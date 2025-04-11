import torch
import torch.nn as nn
import sys

from types import MethodType
from typing import Optional, Tuple, Union
from transformers import GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


def gpt2_dataflow(model):
    for name, module in model.named_modules():
        if isinstance(module, GPT2Model):
            module.forward = MethodType(gpt2model_forward, module)
        if isinstance(module, GPT2Block):
            module.forward = MethodType(gpt2block_forward, module)
    return model


def gpt2model_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

    use_cache = use_cache if use_cache is not None else self.config.use_cache

    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    batch_size = input_ids.shape[0]
    device = input_ids.device

    if past_key_values is None:
        past_key_values = tuple([None] * len(self.h))

    past_length = 0
    if past_key_values[0] is not None:
        past_length = past_key_values[0][0].size(-2)
    position_ids = torch.arange(
        past_length, input_shape[-1] + past_length,
        dtype=torch.long, device=device
    ).unsqueeze(0).expand(batch_size, -1)

    inputs_embeds = self.wte(input_ids)
    position_embeds = self.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds

    presents = () if use_cache else None
    for block, layer_past in zip(self.h, past_key_values):
        outputs = block(
            hidden_states,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        hidden_states = outputs

    hidden_states = self.ln_f(hidden_states)
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents
    )


class PipelineCache:
    def __init__(self):
        self.reset_cache()
        self.cycle = 0


    def reset_cache(self):

        self.input_cache = []

        self.k_cache = []
        self.v_cache = []

        self.out_cache = []


def gpt2block_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:

    cache = PipelineCache()
    cache.reset_cache()
    if hidden_states.shape[0] != 1:
        raise ValueError("Batch_size should be set to 1 when evaluating dataflow!")
    input = hidden_states.squeeze(0)
    embed_dim = input.size(1)

    for cycle in range(input.shape[0]):
        # Stage 1
        cache.input_cache.append(input[cycle])
        x = cache.input_cache[cycle]
        res = x
        x = self.ln_1(x)

        # Stage 2
        q, k, v = self.attn.c_attn(x).split(embed_dim, dim=0)
        q = q.reshape(embed_dim//64, 64)
        k = k.reshape(embed_dim//64, 64)
        v = v.reshape(embed_dim//64, 64)
        cache.k_cache.append(k)
        cache.v_cache.append(v)

        scores = []
        for i in range(cycle+1):
            score = (q * cache.k_cache[i]).sum(dim=1, keepdim=True) / 64**0.5
            scores.append(score)
        scores = torch.cat(scores, dim=1)
        p = torch.softmax(scores, dim=1)

        # Stage 3
        v = torch.stack(cache.v_cache, dim=1)
        y = (p.unsqueeze(-1) * v).sum(dim=1).flatten()

        # Stage 4
        out1 = self.attn.c_proj(y)
        out1 = out1 + res
        res = out1
        out1 = self.ln_2(out1)

        # Stage 5
        out1 = self.mlp.c_fc(out1)
        out1 = nn.GELU(approximate='tanh')(out1)

        # Stage 6
        out1 = self.mlp.c_proj(out1)
        out = out1 + res
        cache.out_cache.append(out)

    out = torch.stack(cache.out_cache, dim=0).unsqueeze(0)

    return out
