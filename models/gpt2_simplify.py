import sys

import torch

from types import MethodType
from typing import Callable, Optional, Tuple, Union
from transformers import GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP, eager_attention_forward
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


def gpt2_simplify(model):
    for name, module in model.named_modules():
        if isinstance(module, GPT2Model):
            module.forward = MethodType(gpt2model_forward, module)
        if isinstance(module, GPT2Block):
            module.forward = MethodType(gpt2block_forward, module)
        if isinstance(module, GPT2Attention):
            module.forward = MethodType(gpt2attention_forward, module)
        if isinstance(module, GPT2MLP):
            module.forward = MethodType(gpt2mlp_forward, module)
    return model


# GPT2 Simplify
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

    # 配置参数处理
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    # 输入预处理
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    batch_size = input_ids.shape[0]
    device = input_ids.device

    # 初始化past_key_values
    if past_key_values is None:
        past_key_values = tuple([None] * len(self.h))

    # 自动生成position_ids
    past_length = 0
    if past_key_values[0] is not None:
        past_length = past_key_values[0][0].size(-2)  # 从第一个key获取序列长度
    position_ids = torch.arange(
        past_length, input_shape[-1] + past_length,
        dtype=torch.long, device=device
    ).unsqueeze(0).expand(batch_size, -1)

    # 嵌入层处理
    inputs_embeds = self.wte(input_ids)
    position_embeds = self.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds

    # Transformer层处理
    presents = () if use_cache else None
    for block, layer_past in zip(self.h, past_key_values):
        outputs = block(
            hidden_states,
            layer_past=layer_past,
            use_cache=use_cache,
        )

        hidden_states = outputs[0]
        if use_cache:
            presents += (outputs[1],)
    # 最终层归一化
    hidden_states = self.ln_f(hidden_states)

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents
    )


def gpt2block_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:

    # 自注意力模块
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_outputs = self.attn(
        hidden_states,
        layer_past=layer_past,
        use_cache=use_cache
    )
    attn_output = attn_outputs[0]
    outputs = attn_outputs[1:]
    hidden_states = residual + attn_output  # 残差连接

    # 前馈网络模块
    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)
    feed_forward = self.mlp(hidden_states)
    hidden_states = residual + feed_forward  # 残差连接

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions, cross_attentions)


def gpt2attention_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    **kwargs,
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

    bsz, q_len, _ = hidden_states.size()

    # Compute query, key, value
    query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

    # Split into heads
    shape_q = (*query.shape[:-1], -1, self.head_dim)
    shape_kv = (*key.shape[:-1], -1, self.head_dim)

    query = query.view(shape_q).transpose(1, 2)
    key = key.view(shape_kv).transpose(1, 2)
    value = value.view(shape_kv).transpose(1, 2)

    # Optional kv caching
    if layer_past is not None:
        past_key = layer_past[0]
        past_value = layer_past[1]
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)

    present = None
    if use_cache is True:
        present = (key, value)

    is_causal = True if attention_mask is None and q_len > 1 else False

    # Compute attention
    attention_interface: Callable = eager_attention_forward

    attn_output, attn_weights = attention_interface(
        self,
        query,
        key,
        value,
        attention_mask,
        head_mask=head_mask,
        dropout=self.attn_dropout.p if self.training else 0.0,
        is_causal=is_causal,
        **kwargs,
    )

    # Reshape outputs
    attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()

    # Final projection
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # a, present, (attentions)


def gpt2mlp_forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
    hidden_states = self.c_fc(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.c_proj(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states
