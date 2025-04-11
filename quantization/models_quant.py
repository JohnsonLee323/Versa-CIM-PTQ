import sys
import torch
import torch.nn as nn

from typing import Callable, Optional, Tuple, Union
from models import build_model, gpt2_simplify, gpt2_dataflow, qwen2_simplify
from types import MethodType
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from timm.models.vision_transformer import Attention
from transformers.utils import logging

logger = logging.get_logger(__name__)

def get_model(config):
    model, tokenizer = build_model(config)

    if not config.FP32:

        if config.MODEL.TYPE == "gpt2":
            if config.DATAFLOW_VAL:
                model = gpt2_dataflow(model)
            else:
                model = gpt2_simplify(model)

        if config.MODEL.TYPE == "qwen2":
            model = qwen2_simplify(model)

        for name, module in model.named_modules():
            if isinstance(module, Attention):
                setattr(module, "matmul1", MatMul())
                setattr(module, "matmul2", MatMul())
                module.forward = MethodType(attention_forward_quant, module)
            if isinstance(module, GPT2Attention):
                setattr(module, "matmul1", MatMul())
                setattr(module, "matmul2", MatMul())
                module.forward = MethodType(gpt2_attention_forward_quant, module)

    return model, tokenizer

# Quantization
class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B

def attention_forward_quant(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    dots = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = dots.softmax(dim=-1)
    attn = self.attn_drop(attn)
    del q, k

    out = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    del attn, v

    out = self.proj(out)
    out = self.proj_drop(out)

    return out

def gpt2_attention_forward_quant(
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

def eager_attention_forward(module, query, key, value, attention_mask, head_mask=None, **kwargs):
    attn_weights = module.matmul1(query, key.transpose(-1, -2))

    attn_weights = attn_weights / torch.full(
        [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
    )

    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
    mask_value = torch.finfo(attn_weights.dtype).min
    mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
    attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    attn_weights = attn_weights.type(value.dtype)

    attn_output = module.matmul2(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights
