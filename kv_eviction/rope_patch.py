"""
RoPE patching for eviction-safe KV caching.

Monkey-patches attention layers to store **raw (unrotated) K** in the
DynamicCache and apply RoPE on-the-fly at attention time using
cache-slot positions ``[0 .. kv_len-1]``.

Supports: LlamaFlashAttention2, Qwen2FlashAttention2.
"""
from __future__ import annotations

from typing import Any

import torch


# ---------------------------------------------------------------------------
# RoPE helper
# ---------------------------------------------------------------------------

def _apply_rotary_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to a single tensor (Q or K).

    Args:
        x:   [bsz, n_heads, seq_len, head_dim]
        cos: [1, seq_len, head_dim] or broadcastable
        sin: same shape as cos
    """
    cos = cos.unsqueeze(1)  # [1, 1, seq_len, hd]
    sin = sin.unsqueeze(1)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------

def patch_model_raw_kv(model: Any) -> None:
    """Monkey-patch attention layers to store raw (unrotated) K in cache
    and apply RoPE on-the-fly at attention time.

    After eviction, surviving raw keys get contiguous cache-slot positions,
    keeping everything within the model's trained RoPE range.
    """
    # ── Import attention classes ───────────────────────────────────────
    LlamaFA2 = None
    Qwen2FA2 = None
    _repeat_kv = None

    try:
        from transformers.models.llama.modeling_llama import (
            LlamaFlashAttention2, repeat_kv,
        )
        LlamaFA2 = LlamaFlashAttention2
        _repeat_kv = repeat_kv
    except ImportError:
        pass

    try:
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2FlashAttention2, repeat_kv as qwen2_repeat_kv,
        )
        Qwen2FA2 = Qwen2FlashAttention2
        if _repeat_kv is None:
            _repeat_kv = qwen2_repeat_kv
    except ImportError:
        pass

    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except ImportError:
        _flash_attention_forward = None

    if LlamaFA2 is None and Qwen2FA2 is None:
        print("WARNING: Neither LlamaFlashAttention2 nor Qwen2FlashAttention2 found. "
              "raw_rel mode requires Flash Attention 2.")
        return

    # ── Llama patched forward ──────────────────────────────────────────
    def _patched_forward_llama(
        self, hidden_states, attention_mask=None, position_ids=None,
        past_key_value=None, output_attentions=False, use_cache=False,
        cache_position=None, position_embeddings=None, **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # RoPE for Q only
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states = _apply_rotary_single(query_states, cos, sin)

        # Store RAW (unrotated) K in cache
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, kwargs,
            )

        # On-the-fly RoPE for ALL cached K
        kv_len = key_states.shape[2]
        slot_pos = torch.arange(kv_len, device=key_states.device, dtype=torch.long).unsqueeze(0)
        cos_k, sin_k = self.rotary_emb(value_states, slot_pos)
        key_states_rot = _apply_rotary_single(key_states, cos_k, sin_k)

        # GQA
        key_states_rot = _repeat_kv(key_states_rot, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        # Attention
        if _flash_attention_forward is not None:
            query_states = query_states.transpose(1, 2)
            key_states_rot = key_states_rot.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            attn_output = _flash_attention_forward(
                query_states, key_states_rot, value_states,
                attention_mask, q_len, dropout=0.0,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
                is_causal=self.is_causal,
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states_rot, value_states,
                attn_mask=attention_mask,
                is_causal=(attention_mask is None and q_len > 1),
            )
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    # ── Qwen2 patched forward ─────────────────────────────────────────
    def _patched_forward_qwen2(
        self, hidden_states, attention_mask=None, position_ids=None,
        past_key_value=None, output_attentions=False, use_cache=False,
        cache_position=None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError("Cache structure requires layer_idx.")
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        rotary_seq_len = max(kv_seq_len, int(position_ids[:, -1].max().item()) + 1)
        cos_table, sin_table = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        # RoPE for Q only
        cos_q = cos_table[position_ids]
        sin_q = sin_table[position_ids]
        query_states = _apply_rotary_single(query_states, cos_q, sin_q)

        # Store RAW K in cache
        if past_key_value is not None:
            cache_position = torch.arange(
                kv_seq_len - q_len, kv_seq_len,
                device=key_states.device, dtype=torch.long,
            )
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs,
            )

        # On-the-fly RoPE for ALL cached K
        kv_len = key_states.shape[2]
        if kv_len > cos_table.shape[0]:
            cos_table, sin_table = self.rotary_emb(value_states, seq_len=kv_len)
        cos_k = cos_table[:kv_len].unsqueeze(0)
        sin_k = sin_table[:kv_len].unsqueeze(0)
        key_states_rot = _apply_rotary_single(key_states, cos_k, sin_k)

        # GQA
        key_states_rot = _repeat_kv(key_states_rot, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        # Attention
        if _flash_attention_forward is not None:
            query_states = query_states.transpose(1, 2)
            key_states_rot = key_states_rot.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            attn_output = _flash_attention_forward(
                query_states, key_states_rot, value_states,
                attention_mask, q_len, dropout=0.0,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
                is_causal=self.is_causal,
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states_rot, value_states,
                attn_mask=attention_mask,
                is_causal=(attention_mask is None and q_len > 1),
            )
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    # ── Apply patches ──────────────────────────────────────────────────
    patched_llama = 0
    patched_qwen2 = 0
    for _name, module in model.named_modules():
        if LlamaFA2 is not None and isinstance(module, LlamaFA2):
            module.forward = _patched_forward_llama.__get__(module, type(module))
            patched_llama += 1
        elif Qwen2FA2 is not None and isinstance(module, Qwen2FA2):
            module.forward = _patched_forward_qwen2.__get__(module, type(module))
            patched_qwen2 += 1

    total = patched_llama + patched_qwen2
    if total > 0:
        parts = []
        if patched_llama > 0:
            parts.append(f"{patched_llama} Llama")
        if patched_qwen2 > 0:
            parts.append(f"{patched_qwen2} Qwen2")
        print(f"  Patched {' + '.join(parts)} attention layers for raw KV + on-the-fly RoPE")
    else:
        print("WARNING: No supported FlashAttention2 layers found to patch.")
