"""
Streaming engine — chunk-based prefill with eviction + decode.

Provides:
  - streaming_prefill(): for generation tasks (QA, NIAH, RULER)
  - streaming_ppl(): for perplexity evaluation (PG-19)
  - greedy_decode(): greedy generation from prefilled cache
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .eviction import (
    BLOCK_N,
    SimpleEvictConfig,
    build_simple_keep_token_idx,
    evict_dynamic_cache_inplace,
)


@torch.no_grad()
def streaming_prefill(
    model: Any,
    input_ids: torch.Tensor,
    evict_cfg: SimpleEvictConfig,
    chunk_size: int = 512,
    block_n: int = BLOCK_N,
    rope_mode: str = "abs",
    verbose: bool = False,
) -> Tuple[Any, torch.Tensor, int, Dict[str, Any]]:
    """Feed ``input_ids`` chunk-by-chunk, evicting when cache exceeds target.

    Use this for **generation tasks** (QA, summarization, NIAH).
    For perplexity evaluation, use :func:`streaming_ppl` instead.

    Returns:
        ``(cache, last_logits, prompt_len, stats)``
    """
    from transformers import DynamicCache

    device = input_ids.device
    seq_len = int(input_ids.shape[1])
    chunk_size = max(1, int(chunk_size))
    target = evict_cfg.cache_target

    cache = DynamicCache()
    last_logits = None
    evict_count = 0
    total_evicted_tokens = 0

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end]
        chunk_len = end - start

        if rope_mode == "abs":
            position_ids = torch.arange(start, end, device=device).unsqueeze(0)
        else:
            cur_cache_len = cache.get_seq_length()
            position_ids = torch.arange(
                cur_cache_len, cur_cache_len + chunk_len, device=device,
            ).unsqueeze(0)

        out = model(
            input_ids=chunk, past_key_values=cache,
            use_cache=True, position_ids=position_ids,
        )
        cache = out.past_key_values
        last_logits = out.logits

        cur_len = cache.get_seq_length()
        if cur_len > target:
            before = cur_len
            keep_idx = build_simple_keep_token_idx(
                total_len=cur_len, block_n=block_n, cfg=evict_cfg, device=device,
            )
            evict_dynamic_cache_inplace(cache, keep_idx)
            after = cache.get_seq_length()
            evict_count += 1
            total_evicted_tokens += before - after
            if verbose:
                print(f"  [EVICT] chunk [{start}:{end}]  cache {before} -> {after}  "
                      f"dropped {before - after}")
        elif verbose:
            print(f"  [CHUNK] [{start}:{end}]  cache {cur_len}")

    stats = {
        "eviction_count": evict_count,
        "total_evicted_tokens": total_evicted_tokens,
        "final_cache_len": cache.get_seq_length(),
        "target_cache_len": target,
    }
    return cache, last_logits, seq_len, stats


@torch.no_grad()
def streaming_ppl(
    model: Any,
    input_ids: torch.Tensor,
    evict_cfg: Optional[SimpleEvictConfig],
    chunk_size: int = 512,
    block_n: int = BLOCK_N,
    rope_mode: str = "abs",
    eval_last_tokens: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compute perplexity via chunk-based streaming prefill with eviction.

    Use this for **language modeling evaluation** (PG-19, etc.).

    Returns:
        ``dict`` with ``ppl``, ``total_nll``, ``total_tokens_scored``,
        ``eviction_count``, ``total_evicted_tokens``, ``final_cache_len``,
        ``per_chunk``.
    """
    from transformers import DynamicCache

    device = input_ids.device
    seq_len = int(input_ids.shape[1])
    vocab_size = model.config.vocab_size
    chunk_size = max(1, int(chunk_size))

    cache_target = evict_cfg.cache_target if evict_cfg is not None else seq_len + 1
    score_from = (seq_len - eval_last_tokens) if eval_last_tokens > 0 else 0

    cache = DynamicCache()
    total_nll = 0.0
    total_tokens = 0
    evict_count = 0
    total_evicted_tokens = 0
    per_chunk: List[Dict[str, Any]] = []
    prev_last_logit: Optional[torch.Tensor] = None

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end]
        chunk_len = end - start

        if rope_mode == "abs":
            position_ids = torch.arange(start, end, device=device).unsqueeze(0)
        else:
            cur_cache_len = cache.get_seq_length()
            position_ids = torch.arange(
                cur_cache_len, cur_cache_len + chunk_len, device=device,
            ).unsqueeze(0)

        out = model(
            input_ids=chunk, past_key_values=cache,
            use_cache=True, position_ids=position_ids,
        )
        cache = out.past_key_values
        logits = out.logits

        # ── CE loss ──
        chunk_nll = 0.0
        chunk_count = 0

        # (a) Boundary loss
        if prev_last_logit is not None:
            boundary_loss = F.cross_entropy(
                prev_last_logit.view(1, -1), chunk[:, 0], reduction="sum",
            )
            if start >= score_from:
                chunk_nll += boundary_loss.item()
                chunk_count += 1

        # (b) Within-chunk loss
        if chunk_len > 1:
            shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
            shift_labels = chunk[:, 1:].contiguous().view(-1)

            if eval_last_tokens == 0:
                chunk_nll += F.cross_entropy(shift_logits, shift_labels, reduction="sum").item()
                chunk_count += chunk_len - 1
            else:
                per_token_loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
                positions = torch.arange(start + 1, end, device=device)
                mask = positions >= score_from
                if mask.any():
                    chunk_nll += per_token_loss[mask].sum().item()
                    chunk_count += int(mask.sum().item())

        prev_last_logit = logits[:, -1:, :].detach()

        # ── Evict ──
        cur_len = cache.get_seq_length()
        evicted_this = 0
        if evict_cfg is not None and cur_len > cache_target:
            before = cur_len
            keep_idx = build_simple_keep_token_idx(
                total_len=cur_len, block_n=block_n, cfg=evict_cfg, device=device,
            )
            evict_dynamic_cache_inplace(cache, keep_idx)
            after = cache.get_seq_length()
            evict_count += 1
            evicted_this = before - after
            total_evicted_tokens += evicted_this
            if verbose:
                print(f"  [EVICT] chunk [{start}:{end}]  cache {before} -> {after}  "
                      f"dropped {evicted_this}")
        elif verbose:
            print(f"  [CHUNK] [{start}:{end}]  cache {cur_len}")

        if chunk_count > 0:
            per_chunk.append({
                "start": start, "end": end,
                "ppl": round(math.exp(chunk_nll / chunk_count), 4),
                "nll": round(chunk_nll, 6), "tokens_scored": chunk_count,
                "cache_len": cache.get_seq_length(), "evicted": evicted_this,
            })

        total_nll += chunk_nll
        total_tokens += chunk_count

    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")

    return {
        "ppl": round(ppl, 4), "total_nll": round(total_nll, 6),
        "total_tokens_scored": total_tokens, "eviction_count": evict_count,
        "total_evicted_tokens": total_evicted_tokens,
        "final_cache_len": cache.get_seq_length(), "per_chunk": per_chunk,
    }


@torch.no_grad()
def greedy_decode(
    model: Any,
    tokenizer: Any,
    cache: Any,
    last_logits: torch.Tensor,
    prompt_len: int,
    max_new_tokens: int = 128,
    rope_mode: str = "abs",
) -> Tuple[str, List[int], int]:
    """Greedy decode from a prefilled cache.

    Returns:
        ``(text, token_ids, n_generated)``
    """
    device = last_logits.device
    eos_id = tokenizer.eos_token_id

    next_token = last_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [int(next_token.item())]

    if next_token.item() == eos_id:
        return tokenizer.decode(generated, skip_special_tokens=True).strip(), generated, 1

    for step in range(max_new_tokens - 1):
        if rope_mode == "abs":
            position_ids = torch.tensor([[prompt_len + step]], device=device)
        else:
            position_ids = torch.tensor([[cache.get_seq_length()]], device=device)

        out = model(
            input_ids=next_token, past_key_values=cache,
            use_cache=True, position_ids=position_ids,
        )
        cache = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
        if next_token.item() == eos_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, generated, len(generated)
