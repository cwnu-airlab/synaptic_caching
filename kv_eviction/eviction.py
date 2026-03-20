"""
Pure eviction logic — no transformers dependency, torch only.

Provides:
  - SimpleEvictConfig: eviction strategy configuration
  - build_eviction_config(): CLI parameters → config
  - build_simple_keep_token_idx(): compute which token indices to keep
  - evict_dynamic_cache_inplace(): physically slice KV cache tensors
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_N = 128
"""Block size (tokens per block) for uniform middle sampling.
Matches Flash Attention's typical block size."""

EVICT_VERBOSE = os.environ.get("EVICT_VERBOSE", "0") == "1"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SimpleEvictConfig:
    """Configuration for simple eviction strategies.

    Attributes:
        sink_tokens:          Number of initial tokens to always keep.
        recent_tokens:        Number of most-recent tokens to always keep.
        middle_strategy:      "none" = drop middle entirely.
                              "uniform" = keep uniformly-spaced middle blocks.
        middle_budget_tokens: How many middle tokens to keep (budget mode).
                              Ignored when uniform_stride > 0.
        uniform_stride:       Keep every N-th block in middle region.
                              If > 0, overrides middle_budget_tokens.
    """
    sink_tokens: int = 256
    recent_tokens: int = 512
    middle_strategy: str = "none"
    middle_budget_tokens: int = 0
    uniform_stride: int = 0

    @property
    def cache_target(self) -> int:
        """Target cache length — eviction fires when cache exceeds this."""
        mid = (self.middle_budget_tokens
               if self.middle_strategy == "uniform" and self.uniform_stride == 0
               else 0)
        return self.sink_tokens + self.recent_tokens + mid


def build_eviction_config(
    strategy: str = "sink_recent_uniform",
    sink_tokens: int = 256,
    recent_tokens: int = 512,
    middle_budget: int = 256,
    uniform_stride: int = 0,
    no_eviction: bool = False,
) -> SimpleEvictConfig:
    """Build ``SimpleEvictConfig`` from human-readable parameters.

    Args:
        strategy:       "sink_recent" or "sink_recent_uniform".
        sink_tokens:    Tokens to keep at the beginning.
        recent_tokens:  Tokens to keep at the end.
        middle_budget:  Token budget for uniform middle.
        uniform_stride: Keep every N-th middle block (overrides budget if > 0).
        no_eviction:    If True, return a config that never evicts.
    """
    if no_eviction:
        return SimpleEvictConfig(
            sink_tokens=999_999, recent_tokens=999_999, middle_strategy="none",
        )
    if strategy == "sink_recent":
        return SimpleEvictConfig(
            sink_tokens=sink_tokens, recent_tokens=recent_tokens,
            middle_strategy="none",
        )
    return SimpleEvictConfig(
        sink_tokens=sink_tokens, recent_tokens=recent_tokens,
        middle_strategy="uniform", middle_budget_tokens=middle_budget,
        uniform_stride=uniform_stride,
    )


# ---------------------------------------------------------------------------
# Index Builder
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_simple_keep_token_idx(
    *,
    total_len: int,
    block_n: int = BLOCK_N,
    cfg: SimpleEvictConfig,
    abs_pos: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Construct sorted token indices to **keep** after eviction.

    Partitions the cache into three regions::

        [0 .......... sink_end)     -> SINK   (always kept)
        [sink_end .. recent_start)  -> MIDDLE (evict or uniform-sample)
        [recent_start .. total_len) -> RECENT (always kept)

    Returns:
        1-D ``int64`` tensor of sorted, deduplicated indices to keep.
    """
    total_len = int(total_len)
    sink = max(0, int(cfg.sink_tokens))
    recent = max(0, int(cfg.recent_tokens))
    strategy = str(cfg.middle_strategy).lower()

    if device is None:
        device = torch.device("cpu")

    # Nothing to evict
    if total_len <= sink + recent:
        return torch.arange(total_len, device=device, dtype=torch.long)

    sink_end = min(total_len, sink)
    recent_start = max(0, total_len - recent)

    # ── Strategy: "none" ───────────────────────────────────────────────
    if strategy == "none":
        if EVICT_VERBOSE:
            print(f"  [evict] none  total={total_len}  "
                  f"sink=[0..{sink_end-1}]  recent=[{recent_start}..{total_len-1}]")
        keep = torch.cat([
            torch.arange(0, sink_end, device=device, dtype=torch.long),
            torch.arange(recent_start, total_len, device=device, dtype=torch.long),
        ])
        return torch.unique(keep, sorted=True)

    # ── Strategy: "uniform" ────────────────────────────────────────────
    if strategy == "uniform":
        n_blocks = (total_len + block_n - 1) // block_n
        middle_start_block = (sink_end + block_n - 1) // block_n
        middle_end_block = recent_start // block_n

        middle_blocks: List[int] = []
        selection_method = "no middle"

        if middle_start_block < middle_end_block:
            stride = int(cfg.uniform_stride)
            budget = int(cfg.middle_budget_tokens)

            if stride > 0:
                middle_blocks = list(range(middle_start_block, middle_end_block, stride))
                selection_method = f"stride={stride}"
            elif budget > 0:
                n_middle_blocks = middle_end_block - middle_start_block
                if n_middle_blocks > 0:
                    keep_blocks = max(1, budget // block_n)
                    if keep_blocks >= n_middle_blocks:
                        middle_blocks = list(range(middle_start_block, middle_end_block))
                        selection_method = f"budget={budget}, keep ALL {n_middle_blocks}"
                    else:
                        indices = torch.linspace(0, n_middle_blocks - 1, keep_blocks).long()
                        middle_blocks = [middle_start_block + int(i) for i in indices]
                        selection_method = f"budget={budget}, keep {keep_blocks}/{n_middle_blocks}"

        # Expand blocks → token indices
        middle_tokens: List[int] = []
        for blk in middle_blocks:
            start_tok = blk * block_n
            end_tok = min((blk + 1) * block_n, total_len)
            middle_tokens.extend(range(start_tok, end_tok))

        if EVICT_VERBOSE:
            print(f"  [evict] uniform  total={total_len}  blocks={n_blocks}\n"
                  f"    sink=[0..{sink_end-1}]  mid=[{sink_end}..{recent_start})  "
                  f"recent=[{recent_start}..{total_len-1}]\n"
                  f"    {selection_method} -> {len(middle_blocks)} blocks, {len(middle_tokens)} tokens")

        parts = [torch.arange(0, sink_end, device=device, dtype=torch.long)]
        if middle_tokens:
            parts.append(torch.tensor(middle_tokens, device=device, dtype=torch.long))
        parts.append(torch.arange(recent_start, total_len, device=device, dtype=torch.long))

        return torch.unique(torch.cat(parts), sorted=True)

    # Fallback: unknown strategy → keep all
    return torch.arange(total_len, device=device, dtype=torch.long)


# ---------------------------------------------------------------------------
# In-Place Cache Eviction
# ---------------------------------------------------------------------------

@torch.no_grad()
def evict_dynamic_cache_inplace(
    past_key_values: Any,
    keep_token_idx: torch.Tensor,
) -> None:
    """In-place eviction for HuggingFace ``DynamicCache``.

    Slices every layer's K/V tensors along the sequence dimension,
    then syncs ``_seen_tokens`` and ``_ub_abs_pos`` metadata.
    """
    if not (hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache")):
        return

    new_len = int(keep_token_idx.numel())

    for layer_idx in range(len(past_key_values.key_cache)):
        k = past_key_values.key_cache[layer_idx]
        v = past_key_values.value_cache[layer_idx]
        idx = keep_token_idx.to(device=k.device)
        past_key_values.key_cache[layer_idx] = k.index_select(-2, idx)
        past_key_values.value_cache[layer_idx] = v.index_select(-2, idx)

    if hasattr(past_key_values, "_seen_tokens"):
        try:
            past_key_values._seen_tokens = new_len
        except Exception:
            pass

    if hasattr(past_key_values, "_ub_abs_pos"):
        try:
            abs_pos = past_key_values._ub_abs_pos
            if isinstance(abs_pos, torch.Tensor) and abs_pos.dim() == 1:
                past_key_values._ub_abs_pos = abs_pos.index_select(
                    0, keep_token_idx.to(device=abs_pos.device),
                )
        except Exception:
            pass
