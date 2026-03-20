"""
kv_eviction — Modular KV Cache Eviction Package
================================================

Usage (same as before):
    from kv_eviction import (
        SimpleEvictConfig, build_eviction_config,
        build_simple_keep_token_idx, evict_dynamic_cache_inplace,
        streaming_prefill, streaming_ppl, greedy_decode,
        load_model_and_tokenizer, patch_model_raw_kv,
        save_eviction_heatmap, BLOCK_N,
    )

Or import specific modules:
    from kv_eviction.eviction import SimpleEvictConfig  # torch only
    from kv_eviction.rope_patch import patch_model_raw_kv
"""

# Core eviction (torch only — no transformers dependency)
from .eviction import (
    BLOCK_N,
    SimpleEvictConfig,
    build_eviction_config,
    build_simple_keep_token_idx,
    evict_dynamic_cache_inplace,
)

# Streaming engine
from .streaming import (
    streaming_prefill,
    streaming_ppl,
    greedy_decode,
)

# Model utilities
from .model_utils import load_model_and_tokenizer

# RoPE patching
from .rope_patch import patch_model_raw_kv

# Visualization
from .visualization import save_eviction_heatmap

__all__ = [
    # Core eviction
    "BLOCK_N",
    "SimpleEvictConfig",
    "build_eviction_config",
    "build_simple_keep_token_idx",
    "evict_dynamic_cache_inplace",
    # Streaming
    "streaming_prefill",
    "streaming_ppl",
    "greedy_decode",
    # Model
    "load_model_and_tokenizer",
    "patch_model_raw_kv",
    # Visualization
    "save_eviction_heatmap",
]
