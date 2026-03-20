"""
Model loading utilities.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
) -> Tuple[Any, Any]:
    """Load a causal LM with Flash Attention 2 (graceful fallback).

    Args:
        model_name: HuggingFace model name or local path.
        device:     "auto" (GPU if available), "cuda", or "cpu".

    Returns:
        (model, tokenizer) tuple.  Model is in eval mode, bfloat16 on CUDA.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model: {model_name}")
    print(f"Device: {device}")

    if device == "auto" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
    }
    if device == "auto":
        model_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="flash_attention_2", **model_kwargs,
        )
        print("  Flash Attention 2 enabled")
    except Exception as e:
        print(f"  Flash Attention 2 not available ({e}), using standard attention")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if device != "auto":
        model = model.to(device)

    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model loaded ({n_params:.1f}M params)")
    return model, tokenizer
