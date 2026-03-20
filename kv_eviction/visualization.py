"""
Eviction visualization — heatmap generation.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from .eviction import BLOCK_N, SimpleEvictConfig, build_simple_keep_token_idx


def save_eviction_heatmap(
    evict_cfg: SimpleEvictConfig,
    total_len: int,
    num_layers: int,
    output_path: str,
    block_n: int = BLOCK_N,
    title_prefix: str = "KV Cache Eviction Pattern",
) -> None:
    """Simulate eviction and save a heatmap PNG.

    Y-axis: layer index, X-axis: token position.
    Blue = kept, gray = evicted.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        import numpy as np
    except ImportError:
        print("  matplotlib/numpy not available, skipping heatmap")
        return

    keep_idx = build_simple_keep_token_idx(
        total_len=total_len, block_n=block_n, cfg=evict_cfg,
        device=torch.device("cpu"),
    )
    keep_set = set(keep_idx.tolist())
    mask = np.zeros((num_layers, total_len), dtype=np.uint8)
    for pos in keep_set:
        mask[:, pos] = 1

    cmap = ListedColormap(["#D9D9D9", "#3274A1"])
    fig, ax = plt.subplots(figsize=(16, max(4, num_layers * 0.22)))
    ax.imshow(mask, aspect="auto", cmap=cmap, interpolation="none", origin="lower")

    sink_end = min(total_len, evict_cfg.sink_tokens)
    recent_start = max(0, total_len - evict_cfg.recent_tokens)
    ax.axvline(x=sink_end - 0.5, color="#E74C3C", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.axvline(x=recent_start - 0.5, color="#2ECC71", linewidth=1.5, linestyle="--", alpha=0.8)

    ax.set_xlabel("Position Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)

    mid_info = "no mid"
    if evict_cfg.middle_strategy == "uniform":
        mid_info = (f"stride={evict_cfg.uniform_stride}" if evict_cfg.uniform_stride > 0
                    else f"mid={evict_cfg.middle_budget_tokens}")

    kept_count = int(mask[0].sum())
    ax.set_title(
        f"{title_prefix}\n"
        f"total={total_len}  sink={evict_cfg.sink_tokens}  "
        f"recent={evict_cfg.recent_tokens}  {mid_info}  |  "
        f"kept={kept_count}  evicted={total_len - kept_count}",
        fontsize=13, fontweight="bold",
    )

    if num_layers <= 16:
        ax.set_yticks(range(num_layers))
    else:
        ax.set_yticks(range(0, num_layers, max(1, num_layers // 8)))

    xtick_step = 256 if total_len <= 2048 else (512 if total_len <= 8192 else 1024)
    ax.set_xticks(range(0, total_len, xtick_step))

    legend_elements = [
        Patch(facecolor="#3274A1", label="Kept"),
        Patch(facecolor="#D9D9D9", label="Evicted"),
        Patch(facecolor="none", edgecolor="#E74C3C", linestyle="--",
              label=f"Sink boundary ({sink_end})"),
        Patch(facecolor="none", edgecolor="#2ECC71", linestyle="--",
              label=f"Recent boundary ({recent_start})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)
    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved to: {output_path}")
