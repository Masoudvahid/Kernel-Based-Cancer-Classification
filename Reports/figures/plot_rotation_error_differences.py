#!/usr/bin/env python3
"""Bar plot of rotation-wise error differences against two baselines."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    rotations = np.arange(0, 360, 10)

    # Invariant-kernel accuracies from the every-10-degree sweep in the report.
    invariant_acc = np.array(
        [
            0.809091,
            0.814973,
            0.818182,
            0.818717,
            0.817647,
            0.811230,
            0.826738,
            0.823529,
            0.814973,
            0.809091,
            0.814973,
            0.818182,
            0.818717,
            0.817647,
            0.811230,
            0.826738,
            0.823529,
            0.814973,
            0.809091,
            0.814973,
            0.818182,
            0.818717,
            0.817647,
            0.811230,
            0.826738,
            0.823529,
            0.814973,
            0.809091,
            0.814973,
            0.818182,
            0.818717,
            0.817647,
            0.811230,
            0.826738,
            0.823529,
            0.814973,
        ]
    )

    invariant_base_acc = 0.809091
    non_invariant_base_acc = 0.814973

    invariant_err = 1.0 - invariant_acc
    err_invariant_base = 1.0 - invariant_base_acc
    err_non_invariant_base = 1.0 - non_invariant_base_acc

    # Positive values mean rotated case has higher error than baseline.
    d_err_vs_invariant = invariant_err - err_invariant_base
    d_err_vs_non_invariant = invariant_err - err_non_invariant_base

    x = np.arange(rotations.size)
    width = 0.42

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
    ax.bar(
        x - width / 2,
        d_err_vs_invariant,
        width=width,
        label="vs invariant baseline (0°)",
        color="#f4a261",
    )
    ax.bar(
        x + width / 2,
        d_err_vs_non_invariant,
        width=width,
        label="vs non-invariant baseline (0°)",
        color="#2a9d8f",
    )

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title("Error Difference Across Rotations (0° to 350°)")
    ax.set_xlabel("Rotation angle (degrees)")
    ax.set_ylabel("Error difference: rotated - baseline")
    ax.set_xlim(-1, rotations.size)
    ax.grid(axis="y", alpha=0.25)

    tick_idx = np.arange(0, rotations.size, 3)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(int(rotations[i])) for i in tick_idx], rotation=0)

    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    out_path = Path(__file__).resolve().parent / "rotation_error_difference_barplot.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
