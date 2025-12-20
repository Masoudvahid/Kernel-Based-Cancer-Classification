from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


def plot_2d_scatter(
    X_feat: np.ndarray,
    y: np.ndarray,
    subset: Sequence[int],
    kernel_idxs: Sequence[int],
    out_path: Path,
    boundary: dict | None = None,
    title: str = "Best 2-kernel feature space",
) -> None:
    if X_feat.size == 0 or y.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    cancer = y == 1
    healthy = y == 0
    ax.scatter(
        X_feat[cancer, subset[0]],
        X_feat[cancer, subset[1]],
        c="crimson",
        label="cancer",
        alpha=0.75,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.scatter(
        X_feat[healthy, subset[0]],
        X_feat[healthy, subset[1]],
        c="teal",
        label="healthy",
        alpha=0.65,
        edgecolors="k",
        linewidths=0.3,
    )

    if boundary is not None and boundary.get("model") is not None:
        model = boundary["model"]
        device = boundary.get("device", "cpu")
        mean = boundary.get("mean", None)
        std = boundary.get("std", None)
        standardize = boundary.get("standardize", False)
        x_min, x_max = X_feat[:, subset[0]].min(), X_feat[:, subset[0]].max()
        y_min, y_max = X_feat[:, subset[1]].min(), X_feat[:, subset[1]].max()
        pad_x = 0.1 * (x_max - x_min + 1e-6)
        pad_y = 0.1 * (y_max - y_min + 1e-6)
        xs = np.linspace(x_min - pad_x, x_max + pad_x, 200)
        ys = np.linspace(y_min - pad_y, y_max + pad_y, 200)
        xx, yy = np.meshgrid(xs, ys)
        grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
        if standardize and mean is not None and std is not None:
            grid = (grid - mean) / std
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(grid).float().to(device))).cpu().numpy()
        zz = probs.reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], alpha=0.18, colors=["teal", "crimson"])
        ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.0, linestyles="--")

    ax.set_xlabel(f"Kernel {kernel_idxs[0]} response")
    ax.set_ylabel(f"Kernel {kernel_idxs[1]} response")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_3d_scatter(X_feat: np.ndarray, y: np.ndarray, subset: Sequence[int], kernel_idxs: Sequence[int], out_path: Path) -> None:
    if X_feat.size == 0 or y.size == 0:
        return
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    cancer = y == 1
    healthy = y == 0
    ax.scatter(
        X_feat[cancer, subset[0]],
        X_feat[cancer, subset[1]],
        X_feat[cancer, subset[2]],
        c="crimson",
        label="cancer",
        alpha=0.75,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.scatter(
        X_feat[healthy, subset[0]],
        X_feat[healthy, subset[1]],
        X_feat[healthy, subset[2]],
        c="teal",
        label="healthy",
        alpha=0.65,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.set_xlabel(f"Kernel {kernel_idxs[0]} response")
    ax.set_ylabel(f"Kernel {kernel_idxs[1]} response")
    ax.set_zlabel(f"Kernel {kernel_idxs[2]} response")
    ax.set_title("Best 3-kernel feature space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_3d_scatter_interactive(
    X_feat: np.ndarray,
    y: np.ndarray,
    subset: Sequence[int],
    kernel_idxs: Sequence[int],
    out_path: Path,
    project_cancer_to_plane: bool = False,
    project_healthy_to_plane: bool = False,
) -> None:
    if X_feat.size == 0 or y.size == 0:
        return
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - import guard
        msg = "plotly is required for interactive plots. Install with `pip install plotly`."
        raise ImportError(msg) from exc

    labels = np.where(y == 1, "cancer", "healthy")
    fig = px.scatter_3d(
        x=X_feat[:, subset[0]],
        y=X_feat[:, subset[1]],
        z=X_feat[:, subset[2]],
        color=labels,
        color_discrete_map={"cancer": "crimson", "healthy": "teal"},
        opacity=0.8,
        title="Best 3-kernel feature space",
    )
    fig.update_traces(marker=dict(size=4, line=dict(color="black", width=0.5)))
    fig.update_layout(
        legend_title_text="Class",
        scene=dict(
            xaxis_title=f"Kernel {kernel_idxs[0]} response",
            yaxis_title=f"Kernel {kernel_idxs[1]} response",
            zaxis_title=f"Kernel {kernel_idxs[2]} response",
        ),
    )
    if project_cancer_to_plane:
        cancer = y == 1
        z_base = float(np.min(X_feat[:, subset[2]]))
        fig.add_trace(
            go.Scatter3d(
                x=X_feat[cancer, subset[0]],
                y=X_feat[cancer, subset[1]],
                z=np.full(np.sum(cancer), z_base),
                mode="markers",
                marker=dict(size=3, color="crimson", opacity=0.5),
                name="cancer (projection)",
                showlegend=True,
            )
        )
    if project_healthy_to_plane:
        healthy = y == 0
        z_base = float(np.min(X_feat[:, subset[2]]))
        fig.add_trace(
            go.Scatter3d(
                x=X_feat[healthy, subset[0]],
                y=X_feat[healthy, subset[1]],
                z=np.full(np.sum(healthy), z_base),
                mode="markers",
                marker=dict(size=3, color="teal", opacity=0.5),
                name="healthy (projection)",
                showlegend=True,
            )
        )
    fig.write_html(str(out_path))


def plot_roc_pr(y_true: np.ndarray, probs: np.ndarray, out_prefix: Path) -> None:
    if probs is None or y_true is None or len(y_true) == 0:
        return
    fpr, tpr, _ = roc_curve(y_true, probs)
    precision, recall, _ = precision_recall_curve(y_true, probs)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f"AUC={roc_curve_area(y_true, probs):.3f}")
    ax[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax[0].set_title("ROC")
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(recall, precision)
    ax[1].set_title("Precision-Recall")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_roc_pr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, probs: np.ndarray, out_path: Path, threshold: float = 0.5) -> None:
    if probs is None or y_true is None or len(y_true) == 0:
        return
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["healthy", "cancer"])
    ax.set_yticklabels(["healthy", "cancer"])
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def roc_curve_area(y_true: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, probs))
    except Exception:
        return 0.5


__all__ = [
    "plot_2d_scatter",
    "plot_3d_scatter",
    "plot_3d_scatter_interactive",
    "plot_confusion",
    "plot_roc_pr",
    "roc_curve_area",
]
