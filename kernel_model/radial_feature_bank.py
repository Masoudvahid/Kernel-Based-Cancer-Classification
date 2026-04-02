from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def init_radial_feature_bank(
    f_init: np.ndarray,
    n_feature_kernels: int,
    seed: int = 0,
    scale_span: Tuple[float, float] = (0.9, 1.1),
    noise_scale: float = 0.05,
) -> np.ndarray:
    """
    Create a bank of radial coefficient vectors with small deterministic variation.

    The first feature kernel is kept identical to the seed profile. The others get a
    mild scale spread and noise so training does not collapse into a symmetric bank.
    """
    if n_feature_kernels <= 0:
        raise ValueError("n_feature_kernels must be positive.")

    f_init = np.asarray(f_init, dtype=np.float32)
    bank = np.repeat(f_init[None, :], int(n_feature_kernels), axis=0)
    if n_feature_kernels == 1:
        return bank

    lo, hi = scale_span
    scales = np.linspace(float(lo), float(hi), int(n_feature_kernels), dtype=np.float32)
    bank *= scales[:, None]

    amp = max(float(np.max(np.abs(f_init))), 1e-6)
    if noise_scale > 0:
        rng = np.random.default_rng(int(seed))
        noise = rng.normal(0.0, float(noise_scale) * amp, size=bank.shape).astype(np.float32)
        bank += noise

    bank[0] = f_init
    return bank.astype(np.float32)


def kernels_from_coeff_bank(coeff_bank: torch.Tensor, basis_t: torch.Tensor) -> torch.Tensor:
    """
    Build a radial kernel bank.

    coeff_bank: (F, R+1)
    basis_t: (R+1, H, W)
    returns: (F, H, W)
    """
    return torch.einsum("fr,rhw->fhw", coeff_bank, basis_t)


def response_features_from_kernel_bank(
    x_t: torch.Tensor,
    kernels_t: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """
    Apply every kernel in the bank and pool each response map to one scalar feature.

    x_t: (B, 1, H, W)
    kernels_t: (F, Kh, Kw)
    returns: (B, F)
    """
    out = F.conv2d(x_t, kernels_t[:, None, :, :], padding=kernels_t.shape[-1] // 2)
    if mode == "abs_max":
        return out.abs().amax(dim=(2, 3))
    if mode == "mean_abs":
        return out.abs().mean(dim=(2, 3))
    raise ValueError(f"Unknown response mode '{mode}'")


def logits_from_feature_bank(
    features_t: torch.Tensor,
    head_w: torch.Tensor,
    head_b: torch.Tensor,
) -> torch.Tensor:
    """
    Linear logistic head over a bank of pooled radial features.
    """
    return features_t @ head_w + head_b


def dense_logits_from_kernel_bank(
    x_t: torch.Tensor,
    kernels_t: torch.Tensor,
    head_w: torch.Tensor,
    head_b: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """
    Dense per-pixel logits for a feature-kernel bank.

    x_t: (B, 1, H, W)
    kernels_t: (F, Kh, Kw)
    returns: (B, H, W)
    """
    out = F.conv2d(x_t, kernels_t[:, None, :, :], padding=kernels_t.shape[-1] // 2)
    if mode == "abs_max" or mode == "mean_abs":
        feat_maps = out.abs()
    else:
        raise ValueError(f"Unknown response mode '{mode}'")
    return (feat_maps * head_w[None, :, None, None]).sum(dim=1) + head_b

