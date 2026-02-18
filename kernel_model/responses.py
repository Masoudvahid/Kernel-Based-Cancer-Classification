from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import to_tensor_batch


def compute_responses(
    bank: List[Dict],
    X_in: np.ndarray,
    X_out: np.ndarray,
    device: str = "cpu",
    batch_size: int = 64,
    response_fn: str = "abs_max",
    standardize_patches: bool = True,
    rotation_aug: bool = False,
    rotation_aug_choices: Sequence[int] = (0, 1, 2, 3),
) -> List[Dict[str, np.ndarray]]:
    """
    Convolve every kernel in the bank over the input patches and aggregate responses.
    """
    device = torch.device(device)
    Xin_t = to_tensor_batch(X_in, device)
    Xout_t = to_tensor_batch(X_out, device)

    if standardize_patches:
        def _standardize(t: torch.Tensor) -> torch.Tensor:
            mean = t.mean(dim=[2, 3], keepdim=True)
            std = t.std(dim=[2, 3], keepdim=True)
            std = torch.where(std < 1e-6, torch.ones_like(std), std)
            return (t - mean) / std

        if Xin_t.numel() > 0:
            Xin_t = _standardize(Xin_t)
        if Xout_t.numel() > 0:
            Xout_t = _standardize(Xout_t)

    if not rotation_aug_choices:
        rotation_aug_choices = (0,)
    if any(int(k) not in (0, 1, 2, 3) for k in rotation_aug_choices):
        raise ValueError("rotation_aug_choices must contain quarter-turn values in {0,1,2,3}.")
    rotation_choices = tuple(int(k) % 4 for k in rotation_aug_choices)

    def _maybe_rotate_batch(batch: torch.Tensor) -> torch.Tensor:
        if not rotation_aug or batch.numel() == 0:
            return batch
        if len(rotation_choices) == 1 and rotation_choices[0] == 0:
            return batch
        idx = torch.randint(0, len(rotation_choices), (batch.shape[0],), device=batch.device)
        out = batch.clone()
        for choice_idx, k in enumerate(rotation_choices):
            if k == 0:
                continue
            mask = idx == choice_idx
            if mask.any():
                out[mask] = torch.rot90(out[mask], k=k, dims=(2, 3))
        return out
    responses: List[Dict[str, np.ndarray]] = []

    filters = [
        torch.from_numpy(entry["kernel"]).unsqueeze(0).unsqueeze(0).to(device) for entry in bank
    ]

    for k_t in tqdm(filters, desc="Kernels"):
        r_in_batches: List[np.ndarray] = []
        for i in range(0, Xin_t.shape[0], batch_size):
            batch = Xin_t[i : i + batch_size]
            batch = _maybe_rotate_batch(batch)
            with torch.no_grad():
                out = F.conv2d(batch, k_t, padding=k_t.shape[-1] // 2)
                if response_fn == "mean_abs":
                    val = out.abs().mean(dim=[1, 2, 3]).cpu().numpy()
                else:
                    val = out.abs().amax(dim=[1, 2, 3]).cpu().numpy()
                r_in_batches.append(val)
        r_in = np.concatenate(r_in_batches, axis=0) if r_in_batches else np.zeros((0,))

        r_out_batches: List[np.ndarray] = []
        for i in range(0, Xout_t.shape[0], batch_size):
            batch = Xout_t[i : i + batch_size]
            batch = _maybe_rotate_batch(batch)
            with torch.no_grad():
                out = F.conv2d(batch, k_t, padding=k_t.shape[-1] // 2)
                if response_fn == "mean_abs":
                    val = out.abs().mean(dim=[1, 2, 3]).cpu().numpy()
                else:
                    val = out.abs().amax(dim=[1, 2, 3]).cpu().numpy()
                r_out_batches.append(val)
        r_out = np.concatenate(r_out_batches, axis=0) if r_out_batches else np.zeros((0,))

        responses.append({"r_in": r_in, "r_out": r_out})

    return responses


__all__ = ["compute_responses"]
