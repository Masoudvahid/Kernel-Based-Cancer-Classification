from typing import Dict, List

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
) -> List[Dict[str, np.ndarray]]:
    """
    Convolve every kernel in the bank over the input patches and aggregate responses.
    """
    device = torch.device(device)
    Xin_t = to_tensor_batch(X_in, device)
    Xout_t = to_tensor_batch(X_out, device)
    responses: List[Dict[str, np.ndarray]] = []

    filters = [
        torch.from_numpy(entry["kernel"]).unsqueeze(0).unsqueeze(0).to(device) for entry in bank
    ]

    for k_t in tqdm(filters, desc="Kernels"):
        r_in_batches: List[np.ndarray] = []
        for i in range(0, Xin_t.shape[0], batch_size):
            batch = Xin_t[i : i + batch_size]
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
