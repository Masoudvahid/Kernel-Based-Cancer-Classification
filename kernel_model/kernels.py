import math
from typing import Dict, List, Optional, Sequence

import numpy as np


def make_gaussian_kernel(size: int, sigma_x: float, sigma_y: Optional[float] = None, theta: float = 0.0) -> np.ndarray:
    if sigma_y is None:
        sigma_y = sigma_x
    assert size % 2 == 1, "size should be odd"
    half = size // 2
    xs = np.arange(-half, half + 1, 1)
    ys = np.arange(-half, half + 1, 1)
    X, Y = np.meshgrid(xs, ys)
    ct = math.cos(theta)
    st = math.sin(theta)
    Xr = ct * X + st * Y
    Yr = -st * X + ct * Y
    G = np.exp(-0.5 * ((Xr ** 2) / (sigma_x ** 2 + 1e-12) + (Yr ** 2) / (sigma_y ** 2 + 1e-12)))
    G = G / (G.sum() + 1e-12)
    return G.astype(np.float32)


def make_dog_kernel(size: int, sigma1: float, sigma2: float, theta: float = 0.0) -> np.ndarray:
    g1 = make_gaussian_kernel(size, sigma1, sigma1, theta)
    g2 = make_gaussian_kernel(size, sigma2, sigma2, theta)
    k = g1 - g2
    k = k - k.mean()
    return k.astype(np.float32)


def make_log_kernel(size: int, sigma: float, theta: float = 0.0) -> np.ndarray:
    assert size % 2 == 1
    half = size // 2
    xs = np.arange(-half, half + 1, 1)
    ys = np.arange(-half, half + 1, 1)
    X, Y = np.meshgrid(xs, ys)
    ct = math.cos(theta)
    st = math.sin(theta)
    Xr = ct * X + st * Y
    Yr = -st * X + ct * Y
    r2 = Xr ** 2 + Yr ** 2
    s2 = sigma ** 2
    LoG = ((r2 - 2 * s2) / (s2 ** 2)) * np.exp(-r2 / (2 * s2))
    LoG = LoG - LoG.mean()
    return LoG.astype(np.float32)


def make_gabor_kernel(
    size: int,
    sigma: float,
    freq: float,
    theta: float = 0.0,
    phase: float = 0.0,
    gamma: float = 1.0,
) -> np.ndarray:
    assert size % 2 == 1
    half = size // 2
    xs = np.arange(-half, half + 1, 1)
    ys = np.arange(-half, half + 1, 1)
    X, Y = np.meshgrid(xs, ys)
    ct = math.cos(theta)
    st = math.sin(theta)
    Xr = ct * X + st * Y
    Yr = -st * X + ct * Y
    Yr = Yr * gamma
    gaussian = np.exp(-(Xr ** 2 + Yr ** 2) / (2 * (sigma ** 2)))
    sinusoid = np.cos(2 * np.pi * freq * Xr + phase)
    K = gaussian * sinusoid
    K = K - K.mean()
    if K.sum() != 0:
        K = K / (np.abs(K).sum() + 1e-12)
    return K.astype(np.float32)


def make_hog_kernel(size: int, sigma: float, theta: float = 0.0) -> np.ndarray:
    assert size % 2 == 1
    half = size // 2
    xs = np.arange(-half, half + 1, 1)
    ys = np.arange(-half, half + 1, 1)
    X, Y = np.meshgrid(xs, ys)
    ct = math.cos(theta)
    st = math.sin(theta)
    Xr = ct * X + st * Y
    Yr = -st * X + ct * Y
    gaussian = np.exp(-(Xr ** 2 + Yr ** 2) / (2 * (sigma ** 2)))
    K = -(Xr / (sigma ** 2 + 1e-12)) * gaussian
    K = K - K.mean()
    K = K / (np.abs(K).sum() + 1e-12)
    return K.astype(np.float32)


def make_lbp_kernel(size: int, radius: int, theta: float = 0.0) -> np.ndarray:
    # LBP-inspired: compare one neighbor against the center.
    assert size % 2 == 1
    half = size // 2
    dx = int(round(radius * math.cos(theta)))
    dy = int(round(radius * math.sin(theta)))
    if dx == 0 and dy == 0:
        dx = int(round(radius)) or 1
    dx = max(-half, min(half, dx))
    dy = max(-half, min(half, dy))
    K = np.zeros((size, size), dtype=np.float32)
    K[half, half] = -1.0
    K[half + dy, half + dx] = 1.0
    K = K - K.mean()
    denom = np.abs(K).sum()
    if denom > 0:
        K = K / denom
    return K.astype(np.float32)


def make_glcm_kernel(size: int, radius: int, theta: float = 0.0, offset_weight: float = 1.0) -> np.ndarray:
    # GLCM-inspired: two-point offset kernel.
    assert size % 2 == 1
    half = size // 2
    dx = int(round(radius * math.cos(theta)))
    dy = int(round(radius * math.sin(theta)))
    if dx == 0 and dy == 0:
        dx = int(round(radius)) or 1
    dx = max(-half, min(half, dx))
    dy = max(-half, min(half, dy))
    K = np.zeros((size, size), dtype=np.float32)
    K[half, half] = 1.0
    K[half + dy, half + dx] = float(offset_weight)
    K = K - K.mean()
    denom = np.abs(K).sum()
    if denom > 0:
        K = K / denom
    return K.astype(np.float32)


def make_mrf_kernel(size: int, radius: int, beta: float = 1.0, neighborhood: str = "cross") -> np.ndarray:
    # MRF-inspired: local smoothness (discrete Laplacian) kernel.
    assert size % 2 == 1
    half = size // 2
    r = max(1, min(half, int(round(radius))))
    K = np.zeros((size, size), dtype=np.float32)
    if neighborhood == "full":
        offsets = [
            (-r, -r), (-r, 0), (-r, r),
            (0, -r), (0, r),
            (r, -r), (r, 0), (r, r),
        ]
    elif neighborhood == "cross":
        offsets = [(-r, 0), (r, 0), (0, -r), (0, r)]
    else:
        raise ValueError(f"Unknown neighborhood '{neighborhood}'")
    for dy, dx in offsets:
        if -half <= dy <= half and -half <= dx <= half:
            K[half + dy, half + dx] = -1.0
    K[half, half] = float(len(offsets))
    K = K * float(beta)
    K = K - K.mean()
    denom = np.abs(K).sum()
    if denom > 0:
        K = K / denom
    return K.astype(np.float32)


def sample_parameters(family: str, n_samples: int, size: int) -> List[Dict]:
    params: List[Dict] = []
    for _ in range(n_samples):
        if family == "gaussian":
            sigma = float(10 ** np.random.uniform(np.log10(0.5), np.log10(size / 2)))
            theta = np.random.uniform(0, math.pi)
            params.append({"sigma_x": sigma, "sigma_y": sigma, "theta": theta, "size": size})
        elif family == "anisotropic_gaussian":
            sigma_x = float(10 ** np.random.uniform(np.log10(0.5), np.log10(size / 2)))
            sigma_y = float(sigma_x * np.random.uniform(0.5, 3.0))
            theta = np.random.uniform(0, math.pi)
            params.append({"sigma_x": sigma_x, "sigma_y": sigma_y, "theta": theta, "size": size})
        elif family == "dog":
            s1 = float(np.random.uniform(0.5, size / 2))
            s2 = float(s1 * np.random.uniform(1.2, 3.0))
            theta = np.random.uniform(0, math.pi)
            params.append({"sigma1": s1, "sigma2": s2, "theta": theta, "size": size})
        elif family == "log":
            s = float(np.random.uniform(0.5, size / 2))
            theta = np.random.uniform(0, math.pi)
            params.append({"sigma": s, "theta": theta, "size": size})
        elif family in ("gabor", "gabor_filter"):
            sigma = float(np.random.uniform(0.5, size / 2))
            freq = float(np.random.uniform(0.02, 0.5))
            theta = np.random.uniform(0, math.pi)
            phase = float(np.random.uniform(0, 2 * math.pi))
            gamma = float(np.random.uniform(0.5, 1.5))
            params.append(
                {"sigma": sigma, "freq": freq, "theta": theta, "phase": phase, "gamma": gamma, "size": size}
            )
        elif family == "hog":
            sigma = float(np.random.uniform(0.5, size / 2))
            theta = np.random.uniform(0, math.pi)
            params.append({"sigma": sigma, "theta": theta, "size": size})
        elif family == "lbp":
            half = max(1, size // 2)
            radius = int(np.random.randint(1, half + 1))
            theta = np.random.uniform(0, 2 * math.pi)
            params.append({"radius": radius, "theta": theta, "size": size})
        elif family == "glcm":
            half = max(1, size // 2)
            radius = int(np.random.randint(1, half + 1))
            theta = np.random.uniform(0, math.pi)
            offset_weight = float(np.random.choice([-1.0, 1.0]))
            params.append(
                {"radius": radius, "theta": theta, "offset_weight": offset_weight, "size": size}
            )
        elif family == "mrf":
            half = max(1, size // 2)
            radius = int(np.random.randint(1, half + 1))
            beta = float(np.random.uniform(0.5, 2.0))
            neighborhood = str(np.random.choice(["cross", "full"]))
            params.append(
                {"radius": radius, "beta": beta, "neighborhood": neighborhood, "size": size}
            )
        else:
            raise ValueError(f"Unknown family '{family}'")
    return params


def build_kernel_bank(families: Sequence[str], n_per_family: int, size: int) -> List[Dict]:
    bank: List[Dict] = []
    for fam in families:
        params = sample_parameters(fam, n_per_family, size)
        for p in params:
            if fam in ("gaussian", "anisotropic_gaussian"):
                kernel = make_gaussian_kernel(p["size"], p["sigma_x"], p.get("sigma_y", None), p["theta"])
            elif fam == "dog":
                kernel = make_dog_kernel(p["size"], p["sigma1"], p["sigma2"], p["theta"])
            elif fam == "log":
                kernel = make_log_kernel(p["size"], p["sigma"], p["theta"])
            elif fam in ("gabor", "gabor_filter"):
                kernel = make_gabor_kernel(p["size"], p["sigma"], p["freq"], p["theta"], p["phase"], p["gamma"])
            elif fam == "hog":
                kernel = make_hog_kernel(p["size"], p["sigma"], p["theta"])
            elif fam == "lbp":
                kernel = make_lbp_kernel(p["size"], p["radius"], p["theta"])
            elif fam == "glcm":
                kernel = make_glcm_kernel(p["size"], p["radius"], p["theta"], p["offset_weight"])
            elif fam == "mrf":
                kernel = make_mrf_kernel(p["size"], p["radius"], p["beta"], p["neighborhood"])
            else:
                continue
            bank.append({"family": fam, "params": p, "kernel": kernel})
    return bank


def combine_kernels(
    kernels: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
    normalize: str = "l1",
) -> np.ndarray:
    """
    Combine multiple kernels into a single composite kernel via weighted sum.
    """
    if not kernels:
        raise ValueError("kernels is empty")
    base_shape = kernels[0].shape
    if any(k.shape != base_shape for k in kernels):
        raise ValueError("All kernels must have the same shape to combine.")
    if weights is None:
        weights_arr = np.ones(len(kernels), dtype=np.float32)
    else:
        weights_arr = np.asarray(weights, dtype=np.float32)
        if weights_arr.shape[0] != len(kernels):
            raise ValueError(f"weights length {len(weights_arr)} does not match kernels length {len(kernels)}")
    combined = np.zeros(base_shape, dtype=np.float32)
    for w, k in zip(weights_arr, kernels):
        combined += float(w) * k.astype(np.float32)
    combined = combined - combined.mean()
    if normalize == "l1":
        denom = np.abs(combined).sum()
        if denom > 0:
            combined = combined / denom
    elif normalize == "l2":
        denom = np.linalg.norm(combined)
        if denom > 0:
            combined = combined / denom
    elif normalize in ("none", None):
        pass
    else:
        raise ValueError(f"Unknown normalize option '{normalize}'. Use 'l1', 'l2', or 'none'.")
    return combined.astype(np.float32)


__all__ = [
    "build_kernel_bank",
    "combine_kernels",
    "make_dog_kernel",
    "make_gabor_kernel",
    "make_gaussian_kernel",
    "make_glcm_kernel",
    "make_hog_kernel",
    "make_lbp_kernel",
    "make_log_kernel",
    "make_mrf_kernel",
    "sample_parameters",
]
