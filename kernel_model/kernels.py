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
        elif family == "gabor":
            sigma = float(np.random.uniform(0.5, size / 2))
            freq = float(np.random.uniform(0.02, 0.5))
            theta = np.random.uniform(0, math.pi)
            phase = float(np.random.uniform(0, 2 * math.pi))
            gamma = float(np.random.uniform(0.5, 1.5))
            params.append(
                {"sigma": sigma, "freq": freq, "theta": theta, "phase": phase, "gamma": gamma, "size": size}
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
            elif fam == "gabor":
                kernel = make_gabor_kernel(p["size"], p["sigma"], p["freq"], p["theta"], p["phase"], p["gamma"])
            else:
                continue
            bank.append({"family": fam, "params": p, "kernel": kernel})
    return bank


__all__ = [
    "build_kernel_bank",
    "make_dog_kernel",
    "make_gabor_kernel",
    "make_gaussian_kernel",
    "make_log_kernel",
    "sample_parameters",
]
