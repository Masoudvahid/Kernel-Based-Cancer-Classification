import math
import warnings
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


def _normalize_signed_shape(mask: np.ndarray) -> np.ndarray:
    inside = mask.astype(bool)
    outside = ~inside
    n_inside = int(inside.sum())
    n_outside = int(outside.sum())
    if n_inside == 0 or n_outside == 0:
        raise ValueError("Shape mask must occupy part, but not all, of the kernel window.")
    K = np.zeros(mask.shape, dtype=np.float32)
    K[inside] = 1.0 / n_inside
    K[outside] = -1.0 / n_outside
    K = K - K.mean()
    denom = np.abs(K).sum()
    if denom > 0:
        K = K / denom
    return K.astype(np.float32)


def make_square_kernel(size: int, side: float, theta: float = 0.0) -> np.ndarray:
    assert size % 2 == 1, "size should be odd"
    half = size // 2
    xs = np.arange(-half, half + 1, 1, dtype=np.float32)
    ys = np.arange(-half, half + 1, 1, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    ct = math.cos(theta)
    st = math.sin(theta)
    Xr = ct * X + st * Y
    Yr = -st * X + ct * Y
    half_side = max(0.5, min(float(side), float(size)) / 2.0)
    mask = (np.abs(Xr) <= half_side) & (np.abs(Yr) <= half_side)
    return _normalize_signed_shape(mask)


def make_triangular_kernel(size: int, base: float, height: float, theta: float = 0.0) -> np.ndarray:
    assert size % 2 == 1, "size should be odd"
    half = size // 2
    xs = np.arange(-half, half + 1, 1, dtype=np.float32)
    ys = np.arange(-half, half + 1, 1, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    ct = math.cos(theta)
    st = math.sin(theta)
    Xr = ct * X + st * Y
    Yr = -st * X + ct * Y

    base = max(1.0, min(float(base), float(size)))
    height = max(1.0, min(float(height), float(size)))
    y_min = -height / 2.0
    y_max = height / 2.0
    width = ((Yr - y_min) / max(height, 1e-12)) * (base / 2.0)
    inside = (Yr >= y_min) & (Yr <= y_max) & (np.abs(Xr) <= width)
    return _normalize_signed_shape(inside)


def _clip_unit_interval(u: float) -> float:
    # Keep values in [0, 1) so integer binning is stable even if samplers return edge values.
    return float(np.clip(u, 0.0, np.nextafter(1.0, 0.0)))


def _sample_uniform(u: float, low: float, high: float) -> float:
    u = _clip_unit_interval(u)
    return float(low + (high - low) * u)


def _sample_log_uniform(u: float, low: float, high: float) -> float:
    u = _clip_unit_interval(u)
    log_low = math.log10(low)
    log_high = math.log10(high)
    return float(10 ** (log_low + (log_high - log_low) * u))


def _sample_int_uniform(u: float, low: int, high: int) -> int:
    if high < low:
        raise ValueError(f"Invalid integer bounds: low={low} high={high}")
    u = _clip_unit_interval(u)
    span = high - low + 1
    return int(low + math.floor(u * span))


def _unit_samples(
    n_samples: int,
    dim: int,
    sampling_method: str = "random",
    seed: Optional[int] = None,
    qmc_scramble: bool = True,
) -> np.ndarray:
    method = (sampling_method or "random").lower()
    if dim <= 0:
        return np.zeros((n_samples, 0), dtype=np.float64)

    if method == "random":
        if seed is None:
            # Legacy behavior: use numpy global RNG when no explicit seed is provided.
            return np.random.random((n_samples, dim))
        rng = np.random.default_rng(seed)
        return rng.random((n_samples, dim))

    qmc_aliases = {"qmc", "sobol", "low_discrepancy", "low-discrepancy"}
    lhs_aliases = {"lhs", "latin_hypercube", "latin-hypercube"}
    if method not in qmc_aliases and method not in lhs_aliases:
        raise ValueError(
            f"Unknown sampling_method '{sampling_method}'. Use 'random', 'qmc', or 'lhs'."
        )

    try:
        from scipy.stats import qmc
    except Exception as exc:
        warnings.warn(
            f"SciPy qmc is unavailable ({exc}); falling back to random sampling.",
            RuntimeWarning,
        )
        if seed is None:
            return np.random.random((n_samples, dim))
        rng = np.random.default_rng(seed)
        return rng.random((n_samples, dim))

    if method in lhs_aliases:
        sampler = qmc.LatinHypercube(d=dim, scramble=qmc_scramble, seed=seed)
        return sampler.random(n=n_samples)

    sampler = qmc.Sobol(d=dim, scramble=qmc_scramble, seed=seed)
    # Prefer balance properties when n is a power-of-two; otherwise still support arbitrary n.
    if n_samples > 0 and (n_samples & (n_samples - 1)) == 0:
        return sampler.random_base2(m=int(math.log2(n_samples)))
    return sampler.random(n=n_samples)


def sample_parameters(
    family: str,
    n_samples: int,
    size: int,
    sampling_method: str = "random",
    sampling_seed: Optional[int] = None,
    qmc_scramble: bool = True,
) -> List[Dict]:
    """
    Sample kernel-family parameters using one of:
    - random: legacy iid uniform sampling
    - qmc: Sobol low-discrepancy sampling
    - lhs: Latin Hypercube sampling
    """
    params: List[Dict] = []

    if family == "gaussian":
        u = _unit_samples(
            n_samples,
            dim=2,
            sampling_method=sampling_method,
            seed=sampling_seed,
            qmc_scramble=qmc_scramble,
        )
        for ui in u:
            sigma = _sample_log_uniform(ui[0], 0.5, size / 2)
            theta = _sample_uniform(ui[1], 0.0, math.pi)
            params.append({"sigma_x": sigma, "sigma_y": sigma, "theta": theta, "size": size})
    elif family == "anisotropic_gaussian":
        u = _unit_samples(
            n_samples,
            dim=3,
            sampling_method=sampling_method,
            seed=sampling_seed,
            qmc_scramble=qmc_scramble,
        )
        for ui in u:
            sigma_x = _sample_log_uniform(ui[0], 0.5, size / 2)
            sigma_y = float(sigma_x * _sample_uniform(ui[1], 0.5, 3.0))
            theta = _sample_uniform(ui[2], 0.0, math.pi)
            params.append({"sigma_x": sigma_x, "sigma_y": sigma_y, "theta": theta, "size": size})
    elif family == "dog":
        u = _unit_samples(
            n_samples,
            dim=3,
            sampling_method=sampling_method,
            seed=sampling_seed,
            qmc_scramble=qmc_scramble,
        )
        for ui in u:
            s1 = _sample_uniform(ui[0], 0.5, size / 2)
            s2 = float(s1 * _sample_uniform(ui[1], 1.2, 3.0))
            theta = _sample_uniform(ui[2], 0.0, math.pi)
            params.append({"sigma1": s1, "sigma2": s2, "theta": theta, "size": size})
    elif family == "log":
        u = _unit_samples(
            n_samples,
            dim=2,
            sampling_method=sampling_method,
            seed=sampling_seed,
            qmc_scramble=qmc_scramble,
        )
        for ui in u:
            sigma = _sample_uniform(ui[0], 0.5, size / 2)
            theta = _sample_uniform(ui[1], 0.0, math.pi)
            params.append({"sigma": sigma, "theta": theta, "size": size})
    elif family in ("gabor", "gabor_filter"):
        u = _unit_samples(
            n_samples,
            dim=5,
            sampling_method=sampling_method,
            seed=sampling_seed,
            qmc_scramble=qmc_scramble,
        )
        for ui in u:
            sigma = _sample_uniform(ui[0], 0.5, size / 2)
            freq = _sample_uniform(ui[1], 0.02, 0.5)
            theta = _sample_uniform(ui[2], 0.0, math.pi)
            phase = _sample_uniform(ui[3], 0.0, 2 * math.pi)
            gamma = _sample_uniform(ui[4], 0.5, 1.5)
            params.append(
                {"sigma": sigma, "freq": freq, "theta": theta, "phase": phase, "gamma": gamma, "size": size}
            )
    elif family == "hog":
        u = _unit_samples(
            n_samples,
            dim=2,
            sampling_method=sampling_method,
            seed=sampling_seed,
            qmc_scramble=qmc_scramble,
        )
        for ui in u:
            sigma = _sample_uniform(ui[0], 0.5, size / 2)
            theta = _sample_uniform(ui[1], 0.0, math.pi)
            params.append({"sigma": sigma, "theta": theta, "size": size})
    elif family == "lbp":
        half = max(1, size // 2)
        u = _unit_samples(
            n_samples,
            dim=2,
            sampling_method=sampling_method,
            seed=sampling_seed,
            qmc_scramble=qmc_scramble,
        )
        for ui in u:
            radius = _sample_int_uniform(ui[0], 1, half)
            theta = _sample_uniform(ui[1], 0.0, 2 * math.pi)
            params.append({"radius": radius, "theta": theta, "size": size})
    elif family == "glcm":
        half = max(1, size // 2)
        u = _unit_samples(
            n_samples,
            dim=3,
            sampling_method=sampling_method,
            seed=sampling_seed,
            qmc_scramble=qmc_scramble,
        )
        for ui in u:
            radius = _sample_int_uniform(ui[0], 1, half)
            theta = _sample_uniform(ui[1], 0.0, math.pi)
            offset_weight = -1.0 if _clip_unit_interval(ui[2]) < 0.5 else 1.0
            params.append({"radius": radius, "theta": theta, "offset_weight": offset_weight, "size": size})
    elif family == "mrf":
        half = max(1, size // 2)
        u = _unit_samples(
            n_samples,
            dim=3,
            sampling_method=sampling_method,
            seed=sampling_seed,
            qmc_scramble=qmc_scramble,
        )
        for ui in u:
            radius = _sample_int_uniform(ui[0], 1, half)
            beta = _sample_uniform(ui[1], 0.5, 2.0)
            neighborhood = "cross" if _clip_unit_interval(ui[2]) < 0.5 else "full"
            params.append({"radius": radius, "beta": beta, "neighborhood": neighborhood, "size": size})
    elif family == "square":
        u = _unit_samples(
            n_samples,
            dim=2,
            sampling_method=sampling_method,
            seed=sampling_seed,
            qmc_scramble=qmc_scramble,
        )
        for ui in u:
            side = _sample_uniform(ui[0], max(3.0, size * 0.2), max(4.0, size * 0.75))
            theta = _sample_uniform(ui[1], 0.0, math.pi / 4.0)
            params.append({"side": side, "theta": theta, "size": size})
    elif family == "triangular":
        u = _unit_samples(
            n_samples,
            dim=3,
            sampling_method=sampling_method,
            seed=sampling_seed,
            qmc_scramble=qmc_scramble,
        )
        for ui in u:
            base = _sample_uniform(ui[0], max(3.0, size * 0.25), max(4.0, size * 0.85))
            height = _sample_uniform(ui[1], max(3.0, size * 0.25), max(4.0, size * 0.85))
            theta = _sample_uniform(ui[2], 0.0, 2.0 * math.pi)
            params.append({"base": base, "height": height, "theta": theta, "size": size})
    else:
        raise ValueError(f"Unknown family '{family}'")
    return params


def build_kernel_bank(
    families: Sequence[str],
    n_per_family: int,
    size: int,
    sampling_method: str = "random",
    sampling_seed: Optional[int] = None,
    qmc_scramble: bool = True,
) -> List[Dict]:
    if size <= 0:
        raise ValueError(f"Kernel size must be positive, got {size}.")
    if size % 2 == 0:
        raise ValueError(
            f"Kernel size must be odd, got {size}. "
            f"Use {size - 1} or {size + 1}."
        )
    bank: List[Dict] = []
    spawned_seeds: Optional[List[Optional[int]]] = None
    if sampling_seed is not None:
        seq = np.random.SeedSequence(int(sampling_seed))
        children = seq.spawn(len(families))
        spawned_seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]

    for i, fam in enumerate(families):
        fam_seed = spawned_seeds[i] if spawned_seeds is not None else None
        params = sample_parameters(
            fam,
            n_per_family,
            size,
            sampling_method=sampling_method,
            sampling_seed=fam_seed,
            qmc_scramble=qmc_scramble,
        )
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
            elif fam == "square":
                kernel = make_square_kernel(p["size"], p["side"], p["theta"])
            elif fam == "triangular":
                kernel = make_triangular_kernel(p["size"], p["base"], p["height"], p["theta"])
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
    "make_square_kernel",
    "make_triangular_kernel",
    "sample_parameters",
]
