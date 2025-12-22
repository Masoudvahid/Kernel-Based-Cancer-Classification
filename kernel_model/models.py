from itertools import combinations
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .selection import pairwise_response_corr

class SimpleLogistic(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lin = torch.nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).squeeze(1)


class SimpleMLP(torch.nn.Module):
    def __init__(self, n_features: int, hidden_dims: Tuple[int, int] = (64, 32), dropout: float = 0.2):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, h1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(h1, h2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def standardize_split(X_tr: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_tr.mean(axis=0, keepdims=True)
    std = X_tr.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (X_tr - mean) / std, (X_val - mean) / std


def subsample_features(
    X_feat: np.ndarray,
    y: np.ndarray,
    subset_frac: Optional[float] = None,
    subset_size: Optional[int] = None,
    seed: int = 42,
    groups: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Randomly subsample rows of a feature matrix for faster experiments.
    """
    n = len(y)
    groups_arr = np.asarray(groups) if groups is not None else None
    if groups_arr is not None and len(groups_arr) != n:
        raise ValueError(f"groups length {len(groups_arr)} does not match y length {n}")
    if subset_frac is None and subset_size is None:
        return X_feat, y, groups_arr
    if n == 0:
        return X_feat, y, groups_arr
    target = subset_size if subset_size is not None else int(np.ceil(n * float(subset_frac)))
    target = min(max(target, 0), n)
    if target == 0:
        return X_feat[:0], y[:0], groups_arr[:0] if groups_arr is not None else None
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=target, replace=False)
    groups_sub = groups_arr[idx] if groups_arr is not None else None
    return X_feat[idx], y[idx], groups_sub


def _build_model(
    n_features: int,
    model_type: str,
    hidden_dims: Tuple[int, int],
    dropout: float,
) -> torch.nn.Module:
    if model_type == "mlp":
        return SimpleMLP(n_features, hidden_dims=hidden_dims, dropout=dropout)
    return SimpleLogistic(n_features)


def train_binary_model(
    X_feat: np.ndarray,
    y: np.ndarray,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    return_model: bool = True,
    model_type: str = "logistic",
    hidden_dims: Tuple[int, int] = (64, 32),
    dropout: float = 0.2,
    standardize: bool = True,
    groups: Optional[Sequence[str]] = None,
    val_size: float = 0.2,
) -> Dict:
    """
    Train a simple binary classifier on feature matrix X_feat.
    """
    if X_feat.size == 0 or y.size == 0:
        return {"model": None, "auc": 0.5, "acc": 0.5}

    device = torch.device(device)
    groups_arr = None
    if groups is not None:
        groups_arr = np.asarray(groups)
        if len(groups_arr) != len(y):
            raise ValueError(f"groups length {len(groups_arr)} does not match labels length {len(y)}")

    if groups_arr is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
        train_idx, val_idx = next(splitter.split(X_feat, y, groups=groups_arr))
        if len(train_idx) == 0 or len(val_idx) == 0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_feat, y, test_size=val_size, stratify=y, random_state=42
            )
        else:
            X_tr, X_val = X_feat[train_idx], X_feat[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_feat, y, test_size=val_size, stratify=y, random_state=42
        )
    mean = std = None
    if standardize:
        mean = X_tr.mean(axis=0, keepdims=True)
        std = X_tr.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        X_tr = (X_tr - mean) / std
        X_val = (X_val - mean) / std

    tr_ds = TensorDataset(
        torch.from_numpy(X_tr).float().to(device),
        torch.from_numpy(y_tr).float().to(device),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float().to(device),
        torch.from_numpy(y_val).float().to(device),
    )
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = _build_model(X_feat.shape[1], model_type, hidden_dims, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_state: Optional[Dict] = None
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        ys = []
        ps = []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                ys.append(yb.cpu().numpy())
                ps.append(probs)
        ys = np.concatenate(ys)
        ps = np.concatenate(ps)
        try:
            auc = roc_auc_score(ys, ps)
        except Exception:
            auc = 0.5
        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    X_val_t = torch.from_numpy(X_val).float().to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(X_val_t)).cpu().numpy()
    acc = accuracy_score(y_val, (probs > 0.5).astype(int))
    try:
        auc = roc_auc_score(y_val, probs)
    except Exception:
        auc = 0.5
    return {
        "model": model if return_model else None,
        "auc": auc,
        "acc": acc,
        "val_probs": probs,
        "val_labels": y_val,
        "mean": mean,
        "std": std,
        "standardize": standardize,
    }


def find_best_subsets(
    X_feat: np.ndarray,
    y: np.ndarray,
    subset_sizes: Iterable[int],
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    model_type: str = "logistic",
    hidden_dims: Tuple[int, int] = (64, 32),
    dropout: float = 0.2,
    standardize: bool = True,
    corr_thresholds: Optional[Dict[int, float]] = None,
    groups: Optional[Sequence[str]] = None,
) -> Dict[int, Dict]:
    """
    Brute-force search over small subset sizes to find best AUC subsets, optionally enforcing a
    correlation ceiling per subset size via ``corr_thresholds`` (e.g., {3: 0.2}).
    """
    best: Dict[int, Dict] = {}
    n_features = X_feat.shape[1]
    for k in subset_sizes:
        if n_features < k:
            continue
        best_entry = None
        fallback_entry = None  # used when nothing meets correlation threshold
        combos = list(combinations(range(n_features), k))
        for combo in tqdm(combos, desc=f"Subset search k={k}", leave=False):
            max_abs_corr = 0.0
            if k >= 2:
                pair_corrs = [
                    abs(pairwise_response_corr(X_feat[:, a], X_feat[:, b]))
                    for a, b in combinations(combo, 2)
                ]
                max_abs_corr = max(pair_corrs) if pair_corrs else 0.0

            res = train_binary_model(
                X_feat[:, combo],
                y,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                return_model=False,
                model_type=model_type,
                hidden_dims=hidden_dims,
                dropout=dropout,
                standardize=standardize,
                groups=groups,
            )
            entry = {"subset": combo, "auc": res["auc"], "acc": res["acc"], "max_abs_corr": max_abs_corr}

            threshold = None
            if corr_thresholds is not None:
                threshold = corr_thresholds.get(k)
            eligible = threshold is None or max_abs_corr <= threshold

            if not eligible:
                # track the least-correlated option as a fallback when nothing meets the threshold
                if (
                    fallback_entry is None
                    or max_abs_corr < fallback_entry["max_abs_corr"] - 1e-9
                    or (
                        abs(max_abs_corr - fallback_entry["max_abs_corr"]) <= 1e-9
                        and entry["auc"] > fallback_entry["auc"]
                    )
                ):
                    fallback_entry = entry
                continue

            if (
                best_entry is None
                or entry["auc"] > best_entry["auc"] + 1e-9
                or (
                    abs(entry["auc"] - best_entry["auc"]) <= 1e-9
                    and entry["max_abs_corr"] < best_entry["max_abs_corr"]
                )
            ):
                best_entry = entry
        if best_entry:
            best_entry["threshold_met"] = True if corr_thresholds and corr_thresholds.get(k) is not None else None
            best[k] = best_entry
        elif fallback_entry:
            fallback_entry["threshold_met"] = False
            best[k] = fallback_entry
    return best


def find_best_low_corr_pair(
    X_feat: np.ndarray,
    y: np.ndarray,
    candidate_idxs: Sequence[int],
    corr_threshold: float = 0.1,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    model_type: str = "logistic",
    hidden_dims: Tuple[int, int] = (64, 32),
    dropout: float = 0.2,
    standardize: bool = True,
    refit_best: bool = False,
    groups: Optional[Sequence[str]] = None,
) -> Dict:
    """
    Iterate over kernel pairs with correlation below ``corr_threshold`` and return the best-performing pair.
    ``candidate_idxs`` maps columns of ``X_feat`` back to bank indices for reporting.
    """
    n_features = X_feat.shape[1] if X_feat.ndim == 2 else 0
    if X_feat.size == 0 or y.size == 0 or n_features < 2 or n_features != len(candidate_idxs):
        return {
            "subset": None,
            "kernel_idxs": [],
            "auc": 0.5,
            "acc": 0.5,
            "corr": None,
            "abs_corr": None,
            "threshold_met": False,
            "model": None,
        }

    best: Optional[Dict] = None
    have_within_threshold = False

    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr = pairwise_response_corr(X_feat[:, i], X_feat[:, j])
            abs_corr = abs(corr)
            eligible = abs_corr <= corr_threshold
            have_within_threshold = have_within_threshold or eligible
            if not eligible and have_within_threshold:
                # once we've seen at least one eligible pair, skip higher-corr pairs
                continue

            res = train_binary_model(
                X_feat[:, [i, j]],
                y,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                return_model=False,
                model_type=model_type,
                hidden_dims=hidden_dims,
                dropout=dropout,
                standardize=standardize,
                groups=groups,
            )
            entry = {
                "subset": (i, j),
                "kernel_idxs": [candidate_idxs[i], candidate_idxs[j]],
                "auc": res["auc"],
                "acc": res["acc"],
                "corr": corr,
                "abs_corr": abs_corr,
            }
            if best is None:
                best = entry
                continue
            if abs_corr < best["abs_corr"] - 1e-9:
                best = entry
            elif abs(abs_corr - best["abs_corr"]) <= 1e-9 and entry["auc"] > best["auc"]:
                best = entry

    if best is None:
        return {
            "subset": None,
            "kernel_idxs": [],
            "auc": 0.5,
            "acc": 0.5,
            "corr": None,
            "abs_corr": None,
            "threshold_met": False,
            "model": None,
        }

    best["threshold_met"] = have_within_threshold

    if refit_best:
        res = train_binary_model(
            X_feat[:, list(best["subset"])],
            y,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            return_model=True,
            model_type=model_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
            standardize=standardize,
            groups=groups,
        )
        best["auc"] = res["auc"]
        best["acc"] = res["acc"]
        best["model"] = res["model"]
    else:
        best["model"] = None

    return best


def fit_subset_model(
    X_feat: np.ndarray,
    y: np.ndarray,
    subset: Sequence[int],
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    model_type: str = "logistic",
    hidden_dims: Tuple[int, int] = (64, 32),
    dropout: float = 0.2,
    standardize: bool = True,
) -> Dict:
    X_sub = X_feat[:, subset]
    mean = std = None
    if standardize:
        mean = X_sub.mean(axis=0, keepdims=True)
        std = X_sub.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        X_sub = (X_sub - mean) / std
    device = torch.device(device)
    ds = TensorDataset(
        torch.from_numpy(X_sub).float().to(device),
        torch.from_numpy(y).float().to(device),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = _build_model(X_sub.shape[1], model_type, hidden_dims, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return {
        "model": model,
        "mean": mean,
        "std": std,
        "device": device,
        "standardize": standardize,
    }


def train_classifier(
    Xin: np.ndarray,
    Xout: np.ndarray,
    selected_idxs: Sequence[int],
    responses: Sequence[Dict[str, np.ndarray]],
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    model_type: str = "logistic",
    hidden_dims: Tuple[int, int] = (64, 32),
    dropout: float = 0.2,
    standardize: bool = True,
    subset_frac: Optional[float] = None,
    subset_size: Optional[int] = None,
    subset_seed: int = 42,
    patient_ids_in: Optional[Sequence[str]] = None,
    patient_ids_out: Optional[Sequence[str]] = None,
) -> Dict:
    from .selection import build_feature_matrix

    X_feat, y, patient_ids = build_feature_matrix(
        selected_idxs,
        responses,
        patient_ids_in=patient_ids_in,
        patient_ids_out=patient_ids_out,
    )
    X_feat, y, patient_ids = subsample_features(
        X_feat,
        y,
        groups=patient_ids,
        subset_frac=subset_frac,
        subset_size=subset_size,
        seed=subset_seed,
    )
    return train_binary_model(
        X_feat,
        y,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        model_type=model_type,
        hidden_dims=hidden_dims,
        dropout=dropout,
        standardize=standardize,
        groups=patient_ids,
    )


__all__ = [
    "SimpleLogistic",
    "SimpleMLP",
    "fit_subset_model",
    "find_best_low_corr_pair",
    "find_best_subsets",
    "standardize_split",
    "subsample_features",
    "train_binary_model",
    "train_classifier",
]
