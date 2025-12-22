from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def fisher_score(r_in: np.ndarray, r_out: np.ndarray) -> float:
    mu_in = r_in.mean() if r_in.size else 0.0
    mu_out = r_out.mean() if r_out.size else 0.0
    var_in = r_in.var(ddof=1) if r_in.size > 1 else 0.0
    var_out = r_out.var(ddof=1) if r_out.size > 1 else 0.0
    num = (mu_in - mu_out) ** 2
    den = var_in + var_out + 1e-12
    return float(num / den)


def auc_score(r_in: np.ndarray, r_out: np.ndarray) -> float:
    y_true = np.concatenate([np.ones(len(r_in)), np.zeros(len(r_out))])
    y_score = np.concatenate([r_in, r_out])
    try:
        if y_score.max() == y_score.min():
            return 0.5
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def pairwise_response_corr(resp_i: np.ndarray, resp_j: np.ndarray) -> float:
    a = resp_i - resp_i.mean()
    b = resp_j - resp_j.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)


def select_diverse(
    bank: Sequence[Dict],
    responses: Sequence[Dict[str, np.ndarray]],
    topM: int = 200,
    K: int = 20,
    lambda_mm: float = 0.75,
) -> List[int]:
    """
    Select top kernels using Maximal Marginal Relevance for diversity.
    """
    scores = []
    for i, resp in enumerate(responses):
        scores.append({"idx": i, "auc": auc_score(resp["r_in"], resp["r_out"]), "fisher": fisher_score(resp["r_in"], resp["r_out"])})
    scores = sorted(scores, key=lambda x: (x["auc"], x["fisher"]), reverse=True)
    top_idxs = [s["idx"] for s in scores[:topM]]

    combined = [np.concatenate([responses[i]["r_in"], responses[i]["r_out"]]) for i in top_idxs]
    selected_idxs: List[int] = []

    for idx in top_idxs:
        if not selected_idxs:
            selected_idxs.append(idx)
            if len(selected_idxs) >= K:
                break
            continue

        best_score = -1e9
        best_idx = None
        for cand in top_idxs:
            if cand in selected_idxs:
                continue
            auc_val = next(s for s in scores if s["idx"] == cand)["auc"]
            sims = [abs(pairwise_response_corr(combined[top_idxs.index(cand)], combined[top_idxs.index(sel)])) for sel in selected_idxs]
            max_sim = max(sims) if sims else 0.0
            mmr = lambda_mm * auc_val - (1 - lambda_mm) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = cand
        if best_idx is None:
            break
        selected_idxs.append(best_idx)
        if len(selected_idxs) >= K:
            break

    return selected_idxs


def select_abess(
    X_feat: np.ndarray,
    y: np.ndarray,
    K: int,
) -> List[int]:
    """
    Feature selection using abess (L0) on the candidate feature matrix.
    """
    if X_feat.size == 0 or y.size == 0 or K <= 0:
        return []
    try:
        from abess.linear import LogisticRegression
    except ImportError as exc:
        raise ImportError(
            "abess is not installed. Install with `pip install abess` to use abess selection."
        ) from exc

    try:
        model = LogisticRegression(support_size=K)
        model.fit(X_feat, y)
    except Exception as exc:  # pragma: no cover - depends on optional abess install
        raise RuntimeError(f"abess selection failed: {exc}") from exc
    coef = np.abs(model.coef_).ravel()
    selected = np.nonzero(coef)[0]
    if selected.size == 0:
        selected = np.argsort(-coef)[:K]
    return selected.tolist()


def build_feature_matrix(
    selected_idxs: Iterable[int],
    responses: Sequence[Dict[str, np.ndarray]],
    patient_ids_in: Optional[Sequence[str]] = None,
    patient_ids_out: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    features: List[np.ndarray] = []
    for idx in selected_idxs:
        resp = responses[idx]
        features.append(np.concatenate([resp["r_in"], resp["r_out"]]).reshape(-1, 1))
    if not features:
        return np.zeros((0, 0)), np.array([]), None
    X_feat = np.concatenate(features, axis=1)
    n_in = len(responses[0]["r_in"]) if responses else 0
    n_out = len(responses[0]["r_out"]) if responses else 0
    y = np.concatenate([np.ones(n_in), np.zeros(n_out)]) if (n_in or n_out) else np.array([])

    patient_ids = None
    if patient_ids_in is not None or patient_ids_out is not None:
        patient_ids_in = list(patient_ids_in) if patient_ids_in is not None else ["unknown"] * n_in
        patient_ids_out = list(patient_ids_out) if patient_ids_out is not None else ["unknown"] * n_out
        if len(patient_ids_in) != n_in or len(patient_ids_out) != n_out:
            raise ValueError(
                f"Patient id length mismatch: n_in={n_in} len(in_ids)={len(patient_ids_in)} "
                f"n_out={n_out} len(out_ids)={len(patient_ids_out)}"
            )
        patient_ids = np.array(list(patient_ids_in) + list(patient_ids_out))

    return X_feat, y, patient_ids


def rank_kernels(responses: Sequence[Dict[str, np.ndarray]]) -> List[Dict]:
    scores = []
    for i, resp in enumerate(responses):
        scores.append({"idx": i, "auc": auc_score(resp["r_in"], resp["r_out"]), "fisher": fisher_score(resp["r_in"], resp["r_out"])})
    scores.sort(key=lambda x: (x["auc"], x["fisher"]), reverse=True)
    return scores


__all__ = [
    "auc_score",
    "build_feature_matrix",
    "fisher_score",
    "pairwise_response_corr",
    "rank_kernels",
    "select_abess",
    "select_diverse",
]
