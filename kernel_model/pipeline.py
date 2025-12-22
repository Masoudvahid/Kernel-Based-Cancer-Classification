import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from .config import PipelineConfig
from .data import extract_patches, load_patches_from_folders
from .kernels import build_kernel_bank
from .models import (
    fit_subset_model,
    find_best_low_corr_pair,
    find_best_subsets,
    subsample_features,
    train_classifier,
)
from .plots import plot_2d_scatter, plot_3d_scatter, plot_confusion, plot_roc_pr
from .responses import compute_responses
from .selection import build_feature_matrix, rank_kernels, select_abess, select_diverse


def _resolve_device(cfg: PipelineConfig) -> str:
    if cfg.device:
        return cfg.device
    if not cfg.training.force_cpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _next_experiment_dir(base: Path, prefix: str = "exp_", digits: int = 3) -> Path:
    """
    Return a child directory under ``base`` named with the next experiment id (exp_001, exp_002, ...).
    If the provided base already looks like an experiment directory (starts with prefix), return it unchanged.
    """
    if base.name.startswith(prefix):
        return base
    base.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            suffix = p.name[len(prefix) :]
            if suffix.isdigit():
                existing.append(int(suffix))
    next_id = max(existing, default=0) + 1
    return base / f"{prefix}{next_id:0{digits}d}"


def _setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("kernel_model")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(out_dir / "run.log")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def run_pipeline(cfg: PipelineConfig) -> Dict:
    """
    End-to-end pipeline for training the kernel model.
    """
    out_dir = _next_experiment_dir(cfg.out_dir)
    cfg.out_dir = out_dir 
    logger = _setup_logger(out_dir)
    device = _resolve_device(cfg)
    logger.info("Starting pipeline. Output: %s Device: %s Selection: %s", out_dir, device, cfg.selection.method)

    if cfg.data.run_extraction:
        logger.info("Extracting patches from %s", cfg.data.image_dir)
        inside_count, outside_count = extract_patches(
            image_dir=cfg.data.image_dir,
            annotation_dir=cfg.data.annotation_dir,
            output_dir=cfg.data.output_dir,
            patch_size=cfg.data.patch_size,
            n_inside_per_image=cfg.data.n_inside_per_image,
            n_outside_per_image=cfg.data.n_outside_per_image,
            max_tries=cfg.data.max_tries * 20,
            red_threshold=cfg.data.green_threshold,
            min_pos_coverage=cfg.data.min_pos_coverage,
            max_neg_coverage=cfg.data.max_neg_coverage,
            near_neg_fraction=cfg.data.near_neg_fraction,
            near_neg_radius=cfg.data.near_neg_radius,
            far_neg_radius=cfg.data.far_neg_radius,
            use_bbox_for_positives=cfg.data.use_bbox_for_positives,
            split_patients=cfg.data.split_patients,
            train_frac=cfg.data.train_frac,
            val_frac=cfg.data.val_frac,
            test_frac=cfg.data.test_frac,
            split_seed=cfg.data.split_seed,
            min_nonzero_frac=cfg.data.min_nonzero_frac,
            min_intensity_rel=cfg.data.min_intensity_rel,
        )
        logger.info("Saved patches: inside=%s outside=%s", inside_count, outside_count)

    splits_to_load = tuple(cfg.load_splits) if cfg.load_splits else ("train",)
    if len(splits_to_load) == 0:
        splits_to_load = ("train",)
    train_split = "train" if "train" in splits_to_load else splits_to_load[0]
    eval_splits = [s for s in splits_to_load if s != train_split]

    def _load_split(split: str):
        return load_patches_from_folders(
            cfg.data_root,
            max_per_class=cfg.max_per_class,
            resize=(cfg.resize_patch_size, cfg.resize_patch_size),
            splits=(split,),
            return_patient_ids=True,
        )

    Xin, Xout, patients_in, patients_out = _load_split(train_split)
    logger.info(
        "Loaded %s split: Xin=%s Xout=%s resize=%s (patients in=%s out=%s)",
        train_split,
        len(Xin),
        len(Xout),
        cfg.resize_patch_size,
        len(set(patients_in)),
        len(set(patients_out)),
    )

    eval_split_data: Dict[str, tuple] = {}
    for split in eval_splits:
        try:
            split_data = _load_split(split)
            eval_split_data[split] = split_data
            logger.info(
                "Loaded %s split: Xin=%s Xout=%s resize=%s (patients in=%s out=%s)",
                split,
                len(split_data[0]),
                len(split_data[1]),
                cfg.resize_patch_size,
                len(set(split_data[2])),
                len(set(split_data[3])),
            )
        except FileNotFoundError:
            logger.warning("Split '%s' not found under %s; skipping.", split, cfg.data_root)

    bank = build_kernel_bank(cfg.bank.families, cfg.bank.n_per_family, cfg.bank.kernel_size)
    logger.info("Built kernel bank: %s kernels", len(bank))
    responses = compute_responses(
        bank,
        Xin,
        Xout,
        device=device,
        batch_size=cfg.training.batch_size,
        response_fn=cfg.selection.response_fn,
    )

    kernel_scores = rank_kernels(responses)
    candidate_kernel_idxs = [s["idx"] for s in kernel_scores[: cfg.selection.topM]]
    X_candidates, y_labels, patient_labels = build_feature_matrix(
        candidate_kernel_idxs,
        responses,
        patient_ids_in=patients_in,
        patient_ids_out=patients_out,
    )
    if patient_labels is not None:
        logger.info("Patient-level grouping enabled (%s unique ids)", len(np.unique(patient_labels)))

    X_for_selection, y_for_selection, patients_for_selection = subsample_features(
        X_candidates,
        y_labels,
        groups=patient_labels,
        subset_frac=cfg.training.train_subset_frac,
        subset_size=cfg.training.train_subset_size,
        seed=cfg.training.subset_seed,
    )

    method = cfg.selection.method.lower()

    if method == "abess":
        selected_subset = select_abess(X_for_selection, y_for_selection, cfg.selection.K)
        selected_idxs = [candidate_kernel_idxs[i] for i in selected_subset]
        logger.info("abess selected kernels (candidate idx): %s", selected_subset)
    elif method == "mmr":
        selected_idxs = select_diverse(
            bank,
            responses,
            topM=cfg.selection.topM,
            K=cfg.selection.K,
            lambda_mm=cfg.selection.lambda_mm,
        )
        logger.info("MMR selected kernels (bank idx): %s", selected_idxs)
    else:
        raise ValueError(f"Unknown selection method '{cfg.selection.method}'. Use 'mmr' or 'abess'.")

    clf_res = train_classifier(
        Xin,
        Xout,
        selected_idxs,
        responses,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        device=device,
        model_type=cfg.training.model_type,
        hidden_dims=cfg.training.hidden_dims,
        dropout=cfg.training.dropout,
        standardize=cfg.training.standardize_features,
        subset_frac=cfg.training.train_subset_frac,
        subset_size=cfg.training.train_subset_size,
        subset_seed=cfg.training.subset_seed,
        patient_ids_in=patients_in,
        patient_ids_out=patients_out,
    )
    logger.info("Classifier results: AUC=%.4f ACC=%.4f", clf_res.get("auc", 0.0), clf_res.get("acc", 0.0))

    plot_roc_pr(clf_res.get("val_labels"), clf_res.get("val_probs"), out_dir / "clf")
    plot_confusion(clf_res.get("val_labels"), clf_res.get("val_probs"), out_dir / "confusion.png")

    eval_results: Dict[str, Dict] = {}
    model = clf_res.get("model")
    model_device = next(model.parameters()).device if model is not None else None

    def _eval_split(split_name: str, split_data: tuple) -> Dict:
        if model is None:
            return {"auc": 0.5, "acc": 0.5, "n": 0, "n_pos": 0, "n_neg": 0}
        Xin_split, Xout_split, patients_in_split, patients_out_split = split_data
        responses_split = compute_responses(
            bank,
            Xin_split,
            Xout_split,
            device=device,
            batch_size=cfg.training.batch_size,
            response_fn=cfg.selection.response_fn,
        )
        X_feat, y_true, _ = build_feature_matrix(
            selected_idxs,
            responses_split,
            patient_ids_in=patients_in_split,
            patient_ids_out=patients_out_split,
        )
        if X_feat.size == 0 or y_true.size == 0:
            return {"auc": 0.5, "acc": 0.5, "n": 0, "n_pos": 0, "n_neg": 0}
        X_eval = X_feat
        mean = clf_res.get("mean")
        std = clf_res.get("std")
        if clf_res.get("standardize") and mean is not None and std is not None:
            X_eval = (X_eval - mean) / std
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(X_eval).float().to(model_device))).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = 0.5
        acc = accuracy_score(y_true, preds)
        return {
            "auc": float(auc),
            "acc": float(acc),
            "n": int(len(y_true)),
            "n_pos": int(np.sum(y_true)),
            "n_neg": int(len(y_true) - np.sum(y_true)),
        }

    for split, split_data in eval_split_data.items():
        res = _eval_split(split, split_data)
        eval_results[split] = res
        logger.info("Eval %s: AUC=%.4f ACC=%.4f", split, res.get("auc", 0.0), res.get("acc", 0.0))

    plot_candidate_idxs = candidate_kernel_idxs[: cfg.selection.plot_top_kernels]
    X_plot, y_plot, patient_plot = build_feature_matrix(
        plot_candidate_idxs,
        responses,
        patient_ids_in=patients_in,
        patient_ids_out=patients_out,
    )

    X_for_search, y_for_search, patients_for_search = subsample_features(
        X_plot,
        y_plot,
        groups=patient_plot,
        subset_frac=cfg.training.train_subset_frac,
        subset_size=cfg.training.train_subset_size,
        seed=cfg.training.subset_seed,
    )

    subset_results = find_best_subsets(
        X_for_search,
        y_for_search,
        subset_sizes=[2, 3],
        epochs=cfg.subset.subset_search_epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        device=device,
        model_type=cfg.training.model_type,
        hidden_dims=cfg.training.hidden_dims,
        dropout=cfg.training.dropout,
        standardize=cfg.training.standardize_features,
        corr_thresholds={3: cfg.subset.triple_corr_threshold},
        groups=patients_for_search,
    )

    low_corr_pair = find_best_low_corr_pair(
        X_for_search,
        y_for_search,
        plot_candidate_idxs,
        corr_threshold=cfg.subset.pair_corr_threshold,
        epochs=cfg.subset.subset_search_epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        device=device,
        model_type=cfg.training.model_type,
        hidden_dims=cfg.training.hidden_dims,
        dropout=cfg.training.dropout,
        standardize=cfg.training.standardize_features,
        refit_best=False,  # keep JSON-serializable
        groups=patients_for_search,
    )
    if low_corr_pair.get("subset") is not None:
        if low_corr_pair.get("threshold_met"):
            logger.info(
                "Best low-corr pair (|corr|<=%.3f): kernels %s (cols %s) AUC=%.4f ACC=%.4f Corr=%.4f",
                cfg.subset.pair_corr_threshold,
                low_corr_pair["kernel_idxs"],
                low_corr_pair["subset"],
                low_corr_pair["auc"],
                low_corr_pair["acc"],
                low_corr_pair["corr"],
            )
        else:
            logger.info(
                "No pair met |corr|<=%.3f; best available (min corr=%.4f) kernels %s (cols %s) AUC=%.4f ACC=%.4f Corr=%.4f",
                cfg.subset.pair_corr_threshold,
                low_corr_pair.get("abs_corr", 0.0),
                low_corr_pair["kernel_idxs"],
                low_corr_pair["subset"],
                low_corr_pair["auc"],
                low_corr_pair["acc"],
                low_corr_pair["corr"],
            )
    else:
        logger.info("No kernel pair met correlation threshold %.3f", cfg.subset.pair_corr_threshold)

    pair_plot = out_dir / "scatter_best_pair.png"
    triple_plot = out_dir / "scatter_best_triple.png"
    if 2 in subset_results:
        pair = subset_results[2]
        pair_kernel_idxs = [plot_candidate_idxs[i] for i in pair["subset"]]
        logger.info("Best pair (plot candidate idx): %s AUC=%.4f", pair["subset"], pair["auc"])
        boundary_model = fit_subset_model(
            X_for_search,
            y_for_search,
            pair["subset"],
            epochs=cfg.subset.boundary_epochs,
            batch_size=cfg.training.batch_size,
            lr=cfg.training.lr,
            device=device,
            model_type=cfg.training.model_type,
            hidden_dims=cfg.training.hidden_dims,
            dropout=cfg.training.dropout,
            standardize=cfg.training.standardize_features,
        )
        plot_2d_scatter(
            X_for_search,
            y_for_search,
            pair["subset"],
            pair_kernel_idxs,
            pair_plot,
            boundary=boundary_model,
            title="Best 2-kernel feature space with decision boundary",
        )
    if 3 in subset_results:
        triple = subset_results[3]
        triple_kernel_idxs = [plot_candidate_idxs[i] for i in triple["subset"]]
        logger.info("Best triple (plot candidate idx): %s AUC=%.4f", triple["subset"], triple["auc"])
        plot_3d_scatter(X_for_search, y_for_search, triple["subset"], triple_kernel_idxs, triple_plot)

    out = {
        "selected_idxs": selected_idxs,
        "bank_meta": [{"family": b["family"], "params": b["params"]} for b in bank],
        "clf_auc": float(clf_res["auc"]),
        "clf_acc": float(clf_res["acc"]),
        "selection_candidate_idxs": candidate_kernel_idxs,
        "plot_candidate_idxs": plot_candidate_idxs,
        "subset_results": subset_results,
        "low_corr_pair": {k: v for k, v in low_corr_pair.items() if k != "model"},
        "selection_method": cfg.selection.method,
        "train_subset_frac": cfg.training.train_subset_frac,
        "train_subset_size": cfg.training.train_subset_size,
        "eval_results": eval_results,
    }

    np.savez(
        out_dir / "feature_cache.npz",
        X_candidates=X_plot,
        y_labels=y_plot,
        subset_results=subset_results,
        candidate_kernel_idxs=np.array(plot_candidate_idxs),
    )
    np.savez_compressed(out_dir / "results.npz", selected_idxs=np.array(selected_idxs))
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    np.save(out_dir / "kernels.npy", np.stack([b["kernel"] for b in bank], axis=0))
    logger.info("Saved artifacts to %s", out_dir)

    return {
        "config": cfg,
        "device": device,
        "clf": clf_res,
        "selected_idxs": selected_idxs,
        "subset_results": subset_results,
        "low_corr_pair": low_corr_pair,
        "candidate_kernel_idxs": candidate_kernel_idxs,
        "plot_candidate_idxs": plot_candidate_idxs,
        "selection_method": cfg.selection.method,
        "eval_results": eval_results,
        "out_dir": out_dir,
    }


__all__ = ["run_pipeline"]
