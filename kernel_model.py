import argparse
from pathlib import Path

from kernel_model.config import (
    KernelBankConfig,
    PatchExtractionConfig,
    PipelineConfig,
    SelectionConfig,
    SubsetSearchConfig,
    TrainingConfig,
)
from kernel_model.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kernel-based cancer classification pipeline")
    # data
    parser.add_argument("--image-dir", type=Path, default=Path("data/TIFF Images/all"), help="Directory with source TIFF images")
    parser.add_argument("--annotation-dir", type=Path, default=Path("data/Pixel-level annotation"), help="Directory with annotation TIFFs")
    parser.add_argument("--data-root", type=Path, default=Path("data/patches"), help="Root containing inside/outside patch folders")
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="Directory to save outputs")
    parser.add_argument("--patch-size", type=int, default=128, help="Patch size to extract from raw images")
    parser.add_argument("--resize-patch-size", type=int, default=64, help="Patch size to resize to for model input")
    parser.add_argument("--n-inside-per-image", type=int, default=50, help="Patches per image inside tumor")
    parser.add_argument("--n-outside-per-image", type=int, default=50, help="Patches per image outside tumor")
    parser.add_argument("--green-threshold", type=int, default=150, help="Threshold for red/green contour extraction")
    parser.add_argument("--max-tries", type=int, default=500, help="Max attempts per image when sampling patches")
    parser.add_argument("--min-pos-coverage", type=float, default=0.5, help="Minimum lesion fraction required in a positive patch")
    parser.add_argument("--max-neg-coverage", type=float, default=0.05, help="Maximum lesion fraction allowed in a negative patch")
    parser.add_argument("--near-neg-fraction", type=float, default=0.5, help="Fraction of negatives sampled near lesion boundary")
    parser.add_argument("--near-neg-radius", type=int, default=None, help="Radius (pixels) for near-lesion negatives; defaults to patch size when unset")
    parser.add_argument("--far-neg-radius", type=int, default=None, help="Minimum distance (pixels) for far negatives; defaults to 2x patch size when unset")
    parser.add_argument("--no-bbox-positives", action="store_true", help="Disable sampling positives whose centers are only inside the lesion bounding box")
    parser.add_argument("--min-nonzero-frac", type=float, default=0.0, help="Minimum fraction of nonzero pixels required to keep a patch")
    parser.add_argument("--min-intensity-rel", type=float, default=0.0, help="Minimum mean intensity as a fraction of dtype max to keep a patch (0-1)")
    parser.add_argument("--no-split-patients", action="store_true", help="Disable patient-level train/val/test partition when extracting patches")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Train fraction for patient-level split when extracting patches")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction for patient-level split when extracting patches")
    parser.add_argument("--test-frac", type=float, default=0.15, help="Test fraction for patient-level split when extracting patches")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for patient split")
    parser.add_argument("--max-per-class", type=int, default=2000, help="Maximum patches to load per class")
    parser.add_argument("--load-splits", type=str, default="train,val,test", help="Comma-separated splits to load (e.g., 'train' or 'train,val')")
    parser.add_argument("--skip-extract", action="store_true", help="Skip patch extraction step and use existing patches")
    # kernels
    parser.add_argument(
        "--families",
        type=str,
        default="gaussian,anisotropic_gaussian,dog,log,gabor",
        help="Comma-separated kernel families",
    )
    parser.add_argument("--n-per-family", type=int, default=200, help="Kernels per family")
    parser.add_argument("--kernel-size", type=int, default=31, help="Kernel spatial size (odd)")
    # selection/training
    parser.add_argument("--selection-method", type=str, default="mmr", choices=["mmr", "abess"], help="Kernel selector: mmr or abess")
    parser.add_argument("--response-fn", type=str, default="mean_abs", choices=["mean_abs", "abs_max"], help="Response aggregation")
    parser.add_argument("--topM", type=int, default=200, help="Top kernels to consider before diversity selection")
    parser.add_argument("--K", type=int, default=20, help="Number of kernels to keep after diversity selection")
    parser.add_argument("--lambda-mm", type=float, default=0.75, help="MMR tradeoff between relevance and diversity")
    parser.add_argument("--plot-top-kernels", type=int, default=20, help="Top kernels to visualize/search subsets on")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs for classifier")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--model-type", type=str, default="mlp", choices=["mlp", "logistic"], help="Classifier architecture")
    parser.add_argument("--hidden-dim1", type=int, default=64, help="First hidden layer width")
    parser.add_argument("--hidden-dim2", type=int, default=32, help="Second hidden layer width")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for MLP")
    parser.add_argument("--no-standardize", action="store_true", help="Disable feature standardization before training")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--train-subset-frac", type=float, default=None, help="Optional fraction of data to train/select on (e.g., 0.25)")
    parser.add_argument("--train-subset-size", type=int, default=None, help="Optional fixed number of samples to train/select on")
    parser.add_argument("--subset-seed", type=int, default=42, help="Seed for subsampling")
    # subset search
    parser.add_argument("--subset-search-epochs", type=int, default=20, help="Epochs for brute-force subset search models")
    parser.add_argument("--boundary-epochs", type=int, default=25, help="Epochs for boundary refit on best pair")
    parser.add_argument("--pair-corr-threshold", type=float, default=0.1, help="Max |corr| allowed when searching best kernel pair")
    parser.add_argument("--device", type=str, default=None, help="Manually set device (cpu/cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = PatchExtractionConfig(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.data_root,
        patch_size=args.patch_size,
        n_inside_per_image=args.n_inside_per_image,
        n_outside_per_image=args.n_outside_per_image,
        green_threshold=args.green_threshold,
        max_tries=args.max_tries,
        min_pos_coverage=args.min_pos_coverage,
        max_neg_coverage=args.max_neg_coverage,
        near_neg_fraction=args.near_neg_fraction,
        near_neg_radius=args.near_neg_radius,
        far_neg_radius=args.far_neg_radius,
        use_bbox_for_positives=not args.no_bbox_positives,
        min_nonzero_frac=args.min_nonzero_frac,
        min_intensity_rel=args.min_intensity_rel,
        split_patients=not args.no_split_patients,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        split_seed=args.split_seed,
        run_extraction=not args.skip_extract,
    )
    bank_cfg = KernelBankConfig(
        families=[f.strip() for f in args.families.split(",") if f.strip()],
        n_per_family=args.n_per_family,
        kernel_size=args.kernel_size,
    )
    sel_cfg = SelectionConfig(
        method=args.selection_method,
        response_fn=args.response_fn,
        topM=args.topM,
        K=args.K,
        lambda_mm=args.lambda_mm,
        plot_top_kernels=args.plot_top_kernels,
    )
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_type=args.model_type,
        hidden_dims=(args.hidden_dim1, args.hidden_dim2),
        dropout=args.dropout,
        standardize_features=not args.no_standardize,
        force_cpu=args.force_cpu,
        train_subset_frac=args.train_subset_frac,
        train_subset_size=args.train_subset_size,
        subset_seed=args.subset_seed,
    )
    subset_cfg = SubsetSearchConfig(
        subset_search_epochs=args.subset_search_epochs,
        boundary_epochs=args.boundary_epochs,
        pair_corr_threshold=args.pair_corr_threshold,
    )
    cfg = PipelineConfig(
        data=data_cfg,
        bank=bank_cfg,
        selection=sel_cfg,
        training=train_cfg,
        subset=subset_cfg,
        data_root=args.data_root,
        out_dir=args.out_dir,
        max_per_class=args.max_per_class,
        resize_patch_size=args.resize_patch_size,
        device=args.device,
        load_splits=tuple(s.strip() for s in args.load_splits.split(",") if s.strip()),
    )
    result = run_pipeline(cfg)
    print(f"Done. AUC={result['clf']['auc']:.4f} ACC={result['clf']['acc']:.4f}. Saved to {cfg.out_dir}")


if __name__ == "__main__":
    main()
