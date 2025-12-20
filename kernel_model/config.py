from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple


@dataclass
class PatchExtractionConfig:
    image_dir: Path = Path("data/TIFF Images/all")
    annotation_dir: Path = Path("data/Pixel-level annotation")
    output_dir: Path = Path("data/patches")
    patch_size: int = 128
    n_inside_per_image: int = 50
    n_outside_per_image: int = 100
    green_threshold: int = 150
    max_tries: int = 500
    run_extraction: bool = True


@dataclass
class KernelBankConfig:
    families: Sequence[str] = field(
        default_factory=lambda: (
            "gaussian",
            "anisotropic_gaussian",
            "dog",
            "log",
            "gabor",
        )
    )
    n_per_family: int = 200
    kernel_size: int = 31


@dataclass
class SelectionConfig:
    method: str = "mmr"  # "mmr" or "abess"
    response_fn: str = "mean_abs"
    topM: int = 200
    K: int = 20
    lambda_mm: float = 0.75
    plot_top_kernels: int = 20


@dataclass
class TrainingConfig:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 5e-4
    model_type: str = "mlp"  # mlp or logistic
    hidden_dims: Tuple[int, int] = (64, 32)
    dropout: float = 0.2
    standardize_features: bool = True
    force_cpu: bool = False
    train_subset_frac: Optional[float] = None  # e.g., 0.25 to train on 25%
    train_subset_size: Optional[int] = None    # or a fixed count to subsample
    subset_seed: int = 42


@dataclass
class SubsetSearchConfig:
    subset_search_epochs: int = 20
    boundary_epochs: int = 25
    pair_corr_threshold: float = 0.1
    triple_corr_threshold: float = 0.2


@dataclass
class PipelineConfig:
    data: PatchExtractionConfig = field(default_factory=PatchExtractionConfig)
    bank: KernelBankConfig = field(default_factory=KernelBankConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    subset: SubsetSearchConfig = field(default_factory=SubsetSearchConfig)
    data_root: Path = Path("data/patches")
    out_dir: Path = Path("./results")
    max_per_class: int = 2000
    resize_patch_size: int = 64
    device: Optional[str] = None  # auto-detect when None
