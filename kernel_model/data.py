import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm


def get_malignant_mask(
    annot_img: np.ndarray,
    red_threshold: int = 150,
    low_threshold: int = 70,
) -> np.ndarray:
    """
    Build a binary mask from the malignant (red) contour in the annotation image.
    """
    r, g, b = cv2.split(annot_img)
    mask = ((g < low_threshold) & 
            (b < low_threshold) & 
            (r > red_threshold)).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = binary_fill_holes(mask).astype(np.uint8)
    mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)[:, :, 0] > 0
    return mask


def _safe_center_mask(mask: np.ndarray, pad: int) -> np.ndarray:
    center_mask = mask.copy()
    if pad > 0:
        center_mask[:pad, :] = False
        center_mask[-pad:, :] = False
        center_mask[:, :pad] = False
        center_mask[:, -pad:] = False
    return center_mask


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _extract_patient_id(path: Path) -> str:
    """
    Infer a patient/study id from a patch filename.
    Expected formats:
      - inside_IMG001_00001.tif -> IMG001
      - outside_patientA_01234.tif -> patientA
    Falls back to the full stem when no prefix is present.
    """
    stem = path.stem
    parts = stem.split("_")
    if parts and parts[0] in ("inside", "outside"):
        if len(parts) >= 3 and parts[-1].isdigit():
            mid = "_".join(parts[1:-1])
            return mid or parts[1]
        if len(parts) >= 2:
            return parts[1]
    return stem


def _assign_patient_splits(
    image_paths: Sequence[Path],
    train_frac: float,
    val_frac: float,
    test_frac: Optional[float],
    seed: int,
    enabled: bool = True,
) -> Dict[str, Optional[str]]:
    """
    Deterministically assign each patient (image stem) to a split.
    Returns a mapping of image stem -> split label ("train", "val", "test" or None).
    """
    if not enabled:
        return {p.stem: None for p in image_paths}
    if train_frac < 0 or val_frac < 0 or (test_frac is not None and test_frac < 0):
        raise ValueError("Split fractions must be non-negative.")
    remaining = 1.0 - train_frac - val_frac if test_frac is None else 1.0 - train_frac - val_frac - test_frac
    if remaining < -1e-6:
        raise ValueError("Split fractions must sum to <= 1.0.")
    test_frac_eff = test_frac if test_frac is not None else max(remaining, 0.0)

    rng = np.random.default_rng(seed)
    n = len(image_paths)
    order = rng.permutation(n)
    train_n = int(round(train_frac * n))
    val_n = int(round(val_frac * n))
    test_n = int(round(test_frac_eff * n))
    total = train_n + val_n + test_n
    if total > n:
        test_n = max(0, test_n - (total - n))
    elif total < n:
        test_n += n - total

    split_map: Dict[str, Optional[str]] = {}
    for idx_pos, perm_idx in enumerate(order):
        stem = image_paths[perm_idx].stem
        if idx_pos < train_n:
            split = "train"
        elif idx_pos < train_n + val_n:
            split = "val"
        elif idx_pos < train_n + val_n + test_n:
            split = "test"
        else:
            split = "train"
        split_map[stem] = split
    return split_map


def sample_patches(
    img: np.ndarray,
    mask: Optional[np.ndarray],
    patch_size: int,
    n_inside: int,
    n_outside: int,
    max_tries: int = 500,
    min_pos_coverage: float = 0.5,
    max_neg_coverage: float = 0.05,
    near_neg_fraction: float = 0.5,
    near_neg_radius: Optional[int] = None,
    far_neg_radius: Optional[int] = None,
    use_bbox_for_positives: bool = True,
    min_nonzero_frac: float = 0.0,
    min_intensity_rel: float = 0.0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Sample inside (tumor) and outside (healthy) patches from an image.

    Positives require the patch center to be inside the lesion mask (or its bounding box)
    with at least ``min_pos_coverage`` lesion pixels inside the patch.
    Negatives are split into near- and far-lesion subsets using a distance transform.
    """
    height, width = img.shape[:2]
    pad = patch_size // 2
    rng = np.random.default_rng()
    near_neg_radius = near_neg_radius or patch_size
    far_neg_radius = far_neg_radius or max(patch_size * 2, near_neg_radius + patch_size)
    near_neg_fraction = float(np.clip(near_neg_fraction, 0.0, 1.0))
    min_pos_coverage = float(min_pos_coverage)
    max_neg_coverage = float(max_neg_coverage)

    mask_bool = mask.astype(bool) if mask is not None else None

    def patch_intensity_ok(patch: np.ndarray) -> bool:
        if min_nonzero_frac > 0.0 and (patch > 0).mean() < min_nonzero_frac:
            return False
        if min_intensity_rel > 0.0:
            if np.issubdtype(patch.dtype, np.integer):
                max_val = np.iinfo(patch.dtype).max
            else:
                max_val = 1.0
            if float(patch.mean()) < min_intensity_rel * float(max_val):
                return False
        return True

    def coverage_ok_positive(patch_mask: Optional[np.ndarray]) -> bool:
        return patch_mask is not None and patch_mask.mean() >= min_pos_coverage

    def coverage_ok_negative(patch_mask: Optional[np.ndarray]) -> bool:
        return patch_mask is None or patch_mask.mean() <= max_neg_coverage

    def sample_from_coords(coords: np.ndarray, target: int, coverage_check) -> List[np.ndarray]:
        collected: List[np.ndarray] = []
        if coords.size == 0 or target <= 0:
            return collected
        coords = coords.copy()
        rng.shuffle(coords)
        for cy, cx in coords:
            if len(collected) >= target:
                break
            x0 = int(cx - pad)
            y0 = int(cy - pad)
            if x0 < 0 or y0 < 0 or x0 + patch_size > width or y0 + patch_size > height:
                continue
            patch_mask = None
            if mask_bool is not None:
                patch_mask = mask_bool[y0 : y0 + patch_size, x0 : x0 + patch_size]
            if coverage_check and not coverage_check(patch_mask):
                continue
            patch = img[y0 : y0 + patch_size, x0 : x0 + patch_size]
            if not patch_intensity_ok(patch):
                continue
            collected.append(patch)
        return collected

    # Positive candidates: centers inside mask or bounding box, with lesion coverage check.
    if mask_bool is not None:
        pos_mask = _safe_center_mask(mask_bool, pad)
        if use_bbox_for_positives:
            bbox = _mask_bbox(mask_bool)
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                bbox_mask = np.zeros_like(mask_bool)
                bbox_mask[y_min : y_max + 1, x_min : x_max + 1] = True
                pos_mask = pos_mask | _safe_center_mask(bbox_mask, pad)
        pos_coords = np.argwhere(pos_mask)
    else:
        pos_coords = np.empty((0, 2), dtype=int)

    inside_patches = sample_from_coords(pos_coords, n_inside, coverage_ok_positive)
    if len(inside_patches) < n_inside and mask_bool is not None:
        tries = 0
        while len(inside_patches) < n_inside and tries < max_tries:
            x0 = rng.integers(0, max(1, width - patch_size + 1))
            y0 = rng.integers(0, max(1, height - patch_size + 1))
            patch_mask = mask_bool[y0 : y0 + patch_size, x0 : x0 + patch_size]
            if coverage_ok_positive(patch_mask):
                patch = img[y0 : y0 + patch_size, x0 : x0 + patch_size]
                if not patch_intensity_ok(patch):
                    tries += 1
                    continue
                inside_patches.append(patch)
            tries += 1

    # Negative candidates: centers outside lesion, split into near/far regions.
    outside_mask = np.ones((height, width), dtype=bool) if mask_bool is None else ~mask_bool
    outside_centers = _safe_center_mask(outside_mask, pad)
    dist_map = None
    if mask_bool is not None and mask_bool.any():
        dist_map = cv2.distanceTransform(outside_mask.astype(np.uint8), cv2.DIST_L2, 3)

    near_mask = np.zeros_like(outside_centers)
    far_mask = np.zeros_like(outside_centers)
    if dist_map is not None:
        near_mask = outside_centers & (dist_map > 0) & (dist_map <= near_neg_radius)
        far_mask = outside_centers & (dist_map >= far_neg_radius)

    near_coords = np.argwhere(near_mask)
    far_coords = np.argwhere(far_mask)
    outside_coords = np.argwhere(outside_centers)

    near_target = min(max(int(round(n_outside * near_neg_fraction)), 0), n_outside)
    far_target = n_outside - near_target

    outside_patches = sample_from_coords(near_coords, near_target, coverage_ok_negative)
    outside_patches.extend(sample_from_coords(far_coords, far_target, coverage_ok_negative))

    remaining = n_outside - len(outside_patches)
    if remaining > 0:
        outside_patches.extend(sample_from_coords(outside_coords, remaining, coverage_ok_negative))

    tries = 0
    while len(outside_patches) < n_outside and tries < max_tries:
        x0 = rng.integers(0, max(1, width - patch_size + 1))
        y0 = rng.integers(0, max(1, height - patch_size + 1))
        patch_mask = None
        if mask_bool is not None:
            patch_mask = mask_bool[y0 : y0 + patch_size, x0 : x0 + patch_size]
        if coverage_ok_negative(patch_mask):
            patch = img[y0 : y0 + patch_size, x0 : x0 + patch_size]
            if not patch_intensity_ok(patch):
                tries += 1
                continue
            outside_patches.append(patch)
        tries += 1

    return inside_patches, outside_patches


def load_patches_from_folders(
    root_dir: Path,
    max_per_class: Optional[int] = None,
    gray: bool = True,
    resize: Optional[Tuple[int, int]] = None,
    return_patient_ids: bool = False,
    splits: Optional[Sequence[str]] = None,
):
    """
    Load pre-extracted patches from ``root_dir/<split>/inside`` and ``root_dir/<split>/outside``.

    When ``splits`` is None, this will use ``root_dir/train`` if it exists, otherwise it falls back
    to the legacy flat layout under ``root_dir``. When ``return_patient_ids`` is True, also returns
    patient/study ids inferred from filenames (prefixed by split label when present).
    """
    root_dir = Path(root_dir)

    def resolve_splits() -> List[Tuple[Optional[str], Path]]:
        if splits is None:
            chosen: List[Optional[str]] = ["train"] if (root_dir / "train").exists() else [None]
        else:
            chosen = [splits] if isinstance(splits, (str, Path)) else list(splits)
            if len(chosen) == 0:
                chosen = ["train"] if (root_dir / "train").exists() else [None]
        resolved: List[Tuple[Optional[str], Path]] = []
        for split in chosen:
            if split is None:
                resolved.append((None, root_dir))
                continue
            split_dir = root_dir / str(split)
            if not split_dir.exists():
                if split == "train" and (root_dir / "inside").exists():
                    resolved.append((None, root_dir))
                    continue
                raise FileNotFoundError(f"Split '{split}' not found under {root_dir}")
            resolved.append((str(split), split_dir))
        return resolved

    def load_from(folder: Path, split_label: Optional[str]) -> Tuple[List[np.ndarray], List[str]]:
        files = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".jpg", ".jpeg", ".tiff")]
        )
        imgs: List[np.ndarray] = []
        patient_ids: List[str] = []
        for path in files:
            image = Image.open(path)
            image = image.convert("L") if gray else image.convert("RGB")
            if resize:
                image = image.resize(resize, Image.BILINEAR)
            arr = np.array(image, dtype=np.float32) / 255.0
            imgs.append(arr)
            pid = _extract_patient_id(path)
            if split_label:
                pid = f"{split_label}:{pid}"
            patient_ids.append(pid)
        return imgs, patient_ids

    inside_imgs: List[np.ndarray] = []
    outside_imgs: List[np.ndarray] = []
    inside_patients: List[str] = []
    outside_patients: List[str] = []

    split_dirs = resolve_splits()
    for split_label, base in split_dirs:
        inside_dir = base / "inside"
        outside_dir = base / "outside"
        os.makedirs(inside_dir, exist_ok=True)
        os.makedirs(outside_dir, exist_ok=True)

        split_inside, split_inside_pids = load_from(inside_dir, split_label)
        split_outside, split_outside_pids = load_from(outside_dir, split_label)
        inside_imgs.extend(split_inside)
        outside_imgs.extend(split_outside)
        inside_patients.extend(split_inside_pids)
        outside_patients.extend(split_outside_pids)

    if max_per_class is not None:
        inside_imgs = inside_imgs[:max_per_class]
        outside_imgs = outside_imgs[:max_per_class]
        inside_patients = inside_patients[:max_per_class]
        outside_patients = outside_patients[:max_per_class]

    def stack_or_empty(items: List[np.ndarray]) -> np.ndarray:
        if not items:
            if resize:
                return np.zeros((0, resize[1], resize[0]), dtype=np.float32)
            return np.zeros((0, 1, 1), dtype=np.float32)
        return np.stack(items, axis=0)

    Xin = stack_or_empty(inside_imgs)
    Xout = stack_or_empty(outside_imgs)
    if return_patient_ids:
        return Xin, Xout, inside_patients, outside_patients
    return Xin, Xout


def to_tensor_batch(X: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert an array of patches (N,H,W) into a torch tensor (N,1,H,W).
    """
    return torch.from_numpy(X).unsqueeze(1).to(device)


def extract_patches(
    healthy_dir: Path,
    malignant_dir: Path,
    annotation_dir: Path,
    output_dir: Path,
    patch_size: int,
    n_inside_per_image: int,
    n_outside_per_image: int,
    max_tries: int,
    red_threshold: int,
    low_threshold: int = 70,
    min_pos_coverage: float = 0.5,
    max_neg_coverage: float = 0.05,
    near_neg_fraction: float = 0.5,
    near_neg_radius: Optional[int] = None,
    far_neg_radius: Optional[int] = None,
    use_bbox_for_positives: bool = True,
    split_patients: bool = True,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: Optional[float] = 0.15,
    split_seed: int = 42,
    min_nonzero_frac: float = 0.0,
    min_intensity_rel: float = 0.0,
) -> Tuple[int, int]:
    """
    Extract patches from all images in ``healthy and malignant`` using optional annotations.

    Returns the count of inside and outside patches written to disk.
    """

    malignant_images = sorted(malignant_dir.glob("*.tif"))
    annotation_images = sorted(annotation_dir.glob("*.tif"))
    healthy_images = sorted(healthy_dir.glob("*.tif"))

    # annotated_pairs = [
    #     (malignant_dir / annot.name, annot)
    #     for annot in sorted(annotation_dir.glob("*.tif"))
    #     if (malignant_dir / annot.name).exists()
    # ]
    # annotated_stems = {p.stem for p, _ in annotated_pairs}
    # healthy_pairs = [
    #     (p, None)
    #     for p in sorted(healthy_dir.glob("*.tif"))
    #     if p.stem not in annotated_stems
    # ][: len(annotated_pairs)]

    # print(f"Found {len(annotated_pairs)} annotated malignant images.")
    # print(f"Found {len(healthy_pairs)} healthy images.")

    # image_entries = annotated_pairs + healthy_pairs
    # image_paths = [p for p, _ in image_entries]

    malignant_pairs = [
        (malignant_dir / annot.name, annot)
        for annot in sorted(annotation_dir.glob("*.tif"))
        if (malignant_dir / annot.name).exists()
    ]

    image_entries = malignant_pairs
    image_paths = [p for p, _ in image_entries]

    split_map = _assign_patient_splits(
        image_paths,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=split_seed,
        enabled=split_patients,
    )
    split_labels = sorted({s for s in split_map.values() if s is not None}) or [None]
    for split in split_labels:
        base = output_dir if split is None else output_dir / split
        (base / "inside").mkdir(parents=True, exist_ok=True)
        (base / "outside").mkdir(parents=True, exist_ok=True)

    inside_idx: Dict[Optional[str], int] = defaultdict(int)
    outside_idx: Dict[Optional[str], int] = defaultdict(int)

    for img_path, annot_path in tqdm(image_entries, total=len(image_entries), desc="Extracting patches"):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        annot = cv2.imread(str(annot_path), cv2.IMREAD_COLOR_RGB) if annot_path is not None else None

        mask = (
            get_malignant_mask(annot, red_threshold=red_threshold, low_threshold=low_threshold)
            if annot is not None
            else np.zeros(img.shape[:2], dtype=bool)
        )

        inside_target = n_inside_per_image if annot is not None else 0

        inside_patches, outside_patches = sample_patches(
            img,
            mask,
            patch_size,
            inside_target,
            n_outside_per_image,
            max_tries=max_tries,
            min_pos_coverage=min_pos_coverage,
            max_neg_coverage=max_neg_coverage,
            near_neg_fraction=near_neg_fraction,
            near_neg_radius=near_neg_radius,
            far_neg_radius=far_neg_radius,
            use_bbox_for_positives=use_bbox_for_positives,
            min_nonzero_frac=min_nonzero_frac,
            min_intensity_rel=min_intensity_rel,
        )

        split = split_map.get(img_path.stem)
        base = output_dir if split is None else output_dir / split
        split_key = split or "all"

        for patch in inside_patches:
            Image.fromarray(patch).save(
                base / "inside" / f"inside_{img_path.stem}_{inside_idx[split_key]:05d}.tif"
            )
            inside_idx[split_key] += 1

        for patch in outside_patches:
            Image.fromarray(patch).save(
                base / "outside" / f"outside_{img_path.stem}_{outside_idx[split_key]:05d}.tif"
            )
            outside_idx[split_key] += 1

    return sum(inside_idx.values()), sum(outside_idx.values())


__all__ = [
    "extract_patches",
    "load_patches_from_folders",
    "get_malignant_mask",
    "sample_patches",
    "to_tensor_batch",
]
