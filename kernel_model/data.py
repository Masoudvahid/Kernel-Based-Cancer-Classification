import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm


def mask_from_green_contour(
    annot_img: np.ndarray,
    red_threshold: int = 150,
    low_threshold: int = 70,
) -> np.ndarray:
    """
    Build a binary mask from the red contour in the annotation image.
    """
    r, g, b = cv2.split(annot_img)
    mask = ((g < low_threshold) & (b < low_threshold) & (r > red_threshold)).astype(
        np.uint8
    )
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = binary_fill_holes(mask).astype(np.uint8)
    mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)[:, :, 0] > 0
    return mask


def sample_patches(
    img: np.ndarray,
    mask: Optional[np.ndarray],
    patch_size: int,
    n_inside: int,
    n_outside: int,
    max_tries: int = 500,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Sample inside (tumor) and outside (healthy) patches from an image.
    """
    height, width = img.shape[:2]
    inside_patches, outside_patches = [], []

    if mask is not None:
        tries = 0
        while len(inside_patches) < n_inside and tries < max_tries:
            x = random.randint(0, width - patch_size)
            y = random.randint(0, height - patch_size)
            patch_mask = mask[y : y + patch_size, x : x + patch_size]
            if patch_mask.mean() > 0.8:
                patch = img[y : y + patch_size, x : x + patch_size]
                inside_patches.append(patch)
            tries += 1
    else:
        mask = np.zeros((*img.shape[:2], 3), dtype=np.uint8)

    tries = 0
    while len(outside_patches) < n_outside and tries < max_tries:
        x = random.randint(0, width - patch_size)
        y = random.randint(0, height - patch_size)
        patch_mask = mask[y : y + patch_size, x : x + patch_size]
        if patch_mask.mean() < 0.05:
            patch = img[y : y + patch_size, x : x + patch_size]
            outside_patches.append(patch)
        tries += 1

    return inside_patches, outside_patches


def load_patches_from_folders(
    root_dir: Path,
    max_per_class: Optional[int] = None,
    gray: bool = True,
    resize: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pre-extracted patches from ``root_dir/inside`` and ``root_dir/outside``.
    """
    inside_dir = Path(root_dir) / "inside"
    outside_dir = Path(root_dir) / "outside"
    os.makedirs(inside_dir, exist_ok=True)
    os.makedirs(outside_dir, exist_ok=True)

    def load_from(folder: Path, limit: Optional[int]) -> List[np.ndarray]:
        files = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".jpg", ".jpeg", ".tiff")]
        )
        if limit:
            files = files[:limit]
        imgs: List[np.ndarray] = []
        for path in files:
            image = Image.open(path)
            image = image.convert("L") if gray else image.convert("RGB")
            if resize:
                image = image.resize(resize, Image.BILINEAR)
            arr = np.array(image, dtype=np.float32) / 255.0
            imgs.append(arr)
        return imgs

    inside_imgs = load_from(inside_dir, max_per_class)
    outside_imgs = load_from(outside_dir, max_per_class)

    def stack_or_empty(items: List[np.ndarray]) -> np.ndarray:
        if not items:
            if resize:
                return np.zeros((0, resize[1], resize[0]), dtype=np.float32)
            return np.zeros((0, 1, 1), dtype=np.float32)
        return np.stack(items, axis=0)

    Xin = stack_or_empty(inside_imgs)
    Xout = stack_or_empty(outside_imgs)
    return Xin, Xout


def to_tensor_batch(X: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert an array of patches (N,H,W) into a torch tensor (N,1,H,W).
    """
    return torch.from_numpy(X).unsqueeze(1).to(device)


def extract_patches(
    image_dir: Path,
    annotation_dir: Path,
    output_dir: Path,
    patch_size: int,
    n_inside_per_image: int,
    n_outside_per_image: int,
    max_tries: int,
    red_threshold: int,
    low_threshold: int = 70,
) -> Tuple[int, int]:
    """
    Extract patches from all images in ``image_dir`` using optional annotations.

    Returns the count of inside and outside patches written to disk.
    """
    output_dir = Path(output_dir)
    (output_dir / "inside").mkdir(parents=True, exist_ok=True)
    (output_dir / "outside").mkdir(parents=True, exist_ok=True)

    image_paths = sorted(Path(image_dir).glob("*.tif"))

    inside_idx = 0
    outside_idx = 0

    for img_path in tqdm(image_paths, total=len(image_paths), desc="Extracting patches"):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        annot_path = Path(annotation_dir) / f"{img_path.stem}.tif"
        annot = None
        if annot_path.exists():
            annot = cv2.imread(str(annot_path), cv2.IMREAD_COLOR)

        mask = mask_from_green_contour(
            annot, red_threshold=red_threshold, low_threshold=low_threshold
        ) if annot is not None else None

        inside_patches, outside_patches = sample_patches(
            img,
            mask,
            patch_size,
            n_inside_per_image,
            n_outside_per_image,
            max_tries=max_tries,
        )

        for patch in inside_patches:
            Image.fromarray(patch).save(output_dir / "inside" / f"inside_{inside_idx:05d}.tif")
            inside_idx += 1

        for patch in outside_patches:
            Image.fromarray(patch).save(output_dir / "outside" / f"outside_{outside_idx:05d}.tif")
            outside_idx += 1

    return inside_idx, outside_idx


__all__ = [
    "extract_patches",
    "load_patches_from_folders",
    "mask_from_green_contour",
    "sample_patches",
    "to_tensor_batch",
]
