# Kernel-Based Cancer Classification

This notebook implements a kernel-based feature extraction and classification pipeline for medical image analysis.

## Overview

The pipeline extracts image patches from annotated medical images, generates a diverse set of convolutional kernels, computes filter responses, and trains a logistic classifier to distinguish between tumor (inside) and healthy (outside) tissue patches.

## Key Components

### 1. **Patch Extraction**
- Extracts patches from medical images based on pixel-level annotations
- Samples patches from inside tumor regions (red contours) and outside healthy regions
- Saves patches to `inside/` and `outside/` directories

### 2. **Kernel Bank Generation**
Generates a diverse set of convolutional kernels from multiple families:
- **Gaussian**: Isotropic Gaussian filters
- **Anisotropic Gaussian**: Elliptical Gaussian filters with rotation
- **DoG (Difference of Gaussians)**: Edge detection kernels
- **LoG (Laplacian of Gaussian)**: Blob detection kernels
- **Gabor**: Oriented texture analysis kernels

### 3. **Response Computation**
- Computes filter responses for each kernel on all patches
- Uses PyTorch for efficient batch processing
- Supports multiple response functions (absolute max, mean absolute)

### 4. **Kernel Selection**
- Ranks kernels by discriminative power (AUC and Fisher score)
- Selects diverse kernels using Maximal Marginal Relevance (MMR) to avoid redundancy
- Balances discriminative power and diversity

### 5. **Classification**
- Trains a logistic regression classifier on selected kernel responses
- Evaluates performance using AUC and accuracy metrics

## Usage

The notebook is configured to:
- Load images from `data/TIFF Images/all/` (or from `data/TIFF Images/Malignant/` and `data/TIFF Images/Normal/` for specific classes)
- Use annotations from `data/Pixel-level annotation/`
- Save patches to `data/patches/` (with subdirectories `inside/` and `outside/`)
- Save results to `results/`

### Default Parameters
- Patch size: 128×128 pixels
- Kernel size: 31×31 pixels
- Kernel families: All 5 types
- Kernels per family: 200
- Selected kernels: Top 20 diverse kernels
- Training epochs: 30

## Output

The pipeline generates:
- `results/results.npz`: Selected kernel indices
- `results/results.json`: Metadata including kernel parameters and classifier performance
- `results/kernels.npy`: All generated kernels

## Results

Example run results:

- **Data**: 130 images of Malignant tumor were used (from `data/TIFF Images/Malignant/`)
<!-- - **Patches extracted**: 5,277 inside (tumor) patches and 3,220 outside (healthy) patches -->
- **Training data**: 2,000 inside patches (small windows of size 128×128) and 2,000 outside patches (limited by `max_per_class`) were extracted from the total 130 images
  - Inside patches saved to: `data/patches/inside/`
  - Outside patches saved to: `data/patches/outside/`
- **Kernel bank**: 1,000 kernels generated (200 per family × 5 families)
- **Selected kernels**: 20 diverse kernels (indices: [340, 852, 970, 964, 938, 315, 282, 234, 252, 346, 355, 246, 325, 60, 378, 37, 385, 128, 107, 97])
- **Classifier performance**:
  - Validation AUC: **0.9201**
  - Validation Accuracy: **0.6763**

## Changes

### 21.10.2025
- [x] Switched response aggregation to mean_abs for smoother, less noisy features.
- [x] Added feature standardization so all kernel responses are on comparable scale before training.
- [x] Introduced an optional 2-layer MLP with dropout (used by default) to model nonlinear separations beyond logistic.
- [x] Extended training (more epochs) and slightly reduced learning rate for steadier convergence.
- [x] Searched over a larger candidate set for best 2/3-kernel subsets to pick stronger features.


## TODO

### 21.10.2025
- [x] Select the top 2 kernels (or best-performing 2-kernel model).
- [x] Extract their outputs and use them as features **x** and **y**.
- [X] Generate a 2D scatterplot of segmented areas:
  - [x] Use distinct colors for *cancer* vs *non-cancer* regions.
  - [x] Check visually whether the classes separate clearly.
- [x] Select the best 3-feature model.
- [x] Create a 3D scatterplot using the three outputs 

- [ ] Experiment with different kernel families and parameter ranges
- [ ] Tune hyperparameters (topM, K, lambda_mm) for better kernel selection
- [ ] Try different response functions (mean_abs, etc.)
- [ ] Evaluate on test set (currently only validation split)
- [ ] Add cross-validation for more robust performance estimates
- [ ] Visualize selected kernels and their responses
- [ ] Compare with deep learning baselines
- [ ] Optimize patch sampling strategy (currently random sampling)
- [ ] Add support for multi-class classification
- [ ] Implement feature importance analysis for selected kernels

## Dependencies

- PyTorch
- NumPy
- OpenCV
- PIL/Pillow
- scikit-learn
- scipy
- matplotlib
- tqdm

