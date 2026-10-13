# Revision based on Masoud_paper.rev01.OP.pdf

The revision applies the annotated PDF's 74 annotations and the accompanying editorial instructions throughout `paper.tex`, including the sections after 3.2. No classifier was retrained and no performance measurements were added or changed. The new figure is generated from a saved learned profile.

## Main changes

- Put training with few independent cases at the center of the title, introduction, and discussion. Separate that motivation from the demonstrated result: compactness at the recorded training sizes. A training-size experiment is still needed to establish a data-efficiency advantage.
- Define and consistently use **predefined-kernel bank**, **learned radial bank**, and **merged radial bank**. Use **Breast** and **Brain** as dataset names. Replace informal terms such as “head,” “readout,” “spans the same evidence,” and “visual primitives” with defined quantities or procedures.
- Define region extraction, annotation coverage, labels, the training set, convolution, response maps, pooled statistics, and the classifier feature vector. Use `h_j` for kernels, `K` for kernel count, and `gamma_i` for annotation coverage.
- Reorder the method into radial representation, response statistics, classifier, initialization, training, and merging. Distinguish initialized profiles from optimized profiles and the temporary neural classifier from the final logistic classifier. Add regularization weights and optimization details.
- Define the top-value fraction and the integer number of values separately. Explain the three response statistics and reference every numbered equation in the text.
- Explain hierarchical clustering, sign alignment, averaging, refitting, and selection of the smallest eligible partition. Remove the unsupported assertion that correlated maps necessarily preserve all classifier decisions.
- Distinguish MMR's correlations between pooled region features from merging's correlations between response-map values.
- Move the contribution discussion to the final section and explain the meaning and limits of each result. Remove the claim that the existing single-kernel comparison isolates the effect of pooling.
- Add a reproducible example of learned-filter interpretation using profile 14 and its actual logistic coefficients. The example explicitly belongs to the 16-kernel Breast model. Its profile survives merging as a singleton, but the classifier coefficients are refitted after merging.
- Replace ambiguous baseline labels where saved configurations identify candidate counts, sampling, or validation sample size.

## Factual corrections and provenance

### Brain positive-region coverage

The previous manuscript stated 0.80. The saved preprocessing record for the dataset referenced by the reported Brain bank specifies `min_pos_coverage: 0.2`:

- `data_brain2/patches_radial_v2_p64_strict/prep_summary.json`
- `results/brain2_radial_single_kernel/radial_kernel_bank_summary.json`, `config.patch_output_root`

The manuscript now reports 0.20. The record also specifies 64-pixel crops, zero negative coverage, a 0.25 near-negative fraction, and 12-pixel mask dilation. The current Brain notebook includes newer 32-pixel localization experiments; those are not substituted for the reported 64-pixel results. The region image files were not present in the inspected cache directories, so the saved metadata could not be checked against individual crops.

### Breast splitting

`kernel_model/data.py`, `_assign_patient_splits`, groups by source-image stem. An image identifier alone does not prove that different images come from different patients. The revised paper therefore states source-image separation for Breast and patient separation for Brain, and identifies the Breast limitation.

### Radial-bin definition and outer coefficients

The implementation uses the floor of Euclidean distance to define bins. `B_r` is a binary basis mask; `f_j(r)` is a kernel weight, not an image intensity. When deriving an initial profile from another kernel, the kernel weights within each bin are averaged.

Although one annotation suggested that outer pixels are ignored, the saved Breast profiles have a nonzero fixed coefficient at radius 21 (`-0.038827255...`). The Brain profiles have zero coefficients beyond radius 30. The revision follows the actual saved model: `R_max` bounds the **trainable radius**, not necessarily nonzero support.

Sources:

- `breast_radial_kernel_bank.ipynb`, `integer_radius_bins`, `build_kernels`
- `results/exp_022_mmr_nperFamily500_nComposition1250/radial_kernel_bank_k16_r20/radial_kernel_bank_summary.json`
- `results/brain2_radial_single_kernel/radial_kernel_bank_summary.json`

### Interpretation example

Profile 14 uses saved zero-based index 13. The feature ordering is all mean magnitudes, then all maximum magnitudes, then all signed extreme responses. Its classifier coefficients are therefore entries 13, 29, and 45 of `metrics.calib_w`: approximately -2.769, -1.727, and +0.307. Its center coefficient is 0.4254475 and its radius-18 coefficient is -0.1826091. The expanded kernel has a nonzero sum, so the text does not describe it as a pure contrast detector. The merging record includes `[13]` as a singleton.

Sources:

- Breast `radial_kernel_bank_summary.json` cited above
- Adjacent `kernel_reduction_summary.json`
- Figure generator: `publication/scripts/plot_profile_interpretation.py`

### Archived baseline differences

Several archived predefined-kernel experiments use 1,790 validation regions, whereas the radial-bank experiments use 1,870. Some configurations additionally cap training regions per class. The revised methods explicitly state that the collected results are not a fully matched architecture comparison.

Examples:

- `results/exp_024_8compositing/results.json`: QMC bank, test AUC 0.9386990, validation N=1,790.
- `results/exp_025/results.json`: QMC bank, test AUC 0.9377259, validation N=1,870.
- `results/exp_022/results.json`: random bank, test AUC 0.9376321, validation N=1,790.
- `results/exp_022_mmr_n1_k1/config.json`: nine candidates, one retained kernel.
- `results/exp_022_mmr_n3_k1/config.json`: 27 candidates, one retained kernel.
- `results/exp_022_mmr_n3_k3/config.json`: 27 candidates, three retained kernels.
- `results/exp_022_mmr/config.json`: 1,800 candidates, three retained kernels.
- `results/exp_023_mmr/config.json`: 1,800 candidates, five retained kernels.

The old predefined-kernel confusion matrix contains 1,090 observations (521+89+56+424), so it cannot be presented alongside the 1,870-region CNN matrix as a matched test-set comparison. The revised figure retains only the CNN matrix and specifies its sample size. The underlying image assets remain unchanged.

### Claims narrowed to the available evidence

- Small numerical AUC differences are not presented as statistically established improvements.
- Logistic regression is not described as providing validated calibration.
- The Brain threshold convention can materially change accuracy; the previous statement that threshold conventions differ by less than 0.001 was removed.
- Brain merging's 0.0289 test-AUC loss is reported directly. No unmeasured explanation is presented as its established cause.
- The Breast standardization experiment does not establish that standardization is necessary for Brain.
- The CNN comparison uses color while kernel models use grayscale, and a Brain CNN result is not available.
- The probability overlays are illustrative region-score maps. Unverified held-out status and claims of pixel-level localization are omitted.

## Remaining empirical work

The central small-training-set claim requires grouped learning curves with repeated selections of independent training cases, plus matched preprocessing and model selection. A fixed-profile pooling comparison would separately test the value of the three response statistics. These are new experiments, not editorial corrections, and were not invented or run as part of this revision.

## Build and checks

Run from the repository root:

```bash
python3 publication/scripts/plot_profile_interpretation.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error publication/paper.tex
```

The final PDF is `publication/paper.pdf`. The revision checks equation references, unchanged numerical table entries, compilation, and rendered page layout.

Final verification: the nine-page PDF compiled without unresolved references or overfull boxes. All six tables retain the original numerical entries, every numbered equation is referenced, and all rendered pages were inspected.
