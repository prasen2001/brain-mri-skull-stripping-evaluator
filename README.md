# Brain MRI Skull-Stripping Evaluator: Classical Image Processing Baselines

An educational, beginner-friendly research prototype that asks a simple question:

> How far can understandable classical image-processing methods go when separating brain tissue from non-brain pixels in varied 2D medical images?

This project evaluates three fixed baselines on the official SynthStrip 2D v1.5 dataset. It does **not** use deep learning, train a model, or provide medical advice.

## What the project does

For each 2D image, the program loads its reference brain mask, robustly normalizes image intensity, runs three methods, measures their predictions, and writes tables and figures:

1. `otsu_threshold`: one automatically selected threshold, with deterministic polarity handling.
2. `otsu_morphology`: Otsu plus opening, closing, hole filling, and small-object removal.
3. `central_component`: Method 2 plus a deterministic largest-central-component heuristic and final cleanup.

The implementation is separated into dataset loading, preprocessing, segmentation, metrics, evaluation, and visualization modules. There is no notebook-only logic and no hidden metrics library.

## Why skull stripping matters

Brain MRI is an imaging technique that uses magnetic fields and radio waves to show soft tissue. A **pixel** is one number at one location in a 2D image; a **voxel** is its 3D equivalent. A scan can include brain tissue, skull, skin, eyes, neck, empty background, and scanner artifacts.

**Segmentation** assigns pixels to categories. **Skull stripping** is a binary segmentation task: keep the brain and reject everything else. A clean brain mask can simplify later measurement, registration, visualization, or disease-research pipelines. A bad mask can remove real brain tissue or keep distracting non-brain anatomy, so its errors must be measured.

An **image** stores intensity values. A **mask** stores labels aligned to the image: here, `1` means reference brain and `0` means non-brain.

## Methods in plain language

### Intensity normalization

MRI intensity has no universal brightness scale. For each image, this project clips values to its 1st and 99th percentiles and scales the result to `[0, 1]`. Percentile clipping stops a few extreme values from controlling the scale. A constant image becomes all zeros and is handled safely.

### Otsu thresholding

Otsu's method examines the intensity histogram and selects the threshold that best separates two intensity groups. Pixels on one side become foreground. Brain tissue is not always the brighter group, so the code scores both polarities using center coverage, border contact, and a broad plausible-area preference. It never consults the ground-truth mask to choose polarity.

Limitation: a histogram knows brightness, not anatomy. Skull, fluid, or internal structures can share the selected intensity range.

### Morphological cleanup

Morphology changes shapes using a small neighborhood called a structuring element:

- **Erosion** shrinks foreground and can remove thin protrusions.
- **Dilation** expands foreground and can bridge small gaps.
- **Opening** is erosion followed by dilation; it removes isolated foreground noise while approximately restoring larger regions.
- **Closing** is dilation followed by erosion; it connects nearby regions and closes narrow gaps.
- **Hole filling** fills enclosed background pockets.
- **Small-object removal** removes connected regions below a scale-aware area threshold.

Radii and minimum area are fractions of the real image dimensions, not assumptions about a 256×256 image.

### Connected components

A connected component is a group of touching foreground pixels. The third method scores each component as:

`area × (1 + centrality_weight × centrality)`

Centrality is 1 at the image center and falls toward the corners. Area remains important, but similarly sized central anatomy is preferred over a peripheral artifact. Exact ties are resolved by label order, so the result is deterministic. This is still a heuristic: the brain may not be centered, and a bright central structure may not represent the whole brain.

## Metrics

For a predicted mask and reference mask:

- **True positive (TP):** correctly predicted brain pixel.
- **True negative (TN):** correctly rejected non-brain pixel.
- **False positive (FP):** non-brain pixel incorrectly kept.
- **False negative (FN):** brain pixel incorrectly removed.

The formulas implemented in `src/metrics.py` are:

- Dice = `2TP / (2TP + FP + FN)`
- IoU = `TP / (TP + FP + FN)`
- Precision = `TP / (TP + FP)`
- Sensitivity/recall = `TP / (TP + FN)`
- Specificity = `TN / (TN + FP)`

Dice and IoU measure overlap. Precision asks, “of what I called brain, how much was correct?” Sensitivity asks, “of the real brain, how much did I find?” Specificity asks, “of the non-brain background, how much did I reject?” Pixel accuracy is recorded but is not a headline result because a large background can make a poor brain mask look deceptively accurate.

For two empty masks, Dice and IoU equal 1 because the masks agree perfectly. If a denominator contains no relevant positives, the metric is 1 only when no corresponding error was made. Every convention is tested in `tests/test_metrics.py`.

## Dataset, license, and citation

The project uses the [official SynthStrip dataset page](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) and the 39 MB `synthstrip_data_v1.5_2d.tar` archive. Version 1.5 was released on 2024-06-20. Its official SHA-256 is:

```text
69b3c3d9e37ef63b800137e25386f0cfb6851803916900923af1cedd4e0d55f4
```

The archive contains one sagittal slice per scan, stored as `image.nii.gz` and `mask.nii.gz`. The official scan lists provide 24 validation scans and 558 test scans. This project calls validation **development**, uses it to select morphology parameters, and reports final comparisons on the untouched 558-scan test split. Because each 2D item comes from a different scan, the division is scan-level and related slices cannot cross splits.

According to the included dataset README, IXI images use [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and the other redistributed SynthStrip content uses [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original-source citations are listed in `data/processed/synthstrip_data_v1.5_2d/README.txt` after download.

Please cite:

> Hoopes A, Mora JS, Dalca AV, Fischl B, Hoffmann M. SynthStrip: Skull-Stripping for Any Brain Image. *NeuroImage*. 2022;260:119474. [doi:10.1016/j.neuroimage.2022.119474](https://doi.org/10.1016/j.neuroimage.2022.119474)

Dataset files are excluded from Git. Do not add private or clinical scans to this repository.

## Installation on Apple silicon macOS

Python 3.11 or newer is supported. The verified run used Python 3.13.15 on Apple silicon.

```bash
cd brain-mri-skull-stripping-evaluator
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependencies have native Apple-silicon wheels and do not include PyTorch, TensorFlow, MONAI, or another deep-learning framework.

## Download and verify the official data

```bash
python -m scripts.download_data
```

The script downloads only the official MGH URL, calculates SHA-256, checks that the file is a tar archive, rejects unsafe archive paths/links, and extracts without modifying source files. The resulting data directory is:

```text
data/processed/synthstrip_data_v1.5_2d
```

## Run tests

```bash
python -m pytest -q
```

Verified result: `17 passed`.

## Reproduce development tuning

This optional command evaluates 81 small morphology configurations on only the 24 official development scans:

```bash
python -m scripts.tune_on_development \
  --data-dir data/processed/synthstrip_data_v1.5_2d \
  --output outputs/development_tuning.csv \
  --seed 42
```

The chosen values are recorded in `configs/default.yaml`: opening fraction `0.008`, closing fraction `0.004`, minimum-component area fraction `0.005`, and centrality weight `0.75`. Several centrality weights tied on development data; `0.75` keeps a moderate preference.

## Quick run

This verified command samples 20 test scans deterministically:

```bash
python -m src.evaluate \
  --data-dir data/processed/synthstrip_data_v1.5_2d \
  --output-dir outputs/quick_run \
  --subset test \
  --limit 20 \
  --seed 42
```

## Full evaluation

```bash
python -m src.evaluate \
  --data-dir data/processed/synthstrip_data_v1.5_2d \
  --output-dir outputs/full_run \
  --subset test \
  --seed 42
```

On a system where Matplotlib cannot write its normal user cache, prefix either evaluation command with `MPLCONFIGDIR=outputs/.matplotlib`. This affects only plot caching, not results.

The CLI also accepts overrides such as `--opening-radius-fraction`, `--closing-radius-fraction`, `--minimum-component-area-fraction`, `--centrality-weight`, and normalization percentiles. Do not tune these on the held-out test metrics.

## Output directory

Each run writes:

```text
per_image_metrics.csv          one row per scan and method
summary_metrics.csv            aggregate statistics per method
run_config.json                seed, parameters, versions, platform, and counts
invalid_pairs.csv              only when invalid inputs are found
method_comparison.png
dice_distribution.png
iou_distribution.png
processing_time_comparison.png
best_cases.png
worst_cases.png
overlays/                      six individual five-panel case figures
```

Every case figure shows the normalized MRI, ground truth, prediction, contour overlay, and labeled error map. Error maps use both color and text: TP = correct brain, FP = predicted outside the reference, and FN = missed brain.

Generated data and outputs are intentionally ignored by Git; rerun the commands to reproduce them.

## Measured held-out results

These are real measurements from the verified full run on all **558 official test scans**, not synthetic examples. Values are per-image means; timing is the mean segmentation time measured on this machine and excludes file loading/plotting.

| Method | Mean Dice | Mean IoU | Precision | Sensitivity | Specificity | Mean time/image |
|---|---:|---:|---:|---:|---:|---:|
| `otsu_threshold` | 0.7494 | 0.6184 | 0.6811 | 0.8843 | 0.7649 | 0.00031 s |
| `otsu_morphology` | 0.7717 | 0.6499 | 0.7087 | 0.9044 | 0.7858 | 0.00219 s |
| `central_component` | **0.7903** | **0.6777** | **0.7491** | 0.9010 | **0.8174** | 0.00379 s |

`central_component` had the best mean Dice and IoU. It improved mean Dice by about 0.041 over raw Otsu. Its median Dice was 0.823, but its minimum was only 0.085; the mean alone hides severe failures.

Quick-run measurements on the deterministic 20-scan sample were Dice 0.7700, 0.7995, and 0.8236. These are a convenience check, not the headline comparison.

## Failure analysis

Successful cases such as `ixi_mra_138` (Dice 0.989), `asl_epi_124` (0.983), and `ixi_mra_013` (0.979) contain a dominant connected brain region with strong separation from background. Morphology removes small fragments and the central-component rule discards peripheral anatomy.

The worst cases are instructive:

- `ixi_t2_022` (Dice 0.085): Otsu selected a bright internal structure rather than the whole heterogeneous T2 brain. Precision was 1.000 but sensitivity was only 0.045—almost everything predicted was brain, but nearly all brain pixels were missed.
- `ixi_t2_141` (Dice 0.096): the intensity grouping included a skull/scalp rim and only small parts of brain, producing both false positives and many false negatives.
- `ixi_t2_020` (Dice 0.132): the connected-component heuristic retained a small bright central structure, showing that “central” does not necessarily mean “whole brain.”

The source-level central-component mean Dice also varied: ASL 0.851, QIN 0.852, IXI 0.799, Infant 0.714, and FSM 0.700. This is evidence that one thresholding heuristic does not generalize uniformly across acquisition types.

## Limitations

- The dataset contains one 2D sagittal slice per scan, not full 3D volumes; 2D results do not establish 3D performance.
- Otsu assumes two separable intensity groups, while MRI anatomy and artifacts are multi-modal and spatially complex.
- The centrality heuristic can select the wrong anatomy when the brain is off-center or internal tissue is especially bright.
- Ground-truth definitions can differ between datasets, especially around cerebrospinal fluid and boundaries.
- Development tuning used only 24 scans. Test metrics describe this dataset and split, not every scanner, population, disease, or protocol.
- Timing is machine- and software-dependent.
- This is a retrospective educational evaluation, not clinical validation.

## Ethical and clinical disclaimer

This software is an educational research prototype. It is **not** diagnostic software, a medical device, clinically validated, or approved by a regulator. Do not use its masks for patient care or clinical decisions. The project makes no performance claim about populations, scanners, or conditions not represented in the evaluated dataset.

## Future work

A sensible next experiment is a small 2D U-Net trained with a strict scan-level development/test protocol and compared against these unchanged baselines. Additional work could evaluate 3D consistency, cross-dataset generalization, uncertainty, bias across acquisition groups, and boundary metrics. An autoencoder, pretrained encoder, transformer, or regression model should be introduced only with a clear question and leakage-safe evaluation—not because it sounds more advanced.

## Project map

```text
configs/default.yaml               tuned classical parameters
scripts/download_data.py           official download, checksum, safe extraction
scripts/tune_on_development.py      development-only grid search
src/dataset.py                      discovery, NIfTI loading, validation
src/preprocessing.py                robust intensity normalization
src/segmentation.py                 three classical methods
src/metrics.py                      explicit confusion counts and formulas
src/evaluate.py                     command-line evaluation
src/visualization.py                charts and five-panel cases
tests/                              17 focused unit tests
INTERVIEW_PREP.md                   beginner study guide
PROJECT_REPORT.md                   concise experimental report
```

