# Project Report

## Objective

Evaluate three understandable classical image-processing baselines for 2D brain MRI skull stripping before attempting deep learning. The research question is whether morphology and anatomical location heuristics improve a simple intensity threshold across varied scans.

## Dataset

The experiment uses the official SynthStrip 2D dataset, version 1.5. It contains one sagittal NIfTI slice and one binary brain mask for each scan. The authors provide a scan-level split: 24 validation scans and 558 test scans. This project uses validation as development and reports final results once on all 558 test scans.

The archive SHA-256 matched the official value before extraction. All 582 image-mask pairs loaded successfully, had matching spatial dimensions, and used binary masks. No test pair was skipped. IXI content is CC BY-SA 4.0; other redistributed content is CC BY 4.0. See the dataset README and the citation in the project README.

## Methods

All images were clipped at their per-image 1st and 99th intensity percentiles and scaled to `[0, 1]`.

1. `otsu_threshold` applies Otsu's two-class threshold and chooses bright or dark polarity with a deterministic center/area/border score.
2. `otsu_morphology` adds opening, closing, hole filling, and scale-aware small-object removal.
3. `central_component` adds a deterministic area-and-centrality component score, retains one component, fills holes, and performs final closing.

No ground-truth mask is used during prediction. No method learns from the test set.

The evaluator also records surface Dice at one-pixel tolerance, HD95, and ASSD. Reference-free checks measure predicted area, connected components, centroid location, border contact, and disagreement among methods to flag masks for review.

## Experimental design

A grid of 81 parameter combinations was evaluated on only the 24 development scans. The chosen configuration used opening fraction 0.008, closing fraction 0.004, minimum component area fraction 0.005, and centrality weight 0.75. On development data, Method 2 reached mean Dice 0.744 and Method 3 reached 0.750.

The final CLI then evaluated the fixed configuration on 558 held-out test scans using seed 42. Metrics were calculated per image, then summarized across images. Confidence intervals used 1,000 scan-level bootstrap resamples with the same seed. Processing time covers segmentation only. The complete configuration and package versions are in `outputs/full_run/run_config.json`.

## Results

| Method | N | Dice mean (median) | IoU mean | Precision | Sensitivity | Specificity | Mean time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `otsu_threshold` | 558 | 0.749 (0.764) | 0.618 | 0.681 | 0.884 | 0.765 | 0.00032 s |
| `otsu_morphology` | 558 | 0.772 (0.787) | 0.650 | 0.709 | **0.904** | 0.786 | 0.00218 s |
| `central_component` | 558 | **0.790 (0.823)** | **0.678** | **0.749** | 0.901 | **0.817** | 0.00380 s |

![Method comparison](docs/assets/method_comparison.png)

The methods improve progressively in mean Dice, IoU, precision, and specificity. Method 2 has slightly higher mean sensitivity than Method 3 because component selection can discard true disconnected brain pixels. Method 3's extra precision and specificity produce the best overall overlap.

| Method | Mean Dice 95% CI | Surface Dice | HD95 (pixels) | ASSD (pixels) |
|---|---:|---:|---:|---:|
| `otsu_threshold` | 0.737–0.761 | 0.278 | 47.69 | 14.92 |
| `otsu_morphology` | 0.759–0.783 | 0.307 | 43.49 | 13.15 |
| `central_component` | **0.777–0.803** | **0.333** | **40.33** | **12.09** |

![Boundary metrics](docs/assets/boundary_metric_comparison.png)

## Reliability warnings

The `central_component` checks flagged 31 of 558 predictions. Flagged predictions had mean Dice 0.551; unflagged predictions had mean Dice 0.804. All four cases below 0.20 Dice and 12 of 13 below 0.40 were flagged, largely through method disagreement. Only 17 of 50 cases below 0.60 were flagged, so this is a conservative review mechanism rather than a guarantee.

![Reference-free warning rates](docs/assets/quality_warning_rates.png)

## Successful cases

![Best cases](outputs/full_run/best_cases.png)

- `ixi_mra_138`: Dice 0.989, IoU 0.977. Prediction closely follows the reference with only a thin boundary error.
- `asl_epi_124`: Dice 0.983, IoU 0.968. The low-resolution brain is a dominant compact component, so cleanup is reliable.
- `ixi_mra_013`: Dice 0.979, IoU 0.958. Strong foreground separation and a central connected region suit the heuristic.

These cases show why the later methods help: morphology removes fragments, while component selection rejects competing peripheral regions.

## Failure cases

![Worst cases](outputs/full_run/worst_cases.png)

- `ixi_t2_022`: Dice 0.085, IoU 0.045. The method retains a bright internal structure. Precision is 1.000, but sensitivity is only 0.045, so almost all true brain tissue is absent.
- `ixi_t2_141`: Dice 0.096, IoU 0.050. Bright scalp/skull and cortical edges compete with the heterogeneous brain, creating both false positives and extensive false negatives.
- `ixi_t2_020`: Dice 0.132, IoU 0.070. A small bright central structure wins the component score even though it is not the whole brain.

These failures explain why the later method is not uniformly better. Selecting one component helps when non-brain clutter is separate, but hurts when thresholding fragments the actual brain or a wrong central structure looks most plausible.

## Interpretation

The experiment supports a narrow conclusion: on this 2D test split, deterministic cleanup and component selection improve average overlap over raw Otsu. It does not show that thresholding is clinically adequate. Method 3's standard deviation is 0.156 and its Dice range is 0.085–0.989, so a good average coexists with catastrophic failures.

Performance varies by source group for Method 3: ASL 0.851, QIN 0.852, IXI 0.799, Infant 0.714, and FSM 0.700 mean Dice. MRI contrast, resolution, anatomy, and acquisition differences challenge a single intensity rule.

![Source-level Dice](docs/assets/source_dice_comparison.png)

## Limitations

- One sagittal slice per scan cannot measure 3D continuity or volumetric accuracy.
- Only 24 scans informed parameter choice.
- The heuristic encodes assumptions about intensity, size, connectivity, and location.
- Warning thresholds are uncalibrated geometry heuristics and miss some poor predictions.
- Surface distances are in pixels rather than physical millimeters.
- Results may not transfer to another scanner, protocol, population, or mask definition.
- Timing is specific to the local Apple-silicon software environment.
- This is not a prospective, clinical, diagnostic, or regulatory evaluation.

## Proposed next experiment

Train a small 2D U-Net using a leakage-safe scan-level training/development split, freeze the current test set, and compare it with these unchanged baselines using the same metrics and case analysis. Add a second external dataset if licensing permits. The key question should be whether learned spatial features reduce the T2 and cross-source failures—not merely whether the model has more parameters.

