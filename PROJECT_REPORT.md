# Project Report

## Objective

Evaluate three understandable classical image-processing baselines for 2D brain MRI skull stripping before attempting deep learning. The research question is whether morphology and anatomical location heuristics improve a simple intensity threshold across varied scans.

## Dataset

The experiment uses the official SynthStrip 2D dataset, version 1.5. It contains one sagittal NIfTI slice and one binary brain mask for each scan. The authors provide a scan-level split: 24 validation scans and 558 test scans. This project uses validation as development and reports final results once on all 558 test scans.

The archive SHA-256 matched the official value before extraction. All 582 image-mask pairs loaded successfully, had matching spatial dimensions, and used binary masks. No test pair was skipped. IXI content is CC BY-SA 4.0; other redistributed content is CC BY 4.0. See the dataset README and the citation in the project README.

External validation uses the open NFBS repository: 125 defaced, anonymized T1 volumes with manually corrected brain masks. The 1,751,464,473-byte archive matched the project-pinned SHA-256 `fd616b9ea21aad3d0d4cd05a25a71f4a3c251140df38188e58e499254440d2cc`. Each volume was reoriented to closest canonical coordinates and its center sagittal slice was selected from image geometry alone. No NFBS mask influenced slice selection, parameter selection, or tuning.

## Methods

All images were clipped at their per-image 1st and 99th intensity percentiles and scaled to `[0, 1]`.

1. `otsu_threshold` applies Otsu's two-class threshold and chooses bright or dark polarity with a deterministic center/area/border score.
2. `otsu_morphology` adds opening, closing, hole filling, and scale-aware small-object removal.
3. `central_component` adds a deterministic area-and-centrality component score, retains one component, fills holes, and performs final closing.

No ground-truth mask is used during prediction. No method learns from the test set.

The evaluator also records surface Dice at one-pixel tolerance plus HD95 and ASSD in both pixels and physical millimeters read from the NIfTI header. Reference-free checks measure predicted area, connected components, centroid location, border contact, and disagreement among methods to flag masks for review.

A separate retrospective taxonomy uses the reference-derived metrics after evaluation. It labels dominant outcomes such as under-segmentation, over-segmentation, boundary mismatch, fragmentation, border leakage, likely polarity failure, and wrong-component selection. These labels explain known results; they are not deploy-time confidence predictions.

## Experimental design

A grid of 81 parameter combinations was evaluated on only the 24 development scans. The chosen configuration used opening fraction 0.008, closing fraction 0.004, minimum component area fraction 0.005, and centrality weight 0.75. On development data, Method 2 reached mean Dice 0.744 and Method 3 reached 0.750.

The warning thresholds were separately audited on the same 24 development scans. For a predefined failure threshold of Dice below 0.40, a small grid recommended the existing defaults and identified both development failures with no false warning. This tiny calibration result is not a general accuracy claim.

The final CLI then evaluated the fixed configuration on 558 held-out test scans using seed 42. Metrics were calculated per image, then summarized across images. Confidence intervals and paired method differences used 1,000 scan-level bootstrap resamples with the same seed. Paired Wilcoxon tests used Holm correction across reported comparisons. Processing time covers segmentation only. The complete configuration and package versions are in `outputs/full_run/run_config.json`.

The exact same configuration was then frozen and run once on all 125 NFBS subjects. NFBS was not used to choose thresholds, morphology radii, centrality weight, warning rules, or failure-category thresholds.

## Results

| Method | N | Dice mean (median) | IoU mean | Precision | Sensitivity | Specificity | Mean time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `otsu_threshold` | 558 | 0.749 (0.764) | 0.618 | 0.681 | 0.884 | 0.765 | 0.00034 s |
| `otsu_morphology` | 558 | 0.772 (0.787) | 0.650 | 0.709 | **0.904** | 0.786 | 0.00225 s |
| `central_component` | 558 | **0.790 (0.823)** | **0.678** | **0.749** | 0.901 | **0.817** | 0.00397 s |

![Method comparison](docs/assets/method_comparison.png)

The methods improve progressively in mean Dice, IoU, precision, and specificity. Method 2 has slightly higher mean sensitivity than Method 3 because component selection can discard true disconnected brain pixels. Method 3's extra precision and specificity produce the best overall overlap.

| Method | Mean Dice 95% CI | Surface Dice | HD95 (mm) | ASSD (mm) |
|---|---:|---:|---:|---:|
| `otsu_threshold` | 0.737–0.761 | 0.278 | 53.09 | 16.36 |
| `otsu_morphology` | 0.759–0.783 | 0.307 | 46.90 | 14.25 |
| `central_component` | **0.777–0.803** | **0.333** | **43.88** | **13.26** |

![Boundary metrics](docs/assets/boundary_metric_comparison.png)

## Paired improvements

Every method processed the same scans, so the analysis subtracts scores scan by scan. Morphology minus Otsu improved Dice by 0.0223 on average (95% bootstrap CI 0.0196–0.0251), winning on 449 scans and losing on 108. Central component minus Otsu improved Dice by 0.0409 (0.0355–0.0468), winning on 456 scans and losing on 101. Central component minus morphology improved Dice by 0.0187 (0.0150–0.0228), with 302 wins, 21 losses, and 235 ties. All Dice comparisons remained significant after Holm correction, but the effect sizes and failures matter more than a small p-value.

![Paired per-scan Dice improvements](docs/assets/paired_dice_improvement.png)

## Reliability warnings

The `central_component` checks flagged 31 of 558 predictions. Flagged predictions had mean Dice 0.551; unflagged predictions had mean Dice 0.804. All four cases below 0.20 Dice and 12 of 13 below 0.40 were flagged, largely through method disagreement. Only 17 of 50 cases below 0.60 were flagged, so this is a conservative review mechanism rather than a guarantee.

![Reference-free warning rates](docs/assets/quality_warning_rates.png)

## Retrospective failure taxonomy

For `central_component` on SynthStrip, the primary categories were 273 over-segmentations, 221 acceptable cases, 37 under-segmentations, nine mixed errors, eight boundary mismatches, eight wrong-component selections, and two border leakages. The strict acceptable category requires Dice at least 0.80 and no dominant error reason.

![SynthStrip failure categories](docs/assets/failure_category_distribution.png)

## External NFBS evaluation

| Method | N | Dice mean (median) | IoU mean | Precision | Sensitivity | HD95 (mm) |
|---|---:|---:|---:|---:|---:|---:|
| `otsu_threshold` | 125 | 0.506 (0.515) | 0.341 | 0.450 | **0.582** | 75.49 |
| `otsu_morphology` | 125 | 0.525 (0.550) | 0.362 | 0.504 | 0.556 | 74.09 |
| `central_component` | 125 | **0.537 (0.596)** | **0.396** | **0.615** | 0.500 | **66.51** |

![NFBS external method comparison](docs/assets/nfbs_external_method_comparison.png)

The central-component method still had the best mean overlap, but its mean Dice dropped by 0.2535 relative to SynthStrip. Its NFBS standard deviation was 0.2257 and its range was 0.000–0.888. The external taxonomy found 44 under-segmentations, 33 boundary mismatches, 31 mixed errors, 12 wrong-component selections, and five over-segmentations. No prediction met the strict acceptable rule.

![NFBS failure categories](docs/assets/nfbs_external_failure_categories.png)

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

The internal experiment supports a narrow conclusion: on the SynthStrip 2D test split, deterministic cleanup and component selection improve average overlap over raw Otsu. The NFBS result adds a stronger conclusion: the absolute performance does not transfer reliably to a different T1 acquisition and field of view. It does not show that thresholding is clinically adequate.

Performance varies by source group for Method 3: ASL 0.851, QIN 0.852, IXI 0.799, Infant 0.714, and FSM 0.700 mean Dice. It also varies by modality encoded in the identifiers: ASL/EPI 0.952, MRA 0.947, DWI 0.912, FLAIR 0.904, T2 0.775, PD 0.757, T1 0.755, and qT1 0.514. MRI contrast, resolution, anatomy, and acquisition differences challenge a single intensity rule.

![Source-level Dice](docs/assets/source_dice_comparison.png)

![Modality-level Dice](docs/assets/modality_dice_comparison.png)

## Limitations

- One sagittal slice per scan cannot measure 3D continuity or volumetric accuracy.
- Only 24 scans informed parameter choice.
- The heuristic encodes assumptions about intensity, size, connectivity, and location.
- Warning thresholds were calibrated on only 24 development scans. They remain geometry heuristics rather than confidence probabilities and miss some poor predictions.
- Millimeter boundary distances depend on NIfTI header spacing and measure only a 2D contour, not a complete 3D surface.
- Results may not transfer to another scanner, protocol, population, or mask definition.
- NFBS evaluates a deterministic 2D center slice derived from each 3D volume. It is external data, but it is not a full-volume experiment and its slice construction differs from SynthStrip's published 2D archive.
- Timing is specific to the local Apple-silicon software environment.
- This is not a prospective, clinical, diagnostic, or regulatory evaluation.

## Proposed next experiment

Train a small 2D U-Net with a leakage-safe training/development/test design and compare it with these unchanged baselines. Keep NFBS untouched as external evaluation. The specific question is whether learned spatial features reduce the observed NFBS under-segmentation and wrong-component failures—not merely whether a model has more parameters. A later experiment should process complete 3D volumes and measure volumetric continuity.

