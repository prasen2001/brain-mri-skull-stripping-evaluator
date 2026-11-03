# NFBS External Validation

## Purpose

This experiment tests whether the fixed classical baselines transfer beyond the dataset used for parameter selection. It is deliberately reported separately from SynthStrip.

## Dataset

The Neurofeedback Skull-stripped (NFBS) repository contains 125 defaced, anonymized T1-weighted MRI volumes and manually corrected brain masks. The source paper states that the repository is completely open and requires neither an application nor a data-usage agreement.

- Project: <https://preprocessed-connectomes-project.org/NFB_skullstripped/>
- Archive: `NFBS_Dataset.tar.gz`, 1,751,464,473 bytes
- Verified SHA-256: `fd616b9ea21aad3d0d4cd05a25a71f4a3c251140df38188e58e499254440d2cc`
- Citation: Puccio et al., *GigaScience* 2016;5:45, <https://doi.org/10.1186/s13742-016-0150-5>

## Leakage controls

1. The SynthStrip morphology, centrality, normalization, warning, and category thresholds were frozen before NFBS evaluation.
2. Each NFBS volume was converted to closest-canonical orientation.
3. The center index of the sagittal image axis was selected using image geometry only.
4. The reference mask was sliced at the same index but never used to select the slice.
5. No setting was changed after inspecting NFBS results.
6. NFBS outputs remain separate from SynthStrip outputs.

This is still a constructed 2D external experiment. It does not evaluate complete 3D volumes, and the slice-selection procedure may differ from the one used to create the official SynthStrip 2D archive.

## Reproduction

```bash
python -m scripts.prepare_nfbs_external

python -m src.evaluate \
  --data-dir data/processed/nfbs_2d_external \
  --output-dir outputs/nfbs_external_run \
  --subset test \
  --seed 42
```

## Results

| Method | N | Mean Dice | Median Dice | Mean IoU | Precision | Sensitivity | HD95 (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `otsu_threshold` | 125 | 0.5056 | 0.5152 | 0.3414 | 0.4497 | 0.5824 | 75.49 |
| `otsu_morphology` | 125 | 0.5252 | 0.5504 | 0.3625 | 0.5043 | 0.5559 | 74.09 |
| `central_component` | 125 | **0.5368** | **0.5955** | **0.3960** | **0.6145** | 0.4999 | **66.51** |

The central-component method improved mean Dice over raw Otsu by 0.0312. Its paired 95% bootstrap interval was 0.0014–0.0602. However, its absolute mean Dice was 0.2535 lower than on SynthStrip, and its range was 0.000–0.888.

## Failure analysis

For `central_component`:

| Primary retrospective category | Images | Percentage |
|---|---:|---:|
| Under-segmentation | 44 | 35.2% |
| Boundary mismatch | 33 | 26.4% |
| Mixed overlap error | 31 | 24.8% |
| Wrong component selected | 12 | 9.6% |
| Over-segmentation | 5 | 4.0% |

No NFBS result met the strict acceptable category. Representative failures show scalp, neck, or peripheral bright tissue becoming a more plausible connected component than the brain. This is a limitation of intensity grouping and simple centrality—not an implementation result to hide.

## Interpretation

The ordering of the baselines transfers: morphology and central-component selection improve the average. Their absolute accuracy does not. One external T1 dataset cannot establish universal failure, but it is enough to reject the claim that the SynthStrip score generalizes automatically.

The next experiment should preserve NFBS as untouched external evaluation and ask whether a learned spatial model reduces these specific failure modes. This remains educational research, not clinical validation.
