# Changelog

All notable project changes are documented here. The project has not yet published a formal release.

## Unreleased

### Added

- Reproducible external evaluation on 125 NFBS subjects.
- Complete 3D NFBS loader, canonical affine/spacing validation, and full-volume evaluator.
- Spherical 3D morphology, 26-connected component selection, and physical 3D surface metrics.
- Complete-volume aggregate charts and orthogonal best/worst review figures.
- Optional canonical 3D spatial-prior method with millimetre-aware morphology.
- Validated reuse of verified baseline rows for faster follow-up experiments.
- Generic paired statistics that include optional methods.
- Deterministic retrospective failure-mode taxonomy and summary chart.
- Physical-space HD95 and ASSD from NIfTI spacing.
- Paired bootstrap intervals, Wilcoxon tests, and Holm correction.
- Modality- and source-specific analysis.
- Development-only quality-warning calibration.
- Ranked multi-method failure-review report.
- Optional public-dataset Streamlit explorer.
- Complete visualization/report tests and a 75% coverage gate.

### Changed

- Documentation now distinguishes internal SynthStrip results, prediction-only warnings, retrospective error categories, constructed 2D NFBS measurements, and complete 3D NFBS measurements.
- NFBS spatial-prior results are labeled as exploratory second-look analysis rather than fresh external validation.

### Known limitations

- SynthStrip evaluation is based on one 2D sagittal slice per scan; complete 3D evidence is limited to one external T1 dataset.
- Frozen classical parameters reach only 0.5368 mean Dice on external NFBS slices.
- Transferred 3D morphology reaches only 0.5589 mean Dice on complete NFBS volumes and strongly over-segments scalp/neck tissue.
- The spatial prior assumes typical positioning in canonical coordinates and still reaches only 0.5146 Dice in its worst NFBS case.
- The software is an educational prototype and is not clinically validated.
