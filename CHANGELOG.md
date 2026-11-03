# Changelog

All notable project changes are documented here. The project has not yet published a formal release.

## Unreleased

### Added

- Reproducible external evaluation on 125 NFBS subjects.
- Deterministic retrospective failure-mode taxonomy and summary chart.
- Physical-space HD95 and ASSD from NIfTI spacing.
- Paired bootstrap intervals, Wilcoxon tests, and Holm correction.
- Modality- and source-specific analysis.
- Development-only quality-warning calibration.
- Ranked multi-method failure-review report.
- Optional public-dataset Streamlit explorer.
- Complete visualization/report tests and a 75% coverage gate.

### Changed

- Documentation now distinguishes internal SynthStrip results, prediction-only warnings, retrospective error categories, and external NFBS measurements.

### Known limitations

- Evaluation is based on one 2D sagittal slice per scan rather than complete 3D volumes.
- Frozen classical parameters reach only 0.5368 mean Dice on external NFBS slices.
- The software is an educational prototype and is not clinically validated.
