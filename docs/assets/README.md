# Tracked aggregate figures

These aggregate charts are generated from the official SynthStrip 2D v1.5 test
split by `python -m src.evaluate`. They contain summary statistics only and no
individual MRI images. Full outputs and case overlays remain under ignored
`outputs/` directories and can be regenerated using the project README.

The tracked figures cover overall overlap, source and modality stratification,
physical boundary quality, paired per-scan improvement, and prediction-only
warning rates. They were refreshed from the verified 558-scan test run with
seed 42.

The two `nfbs_external_*` figures summarize a separate 125-subject NFBS run
using frozen parameters. Only aggregate charts are tracked; individual MRI case
figures remain in ignored local outputs.

The two `nfbs_3d_*` figures summarize the verified complete-volume run on the
same 125 subjects. They contain only aggregate measurements. Orthogonal MRI
review figures remain under ignored `outputs/nfbs_3d_full/` paths and are not
published in the repository.

The two `nfbs_3d_spatial_*` figures summarize an exploratory follow-up performed
after the baseline NFBS failures were inspected. They are tracked to document
the measured hypothesis test, but they must not be described as untouched
external validation.
