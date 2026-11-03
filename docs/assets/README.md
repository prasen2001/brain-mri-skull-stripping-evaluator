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
