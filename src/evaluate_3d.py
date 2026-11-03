"""Evaluate the three frozen classical baselines on complete 3D NFBS volumes."""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluate import (
    _bootstrap_confidence_intervals,
    _package_versions,
    _paired_method_comparisons,
    _summarize,
)
from src.failure_analysis import categorize_failures, summarize_failure_categories
from src.metrics import compute_binary_metrics, compute_boundary_metrics
from src.preprocessing import robust_normalize
from src.segmentation import METHOD_NAMES
from src.volume_dataset import discover_nfbs_volumes, load_volume_pair
from src.volume_segmentation import (
    VOLUME_METHOD_NAMES,
    SpatialPriorConfig,
    VolumeMorphologyConfig,
    segment_volume,
)
from src.volume_visualization import create_volume_visualizations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/volume_default.yaml")
    )
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument(
        "--include-spatial-prior",
        action="store_true",
        help=(
            "Include the optional exploratory fourth method. NFBS has already "
            "been inspected, so this is a second-look analysis, not fresh validation."
        ),
    )
    parser.add_argument(
        "--reuse-baseline-metrics",
        type=Path,
        default=None,
        help=(
            "Reuse a verified three-method per_volume_metrics.csv and compute "
            "only the spatial prior. Requires --include-spatial-prior."
        ),
    )
    return parser


def _triple(
    value: object, name: str, default: tuple[float, float, float]
) -> tuple[float, float, float]:
    candidate = default if value is None else value
    if not isinstance(candidate, (list, tuple)) or len(candidate) != 3:
        raise ValueError(f"{name} must contain exactly three numeric values.")
    return (float(candidate[0]), float(candidate[1]), float(candidate[2]))


def _load_config(
    path: Path, bootstrap_override: int | None
) -> tuple[
    VolumeMorphologyConfig,
    SpatialPriorConfig,
    float,
    float,
    float,
    int,
    dict[str, Any],
]:
    if not path.is_file():
        raise FileNotFoundError(f"3D configuration file does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    normalization = raw.get("normalization", {})
    morphology = raw.get("morphology", {})
    central = raw.get("central_component", {})
    spatial = raw.get("spatial_prior", {})
    evaluation = raw.get("evaluation", {})
    lower = float(normalization.get("lower_percentile", 1.0))
    upper = float(normalization.get("upper_percentile", 99.0))
    config = VolumeMorphologyConfig(
        opening_radius_fraction=float(
            morphology.get("opening_radius_fraction", 0.008)
        ),
        closing_radius_fraction=float(
            morphology.get("closing_radius_fraction", 0.004)
        ),
        minimum_component_volume_fraction=float(
            morphology.get("minimum_component_volume_fraction", 0.005)
        ),
        centrality_weight=float(central.get("centrality_weight", 0.75)),
    )
    spatial_config = SpatialPriorConfig(
        center_fraction=_triple(
            spatial.get("center_fraction"),
            "spatial_prior.center_fraction",
            (0.50, 0.46, 0.61),
        ),
        radius_fraction=_triple(
            spatial.get("radius_fraction"),
            "spatial_prior.radius_fraction",
            (0.39, 0.42, 0.34),
        ),
        inferior_exclusion_fraction=float(
            spatial.get("inferior_exclusion_fraction", 0.25)
        ),
        opening_radius_mm=float(spatial.get("opening_radius_mm", 1.0)),
        closing_radius_mm=float(spatial.get("closing_radius_mm", 2.0)),
        minimum_component_volume_ml=float(
            spatial.get("minimum_component_volume_ml", 1.0)
        ),
    )
    tolerance = float(evaluation.get("boundary_tolerance_voxels", 1.0))
    bootstrap_samples = int(
        bootstrap_override
        if bootstrap_override is not None
        else evaluation.get("bootstrap_samples", 1000)
    )
    effective = {
        "normalization": {
            "lower_percentile": lower,
            "upper_percentile": upper,
        },
        "morphology": asdict(config),
        "spatial_prior": asdict(spatial_config),
        "evaluation": {
            "boundary_tolerance_voxels": tolerance,
            "bootstrap_samples": bootstrap_samples,
        },
    }
    return (
        config,
        spatial_config,
        lower,
        upper,
        tolerance,
        bootstrap_samples,
        effective,
    )


def _load_reused_baselines(path: Path, identifiers: set[str]) -> pd.DataFrame:
    """Load and validate complete baseline rows for the requested subjects."""
    if not path.is_file():
        raise FileNotFoundError(f"Baseline metrics file does not exist: {path}")
    metrics = pd.read_csv(path)
    required = {
        "identifier",
        "method",
        "tp",
        "tn",
        "fp",
        "fn",
        "dice",
        "iou",
        "precision",
        "sensitivity",
        "specificity",
        "surface_dice",
        "hd95_mm",
        "assd_mm",
        "quality_warnings",
        "processing_time_seconds",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Baseline metrics are missing columns: {sorted(missing)}")
    metrics["identifier"] = metrics["identifier"].astype(str)
    metrics["method"] = metrics["method"].astype(str)
    selected = metrics[metrics["identifier"].isin(identifiers)].copy()
    observed_identifiers = set(selected["identifier"])
    if observed_identifiers != identifiers:
        missing_ids = sorted(identifiers - observed_identifiers)
        raise ValueError(f"Baseline metrics are missing identifiers: {missing_ids}")
    if selected.duplicated(["identifier", "method"]).any():
        raise ValueError("Baseline metrics contain duplicate identifier-method rows.")
    expected_methods = set(METHOD_NAMES)
    observed_methods = set(selected["method"])
    if observed_methods != expected_methods:
        raise ValueError(
            "Baseline metrics must contain exactly the original methods: "
            f"{sorted(expected_methods)}; found {sorted(observed_methods)}."
        )
    counts = selected.groupby("identifier")["method"].nunique()
    if not counts.eq(len(METHOD_NAMES)).all():
        raise ValueError("Each reused subject must contain all original methods.")
    return selected


def run(args: argparse.Namespace) -> None:
    """Evaluate full volumes, write measurements, and render review figures."""
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")
    if args.reuse_baseline_metrics is not None and not args.include_spatial_prior:
        raise ValueError(
            "--reuse-baseline-metrics requires --include-spatial-prior."
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    (
        config,
        spatial_config,
        lower,
        upper,
        tolerance,
        bootstrap_samples,
        effective,
    ) = _load_config(args.config, args.bootstrap_samples)
    if tolerance < 0:
        raise ValueError("boundary tolerance must be non-negative.")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive.")

    pairs, invalid = discover_nfbs_volumes(args.data_dir)
    if args.limit is not None and args.limit < len(pairs):
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(len(pairs), args.limit, replace=False).tolist())
        pairs = [pairs[index] for index in indices]
    if not pairs:
        raise RuntimeError("No candidate NFBS volume pairs were found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reported_methods = (
        VOLUME_METHOD_NAMES if args.include_spatial_prior else METHOD_NAMES
    )
    methods_to_run = (
        (VOLUME_METHOD_NAMES[-1],)
        if args.reuse_baseline_metrics is not None
        else reported_methods
    )
    reused_baselines = (
        _load_reused_baselines(
            args.reuse_baseline_metrics, {pair.identifier for pair in pairs}
        )
        if args.reuse_baseline_metrics is not None
        else None
    )

    records: list[dict[str, Any]] = []
    evaluated = []
    for index, pair in enumerate(pairs, start=1):
        try:
            image, reference, spacing = load_volume_pair(pair)
            normalized = robust_normalize(image, lower, upper)
            predictions: dict[str, np.ndarray] = {}
            thresholds: dict[str, float] = {}
            timings: dict[str, float] = {}
            for method in methods_to_run:
                started = time.perf_counter()
                prediction, threshold = segment_volume(
                    normalized, method, config, spacing, spatial_config
                )
                timings[method] = time.perf_counter() - started
                predictions[method] = prediction
                thresholds[method] = threshold

            for method in methods_to_run:
                prediction = predictions[method]
                overlap = compute_binary_metrics(prediction, reference)
                boundary = compute_boundary_metrics(
                    prediction, reference, tolerance, spacing
                )
                records.append(
                    {
                        "identifier": pair.identifier,
                        "subset": "external_test",
                        "method": method,
                        "volume_shape": "x".join(str(size) for size in image.shape),
                        "voxel_spacing_mm": "x".join(f"{value:.6g}" for value in spacing),
                        "otsu_threshold": thresholds[method],
                        **overlap,
                        **boundary,
                        "hd95_voxels": boundary["hd95_pixels"],
                        "assd_voxels": boundary["assd_pixels"],
                        "predicted_volume_fraction": float(prediction.mean()),
                        # The 2D warning thresholds were not transferred to 3D.
                        "quality_warnings": "none",
                        "processing_time_seconds": timings[method],
                    }
                )
            evaluated.append(pair)
        except Exception as exc:  # one corrupt volume should not discard a run
            invalid.append(
                {"identifier": pair.identifier, "reason": f"Evaluation failed: {exc}"}
            )
        if index == 1 or index % 5 == 0 or index == len(pairs):
            print(f"Processed {index}/{len(pairs)} candidate volumes", flush=True)

    new_metrics = pd.DataFrame.from_records(records)
    if new_metrics.empty:
        raise RuntimeError("Every candidate NFBS volume failed during evaluation.")
    if reused_baselines is not None:
        evaluated_ids = {pair.identifier for pair in evaluated}
        reused = reused_baselines[
            reused_baselines["identifier"].isin(evaluated_ids)
        ].copy()
        metrics = pd.concat((reused, new_metrics), ignore_index=True, sort=False)
    else:
        metrics = new_metrics
    metrics = categorize_failures(metrics)
    metrics.to_csv(args.output_dir / "per_volume_metrics.csv", index=False)
    summary = _summarize(metrics)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    confidence = _bootstrap_confidence_intervals(metrics, bootstrap_samples, args.seed)
    confidence.to_csv(
        args.output_dir / "bootstrap_confidence_intervals.csv", index=False
    )
    paired = _paired_method_comparisons(metrics, bootstrap_samples, args.seed)
    paired.to_csv(args.output_dir / "paired_method_comparisons.csv", index=False)
    failure_summary = summarize_failure_categories(metrics)
    failure_summary.to_csv(
        args.output_dir / "failure_category_summary.csv", index=False
    )
    if invalid:
        pd.DataFrame(invalid).to_csv(args.output_dir / "invalid_pairs.csv", index=False)

    pair_map = {pair.identifier: pair for pair in evaluated}
    best_method = create_volume_visualizations(
        metrics,
        pair_map,
        args.output_dir,
        config,
        spatial_config,
        lower,
        upper,
    )
    run_config = {
        "created_utc": datetime.now(UTC).isoformat(),
        "experiment": (
            "exploratory_3d_nfbs_spatial_prior_second_look"
            if args.include_spatial_prior
            else "complete_3d_nfbs_external_evaluation"
        ),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "limit": args.limit,
        "n_discovered": len(pairs),
        "n_evaluated": len(evaluated),
        "n_invalid": len(invalid),
        "methods": list(reported_methods),
        "best_method_by_mean_dice": best_method,
        "effective_parameters": effective,
        "parameters_tuned_on_nfbs": False,
        "nfbs_previously_inspected": bool(args.include_spatial_prior),
        "reused_baseline_metrics": (
            str(args.reuse_baseline_metrics)
            if args.reuse_baseline_metrics is not None
            else None
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    print(f"Evaluated {len(evaluated)} complete volumes. Results: {args.output_dir}")
    print(summary[["method", "n_images", "dice_mean", "iou_mean"]].to_string(index=False))


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
