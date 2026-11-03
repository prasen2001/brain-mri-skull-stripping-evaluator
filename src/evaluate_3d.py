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
from src.volume_segmentation import VolumeMorphologyConfig, segment_volume
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
    return parser


def _load_config(
    path: Path, bootstrap_override: int | None
) -> tuple[VolumeMorphologyConfig, float, float, float, int, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"3D configuration file does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    normalization = raw.get("normalization", {})
    morphology = raw.get("morphology", {})
    central = raw.get("central_component", {})
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
        "evaluation": {
            "boundary_tolerance_voxels": tolerance,
            "bootstrap_samples": bootstrap_samples,
        },
    }
    return config, lower, upper, tolerance, bootstrap_samples, effective


def run(args: argparse.Namespace) -> None:
    """Evaluate full volumes, write measurements, and render review figures."""
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    config, lower, upper, tolerance, bootstrap_samples, effective = _load_config(
        args.config, args.bootstrap_samples
    )
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

    records: list[dict[str, Any]] = []
    evaluated = []
    for index, pair in enumerate(pairs, start=1):
        try:
            image, reference, spacing = load_volume_pair(pair)
            normalized = robust_normalize(image, lower, upper)
            predictions: dict[str, np.ndarray] = {}
            thresholds: dict[str, float] = {}
            timings: dict[str, float] = {}
            for method in METHOD_NAMES:
                started = time.perf_counter()
                prediction, threshold = segment_volume(normalized, method, config)
                timings[method] = time.perf_counter() - started
                predictions[method] = prediction
                thresholds[method] = threshold

            for method in METHOD_NAMES:
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

    metrics = pd.DataFrame.from_records(records)
    if metrics.empty:
        raise RuntimeError("Every candidate NFBS volume failed during evaluation.")
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
        metrics, pair_map, args.output_dir, config, lower, upper
    )
    run_config = {
        "created_utc": datetime.now(UTC).isoformat(),
        "experiment": "complete_3d_nfbs_external_evaluation",
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "limit": args.limit,
        "n_discovered": len(pairs),
        "n_evaluated": len(evaluated),
        "n_invalid": len(invalid),
        "methods": list(METHOD_NAMES),
        "best_method_by_mean_dice": best_method,
        "effective_parameters": effective,
        "parameters_tuned_on_nfbs": False,
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
