"""Command-line evaluation for the three classical skull-stripping baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.dataset import discover_pairs, load_pair, validate_pairs
from src.metrics import compute_binary_metrics
from src.preprocessing import robust_normalize
from src.segmentation import METHOD_NAMES, MorphologyConfig, segment
from src.visualization import create_all_visualizations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate three classical skull-stripping baselines on 2D NIfTI pairs."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--subset",
        choices=("all", "development", "dev", "validation", "valid", "test"),
        default="test",
    )
    parser.add_argument("--limit", type=int, default=None, help="Deterministic sample size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--opening-radius-fraction", type=float, default=None)
    parser.add_argument("--closing-radius-fraction", type=float, default=None)
    parser.add_argument("--minimum-component-area-fraction", type=float, default=None)
    parser.add_argument("--centrality-weight", type=float, default=None)
    parser.add_argument("--lower-percentile", type=float, default=None)
    parser.add_argument("--upper-percentile", type=float, default=None)
    return parser


def _load_config(args: argparse.Namespace) -> tuple[MorphologyConfig, float, float, dict[str, Any]]:
    if not args.config.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {args.config}")
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    normalization = raw.get("normalization", {})
    morphology = raw.get("morphology", {})
    central = raw.get("central_component", {})

    def chosen(cli_value: float | None, section: dict[str, Any], key: str, default: float) -> float:
        return float(cli_value if cli_value is not None else section.get(key, default))

    lower = chosen(args.lower_percentile, normalization, "lower_percentile", 1.0)
    upper = chosen(args.upper_percentile, normalization, "upper_percentile", 99.0)
    config = MorphologyConfig(
        opening_radius_fraction=chosen(
            args.opening_radius_fraction, morphology, "opening_radius_fraction", 0.004
        ),
        closing_radius_fraction=chosen(
            args.closing_radius_fraction, morphology, "closing_radius_fraction", 0.008
        ),
        minimum_component_area_fraction=chosen(
            args.minimum_component_area_fraction,
            morphology,
            "minimum_component_area_fraction",
            0.002,
        ),
        centrality_weight=chosen(
            args.centrality_weight, central, "centrality_weight", 0.75
        ),
    )
    effective = {
        "normalization": {"lower_percentile": lower, "upper_percentile": upper},
        "morphology": {
            "opening_radius_fraction": config.opening_radius_fraction,
            "closing_radius_fraction": config.closing_radius_fraction,
            "minimum_component_area_fraction": config.minimum_component_area_fraction,
        },
        "central_component": {"centrality_weight": config.centrality_weight},
    }
    return config, lower, upper, effective


def _package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "matplotlib", "seaborn", "scikit-image", "scipy", "nibabel", "PyYAML"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    score_columns = ["dice", "iou", "precision", "sensitivity", "specificity", "pixel_accuracy"]
    rows: list[dict[str, Any]] = []
    for method, group in metrics.groupby("method", sort=False):
        row: dict[str, Any] = {"method": method, "n_images": int(len(group))}
        for column in score_columns:
            values = group[column]
            for statistic in ("mean", "median", "std", "min", "max"):
                row[f"{column}_{statistic}"] = float(getattr(values, statistic)())
        row["processing_time_total_seconds"] = float(group["processing_time_seconds"].sum())
        row["processing_time_average_seconds"] = float(group["processing_time_seconds"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    """Validate inputs, evaluate methods, save tables, and render figures."""
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    config, lower, upper, effective_config = _load_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    discovered, discovery_invalid = discover_pairs(args.data_dir, args.subset, args.seed)
    valid_pairs, validation_invalid = validate_pairs(discovered)
    invalid = discovery_invalid + validation_invalid
    if args.limit is not None and args.limit < len(valid_pairs):
        rng = np.random.default_rng(args.seed)
        selected = sorted(rng.choice(len(valid_pairs), size=args.limit, replace=False).tolist())
        valid_pairs = [valid_pairs[index] for index in selected]
    if not valid_pairs:
        raise RuntimeError("No valid image-mask pairs matched the requested subset.")

    records: list[dict[str, Any]] = []
    evaluated_pairs = []
    for index, pair in enumerate(valid_pairs, start=1):
        try:
            image, reference = load_pair(pair)
            normalized = robust_normalize(image, lower, upper)
            for method in METHOD_NAMES:
                started = time.perf_counter()
                prediction, threshold = segment(normalized, method, config)
                elapsed = time.perf_counter() - started
                metric_values = compute_binary_metrics(prediction, reference)
                records.append(
                    {
                        "identifier": pair.identifier,
                        "subset": pair.subset,
                        "method": method,
                        "image_height": int(image.shape[0]),
                        "image_width": int(image.shape[1]),
                        "otsu_threshold": threshold,
                        **metric_values,
                        "processing_time_seconds": elapsed,
                    }
                )
            evaluated_pairs.append(pair)
        except Exception as exc:  # one corrupt scan should not discard a run
            invalid.append({"identifier": pair.identifier, "reason": f"Evaluation failed: {exc}"})
        if index == 1 or index % 50 == 0 or index == len(valid_pairs):
            print(f"Processed {index}/{len(valid_pairs)} candidate pairs", flush=True)

    metrics = pd.DataFrame.from_records(records)
    if metrics.empty:
        raise RuntimeError("Every candidate pair failed during evaluation.")
    metrics.to_csv(args.output_dir / "per_image_metrics.csv", index=False)
    summary = _summarize(metrics)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    if invalid:
        pd.DataFrame(invalid).to_csv(args.output_dir / "invalid_pairs.csv", index=False)

    pair_map = {pair.identifier: pair for pair in evaluated_pairs}
    best_method = create_all_visualizations(
        metrics, pair_map, args.output_dir, config, lower, upper
    )
    run_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "requested_subset": args.subset,
        "seed": args.seed,
        "limit": args.limit,
        "n_discovered_for_subset": len(discovered),
        "n_evaluated": len(evaluated_pairs),
        "n_invalid": len(invalid),
        "methods": list(METHOD_NAMES),
        "best_method_by_mean_dice": best_method,
        "effective_parameters": effective_config,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    print(f"Evaluated {len(evaluated_pairs)} pairs. Results: {args.output_dir}")
    print(summary[["method", "n_images", "dice_mean", "iou_mean"]].to_string(index=False))


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()

