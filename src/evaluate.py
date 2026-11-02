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
from scipy import stats
import yaml

from src.dataset import discover_pairs, load_pair_with_spacing, validate_pairs
from src.metrics import compute_binary_metrics, compute_boundary_metrics
from src.preprocessing import robust_normalize
from src.quality import QualityConfig, mean_method_disagreement, prediction_diagnostics
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
    parser.add_argument("--boundary-tolerance-pixels", type=float, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    return parser


def _load_config(
    args: argparse.Namespace,
) -> tuple[MorphologyConfig, QualityConfig, float, float, float, int, dict[str, Any]]:
    if not args.config.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {args.config}")
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    normalization = raw.get("normalization", {})
    morphology = raw.get("morphology", {})
    central = raw.get("central_component", {})
    quality = raw.get("quality", {})
    evaluation = raw.get("evaluation", {})

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
    quality_config = QualityConfig(
        minimum_area_fraction=float(quality.get("minimum_area_fraction", 0.03)),
        maximum_area_fraction=float(quality.get("maximum_area_fraction", 0.80)),
        maximum_centroid_distance=float(quality.get("maximum_centroid_distance", 0.55)),
        maximum_border_contact_fraction=float(
            quality.get("maximum_border_contact_fraction", 0.65)
        ),
        maximum_component_count=int(quality.get("maximum_component_count", 20)),
        minimum_largest_component_fraction=float(
            quality.get("minimum_largest_component_fraction", 0.80)
        ),
        maximum_method_disagreement=float(
            quality.get("maximum_method_disagreement", 0.35)
        ),
    )
    boundary_tolerance = chosen(
        args.boundary_tolerance_pixels,
        evaluation,
        "boundary_tolerance_pixels",
        1.0,
    )
    bootstrap_samples = int(
        args.bootstrap_samples
        if args.bootstrap_samples is not None
        else evaluation.get("bootstrap_samples", 1000)
    )
    effective = {
        "normalization": {"lower_percentile": lower, "upper_percentile": upper},
        "morphology": {
            "opening_radius_fraction": config.opening_radius_fraction,
            "closing_radius_fraction": config.closing_radius_fraction,
            "minimum_component_area_fraction": config.minimum_component_area_fraction,
        },
        "central_component": {"centrality_weight": config.centrality_weight},
        "quality": {
            "minimum_area_fraction": quality_config.minimum_area_fraction,
            "maximum_area_fraction": quality_config.maximum_area_fraction,
            "maximum_centroid_distance": quality_config.maximum_centroid_distance,
            "maximum_border_contact_fraction": quality_config.maximum_border_contact_fraction,
            "maximum_component_count": quality_config.maximum_component_count,
            "minimum_largest_component_fraction": quality_config.minimum_largest_component_fraction,
            "maximum_method_disagreement": quality_config.maximum_method_disagreement,
        },
        "evaluation": {
            "boundary_tolerance_pixels": boundary_tolerance,
            "bootstrap_samples": bootstrap_samples,
        },
    }
    return (
        config,
        quality_config,
        lower,
        upper,
        boundary_tolerance,
        bootstrap_samples,
        effective,
    )


def _package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "matplotlib", "seaborn", "scikit-image", "scipy", "nibabel", "PyYAML"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


SUMMARY_METRICS = [
    "dice",
    "iou",
    "precision",
    "sensitivity",
    "specificity",
    "pixel_accuracy",
    "surface_dice",
    "hd95_pixels",
    "assd_pixels",
    "hd95_mm",
    "assd_mm",
]
PAIRED_METRICS = ["dice", "iou", "surface_dice", "hd95_mm", "assd_mm"]
HIGHER_IS_BETTER = {"dice", "iou", "surface_dice"}


def _summarize_grouped(
    metrics: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    """Aggregate overlap, boundary, and timing metrics for requested groups."""
    rows: list[dict[str, Any]] = []
    group_key: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in metrics.groupby(group_key, sort=False):
        key_values = (keys,) if len(group_columns) == 1 else keys
        row: dict[str, Any] = dict(zip(group_columns, key_values, strict=True))
        row["n_images"] = int(len(group))
        for column in SUMMARY_METRICS:
            values = group[column]
            for statistic in ("mean", "median", "std", "min", "max"):
                row[f"{column}_{statistic}"] = float(getattr(values, statistic)())
        row["processing_time_total_seconds"] = float(group["processing_time_seconds"].sum())
        row["processing_time_average_seconds"] = float(group["processing_time_seconds"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return _summarize_grouped(metrics, ["method"])


def _bootstrap_confidence_intervals(
    metrics: pd.DataFrame, sample_count: int, seed: int
) -> pd.DataFrame:
    """Compute deterministic scan-level percentile CIs for mean metrics."""
    rows: list[dict[str, Any]] = []
    for method, group in metrics.groupby("method", sort=False):
        ordered = group.sort_values("identifier")
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(ordered), size=(sample_count, len(ordered)))
        for metric in SUMMARY_METRICS:
            values = ordered[metric].to_numpy(dtype=float)
            bootstrap_means = values[indices].mean(axis=1)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "ci_95_lower": float(np.percentile(bootstrap_means, 2.5)),
                    "ci_95_upper": float(np.percentile(bootstrap_means, 97.5)),
                    "bootstrap_samples": sample_count,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def _quality_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize warning frequency and observed Dice for audit purposes."""
    rows: list[dict[str, Any]] = []
    for method, group in metrics.groupby("method", sort=False):
        warned = group["has_quality_warning"].astype(bool)
        warning_counts: dict[str, int] = {}
        for warning_text in group.loc[warned, "quality_warnings"]:
            for warning in str(warning_text).split(";"):
                warning_counts[warning] = warning_counts.get(warning, 0) + 1
        rows.append(
            {
                "method": method,
                "n_images": int(len(group)),
                "n_warned": int(warned.sum()),
                "warning_rate": float(warned.mean()),
                "mean_warning_count": float(group["quality_warning_count"].mean()),
                "mean_dice_warned": float(group.loc[warned, "dice"].mean())
                if warned.any()
                else float("nan"),
                "mean_dice_not_warned": float(group.loc[~warned, "dice"].mean())
                if (~warned).any()
                else float("nan"),
                "warning_type_counts": ";".join(
                    f"{name}:{count}" for name, count in sorted(warning_counts.items())
                )
                or "none",
            }
        )
    return pd.DataFrame(rows)


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values while preserving original order."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_maximum = 0.0
    count = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[original_index])
        running_maximum = max(running_maximum, candidate)
        adjusted[original_index] = running_maximum
    return adjusted.tolist()


def _paired_method_comparisons(
    metrics: pd.DataFrame, sample_count: int, seed: int
) -> pd.DataFrame:
    """Compare methods on the same scans using paired bootstrap differences."""
    rows: list[dict[str, Any]] = []
    for baseline_index, baseline_method in enumerate(METHOD_NAMES[:-1]):
        for comparison_method in METHOD_NAMES[baseline_index + 1 :]:
            baseline = metrics[metrics["method"] == baseline_method].set_index(
                "identifier"
            )
            comparison = metrics[metrics["method"] == comparison_method].set_index(
                "identifier"
            )
            identifiers = sorted(set(baseline.index) & set(comparison.index))
            if not identifiers:
                continue
            for metric in PAIRED_METRICS:
                baseline_values = baseline.loc[identifiers, metric].to_numpy(dtype=float)
                comparison_values = comparison.loc[identifiers, metric].to_numpy(
                    dtype=float
                )
                raw_difference = comparison_values - baseline_values
                improvement = (
                    raw_difference
                    if metric in HIGHER_IS_BETTER
                    else -raw_difference
                )
                rng = np.random.default_rng(seed)
                indices = rng.integers(
                    0, len(identifiers), size=(sample_count, len(identifiers))
                )
                bootstrap_means = improvement[indices].mean(axis=1)
                if np.allclose(improvement, 0):
                    p_value = 1.0
                else:
                    p_value = float(
                        stats.wilcoxon(
                            improvement,
                            alternative="two-sided",
                            zero_method="wilcox",
                        ).pvalue
                    )
                tolerance = 1e-12
                rows.append(
                    {
                        "baseline_method": baseline_method,
                        "comparison_method": comparison_method,
                        "metric": metric,
                        "n_paired_scans": len(identifiers),
                        "baseline_mean": float(baseline_values.mean()),
                        "comparison_mean": float(comparison_values.mean()),
                        "raw_comparison_minus_baseline": float(raw_difference.mean()),
                        "mean_improvement": float(improvement.mean()),
                        "median_improvement": float(np.median(improvement)),
                        "improvement_ci_95_lower": float(
                            np.percentile(bootstrap_means, 2.5)
                        ),
                        "improvement_ci_95_upper": float(
                            np.percentile(bootstrap_means, 97.5)
                        ),
                        "comparison_wins": int(np.count_nonzero(improvement > tolerance)),
                        "baseline_wins": int(np.count_nonzero(improvement < -tolerance)),
                        "ties": int(np.count_nonzero(np.abs(improvement) <= tolerance)),
                        "comparison_win_rate": float(
                            np.count_nonzero(improvement > tolerance) / len(improvement)
                        ),
                        "wilcoxon_p_value": p_value,
                        "bootstrap_samples": sample_count,
                        "seed": seed,
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["holm_adjusted_p_value"] = _holm_adjust(
            result["wilcoxon_p_value"].tolist()
        )
    return result


def _dataset_source(identifier: str) -> str:
    """Return the official source-family prefix embedded in a scan identifier."""
    return identifier.split("_", maxsplit=1)[0].upper()


def run(args: argparse.Namespace) -> None:
    """Validate inputs, evaluate methods, save tables, and render figures."""
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    (
        config,
        quality_config,
        lower,
        upper,
        boundary_tolerance,
        bootstrap_samples,
        effective_config,
    ) = _load_config(args)
    if boundary_tolerance < 0:
        raise ValueError("--boundary-tolerance-pixels must be non-negative.")
    if bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be a positive integer.")
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
            image, reference, spacing_mm = load_pair_with_spacing(pair)
            normalized = robust_normalize(image, lower, upper)
            predictions: dict[str, np.ndarray] = {}
            thresholds: dict[str, float] = {}
            timings: dict[str, float] = {}
            for method in METHOD_NAMES:
                started = time.perf_counter()
                prediction, threshold = segment(normalized, method, config)
                timings[method] = time.perf_counter() - started
                predictions[method] = prediction
                thresholds[method] = threshold

            for method in METHOD_NAMES:
                prediction = predictions[method]
                metric_values = compute_binary_metrics(prediction, reference)
                boundary_values = compute_boundary_metrics(
                    prediction, reference, boundary_tolerance, spacing_mm
                )
                disagreement = mean_method_disagreement(
                    prediction,
                    [predictions[other] for other in METHOD_NAMES if other != method],
                )
                diagnostics = prediction_diagnostics(
                    prediction, disagreement, quality_config
                )
                records.append(
                    {
                        "identifier": pair.identifier,
                        "dataset_source": _dataset_source(pair.identifier),
                        "subset": pair.subset,
                        "method": method,
                        "image_height": int(image.shape[0]),
                        "image_width": int(image.shape[1]),
                        "row_spacing_mm": spacing_mm[0],
                        "column_spacing_mm": spacing_mm[1],
                        "otsu_threshold": thresholds[method],
                        **metric_values,
                        **boundary_values,
                        **diagnostics,
                        "processing_time_seconds": timings[method],
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
    source_summary = _summarize_grouped(metrics, ["dataset_source", "method"])
    source_summary.to_csv(args.output_dir / "source_summary_metrics.csv", index=False)
    confidence_intervals = _bootstrap_confidence_intervals(
        metrics, bootstrap_samples, args.seed
    )
    confidence_intervals.to_csv(
        args.output_dir / "bootstrap_confidence_intervals.csv", index=False
    )
    quality_summary = _quality_summary(metrics)
    quality_summary.to_csv(args.output_dir / "quality_summary.csv", index=False)
    paired_comparisons = _paired_method_comparisons(
        metrics, bootstrap_samples, args.seed
    )
    paired_comparisons.to_csv(
        args.output_dir / "paired_method_comparisons.csv", index=False
    )
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

