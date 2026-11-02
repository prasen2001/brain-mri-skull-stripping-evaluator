"""Calibrate warning thresholds on the official development split only."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.calibration import detection_metrics, rank_calibration_results
from src.dataset import discover_pairs, load_pair
from src.metrics import compute_binary_metrics
from src.preprocessing import robust_normalize
from src.quality import QualityConfig, mean_method_disagreement, prediction_diagnostics
from src.segmentation import METHOD_NAMES, MorphologyConfig, segment


def _configs(path: Path) -> tuple[MorphologyConfig, QualityConfig, float, float]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    normalization = raw.get("normalization", {})
    morphology = raw.get("morphology", {})
    central = raw.get("central_component", {})
    quality = raw.get("quality", {})
    return (
        MorphologyConfig(
            opening_radius_fraction=float(morphology.get("opening_radius_fraction", 0.008)),
            closing_radius_fraction=float(morphology.get("closing_radius_fraction", 0.004)),
            minimum_component_area_fraction=float(
                morphology.get("minimum_component_area_fraction", 0.005)
            ),
            centrality_weight=float(central.get("centrality_weight", 0.75)),
        ),
        QualityConfig(
            minimum_area_fraction=float(quality.get("minimum_area_fraction", 0.03)),
            maximum_area_fraction=float(quality.get("maximum_area_fraction", 0.80)),
            maximum_centroid_distance=float(
                quality.get("maximum_centroid_distance", 0.55)
            ),
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
        ),
        float(normalization.get("lower_percentile", 1.0)),
        float(normalization.get("upper_percentile", 99.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/quality_calibration"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--failure-dice-threshold", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.failure_dice_threshold < 1:
        raise ValueError("--failure-dice-threshold must lie between 0 and 1.")

    morphology, default_quality, lower, upper = _configs(args.config)
    pairs, invalid = discover_pairs(args.data_dir, "development", args.seed)
    if invalid:
        print(f"Warning: discovery reported {len(invalid)} invalid candidates")

    cases = []
    for pair in pairs:
        image, reference = load_pair(pair)
        normalized = robust_normalize(image, lower, upper)
        predictions = {
            method: segment(normalized, method, morphology)[0] for method in METHOD_NAMES
        }
        prediction = predictions["central_component"]
        disagreement = mean_method_disagreement(
            prediction,
            [predictions[method] for method in METHOD_NAMES if method != "central_component"],
        )
        cases.append(
            {
                "identifier": pair.identifier,
                "prediction": prediction,
                "method_disagreement": disagreement,
                "dice": float(compute_binary_metrics(prediction, reference)["dice"]),
            }
        )

    rows = []
    minimum_areas = sorted(
        {0.01, 0.02, 0.03, 0.04, 0.05, default_quality.minimum_area_fraction}
    )
    disagreements = sorted(
        {0.20, 0.25, 0.30, 0.35, 0.40, 0.50, default_quality.maximum_method_disagreement}
    )
    failures = np.asarray(
        [case["dice"] < args.failure_dice_threshold for case in cases], dtype=bool
    )
    for minimum_area in minimum_areas:
        for maximum_disagreement in disagreements:
            candidate = replace(
                default_quality,
                minimum_area_fraction=minimum_area,
                maximum_method_disagreement=maximum_disagreement,
            )
            flagged = np.asarray(
                [
                    prediction_diagnostics(
                        case["prediction"], case["method_disagreement"], candidate
                    )["has_quality_warning"]
                    for case in cases
                ],
                dtype=bool,
            )
            rows.append(
                {
                    "minimum_area_fraction": minimum_area,
                    "maximum_method_disagreement": maximum_disagreement,
                    "failure_dice_threshold": args.failure_dice_threshold,
                    "n_development_scans": len(cases),
                    "n_failures": int(failures.sum()),
                    **detection_metrics(flagged, failures),
                    "distance_from_default": abs(
                        minimum_area - default_quality.minimum_area_fraction
                    )
                    + abs(
                        maximum_disagreement
                        - default_quality.maximum_method_disagreement
                    ),
                }
            )
    ranked = rank_calibration_results(pd.DataFrame(rows))
    recommendation = ranked.iloc[0].to_dict()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output_dir / "calibration_grid.csv", index=False)
    pd.DataFrame(
        [
            {
                "identifier": case["identifier"],
                "dice": case["dice"],
                "method_disagreement": case["method_disagreement"],
                "is_failure": case["dice"] < args.failure_dice_threshold,
            }
            for case in cases
        ]
    ).to_csv(args.output_dir / "development_cases.csv", index=False)
    (args.output_dir / "recommended_quality_config.json").write_text(
        json.dumps(recommendation, indent=2), encoding="utf-8"
    )
    print(ranked.head(10).to_string(index=False))
    print(
        "Recommendation is development-only; review it before changing configs/default.yaml."
    )


if __name__ == "__main__":
    main()
