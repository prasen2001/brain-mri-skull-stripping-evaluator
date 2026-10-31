"""Small reproducible morphology grid search on the development split only."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from src.dataset import discover_pairs, load_pair
from src.metrics import compute_binary_metrics
from src.preprocessing import robust_normalize
from src.segmentation import MorphologyConfig, central_component, otsu_morphology


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/development_tuning.csv")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs, invalid = discover_pairs(args.data_dir, "development", args.seed)
    if invalid:
        print(f"Warning: discovery reported {len(invalid)} invalid candidates")
    loaded = []
    for pair in pairs:
        image, reference = load_pair(pair)
        loaded.append((robust_normalize(image), reference))

    rows = []
    grid = product(
        (0.002, 0.004, 0.008),
        (0.004, 0.008, 0.015),
        (0.001, 0.002, 0.005),
        (0.5, 0.75, 1.0),
    )
    for opening, closing, minimum_area, centrality in grid:
        config = MorphologyConfig(opening, closing, minimum_area, centrality)
        morphology_dice = []
        component_dice = []
        for image, reference in loaded:
            method_2, _ = otsu_morphology(image, config)
            method_3, _ = central_component(image, config)
            morphology_dice.append(compute_binary_metrics(method_2, reference)["dice"])
            component_dice.append(compute_binary_metrics(method_3, reference)["dice"])
        rows.append(
            {
                "opening_radius_fraction": opening,
                "closing_radius_fraction": closing,
                "minimum_component_area_fraction": minimum_area,
                "centrality_weight": centrality,
                "otsu_morphology_mean_dice": sum(morphology_dice) / len(morphology_dice),
                "central_component_mean_dice": sum(component_dice) / len(component_dice),
                "combined_mean_dice": (
                    sum(morphology_dice) + sum(component_dice)
                )
                / (2 * len(loaded)),
                "n_development_scans": len(loaded),
            }
        )
    results = pd.DataFrame(rows).sort_values(
        ["central_component_mean_dice", "combined_mean_dice"], ascending=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.head(10).to_string(index=False))
    print(f"Saved all {len(results)} configurations to {args.output}")


if __name__ == "__main__":
    main()
