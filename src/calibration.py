"""Development-only helpers for calibrating prediction warning rules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detection_metrics(
    flagged: np.ndarray, failures: np.ndarray
) -> dict[str, int | float]:
    """Measure how well binary warning flags identify defined failures."""
    flags = np.asarray(flagged, dtype=bool)
    targets = np.asarray(failures, dtype=bool)
    if flags.shape != targets.shape:
        raise ValueError(f"Flag and failure shapes differ: {flags.shape} and {targets.shape}.")
    tp = int(np.count_nonzero(flags & targets))
    fp = int(np.count_nonzero(flags & ~targets))
    fn = int(np.count_nonzero(~flags & targets))
    tn = int(np.count_nonzero(~flags & ~targets))
    precision = float(tp / (tp + fp)) if tp + fp else 1.0
    recall = float(tp / (tp + fn)) if tp + fn else 1.0
    specificity = float(tn / (tn + fp)) if tn + fp else 1.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "warning_rate": float(flags.mean()) if flags.size else 0.0,
    }


def rank_calibration_results(results: pd.DataFrame) -> pd.DataFrame:
    """Rank warning configurations by detection quality and conservatism."""
    required = {"f1", "precision", "recall", "specificity", "warning_rate", "distance_from_default"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Calibration results are missing columns: {sorted(missing)}")
    return results.sort_values(
        [
            "f1",
            "recall",
            "precision",
            "specificity",
            "warning_rate",
            "distance_from_default",
        ],
        ascending=[False, False, False, False, True, True],
        ignore_index=True,
    )
