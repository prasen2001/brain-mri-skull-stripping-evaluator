"""Local, public-dataset-only explorer for the classical baselines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.dataset import discover_pairs, load_pair_with_spacing
from src.metrics import compute_binary_metrics, compute_boundary_metrics
from src.preprocessing import robust_normalize
from src.quality import QualityConfig, mean_method_disagreement, prediction_diagnostics
from src.segmentation import METHOD_NAMES, MorphologyConfig, segment

DATA_DIRECTORY = Path("data/processed/synthstrip_data_v1.5_2d")


def _load_defaults() -> tuple[MorphologyConfig, QualityConfig, float, float]:
    raw = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    morphology = raw["morphology"]
    quality = raw["quality"]
    normalization = raw["normalization"]
    central = raw["central_component"]
    return (
        MorphologyConfig(
            opening_radius_fraction=float(morphology["opening_radius_fraction"]),
            closing_radius_fraction=float(morphology["closing_radius_fraction"]),
            minimum_component_area_fraction=float(
                morphology["minimum_component_area_fraction"]
            ),
            centrality_weight=float(central["centrality_weight"]),
        ),
        QualityConfig(**quality),
        float(normalization["lower_percentile"]),
        float(normalization["upper_percentile"]),
    )


def _overlay_figure(
    image: np.ndarray, reference: np.ndarray, prediction: np.ndarray
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(image, cmap="gray", vmin=0, vmax=1)
    if reference.any():
        axis.contour(reference, levels=[0.5], colors=["cyan"], linewidths=1.5)
    if prediction.any():
        axis.contour(prediction, levels=[0.5], colors=["magenta"], linewidths=1.5)
    axis.set_title("Reference=cyan, prediction=magenta")
    axis.axis("off")
    figure.tight_layout()
    return figure


def _error_rgb(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    rgb = np.full((*prediction.shape, 3), 35, dtype=np.uint8)
    rgb[prediction & reference] = (44, 162, 95)
    rgb[prediction & ~reference] = (242, 142, 43)
    rgb[~prediction & reference] = (106, 81, 163)
    return rgb


def main() -> None:
    st.set_page_config(page_title="Skull-Stripping Evaluator", layout="wide")
    st.title("Brain MRI Skull-Stripping Evaluator")
    st.caption(
        "Educational classical baselines using only the downloaded public SynthStrip dataset. "
        "No upload control is provided."
    )
    if not DATA_DIRECTORY.is_dir():
        st.error("Dataset not found. Run `python -m scripts.download_data` first.")
        st.stop()

    default_morphology, quality_config, lower, upper = _load_defaults()
    subset = st.sidebar.selectbox("Dataset subset", ("test", "development"))
    pairs, invalid = discover_pairs(DATA_DIRECTORY, subset=subset, seed=42)
    if invalid:
        st.sidebar.warning(f"{len(invalid)} dataset candidates were invalid.")
    pair_by_identifier = {pair.identifier: pair for pair in pairs}
    identifier = st.sidebar.selectbox("Public dataset image", sorted(pair_by_identifier))

    st.sidebar.subheader("Morphology parameters")
    opening = st.sidebar.slider(
        "Opening radius fraction",
        0.001,
        0.030,
        float(default_morphology.opening_radius_fraction),
        0.001,
    )
    closing = st.sidebar.slider(
        "Closing radius fraction",
        0.001,
        0.030,
        float(default_morphology.closing_radius_fraction),
        0.001,
    )
    minimum_area = st.sidebar.slider(
        "Minimum component area fraction",
        0.001,
        0.050,
        float(default_morphology.minimum_component_area_fraction),
        0.001,
    )
    centrality = st.sidebar.slider(
        "Centrality weight",
        0.0,
        2.0,
        float(default_morphology.centrality_weight),
        0.05,
    )
    morphology = MorphologyConfig(opening, closing, minimum_area, centrality)

    image, reference, spacing_mm = load_pair_with_spacing(pair_by_identifier[identifier])
    normalized = robust_normalize(image, lower, upper)
    predictions = {
        method: segment(normalized, method, morphology)[0] for method in METHOD_NAMES
    }

    image_column, reference_column = st.columns(2)
    image_column.image(normalized, caption="Normalized MRI", clamp=True)
    reference_column.image(
        reference.astype(np.uint8) * 255, caption="Ground-truth brain mask", clamp=True
    )
    st.write(
        f"Image: `{identifier}` · shape: {image.shape} · spacing: "
        f"{spacing_mm[0]:.3f} × {spacing_mm[1]:.3f} mm"
    )

    metric_rows = []
    diagnostics_by_method = {}
    for method, prediction in predictions.items():
        disagreement = mean_method_disagreement(
            prediction,
            [other for name, other in predictions.items() if name != method],
        )
        diagnostics = prediction_diagnostics(
            prediction, disagreement, quality_config
        )
        diagnostics_by_method[method] = diagnostics
        metric_rows.append(
            {
                "method": method,
                **compute_binary_metrics(prediction, reference),
                **compute_boundary_metrics(
                    prediction, reference, tolerance_pixels=1.0, spacing_mm=spacing_mm
                ),
                "warnings": diagnostics["quality_warnings"],
            }
        )
    metric_frame = pd.DataFrame(metric_rows)
    st.subheader("Metrics")
    st.dataframe(
        metric_frame[
            [
                "method",
                "dice",
                "iou",
                "precision",
                "sensitivity",
                "specificity",
                "surface_dice",
                "hd95_mm",
                "assd_mm",
                "warnings",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    tabs = st.tabs(list(METHOD_NAMES))
    for tab, method in zip(tabs, METHOD_NAMES, strict=True):
        prediction = predictions[method]
        diagnostics = diagnostics_by_method[method]
        with tab:
            if diagnostics["has_quality_warning"]:
                st.warning(f"Review suggested: {diagnostics['quality_warnings']}")
            else:
                st.success("No geometry warning was triggered. This does not prove correctness.")
            prediction_column, overlay_column, error_column = st.columns(3)
            prediction_column.image(
                prediction.astype(np.uint8) * 255,
                caption="Predicted mask",
                clamp=True,
            )
            overlay = _overlay_figure(normalized, reference, prediction)
            overlay_column.pyplot(overlay, use_container_width=True)
            plt.close(overlay)
            error_column.image(
                _error_rgb(prediction, reference),
                caption="Error: TP=green, FP=orange, FN=purple",
                clamp=True,
            )
            st.json(diagnostics)

    st.divider()
    st.info(
        "Educational research prototype only. It is not clinically validated, diagnostic "
        "software, or suitable for patient-care decisions."
    )


if __name__ == "__main__":
    main()
