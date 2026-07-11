"""Strict artifact reader for the Study 2 Haufe figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study2.sensor_patterns import SensorPatternSummary

FIGURE_CONFIG_KEY = "study2.figures.haufe_forward_patterns"


def load_haufe_forward_pattern_artifacts(
    *,
    fold_path: Path,
    aggregate_path: Path,
    stability_path: Path,
    config: Any,
) -> SensorPatternSummary:
    """Read and validate the three artifacts written by the Study 2 Haufe stage."""

    figure_config = require_config_value(config, FIGURE_CONFIG_KEY)
    if not isinstance(figure_config, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY} must be a mapping.")
    bands = tuple(str(spec["name"]) for spec in figure_config["bands"])
    fold_patterns = _read_table(fold_path, "fold patterns")
    aggregate = _read_table(aggregate_path, "aggregate patterns")
    stability = _read_table(stability_path, "stability")
    _require_columns(
        fold_patterns,
        {"target", "fold", "test_subject", "band", "channel", "normalized_pattern"},
        "Fold patterns",
    )
    _require_columns(
        aggregate,
        {"target", "band", "channel", "median_normalized_pattern", "n_folds"},
        "Aggregate patterns",
    )
    _require_columns(
        stability,
        {"target", "band", "comparison_index", "fold_a", "fold_b", "spatial_correlation"},
        "Stability",
    )
    for label, frame in (
        ("fold patterns", fold_patterns),
        ("aggregate patterns", aggregate),
        ("stability", stability),
    ):
        if set(frame["target"].astype(str)) != {str(figure_config["target"])}:
            raise ValueError(f"Study 2 {label} must contain only the configured NPS target.")
        if set(frame["band"].astype(str)) != set(bands):
            raise ValueError(f"Study 2 {label} does not contain every configured band.")
    numeric_columns = (
        (fold_patterns, "normalized_pattern"),
        (aggregate, "median_normalized_pattern"),
        (stability, "spatial_correlation"),
    )
    if any(
        not np.isfinite(pd.to_numeric(frame[column], errors="coerce").to_numpy()).all()
        for frame, column in numeric_columns
    ):
        raise ValueError("Study 2 Haufe artifacts contain non-finite numerical values.")
    if not stability["spatial_correlation"].between(-1.0, 1.0).all():
        raise ValueError("Study 2 fold-map correlations must lie in [-1, 1].")

    subjects = tuple(sorted(fold_patterns["test_subject"].astype(str).unique()))
    folds = tuple(sorted(pd.to_numeric(fold_patterns["fold"], errors="raise").unique()))
    if folds != tuple(range(len(subjects))):
        raise ValueError("Study 2 Haufe artifacts must use consecutive LOSO folds from zero.")
    channels = tuple(sorted(aggregate["channel"].astype(str).unique()))
    expected_map_rows = len(bands) * len(channels)
    if len(aggregate) != expected_map_rows or aggregate.duplicated(["band", "channel"]).any():
        raise ValueError("Aggregate Haufe artifacts must contain one complete map per band.")
    if set(pd.to_numeric(aggregate["n_folds"], errors="raise")) != {len(subjects)}:
        raise ValueError("Aggregate Haufe fold counts disagree with the fold artifact.")
    return SensorPatternSummary(
        fold_patterns=fold_patterns,
        aggregate_patterns=aggregate,
        stability=stability,
        target=str(figure_config["target"]),
        bands=bands,
        channels=channels,
        n_subjects=len(subjects),
        article_ready=len(subjects) >= int(figure_config["minimum_article_subjects"]),
    )


def _read_table(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Study 2 Haufe {label} artifact not found: {path}")
    frame = pd.read_csv(path, sep="\t")
    if frame.empty:
        raise ValueError(f"Study 2 Haufe {label} artifact is empty: {path}")
    return frame


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} artifact is missing required columns: {missing}.")


__all__ = ["load_haufe_forward_pattern_artifacts"]
