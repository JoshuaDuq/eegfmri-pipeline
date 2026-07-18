"""Publication rendering for band-resolved Study 1 time-frequency maps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.band_time_frequency import (
    BAND_CONFIG_KEY,
    FIGURE_CONFIG_KEY,
    BandTfrSummary,
)
from studies.pain_study.study1.figures.validity_style import (
    figure_size_inches,
    publication_style,
)


def participant_band_color_limit(
    summary: BandTfrSummary,
    band: str,
    *,
    percentile: float,
) -> float:
    """Return a robust scale shared by participant maps within one band."""

    percentile_value = float(percentile)
    if not 90.0 <= percentile_value <= 100.0:
        raise ValueError("Study 1 band TFR color percentile must be between 90 and 100.")
    values = summary.subject_maps.loc[
        summary.subject_maps["band"].eq(band),
        "temperature_slope_db_per_c",
    ].to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Study 1 band TFR has no finite participant values for {band}.")
    limit = float(np.percentile(np.abs(values), percentile_value))
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError(f"Study 1 band TFR requires non-zero display values for {band}.")
    return limit


def cohort_band_color_limit(
    summary: BandTfrSummary,
    band: str,
    *,
    percentile: float,
) -> float:
    """Return a cohort-specific robust scale for one band."""

    percentile_value = float(percentile)
    if not 90.0 <= percentile_value <= 100.0:
        raise ValueError("Study 1 band TFR color percentile must be between 90 and 100.")
    values = summary.cohort_maps.loc[
        summary.cohort_maps["band"].eq(band),
        "mean_temperature_slope_db_per_c",
    ].to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Study 1 band TFR has no finite cohort values for {band}.")
    limit = float(np.percentile(np.abs(values), percentile_value))
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError(f"Study 1 band TFR requires non-zero cohort values for {band}.")
    return limit


def build_participant_band_tfr_figure(
    summary: BandTfrSummary,
    *,
    band: str,
    subject_id: str,
    color_limit: float,
    config: Any,
) -> Figure:
    """Render one participant's nuisance-adjusted temperature-slope TFR."""

    rows = summary.subject_maps.loc[
        summary.subject_maps["band"].eq(band) & summary.subject_maps["subject_id"].eq(subject_id)
    ]
    if rows.empty:
        raise ValueError(f"Study 1 band TFR has no map for {subject_id}/{band}.")
    audit = summary.source_audit.loc[summary.source_audit["subject_id"].eq(subject_id)]
    if len(audit) != 1:
        raise ValueError(f"Study 1 band TFR source audit is invalid for {subject_id}.")
    record = audit.iloc[0]
    metadata = (
        f"{int(record['n_model_trials'])}/{int(record['n_clean_trials'])} modelled/clean trials · "
        f"{int(record['n_excluded_missing_metadata'])} excluded for missing metadata · "
        f"{int(record['n_eeg_channels'])} EEG channels · "
        f"source modified {_display_timestamp(record['modified_time_utc'])}"
    )
    return _build_band_tfr_figure(
        rows,
        value_column="temperature_slope_db_per_c",
        band=band,
        title_suffix=subject_id,
        metadata=metadata,
        scale_label="Participant-family",
        color_limit=color_limit,
        config=config,
    )


def build_cohort_band_tfr_figure(
    summary: BandTfrSummary,
    *,
    band: str,
    color_limit: float,
    config: Any,
) -> Figure:
    """Render the equal-weight mean of participant temperature-slope TFRs."""

    rows = summary.cohort_maps.loc[summary.cohort_maps["band"].eq(band)]
    if rows.empty:
        raise ValueError(f"Study 1 band TFR has no cohort map for {band}.")
    trials = summary.source_audit["n_model_trials"].to_numpy(dtype=int)
    excluded = summary.source_audit["n_excluded_missing_metadata"].to_numpy(dtype=int)
    channels = summary.source_audit["n_eeg_channels"].to_numpy(dtype=int)
    metadata = (
        f"Equal-weight participant slopes · n={len(summary.subject_ids)} · "
        f"modelled trials: range {trials.min()}–{trials.max()} · "
        f"missing-metadata exclusions: total {excluded.sum()} · "
        f"EEG channels/participant: range {channels.min()}–{channels.max()}"
    )
    return _build_band_tfr_figure(
        rows,
        value_column="mean_temperature_slope_db_per_c",
        band=band,
        title_suffix="cohort",
        metadata=metadata,
        scale_label="Cohort-specific",
        color_limit=color_limit,
        config=config,
    )


def _build_band_tfr_figure(
    rows: pd.DataFrame,
    *,
    value_column: str,
    band: str,
    title_suffix: str,
    metadata: str,
    scale_label: str,
    color_limit: float,
    config: Any,
) -> Figure:
    figure_config = require_config_value(config, FIGURE_CONFIG_KEY)
    if not isinstance(figure_config, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY} must be a mapping.")
    dimensions = figure_config.get("dimensions_mm")
    if not isinstance(dimensions, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY}.dimensions_mm must be a mapping.")
    display_window = _window(figure_config.get("display_window_s"))
    percentile = float(figure_config.get("color_percentile"))
    n_cycles = float(figure_config.get("n_cycles"))
    time_step_s = float(figure_config.get("time_step_s"))
    if not np.isfinite((n_cycles, time_step_s)).all() or n_cycles <= 0.0 or time_step_s <= 0.0:
        raise ValueError("Study 1 temperature TFR cycles and time step must be positive.")
    band_spec = _band_spec(config, band)
    phases = _phase_specs(config, display_window)
    limit = float(color_limit)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("Study 1 band TFR color limit must be positive and finite.")
    matrix = _map_matrix(rows, value_column=value_column)

    with publication_style(config):
        figure = plt.figure(figsize=figure_size_inches(dimensions))
        figure.set_size_inches(*figure_size_inches(dimensions), forward=False)
        axis = figure.add_axes((0.10, 0.18, 0.76, 0.53), label="band-tfr")
        color_axis = figure.add_axes((0.89, 0.18, 0.018, 0.53), label="colorbar")
        image = axis.pcolormesh(
            matrix.columns.to_numpy(dtype=float),
            matrix.index.to_numpy(dtype=float),
            matrix.to_numpy(dtype=float),
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            shading="auto",
            rasterized=True,
        )
        colorbar = figure.colorbar(image, cax=color_axis, extend="both")
        colorbar.set_label("Temperature slope (dB/°C)")
        _draw_protocol(axis, phases)
        axis.set_xlim(display_window)
        axis.set_ylim(
            float(matrix.index.min()),
            float(matrix.index.max()),
        )
        axis.set_xlabel("Time from stimulus onset (s)")
        axis.set_ylabel("Frequency (Hz)")
        axis.spines[["top", "right"]].set_visible(False)

        figure.text(
            0.10,
            0.955,
            f"{band_spec['label']}-band EEG temperature modulation · {title_suffix}",
            ha="left",
            va="top",
            fontsize=7.0,
            fontweight="bold",
        )
        figure.text(
            0.10,
            0.905,
            "Adjusted OLS slope (dB/°C) · run + thermode surface + within-run trial order",
            ha="left",
            va="top",
            fontsize=6.0,
        )
        figure.text(
            0.10,
            0.855,
            f"All EEG channels retained through normalization · {n_cycles:g}-cycle Hanning "
            f"mtmconvol · {time_step_s:g} s step",
            ha="left",
            va="top",
            fontsize=6.0,
        )
        figure.text(
            0.10,
            0.81,
            metadata,
            ha="left",
            va="top",
            fontsize=6.0,
        )
        figure.text(
            0.91,
            0.035,
            f"{scale_label} {band_spec['label'].lower()} scale: ±{limit:.3f} dB/°C "
            f"({percentile:g}th percentile); exact values in audits.",
            ha="right",
            va="bottom",
            fontsize=5.5,
            color="#555555",
        )
    return figure


def _draw_protocol(axis, phases: Sequence[Mapping[str, object]]) -> None:
    for phase in phases:
        start = float(phase["start_s"])
        end = float(phase["end_s"])
        axis.text(
            (start + end) / 2.0,
            1.035,
            str(phase["label"]),
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=5.2,
            color="#555555",
            clip_on=False,
        )
    for phase in phases[1:]:
        boundary = float(phase["start_s"])
        axis.axvline(
            boundary,
            color="#333333" if np.isclose(boundary, 0.0) else "#888888",
            linewidth=0.55 if np.isclose(boundary, 0.0) else 0.4,
            linestyle="-" if np.isclose(boundary, 0.0) else (0, (2, 2)),
            zorder=3,
        )


def _map_matrix(rows: pd.DataFrame, *, value_column: str) -> pd.DataFrame:
    if rows.duplicated(["frequency_hz", "time_s"]).any():
        raise ValueError("Study 1 band TFR figure contains duplicate map coordinates.")
    matrix = (
        rows.pivot(
            index="frequency_hz",
            columns="time_s",
            values=value_column,
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    if matrix.shape[0] < 2 or matrix.shape[1] < 2 or matrix.isna().any().any():
        raise ValueError("Study 1 band TFR figure requires a complete two-dimensional map.")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("Study 1 band TFR figure values must be finite.")
    return matrix


def _display_timestamp(value: object) -> str:
    try:
        timestamp = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError("Study 1 band TFR source timestamp must use ISO 8601.") from exc
    if timestamp.tzinfo is None:
        raise ValueError("Study 1 band TFR source timestamp must include a timezone.")
    return timestamp.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _band_spec(config: Any, band: str) -> Mapping[str, object]:
    specs = require_config_value(config, BAND_CONFIG_KEY)
    if not isinstance(specs, list):
        raise ValueError(f"{BAND_CONFIG_KEY} must be a list.")
    matches = [spec for spec in specs if isinstance(spec, Mapping) and spec.get("name") == band]
    if len(matches) != 1:
        raise ValueError(f"Study 1 band TFR requires one configured specification for {band}.")
    return matches[0]


def _phase_specs(
    config: Any,
    display_window: tuple[float, float],
) -> tuple[Mapping[str, object], ...]:
    phases = require_config_value(
        config,
        "study1.figures.band_power_epoch_evolution.phases",
    )
    if not isinstance(phases, list) or not all(isinstance(phase, Mapping) for phase in phases):
        raise ValueError("Study 1 band TFR phases must be a list of mappings.")
    boundaries = [float(phases[0]["start_s"])]
    for phase in phases:
        start = float(phase["start_s"])
        end = float(phase["end_s"])
        if not np.isclose(start, boundaries[-1]) or end <= start:
            raise ValueError("Study 1 band TFR phases must be contiguous and increasing.")
        boundaries.append(end)
    if not np.allclose((boundaries[0], boundaries[-1]), display_window):
        raise ValueError("Study 1 band TFR phases must span the display window.")
    return tuple(phases)


def _window(value: object) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Study 1 band TFR display window requires two values.")
    start, end = (float(item) for item in value)
    if not np.isfinite((start, end)).all() or start >= end:
        raise ValueError("Study 1 band TFR display window must be finite and increasing.")
    return start, end


__all__ = [
    "cohort_band_color_limit",
    "build_cohort_band_tfr_figure",
    "build_participant_band_tfr_figure",
    "participant_band_color_limit",
]
