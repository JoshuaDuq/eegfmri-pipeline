"""Publication rendering for Study 1 time-resolved band power."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.band_power_epoch_evolution import (
    FIGURE_CONFIG_KEY,
    BandPowerEpochSummary,
)
from studies.pain_study.study1.figures.validity_style import (
    figure_size_inches,
    publication_style,
)

PARTICIPANT_COLOR = "#6E6E6E"
REFERENCE_COLOR = "#555555"
LABEL_BACKGROUND = {"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.6}


def build_band_power_epoch_figure(
    summary: BandPowerEpochSummary,
    config: Any,
) -> Figure:
    """Render five retained-cohort band-power trajectories."""

    figure_config = require_config_value(config, FIGURE_CONFIG_KEY)
    if not isinstance(figure_config, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY} must be a mapping.")
    band_specs = _band_specs(figure_config)
    display_window = _window(figure_config.get("display_window_s"))
    phases = _phase_specs(figure_config, display_window=display_window)
    _validate_summary(summary, band_specs, display_window)
    dimensions = figure_config.get("dimensions_mm")
    if not isinstance(dimensions, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY}.dimensions_mm must be a mapping.")

    with publication_style(config):
        figure = plt.figure(figsize=figure_size_inches(dimensions))
        figure.set_size_inches(*figure_size_inches(dimensions), forward=False)
        grid = figure.add_gridspec(
            6,
            1,
            height_ratios=(0.34, 1.0, 1.0, 1.0, 1.0, 1.0),
            left=0.11,
            right=0.985,
            bottom=0.105,
            top=0.79,
            hspace=0.12,
        )
        protocol_axis = figure.add_subplot(grid[0, 0], label="protocol")
        _draw_protocol_bar(
            protocol_axis,
            phases=phases,
            display_window=display_window,
        )
        axes = []
        for index in range(len(band_specs)):
            shared_axis = axes[0] if axes else None
            axes.append(
                figure.add_subplot(
                    grid[index + 1, 0],
                    sharex=protocol_axis,
                    sharey=shared_axis,
                )
            )
        y_limit = _shared_y_limit(summary)
        for index, (axis, band_spec) in enumerate(zip(axes, band_specs, strict=True)):
            _draw_band_panel(
                axis,
                summary,
                band_spec=band_spec,
                phases=phases,
                display_window=display_window,
                y_limit=y_limit,
                show_x_label=index == len(axes) - 1,
            )

        figure.legend(
            handles=_legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.865),
            ncol=3,
            frameon=False,
            handlelength=1.7,
            handletextpad=0.45,
            columnspacing=1.25,
        )
        figure.text(
            0.11,
            0.965,
            "Global EEG band-power change during thermal stimulation",
            ha="left",
            va="top",
            fontsize=7.0,
            fontweight="bold",
        )
        figure.text(
            0.11,
            0.925,
            _sample_metadata(summary),
            ha="left",
            va="top",
            fontsize=6.0,
        )
        figure.text(
            0.025,
            0.43,
            "Change from −5.0 to −0.01 s baseline (dB)",
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.0,
        )
        figure.text(
            0.985,
            0.018,
            "Final 0.5 s omitted to limit right-edge Morlet convolution effects.",
            ha="right",
            va="bottom",
            fontsize=5.5,
            color=REFERENCE_COLOR,
        )
    return figure


def _draw_protocol_bar(
    axis,
    *,
    phases: Sequence[Mapping[str, object]],
    display_window: tuple[float, float],
) -> None:
    axis.set_xlim(display_window)
    axis.set_ylim(0.0, 1.0)
    for phase in phases:
        start = float(phase["start_s"])
        end = float(phase["end_s"])
        axis.axvspan(
            start,
            end,
            ymin=0.12,
            ymax=0.72,
            facecolor=str(phase["color"]),
            edgecolor="white",
            linewidth=0.6,
        )
        axis.text(
            (start + end) / 2.0,
            0.42,
            str(phase["label"]),
            ha="center",
            va="center",
            fontsize=5.5,
        )
    axis.set_axis_off()


def _draw_band_panel(
    axis,
    summary: BandPowerEpochSummary,
    *,
    band_spec: Mapping[str, object],
    phases: Sequence[Mapping[str, object]],
    display_window: tuple[float, float],
    y_limit: float,
    show_x_label: bool,
) -> None:
    band = str(band_spec["name"])
    color = str(band_spec["color"])
    participant = summary.subject_timecourses.loc[summary.subject_timecourses["band"].eq(band)]
    cohort = summary.cohort_timecourses.loc[
        summary.cohort_timecourses["band"].eq(band)
    ].sort_values("time_s", kind="stable")

    for _subject_id, subject_rows in participant.groupby("subject_id", sort=True):
        subject_rows = subject_rows.sort_values("time_s", kind="stable")
        axis.plot(
            subject_rows["time_s"],
            subject_rows["power_db"],
            color=PARTICIPANT_COLOR,
            linewidth=0.4,
            alpha=0.22,
            zorder=1,
        )
    axis.fill_between(
        cohort["time_s"].to_numpy(dtype=float),
        cohort["ci_low"].to_numpy(dtype=float),
        cohort["ci_high"].to_numpy(dtype=float),
        color=color,
        alpha=0.18,
        linewidth=0.0,
        zorder=2,
    )
    axis.plot(
        cohort["time_s"],
        cohort["mean_power_db"],
        color=color,
        linewidth=1.25,
        zorder=3,
    )
    axis.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.45, linestyle=(0, (3, 2)), zorder=0)
    for boundary in _phase_boundaries(phases):
        is_onset = np.isclose(boundary, 0.0)
        axis.axvline(
            boundary,
            color=REFERENCE_COLOR if is_onset else "#B7B7B7",
            linewidth=0.55 if is_onset else 0.4,
            linestyle="-" if is_onset else (0, (2, 2)),
            zorder=0,
        )
    axis.set_xlim(display_window)
    axis.set_ylim(-y_limit, y_limit)
    bounds = band_spec["frequency_hz"]
    axis.set_label(f"band:{band}")
    axis.text(
        0.008,
        0.91,
        str(band_spec["label"]),
        color=color,
        fontweight="bold",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
        bbox=LABEL_BACKGROUND,
    )
    axis.text(
        0.13,
        0.91,
        f"{float(bounds[0]):g}–{float(bounds[1]):g} Hz",
        color=REFERENCE_COLOR,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        bbox=LABEL_BACKGROUND,
    )
    axis.set_xlabel("Time from stimulus onset (s)" if show_x_label else "")
    axis.tick_params(axis="x", labelbottom=show_x_label)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)


def _shared_y_limit(summary: BandPowerEpochSummary) -> float:
    values = np.concatenate(
        [
            summary.subject_timecourses["power_db"].to_numpy(dtype=float),
            summary.cohort_timecourses["ci_low"].to_numpy(dtype=float),
            summary.cohort_timecourses["ci_high"].to_numpy(dtype=float),
        ]
    )
    if not np.isfinite(values).all():
        raise ValueError("Study 1 band-power figure contains non-finite display values.")
    maximum = float(np.max(np.abs(values)))
    if maximum <= 0.0:
        raise ValueError("Study 1 band-power figure requires non-zero trajectories.")
    padded = maximum * 1.05
    scale = 10.0 ** np.floor(np.log10(padded))
    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        candidate = multiplier * scale
        if candidate >= padded:
            return float(candidate)
    raise RuntimeError("Study 1 band-power display limit selection failed.")


def _legend_handles() -> list[object]:
    return [
        Line2D(
            [],
            [],
            color=PARTICIPANT_COLOR,
            linewidth=0.6,
            alpha=0.55,
            label="Participant retained-trial mean",
        ),
        Line2D(
            [],
            [],
            color="#222222",
            linewidth=1.25,
            label="Equally weighted cohort mean",
        ),
        Patch(
            facecolor="#777777",
            alpha=0.18,
            edgecolor="none",
            label="Pointwise 95% participant-bootstrap CI",
        ),
    ]


def _sample_metadata(summary: BandPowerEpochSummary) -> str:
    metadata = summary.subject_timecourses.loc[
        :, ["subject_id", "n_retained_trials", "n_channels"]
    ].drop_duplicates()
    if metadata["subject_id"].duplicated().any():
        raise ValueError(
            "Study 1 band-power participant trial and channel counts must be constant."
        )
    trials = metadata["n_retained_trials"].to_numpy(dtype=int)
    channels = metadata["n_channels"].to_numpy(dtype=int)
    if np.any(trials < 1) or np.any(channels < 1):
        raise ValueError("Study 1 band-power sample metadata must contain positive counts.")
    unique_channels = np.unique(channels)
    if len(unique_channels) != 1:
        raise ValueError("Study 1 band-power figure requires a common EEG channel scope.")
    median_trials = float(np.median(trials))
    return (
        f"Retained-trial means · n={len(summary.subject_ids)} participants · "
        f"trials/participant: median {median_trials:g}, range {trials.min()}–{trials.max()} · "
        f"{unique_channels[0]} EEG channels"
    )


def _phase_boundaries(phases: Sequence[Mapping[str, object]]) -> tuple[float, ...]:
    return tuple(float(phase["end_s"]) for phase in phases[:-1])


def _validate_summary(
    summary: BandPowerEpochSummary,
    band_specs: Sequence[Mapping[str, object]],
    display_window: tuple[float, float],
) -> None:
    configured_bands = tuple(str(spec["name"]) for spec in band_specs)
    if summary.bands != configured_bands:
        raise ValueError("Study 1 band-power figure bands must match the configured order exactly.")
    if len(summary.subject_ids) < 2:
        raise ValueError("Study 1 band-power figure requires at least two participants.")
    if len(summary.times) < 2 or not np.all(np.diff(summary.times) > 0.0):
        raise ValueError("Study 1 band-power figure requires an increasing time axis.")
    if summary.times[0] < display_window[0] or summary.times[-1] > display_window[1]:
        raise ValueError("Study 1 band-power summary exceeds the configured display window.")
    expected_subject_rows = len(summary.subject_ids) * len(summary.bands) * len(summary.times)
    expected_cohort_rows = len(summary.bands) * len(summary.times)
    if len(summary.subject_timecourses) != expected_subject_rows:
        raise ValueError("Study 1 band-power participant audit is incomplete.")
    if len(summary.cohort_timecourses) != expected_cohort_rows:
        raise ValueError("Study 1 band-power cohort audit is incomplete.")


def _band_specs(figure_config: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw_specs = figure_config.get("bands")
    if not isinstance(raw_specs, list) or len(raw_specs) != 5:
        raise ValueError("Study 1 band-power figure requires exactly five band specifications.")
    if not all(isinstance(spec, Mapping) for spec in raw_specs):
        raise ValueError("Study 1 band-power figure band specifications must be mappings.")
    for spec in raw_specs:
        if not str(spec.get("label", "")).strip() or not str(spec.get("color", "")).strip():
            raise ValueError("Study 1 band-power bands require labels and colors.")
    return tuple(raw_specs)


def _phase_specs(
    figure_config: Mapping[str, object],
    *,
    display_window: tuple[float, float],
) -> tuple[Mapping[str, object], ...]:
    raw_phases = figure_config.get("phases")
    if not isinstance(raw_phases, list) or len(raw_phases) != 5:
        raise ValueError("Study 1 band-power figure requires exactly five epoch phases.")
    if not all(isinstance(phase, Mapping) for phase in raw_phases):
        raise ValueError("Study 1 band-power phase specifications must be mappings.")
    boundaries = [float(raw_phases[0].get("start_s"))]
    for phase in raw_phases:
        if not str(phase.get("label", "")).strip() or not str(phase.get("color", "")).strip():
            raise ValueError("Study 1 band-power phases require labels and colors.")
        start = float(phase.get("start_s"))
        end = float(phase.get("end_s"))
        if not np.isclose(start, boundaries[-1]) or end <= start:
            raise ValueError("Study 1 band-power phases must be contiguous and increasing.")
        boundaries.append(end)
    if not np.allclose((boundaries[0], boundaries[-1]), display_window):
        raise ValueError("Study 1 band-power phases must span the complete display window.")
    return tuple(raw_phases)


def _window(value: object) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Study 1 band-power display window requires two values.")
    start, end = float(value[0]), float(value[1])
    if not np.isfinite([start, end]).all() or start >= end:
        raise ValueError("Study 1 band-power display window must satisfy start < end.")
    return start, end


__all__ = ["build_band_power_epoch_figure"]
