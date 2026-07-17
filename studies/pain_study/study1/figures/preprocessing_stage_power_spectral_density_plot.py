"""Publication rendering for Study 1 preprocessing-stage power spectra."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from studies.pain_study.study1.figures.cohort_power_spectral_density import (
    CohortPsdSummary,
)
from studies.pain_study.study1.figures.cohort_power_spectral_density_plot import (
    build_cohort_psd_figure,
)
from studies.pain_study.study1.figures.preprocessing_stage_power_spectral_density import (
    PreprocessingStagePsdSpecification,
)


def build_preprocessing_stage_psd_figure(
    summary: CohortPsdSummary,
    specification: PreprocessingStagePsdSpecification,
    config: Any,
) -> Figure:
    """Render a cohort PSD with explicit preprocessing-checkpoint provenance."""
    _validate_stage_summary(summary, specification)
    figure = build_cohort_psd_figure(summary, config)
    headers = {text.get_gid(): text for text in figure.texts if text.get_gid()}
    required_headers = {"cohort-psd-title", "cohort-psd-sample"}
    missing_headers = required_headers.difference(headers)
    if missing_headers:
        missing = ", ".join(sorted(missing_headers))
        raise ValueError(f"Cohort PSD figure is missing required headers: {missing}.")

    stage = specification.stage
    headers["cohort-psd-title"].set_text(
        f"Continuous EEG power spectrum · {stage.label} checkpoint"
    )
    headers["cohort-psd-sample"].set_text(_checkpoint_metadata(summary, specification))
    return figure


def _validate_stage_summary(
    summary: CohortPsdSummary,
    specification: PreprocessingStagePsdSpecification,
) -> None:
    expected_stage = specification.stage.identifier
    for frame_name, frame in (
        ("participant spectra", summary.participant_spectra),
        ("cohort spectrum", summary.cohort_spectrum),
        ("run audit", summary.run_audit),
    ):
        if "stage" not in frame:
            raise ValueError(f"Preprocessing PSD {frame_name} is missing stage metadata.")
        stages = set(frame["stage"].dropna().astype(str))
        if stages != {expected_stage} or frame["stage"].isna().any():
            raise ValueError(
                f"Preprocessing PSD {frame_name} stage metadata must equal " f"{expected_stage!r}."
            )

    _validate_constant_setting(
        summary.run_audit,
        column="sampling_frequency_hz",
        expected=specification.stage.sampling_frequency_hz,
        label="sampling frequency",
    )
    _validate_constant_setting(
        summary.run_audit,
        column="segment_duration_s",
        expected=specification.segment_duration_s,
        label="segment duration",
    )
    _validate_constant_setting(
        summary.run_audit,
        column="overlap_fraction",
        expected=specification.overlap_fraction,
        label="overlap fraction",
    )


def _validate_constant_setting(
    run_audit: pd.DataFrame,
    *,
    column: str,
    expected: float,
    label: str,
) -> None:
    if column not in run_audit:
        raise ValueError(f"Preprocessing PSD run audit is missing {label} metadata.")
    values = pd.to_numeric(run_audit[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.allclose(values, expected):
        raise ValueError(
            f"Preprocessing PSD {label} must equal the configured value " f"{expected:g}."
        )


def _checkpoint_metadata(
    summary: CohortPsdSummary,
    specification: PreprocessingStagePsdSpecification,
) -> str:
    stage = specification.stage
    overlap_percent = 100.0 * specification.overlap_fraction
    return (
        f"{_checkpoint_label(stage.identifier)} checkpoint · "
        f"{stage.sampling_frequency_hz:g} Hz source · "
        f"Welch {specification.segment_duration_s:g} s, "
        f"{overlap_percent:g}% overlap · "
        f"n={_count_label(summary.n_subjects, 'participant')} · "
        f"{_count_label(summary.n_runs, 'run')}"
    )


def _checkpoint_label(stage_identifier: str) -> str:
    if stage_identifier == "mne":
        return stage_identifier.upper()
    return stage_identifier.capitalize()


def _count_label(count: int, noun: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {noun}{suffix}"


__all__ = ["build_preprocessing_stage_psd_figure"]
