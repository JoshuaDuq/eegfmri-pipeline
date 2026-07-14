"""Publication rendering for Study 1 preprocessing-stage power spectra."""

from __future__ import annotations

from typing import Any

from matplotlib.figure import Figure

from eeg_pipeline.utils.config.loader import require_config_value
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
    """Render the shared cohort PSD with an explicit checkpoint annotation."""
    figure = build_cohort_psd_figure(summary, config)
    annotation_size = float(
        require_config_value(config, "study1.figures.validity.font.annotation_pt")
    )
    stage = specification.stage
    figure.axes[0].text(
        0.99,
        0.98,
        f"{stage.label} · {stage.sampling_frequency_hz:g} Hz",
        ha="right",
        va="top",
        transform=figure.axes[0].transAxes,
        fontsize=annotation_size,
    )
    return figure


__all__ = ["build_preprocessing_stage_psd_figure"]
