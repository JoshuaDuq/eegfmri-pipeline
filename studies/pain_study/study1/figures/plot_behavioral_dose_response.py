"""Write the standalone Study 1 behavioral dose-response SVG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from studies.pain_study.study1.figures.dose_response import (
    DoseResponseSpecification,
    HorizontalReference,
    build_dose_response_figure,
)
from studies.pain_study.study1.figures.validity_data import (
    ValidityTrialData,
    build_dose_response_summary,
)
from studies.pain_study.study1.figures.validity_style import (
    save_validity_svg,
    validity_output_dir,
)

OUTPUT_FILENAME = "behavioral_dose_response.svg"
SPECIFICATION = DoseResponseSpecification(
    ylabel="Displayed rating (0–200)",
    color_config_key="behavioral",
    y_limits=(0.0, 200.0),
    reference=HorizontalReference(value=100.0, label="Pain threshold"),
)


def write_behavioral_dose_response(
    *,
    trial_data: ValidityTrialData,
    config: Any,
) -> Path:
    summary = build_dose_response_summary(
        trial_data.enriched_targets,
        outcome="vas_final_coded_rating",
        config=config,
    )
    figure = build_dose_response_figure(summary, SPECIFICATION, config)
    return save_validity_svg(
        figure,
        validity_output_dir(config) / OUTPUT_FILENAME,
        config,
    )


__all__ = ["write_behavioral_dose_response"]
