"""Write the standalone Study 1 NPS dose-response SVG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from studies.pain_study.study1.figures.dose_response import (
    DoseResponseSpecification,
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

OUTPUT_FILENAME = "nps_dose_response.svg"
SPECIFICATION = DoseResponseSpecification(
    ylabel="NPS expression (a.u.)",
    color_config_key="nps",
)


def write_nps_dose_response(*, trial_data: ValidityTrialData, config: Any) -> Path:
    summary = build_dose_response_summary(
        trial_data.targets,
        outcome="NPS",
        config=config,
    )
    figure = build_dose_response_figure(summary, SPECIFICATION, config)
    return save_validity_svg(
        figure,
        validity_output_dir(config) / OUTPUT_FILENAME,
        config,
    )


__all__ = ["write_nps_dose_response"]
