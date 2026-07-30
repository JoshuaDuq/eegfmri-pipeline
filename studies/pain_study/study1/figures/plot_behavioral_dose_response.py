"""Write the standalone Study 1 behavioral dose-response SVG."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eeg_pipeline.utils.config.loader import load_config
from studies.pain_study.study1.config.loader import apply_study1_config_defaults
from studies.pain_study.study1.figures.dose_response import (
    DoseResponseSpecification,
    HorizontalReference,
    build_dose_response_figure,
)
from studies.pain_study.study1.figures.validity_data import (
    ValidityTrialData,
    build_dose_response_summary,
    load_validity_trial_data,
    validate_behavioral_ratings,
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
    validate_behavioral_ratings(trial_data.enriched_targets)
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


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Write the Study 1 behavioral dose-response SVG.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study1-config", type=Path)
    parser.add_argument("--task", required=True)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    apply_study1_config_defaults(config, arguments.study1_config)
    trial_data = load_validity_trial_data(task=arguments.task, config=config)
    output_path = write_behavioral_dose_response(
        trial_data=trial_data,
        config=config,
    )
    print(output_path)
    return output_path


if __name__ == "__main__":
    main()


__all__ = ["main", "write_behavioral_dose_response"]
