"""Write the standalone Study 1 NPS behavioral-validity SVG."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eeg_pipeline.utils.config.loader import load_config
from studies.pain_study.study1.config.loader import apply_study1_config_defaults
from studies.pain_study.study1.figures.behavioral_validity import (
    NPS_SPECIFICATION,
    BehavioralValiditySummary,
    build_behavioral_validity_summary,
    require_behavioral_validity_target,
)
from studies.pain_study.study1.figures.coefficient_plot import (
    build_behavioral_validity_figure,
)
from studies.pain_study.study1.figures.validity_data import (
    load_validity_trial_data,
)
from studies.pain_study.study1.figures.validity_style import (
    save_validity_svg,
    validity_output_dir,
)

OUTPUT_FILENAME = "nps_behavioral_validity.svg"


def write_nps_behavioral_validity(
    *,
    summary: BehavioralValiditySummary,
    config: Any,
) -> Path:
    require_behavioral_validity_target(summary, NPS_SPECIFICATION.target)
    figure = build_behavioral_validity_figure(
        summary,
        color_config_key=NPS_SPECIFICATION.color_config_key,
        config=config,
    )
    return save_validity_svg(
        figure,
        validity_output_dir(config) / OUTPUT_FILENAME,
        config,
    )


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Write the Study 1 NPS behavioral-validity SVG."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study1-config", type=Path)
    parser.add_argument("--task", required=True)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    apply_study1_config_defaults(config, arguments.study1_config)
    trial_data = load_validity_trial_data(task=arguments.task, config=config)
    summary = build_behavioral_validity_summary(
        trial_data.enriched_targets,
        specification=NPS_SPECIFICATION,
        config=config,
    )
    output_path = write_nps_behavioral_validity(summary=summary, config=config)
    print(output_path)
    return output_path


if __name__ == "__main__":
    main()


__all__ = ["main", "write_nps_behavioral_validity"]
