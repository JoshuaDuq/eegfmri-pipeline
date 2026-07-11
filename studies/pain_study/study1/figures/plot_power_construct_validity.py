"""Write the standalone Study 1 EEG power construct-validity figure."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eeg_pipeline.infra.tsv import write_parquet, write_tsv
from eeg_pipeline.utils.config.loader import load_config, require_config_value
from studies.pain_study.study1.config.loader import apply_study1_config_defaults
from studies.pain_study.study1.figures.power_construct_validity import (
    load_power_construct_validity_summary,
)
from studies.pain_study.study1.figures.power_construct_validity_plot import (
    build_power_construct_validity_figure,
)
from studies.pain_study.study1.figures.validity_style import (
    save_publication_svg,
    validity_output_dir,
)

OUTPUT_FILENAME = "power_construct_validity.svg"


@dataclass(frozen=True)
class PowerConstructValidityFigurePaths:
    """SVG and seven paired reproducibility-audit paths."""

    svg: Path
    trials_tsv: Path
    trials_parquet: Path
    temperature_subject_tsv: Path
    temperature_subject_parquet: Path
    temperature_summary_tsv: Path
    temperature_summary_parquet: Path
    rating_subject_tsv: Path
    rating_subject_parquet: Path
    rating_summary_tsv: Path
    rating_summary_parquet: Path
    sensitivity_subject_tsv: Path
    sensitivity_subject_parquet: Path
    sensitivity_summary_tsv: Path
    sensitivity_summary_parquet: Path


def write_power_construct_validity(
    *,
    task: str,
    config: Any,
    output_path: Path | None = None,
) -> PowerConstructValidityFigurePaths:
    """Validate inputs, render one SVG, and write complete analysis audits."""

    summary = load_power_construct_validity_summary(task=task, config=config)
    resolved_output = output_path or validity_output_dir(config) / OUTPUT_FILENAME
    figure = build_power_construct_validity_figure(summary, config)
    save_publication_svg(
        figure,
        resolved_output,
        config,
        dimensions_mm=require_config_value(
            config,
            "study1.figures.power_construct_validity.dimensions_mm",
        ),
    )

    tables = {
        "power_construct_validity_trials": summary.trials,
        "power_temperature_by_subject": summary.temperature_by_subject,
        "power_temperature_summary": summary.temperature_summary,
        "power_rating_by_subject": summary.rating_by_subject,
        "power_rating_summary": summary.rating_summary,
        "power_fp1_fp2_sensitivity_by_subject": summary.sensitivity_by_subject,
        "power_fp1_fp2_sensitivity_summary": summary.sensitivity_summary,
    }
    paths: dict[str, tuple[Path, Path]] = {}
    for stem, frame in tables.items():
        tsv_path = resolved_output.with_name(f"{stem}.tsv")
        parquet_path = tsv_path.with_suffix(".parquet")
        write_tsv(frame, tsv_path)
        write_parquet(frame, parquet_path)
        paths[stem] = (tsv_path, parquet_path)
    return PowerConstructValidityFigurePaths(
        svg=resolved_output,
        trials_tsv=paths["power_construct_validity_trials"][0],
        trials_parquet=paths["power_construct_validity_trials"][1],
        temperature_subject_tsv=paths["power_temperature_by_subject"][0],
        temperature_subject_parquet=paths["power_temperature_by_subject"][1],
        temperature_summary_tsv=paths["power_temperature_summary"][0],
        temperature_summary_parquet=paths["power_temperature_summary"][1],
        rating_subject_tsv=paths["power_rating_by_subject"][0],
        rating_subject_parquet=paths["power_rating_by_subject"][1],
        rating_summary_tsv=paths["power_rating_summary"][0],
        rating_summary_parquet=paths["power_rating_summary"][1],
        sensitivity_subject_tsv=paths["power_fp1_fp2_sensitivity_by_subject"][0],
        sensitivity_subject_parquet=paths["power_fp1_fp2_sensitivity_by_subject"][1],
        sensitivity_summary_tsv=paths["power_fp1_fp2_sensitivity_summary"][0],
        sensitivity_summary_parquet=paths["power_fp1_fp2_sensitivity_summary"][1],
    )


def main(argv: Sequence[str] | None = None) -> PowerConstructValidityFigurePaths:
    parser = argparse.ArgumentParser(
        description="Write the Study 1 EEG power construct-validity SVG and audits."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study1-config", type=Path)
    parser.add_argument("--deriv-root", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    apply_study1_config_defaults(config, arguments.study1_config)
    if arguments.deriv_root is not None:
        config["paths.deriv_root"] = str(arguments.deriv_root.expanduser().resolve())
    output_paths = write_power_construct_validity(
        task=arguments.task,
        config=config,
        output_path=arguments.output,
    )
    print(output_paths.svg)
    return output_paths


if __name__ == "__main__":
    main()


__all__ = [
    "PowerConstructValidityFigurePaths",
    "main",
    "write_power_construct_validity",
]
