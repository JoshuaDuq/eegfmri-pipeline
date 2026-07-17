"""Write the standalone Study 1 band-power epoch-evolution figure."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eeg_pipeline.infra.tsv import write_parquet, write_tsv
from eeg_pipeline.utils.config.loader import load_config, require_config_value
from studies.pain_study.study1.config.loader import apply_study1_config_defaults
from studies.pain_study.study1.figures.band_power_epoch_evolution import (
    FIGURE_CONFIG_KEY,
    load_band_power_epoch_summary,
)
from studies.pain_study.study1.figures.band_power_epoch_evolution_plot import (
    build_band_power_epoch_figure,
)
from studies.pain_study.study1.figures.validity_style import (
    save_publication_svg,
    validity_output_dir,
)

OUTPUT_FILENAME = "band_power_epoch_evolution.svg"


@dataclass(frozen=True)
class BandPowerEpochFigurePaths:
    """Figure and paired participant/cohort audit paths."""

    svg: Path
    subject_tsv: Path
    subject_parquet: Path
    summary_tsv: Path
    summary_parquet: Path


def write_band_power_epoch_evolution(
    *,
    task: str,
    config: Any,
    output_path: Path | None = None,
) -> BandPowerEpochFigurePaths:
    """Compute, render, and audit the retained Study 1 band trajectories."""

    summary = load_band_power_epoch_summary(task=task, config=config)
    resolved_output = output_path or validity_output_dir(config) / OUTPUT_FILENAME
    figure = build_band_power_epoch_figure(summary, config)
    save_publication_svg(
        figure,
        resolved_output,
        config,
        dimensions_mm=require_config_value(
            config,
            f"{FIGURE_CONFIG_KEY}.dimensions_mm",
        ),
    )

    subject_tsv = resolved_output.with_name("band_power_epoch_evolution_by_subject.tsv")
    subject_parquet = subject_tsv.with_suffix(".parquet")
    summary_tsv = resolved_output.with_name("band_power_epoch_evolution_summary.tsv")
    summary_parquet = summary_tsv.with_suffix(".parquet")
    write_tsv(summary.subject_timecourses, subject_tsv)
    write_parquet(summary.subject_timecourses, subject_parquet)
    write_tsv(summary.cohort_timecourses, summary_tsv)
    write_parquet(summary.cohort_timecourses, summary_parquet)
    return BandPowerEpochFigurePaths(
        svg=resolved_output,
        subject_tsv=subject_tsv,
        subject_parquet=subject_parquet,
        summary_tsv=summary_tsv,
        summary_parquet=summary_parquet,
    )


def main(argv: Sequence[str] | None = None) -> BandPowerEpochFigurePaths:
    parser = argparse.ArgumentParser(
        description="Write the Study 1 band-power epoch-evolution SVG and audits."
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
    outputs = write_band_power_epoch_evolution(
        task=arguments.task,
        config=config,
        output_path=arguments.output,
    )
    print(outputs.svg)
    return outputs


if __name__ == "__main__":
    main()


__all__ = [
    "BandPowerEpochFigurePaths",
    "main",
    "write_band_power_epoch_evolution",
]
