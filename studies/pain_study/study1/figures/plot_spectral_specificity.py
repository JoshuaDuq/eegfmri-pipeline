"""Write the standalone Study 1 spectral-specificity figure."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eeg_pipeline.infra.tsv import write_parquet, write_tsv
from eeg_pipeline.utils.config.loader import load_config, require_config_value
from studies.pain_study.study1.cohort import study1_output_root
from studies.pain_study.study1.config.loader import apply_study1_config_defaults
from studies.pain_study.study1.figures.spectral_specificity import (
    load_spectral_specificity_summary,
)
from studies.pain_study.study1.figures.spectral_specificity_plot import (
    build_spectral_specificity_figure,
)
from studies.pain_study.study1.figures.validity_style import (
    save_publication_svg,
    validity_output_dir,
)

OUTPUT_FILENAME = "spectral_specificity.svg"
SUBJECT_AUDIT_FILENAME = "spectral_specificity_by_subject.tsv"
SUMMARY_AUDIT_FILENAME = "spectral_specificity_summary.tsv"


@dataclass(frozen=True)
class SpectralSpecificityFigurePaths:
    """Figure and audit paths written by the spectral-specificity stage."""

    svg: Path
    subject_tsv: Path
    subject_parquet: Path
    summary_tsv: Path
    summary_parquet: Path


def write_spectral_specificity(
    *,
    report_path: Path,
    config: Any,
    output_path: Path | None = None,
) -> SpectralSpecificityFigurePaths:
    """Validate the report and write one SVG plus reproducibility audits."""

    summary = load_spectral_specificity_summary(report_path, config)
    resolved_output = output_path or validity_output_dir(config) / OUTPUT_FILENAME
    figure = build_spectral_specificity_figure(summary, config)
    save_publication_svg(
        figure,
        resolved_output,
        config,
        dimensions_mm=require_config_value(
            config,
            "study1.figures.spectral_specificity.dimensions_mm",
        ),
    )

    subject_tsv = resolved_output.with_name(SUBJECT_AUDIT_FILENAME)
    subject_parquet = subject_tsv.with_suffix(".parquet")
    summary_tsv = resolved_output.with_name(SUMMARY_AUDIT_FILENAME)
    summary_parquet = summary_tsv.with_suffix(".parquet")
    write_tsv(summary.participant_effects, subject_tsv)
    write_parquet(summary.participant_effects, subject_parquet)
    write_tsv(summary.cohort_effects, summary_tsv)
    write_parquet(summary.cohort_effects, summary_parquet)
    return SpectralSpecificityFigurePaths(
        svg=resolved_output,
        subject_tsv=subject_tsv,
        subject_parquet=subject_parquet,
        summary_tsv=summary_tsv,
        summary_parquet=summary_parquet,
    )


def main(argv: Sequence[str] | None = None) -> SpectralSpecificityFigurePaths:
    parser = argparse.ArgumentParser(
        description="Write the Study 1 spectral-specificity SVG and audit tables."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study1-config", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    apply_study1_config_defaults(config, arguments.study1_config)
    report_path = arguments.report or study1_output_root(config) / "reports" / "study1_report.tsv"
    output_paths = write_spectral_specificity(
        report_path=report_path,
        config=config,
        output_path=arguments.output,
    )
    print(output_paths.svg)
    return output_paths


if __name__ == "__main__":
    main()


__all__ = [
    "SpectralSpecificityFigurePaths",
    "main",
    "write_spectral_specificity",
]
