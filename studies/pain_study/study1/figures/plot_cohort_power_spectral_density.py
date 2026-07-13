"""Write the standalone Study 1 cohort power spectral density artifact."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eeg_pipeline.infra.tsv import write_parquet, write_tsv
from eeg_pipeline.utils.config.loader import load_config, require_config_value
from eeg_pipeline.utils.config.roots import resolve_eeg_deriv_root
from studies.pain_study.study1.config.loader import apply_study1_config_defaults
from studies.pain_study.study1.figures.cohort_power_spectral_density import (
    CohortPsdSummary,
    build_cohort_psd_summary,
    cohort_psd_specification,
)
from studies.pain_study.study1.figures.cohort_power_spectral_density_plot import (
    build_cohort_psd_figure,
)
from studies.pain_study.study1.figures.continuous_spectrum import (
    discover_final_clean_runs,
    estimate_continuous_run_spectrum,
)
from studies.pain_study.study1.figures.spectral_statistics import (
    validity_bootstrap_specification,
)
from studies.pain_study.study1.figures.validity_style import (
    save_publication_svg,
    validity_output_dir,
)

OUTPUT_FILENAME = "cohort_power_spectral_density.svg"
RUN_AUDIT_FILENAME = "cohort_power_spectral_density_by_run.tsv"
PARTICIPANT_AUDIT_FILENAME = "cohort_power_spectral_density_by_subject.tsv"
SUMMARY_AUDIT_FILENAME = "cohort_power_spectral_density_summary.tsv"


@dataclass(frozen=True)
class CohortPsdFigurePaths:
    """Figure and audit paths written by the cohort-PSD stage."""

    svg: Path
    run_tsv: Path
    run_parquet: Path
    participant_tsv: Path
    participant_parquet: Path
    summary_tsv: Path
    summary_parquet: Path


def write_cohort_power_spectral_density(
    *,
    derivative_root: Path,
    task: str,
    config: Any,
    subjects: Sequence[str] = (),
    output_path: Path | None = None,
) -> CohortPsdFigurePaths:
    """Analyze final-clean EEG and write one SVG plus exact audit tables."""
    summary = _build_summary(
        derivative_root=Path(derivative_root),
        task=task,
        config=config,
        subjects=subjects,
    )
    resolved_output = output_path or validity_output_dir(config) / OUTPUT_FILENAME
    figure = build_cohort_psd_figure(summary, config)
    save_publication_svg(
        figure,
        resolved_output,
        config,
        dimensions_mm=require_config_value(
            config,
            "study1.figures.cohort_power_spectral_density.dimensions_mm",
        ),
    )

    run_tsv = resolved_output.with_name(RUN_AUDIT_FILENAME)
    run_parquet = run_tsv.with_suffix(".parquet")
    participant_tsv = resolved_output.with_name(PARTICIPANT_AUDIT_FILENAME)
    participant_parquet = participant_tsv.with_suffix(".parquet")
    summary_tsv = resolved_output.with_name(SUMMARY_AUDIT_FILENAME)
    summary_parquet = summary_tsv.with_suffix(".parquet")
    write_tsv(summary.run_audit, run_tsv)
    write_parquet(summary.run_audit, run_parquet)
    write_tsv(summary.participant_spectra, participant_tsv)
    write_parquet(summary.participant_spectra, participant_parquet)
    write_tsv(summary.cohort_spectrum, summary_tsv)
    write_parquet(summary.cohort_spectrum, summary_parquet)
    return CohortPsdFigurePaths(
        svg=resolved_output,
        run_tsv=run_tsv,
        run_parquet=run_parquet,
        participant_tsv=participant_tsv,
        participant_parquet=participant_parquet,
        summary_tsv=summary_tsv,
        summary_parquet=summary_parquet,
    )


def _build_summary(
    *,
    derivative_root: Path,
    task: str,
    config: Any,
    subjects: Sequence[str],
) -> CohortPsdSummary:
    specification = cohort_psd_specification(config)
    run_paths = discover_final_clean_runs(
        derivative_root,
        task=task,
        excluded_subjects=specification.excluded_subjects,
        requested_subjects=subjects,
    )
    run_spectra = tuple(
        estimate_continuous_run_spectrum(path, specification.spectrum) for path in run_paths
    )
    return build_cohort_psd_summary(
        run_spectra,
        specification,
        bootstrap=validity_bootstrap_specification(config),
    )


def main(argv: Sequence[str] | None = None) -> CohortPsdFigurePaths:
    parser = argparse.ArgumentParser(
        description="Write the Study 1 cohort PSD SVG and audit tables."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study1-config", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--derivative-root", type=Path)
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    apply_study1_config_defaults(config, arguments.study1_config)
    derivative_root = arguments.derivative_root or resolve_eeg_deriv_root(config)
    output_paths = write_cohort_power_spectral_density(
        derivative_root=derivative_root,
        task=arguments.task,
        config=config,
        subjects=tuple(arguments.subject),
        output_path=arguments.output,
    )
    print(output_paths.svg)
    return output_paths


if __name__ == "__main__":
    main()


__all__ = [
    "CohortPsdFigurePaths",
    "main",
    "write_cohort_power_spectral_density",
]
