"""Write Study 1 preprocessing-stage cohort power spectral density artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eeg_pipeline.infra.tsv import write_parquet, write_tsv
from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.cohort_power_spectral_density import (
    CohortPsdSummary,
)
from studies.pain_study.study1.figures.preprocessing_psd_sources import EegRunSource
from studies.pain_study.study1.figures.preprocessing_stage_power_spectral_density import (
    PreprocessingStagePsdSpecification,
    build_preprocessing_stage_psd_summary,
)
from studies.pain_study.study1.figures.preprocessing_stage_power_spectral_density_plot import (
    build_preprocessing_stage_psd_figure,
)
from studies.pain_study.study1.figures.spectral_statistics import (
    validity_bootstrap_specification,
)
from studies.pain_study.study1.figures.validity_style import (
    save_publication_svg,
    validity_output_dir,
)


@dataclass(frozen=True)
class PreprocessingStagePsdPaths:
    """Figure and audit paths for one preprocessing-stage report."""

    svg: Path
    run_tsv: Path
    run_parquet: Path
    participant_tsv: Path
    participant_parquet: Path
    summary_tsv: Path
    summary_parquet: Path


def write_preprocessing_stage_psd(
    *,
    sources: tuple[EegRunSource, ...],
    specification: PreprocessingStagePsdSpecification,
    config: Any,
    output_dir: Path | None = None,
) -> PreprocessingStagePsdPaths:
    """Write one stage SVG and its exact audit-table family."""
    summary = _build_stage_summary(
        sources=sources,
        specification=specification,
        config=config,
    )
    directory = Path(output_dir) if output_dir is not None else validity_output_dir(config)
    paths = _artifact_paths(directory, specification.stage.identifier)
    figure = build_preprocessing_stage_psd_figure(summary, specification, config)
    save_publication_svg(
        figure,
        paths.svg,
        config,
        dimensions_mm=require_config_value(
            config,
            "study1.figures.cohort_power_spectral_density.dimensions_mm",
        ),
    )
    _write_summary_tables(summary, paths)
    return paths


def _build_stage_summary(
    *,
    sources: tuple[EegRunSource, ...],
    specification: PreprocessingStagePsdSpecification,
    config: Any,
) -> CohortPsdSummary:
    return build_preprocessing_stage_psd_summary(
        sources,
        specification,
        bootstrap=validity_bootstrap_specification(config),
    )


def _artifact_paths(directory: Path, stage_identifier: str) -> PreprocessingStagePsdPaths:
    stem = f"cohort_power_spectral_density_{stage_identifier}"
    return PreprocessingStagePsdPaths(
        svg=directory / f"{stem}.svg",
        run_tsv=directory / f"{stem}_by_run.tsv",
        run_parquet=directory / f"{stem}_by_run.parquet",
        participant_tsv=directory / f"{stem}_by_subject.tsv",
        participant_parquet=directory / f"{stem}_by_subject.parquet",
        summary_tsv=directory / f"{stem}_summary.tsv",
        summary_parquet=directory / f"{stem}_summary.parquet",
    )


def _write_summary_tables(
    summary: CohortPsdSummary,
    paths: PreprocessingStagePsdPaths,
) -> None:
    write_tsv(summary.run_audit, paths.run_tsv)
    write_parquet(summary.run_audit, paths.run_parquet)
    write_tsv(summary.participant_spectra, paths.participant_tsv)
    write_parquet(summary.participant_spectra, paths.participant_parquet)
    write_tsv(summary.cohort_spectrum, paths.summary_tsv)
    write_parquet(summary.cohort_spectrum, paths.summary_parquet)


__all__ = [
    "PreprocessingStagePsdPaths",
    "write_preprocessing_stage_psd",
]
