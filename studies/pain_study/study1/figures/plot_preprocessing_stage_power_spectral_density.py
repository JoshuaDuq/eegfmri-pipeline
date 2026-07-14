"""Write Study 1 preprocessing-stage cohort power spectral density artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from eeg_pipeline.infra.tsv import write_parquet, write_tsv
from eeg_pipeline.utils.config.loader import load_config, require_config_value
from eeg_pipeline.utils.config.roots import resolve_eeg_deriv_root
from studies.pain_study.study1.config.loader import apply_study1_config_defaults
from studies.pain_study.study1.figures.cohort_power_spectral_density import (
    CohortPsdSummary,
)
from studies.pain_study.study1.figures.preprocessing_psd_sources import (
    BrainVisionSourceCorrection,
    BrainVisionSourceExclusion,
    EegRunSource,
    discover_mne_runs,
    discover_processed_brainvision_runs,
    discover_raw_brainvision_runs,
)
from studies.pain_study.study1.figures.preprocessing_stage_power_spectral_density import (
    PreprocessingStagePsdSpecification,
    build_preprocessing_stage_psd_summary,
    preprocessing_stage_psd_specification,
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

STAGE_IDENTIFIERS = ("raw", "processed", "mne")
BRAINVISION_TASK = "thermalactive"


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
    return _write_preprocessing_stage_psd_summary(
        summary=summary,
        specification=specification,
        config=config,
        output_dir=output_dir,
    )


def _write_preprocessing_stage_psd_summary(
    *,
    summary: CohortPsdSummary,
    specification: PreprocessingStagePsdSpecification,
    config: Any,
    output_dir: Path | None,
) -> PreprocessingStagePsdPaths:
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


def write_preprocessing_stage_psds(
    *,
    stage_identifiers: Sequence[str],
    kingston_root: Path,
    derivative_root: Path | None,
    task: str,
    config: Any,
    subjects: Sequence[str] = (),
    output_dir: Path | None = None,
) -> tuple[PreprocessingStagePsdPaths, ...]:
    """Validate every requested stage, then write reports in request order."""
    stages = tuple(stage_identifiers)
    if not stages:
        raise ValueError("At least one preprocessing PSD stage is required.")
    if len(set(stages)) != len(stages):
        raise ValueError("Duplicate preprocessing PSD stage requested.")
    if any(stage in {"raw", "processed"} for stage in stages) and task != BRAINVISION_TASK:
        raise ValueError(f"BrainVision preprocessing PSD stages require task {BRAINVISION_TASK!r}.")

    specifications = tuple(
        preprocessing_stage_psd_specification(config, stage_identifier)
        for stage_identifier in stages
    )
    source_sets = tuple(
        _discover_stage_sources(
            stage_identifier=specification.stage.identifier,
            kingston_root=Path(kingston_root),
            derivative_root=Path(derivative_root) if derivative_root is not None else None,
            task=task,
            excluded_subjects=specification.excluded_subjects,
            requested_subjects=subjects,
            source_corrections=specification.source_corrections,
            source_exclusions=specification.source_exclusions,
        )
        for specification in specifications
    )
    summaries = tuple(
        _build_stage_summary(
            sources=sources,
            specification=specification,
            config=config,
        )
        for specification, sources in zip(specifications, source_sets, strict=True)
    )
    directory = Path(output_dir) if output_dir is not None else validity_output_dir(config)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".study1-psd-", dir=directory.parent) as temporary:
        staging_directory = Path(temporary)
        staged_paths = tuple(
            _write_preprocessing_stage_psd_summary(
                summary=summary,
                specification=specification,
                config=config,
                output_dir=staging_directory,
            )
            for summary, specification in zip(summaries, specifications, strict=True)
        )
        directory.mkdir(parents=True, exist_ok=True)
        return tuple(_publish_stage_paths(paths, directory) for paths in staged_paths)


def _discover_stage_sources(
    *,
    stage_identifier: str,
    kingston_root: Path,
    derivative_root: Path | None,
    task: str,
    excluded_subjects: Sequence[str],
    requested_subjects: Sequence[str],
    source_corrections: Sequence[BrainVisionSourceCorrection],
    source_exclusions: Sequence[BrainVisionSourceExclusion],
) -> tuple[EegRunSource, ...]:
    if stage_identifier == "raw":
        return discover_raw_brainvision_runs(
            kingston_root,
            excluded_subjects=excluded_subjects,
            requested_subjects=requested_subjects,
            source_corrections=source_corrections,
            source_exclusions=source_exclusions,
        )
    if stage_identifier == "processed":
        return discover_processed_brainvision_runs(
            kingston_root,
            excluded_subjects=excluded_subjects,
            requested_subjects=requested_subjects,
            source_corrections=source_corrections,
            source_exclusions=source_exclusions,
        )
    if derivative_root is None:
        raise ValueError("The MNE preprocessing PSD stage requires an EEG derivative root.")
    return discover_mne_runs(
        derivative_root,
        task=task,
        excluded_subjects=excluded_subjects,
        requested_subjects=requested_subjects,
    )


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


def _publish_stage_paths(
    staged: PreprocessingStagePsdPaths,
    directory: Path,
) -> PreprocessingStagePsdPaths:
    published = {}
    for field in fields(staged):
        staged_path = getattr(staged, field.name)
        published_path = directory / staged_path.name
        staged_path.replace(published_path)
        published[field.name] = published_path
    return PreprocessingStagePsdPaths(**published)


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


def main(argv: Sequence[str] | None = None) -> tuple[PreprocessingStagePsdPaths, ...]:
    parser = argparse.ArgumentParser(
        description="Write Study 1 raw, BrainVision-processed, and MNE-processed PSD reports."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study1-config", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--kingston-root", type=Path, required=True)
    parser.add_argument("--derivative-root", type=Path)
    parser.add_argument(
        "--stage",
        action="append",
        choices=STAGE_IDENTIFIERS,
        required=True,
    )
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    apply_study1_config_defaults(config, arguments.study1_config)
    derivative_root = arguments.derivative_root
    if "mne" in arguments.stage and derivative_root is None:
        derivative_root = resolve_eeg_deriv_root(config)
    output_paths = write_preprocessing_stage_psds(
        stage_identifiers=tuple(arguments.stage),
        kingston_root=arguments.kingston_root,
        derivative_root=derivative_root,
        task=arguments.task,
        config=config,
        subjects=tuple(arguments.subject),
        output_dir=arguments.output_dir,
    )
    for paths in output_paths:
        print(paths.svg)
    return output_paths


if __name__ == "__main__":
    main()


__all__ = [
    "PreprocessingStagePsdPaths",
    "main",
    "write_preprocessing_stage_psd",
    "write_preprocessing_stage_psds",
]
