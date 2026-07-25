"""Write Study 1 volume-marker-locked rectified ECG artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import pandas as pd

from eeg_pipeline.infra.tsv import write_tsv
from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.config.loader import load_study1_config
from studies.pain_study.study1.figures.preprocessing_psd_sources import (
    BrainVisionFileRunSource,
    discover_processed_brainvision_runs,
    discover_raw_brainvision_runs,
)
from studies.pain_study.study1.figures.validity_style import (
    save_publication_png,
    save_publication_svg,
)
from studies.pain_study.study1.figures.volume_locked_rectified_ecg import (
    VolumeLockedEcgRun,
    VolumeLockedEcgSpecification,
    VolumeLockedEcgSummary,
    build_volume_locked_ecg_summary,
    summarize_volume_locked_ecg_run,
    volume_locked_ecg_specification,
)
from studies.pain_study.study1.figures.volume_locked_rectified_ecg_plot import (
    build_volume_locked_ecg_figure,
)


@dataclass(frozen=True)
class VolumeLockedEcgPaths:
    """Figure and source-data paths for the volume-locked ECG report."""

    svg: Path
    png: Path
    peaks_tsv: Path
    troughs_tsv: Path
    run_traces_tsv: Path
    participant_traces_tsv: Path


def analyze_volume_locked_ecg(
    *,
    source_data_root: Path,
    subjects: Sequence[str],
    config: Any,
) -> VolumeLockedEcgSummary:
    """Load paired native and Analyzer-corrected BrainVision ECG runs."""
    requested_subjects = tuple(subjects)
    if not requested_subjects:
        raise ValueError("At least one participant must be requested.")
    if len(set(requested_subjects)) != len(requested_subjects):
        raise ValueError("Requested participants must be unique.")

    sources = _discover_volume_locked_sources(
        source_data_root=Path(source_data_root),
        requested_subjects=requested_subjects,
        config=config,
    )
    specification = volume_locked_ecg_specification(config)
    runs = tuple(
        _summarize_source(
            source,
            stage=stage,
            expected_sampling_frequency_hz=expected_sampling_frequency_hz,
            specification=specification,
        )
        for stage, expected_sampling_frequency_hz, source in sources
    )
    expected_runs = int(
        require_config_value(
            config,
            "study1.figures.volume_locked_rectified_ecg.expected_runs_per_participant",
        )
    )
    return build_volume_locked_ecg_summary(
        runs,
        specification,
        expected_runs_per_participant=expected_runs,
    )


def _discover_volume_locked_sources(
    *,
    source_data_root: Path,
    requested_subjects: tuple[str, ...],
    config: Any,
) -> tuple[tuple[str, float, BrainVisionFileRunSource], ...]:
    configured_stages = require_config_value(
        config,
        "study1.figures.volume_locked_rectified_ecg.stages",
    )
    sources: list[tuple[str, float, BrainVisionFileRunSource]] = []
    for stage in ("raw", "processed"):
        discovery = (
            discover_raw_brainvision_runs if stage == "raw" else discover_processed_brainvision_runs
        )
        discovered = discovery(
            Path(source_data_root),
            excluded_subjects=(),
            requested_subjects=requested_subjects,
            source_corrections=(),
            source_exclusions=(),
        )
        for source in discovered:
            if not isinstance(source, BrainVisionFileRunSource):
                raise TypeError(
                    "Volume-locked ECG requires on-disk BrainVision triplets; "
                    f"got {source.representation}."
                )
            sources.append(
                (
                    stage,
                    float(configured_stages[stage]["sampling_frequency_hz"]),
                    source,
                )
            )
    return tuple(sources)


def write_volume_locked_ecg_summary(
    summary: VolumeLockedEcgSummary,
    *,
    config: Any,
    output_dir: Path,
) -> VolumeLockedEcgPaths:
    """Write SVG/PNG figures and exact plotted source tables."""
    directory = Path(output_dir)
    paths = _artifact_paths(directory)
    figure_config = require_config_value(
        config,
        "study1.figures.volume_locked_rectified_ecg",
    )
    dimensions = figure_config["dimensions_mm"]
    save_publication_svg(
        build_volume_locked_ecg_figure(
            summary.participants,
            summary.peaks,
            summary.troughs,
            config,
        ),
        paths.svg,
        config,
        dimensions_mm=dimensions,
    )
    save_publication_png(
        build_volume_locked_ecg_figure(
            summary.participants,
            summary.peaks,
            summary.troughs,
            config,
        ),
        paths.png,
        config,
        dimensions_mm=dimensions,
        dpi=int(figure_config["png_dpi"]),
    )
    write_tsv(summary.peaks, paths.peaks_tsv)
    write_tsv(summary.troughs, paths.troughs_tsv)
    write_tsv(_run_trace_frame(summary), paths.run_traces_tsv)
    write_tsv(_participant_trace_frame(summary), paths.participant_traces_tsv)
    return paths


def write_volume_locked_rectified_ecg(
    *,
    source_data_root: Path,
    subjects: Sequence[str],
    config: Any,
    output_dir: Path,
) -> VolumeLockedEcgPaths:
    """Analyze paired recordings and write the complete report."""
    summary = analyze_volume_locked_ecg(
        source_data_root=source_data_root,
        subjects=subjects,
        config=config,
    )
    return write_volume_locked_ecg_summary(
        summary,
        config=config,
        output_dir=output_dir,
    )


def _summarize_source(
    source: BrainVisionFileRunSource,
    *,
    stage: str,
    expected_sampling_frequency_hz: float,
    specification: VolumeLockedEcgSpecification,
) -> VolumeLockedEcgRun:
    raw = mne.io.read_raw_brainvision(
        source.header_path,
        preload=False,
        verbose="ERROR",
    )
    try:
        return summarize_volume_locked_ecg_run(
            raw,
            subject_id=source.subject_id,
            run_id=source.run_id,
            stage=stage,
            source_file=source.source_path,
            expected_sampling_frequency_hz=expected_sampling_frequency_hz,
            specification=specification,
        )
    finally:
        raw.close()


def _artifact_paths(directory: Path) -> VolumeLockedEcgPaths:
    stem = "volume_locked_rectified_ecg"
    return VolumeLockedEcgPaths(
        svg=directory / f"{stem}.svg",
        png=directory / f"{stem}.png",
        peaks_tsv=directory / f"{stem}_artifact_peaks.tsv",
        troughs_tsv=directory / f"{stem}_analyzer_troughs.tsv",
        run_traces_tsv=directory / f"{stem}_by_run.tsv",
        participant_traces_tsv=directory / f"{stem}_by_subject.tsv",
    )


def _run_trace_frame(summary: VolumeLockedEcgSummary) -> pd.DataFrame:
    frames = []
    for run in summary.runs:
        frames.append(
            pd.DataFrame(
                {
                    "subject_id": run.subject_id,
                    "stage": run.stage,
                    "run": int(run.run_id),
                    "time_ms": run.times_ms,
                    "mean_rectified_ecg_uv": run.mean_rectified_ecg_uv,
                    "sampling_frequency_hz": run.sampling_frequency_hz,
                    "n_complete_epochs": run.n_complete_epochs,
                    "rectification_order": "abs_then_volume_mean",
                    "source_file": run.source_file,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _participant_trace_frame(summary: VolumeLockedEcgSummary) -> pd.DataFrame:
    frames = []
    for participant in summary.participants:
        frames.append(
            pd.DataFrame(
                {
                    "subject_id": participant.subject_id,
                    "stage": participant.stage,
                    "time_ms": participant.times_ms,
                    "mean_rectified_ecg_uv": participant.mean_rectified_ecg_uv,
                    "sampling_frequency_hz": participant.sampling_frequency_hz,
                    "n_runs": participant.n_runs,
                    "n_complete_epochs": participant.n_complete_epochs,
                    "run_weighting": "equal_run_mean",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main(argv: Sequence[str] | None = None) -> VolumeLockedEcgPaths:
    parser = argparse.ArgumentParser(
        description="Plot 0–900 ms volume-locked rectified ECG before and after Analyzer correction."
    )
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--subject", action="append", required=True)
    parser.add_argument("--study1-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)

    paths = write_volume_locked_rectified_ecg(
        source_data_root=arguments.source_data_root,
        subjects=tuple(arguments.subject),
        config=load_study1_config(arguments.study1_config),
        output_dir=arguments.output_dir,
    )
    for path in paths.__dict__.values():
        print(path)
    return paths


if __name__ == "__main__":
    main()


__all__ = [
    "VolumeLockedEcgPaths",
    "analyze_volume_locked_ecg",
    "main",
    "write_volume_locked_ecg_summary",
    "write_volume_locked_rectified_ecg",
]
