"""Write participant and cohort Study 1 band time-frequency figures."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from eeg_pipeline.infra.tsv import write_parquet, write_tsv
from eeg_pipeline.utils.config.loader import load_config, require_config_value
from eeg_pipeline.utils.config.roots import resolve_eeg_deriv_root
from studies.pain_study.study1.config.loader import apply_study1_config_defaults
from studies.pain_study.study1.figures.band_time_frequency import (
    FIGURE_CONFIG_KEY,
    BandTfrSummary,
    load_band_tfr_summary,
)
from studies.pain_study.study1.figures.band_time_frequency_plot import (
    build_cohort_band_tfr_figure,
    build_participant_band_tfr_figure,
    cohort_band_color_limit,
    participant_band_color_limit,
)
from studies.pain_study.study1.figures.validity_style import (
    save_publication_svg,
    validity_output_dir,
)


@dataclass(frozen=True)
class BandTfrFigurePaths:
    """Published participant/cohort figures and their numerical audits."""

    output_dir: Path
    participant_svgs: tuple[Path, ...]
    cohort_svgs: tuple[Path, ...]
    subject_tsv: Path
    subject_parquet: Path
    cohort_tsv: Path
    cohort_parquet: Path
    sources_tsv: Path
    sources_parquet: Path


def write_band_time_frequency(
    *,
    task: str,
    config: Any,
    derivative_root: Path | None = None,
    output_dir: Path | None = None,
) -> BandTfrFigurePaths:
    """Compute and atomically publish all participant and cohort band TFRs."""

    root = Path(derivative_root) if derivative_root is not None else resolve_eeg_deriv_root(config)
    summary = load_band_tfr_summary(
        task=task,
        derivative_root=root,
        config=config,
    )
    destination = (
        Path(output_dir)
        if output_dir is not None
        else validity_output_dir(config) / "band_time_frequency"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".study1-band-tfr-", dir=destination.parent) as temporary:
        staged = _write_band_tfr_summary(
            summary,
            config=config,
            output_dir=Path(temporary),
        )
        destination.mkdir(parents=True, exist_ok=True)
        return _publish_paths(staged, destination)


def _write_band_tfr_summary(
    summary: BandTfrSummary,
    *,
    config: Any,
    output_dir: Path,
) -> BandTfrFigurePaths:
    figure_config = require_config_value(config, FIGURE_CONFIG_KEY)
    if not isinstance(figure_config, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY} must be a mapping.")
    dimensions = figure_config.get("dimensions_mm")
    if not isinstance(dimensions, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY}.dimensions_mm must be a mapping.")
    percentile = float(figure_config.get("color_percentile"))
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_svgs = []
    cohort_svgs = []
    for band in summary.bands:
        participant_limit = participant_band_color_limit(
            summary,
            band,
            percentile=percentile,
        )
        for subject_id in summary.subject_ids:
            output_path = output_dir / f"band_time_frequency_{band}_{subject_id}.svg"
            figure = build_participant_band_tfr_figure(
                summary,
                band=band,
                subject_id=subject_id,
                color_limit=participant_limit,
                config=config,
            )
            participant_svgs.append(
                save_publication_svg(
                    figure,
                    output_path,
                    config,
                    dimensions_mm=dimensions,
                )
            )
        output_path = output_dir / f"band_time_frequency_{band}_cohort.svg"
        cohort_limit = cohort_band_color_limit(summary, band, percentile=percentile)
        figure = build_cohort_band_tfr_figure(
            summary,
            band=band,
            color_limit=cohort_limit,
            config=config,
        )
        cohort_svgs.append(
            save_publication_svg(
                figure,
                output_path,
                config,
                dimensions_mm=dimensions,
            )
        )

    paths = _audit_paths(output_dir)
    write_tsv(summary.subject_maps, paths.subject_tsv)
    write_parquet(summary.subject_maps, paths.subject_parquet)
    write_tsv(summary.cohort_maps, paths.cohort_tsv)
    write_parquet(summary.cohort_maps, paths.cohort_parquet)
    write_tsv(summary.source_audit, paths.sources_tsv)
    write_parquet(summary.source_audit, paths.sources_parquet)
    return BandTfrFigurePaths(
        output_dir=output_dir,
        participant_svgs=tuple(participant_svgs),
        cohort_svgs=tuple(cohort_svgs),
        subject_tsv=paths.subject_tsv,
        subject_parquet=paths.subject_parquet,
        cohort_tsv=paths.cohort_tsv,
        cohort_parquet=paths.cohort_parquet,
        sources_tsv=paths.sources_tsv,
        sources_parquet=paths.sources_parquet,
    )


def _audit_paths(output_dir: Path) -> BandTfrFigurePaths:
    subject_tsv = output_dir / "band_time_frequency_by_subject.tsv"
    cohort_tsv = output_dir / "band_time_frequency_summary.tsv"
    sources_tsv = output_dir / "band_time_frequency_sources.tsv"
    return BandTfrFigurePaths(
        output_dir=output_dir,
        participant_svgs=(),
        cohort_svgs=(),
        subject_tsv=subject_tsv,
        subject_parquet=subject_tsv.with_suffix(".parquet"),
        cohort_tsv=cohort_tsv,
        cohort_parquet=cohort_tsv.with_suffix(".parquet"),
        sources_tsv=sources_tsv,
        sources_parquet=sources_tsv.with_suffix(".parquet"),
    )


def _publish_paths(staged: BandTfrFigurePaths, destination: Path) -> BandTfrFigurePaths:
    def publish(path: Path) -> Path:
        published = destination / path.name
        path.replace(published)
        return published

    return BandTfrFigurePaths(
        output_dir=destination,
        participant_svgs=tuple(publish(path) for path in staged.participant_svgs),
        cohort_svgs=tuple(publish(path) for path in staged.cohort_svgs),
        subject_tsv=publish(staged.subject_tsv),
        subject_parquet=publish(staged.subject_parquet),
        cohort_tsv=publish(staged.cohort_tsv),
        cohort_parquet=publish(staged.cohort_parquet),
        sources_tsv=publish(staged.sources_tsv),
        sources_parquet=publish(staged.sources_parquet),
    )


def main(argv: Sequence[str] | None = None) -> BandTfrFigurePaths:
    parser = argparse.ArgumentParser(
        description="Write Study 1 participant and cohort band TFR figures."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study1-config", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--derivative-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    apply_study1_config_defaults(config, arguments.study1_config)
    outputs = write_band_time_frequency(
        task=arguments.task,
        derivative_root=arguments.derivative_root,
        config=config,
        output_dir=arguments.output_dir,
    )
    for path in outputs.participant_svgs + outputs.cohort_svgs:
        print(path)
    return outputs


if __name__ == "__main__":
    main()


__all__ = ["BandTfrFigurePaths", "main", "write_band_time_frequency"]
