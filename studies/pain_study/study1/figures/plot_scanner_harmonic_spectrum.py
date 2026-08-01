"""Write the standalone Study 1 scanner-harmonic QC figure."""

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
from studies.pain_study.study1.figures.scanner_harmonic_spectrum import (
    ScannerHarmonicSummary,
    build_scanner_harmonic_figure,
    build_scanner_harmonic_summary,
    discover_final_clean_runs,
    estimate_run_spectrum,
    scanner_harmonic_specification,
    validity_bootstrap_specification,
)
from studies.pain_study.study1.figures.validity_style import (
    save_publication_svg,
    validity_output_dir,
)

OUTPUT_FILENAME = "scanner_harmonic_spectrum.svg"
RUN_AUDIT_FILENAME = "scanner_harmonic_spectrum_by_run.tsv"
PARTICIPANT_AUDIT_FILENAME = "scanner_harmonic_spectrum_by_subject.tsv"


@dataclass(frozen=True)
class ScannerHarmonicFigurePaths:
    """Figure and audit paths written by the standalone QC stage."""

    svg: Path
    run_tsv: Path
    run_parquet: Path
    participant_tsv: Path
    participant_parquet: Path


def write_scanner_harmonic_spectrum(
    *,
    derivative_root: Path,
    task: str,
    config: Any,
    subjects: Sequence[str] = (),
    output_path: Path | None = None,
) -> ScannerHarmonicFigurePaths:
    """Analyze final-clean EEG and write one SVG plus reproducibility audits."""
    summary = _build_summary(
        derivative_root=Path(derivative_root),
        task=task,
        config=config,
        subjects=subjects,
    )
    resolved_output = output_path or validity_output_dir(config) / OUTPUT_FILENAME
    figure = build_scanner_harmonic_figure(summary, config)
    save_publication_svg(
        figure,
        resolved_output,
        config,
        dimensions_mm=require_config_value(
            config,
            "study1.figures.scanner_harmonics.dimensions_mm",
        ),
    )

    run_tsv = resolved_output.with_name(RUN_AUDIT_FILENAME)
    run_parquet = run_tsv.with_suffix(".parquet")
    participant_tsv = resolved_output.with_name(PARTICIPANT_AUDIT_FILENAME)
    participant_parquet = participant_tsv.with_suffix(".parquet")
    write_tsv(summary.run_audit, run_tsv)
    write_parquet(summary.run_audit, run_parquet)
    write_tsv(summary.participant_audit, participant_tsv)
    write_parquet(summary.participant_audit, participant_parquet)
    return ScannerHarmonicFigurePaths(
        svg=resolved_output,
        run_tsv=run_tsv,
        run_parquet=run_parquet,
        participant_tsv=participant_tsv,
        participant_parquet=participant_parquet,
    )


def _build_summary(
    *,
    derivative_root: Path,
    task: str,
    config: Any,
    subjects: Sequence[str],
) -> ScannerHarmonicSummary:
    specification = scanner_harmonic_specification(config)
    run_paths = discover_final_clean_runs(
        derivative_root,
        task=task,
        excluded_subjects=specification.excluded_subjects,
        requested_subjects=subjects,
    )
    run_spectra = tuple(estimate_run_spectrum(path, specification) for path in run_paths)
    return build_scanner_harmonic_summary(
        run_spectra,
        specification,
        bootstrap=validity_bootstrap_specification(config),
    )


def main(argv: Sequence[str] | None = None) -> ScannerHarmonicFigurePaths:
    parser = argparse.ArgumentParser(
        description="Write the Study 1 scanner-harmonic spectrum SVG and audits."
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
    output_paths = write_scanner_harmonic_spectrum(
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
    "ScannerHarmonicFigurePaths",
    "main",
    "write_scanner_harmonic_spectrum",
]
