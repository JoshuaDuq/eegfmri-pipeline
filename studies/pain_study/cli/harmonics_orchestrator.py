"""Execution orchestrator for scanner-harmonic QC command."""

from __future__ import annotations

import argparse
from typing import Any, Sequence

from studies.pain_study.analysis.gradient.scanner_harmonics import (
    DEFAULT_GAMMA_EXCLUSIONS,
    DEFAULT_HARMONIC_WINDOWS,
    DEFAULT_QC_CHANNELS,
    FrequencyWindow,
    analyze_brainvision_file,
    discover_brainvision_files,
    write_scanner_harmonic_reports,
)


def run_harmonics(args: argparse.Namespace, subjects: list[str], config: Any) -> None:
    """Execute scanner-harmonic QC."""
    selected_subjects = _resolve_subjects(args)
    gamma_window = FrequencyWindow("gamma", args.gamma_band[0], args.gamma_band[1])
    gamma_exclusions = _build_windows(
        args.exclude_band,
        default=DEFAULT_GAMMA_EXCLUSIONS,
        prefix="scanner_exclusion",
    )
    harmonic_windows = _build_windows(
        args.harmonic_window,
        default=DEFAULT_HARMONIC_WINDOWS,
        prefix="scanner_harmonic",
    )
    channels = args.channels if args.channels is not None else DEFAULT_QC_CHANNELS

    vhdr_paths = discover_brainvision_files(
        args.input_root,
        subjects=selected_subjects,
        pattern=args.pattern,
    )
    if not vhdr_paths:
        raise FileNotFoundError(
            f"No BrainVision .vhdr files matched {args.pattern!r} under {args.input_root}"
        )

    rows = [
        analyze_brainvision_file(
            vhdr_path,
            channels=channels,
            nperseg=args.nperseg,
            min_samples=args.min_samples,
            gamma_window=gamma_window,
            gamma_exclusions=gamma_exclusions,
            harmonic_windows=harmonic_windows,
        )
        for vhdr_path in vhdr_paths
    ]
    tsv_path, json_path = write_scanner_harmonic_reports(rows, args.output_dir)
    print(f"Wrote scanner-harmonic QC TSV: {tsv_path}")
    print(f"Wrote scanner-harmonic QC JSON: {json_path}")


def _resolve_subjects(args: argparse.Namespace) -> list[str] | None:
    if getattr(args, "all_subjects", False) or getattr(args, "group", None) == "all":
        return None
    if getattr(args, "group", None):
        return [subject.strip() for subject in args.group.split(",") if subject.strip()]
    return getattr(args, "subject", None)


def _build_windows(
    values: Sequence[Sequence[float]] | None,
    *,
    default: Sequence[FrequencyWindow],
    prefix: str,
) -> tuple[FrequencyWindow, ...]:
    if values is None:
        return tuple(default)
    return tuple(
        FrequencyWindow(f"{prefix}_{index}", low_hz, high_hz)
        for index, (low_hz, high_hz) in enumerate(values, start=1)
    )
