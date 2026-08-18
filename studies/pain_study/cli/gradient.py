"""``eeg-pipeline gradient`` -- measure what the upstream gradient correction left behind.

``measure`` reads each run, measures the volume timing from its marker train, and scores
the comb against its local background before and after the ICA exclusions; it writes the
tables. ``plot`` runs the same measurement and draws it.

The volume rate is measured from the markers in every run rather than read from
configuration, so a run whose markers are missing or too few is reported as not measured
rather than measured and found clean. What each number means, and what it does not, is in
``studies/pain_study/scripts/gradient/README.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

MODES = ("measure", "plot")

FILTERED_SUFFIX = "_proc-filt_raw.fif"
ICA_SUFFIX = "_proc-ica_ica.fif"


def setup_gradient(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the gradient parser."""
    parser = subparsers.add_parser(
        "gradient",
        help="Measure and draw the residual scanner gradient comb",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=list(MODES), help="Stage to run, in the order listed")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Restrict to these subjects (default: every subject found)",
    )
    parser.add_argument(
        # Not --config: the top-level CLI strips that out of argv before argparse runs, so
        # a subcommand declaring it advertises an option it can never be given.
        "--workflow-config",
        dest="config",
        type=str,
        default=None,
        help="Workflow YAML (default: studies/pain_study/scripts/gradient/config.yaml)",
    )
    parser.add_argument("--deriv-root", type=str, default=None, help="Preprocessed EEG root")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Where the tables and figures go"
    )
    return parser


def _subject_dirs(deriv_root: Path, subjects: List[str] | None) -> list[Path]:
    wanted = {s if s.startswith("sub-") else f"sub-{s}" for s in (subjects or [])}
    found = sorted(p for p in deriv_root.glob("sub-*") if p.is_dir())
    return [p for p in found if not wanted or p.name in wanted]


def _runs_of(subject_dir: Path) -> tuple[list[Path], Path | None]:
    filtered = sorted(subject_dir.rglob(f"*{FILTERED_SUFFIX}"))
    icas = sorted(subject_dir.rglob(f"*{ICA_SUFFIX}"))
    return filtered, icas[0] if icas else None


def _measure_subject(subject_dir: Path, settings: dict[str, Any]) -> tuple[list, list, list]:
    import mne

    from studies.pain_study.analysis.gradient.comb import (
        CombNotMeasured,
        compute_comb_residual,
        measure_volume_timing,
    )
    from studies.pain_study.analysis.gradient.locked import compute_volume_locked_average

    combs: list = []
    lockeds: list = []
    declined: list = []

    filtered, ica_path = _runs_of(subject_dir)
    if not filtered or ica_path is None:
        return combs, lockeds, declined
    ica = mne.preprocessing.read_ica(ica_path, verbose="ERROR")

    for path in filtered:
        recording_id = path.name.replace(FILTERED_SUFFIX, "")
        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        timing = measure_volume_timing(
            raw, description=settings["volume_marker_description"]
        )
        # No marker train means this run was not recorded in the scanner. It gets no
        # measurement rather than an empty one.
        if timing is None:
            continue
        cleaned = ica.apply(raw.copy(), verbose="ERROR")
        comb = compute_comb_residual(
            raw,
            cleaned,
            timing=timing,
            recording_id=recording_id,
            band_hz=tuple(settings["comb_frequency_range_hz"]),
            welch_seconds=settings["comb_welch_seconds"],
        )
        (declined if isinstance(comb, CombNotMeasured) else combs).append(comb)
        locked = compute_volume_locked_average(
            raw,
            cleaned,
            timing=timing,
            recording_id=recording_id,
            description=settings["volume_marker_description"],
        )
        if locked is not None:
            lockeds.append(locked)
    return combs, lockeds, declined


def _settings(config_path: Path | None) -> dict[str, Any]:
    import yaml

    default = Path("studies/pain_study/scripts/gradient/config.yaml")
    path = Path(config_path) if config_path else default
    document = yaml.safe_load(path.read_text()) or {}
    return document.get("gradient", {})


def run_gradient(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Dispatch one stage to the module that implements it."""
    from studies.pain_study.scripts.gradient import plot as gradient_plot
    from studies.pain_study.scripts.gradient import tables

    if subjects and not args.subjects:
        args.subjects = list(subjects)

    settings = _settings(args.config)
    deriv_root = Path(args.deriv_root) if args.deriv_root else Path(config.paths.deriv_root)
    output_dir = Path(args.output_dir) if args.output_dir else deriv_root / "gradient"
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject_dir in _subject_dirs(deriv_root, args.subjects):
        combs, lockeds, declined = _measure_subject(subject_dir, settings)
        if not combs and not lockeds and not declined:
            continue
        stem = output_dir / subject_dir.name
        if args.mode == "measure":
            if combs:
                tables.comb_frame(combs).to_csv(f"{stem}_comb.tsv", sep="\t", index=False)
            if lockeds:
                tables.locked_frame(lockeds).to_csv(f"{stem}_locked.tsv", sep="\t", index=False)
            if declined:
                tables.declined_frame(declined).to_csv(
                    f"{stem}_declined.tsv", sep="\t", index=False
                )
        else:
            if combs:
                gradient_plot.save(
                    gradient_plot.plot_comb_residual(combs), Path(f"{stem}_comb.png")
                )
            if lockeds:
                gradient_plot.save(
                    gradient_plot.plot_volume_locked_average(lockeds), Path(f"{stem}_locked.png")
                )
