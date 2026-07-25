"""Create non-destructive BrainVision metadata with unambiguous VAS markers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import mne
import numpy as np

from eeg_pipeline.preprocessing.brainvision_markers import sanitize_vas_marker_text

RUN_PATTERN = re.compile(r"^ThermalPainEEGFMRI_run(?P<run>\d+)_sub(?P<subject>[^_]+)_")
DEFAULT_RECORDING_OVERRIDES_PATH = (
    Path(__file__).parent / "config/native_eeg_fmri_recording_overrides.tsv"
)


@dataclass(frozen=True)
class CohortRecording:
    """One cohort run and source representation."""

    subject: str
    run: int
    source_layout: str
    source_vhdr: Path


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return stream.read()


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)


def _setting_value(text: str, key: str) -> str:
    prefix = f"{key}="
    values = [line[len(prefix) :] for line in text.splitlines() if line.startswith(prefix)]
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {key} setting, found {len(values)}")
    return values[0]


def _replace_setting(text: str, key: str, value: str) -> str:
    prefix = f"{key}="
    replacements = 0
    output_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        newline = line[len(content) :]
        if content.startswith(prefix):
            output_lines.append(f"{prefix}{value}{newline}")
            replacements += 1
        else:
            output_lines.append(line)
    if replacements != 1:
        raise ValueError(f"Expected exactly one {key} setting, found {replacements}")
    return "".join(output_lines)


def _resolve_reference(parent: Path, value: str) -> Path:
    relative_path = Path(value.replace("\\", "/"))
    return (parent / relative_path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_source_header(source_vhdr: Path) -> tuple[str, int]:
    match = RUN_PATTERN.match(source_vhdr.stem)
    if match is None:
        raise ValueError(f"Cannot parse subject and run from {source_vhdr.name}")
    return match.group("subject"), int(match.group("run"))


def load_recording_overrides(path: Path) -> dict[str, int | None]:
    """Load explicit exclusions and logical run identities for exceptional acquisitions."""
    if not path.is_file():
        raise FileNotFoundError(f"Recording overrides do not exist: {path}")
    overrides: dict[str, int | None] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream, delimiter="\t"), start=2):
            required = {"subject", "source_vhdr", "action", "run"}
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"Recording override line {line_number} is missing: {missing}")
            source_name = row["source_vhdr"].strip()
            if Path(source_name).name != source_name or not source_name.endswith(".vhdr"):
                raise ValueError(
                    f"Recording override line {line_number} has an invalid source_vhdr"
                )
            subject, _ = _parse_source_header(Path(source_name))
            if row["subject"].strip() != subject:
                raise ValueError(f"Recording override line {line_number} has a subject mismatch")
            if source_name in overrides:
                raise ValueError(f"Duplicate recording override for {source_name}")

            action = row["action"].strip()
            run_text = row["run"].strip()
            if action == "exclude":
                if run_text:
                    raise ValueError(
                        f"Excluded recording override line {line_number} must not define a run"
                    )
                overrides[source_name] = None
            elif action == "include":
                run = int(run_text)
                if run < 1:
                    raise ValueError(f"Recording override line {line_number} has invalid run {run}")
                overrides[source_name] = run
            else:
                raise ValueError(
                    f"Recording override line {line_number} has invalid action {action!r}"
                )
    return overrides


def discover_cohort_recordings(
    source_data_root: Path,
    *,
    subjects: Sequence[str] | None = None,
    recording_overrides: Mapping[str, int | None] | None = None,
) -> list[CohortRecording]:
    """Discover thermal recordings in the original and Analyzer-processed layouts."""
    source_layouts = ("brainvision_processed_1khz", "original_5khz")
    source_headers = sorted(
        source_vhdr
        for source_layout in source_layouts
        for source_vhdr in source_data_root.glob(
            f"sub-*/eeg/{source_layout}/ThermalPainEEGFMRI_run*_sub*_*.vhdr"
        )
        if not source_vhdr.name.startswith("._")
    )
    if not source_headers:
        raise FileNotFoundError(
            f"No supported thermal EEG-fMRI recordings found in {source_data_root}"
        )

    recordings: list[CohortRecording] = []
    seen_layout_subject_runs: set[tuple[str, str, int]] = set()
    overrides = dict(recording_overrides or {})
    matched_overrides: set[str] = set()
    requested_subjects = set(subjects) if subjects is not None else None
    discovered_subjects: set[str] = set()
    for source_vhdr in source_headers:
        subject, run = _parse_source_header(source_vhdr)
        source_layout = source_vhdr.parent.name
        directory_subject = source_vhdr.parent.parent.parent.name.removeprefix("sub-")
        if subject != directory_subject:
            raise ValueError(
                f"Source subject mismatch for {source_vhdr}: {subject} != {directory_subject}"
            )
        discovered_subjects.add(subject)
        if requested_subjects is not None and subject not in requested_subjects:
            continue

        _source_files(source_vhdr)
        logical_run = run
        if source_vhdr.name in overrides:
            matched_overrides.add(source_vhdr.name)
            override_run = overrides[source_vhdr.name]
            if override_run is None:
                continue
            logical_run = override_run

        layout_subject_run = (source_layout, subject, logical_run)
        if layout_subject_run in seen_layout_subject_runs:
            raise ValueError(
                f"Ambiguous {source_layout} recordings map to sub-{subject} run-{logical_run}; "
                "add explicit recording overrides"
            )
        seen_layout_subject_runs.add(layout_subject_run)
        recordings.append(
            CohortRecording(
                subject=subject,
                run=logical_run,
                source_layout=source_layout,
                source_vhdr=source_vhdr,
            )
        )

    if requested_subjects is not None:
        missing_subjects = sorted(requested_subjects - discovered_subjects)
        if missing_subjects:
            raise FileNotFoundError(
                f"No original 5 kHz recordings found for subjects: {missing_subjects}"
            )
    if not recordings:
        raise FileNotFoundError("Subject selection contains no original 5 kHz recordings")
    relevant_overrides = {
        source_name
        for source_name in overrides
        if requested_subjects is None
        or _parse_source_header(Path(source_name))[0] in requested_subjects
    }
    unmatched_overrides = sorted(relevant_overrides - matched_overrides)
    if unmatched_overrides:
        raise FileNotFoundError(
            f"Recording overrides do not match discovered source headers: {unmatched_overrides}"
        )

    return sorted(
        recordings,
        key=lambda recording: (
            recording.subject,
            recording.source_layout,
            recording.run,
        ),
    )


def _source_files(source_vhdr: Path) -> tuple[Path, Path, str]:
    header_text = _read_text(source_vhdr)
    marker_path = _resolve_reference(
        source_vhdr.parent,
        _setting_value(header_text, "MarkerFile"),
    )
    data_path = _resolve_reference(
        source_vhdr.parent,
        _setting_value(header_text, "DataFile"),
    )
    for companion in (marker_path, data_path):
        if not companion.is_file():
            raise FileNotFoundError(f"Missing BrainVision companion file: {companion}")
    return marker_path, data_path, header_text


def _validate_raw(raw: mne.io.BaseRaw, source_vhdr: Path) -> None:
    if float(raw.info["sfreq"]) <= 0.0:
        raise ValueError(f"Sampling frequency must be positive in {source_vhdr}")
    if len(raw.ch_names) != 64:
        raise ValueError(f"Expected 64 channels in {source_vhdr}, got {len(raw.ch_names)}")
    if raw.ch_names.count("ECG") != 1:
        raise ValueError(f"Expected exactly one ECG channel in {source_vhdr}")


def _verify_staged_raw(source_vhdr: Path, staged_vhdr: Path) -> None:
    source_raw = mne.io.read_raw_brainvision(source_vhdr, preload=False, verbose=False)
    staged_raw = mne.io.read_raw_brainvision(staged_vhdr, preload=False, verbose=False)

    scalar_pairs = (
        ("sampling frequency", source_raw.info["sfreq"], staged_raw.info["sfreq"]),
        ("sample count", source_raw.n_times, staged_raw.n_times),
        ("measurement date", source_raw.info["meas_date"], staged_raw.info["meas_date"]),
    )
    for label, source_value, staged_value in scalar_pairs:
        if source_value != staged_value:
            raise ValueError(f"Staged {label} differs for {source_vhdr}")
    if source_raw.ch_names != staged_raw.ch_names:
        raise ValueError(f"Staged channel order differs for {source_vhdr}")

    expected_descriptions = [
        "Vas_on/VAS_ON" if description == "Vas_on/V  1" else description
        for description in source_raw.annotations.description
    ]
    if list(staged_raw.annotations.description) != expected_descriptions:
        raise ValueError(f"Staged annotation descriptions differ unexpectedly for {source_vhdr}")
    np.testing.assert_array_equal(source_raw.annotations.onset, staged_raw.annotations.onset)
    np.testing.assert_array_equal(source_raw.annotations.duration, staged_raw.annotations.duration)

    window_size = min(100, source_raw.n_times)
    starts = sorted({0, max(0, source_raw.n_times // 2), source_raw.n_times - window_size})
    for start in starts:
        stop = min(source_raw.n_times, start + window_size)
        np.testing.assert_array_equal(
            source_raw.get_data(start=start, stop=stop),
            staged_raw.get_data(start=start, stop=stop),
        )


def stage_recording(
    recording: CohortRecording,
    source_data_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Write and verify one metadata-only sanitized BrainVision view."""
    source_marker, source_data, header_text = _source_files(recording.source_vhdr)
    source_marker_text = _read_text(source_marker)
    source_data_stat = source_data.stat()
    source_raw = mne.io.read_raw_brainvision(
        recording.source_vhdr,
        preload=False,
        verbose=False,
    )
    _validate_raw(source_raw, recording.source_vhdr)
    result = sanitize_vas_marker_text(source_marker_text, n_samples=source_raw.n_times)
    if result.volume_count == 0:
        raise ValueError(f"Marker file contains no Volume,V  1 markers: {source_marker}")

    source_relative_vhdr = recording.source_vhdr.relative_to(source_data_root)
    output_dir = output_root / source_relative_vhdr.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_vhdr = output_dir / recording.source_vhdr.name
    staged_vmrk = staged_vhdr.with_suffix(".vmrk")
    if staged_vhdr.exists() or staged_vmrk.exists():
        raise FileExistsError(f"Staged BrainVision metadata already exists: {staged_vhdr}")

    relative_data = os.path.relpath(source_data, output_dir).replace(os.sep, "/")
    staged_header_text = _replace_setting(header_text, "DataFile", relative_data)
    staged_header_text = _replace_setting(
        staged_header_text,
        "MarkerFile",
        staged_vmrk.name,
    )
    staged_marker_text = _replace_setting(result.text, "DataFile", relative_data)
    _write_text(staged_vhdr, staged_header_text)
    _write_text(staged_vmrk, staged_marker_text)

    staged_data_reference = _resolve_reference(
        staged_vhdr.parent,
        _setting_value(staged_header_text, "DataFile"),
    )
    if staged_data_reference != source_data:
        raise ValueError(f"Staged header does not reference the original EEG: {staged_vhdr}")
    _verify_staged_raw(recording.source_vhdr, staged_vhdr)

    final_data_stat = source_data.stat()
    if (
        source_data_stat.st_size != final_data_stat.st_size
        or source_data_stat.st_mtime_ns != final_data_stat.st_mtime_ns
    ):
        raise RuntimeError(f"Original EEG metadata changed during staging: {source_data}")

    return {
        "subject": f"sub-{recording.subject}",
        "run": recording.run,
        "source_layout": recording.source_layout,
        "source_relative_vhdr": str(source_relative_vhdr),
        "source_vhdr": str(recording.source_vhdr),
        "source_vmrk": str(source_marker),
        "source_eeg": str(source_data),
        "staged_vhdr": str(staged_vhdr),
        "volume_count": result.volume_count,
        "vas_count": result.vas_count,
        "sampling_frequency_hz": float(source_raw.info["sfreq"]),
        "source_vhdr_sha256": _sha256(recording.source_vhdr),
        "source_vmrk_sha256": _sha256(source_marker),
        "source_eeg_size": source_data_stat.st_size,
        "source_eeg_mtime_ns": source_data_stat.st_mtime_ns,
        "verified": True,
    }


def _write_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    manifest_path = output_root / "marker_sanitization_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _remove_appledouble_files(root: Path) -> None:
    for resource_fork in root.rglob("._*"):
        resource_fork.unlink()


def run_sanitization(
    source_data_root: Path,
    output_root: Path,
    *,
    subjects: Sequence[str] | None = None,
    recording_overrides: Mapping[str, int | None] | None = None,
) -> Path:
    """Create and atomically publish sanitized metadata for discovered recordings."""
    if output_root.exists():
        raise FileExistsError(f"Output root already exists: {output_root}")
    temporary_root = output_root.parent / f".{output_root.name}.tmp"
    if temporary_root.exists():
        raise FileExistsError(f"Temporary output root already exists: {temporary_root}")

    recordings = discover_cohort_recordings(
        source_data_root,
        subjects=subjects,
        recording_overrides=recording_overrides,
    )
    succeeded = False
    try:
        rows = [
            stage_recording(recording, source_data_root, temporary_root)
            for recording in recordings
        ]
        published_rows = []
        for row in rows:
            staged_vhdr = Path(str(row["staged_vhdr"]))
            published_row = dict(row)
            published_row["staged_vhdr"] = str(
                output_root / staged_vhdr.relative_to(temporary_root)
            )
            published_rows.append(published_row)
        _write_manifest(temporary_root, published_rows)
        summary = {
            "run_count": len(rows),
            "volume_count": sum(int(row["volume_count"]) for row in rows),
            "vas_count": sum(int(row["vas_count"]) for row in rows),
            "source_eeg_bytes_copied": 0,
            "replacement": "Vas_on/V  1 -> Vas_on/VAS_ON",
        }
        (temporary_root / "marker_sanitization_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary_root / "README.md").write_text(
            "# BrainVision marker-sanitized metadata\n\n"
            "These headers and marker files reference immutable source `.eeg` files.\n"
            "Only `Vas_on,V  1` descriptions were changed to `Vas_on,VAS_ON`.\n",
            encoding="utf-8",
        )
        _remove_appledouble_files(temporary_root)
        temporary_root.replace(output_root)
        succeeded = True
    finally:
        if not succeeded and temporary_root.exists():
            shutil.rmtree(temporary_root)
    return output_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage unambiguous VAS markers for supported BrainVision recordings."
    )
    parser.add_argument(
        "--source-data-root",
        type=Path,
        default=Path("/Volumes/KINGSTON/EEG_fMRI_data/source_data"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Volumes/KINGSTON/EEG_fMRI_data/derivatives/brainvision_marker_sanitized-v2"),
    )
    parser.add_argument(
        "--subject",
        action="append",
        default=None,
        help="Subject label without 'sub-'; repeat to select subjects. Defaults to all discovered.",
    )
    parser.add_argument(
        "--recording-overrides",
        type=Path,
        default=DEFAULT_RECORDING_OVERRIDES_PATH,
        help="TSV containing explicit exclusions and logical run overrides.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = run_sanitization(
        source_data_root=args.source_data_root,
        output_root=args.output_root,
        subjects=args.subject,
        recording_overrides=load_recording_overrides(args.recording_overrides),
    )
    print(output_root)


if __name__ == "__main__":
    main()
