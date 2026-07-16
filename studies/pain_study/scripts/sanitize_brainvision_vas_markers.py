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
from typing import Sequence

import mne
import numpy as np

from eeg_pipeline.preprocessing.brainvision_markers import sanitize_vas_marker_text

DEFAULT_SUBJECTS = ("0001",) + tuple(f"{subject:04d}" for subject in range(3, 16))
EXPECTED_RUN_COUNT = 83
CORRECTED_SUFFIX = "_scannerpulse_corrected.vhdr"
RUN_PATTERN = re.compile(r"^ThermalPainEEGFMRI_run(?P<run>\d+)_sub(?P<subject>[^_]+)_")


@dataclass(frozen=True)
class CohortRecording:
    """One cohort run mapped to its original 5 kHz BrainVision header."""

    subject: str
    run: int
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


def _parse_reference(reference: Path) -> tuple[str, int, str]:
    if not reference.name.endswith(CORRECTED_SUFFIX):
        raise ValueError(f"Unexpected corrected reference filename: {reference}")
    original_basename = reference.name.removesuffix(CORRECTED_SUFFIX)
    match = RUN_PATTERN.match(original_basename)
    if match is None:
        raise ValueError(f"Cannot parse subject and run from {reference.name}")
    return match.group("subject"), int(match.group("run")), original_basename


def discover_cohort_recordings(
    source_data_root: Path,
    *,
    subjects: Sequence[str] = DEFAULT_SUBJECTS,
    expected_count: int = EXPECTED_RUN_COUNT,
) -> list[CohortRecording]:
    """Map the fixed corrected-source inventory to unique original recordings."""
    subject_set = set(subjects)
    references = sorted(
        reference
        for reference in source_data_root.glob(
            "sub-*/eeg/brainvision_processed_1khz/"
            "ThermalPainEEGFMRI_run*_scannerpulse_corrected.vhdr"
        )
        if not reference.name.startswith("._")
        and reference.parent.parent.parent.name.removeprefix("sub-") in subject_set
    )
    if len(references) != expected_count:
        raise ValueError(
            f"Expected {expected_count} corrected cohort references, found {len(references)}"
        )

    recordings: list[CohortRecording] = []
    seen_subject_runs: set[tuple[str, int]] = set()
    for reference in references:
        subject, run, original_basename = _parse_reference(reference)
        reference_subject = reference.parent.parent.parent.name.removeprefix("sub-")
        if subject != reference_subject:
            raise ValueError(
                f"Reference subject mismatch for {reference}: {subject} != {reference_subject}"
            )

        source_name = f"{original_basename}.vhdr"
        matches = sorted(
            path
            for path in source_data_root.glob(f"sub-{subject}/eeg/original_5khz/{source_name}")
            if not path.name.startswith("._")
        )
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one original for {reference}, found {len(matches)}")

        subject_run = (subject, run)
        if subject_run in seen_subject_runs:
            raise ValueError(f"Duplicate cohort subject/run mapping: sub-{subject} run-{run}")
        seen_subject_runs.add(subject_run)
        recordings.append(CohortRecording(subject=subject, run=run, source_vhdr=matches[0]))

    return sorted(recordings, key=lambda recording: (recording.subject, recording.run))


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
    if raw.info["sfreq"] != 5_000.0:
        raise ValueError(f"Expected 5000 Hz in {source_vhdr}, got {raw.info['sfreq']}")
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


def stage_recording(recording: CohortRecording, output_root: Path) -> dict[str, object]:
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

    output_dir = output_root / f"sub-{recording.subject}" / "eeg"
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
        "source_vhdr": str(recording.source_vhdr),
        "source_vmrk": str(source_marker),
        "source_eeg": str(source_data),
        "staged_vhdr": str(staged_vhdr),
        "volume_count": result.volume_count,
        "vas_count": result.vas_count,
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
    subjects: Sequence[str] = DEFAULT_SUBJECTS,
    expected_count: int = EXPECTED_RUN_COUNT,
) -> Path:
    """Create and atomically publish the complete sanitized cohort metadata."""
    if output_root.exists():
        raise FileExistsError(f"Output root already exists: {output_root}")
    temporary_root = output_root.parent / f".{output_root.name}.tmp"
    if temporary_root.exists():
        raise FileExistsError(f"Temporary output root already exists: {temporary_root}")

    recordings = discover_cohort_recordings(
        source_data_root,
        subjects=subjects,
        expected_count=expected_count,
    )
    succeeded = False
    try:
        rows = [stage_recording(recording, temporary_root) for recording in recordings]
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
            "These headers and marker files reference the immutable original 5 kHz `.eeg` files.\n"
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
        description="Stage unambiguous VAS markers for the 5 kHz BrainVision cohort."
    )
    parser.add_argument(
        "--source-data-root",
        type=Path,
        default=Path("/Volumes/KINGSTON/EEG_fMRI_data/source_data"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Volumes/KINGSTON/EEG_fMRI_data/derivatives/brainvision_marker_sanitized-v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = run_sanitization(
        source_data_root=args.source_data_root,
        output_root=args.output_root,
    )
    print(output_root)


if __name__ == "__main__":
    main()
