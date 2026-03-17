#!/usr/bin/env python3
"""
Fix fMRI BIDS outputs and add EEG↔fMRI truncation QC.

What it does
------------
0) Ensures phasediff BIDS: adds EchoTime1 and EchoTime2 to *_phasediff.json when
   missing, inferring from EchoTime/EchoTime1/EchoTime2 in the same fmap dir
   (avoids bids-validator ECHO_TIME1-2_NOT_DEFINED).
1) Renames non-BIDS fMRI events files:
     sub-*_task-*_run-*_bold_events.tsv -> sub-*_task-*_run-*_events.tsv
2) Converts fMRI events onsets to the scanner/EEG time base (0=EEG/scan start):
     - Assumes fMRI onset is in PsychoPy absolute time (e.g. raw_to_bids with
       onset_reference=as_is). If raw_to_bids used first_iti_start, onsets are
       (PsychoPy - first_iti) and this conversion would be wrong—use as_is or
       adapt the formula.
     - We estimate a per-run offset from EEG: offset = median(stim_start_time - trig_onset)
       over Trig_therm* rows, then onset_scan = onset_psychopy - offset.
     - We preserve the original in onset_psychopy and write psychopy_to_scan_offset_s.
3) Adds per-event QC columns to fMRI events.tsv:
     - trial_end_time: max(onset+duration) per trial_number
     - trial_complete_in_eeg: 1 if trial ends before EEG end, else 0
     - trial_complete_in_bold: 1 if trial ends before BOLD end, else 0
4) Writes run_qc.tsv with per-run diagnostics (durations, offset, alignment, truncation).
5) Writes an optional derived EEG phase-events table per run under qc_out/eeg_phase_events/.

Alignment assumption (not done here)
------------------------------------
Acquisition: EEG and BOLD are stopped together after each run and started together
for the next; within a run they are simultaneous. The script converts fMRI onsets
(PsychoPy) to EEG/scan time via an estimated offset. trial_complete_in_eeg compares
trial_end to eeg_recording_duration. trial_complete_in_bold compares trial_end to
BOLD end in EEG time: when fmri_nvols > eeg_volume_markers (dummies), BOLD end in
EEG time = fmri_duration - (fmri_nvols - eeg_volume_markers)*TR.

Uses only the Python standard library (no pandas) so it can run in minimal environments.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class RunQC:
    subject: str
    task: str
    run: int
    fmri_events_path: str
    fmri_bold_path: str
    eeg_events_path: str
    eeg_json_path: str
    eeg_recording_duration: float
    fmri_tr: float
    fmri_nvols: int
    fmri_duration: float
    eeg_volume_markers: int
    eeg_last_volume_onset: float
    n_trials: int
    n_trials_incomplete_in_eeg: int
    max_trial_end_time: float
    stim_alignment_max_diff_s: float
    psychopy_to_scan_offset_s: float
    n_trials_incomplete_in_bold: int


def _iter_files(root: Path, pattern: str) -> Iterable[Path]:
    yield from root.rglob(pattern)


def _rename_bold_events_files(fmri_root: Path) -> int:
    renamed = 0
    for src in _iter_files(fmri_root, "*_bold_events.tsv"):
        dst = Path(str(src).replace("_bold_events.tsv", "_events.tsv"))
        if dst.exists():
            continue
        src.rename(dst)
        renamed += 1
    return renamed


def _read_tsv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _write_tsv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _ensure_phasediff_echo_times(fmri_root: Path) -> int:
    """
    Add EchoTime1 and EchoTime2 to *_phasediff.json sidecars when missing.
    Infers from EchoTime/EchoTime1/EchoTime2 in the same subject fmap dir.
    BIDS requires both for phasediff; bids-validator fails with ECHO_TIME1-2_NOT_DEFINED otherwise.
    """
    n = 0
    for sub_dir in sorted(fmri_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        fmap_dirs: List[Path] = [sub_dir / "fmap"] if (sub_dir / "fmap").is_dir() else []
        fmap_dirs.extend(d for d in sub_dir.glob("ses-*/fmap") if d.is_dir())
        for fmap_dir in fmap_dirs:
            all_tes: List[float] = []
            for j in fmap_dir.glob("*.json"):
                m = _load_json(j)
                for k in ("EchoTime", "EchoTime1", "EchoTime2"):
                    v = m.get(k)
                    if v is not None:
                        try:
                            all_tes.append(float(v))
                        except (TypeError, ValueError):
                            pass
            all_tes = sorted(set(all_tes))
            for j in fmap_dir.glob("*_phasediff.json"):
                m = _load_json(j)
                if "EchoTime1" in m and "EchoTime2" in m:
                    continue
                if len(all_tes) < 2:
                    print(f"Warning: cannot infer EchoTime1/2 for {j} (found TEs: {all_tes})")
                    continue
                m["EchoTime1"] = all_tes[0]
                m["EchoTime2"] = all_tes[-1]
                _save_json(j, m)
                n += 1
    return n


def _nifti_nvols_tr(nifti_gz_path: Path) -> Tuple[int, float]:
    """
    Minimal NIfTI-1 header parse: dim[4] = nvols, pixdim[4] = TR.
    """
    with gzip.open(nifti_gz_path, "rb") as f:
        hdr = f.read(348)
    (sizeof_hdr,) = struct.unpack("<I", hdr[0:4])
    endian = "<" if sizeof_hdr == 348 else ">"
    dim = struct.unpack(endian + "8h", hdr[40:56])
    pixdim = struct.unpack(endian + "8f", hdr[76:108])
    nvols = int(dim[4])
    tr = float(pixdim[4])
    return nvols, tr


def _parse_subject_task_run_from_name(name: str) -> Optional[Tuple[str, str, int]]:
    # Example: sub-0000_task-thermalactive_run-01_events.tsv
    m = re.match(r"^(sub-[^_]+)_task-([^_]+)_run-(\d+)_events\.tsv$", name)
    if not m:
        return None
    sub_label = m.group(1)
    task = m.group(2)
    run = int(m.group(3))
    return sub_label, task, run


def _eeg_paths(eeg_root: Path, sub_label: str, task: str, run: int) -> Tuple[Path, Path]:
    eeg_dir = eeg_root / sub_label / "eeg"
    eeg_events = eeg_dir / f"{sub_label}_task-{task}_run-{run}_events.tsv"
    eeg_json = eeg_dir / f"{sub_label}_task-{task}_run-{run}_eeg.json"
    return eeg_events, eeg_json


def _fmri_paths(fmri_root: Path, sub_label: str, task: str, run: int) -> Tuple[Path, Path]:
    func_dir = fmri_root / sub_label / "func"
    fmri_events = func_dir / f"{sub_label}_task-{task}_run-{run:02d}_events.tsv"
    fmri_bold = func_dir / f"{sub_label}_task-{task}_run-{run:02d}_bold.nii.gz"
    return fmri_events, fmri_bold


def _volume_markers_from_eeg_events(eeg_events_path: Path) -> Tuple[int, float]:
    count = 0
    last_onset = 0.0
    parse_failures = 0
    _, rows = _read_tsv(eeg_events_path)
    for row in rows:
        tt = (row.get("trial_type") or "").strip()
        if not tt.startswith("Volume/"):
            continue
        try:
            onset = float(row.get("onset") or "")
        except (TypeError, ValueError):
            parse_failures += 1
            continue
        count += 1
        if onset > last_onset:
            last_onset = onset
    if parse_failures:
        print(f"Warning: ignored {parse_failures} malformed Volume/* onset row(s) in {eeg_events_path}")
    return count, last_onset


def _psychopy_to_scan_offset_s(eeg_events_path: Path) -> float:
    """
    Estimate a per-run constant offset between PsychoPy times and scan/EEG times.

    We assume:
      EEG trigger onset (events.tsv onset for Trig_therm*) is in scan/EEG time.
      stim_start_time is in PsychoPy clock.

    Then: scan_time = psychopy_time - offset, where offset = median(stim_start_time - trig_onset).
    Returns 0.0 if cannot compute.
    """
    deltas: List[float] = []
    parse_failures = 0
    _, rows = _read_tsv(eeg_events_path)
    for row in rows:
        tt = (row.get("trial_type") or "").strip()
        if not tt.startswith("Trig_therm"):
            continue
        try:
            trig_onset = float(row.get("onset") or "")
            stim_start = float(row.get("stim_start_time") or "")
        except (TypeError, ValueError):
            parse_failures += 1
            continue
        deltas.append(stim_start - trig_onset)
    if parse_failures:
        print(
            f"Warning: ignored {parse_failures} malformed Trig_therm timing row(s) in {eeg_events_path}"
        )
    if len(deltas) < 3:
        return 0.0
    deltas.sort()
    mid = len(deltas) // 2
    return deltas[mid] if len(deltas) % 2 == 1 else 0.5 * (deltas[mid - 1] + deltas[mid])


def _convert_fmri_events_to_scan_timebase(
    fmri_events_path: Path,
    *,
    offset_s: float,
) -> None:
    """
    Convert fMRI events.tsv onset column from PsychoPy time base to scan time base.

    Idempotent:
      - If onset_psychopy exists, assumes conversion already done and does nothing.
      - Otherwise, creates onset_psychopy, psychopy_to_scan_offset_s, and updates onset.
    """
    fieldnames, rows = _read_tsv(fmri_events_path)
    if not rows:
        return

    if "onset_psychopy" in fieldnames:
        # Already converted earlier.
        return

    # Ensure provenance columns exist (append at end for stability)
    if "onset_psychopy" not in fieldnames:
        fieldnames.append("onset_psychopy")
    if "psychopy_to_scan_offset_s" not in fieldnames:
        fieldnames.append("psychopy_to_scan_offset_s")

    parse_failures = 0
    for row in rows:
        try:
            onset_psy = float(row.get("onset") or "")
        except (TypeError, ValueError):
            parse_failures += 1
            continue
        row["onset_psychopy"] = f"{onset_psy:.6f}"
        row["psychopy_to_scan_offset_s"] = f"{offset_s:.6f}"
        row["onset"] = f"{(onset_psy - offset_s):.6f}"
    if parse_failures:
        print(
            f"Warning: left {parse_failures} row(s) unconverted due to malformed onset in {fmri_events_path}"
        )

    # Keep ordering stable for downstream tools
    def sort_key(r: Dict[str, str]) -> Tuple[float, float, str]:
        try:
            o = float(r.get("onset") or "")
        except (TypeError, ValueError):
            o = float("inf")
        try:
            tn = float(r.get("trial_number") or "")
        except (TypeError, ValueError):
            tn = float("inf")
        tt = (r.get("trial_type") or "").strip()
        return o, tn, tt

    rows.sort(key=sort_key)
    _write_tsv(fmri_events_path, fieldnames, rows)


def _annotate_fmri_events_for_qc(
    fmri_events_path: Path,
    *,
    eeg_recording_duration: float,
    fmri_duration: float,
    fmri_nvols: int,
    eeg_volume_markers: int,
    fmri_tr: float,
) -> Tuple[int, int, int, float]:
    """
    Add QC columns (onset/trial_end in EEG/scan time from prior conversion):
      - eeg_recording_duration, trial_end_time, trial_complete_in_eeg, trial_complete_in_bold

    trial_complete_in_bold: compares trial_end to BOLD end. When BOLD has dummy volumes,
    EEG 0 = BOLD time (fmri_nvols - eeg_volume_markers)*TR; BOLD end in EEG time is
    fmri_duration - that offset.

    Returns (n_trials, n_trials_incomplete_in_eeg, n_trials_incomplete_in_bold, max_trial_end_time).
    """
    fieldnames, rows = _read_tsv(fmri_events_path)
    if not rows:
        return 0, 0, 0, 0.0

    if "trial_number" not in fieldnames:
        return 0, 0, 0, 0.0

    trial_end: Dict[str, float] = {}
    parse_failures = 0
    for row in rows:
        tn = str(row.get("trial_number") or "").strip()
        if not tn:
            continue
        try:
            onset = float(row.get("onset") or "")
            dur = float(row.get("duration") or "")
        except (TypeError, ValueError):
            parse_failures += 1
            continue
        end = onset + dur
        prev = trial_end.get(tn, float("-inf"))
        if end > prev:
            trial_end[tn] = end
    if parse_failures:
        print(
            f"Warning: skipped {parse_failures} malformed onset/duration row(s) while annotating {fmri_events_path}"
        )

    eeg_tol_s = 0.10
    bold_tol_s = 1e-3

    eeg_to_bold_s = max(0.0, (fmri_nvols - eeg_volume_markers) * fmri_tr)
    bold_end_in_eeg_time = fmri_duration - eeg_to_bold_s

    incomplete_eeg = 0
    incomplete_bold = 0
    for tn, end in trial_end.items():
        if end > eeg_recording_duration + eeg_tol_s:
            incomplete_eeg += 1
        if end > bold_end_in_eeg_time + bold_tol_s:
            incomplete_bold += 1

    for col in (
        "eeg_recording_duration",
        "trial_end_time",
        "trial_complete_in_eeg",
        "trial_complete_in_bold",
    ):
        if col not in fieldnames:
            fieldnames.append(col)

    for row in rows:
        tn = str(row.get("trial_number") or "").strip()
        end = trial_end.get(tn)
        if end is None:
            continue
        row["eeg_recording_duration"] = f"{eeg_recording_duration:.6f}"
        row["trial_end_time"] = f"{end:.6f}"
        row["trial_complete_in_eeg"] = "1" if end <= eeg_recording_duration + eeg_tol_s else "0"
        row["trial_complete_in_bold"] = "1" if end <= bold_end_in_eeg_time + bold_tol_s else "0"

    _write_tsv(fmri_events_path, fieldnames, rows)
    max_end = max(trial_end.values()) if trial_end else 0.0
    return len(trial_end), incomplete_eeg, incomplete_bold, max_end


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _stim_alignment_max_diff_s(fmri_events_path: Path, eeg_events_path: Path) -> float:
    """
    Max |fMRI stimulation onset (scan time) - EEG Trig_therm onset (scan time)| over trials (seconds).
    With event_granularity=phases there are 3 stimulation rows per trial; we use min(onset) = stim start.
    Returns -1.0 if cannot compute.
    """
    _, fmri_rows = _read_tsv(fmri_events_path)
    _, eeg_rows = _read_tsv(eeg_events_path)
    fmri_stim: Dict[int, float] = {}
    for r in fmri_rows:
        if (r.get("trial_type") or "").strip() != "stimulation":
            continue
        tn = r.get("trial_number")
        on = r.get("onset")
        if tn is None or on is None:
            continue
        try:
            tni = int(float(tn))
            on_f = float(on)
            fmri_stim[tni] = min(fmri_stim[tni], on_f) if tni in fmri_stim else on_f
        except (ValueError, TypeError):
            continue
    eeg_stim: Dict[int, float] = {}
    for r in eeg_rows:
        tn = r.get("trial_number")
        onset = r.get("onset")
        tt = (r.get("trial_type") or "").strip()
        if not tt.startswith("Trig_therm"):
            continue
        if onset is None or tn is None or str(onset).strip() == "":
            continue
        try:
            eeg_stim[int(float(tn))] = float(onset)
        except (ValueError, TypeError):
            continue
    common = set(fmri_stim) & set(eeg_stim)
    if not common:
        return -1.0
    return max(abs(fmri_stim[t] - eeg_stim[t]) for t in common)


def _write_eeg_phase_events(
    *,
    eeg_events_path: Path,
    out_path: Path,
    offset_s: float,
) -> None:
    """
    Create a phase-style events table in scan/EEG time using the PsychoPy timing columns
    embedded in the EEG events.tsv (Trig_therm rows).
    """
    fieldnames, rows = _read_tsv(eeg_events_path)
    if not rows or "trial_number" not in fieldnames:
        return

    trig_rows = [r for r in rows if (r.get("trial_type") or "").strip().startswith("Trig_therm")]
    if not trig_rows:
        return

    out_rows: List[Dict[str, str]] = []
    id_parse_failures = 0
    for r in trig_rows:
        try:
            run_id = int(float(r.get("run_id") or "0"))
            trial_number = int(float(r.get("trial_number") or "0"))
        except (TypeError, ValueError):
            id_parse_failures += 1
            continue

        def f(key: str, _row: Dict[str, str] = r) -> Optional[float]:
            v = _row.get(key)
            if v is None or str(v).strip() == "":
                return None
            try:
                return float(v) - offset_s
            except (TypeError, ValueError):
                return None

        iti_start = f("iti_start_time")
        iti_end = f("iti_end_time")
        stim_start = f("stim_start_time")
        stim_end = f("stim_end_time")
        q_start = f("pain_q_start_time")
        q_end = f("pain_q_end_time")
        vas_start = f("vas_start_time")
        vas_end = f("vas_end_time")

        def add(
            tt: str,
            onset: Optional[float],
            end: Optional[float],
            _row: Dict[str, str] = r,
            _run_id: int = run_id,
            _trial_number: int = trial_number,
        ) -> None:
            if onset is None or end is None:
                return
            dur = max(0.0, end - onset)
            row: Dict[str, str] = {
                "onset": f"{onset:.6f}",
                "duration": f"{dur:.6f}",
                "trial_type": tt,
                "run_id": str(_run_id),
                "trial_number": str(_trial_number),
            }
            # Copy core behavioral columns if present
            for k in ("stimulus_temp", "selected_surface", "pain_binary_coded", "vas_final_coded_rating"):
                if k in _row and str(_row.get(k) or "").strip() != "":
                    row[k] = str(_row.get(k))
            out_rows.append(row)

        add("fixation_rest", iti_start, iti_end)
        add("stimulation", stim_start, stim_end)
        add("pain_question", q_start, q_end)
        add("vas_rating", vas_start, vas_end)

        if stim_end is not None and q_start is not None:
            add("fixation_poststim", stim_end, q_start)
    if id_parse_failures:
        print(
            f"Warning: skipped {id_parse_failures} Trig_therm row(s) with malformed run/trial identifiers in {eeg_events_path}"
        )

    out_rows.sort(key=lambda rr: (float(rr["onset"]), int(rr.get("trial_number") or "0"), rr.get("trial_type") or ""))
    out_fields = ["onset", "duration", "trial_type", "run_id", "trial_number", "stimulus_temp", "selected_surface", "pain_binary_coded", "vas_final_coded_rating"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(out_path, out_fields, out_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix fMRI BIDS outputs + EEG↔fMRI alignment QC.")
    parser.add_argument(
        "--fmri-root",
        type=Path,
        default=Path("data/bids_output/fmri"),
        help="Path to fMRI BIDS root (default: data/bids_output/fmri)",
    )
    parser.add_argument(
        "--eeg-root",
        type=Path,
        default=Path("data/bids_output/eeg"),
        help="Path to EEG BIDS root (default: data/bids_output/eeg)",
    )
    parser.add_argument(
        "--qc-out",
        type=Path,
        default=Path("data/derivatives/qc/eeg_fmri_alignment"),
        help="Output directory for QC tables (default: data/derivatives/qc/eeg_fmri_alignment)",
    )
    args = parser.parse_args()

    fmri_root: Path = args.fmri_root
    eeg_root: Path = args.eeg_root
    qc_out: Path = args.qc_out

    renamed = _rename_bold_events_files(fmri_root)
    n_echo = _ensure_phasediff_echo_times(fmri_root)
    if n_echo:
        print(f"Added EchoTime1/EchoTime2 to {n_echo} phasediff sidecar(s).")

    run_qc_rows: List[RunQC] = []
    for events_path in sorted(_iter_files(fmri_root, "*_events.tsv")):
        parsed = _parse_subject_task_run_from_name(events_path.name)
        if parsed is None:
            continue
        sub_label, task, run = parsed
        # Only process subject-level functional events (ignore task-level JSON, etc).
        if "func" not in events_path.parts:
            continue

        fmri_events, fmri_bold = _fmri_paths(fmri_root, sub_label, task, run)
        eeg_events, eeg_json = _eeg_paths(eeg_root, sub_label, task, run)

        if not (fmri_events.exists() and fmri_bold.exists() and eeg_events.exists() and eeg_json.exists()):
            continue

        eeg_meta = _load_json(eeg_json)
        eeg_dur = float(eeg_meta.get("RecordingDuration"))

        fmri_nvols, fmri_tr = _nifti_nvols_tr(fmri_bold)
        fmri_dur = (fmri_nvols - 1) * fmri_tr

        vol_count, last_vol_onset = _volume_markers_from_eeg_events(eeg_events)

        offset_s = _psychopy_to_scan_offset_s(eeg_events)
        _convert_fmri_events_to_scan_timebase(fmri_events, offset_s=offset_s)
        n_trials, n_incomplete_eeg, n_incomplete_bold, max_trial_end = _annotate_fmri_events_for_qc(
            fmri_events,
            eeg_recording_duration=eeg_dur,
            fmri_duration=fmri_dur,
            fmri_nvols=fmri_nvols,
            eeg_volume_markers=vol_count,
            fmri_tr=fmri_tr,
        )
        stim_diff = _stim_alignment_max_diff_s(fmri_events, eeg_events)

        # Write optional derived EEG phase events in scan time base (for analysis convenience).
        phase_out = qc_out / "eeg_phase_events" / f"{sub_label}_task-{task}_run-{run:02d}_events.tsv"
        _write_eeg_phase_events(eeg_events_path=eeg_events, out_path=phase_out, offset_s=offset_s)

        run_qc_rows.append(
            RunQC(
                subject=sub_label,
                task=task,
                run=run,
                fmri_events_path=str(fmri_events),
                fmri_bold_path=str(fmri_bold),
                eeg_events_path=str(eeg_events),
                eeg_json_path=str(eeg_json),
                eeg_recording_duration=eeg_dur,
                fmri_tr=fmri_tr,
                fmri_nvols=fmri_nvols,
                fmri_duration=fmri_dur,
                eeg_volume_markers=vol_count,
                eeg_last_volume_onset=last_vol_onset,
                n_trials=n_trials,
                n_trials_incomplete_in_eeg=n_incomplete_eeg,
                max_trial_end_time=max_trial_end,
                stim_alignment_max_diff_s=stim_diff,
                psychopy_to_scan_offset_s=offset_s,
                n_trials_incomplete_in_bold=n_incomplete_bold,
            )
        )

    _ensure_dir(qc_out)
    qc_path = qc_out / "run_qc.tsv"
    with qc_path.open("w", newline="") as f:
        fieldnames = [
            "subject",
            "task",
            "run",
            "eeg_recording_duration",
            "fmri_tr",
            "fmri_nvols",
            "fmri_duration",
            "eeg_volume_markers",
            "eeg_last_volume_onset",
            "n_trials",
            "n_trials_incomplete_in_eeg",
            "n_trials_incomplete_in_bold",
            "max_trial_end_time",
            "stim_alignment_max_diff_s",
            "psychopy_to_scan_offset_s",
            "fmri_events_path",
            "fmri_bold_path",
            "eeg_events_path",
            "eeg_json_path",
        ]
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        w.writeheader()
        for r in run_qc_rows:
            w.writerow(
                {
                    "subject": r.subject,
                    "task": r.task,
                    "run": r.run,
                    "eeg_recording_duration": f"{r.eeg_recording_duration:.6f}",
                    "fmri_tr": f"{r.fmri_tr:.6f}",
                    "fmri_nvols": r.fmri_nvols,
                    "fmri_duration": f"{r.fmri_duration:.6f}",
                    "eeg_volume_markers": r.eeg_volume_markers,
                    "eeg_last_volume_onset": f"{r.eeg_last_volume_onset:.6f}",
                    "n_trials": r.n_trials,
                    "n_trials_incomplete_in_eeg": r.n_trials_incomplete_in_eeg,
                    "n_trials_incomplete_in_bold": r.n_trials_incomplete_in_bold,
                    "max_trial_end_time": f"{r.max_trial_end_time:.6f}",
                    "stim_alignment_max_diff_s": f"{r.stim_alignment_max_diff_s:.6f}" if r.stim_alignment_max_diff_s >= 0 else "",
                    "psychopy_to_scan_offset_s": f"{r.psychopy_to_scan_offset_s:.6f}",
                    "fmri_events_path": r.fmri_events_path,
                    "fmri_bold_path": r.fmri_bold_path,
                    "eeg_events_path": r.eeg_events_path,
                    "eeg_json_path": r.eeg_json_path,
                }
            )

    print(f"Renamed fMRI events files: {renamed}")
    print(f"Wrote QC: {qc_path}")
    print(f"Annotated runs: {len(run_qc_rows)}")
    n_incomplete = sum(r.n_trials_incomplete_in_eeg for r in run_qc_rows)
    if n_incomplete > 0:
        print(f"Note: {n_incomplete} trial(s) extend past EEG end (truncation QC).")
    n_incomplete_b = sum(r.n_trials_incomplete_in_bold for r in run_qc_rows)
    if n_incomplete_b > 0:
        print(f"Note: {n_incomplete_b} trial(s) extend past BOLD end (truncation QC).")
    stim_diffs = [r.stim_alignment_max_diff_s for r in run_qc_rows if r.stim_alignment_max_diff_s >= 0]
    if stim_diffs:
        max_stim = max(stim_diffs)
        print(f"Stimulation alignment: max |fMRI stimulation onset − EEG Trig_therm onset| = {max_stim:.6f} s across runs (< 0.01 s ⇒ aligned).")
    def _bold_end_in_eeg(r: RunQC) -> float:
        o = max(0.0, (r.fmri_nvols - r.eeg_volume_markers) * r.fmri_tr)
        return r.fmri_duration - o

    past_bold = [r for r in run_qc_rows if r.max_trial_end_time > _bold_end_in_eeg(r) + 1e-6]
    if past_bold:
        print(
            "Diagnostic: max_trial_end > BOLD end (in EEG time) for %s — "
            "BOLD run shorter than task, or events not in scan time (check PsychoPy→scan offset)."
            % ", ".join(f"{r.subject} run-{r.run:02d}" for r in past_bold)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
