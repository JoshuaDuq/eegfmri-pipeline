"""BIDS metadata helpers for raw-to-BIDS utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


THERMAL_EVENTS_SCHEMA: dict[str, dict[str, Any]] = {
    "run_id": {"Description": "Run index (1-based; matches PsychoPy run_id)."},
    "trial_number": {"Description": "Within-run trial index (1-based)."},
    "stimulus_temp": {"Description": "Thermode target temperature.", "Units": "C"},
    "selected_surface": {"Description": "Stimulus surface index (experiment-defined)."},
    "binary_outcome_coded": {"Description": "Pain yes/no response (1=yes, 0=no)."},
    "vas_final_coded_rating": {
        "Description": "Final VAS-coded rating (non-pain: 0–99 heat; pain: 100–200 pain)."
    },
    "iti_start_time": {"Description": "Trial ITI start time (experiment clock).", "Units": "s"},
    "iti_end_time": {"Description": "Trial ITI end time (experiment clock).", "Units": "s"},
    "stim_start_time": {"Description": "Stimulation start time (experiment clock).", "Units": "s"},
    "stim_end_time": {"Description": "Stimulation end time (experiment clock).", "Units": "s"},
    "pain_q_start_time": {"Description": "Pain question start time (experiment clock).", "Units": "s"},
    "pain_q_end_time": {"Description": "Pain question end time (experiment clock).", "Units": "s"},
    "vas_start_time": {"Description": "VAS start time (experiment clock).", "Units": "s"},
    "vas_end_time": {"Description": "VAS end time (experiment clock).", "Units": "s"},
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


def events_json_path(events_tsv: Path) -> Path:
    return events_tsv.with_suffix(".json")


def ensure_events_sidecar(events_tsv: Path, columns: list[str]) -> None:
    """Ensure the `_events.json` sidecar documents merged columns."""
    sidecar = events_json_path(events_tsv)
    data = _load_json(sidecar)

    data.setdefault(
        "onset",
        {
            "Description": "Onset (in seconds) of the event from the beginning of the first datapoint.",
            "Units": "s",
        },
    )
    data.setdefault(
        "duration",
        {
            "Description": "Duration of the event in seconds from onset (0 indicates an impulse event).",
            "Units": "s",
        },
    )
    data.setdefault(
        "trial_type",
        {"Description": "The type, category, or name of the event."},
    )
    data.setdefault(
        "value",
        {"Description": "The event code (trigger code or event ID) associated with the event."},
    )
    data.setdefault(
        "sample",
        {"Description": "The event onset time in number of sampling points (first sample is 0)."},
    )

    for col in columns:
        if col in {"onset", "duration", "trial_type", "value", "sample"}:
            continue
        if col in data:
            continue
        if col in THERMAL_EVENTS_SCHEMA:
            data[col] = THERMAL_EVENTS_SCHEMA[col]
            continue
        if col.endswith("_time"):
            data[col] = {"Description": f"{col} (experiment clock).", "Units": "s"}
        else:
            data[col] = {"Description": f"{col} (additional per-event column)."}

    _write_json(sidecar, data)


def ensure_task_events_json(bids_root: Path, task: str) -> None:
    out = bids_root / f"task-{task}_events.json"
    if out.exists():
        return
    schema: dict[str, Any] = {
        "onset": {"Description": "Event onset in seconds from the start of the EEG run.", "Units": "s"},
        "duration": {"Description": "Event duration in seconds.", "Units": "s"},
        "trial_type": {"Description": "Event label (BrainVision/MNE annotation description)."},
        "value": {"Description": "Event code (trigger ID)."},
        "sample": {"Description": "Event onset sample index (first sample is 0)."},
    }
    schema.update(THERMAL_EVENTS_SCHEMA)
    _write_json(out, schema)


def ensure_participants_tsv(bids_root: Path, subject_labels: list[str]) -> None:
    path = bids_root / "participants.tsv"
    header_cols: list[str] = []
    existing: set[str] = set()

    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            header_cols = ["participant_id"]
        else:
            header_cols = [c.lstrip("\ufeff") for c in lines[0].split("\t")]
            for line in lines[1:]:
                if not line.strip():
                    continue
                existing.add(line.split("\t")[0].strip())
    else:
        header_cols = ["participant_id"]
        path.write_text("participant_id\n", encoding="utf-8")

    if "participant_id" not in header_cols:
        return

    extra_cols = [c for c in header_cols if c != "participant_id"]
    new_rows: list[str] = []
    for s in sorted(set(subject_labels)):
        pid = f"sub-{s}"
        if pid in existing:
            continue
        if extra_cols:
            new_rows.append("\t".join([pid] + ["n/a"] * len(extra_cols)))
        else:
            new_rows.append(pid)

    if new_rows:
        with path.open("a", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(f"{row}\n")

