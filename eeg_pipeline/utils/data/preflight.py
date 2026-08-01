"""Read-only inspection of a BIDS dataset before any of it is processed.

Issue #14: setting up a study meant discovering its shape by hand. How many subjects and
runs, whether every recording has events, whether the sampling rate is the same
throughout, whether the ECG channel the config names is actually recorded — each was
answerable only by opening files, or by starting preprocessing and waiting for something
to fail.

Two deliberate limits.

*Nothing here opens a recording.* Every answer comes from BIDS metadata — the ``_eeg.json``
sidecars, ``channels.tsv``, ``events.tsv`` — which is where BIDS already requires it to
be. That keeps the check fast enough to run before deciding anything, and free of MNE, so
it works in the half-built environment of someone still installing.

*Nothing here grades.* A sampling rate that differs between runs is reported as the two
rates, not as a failure: whether that is wrong depends on the study, and this module does
not know the study. The three statuses are statements of fact — something is as
configured, something varies across runs, or something is not there — and the reader draws
the conclusion.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: What this module can say. Deliberately not a severity: see the module docstring.
STATUS_OK = "ok"
STATUS_DIFFERS = "differs"
STATUS_ABSENT = "absent"

#: BIDS EEG raw extensions. A run is identified by its raw file; the sidecars are found
#: from its name.
_EEG_EXTENSIONS = (".vhdr", ".edf", ".bdf", ".set", ".fif")

#: How BIDS spells a missing value, plus the spellings that reach these files in practice.
_MISSING_VALUES = frozenset({"", "n/a", "na", "nan", "none", "null"})


@dataclass(frozen=True)
class Observation:
    """One fact about the dataset, named by the config key or topic it concerns."""

    key: str
    status: str
    message: str

    def __str__(self) -> str:
        return f"{self.key}: {self.message}"


@dataclass(frozen=True)
class PreflightReport:
    observations: Tuple[Observation, ...]

    def by_status(self, status: str) -> Tuple[Observation, ...]:
        return tuple(item for item in self.observations if item.status == status)


@dataclass(frozen=True)
class RunFile:
    """One recording and the sidecars that describe it."""

    raw: Path
    entities: Dict[str, str]

    @property
    def subject(self) -> str:
        return self.entities.get("sub", "")

    @property
    def task(self) -> str:
        return self.entities.get("task", "")

    @property
    def session(self) -> str:
        return self.entities.get("ses", "")

    def sidecar(self, suffix: str) -> Path:
        """The companion file with this ``suffix``, e.g. ``channels.tsv``."""
        name = self.raw.name
        for extension in _EEG_EXTENSIONS:
            if name.endswith(f"_eeg{extension}"):
                base = name[: -len(f"_eeg{extension}")]
                return self.raw.with_name(f"{base}_{suffix}")
        return self.raw.with_name(suffix)


def _plural(count: int, noun: str, plural: str = "") -> str:
    """``3 runs`` / ``1 run``. A report that miscounts its own grammar reads as careless."""
    return f"{count} {noun if count == 1 else (plural or noun + 's')}"


def _number(value: Any) -> str:
    """Render a frequency without the trailing ``.0`` that makes 500.0 read as precision."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


def _parse_entities(path: Path) -> Dict[str, str]:
    entities: Dict[str, str] = {}
    for part in path.name.split("_"):
        key, separator, value = part.partition("-")
        if separator:
            entities[key] = value
    return entities


def discover_runs(bids_root: Path) -> List[RunFile]:
    """Every EEG recording under ``bids_root``, in a stable order."""
    runs: List[RunFile] = []
    for path in sorted(bids_root.glob("sub-*/**/*_eeg.*")):
        if path.suffix.lower() not in _EEG_EXTENSIONS or path.name.startswith("._"):
            continue
        runs.append(RunFile(raw=path, entities=_parse_entities(path)))
    return runs


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def _is_missing(value: Optional[str]) -> bool:
    return value is None or str(value).strip().lower() in _MISSING_VALUES


def _channel_types(run: RunFile) -> Dict[str, str]:
    """``name -> type`` from this run's channels.tsv, upper-cased for comparison."""
    rows = _read_tsv(run.sidecar("channels.tsv"))
    return {
        str(row.get("name", "")).strip(): str(row.get("type", "")).strip().upper()
        for row in rows
        if str(row.get("name", "")).strip()
    }


###################################################################
# Individual observations
###################################################################


def _observe_roots(config: Any, observations: List[Observation]) -> Optional[Path]:
    """Report both roots. Returns the BIDS root only if it is there to be read."""
    raw_bids_root = config.get("paths.bids_root", None)
    bids_root = Path(str(raw_bids_root)) if raw_bids_root else None

    if bids_root is None:
        observations.append(
            Observation(
                "paths.bids_root",
                STATUS_ABSENT,
                "is not configured, so there is no dataset to inspect.",
            )
        )
    elif not bids_root.is_dir():
        observations.append(
            Observation(
                "paths.bids_root",
                STATUS_ABSENT,
                f"{bids_root} does not exist. Nothing below could be measured.",
            )
        )
        bids_root = None
    else:
        observations.append(Observation("paths.bids_root", STATUS_OK, f"found at {bids_root}."))

    _observe_deriv_root(config, observations)
    return bids_root


def _observe_deriv_root(config: Any, observations: List[Observation]) -> None:
    raw_deriv_root = config.get("paths.deriv_root", None)
    if not raw_deriv_root:
        observations.append(
            Observation(
                "paths.deriv_root",
                STATUS_ABSENT,
                "is not configured, so there is nowhere to write derivatives.",
            )
        )
        return

    deriv_root = Path(str(raw_deriv_root))
    if deriv_root.is_dir():
        writable = os.access(deriv_root, os.W_OK)
        observations.append(
            Observation(
                "paths.deriv_root",
                STATUS_OK if writable else STATUS_ABSENT,
                (
                    f"exists at {deriv_root} and is writable."
                    if writable
                    else f"exists at {deriv_root} but is not writable."
                ),
            )
        )
        return

    # Absent is the normal state before a first run, so the question is whether it can be
    # made. Answered by asking the nearest existing ancestor, never by creating anything.
    ancestor = deriv_root
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent

    if ancestor.is_dir() and os.access(ancestor, os.W_OK):
        observations.append(
            Observation(
                "paths.deriv_root",
                STATUS_OK,
                f"does not exist yet and will be created at {deriv_root}.",
            )
        )
    else:
        observations.append(
            Observation(
                "paths.deriv_root",
                STATUS_ABSENT,
                f"cannot be created at {deriv_root}: {ancestor} is not writable.",
            )
        )


def _observe_inventory(runs: Sequence[RunFile], observations: List[Observation]) -> None:
    subjects = sorted({run.subject for run in runs if run.subject})
    tasks = sorted({run.task for run in runs if run.task})
    sessions = sorted({run.session for run in runs if run.session})

    if not runs:
        observations.append(
            Observation(
                "inventory",
                STATUS_ABSENT,
                "no EEG recordings found under the BIDS root.",
            )
        )
        return

    parts = [_plural(len(subjects), "subject"), _plural(len(runs), "run")]
    if sessions:
        parts.insert(1, _plural(len(sessions), "session"))
    observations.append(
        Observation(
            "inventory",
            STATUS_OK,
            f"{', '.join(parts)}; task labels present: {', '.join(tasks) or 'none'}.",
        )
    )


def _observe_configured_task(
    config: Any, runs: Sequence[RunFile], observations: List[Observation]
) -> List[RunFile]:
    """Report whether the configured task label is one the recordings carry.

    Returns the runs the rest of the checks apply to. When the label matches nothing,
    every later observation is made over the whole tree rather than over an empty set,
    because reporting the shape of what *is* there is more use than reporting nothing.
    """
    task = config.get("project.task", None)
    tasks_present = sorted({run.task for run in runs if run.task})

    if not task:
        observations.append(
            Observation(
                "project.task",
                STATUS_ABSENT,
                f"is not set. Task labels in the dataset: {', '.join(tasks_present) or 'none'}.",
            )
        )
        return list(runs)

    matching = [run for run in runs if run.task == str(task)]
    if matching:
        observations.append(
            Observation(
                "project.task",
                STATUS_OK,
                f"task-{task} matches {len(matching)} of {len(runs)} recordings.",
            )
        )
        return matching

    observations.append(
        Observation(
            "project.task",
            STATUS_ABSENT,
            f"is {str(task)!r}, which no recording carries. Task labels in the dataset: "
            f"{', '.join(tasks_present) or 'none'}.",
        )
    )
    return list(runs)


def _observe_events_pairing(runs: Sequence[RunFile], observations: List[Observation]) -> None:
    paired = [run for run in runs if run.sidecar("events.tsv").exists()]
    if len(paired) == len(runs):
        observations.append(
            Observation(
                "events_pairing",
                STATUS_OK,
                f"all {_plural(len(runs), 'recording')} have an events.tsv.",
            )
        )
        return

    unpaired = [run.raw.name for run in runs if run not in paired]
    observations.append(
        Observation(
            "events_pairing",
            STATUS_DIFFERS,
            f"{len(paired)}/{len(runs)} recordings have an events.tsv. Without one: "
            f"{', '.join(unpaired[:5])}"
            f"{' …' if len(unpaired) > 5 else ''}.",
        )
    )


def _observe_sampling_frequency(runs: Sequence[RunFile], observations: List[Observation]) -> None:
    counts: Dict[str, int] = {}
    without = 0
    for run in runs:
        value = _read_json(run.sidecar("eeg.json")).get("SamplingFrequency", None)
        if value is None:
            without += 1
            continue
        counts[_number(value)] = counts.get(_number(value), 0) + 1

    if not counts:
        observations.append(
            Observation(
                "sampling_frequency",
                STATUS_ABSENT,
                "no _eeg.json sidecar states SamplingFrequency, which BIDS requires.",
            )
        )
        return

    if len(counts) == 1 and not without:
        rate = next(iter(counts))
        observations.append(
            Observation(
                "sampling_frequency",
                STATUS_OK,
                f"{rate} Hz in all {_plural(len(runs), 'recording')}.",
            )
        )
        return

    listing = ", ".join(f"{rate} Hz in {count}" for rate, count in sorted(counts.items()))
    unstated = f"; {without} sidecars do not state it" if without else ""
    observations.append(
        Observation(
            "sampling_frequency",
            STATUS_DIFFERS,
            f"differs across recordings: {listing}{unstated}.",
        )
    )


def _observe_channel_layout(runs: Sequence[RunFile], observations: List[Observation]) -> None:
    layouts: List[Tuple[RunFile, frozenset]] = []
    for run in runs:
        names = frozenset(_channel_types(run))
        if names:
            layouts.append((run, names))

    if not layouts:
        observations.append(
            Observation(
                "channel_layout",
                STATUS_ABSENT,
                "no channels.tsv found, so the channel layout could not be compared.",
            )
        )
        return

    everywhere = frozenset.intersection(*(names for _, names in layouts))
    anywhere = frozenset.union(*(names for _, names in layouts))
    inconsistent = sorted(anywhere - everywhere)

    if not inconsistent:
        observations.append(
            Observation(
                "channel_layout",
                STATUS_OK,
                f"the same {_plural(len(everywhere), 'channel')} in all "
                f"{_plural(len(layouts), 'recording')}.",
            )
        )
        return

    observations.append(
        Observation(
            "channel_layout",
            STATUS_DIFFERS,
            f"{_plural(len(everywhere), 'channel')} in every recording; "
            f"{len(inconsistent)} {'appears' if len(inconsistent) == 1 else 'appear'} "
            f"in some but not all: "
            f"{', '.join(inconsistent[:10])}{' …' if len(inconsistent) > 10 else ''}.",
        )
    )


def _observe_declared_channels(
    config: Any,
    key: str,
    runs: Sequence[RunFile],
    observations: List[Observation],
) -> None:
    """Report a declared channel against the recordings, and only if one is declared.

    Not declaring an ECG or EOG lead is an ordinary configuration, not an omission, so
    nothing is said about it.
    """
    declared = [str(name) for name in (config.get(key, None) or [])]
    if not declared:
        return

    for name in declared:
        present = [run for run in runs if name in _channel_types(run)]
        if len(present) == len(runs) and runs:
            observations.append(
                Observation(
                    key,
                    STATUS_OK,
                    f"{name!r} is present in all {_plural(len(runs), 'recording')}.",
                )
            )
        elif present:
            observations.append(
                Observation(
                    key,
                    STATUS_DIFFERS,
                    f"{name!r} is present in {len(present)}/{len(runs)} recordings.",
                )
            )
        else:
            recorded = sorted(
                {
                    channel
                    for run in runs
                    for channel, kind in _channel_types(run).items()
                    if kind in {"ECG", "EKG", "EOG"}
                }
            )
            observations.append(
                Observation(
                    key,
                    STATUS_ABSENT,
                    f"{name!r} is in no recording's channels.tsv. Channels typed ECG or EOG "
                    f"in this dataset: {', '.join(recorded) or 'none'}.",
                )
            )


def _observe_trigger_channels(runs: Sequence[RunFile], observations: List[Observation]) -> None:
    with_triggers = [
        run
        for run in runs
        if any(kind in {"TRIG", "STIM"} for kind in _channel_types(run).values())
    ]
    if not with_triggers:
        observations.append(
            Observation(
                "trigger_channels",
                STATUS_ABSENT,
                "no channel is typed TRIG or STIM. This is expected when event timing "
                "comes from events.tsv rather than a recorded trigger line.",
            )
        )
        return

    observations.append(
        Observation(
            "trigger_channels",
            STATUS_OK if len(with_triggers) == len(runs) else STATUS_DIFFERS,
            f"a TRIG or STIM channel is present in {len(with_triggers)}/{len(runs)} recordings.",
        )
    )


def _observe_line_frequency(
    config: Any, runs: Sequence[RunFile], observations: List[Observation]
) -> None:
    notch = config.get("preprocessing.notch_freq", None)
    if notch is None:
        return

    stated = {
        _number(value)
        for value in (
            _read_json(run.sidecar("eeg.json")).get("PowerLineFrequency", None) for run in runs
        )
        if value is not None
    }

    if not stated:
        observations.append(
            Observation(
                "preprocessing.notch_freq",
                STATUS_ABSENT,
                f"is {_number(notch)} Hz; no sidecar states PowerLineFrequency, so it could "
                "not be confirmed against the recordings.",
            )
        )
        return

    if stated == {_number(notch)}:
        observations.append(
            Observation(
                "preprocessing.notch_freq",
                STATUS_OK,
                f"{_number(notch)} Hz matches the PowerLineFrequency of every recording.",
            )
        )
        return

    observations.append(
        Observation(
            "preprocessing.notch_freq",
            STATUS_DIFFERS,
            f"is {_number(notch)} Hz, but the recordings state PowerLineFrequency "
            f"{', '.join(sorted(stated))} Hz.",
        )
    )


def _observe_event_columns(
    config: Any, runs: Sequence[RunFile], observations: List[Observation]
) -> None:
    """For each required logical column, which alias matched and how complete it is."""
    required = [str(group) for group in (config.get("event_columns.required", None) or [])]
    if not required:
        return

    events_files = [run.sidecar("events.tsv") for run in runs]
    events_files = [path for path in events_files if path.exists()]
    if not events_files:
        return

    for group in required:
        key = f"event_columns.{group}"
        aliases = [str(alias) for alias in (config.get(key, None) or [])]
        if not aliases:
            continue

        rows_by_alias: Dict[str, List[Dict[str, str]]] = {}
        for path in events_files:
            rows = _read_tsv(path)
            if not rows:
                continue
            for alias in aliases:
                if alias in rows[0]:
                    rows_by_alias.setdefault(alias, []).extend(rows)
                    break

        if not rows_by_alias:
            observations.append(
                Observation(
                    key,
                    STATUS_ABSENT,
                    f"no events.tsv contains any of its column names: {', '.join(aliases)}.",
                )
            )
            continue

        for alias, rows in sorted(rows_by_alias.items()):
            missing = sum(1 for row in rows if _is_missing(row.get(alias)))
            if missing:
                observations.append(
                    Observation(
                        key,
                        STATUS_DIFFERS,
                        f"resolved to column {alias!r}, which has "
                        f"{_plural(missing, 'missing value')} in "
                        f"{_plural(len(rows), 'row')}.",
                    )
                )
            else:
                observations.append(
                    Observation(
                        key,
                        STATUS_OK,
                        f"resolved to column {alias!r}, complete in "
                        f"{_plural(len(rows), 'row')}.",
                    )
                )


###################################################################
# Entry point
###################################################################


def _is_rest(config: Any) -> bool:
    paradigm = config.get("project.paradigm", None)
    if paradigm is not None:
        return str(paradigm).strip().lower() == "rest"
    return bool(config.get("preprocessing.task_is_rest", False))


def run_preflight(config: Any) -> PreflightReport:
    """Inspect the configured dataset and report what was found, without writing."""
    observations: List[Observation] = []

    bids_root = _observe_roots(config, observations)
    if bids_root is None:
        return PreflightReport(observations=tuple(observations))

    runs = discover_runs(bids_root)
    _observe_inventory(runs, observations)
    if not runs:
        return PreflightReport(observations=tuple(observations))

    runs_of_interest = _observe_configured_task(config, runs, observations)

    # Fixed-length resting-state segments are cut without events, so a recording without
    # an events.tsv is not a finding for that paradigm.
    if not _is_rest(config):
        _observe_events_pairing(runs_of_interest, observations)

    _observe_sampling_frequency(runs_of_interest, observations)
    _observe_channel_layout(runs_of_interest, observations)
    _observe_declared_channels(config, "eeg.ecg_channels", runs_of_interest, observations)
    _observe_declared_channels(config, "eeg.eog_channels", runs_of_interest, observations)
    _observe_trigger_channels(runs_of_interest, observations)
    _observe_line_frequency(config, runs_of_interest, observations)
    _observe_event_columns(config, runs_of_interest, observations)

    return PreflightReport(observations=tuple(observations))


def format_preflight_report(report: PreflightReport) -> str:
    """Render the report as one line per observation, marked by status."""
    marks = {STATUS_OK: "✓", STATUS_DIFFERS: "!", STATUS_ABSENT: "✗"}
    lines = [
        f"  {marks.get(item.status, '?')} {item.key}: {item.message}"
        for item in report.observations
    ]

    counts = {
        status: len(report.by_status(status))
        for status in (STATUS_OK, STATUS_DIFFERS, STATUS_ABSENT)
    }
    summary = (
        f"  {counts[STATUS_OK]} as configured, "
        f"{counts[STATUS_DIFFERS]} varying across runs, "
        f"{counts[STATUS_ABSENT]} not found"
    )
    return "\n".join(
        [
            "=" * 50,
            "       STUDY PREFLIGHT",
            "=" * 50,
            "",
            *lines,
            "",
            "  SUMMARY",
            "  " + "-" * 30,
            summary,
            "",
        ]
    )


__all__ = [
    "Observation",
    "PreflightReport",
    "RunFile",
    "STATUS_ABSENT",
    "STATUS_DIFFERS",
    "STATUS_OK",
    "discover_runs",
    "format_preflight_report",
    "run_preflight",
]
