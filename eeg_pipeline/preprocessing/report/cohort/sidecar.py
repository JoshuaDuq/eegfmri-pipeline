"""The per-subject QC sidecar: what one participant contributes to a cohort.

The measurements a cohort needs already exist. ``run_evidence`` reads each run once,
applies the ICA exclusions, and measures the spectra, the gradient comb, the volume-locked
residual, the time-resolved quality and the beat detection in a single pass -- by a wide
margin the most expensive thing the pipeline does. Today those results are plotted and
dropped, so a cohort document could only be built by paying that cost again, per
participant, from a gigabyte of filtered raw each.

This module is the alternative: the same numbers, written down at the moment they are
already in memory. Two properties follow. The cohort figure is provably the same
measurement as the subject figure beneath it, because there is one computation and two
readers rather than two computations that must be kept agreeing. And the cohort command
becomes cheap enough to re-run whenever a participant is added.

The sidecar deliberately holds what a cohort can legitimately aggregate, not everything
that was measured. ``RunContinuity`` carries a time-by-channel matrix that no cohort can
meaningfully average, so what is written is its reduction to per-run scalars. Reducing at
write time rather than read time is what stops the cohort command from inventing an
aggregation the measuring stage never sanctioned.

Plain tables and JSON, with no MNE import, so the schema can be exercised without a
recording. Turning the measured objects into this payload is the writer's job, at the call
site where those objects live.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

#: Layout version of the sidecar. Bumped when a column changes meaning, not when a stage
#: adds one: a reader that meets an unknown column can ignore it, but one that meets a
#: familiar column holding something else cannot.
#:
#: 3: ``bcg_residual_uv`` changed from the peak-to-peak of the across-channel RMS of the
#: beat-locked average to the RMS of that average, and gained the floor columns beside it.
#: The old number carried an averaging floor nobody could see, so the same run reported
#: 0.14 uV over 493 beats and 2.78 uV over 59 of them. A cohort that read a version-2
#: sidecar as though it were this one would compare two different quantities across
#: participants, which is exactly what this version number exists to prevent.
SCHEMA_VERSION = 3

#: Suffix of the report the sidecar belongs to, mirroring :mod:`build_record`.
_REPORT_SUFFIX = "_report.h5"

SUBJECT_SUFFIX = "_desc-qcsubject.json"
RUNS_SUFFIX = "_desc-qcruns.tsv"
SPECTRUM_SUFFIX = "_desc-qcspectrum_curves.tsv"
COMB_SUFFIX = "_desc-qccomb_curves.tsv"
CHANNELS_SUFFIX = "_desc-qcchannels.tsv"
CONDITIONS_SUFFIX = "_desc-qcconditions.tsv"

#: Columns every table must carry. A sidecar missing one is raised rather than silently
#: contributing a column of nothing to a cohort figure.
#:
#: The distinction that governs this list, and :data:`SCANNER_RUN_COLUMNS` below, is
#: between a column and a value. A column is required when the recorded context implies
#: the measurement was attempted, and its absence is therefore a fault in the writer. The
#: value in it may still be missing, because a measurement that was attempted and did not
#: resolve is an ordinary outcome -- ``compute_comb_residual`` declines when the frequency
#: resolution cannot separate the comb from its background, which is a property of the
#: volume rate rather than of the data. Missing values shrink a panel's denominator, which
#: the panel prints. A missing column would shrink it silently.
RUN_COLUMNS = (
    "run",
    "n_channels",
    "duration_s",
    # Continuity is measured for every run of every acquisition -- the estimator has no
    # declining path -- so its reductions are required unconditionally.
    "flagged_fraction",
    "continuity_median_db",
    "continuity_max_db",
)

#: Additionally required of an in-scanner acquisition.
#:
#: The context is derived from the presence of volume markers, so a sidecar that claims to
#: be in-scanner claims that volume timing was measurable. Timing is then always present;
#: the volume-locked amplitude may be absent in value where too few complete epochs
#: survived, but the column proves the writer attempted it.
#:
#: Observed locked RMS, its estimated floor, and the signed difference in power are all
#: retained. A negative difference is a censored measurement and cannot be reconstructed
#: from a zero-clipped amplitude.
SCANNER_RUN_COLUMNS = (
    "n_volumes",
    "repetition_time_s",
    "volume_jitter_s",
    "volume_locked_rms_before_uv",
    "volume_locked_floor_before_uv",
    "volume_locked_excess_power_before_uv2",
    "volume_locked_resolved_before",
    "volume_locked_rms_after_uv",
    "volume_locked_floor_after_uv",
    "volume_locked_excess_power_after_uv2",
    "volume_locked_resolved_after",
    "median_bpm",
    "n_beats",
    "beat_dropouts",
    "marker_matched_fraction",
    "marker_median_lag_s",
    "marker_lag_iqr_s",
    "n_markers",
    "n_detected_beats",
    "n_matched_beats",
    "pulse_marker_count",
    "beat_source",
    "bcg_residual_uv",
    "bcg_beat_train_coverage",
    # Required, not optional. The residual amplitude above is an average over the beats it
    # was given, so its floor moves with that count and the bare number is not comparable
    # between runs; a sidecar that carried the amplitude without the floor would let a
    # cohort rank runs by beat-detection quality and call it artifact. Same reasoning, and
    # the same estimator, as the volume-locked gradient columns.
    "bcg_noise_floor_uv",
    "bcg_excess_power_uv2",
    "bcg_resolved",
    "bcg_n_beats",
)

#: The across-channel median and the worst channel, per run and stage.
#:
#: The spread quantiles the subject panel draws are a within-participant quantity that no
#: cohort panel pools, and a required column nothing reads is dead weight in a contract.
#: The worst channel is kept because gradient residual is focal: a montage median can sit
#: near zero while individual sensors are unusable.
SPECTRUM_COLUMNS = ("run", "stage", "freq_hz", "median_db", "max_db")
#: One row per EEG channel: where it sat on the head, and how many runs it was bad in.
#:
#: The positions come from the recording rather than from a montage name, so a cohort
#: topography places every electrode where it actually was. Guessing a standard montage
#: would place them plausibly and, when the guess was wrong, silently wrongly -- which is
#: the one failure a figure like that must not have.
CHANNEL_COLUMNS = ("channel", "x", "y", "z", "n_runs_bad")

#: ``notched`` marks harmonics that landed in a notch stopband. Required rather than
#: optional: a comb table without it cannot tell a harmonic the correction removed from one
#: the notch filter did, and the second reads as a far larger improvement than any
#: correction achieves.
COMB_COLUMNS = (
    "run",
    "harmonic_index",
    "harmonic_hz",
    "notched",
    "before_excess_db_median",
    "before_excess_db_max",
    "after_excess_db_median",
    "after_excess_db_max",
)

#: One row per experimental condition: trials presented, and trials that survived rejection.
#:
#: Both counts travel together because neither answers the question on its own. A condition
#: with forty retained trials is well sampled or badly decimated depending on how many were
#: presented, and differential rejection across conditions confounds a contrast rather than
#: merely underpowering it -- which a total-epoch count cannot show. Written only for a task
#: acquisition, since a resting-state recording has no conditions to count.
CONDITION_COLUMNS = ("condition", "n_total", "n_kept")


class AcquisitionContext(Enum):
    """Whether the recording was made inside a scanner.

    Derived per participant from its own evidence -- the presence of volume markers -- and
    never configured, so a mixed cohort classifies itself. This is the axis the report
    refuses to pool across: a variance-removed figure that is unremarkable inside a bore is
    alarming outside one, and a median over both describes neither.
    """

    IN_SCANNER = "in_scanner"
    OUT_OF_SCANNER = "out_of_scanner"


class Paradigm(Enum):
    """Whether the recording has trials to retain.

    Decides whether the epoch-rejection, events and evoked-reliability sections exist at
    all for this participant.
    """

    TASK = "task"
    REST = "rest"


@dataclass(frozen=True)
class SidecarPaths:
    """Where one participant's sidecar lives.

    Derived from the report's own name rather than configured, so a file move cannot
    separate a sidecar from the report it describes or point a stage at the wrong one.
    """

    subject_json: Path
    runs: Path
    spectrum_curves: Path
    comb_curves: Path
    channels: Path
    conditions: Path

    @property
    def required(self) -> tuple[Path, ...]:
        """Files every participant has, whatever the acquisition."""
        return (self.subject_json, self.runs, self.spectrum_curves)

    @property
    def optional(self) -> tuple[Path, ...]:
        """Files only some acquisitions produce."""
        return (self.comb_curves, self.channels, self.conditions)


@dataclass(frozen=True)
class SubjectSidecar:
    """One participant's contribution to the cohort, as written and as read."""

    subject: str
    task: str
    context: AcquisitionContext
    paradigm: Paradigm
    #: Subject-level scalars: variance removed, component counts, rank, and the rest of
    #: what the build record already holds, copied so a cohort read needs one file.
    measurements: Mapping[str, Any] = field(default_factory=dict)
    #: Report settings that shape the numbers, so a homogeneity panel can compare them.
    settings: Mapping[str, Any] = field(default_factory=dict)
    #: Package versions per stage, mirroring the build record.
    versions: Mapping[str, str] = field(default_factory=dict)
    #: Acquisition date, which indexes cap ageing and electrode wear. Distinct from the
    #: processing date in the build record, which indexes pipeline change; conflating the
    #: two makes a drift panel unreadable.
    acquisition_date: str | None = None
    written_at: str | None = None
    schema_version: int = SCHEMA_VERSION
    runs: pd.DataFrame = field(default_factory=lambda: _empty(RUN_COLUMNS))
    spectrum_curves: pd.DataFrame = field(default_factory=lambda: _empty(SPECTRUM_COLUMNS))
    comb_curves: pd.DataFrame = field(default_factory=lambda: _empty(COMB_COLUMNS))
    channels: pd.DataFrame = field(default_factory=lambda: _empty(CHANNEL_COLUMNS))
    conditions: pd.DataFrame = field(default_factory=lambda: _empty(CONDITION_COLUMNS))

    @property
    def n_runs(self) -> int:
        return int(len(self.runs))

    @property
    def has_condition_evidence(self) -> bool:
        """Whether this participant can contribute to the per-condition panels.

        Narrower than :attr:`paradigm`, for the same reason :attr:`has_comb_evidence` is
        narrower than :attr:`context`. A task recording whose events table did not name a
        condition column has trials but no conditions to break them down by, so "did this
        have trials" and "can this join the per-condition figure" are two questions.
        """
        return not self.conditions.empty

    @property
    def has_comb_evidence(self) -> bool:
        """Whether this participant can contribute to the harmonic-comb panel.

        Narrower than :attr:`context`, and deliberately so. ``compute_comb_residual``
        declines when the frequency resolution cannot separate the comb from its
        background, which is a property of the volume rate and the run length rather than
        of the data quality. A participant can therefore be in a scanner, contribute a
        volume-locked amplitude, and still have no comb to pool -- so "was this in a
        scanner" and "can this join the comb figure" are two questions and get two
        answers.
        """
        return not self.comb_curves.empty


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    """An absent table with its columns, so a reader need not special-case it.

    An acquisition that produced no gradient evidence is not a malformed sidecar, it is an
    EEG-only recording. Returning the shape rather than ``None`` means every consumer
    filters an empty frame instead of testing for one.
    """
    return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})


def sidecar_paths(report_path: Path | str) -> SidecarPaths:
    """Return the sidecar that belongs to a report."""
    path = Path(report_path)
    stem = path.name
    for suffix in (_REPORT_SUFFIX, ".h5", ".html"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return SidecarPaths(
        subject_json=path.with_name(f"{stem}{SUBJECT_SUFFIX}"),
        runs=path.with_name(f"{stem}{RUNS_SUFFIX}"),
        spectrum_curves=path.with_name(f"{stem}{SPECTRUM_SUFFIX}"),
        comb_curves=path.with_name(f"{stem}{COMB_SUFFIX}"),
        channels=path.with_name(f"{stem}{CHANNELS_SUFFIX}"),
        conditions=path.with_name(f"{stem}{CONDITIONS_SUFFIX}"),
    )


def has_sidecar(report_path: Path | str) -> bool:
    """Whether a participant can be aggregated at all.

    A participant without one is not an error: it is a participant whose report predates
    this feature, or whose run failed. The cohort report lists it as not aggregated rather
    than dropping it, which is why this is a question and not an exception.
    """
    return all(path.is_file() for path in sidecar_paths(report_path).required)


def run_columns_for(context: AcquisitionContext) -> tuple[str, ...]:
    """Columns the run table must carry for a participant in this context."""
    if context is AcquisitionContext.IN_SCANNER:
        return RUN_COLUMNS + SCANNER_RUN_COLUMNS
    return RUN_COLUMNS


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    source: str,
    context: AcquisitionContext | None = None,
) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if not missing:
        return
    scanner_only = [name for name in missing if name in SCANNER_RUN_COLUMNS]
    detail = ""
    if context is AcquisitionContext.IN_SCANNER and scanner_only:
        detail = (
            " These are required because the sidecar records an in-scanner acquisition, "
            "which means volume timing was measurable and the gradient measurements were "
            "attempted. Record the column with a missing value if one did not resolve."
        )
    raise ValueError(f"{source} is missing the columns {', '.join(missing)}.{detail}")


def _require_no_infinite_values(frame: pd.DataFrame, *, source: str) -> None:
    """Reject infinities while retaining NaN as the missing-value marker."""
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if np.isinf(numeric).any():
            raise ValueError(f"{source} has a non-finite value in {column}.")


def _resolved_value(value: Any, *, column: str) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0"}:
        return True
    if text in {"false", "0", "0.0"}:
        return False
    raise ValueError(f"The run table has an invalid boolean in {column}: {value!r}.")


def _require_locked_power_consistency(frame: pd.DataFrame) -> None:
    """Ensure resolution flags preserve the signed-power meaning."""
    for stage in ("before", "after"):
        power_column = f"volume_locked_excess_power_{stage}_uv2"
        resolved_column = f"volume_locked_resolved_{stage}"
        if power_column not in frame or resolved_column not in frame:
            continue
        powers = pd.to_numeric(frame[power_column], errors="coerce")
        for index, (power, resolved) in enumerate(
            zip(powers, frame[resolved_column], strict=True)
        ):
            flag = _resolved_value(resolved, column=resolved_column)
            if pd.isna(power) or flag is None:
                continue
            if flag is not bool(power > 0.0):
                raise ValueError(
                    f"The run table row {index} has {resolved_column}={flag}, but "
                    f"{power_column}={power:g}. The flag must equal signed power > 0."
                )


def _read_table(
    path: Path,
    columns: tuple[str, ...],
    *,
    required: bool,
    context: AcquisitionContext | None = None,
) -> pd.DataFrame:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"The QC sidecar table {path.name} does not exist.")
        return _empty(columns)
    frame = pd.read_csv(path, sep="\t")
    _require_columns(frame, columns, source=path.name, context=context)
    _require_no_infinite_values(frame, source=path.name)
    return frame


def write_sidecar(report_path: Path | str, sidecar: SubjectSidecar) -> SidecarPaths:
    """Write one participant's sidecar beside its report.

    Tables that hold nothing are not written. An EEG-only recording has no gradient comb,
    and a zero-row file claiming the column names of one would be indistinguishable on disk
    from a scanner recording whose measurement failed.
    """
    # Validated before anything reaches disk. A writer that omits a column its own context
    # requires would otherwise produce a sidecar that reads back fine on the machine that
    # wrote it and fails weeks later, in a cohort run, naming a participant rather than the
    # stage at fault.
    _require_columns(
        sidecar.runs,
        run_columns_for(sidecar.context),
        source=f"The run table for sub-{sidecar.subject}",
        context=sidecar.context,
    )
    for frame, source in (
        (sidecar.runs, "The run table"),
        (sidecar.spectrum_curves, "The spectrum table"),
        (sidecar.comb_curves, "The comb table"),
        (sidecar.channels, "The channel table"),
        (sidecar.conditions, "The condition table"),
    ):
        _require_no_infinite_values(frame, source=source)
    _require_locked_power_consistency(sidecar.runs)

    document = {
        "schema_version": int(sidecar.schema_version),
        "subject": str(sidecar.subject),
        "task": str(sidecar.task),
        "context": sidecar.context.value,
        "paradigm": sidecar.paradigm.value,
        "acquisition_date": sidecar.acquisition_date,
        "written_at": sidecar.written_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "measurements": dict(sidecar.measurements),
        "settings": dict(sidecar.settings),
        "versions": dict(sidecar.versions),
    }
    try:
        serialized = json.dumps(document, indent=1, sort_keys=True, allow_nan=False)
    except ValueError as error:
        raise ValueError(f"The QC subject metadata is not valid finite JSON: {error}") from error

    paths = sidecar_paths(report_path)
    paths.subject_json.parent.mkdir(parents=True, exist_ok=True)
    paths.subject_json.write_text(serialized, encoding="utf-8")

    for frame, path, required in (
        (sidecar.runs, paths.runs, True),
        (sidecar.spectrum_curves, paths.spectrum_curves, True),
        (sidecar.comb_curves, paths.comb_curves, False),
        (sidecar.channels, paths.channels, False),
        (sidecar.conditions, paths.conditions, False),
    ):
        if frame.empty and not required:
            path.unlink(missing_ok=True)
            continue
        frame.to_csv(path, sep="\t", index=False)
    return paths


def read_sidecar(report_path: Path | str) -> SubjectSidecar:
    """Read one participant's sidecar.

    Every failure here is raised rather than absorbed. A sidecar that cannot be parsed, or
    that describes a different participant, or that was written under a layout this code
    does not understand, would otherwise contribute a wrong row to a cohort figure that
    looks exactly like a right one.
    """
    paths = sidecar_paths(report_path)
    if not paths.subject_json.is_file():
        raise FileNotFoundError(f"No QC sidecar at {paths.subject_json}.")
    try:
        document = json.loads(paths.subject_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{paths.subject_json.name} is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{paths.subject_json.name} does not hold a sidecar document.")

    version = int(document.get("schema_version", -1))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{paths.subject_json.name} was written under sidecar schema {version}, and "
            f"this code reads schema {SCHEMA_VERSION}. Rebuild the subject report rather "
            f"than pooling columns whose meaning may have changed."
        )

    subject = str(document.get("subject", ""))
    expected = _subject_from_path(paths.subject_json)
    if expected is not None and subject != expected:
        raise ValueError(
            f"{paths.subject_json.name} describes participant {subject!r}, which does not "
            f"match the {expected!r} in its own path. A sidecar has been moved or renamed."
        )

    context = AcquisitionContext(document["context"])
    runs = _read_table(paths.runs, run_columns_for(context), required=True, context=context)
    _require_locked_power_consistency(runs)
    return SubjectSidecar(
        subject=subject,
        task=str(document.get("task", "")),
        context=context,
        paradigm=Paradigm(document["paradigm"]),
        measurements=dict(document.get("measurements") or {}),
        settings=dict(document.get("settings") or {}),
        versions=dict(document.get("versions") or {}),
        acquisition_date=document.get("acquisition_date"),
        written_at=document.get("written_at"),
        schema_version=version,
        runs=runs,
        spectrum_curves=_read_table(paths.spectrum_curves, SPECTRUM_COLUMNS, required=True),
        comb_curves=_read_table(paths.comb_curves, COMB_COLUMNS, required=False),
        channels=_read_table(paths.channels, CHANNEL_COLUMNS, required=False),
        conditions=_read_table(paths.conditions, CONDITION_COLUMNS, required=False),
    )


def _subject_from_path(path: Path) -> str | None:
    """Recover the participant label a BIDS derivative path encodes, if it encodes one."""
    for part in path.name.split("_"):
        if part.startswith("sub-"):
            return part[len("sub-") :]
    return None


__all__ = [
    "CHANNEL_COLUMNS",
    "COMB_COLUMNS",
    "CONDITION_COLUMNS",
    "RUN_COLUMNS",
    "SCANNER_RUN_COLUMNS",
    "SCHEMA_VERSION",
    "SPECTRUM_COLUMNS",
    "AcquisitionContext",
    "Paradigm",
    "SidecarPaths",
    "SubjectSidecar",
    "has_sidecar",
    "read_sidecar",
    "run_columns_for",
    "sidecar_paths",
    "write_sidecar",
]
