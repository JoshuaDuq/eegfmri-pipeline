"""Turning one subject's measured evidence into the sidecar a cohort reads.

Every number here was measured by the stage that owns the section it belongs to. Nothing
in this module estimates anything: it reads properties the measurement objects already
expose and lays them out as tables. That is the whole point -- a cohort figure and the
subject figure beneath it cannot disagree when there is one computation and two readers,
and they can disagree the moment this file starts recomputing.

Two derivations are made here rather than configured, both from the participant's own
evidence, so that a mixed cohort classifies itself:

* volume markers were observed in at least one run, so the recording was made in a
  scanner, even when too few survived to estimate timing;
* at least one run carried task events, so the paradigm has trials to retain.

What is deliberately *not* carried is as considered as what is. The volume-locked
waveform is a subject-level diagnostic that no cohort can pool -- each participant's copy
carries its own averaging noise floor at every point, and with differing repetition times
there is no shared time axis to pool them on. Quantities derived from other stored
quantities are not carried either, because two copies of one measurement can drift apart
and only one of them can then be right.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.analyzer_qc import (
    CardiacResidual,
    MarkerAgreement,
)
from eeg_pipeline.preprocessing.report.rr_intervals import (
    RrIntervals,
)
from eeg_pipeline.preprocessing.report.cohort.sidecar import (
    CHANNEL_COLUMNS,
    COMB_COLUMNS,
    CONDITION_COLUMNS,
    SPECTRUM_COLUMNS,
    AcquisitionContext,
    Paradigm,
    SubjectSidecar,
    run_columns_for,
)
from eeg_pipeline.preprocessing.report.continuity import RunContinuity
from eeg_pipeline.preprocessing.report.settings import (
    DEFAULT_COMPONENT_LABEL_PATTERNS,
)
from eeg_pipeline.preprocessing.report.preservation import PosteriorAlpha
from eeg_pipeline.preprocessing.report.spectra import RunSpectra

#: Stage labels used in every curve table, so one filter works across all of them.
BEFORE = "before"
AFTER = "after"


def acquisition_context_of(continuity: Sequence[RunContinuity]) -> AcquisitionContext:
    """Classify the acquisition from whether volume markers were observed.

    Timing estimation requires enough markers to characterize the interval distribution.
    Acquisition classification does not: even one observed volume marker is direct
    evidence of an in-scanner recording. This is the axis the cohort report refuses to
    pool across, so failed timing estimation must not silently move a participant into
    the out-of-scanner stratum.
    """
    observed = any(run.has_volume_markers for run in continuity)
    return AcquisitionContext.IN_SCANNER if observed else AcquisitionContext.OUT_OF_SCANNER


def paradigm_of(continuity: Sequence[RunContinuity]) -> Paradigm:
    """Classify the paradigm from whether any run carried task events.

    ``RunContinuity.event_onsets`` is already populated for the event rug the continuity
    panel draws beneath its time axis, and is empty for a resting-state recording.
    """
    return Paradigm.TASK if any(run.event_onsets for run in continuity) else Paradigm.REST


def _by_recording(items: Sequence[Any]) -> dict[str, Any]:
    return {item.recording_id: item for item in items}


def _finite(value: Any) -> float:
    """Coerce to a float, mapping anything unmeasured to a missing value.

    A measurement that was attempted and did not resolve belongs in the table as a blank,
    which shrinks the denominator a panel prints. Rendered as a zero it would be a
    measurement, and an unusually good one.
    """
    if value is None:
        return float("nan")
    number = float(value)
    return number if np.isfinite(number) else float("nan")


def run_table(
    *,
    spectra: Sequence[RunSpectra],
    continuity: Sequence[RunContinuity],
    timings: Mapping[str, Any],
    locked_averages: Sequence[Any] = (),
    rr_intervals: Sequence[RrIntervals] = (),
    marker_agreements: Sequence[MarkerAgreement] = (),
    cardiac_residuals: Sequence[CardiacResidual] = (),
    context: AcquisitionContext,
) -> pd.DataFrame:
    """One row per run, holding every run-level scalar a cohort panel reads.

    ``spectra`` is the spine: it has an entry for every run that was measured at all,
    whatever the acquisition supported afterwards.
    """
    continuity_by_run = _by_recording(continuity)
    locked_by_run = _by_recording(locked_averages)
    rr_by_run = _by_recording(rr_intervals)
    agreement_by_run = _by_recording(marker_agreements)
    residual_by_run = _by_recording(cardiac_residuals)

    rows: list[dict[str, Any]] = []
    for run in spectra:
        recording_id = run.recording_id
        quality = continuity_by_run.get(recording_id)
        excursion = None if quality is None else np.asarray(quality.excursion_db, dtype=float)
        row: dict[str, Any] = {
            "run": recording_id,
            "n_channels": int(run.n_channels),
            "duration_s": _finite(None if quality is None else quality.duration_s),
            "flagged_fraction": _finite(None if quality is None else quality.bad_fraction),
            "continuity_median_db": _finite(
                None if excursion is None or excursion.size == 0 else np.median(excursion)
            ),
            "continuity_max_db": _finite(
                None if excursion is None or excursion.size == 0 else np.max(excursion)
            ),
        }

        # The aperiodic background either side of the exclusions. Broadband artifact
        # raises the spectrum roughly uniformly and flattens the slope, and removing too
        # many components takes the background down with it -- so a cohort whose exponent
        # shifted systematically through cleaning lost broadband signal, which no other
        # panel in the report would show. Already fitted for the subject panel.
        for stage, spectrum in ((BEFORE, run.before), (AFTER, run.after)):
            fit = spectrum.aperiodic
            row[f"aperiodic_exponent_{stage}"] = _finite(None if fit is None else fit.exponent)
            row[f"aperiodic_offset_db_{stage}"] = _finite(None if fit is None else fit.offset_db)
            row[f"aperiodic_r_squared_{stage}"] = _finite(None if fit is None else fit.r_squared)

        beats = rr_by_run.get(recording_id)
        row["median_bpm"] = _finite(None if beats is None else beats.median_bpm)
        row["n_beats"] = _finite(None if beats is None else beats.beat_times_s.size)
        # Intervals long enough to be a missed beat rather than a slow one. This is what
        # separates "the heart rate varied" from "the detector lost beats", and the second
        # decides whether the pulse correction had a complete marker train to work from.
        row["beat_dropouts"] = _finite(None if beats is None else beats.dropout_count)

        agreement = agreement_by_run.get(recording_id)
        row["marker_matched_fraction"] = _finite(
            None if agreement is None else agreement.matched_fraction
        )
        row["n_markers"] = _finite(None if agreement is None else agreement.n_markers)
        row["n_detected_beats"] = _finite(None if agreement is None else agreement.n_detected)
        row["n_matched_beats"] = _finite(None if agreement is None else agreement.n_matched)
        # The pair that says what a low matched fraction is made of. A train that sits a
        # fixed distance from the beats the detector found is a delay between two
        # detectors, and the in-scanner ECG produces one routinely: the magnetohydrodynamic
        # deflection is larger than the R wave, so the detector locks onto it a few hundred
        # milliseconds late and reports a marker train that drove a working correction as
        # complete disagreement. A tight lag says that; a broad one says the markers really
        # do not describe the heartbeat. The subject report prints both already, and
        # without them here the cohort table cannot tell the two apart.
        row["marker_median_lag_s"] = _finite(None if agreement is None else agreement.median_lag_s)
        row["marker_lag_iqr_s"] = _finite(None if agreement is None else agreement.lag_iqr_s)

        if context is AcquisitionContext.IN_SCANNER:
            timing = timings.get(recording_id)
            locked = locked_by_run.get(recording_id)
            row["n_volumes"] = _finite(None if timing is None else timing.n_volumes)
            row["repetition_time_s"] = _finite(
                None if timing is None else timing.repetition_time_s
            )
            # Marker jitter smears the comb across neighbouring bins, which lowers every
            # measured excess without the residual itself having changed, so a timing
            # outlier invalidates the gradient section rather than merely annotating it.
            row["volume_jitter_s"] = _finite(None if timing is None else timing.interval_jitter_s)
            for stage in (BEFORE, AFTER):
                suffix = "before" if stage == BEFORE else "after"
                row[f"volume_locked_rms_{suffix}_uv"] = _finite(
                    None if locked is None else getattr(locked, f"{suffix}_locked_rms_uv")
                )
                row[f"volume_locked_floor_{suffix}_uv"] = _finite(
                    None if locked is None else getattr(locked, f"{suffix}_noise_floor_uv")
                )
                row[f"volume_locked_excess_power_{suffix}_uv2"] = _finite(
                    None if locked is None else getattr(locked, f"{suffix}_excess_power_uv2")
                )
                row[f"volume_locked_resolved_{suffix}"] = (
                    None if locked is None else bool(getattr(locked, f"{suffix}_is_resolved"))
                )
            # What the upstream pulse correction left behind, and whether it had a beat
            # train to work from at all. A run with no markers had no subtraction applied,
            # so this is the column that says which runs need re-exporting from Analyzer.
            residual = residual_by_run.get(recording_id)
            row["pulse_marker_count"] = _finite(
                None if residual is None else residual.marker_count
            )
            row["beat_source"] = None if residual is None else residual.beat_source
            row["bcg_residual_uv"] = _finite(
                None if residual is None else residual.residual_uv
            )
            # The floor this residual has to clear, and what is left after it. Recorded
            # together because the amplitude alone is not comparable across runs: it is an
            # average over the beats it was given, and the floor grows as that count falls.
            # Named to match the volume-locked gradient columns, which measure the same
            # kind of quantity the same way.
            row["bcg_noise_floor_uv"] = _finite(
                None if residual is None else residual.noise_floor_uv
            )
            row["bcg_excess_power_uv2"] = _finite(
                None if residual is None else residual.excess_power_uv2
            )
            row["bcg_resolved"] = None if residual is None else residual.is_resolved
            row["bcg_n_beats"] = _finite(None if residual is None else residual.n_beats)
            # The residual is an average over the beats it was given, so it describes only
            # the share of the run they cover. Recorded beside it because a small residual
            # over a quarter of a run is not a corrected run, and the sidecar is what
            # cross-run analyses read.
            row["bcg_beat_train_coverage"] = _finite(
                None if residual is None else residual.beat_train_coverage
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    missing = [name for name in run_columns_for(context) if name not in frame.columns]
    if missing:
        raise ValueError(
            f"The run table is missing {', '.join(missing)} for a "
            f"{context.value} acquisition."
        )
    return frame


def spectrum_curves(spectra: Sequence[RunSpectra]) -> pd.DataFrame:
    """The across-channel median and worst channel per run, stage and frequency.

    The worst channel travels beside the median because gradient residual is focal: it
    concentrates in the sensors with the largest lead loops, so a montage median can sit
    near zero while individual sensors are unusable.
    """
    rows: list[dict[str, Any]] = []
    for run in spectra:
        frequencies = np.asarray(run.frequencies, dtype=float)
        for stage, spectrum in ((BEFORE, run.before), (AFTER, run.after)):
            rows.append(
                pd.DataFrame(
                    {
                        "run": run.recording_id,
                        "stage": stage,
                        "freq_hz": frequencies,
                        "median_db": np.asarray(spectrum.median_db, dtype=float),
                        "max_db": np.asarray(spectrum.max_db, dtype=float),
                    }
                )
            )
    if not rows:
        return pd.DataFrame({name: pd.Series(dtype="object") for name in SPECTRUM_COLUMNS})
    return pd.concat(rows, ignore_index=True)


def comb_curves(combs: Sequence[Any]) -> pd.DataFrame:
    """Comb excess per harmonic, reduced across channels, before and after ICA.

    Carries the harmonic *index* as well as its frequency. Harmonics sit at multiples of
    the volume rate, so participants scanned at different repetition times have their
    harmonics at different frequencies; pooling by frequency would put one participant's
    third harmonic in the same bin as another's fourth. The index is the axis a mixed
    cohort can be pooled on, and the frequency is what a single-rate cohort draws against.
    """
    rows: list[dict[str, Any]] = []
    for comb in combs:
        harmonics = np.asarray(comb.harmonic_frequencies_hz, dtype=float)
        if harmonics.size == 0:
            continue
        fundamental = comb.timing.fundamental_hz
        before = np.asarray(comb.before_excess_db, dtype=float)
        after = np.asarray(comb.after_excess_db, dtype=float)
        # Carried so the cohort can exclude what the subject panel already excludes. A
        # harmonic landing in the notch stopband measures the depth of the filter, not the
        # residual: on a 0.9 s repetition time the 54th harmonic sits at 60 Hz and reads
        # about -25 dB, which without this column becomes the deepest excursion in the
        # cohort figure and sets its whole vertical scale.
        notched = np.asarray(comb.notched, dtype=bool)
        if notched.size != harmonics.size:
            notched = np.zeros(harmonics.size, dtype=bool)
        rows.append(
            pd.DataFrame(
                {
                    "run": comb.recording_id,
                    "harmonic_index": np.rint(harmonics / fundamental).astype(int),
                    "harmonic_hz": harmonics,
                    "notched": notched,
                    "before_excess_db_median": np.median(before, axis=0),
                    "before_excess_db_max": np.max(before, axis=0),
                    "after_excess_db_median": np.median(after, axis=0),
                    "after_excess_db_max": np.max(after, axis=0),
                }
            )
        )
    if not rows:
        return pd.DataFrame({name: pd.Series(dtype="object") for name in COMB_COLUMNS})
    return pd.concat(rows, ignore_index=True)


#: Detector descriptions mapped to the class a cohort counts them under.
#:
#: ``status_description`` is prose the detectors write -- "Auto-detected eye blink
#: (MNE-ICALabel)", "Auto-detected ECG artifact (MNE)" -- rather than a coded field, so
#: this matches substrings. Ordered, because "channel noise" and "line noise" both contain
#: "noise" and the more specific reading has to win.
#:
#: Reading prose is fragile and is done here, once, at write time, rather than in the
#: cohort command: a description this table does not recognise lands in ``other`` and is
#: still counted, which is the failure mode worth having. A site whose detectors write
#: different prose sets ``report.acquisition.component_label_patterns`` rather than
#: accepting a cohort where every exclusion is unclassified.
_LABEL_PATTERNS: tuple[tuple[str, str], ...] = DEFAULT_COMPONENT_LABEL_PATTERNS

#: Classes a cohort reports, in reading order. ``other`` collects descriptions no pattern
#: matched; ``unrecorded`` counts exclusions no detector explained.
COMPONENT_LABEL_CLASSES = ("eye", "heart", "muscle", "line", "channel", "other", "unrecorded")


def component_label_counts(
    components: pd.DataFrame | None,
    *,
    label_patterns: Sequence[tuple[str, str]] = _LABEL_PATTERNS,
) -> dict[str, int]:
    """Count the excluded components under each detector class.

    A cohort panel built on these separates "twelve components removed" from "twelve
    components removed, nine of them muscle", which are different recordings with the same
    headline number.
    """
    counts = {name: 0 for name in COMPONENT_LABEL_CLASSES}
    if components is None or components.empty:
        return counts
    if "status" not in components.columns:
        raise ValueError("A component table needs a status column to count exclusions.")
    excluded = components[components["status"].astype(str).str.strip() == "bad"]
    descriptions = excluded.get("status_description")
    for index in range(len(excluded)):
        text = "" if descriptions is None else str(descriptions.iloc[index]).strip().lower()
        if not text:
            counts["unrecorded"] += 1
            continue
        for pattern, label in label_patterns:
            if pattern in text:
                counts[label] += 1
                break
        else:
            counts["other"] += 1
    return counts


def channel_table(
    *,
    positions: Mapping[str, Sequence[float]] | None,
    bad_by_run: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """One row per channel: where it sat, and how many runs it was marked bad in.

    ``positions`` are the sensor locations the recording itself carries, in metres, so a
    cohort topography places every electrode where it actually was rather than where a
    montage name suggests it should have been.

    A count per channel rather than a union, because a channel bad in one run of six and
    one bad in all six are different facts about a cap, and a union reports them
    identically.
    """
    if not positions:
        return pd.DataFrame({name: pd.Series(dtype="object") for name in CHANNEL_COLUMNS})
    counts: dict[str, int] = {name: 0 for name in positions}
    for names in (bad_by_run or {}).values():
        for name in names:
            if name in counts:
                counts[name] += 1
    rows = []
    for name, location in positions.items():
        coordinates = np.asarray(location, dtype=float).ravel()[:3]
        if coordinates.size < 3 or not np.all(np.isfinite(coordinates)):
            continue
        rows.append(
            {
                "channel": name,
                "x": float(coordinates[0]),
                "y": float(coordinates[1]),
                "z": float(coordinates[2]),
                "n_runs_bad": counts[name],
            }
        )
    if not rows:
        return pd.DataFrame({name: pd.Series(dtype="object") for name in CHANNEL_COLUMNS})
    return pd.DataFrame(rows)


def condition_table(
    *,
    total_by_condition: Mapping[str, Any] | None = None,
    kept_by_condition: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Trials presented and trials retained, per experimental condition.

    Both counts are carried because neither is interpretable alone. Forty retained trials
    in one condition and a hundred in another is a confounded contrast if both were
    presented equally often, and a balanced design if they were not -- and the retained
    count cannot tell those apart.

    A condition named by only one of the two mappings still gets a row, with the count the
    other mapping did not supply left missing. A condition that was presented and lost
    every trial is exactly the case a cohort panel most needs to see, and dropping it for
    want of a retained count would hide it.
    """
    totals = {str(name): value for name, value in dict(total_by_condition or {}).items()}
    kept = {str(name): value for name, value in dict(kept_by_condition or {}).items()}
    names = sorted(set(totals) | set(kept))
    if not names:
        return pd.DataFrame({name: pd.Series(dtype="object") for name in CONDITION_COLUMNS})
    return pd.DataFrame(
        [
            {
                "condition": name,
                "n_total": _finite(totals.get(name)),
                "n_kept": _finite(kept.get(name)),
            }
            for name in names
        ]
    )


def pool_alpha_runs(measured: Sequence[PosteriorAlpha]) -> PosteriorAlpha | None:
    """Reduce a participant's per-run alpha measurements to one.

    An unweighted median across runs, matching every other run-to-participant reduction of
    a non-rate quantity here: each run is an independent estimate of the same rhythm, and a
    longer run estimates it more precisely rather than more correctly.

    The peak frequency is medianed only over the runs where the peak was resolvable. A run
    with no rhythm contributes the argmax of its own noise, and letting that into the
    median would drag the participant's peak toward the middle of the band -- which is
    exactly the fabricated structure the resolvability test exists to keep out.
    """
    entries = [entry for entry in measured if entry is not None]
    if not entries:
        return None
    resolvable = [entry for entry in entries if entry.is_resolvable()]
    peak_source = resolvable or entries
    return PosteriorAlpha(
        channel_names=entries[0].channel_names,
        frequencies_hz=entries[0].frequencies_hz,
        power_db=np.median(
            np.vstack([np.asarray(entry.power_db, dtype=float) for entry in entries]), axis=0
        ),
        # Pooled the same way as the spectrum above, so the drawn background stays the
        # background of the drawn spectrum rather than one run's line under all of them.
        background_db=np.median(
            np.vstack([np.asarray(entry.background_db, dtype=float) for entry in entries]),
            axis=0,
        ),
        peak_frequency_hz=float(
            np.median([entry.peak_frequency_hz for entry in peak_source])
        ),
        prominence_db=float(np.median([entry.prominence_db for entry in entries])),
        band_hz=entries[0].band_hz,
        background_residual_db=float(
            np.median([entry.background_residual_db for entry in entries])
        ),
        # The runs of one participant are measured over the same band at the same
        # resolution, so the search width is theirs rather than an average of differing
        # ones. Carried through so the pooled measurement is scored against the same
        # multiplicity correction each run was.
        n_search_bins=int(entries[0].n_search_bins),
        # Interior only where every run agreed it was: one run whose maximum sat on the
        # band edge is one run that did not establish a peak, and pooling cannot establish
        # one for it.
        is_interior=all(entry.is_interior for entry in entries),
        runner_up_frequency_hz=float(
            np.median([entry.runner_up_frequency_hz for entry in peak_source])
        ),
        # The *smallest* gap across runs, not the median. A frequency that was contested in
        # any run is a frequency that could have come out differently, and averaging that
        # away is exactly the reassurance this field exists to withhold.
        runner_up_gap_db=float(min(entry.runner_up_gap_db for entry in entries)),
    )


def alpha_measurements(alpha: Mapping[str, PosteriorAlpha]) -> dict[str, Any]:
    """Peak scalars per stage, with the frequency withheld where there is no peak.

    A peak frequency is only recorded when the prominence clears the spectrum's own
    roughness. Below that the argmax of a band is the argmax of noise, and recording it
    would put a plausible-looking number into a cohort histogram that no rhythm produced.
    The resolvability flag is recorded either way, because how many participants had no
    resolvable rhythm is itself a cohort measurement.

    Whether the frequency was contested travels with it. A band holding two comparable
    bumps has no single peak frequency, and the one the argmax returns can move several
    hertz on a change too small to matter anywhere else; a cohort histogram that does not
    know which of its entries are bistable reads sharper than the evidence supports.
    """
    measurements: dict[str, Any] = {}
    for stage, measured in alpha.items():
        resolvable = measured.is_resolvable()
        measurements[f"alpha_prominence_db_{stage}"] = float(measured.prominence_db)
        measurements[f"alpha_background_residual_db_{stage}"] = float(
            measured.background_residual_db
        )
        measurements[f"alpha_peak_resolvable_{stage}"] = bool(resolvable)
        if resolvable:
            measurements[f"alpha_peak_frequency_hz_{stage}"] = float(measured.peak_frequency_hz)
            measurements[f"alpha_peak_contested_{stage}"] = bool(measured.peak_is_contested)
            if np.isfinite(measured.runner_up_frequency_hz):
                measurements[f"alpha_runner_up_hz_{stage}"] = float(
                    measured.runner_up_frequency_hz
                )
                measurements[f"alpha_runner_up_gap_db_{stage}"] = float(
                    measured.runner_up_gap_db
                )
    return measurements


def build_subject_sidecar(
    *,
    subject: str,
    task: str,
    spectra: Sequence[RunSpectra],
    continuity: Sequence[RunContinuity],
    timings: Mapping[str, Any],
    locked_averages: Sequence[Any] = (),
    combs: Sequence[Any] = (),
    rr_intervals: Sequence[RrIntervals] = (),
    marker_agreements: Sequence[MarkerAgreement] = (),
    cardiac_residuals: Sequence[CardiacResidual] = (),
    alpha: Mapping[str, PosteriorAlpha] | None = None,
    components: pd.DataFrame | None = None,
    channel_positions: Mapping[str, Sequence[float]] | None = None,
    bad_channels_by_run: Mapping[str, Sequence[str]] | None = None,
    trials_by_condition: Mapping[str, Any] | None = None,
    retained_by_condition: Mapping[str, Any] | None = None,
    measurements: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    versions: Mapping[str, str] | None = None,
    acquisition_date: str | None = None,
    label_patterns: Sequence[tuple[str, str]] = _LABEL_PATTERNS,
) -> SubjectSidecar:
    """Assemble one participant's sidecar from what the measuring pass produced."""
    if not spectra:
        raise ValueError(
            f"sub-{subject} has no measured runs, so there is nothing for a cohort to read."
        )
    context = acquisition_context_of(continuity)
    paradigm = paradigm_of(continuity)
    alpha = dict(alpha or {})
    combined: dict[str, Any] = dict(measurements or {})
    combined.update(alpha_measurements(alpha))
    if components is not None:
        for label, count in component_label_counts(
            components, label_patterns=label_patterns
        ).items():
            combined[f"n_excluded_{label}"] = count

    return SubjectSidecar(
        subject=str(subject),
        task=str(task),
        context=context,
        paradigm=paradigm,
        measurements=combined,
        settings=dict(settings or {}),
        versions=dict(versions or {}),
        acquisition_date=acquisition_date,
        runs=run_table(
            spectra=spectra,
            continuity=continuity,
            timings=timings,
            locked_averages=locked_averages,
            rr_intervals=rr_intervals,
            marker_agreements=marker_agreements,
            cardiac_residuals=cardiac_residuals,
            context=context,
        ),
        spectrum_curves=spectrum_curves(spectra),
        comb_curves=comb_curves(combs),
        channels=channel_table(
            positions=channel_positions, bad_by_run=bad_channels_by_run
        ),
        # Conditions belong to a paradigm that has them. A resting-state recording that
        # arrived with condition counts was misclassified somewhere upstream, and carrying
        # them anyway would put a rest participant into a task-only panel.
        conditions=(
            condition_table(
                total_by_condition=trials_by_condition,
                kept_by_condition=retained_by_condition,
            )
            if paradigm is Paradigm.TASK
            else condition_table()
        ),
    )


__all__ = [
    "AFTER",
    "BEFORE",
    "COMPONENT_LABEL_CLASSES",
    "acquisition_context_of",
    "alpha_measurements",
    "build_subject_sidecar",
    "channel_table",
    "comb_curves",
    "component_label_counts",
    "condition_table",
    "pool_alpha_runs",
    "paradigm_of",
    "run_table",
    "spectrum_curves",
]
