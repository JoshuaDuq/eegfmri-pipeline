"""Gradient comb and volume-locked residual, as tables.

One function per table the report section used to render. Each returns a DataFrame
carrying the columns that table carried, so the numbers survive the move out of the
MNE report and can be written as TSV instead.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.style import run_label
from studies.pain_study.analysis.gradient.cohort import CohortComb, comb_attenuation
from studies.pain_study.analysis.gradient.comb import CombNotMeasured, CombResidual
from studies.pain_study.analysis.gradient.locked import VolumeLockedAverage

# A non-positive signed excess does not support an amplitude. It is reported as this
# string rather than clipped to zero, because zero would claim a measurement that the
# odd-even floor explicitly declined to make.
UNRESOLVED = "unresolved"


def _locked_stage_measurements(
    locked: VolumeLockedAverage,
) -> tuple[tuple[float, float, float, float | None], ...]:
    values = (
        (
            locked.before_locked_rms_uv,
            locked.before_noise_floor_uv,
            locked.before_excess_power_uv2,
            locked.before_resolved_amplitude_uv,
        ),
        (
            locked.after_locked_rms_uv,
            locked.after_noise_floor_uv,
            locked.after_excess_power_uv2,
            locked.after_resolved_amplitude_uv,
        ),
    )
    if any(value is None for stage in values for value in stage[:3]):
        raise ValueError(
            f"{locked.recording_id} has an incomplete volume-locked noise-floor estimate."
        )
    return tuple(
        (float(observed), float(floor), float(excess), resolved)
        for observed, floor, excess, resolved in values
    )


def _measured(runs: pd.DataFrame, column: str) -> pd.Series:
    # The finite values of one run column, or nothing where it is inapplicable.
    if column not in runs.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(runs[column], errors="coerce")
    return values[np.isfinite(values)]


def _locked_amplitude(runs: pd.DataFrame, stage: str) -> tuple[str | None, float | None]:
    # Participant-median locked amplitude, preserving an unresolved estimate.
    excess = _measured(runs, f"volume_locked_excess_power_{stage}_uv2")
    if excess.empty:
        return None, None
    power = float(excess.median())
    if power <= 0.0:
        return UNRESOLVED, None
    amplitude = float(np.sqrt(power))
    return f"{amplitude:.2f}", amplitude


def comb_frame(combs: Sequence[CombResidual]) -> pd.DataFrame:
    rows = [
        {
            "run": run_label(comb.recording_id),
            "repetition_time_s": comb.timing.repetition_time_s,
            "n_volumes": comb.timing.n_volumes,
            "marker_jitter_ms": comb.timing.interval_jitter_s * 1e3,
            "n_harmonics": int(comb.harmonic_frequencies_hz.size),
            "median_before_excess_db": comb.median_before_excess_db,
            "median_after_excess_db": comb.median_after_excess_db,
            "worst_excess_db": comb.worst_excess_db,
            "worst_harmonic_hz": comb.worst_harmonic_hz,
            "worst_channel": comb.worst_channel,
        }
        for comb in combs
    ]
    return pd.DataFrame(rows)


def locked_frame(averages: Sequence[VolumeLockedAverage]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for locked in averages:
        stages = _locked_stage_measurements(locked)
        correlations = (locked.before_half_correlation, locked.after_half_correlation)
        names = ("Before ICA", "After ICA")
        for name, (observed, floor, excess, resolved), agreement in zip(
            names, stages, correlations, strict=True
        ):
            rows.append(
                {
                    "run": run_label(locked.recording_id),
                    "stage": name,
                    "n_volumes": locked.n_volumes,
                    "observed_locked_rms_uv": observed,
                    "noise_floor_uv": floor,
                    "signed_excess_power_uv2": excess,
                    "half_correlation": (
                        None
                        if agreement is None or not np.isfinite(agreement)
                        else float(agreement)
                    ),
                    "resolved_amplitude_uv": UNRESOLVED if resolved is None else float(resolved),
                }
            )
    return pd.DataFrame(rows)


def declined_frame(declined: Sequence[CombNotMeasured]) -> pd.DataFrame:
    rows = [
        {
            "run": run_label(item.recording_id),
            "n_harmonics": item.n_harmonics if item.n_harmonics else None,
            "n_notched": item.n_notched if item.n_harmonics else None,
            "reason": item.reason,
        }
        for item in declined
    ]
    return pd.DataFrame(rows)


def attenuation_frame(comb: CohortComb) -> pd.DataFrame:
    attenuation = comb_attenuation(comb)
    rows = [
        {
            "subject": subject,
            "before_db": float(np.median(comb.before.per_subject[subject])),
            "after_db": float(np.median(comb.after.per_subject[subject])),
            "removed_db": float(removed),
        }
        for subject, removed in attenuation.items()
    ]
    return pd.DataFrame(rows)


def timing_frame(cohort) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for participant in cohort.participants:
        runs = participant.runs
        if "repetition_time_s" not in runs.columns:
            continue
        rate = _measured(runs, "repetition_time_s")
        jitter = _measured(runs, "volume_jitter_s")
        volumes = _measured(runs, "n_volumes")
        before_text, before_amplitude = _locked_amplitude(runs, "before")
        after_text, after_amplitude = _locked_amplitude(runs, "after")
        observed_after = _measured(runs, "volume_locked_rms_after_uv")
        floor_after = _measured(runs, "volume_locked_floor_after_uv")
        # Paired within the participant: the removed column is one recording measured
        # twice, not a difference between two cohort summaries. Withheld unless both
        # sides are present, because a difference against a missing side is not one.
        removed = (
            before_amplitude - after_amplitude
            if before_amplitude is not None and after_amplitude is not None
            else None
        )
        rows.append(
            {
                "subject": participant.subject,
                "repetition_time_s": None if rate.empty else float(rate.median()),
                "worst_jitter_ms": None if jitter.empty else float(jitter.max() * 1e3),
                "n_volumes": None if volumes.empty else int(volumes.sum()),
                "before_uv": before_text,
                "after_uv": after_text,
                "removed_uv": removed,
                "observed_after_uv": None if observed_after.empty else float(observed_after.median()),
                "floor_after_uv": None if floor_after.empty else float(floor_after.median()),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "UNRESOLVED",
    "attenuation_frame",
    "comb_frame",
    "declined_frame",
    "locked_frame",
    "timing_frame",
]
