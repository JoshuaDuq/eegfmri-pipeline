"""Fill Analyzer's pulse-marker gaps: recover beats, correct only there, and score it.

Analyzer's correction is kept wherever it marked a beat, because it measurably beats ours
on both arms there (0.16% residual against our 2.03%, alpha retained 0.54 against 0.34).
What it never marked is untouched artifact -- 6,954 s across the cohort, up to 77% of a
single run -- and that is all this stage changes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from studies.pain_study.analysis.bcg import correct as bcg_correct
from studies.pain_study.analysis.bcg import detect as bcg_detect
from studies.pain_study.analysis.bcg import markers as bcg_markers
from studies.pain_study.analysis.bcg import metrics as bcg_metrics
from studies.pain_study.analysis.bcg.sources import (
    discover_run_pairs,
    validate_pair,
    write_corrected_recording,
)

WORKFLOW = "cardiac_gaps"

OCCIPITAL = ("O1", "O2", "Oz", "PO3", "PO4", "POz")


@dataclass(frozen=True)
class BenchmarkSettings:
    methods: tuple[str, ...] = ("obs", "aas")
    n_components: tuple[int, ...] = (4, 8)
    band: tuple[float, float] = (1.0, 20.0)
    alpha_band: tuple[float, float] = (8.0, 13.0)
    window: tuple[float, float] = (-0.3, 0.7)
    n_surrogate: int = 20
    seed: int = 0
    picks: tuple[str, ...] = field(default=OCCIPITAL)


def benchmark_arrays(
    data_uv: np.ndarray,
    beats: np.ndarray,
    sfreq: float,
    *,
    methods=("obs", "aas"),
    n_components=(4,),
    band=(1.0, 20.0),
    alpha_band=(8.0, 13.0),
    window=(-0.3, 0.7),
    n_surrogate: int = 20,
    seed: int = 0,
    picks: np.ndarray | None = None,
    ch_names: list[str] | None = None,
    stim_onsets: np.ndarray | None = None,
) -> list[dict]:
    """Score each method on both arms, with a sham control for the preservation arm.

    The sham applies the identical correction at a circularly-shifted beat train, where no
    artifact sits, so everything it removes is signal loss.
    """
    duration = data_uv.shape[1] / sfreq
    rng = np.random.default_rng(seed)
    sham_beats = np.sort((beats + rng.uniform(2.0, duration - 2.0)) % duration)

    rows: list[dict] = []
    for method in methods:
        ranks = n_components if method == "obs" else (0,)
        for rank in ranks:
            kwargs = dict(method=method, window=window, ch_names=ch_names)
            if method == "obs":
                kwargs["n_components"] = rank

            real = bcg_correct.correct_beats(data_uv, beats, sfreq, **kwargs)
            sham = bcg_correct.correct_beats(data_uv, sham_beats, sfreq, **kwargs)
            result = bcg_metrics.rlocked_reduction(
                real, beats, sfreq, window=window, n_surrogate=n_surrogate, seed=seed
            )
            rows.append(
                {
                    "method": method,
                    "n_components": rank,
                    "removal_max": result.max_value,
                    "removal_null_max": result.null_max,
                    "removal_channels_above_null": result.channels_above_null,
                    "real_alpha_retained": bcg_metrics.band_retention(
                        data_uv, real, sfreq, alpha_band, picks
                    ),
                    "sham_alpha_retained": bcg_metrics.band_retention(
                        data_uv, sham, sfreq, alpha_band, picks
                    ),
                    "real_band_retained": bcg_metrics.band_retention(
                        data_uv, real, sfreq, band, picks
                    ),
                    "sham_band_retained": bcg_metrics.band_retention(
                        data_uv, sham, sfreq, band, picks
                    ),
                }
            )
            if stim_onsets is not None and stim_onsets.size >= 8:
                evoked = bcg_metrics.evoked_preservation(
                    data_uv, real, stim_onsets, sfreq, (-0.1, 0.5)
                )
                rows[-1]["evoked_correlation_median"] = float(np.nanmedian(evoked.correlation))
                rows[-1]["evoked_amplitude_ratio_median"] = float(
                    np.nanmedian(evoked.amplitude_ratio)
                )
    return rows


MINIMUM_SCORABLE_BEATS = 8
# A resting adult in the bore runs roughly 45-100 bpm; the cohort's own median is 60.9 and
# its 5th percentile 46.8. Outside this the marker set cannot be the whole beat train.
PLAUSIBLE_BPM = (40.0, 110.0)

# The absolute range above only catches a train missing most of its beats. A run missing a
# third of them lands at an unremarkable rate -- sub-0001 run 1 recovered to 47.0 bpm where
# that subject's other runs sit at 71.1 -- and is passed on as corrected. A rate is only
# plausible against the heart it came from, so each run is also compared with its own
# subject. On this cohort the under-marked runs sit at 0.66-0.79 of their subject's rate and
# the genuinely slower ones at 0.90 or above; this bound is the gap between those two groups
# and is reported alongside the ratio so it can be re-derived elsewhere.
SUBJECT_RATE_RATIO = 0.85


def recovery_status(recovery, minimum: int = MINIMUM_SCORABLE_BEATS) -> str:
    """Why a run is or is not correctable, as a status rather than a bare count.

    A run Analyzer marked without leaving gaps has nothing to correct, which is a different
    outcome from one whose gaps the matcher could not fill. Most skipped runs on this
    cohort are the former, and a report that calls both `too_few_recovered` reads a healthy
    run as a detector failure.
    """
    if recovery.quality.status != "ok":
        return recovery.quality.status
    recovered = int(recovery.recovered_beats.size)
    if recovery.quality.gap_seconds_before <= 0.0:
        return "no_gaps"
    if recovered < minimum:
        return f"too_few_recovered ({recovered})"
    # Structural checks can all pass on a run that is simply under-marked everywhere: the
    # gap rule is relative to the run's own median RR, so when Analyzer marked a fraction
    # of the beats there is no stretch long enough to flag and nothing gets searched.
    bpm = recovery.quality.implied_bpm
    if bpm == bpm and not PLAUSIBLE_BPM[0] <= bpm <= PLAUSIBLE_BPM[1]:
        return f"implausible_rate ({bpm:.1f} bpm)"
    return "ok"


def flag_rates_against_subject(
    rows: list[dict], *, minimum_ratio: float = SUBJECT_RATE_RATIO
) -> list[dict]:
    """Compare each run's recovered rate with its own subject's, and say when it falls short.

    :func:`recovery_status` judges a run alone, so it can only reject a rate that is
    impossible for anybody. That leaves the common failure untouched: Analyzer marks two
    thirds of a train, the gap rule finds nothing to search because it is relative to the
    run's own median RR, and the run reports a perfectly ordinary rate that happens to be a
    third below the rate the same heart shows in every other run of the session.

    The reference is the median of the subject's *other* runs, restricted to ones whose rate
    is possible at all, so one bad run cannot define the standard it is judged against. A
    subject with fewer than two usable runs is left alone rather than guessed at.

    ``subject_reference_bpm`` and ``implied_bpm_ratio`` are written for every row whether or
    not the status changes, so the comparison stays visible and re-derivable at another
    bound. A run already flagged for its own reason keeps that reason.
    """
    import math

    by_subject: dict[str, list[dict]] = {}
    for row in rows:
        by_subject.setdefault(str(row.get("subject")), []).append(row)

    for runs in by_subject.values():
        for row in runs:
            row["subject_reference_bpm"] = float("nan")
            row["implied_bpm_ratio"] = float("nan")
            bpm = row.get("implied_bpm")
            if bpm is None or not isinstance(bpm, (int, float)) or math.isnan(bpm):
                continue
            others = [
                float(other["implied_bpm"])
                for other in runs
                if other is not row
                and isinstance(other.get("implied_bpm"), (int, float))
                and not math.isnan(float(other.get("implied_bpm", float("nan"))))
                and PLAUSIBLE_BPM[0] <= float(other["implied_bpm"]) <= PLAUSIBLE_BPM[1]
            ]
            if len(others) < 2:
                continue
            reference = float(np.median(others))
            row["subject_reference_bpm"] = reference
            ratio = float(bpm) / reference if reference else float("nan")
            row["implied_bpm_ratio"] = ratio
            if row.get("status") == "ok" and ratio < minimum_ratio:
                row["status"] = f"rate_below_subject ({float(bpm):.1f} vs {reference:.1f} bpm)"
    return rows


def quality_row(recovery, crosscheck: dict | None) -> dict:
    """Every per-run recovery measurement, flattened for the cohort report.

    The fields are taken from `BeatQuality` by reflection rather than listed, so a field
    added to the dataclass reaches the report instead of being silently dropped. None of
    them is a verdict: the lock ratio of Analyzer's *own* beats is what separates a run
    whose ECG cannot support detection from one where recovery simply failed.
    """
    from dataclasses import asdict

    row = asdict(recovery.quality)
    # BeatQuality's own `status` describes the beat set, not the run's outcome. Left under
    # that name it overwrites the run status when the row is merged into the report.
    row["beat_status"] = row.pop("status")
    if crosscheck is None:
        crosscheck = {"status": "not_run"}
    row["crosscheck_status"] = crosscheck.get("status", "not_run")
    row["crosscheck_agreement_fraction"] = crosscheck.get("agreement_fraction", float("nan"))
    row["crosscheck_beats"] = crosscheck.get("crosscheck_beats", float("nan"))
    row["crosscheck_lock_ratio"] = crosscheck.get("crosscheck_lock_ratio", float("nan"))
    return row


PROVENANCE_SUFFIX = ".gapfill.json"


def _provenance_path(written_vhdr: Path) -> Path:
    return Path(written_vhdr).with_suffix(PROVENANCE_SUFFIX)


def write_provenance(written_vhdr: Path, record: dict) -> Path:
    """Record how a written recording was produced, beside the recording itself."""
    import datetime

    payload = dict(record)
    payload["written_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    destination = _provenance_path(written_vhdr)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def read_provenance(written_vhdr: Path) -> dict | None:
    """What produced a written recording, or None when it carries no record.

    `verify` scores whatever file it finds at the output path, which after a failed
    `apply` is whatever an earlier run left there. Without this the two are
    indistinguishable, and stale results get reported as current.
    """
    path = _provenance_path(written_vhdr)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def scoring_data(raw) -> tuple[np.ndarray, list[str]]:
    """EEG channels in microvolts, and their names.

    The referee must never see the ECG channel. It carries the cardiac signal itself, so
    its R-locked reduction is near-total by construction -- 0.545 on sub0009 run 1 against
    0.004 for the best EEG channel -- and scoring it reports the ECG rather than any
    residual artifact.
    """
    names = raw.copy().pick("eeg").ch_names
    return raw.copy().pick(names).get_data() * 1e6, names


def _load_pair(pair):
    import mne

    mne.set_log_level("ERROR")
    uncorrected = mne.io.read_raw_brainvision(pair.uncorrected_vhdr, preload=True, verbose="ERROR")
    corrected = mne.io.read_raw_brainvision(pair.corrected_vhdr, preload=True, verbose="ERROR")
    return uncorrected, corrected


def benchmark_run(pair, settings: BenchmarkSettings) -> list[dict]:
    """Benchmark one recording, scoring only the stretches this stage would change."""
    validation = validate_pair(pair)
    if validation.status != "ok":
        return [{"subject": pair.subject, "run": pair.run, "status": validation.status}]

    uncorrected, _ = _load_pair(pair)
    sfreq = uncorrected.info["sfreq"]
    eeg_names = uncorrected.copy().pick("eeg").ch_names
    data = uncorrected.copy().pick(eeg_names).get_data() * 1e6
    ecg = uncorrected.copy().pick(["ECG"]).get_data()[0] * 1e6

    analyzer = bcg_detect.read_analyzer_beats(pair.uncorrected_vhdr)
    recovery = bcg_detect.recover_beats(ecg, analyzer, sfreq)
    status = recovery_status(recovery)
    if status != "ok":
        return [
            {
                "subject": pair.subject,
                "run": pair.run,
                "status": status,
                "analyzer_beats": int(analyzer.size),
                "recovered_beats": int(recovery.recovered_beats.size),
                "gap_seconds_before": recovery.quality.gap_seconds_before,
            }
        ]

    # `np.array(...) or None` raises on a multi-element array; test emptiness explicitly.
    selected = [i for i, n in enumerate(eeg_names) if n in settings.picks]
    picks = np.array(selected, dtype=int) if selected else None

    stim = np.asarray(
        [
            onset
            for onset, description in zip(
                uncorrected.annotations.onset, uncorrected.annotations.description
            )
            if description.split("/")[-1].strip().startswith("S")
        ],
        dtype=float,
    )

    rows = benchmark_arrays(
        data,
        recovery.recovered_beats,
        sfreq,
        methods=settings.methods,
        n_components=settings.n_components,
        band=settings.band,
        alpha_band=settings.alpha_band,
        window=settings.window,
        n_surrogate=settings.n_surrogate,
        seed=settings.seed,
        picks=picks,
        ch_names=eeg_names,
        stim_onsets=stim,
    )
    for row in rows:
        row.update(
            {
                "subject": pair.subject,
                "run": pair.run,
                "status": "ok",
                "analyzer_beats": int(analyzer.size),
                "recovered_beats": int(recovery.recovered_beats.size),
                "analyzer_lock_ratio": recovery.quality.analyzer_lock_ratio,
                "recovered_lock_ratio": recovery.quality.recovered_lock_ratio,
                "gap_seconds_before": recovery.quality.gap_seconds_before,
                "gap_seconds_after": recovery.quality.gap_seconds_after,
            }
        )
    return rows


def report_run(pair, *, crosscheck: bool = True) -> dict:
    """Measure one recording's gaps and recovery quality without correcting anything.

    This is the cohort's inventory: how much of each run Analyzer left unmarked, how much
    recovery closes, and whether the recovered beats sit on the QRS. It reads only the ECG
    channel and the markers, so it runs over all 104 recordings cheaply.
    """
    import mne

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_brainvision(pair.uncorrected_vhdr, preload=True, verbose="ERROR")
    sfreq = raw.info["sfreq"]
    duration = raw.n_times / sfreq

    if "ECG" not in raw.ch_names:
        return {"subject": pair.subject, "run": pair.run, "status": "missing_ecg"}

    ecg = raw.copy().pick(["ECG"]).get_data()[0] * 1e6
    analyzer = bcg_detect.read_analyzer_beats(pair.uncorrected_vhdr)
    recovery = bcg_detect.recover_beats(ecg, analyzer, sfreq)

    agreement = None
    if crosscheck:
        agreement = bcg_detect.crosscheck_agreement(recovery.combined_beats, ecg, sfreq)

    gaps = bcg_detect.gap_summary(analyzer, duration)
    row = {
        "subject": pair.subject,
        "run": pair.run,
        "status": recovery_status(recovery),
        "duration_s": duration,
        "analyzer_beats": int(analyzer.size),
        "combined_beats": int(recovery.combined_beats.size),
        "n_gaps": gaps["n_gaps"],
        "gap_fraction": gaps["gap_fraction"],
        "max_rr_s": gaps["max_rr_s"],
        "implied_missing_beats": gaps["implied_missing_beats"],
    }
    row.update(quality_row(recovery, agreement))
    return row


def markers_run(pair, output_root: Path) -> dict:
    """Re-emit a recording unchanged except for the recovered R markers.

    Analyzer corrects better than we do at the beats it has, so handing the beats back is
    worth more than correcting with them ourselves. The ``.eeg`` and ``.vhdr`` are copied
    byte for byte and only the ``.vmrk`` grows, so Analyzer resumes from its own
    `Pulse Artifact Correction (Mark R peaks)` node with nothing else disturbed.
    """
    import mne

    mne.set_log_level("ERROR")
    source = pair.uncorrected_vhdr
    raw = mne.io.read_raw_brainvision(source, preload=True, verbose="ERROR")
    sfreq = raw.info["sfreq"]

    if "ECG" not in raw.ch_names:
        return {"subject": pair.subject, "run": pair.run, "status": "missing_ecg"}

    ecg = raw.copy().pick(["ECG"]).get_data()[0] * 1e6
    analyzer = bcg_detect.read_analyzer_beats(source)
    recovery = bcg_detect.recover_beats(ecg, analyzer, sfreq)

    # A run with nothing to add is still re-emitted, so the output tree is a complete
    # drop-in replacement rather than a partial one the caller has to merge by hand.
    marker_file = bcg_markers.read_marker_file(source.with_suffix(".vmrk"))
    augmented = bcg_markers.add_pulse_markers(marker_file, recovery.recovered_beats, sfreq)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for suffix in (".eeg", ".vhdr"):
        shutil.copy2(source.with_suffix(suffix), output_root / source.with_suffix(suffix).name)
    destination = output_root / source.name
    bcg_markers.write_marker_file(destination.with_suffix(".vmrk"), augmented)

    check = bcg_detect.read_analyzer_beats(destination)
    expected = recovery.combined_beats.size
    if check.size != expected:
        raise RuntimeError(
            f"{destination.name}: wrote {expected} R markers but read back {check.size}"
        )

    row = {
        "subject": pair.subject,
        "run": pair.run,
        "status": recovery_status(recovery, minimum=1),
        "analyzer_beats": int(analyzer.size),
        "recovered_beats": int(recovery.recovered_beats.size),
        "total_r_markers": int(check.size),
        "markers_total": len(augmented.markers),
        "recovered_lock_ratio": recovery.quality.recovered_lock_ratio,
        "gap_seconds_before": recovery.quality.gap_seconds_before,
        "gap_seconds_after": recovery.quality.gap_seconds_after,
        "output": str(destination),
    }
    write_provenance(destination, {**row, "source": str(source)})
    return row


def _write_tsv(rows: list[dict], destination: Path) -> None:
    import csv

    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


ROUNDTRIP_RELATIVE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ApplySettings:
    method: str = "obs"
    n_components: int = 4
    window: tuple[float, float] = (-0.3, 0.7)
    pad_seconds: float = 0.5


def substitute_gap_stretches(base_uv, replacement_uv, gaps, sfreq, pad_seconds=0.5):
    """Splice corrected gap stretches into Analyzer's output, with a small pad.

    The pad covers epochs of beats sitting just inside a gap edge, whose correction window
    extends slightly beyond the gap itself.
    """
    padded = [(start - pad_seconds, end + pad_seconds) for start, end in gaps]
    return bcg_correct.substitute_stretches(base_uv, replacement_uv, padded, sfreq)


def apply_run(pair, output_root: Path, settings: ApplySettings) -> dict:
    """Correct one recording's gap stretches and write the result beside its sidecars."""
    import mne

    validation = validate_pair(pair)
    if validation.status != "ok":
        return {"subject": pair.subject, "run": pair.run, "status": validation.status}

    uncorrected, corrected = _load_pair(pair)
    sfreq = uncorrected.info["sfreq"]
    eeg_names = uncorrected.copy().pick("eeg").ch_names
    unc = uncorrected.copy().pick(eeg_names).get_data() * 1e6
    cor = corrected.copy().pick(eeg_names).get_data() * 1e6

    ecg = uncorrected.copy().pick(["ECG"]).get_data()[0] * 1e6
    analyzer = bcg_detect.read_analyzer_beats(pair.uncorrected_vhdr)
    recovery = bcg_detect.recover_beats(ecg, analyzer, sfreq)
    # `apply` corrects whatever the matcher found rather than needing the 8 beats the
    # referee wants to score a run -- but OBS cannot form a basis from fewer epochs than
    # components, so the floor follows the method actually being applied.
    minimum = settings.n_components + 1 if settings.method == "obs" else 1
    status = recovery_status(recovery, minimum=minimum)
    if status != "ok":
        return {
            "subject": pair.subject,
            "run": pair.run,
            "status": status,
            "analyzer_beats": int(analyzer.size),
            "recovered_beats": int(recovery.recovered_beats.size),
            "gap_seconds_before": recovery.quality.gap_seconds_before,
        }

    kwargs = dict(method=settings.method, window=settings.window, ch_names=eeg_names)
    if settings.method == "obs":
        kwargs["n_components"] = settings.n_components
    repaired = bcg_correct.correct_beats(unc, recovery.recovered_beats, sfreq, **kwargs)

    gaps = [(g.start_s, g.end_s) for g in bcg_detect.find_gaps(analyzer)]
    merged = substitute_gap_stretches(cor, repaired, gaps, sfreq, settings.pad_seconds)

    # Write every channel the header describes, with the EEG rows replaced and the rest --
    # ECG above all -- carried through from Analyzer's own output untouched.
    full = corrected.get_data() * 1e6
    eeg_rows = [corrected.ch_names.index(name) for name in eeg_names]
    full[eeg_rows] = merged
    destination = write_corrected_recording(pair.corrected_vhdr, output_root, full)

    check = mne.io.read_raw_brainvision(destination, preload=True, verbose="ERROR")
    deviation = float(np.max(np.abs(check.get_data() * 1e6 - full)))
    scale = float(np.max(np.abs(full)))
    if deviation > ROUNDTRIP_RELATIVE_TOLERANCE * scale:
        raise RuntimeError(
            f"{destination.name}: written data differs by {deviation:.3e} uV, "
            f"above the {ROUNDTRIP_RELATIVE_TOLERANCE * scale:.3e} uV round-trip tolerance."
        )

    row = {
        "subject": pair.subject,
        "run": pair.run,
        "status": "ok",
        "method": settings.method,
        "n_components": settings.n_components,
        "recovered_beats": int(recovery.recovered_beats.size),
        "gap_seconds_before": recovery.quality.gap_seconds_before,
        "gap_seconds_after": recovery.quality.gap_seconds_after,
        "gap_fraction_replaced": sum(e - s for s, e in gaps) / (unc.shape[1] / sfreq),
        "roundtrip_max_deviation_uv": deviation,
        "markers_preserved": int(len(check.annotations)),
        "output": str(destination),
    }
    write_provenance(destination, {**row, "source": str(pair.uncorrected_vhdr)})
    return row


def verify_run(pair, output_root: Path) -> dict:
    """Re-score a written recording with the referee, inside the gaps and outside them.

    Scoring the two separately is what shows whether the stage introduced a time-varying
    difference within the run, which is the risk of correcting only part of it.
    """
    import mne

    mne.set_log_level("ERROR")
    destination = output_root / pair.corrected_vhdr.name
    if not destination.exists():
        return {"subject": pair.subject, "run": pair.run, "status": "not_written"}

    provenance = read_provenance(destination)
    if provenance is None:
        # Left by an earlier run, so it may not reflect the current code or settings.
        return {"subject": pair.subject, "run": pair.run, "status": "no_provenance"}

    written = mne.io.read_raw_brainvision(destination, preload=True, verbose="ERROR")
    uncorrected, _ = _load_pair(pair)
    sfreq = written.info["sfreq"]
    data, scored_names = scoring_data(written)

    ecg = uncorrected.copy().pick(["ECG"]).get_data()[0] * 1e6
    analyzer = bcg_detect.read_analyzer_beats(pair.uncorrected_vhdr)
    recovery = bcg_detect.recover_beats(ecg, analyzer, sfreq)

    result = bcg_metrics.rlocked_reduction(
        data, recovery.recovered_beats, sfreq, n_surrogate=20, seed=0
    )
    analyzer_result = bcg_metrics.rlocked_reduction(data, analyzer, sfreq, n_surrogate=20, seed=0)
    return {
        "subject": pair.subject,
        "run": pair.run,
        "status": "ok",
        "channels_scored": len(scored_names),
        "markers_present": int(len(written.annotations)),
        "applied_method": provenance.get("method"),
        "applied_n_components": provenance.get("n_components"),
        "applied_utc": provenance.get("written_utc"),
        "recovered_removal_max": result.max_value,
        "recovered_null_max": result.null_max,
        "recovered_channels_above_null": result.channels_above_null,
        "analyzer_removal_max": analyzer_result.max_value,
        "analyzer_channels_above_null": analyzer_result.channels_above_null,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["report", "markers", "benchmark", "apply", "verify"])
    parser.add_argument("--uncorrected-root", type=Path, default=None)
    parser.add_argument("--corrected-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--method", default=None, choices=["obs", "aas"])
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    run(args)


def run(args: argparse.Namespace) -> None:
    """Execute one command. Split from ``main`` so the CLI can call it with its own args."""
    from studies.pain_study.scripts.workflow_config import load_workflow_config

    config = load_workflow_config(WORKFLOW, getattr(args, "config", None))
    args.uncorrected_root = config.path("uncorrected_root", override=args.uncorrected_root)
    args.corrected_root = config.path("corrected_root", override=args.corrected_root)
    args.output_root = config.path("output_root", override=args.output_root)
    report_dir = config.path("report_dir")
    settings_block = config.get("cardiac_gaps") or {}
    if args.method is None:
        args.method = str(settings_block.get("method", "obs"))
    if args.n_components is None:
        args.n_components = int(settings_block.get("n_components", 4))

    pairs = discover_run_pairs(args.uncorrected_root, args.corrected_root)
    if args.subjects:
        requested = set(args.subjects)
        pairs = [p for p in pairs if p.subject in requested]
        missing = sorted(requested - {p.subject for p in pairs})
        if missing:
            print(f"warning: no paired recordings for {', '.join(missing)}", flush=True)
    if args.limit:
        # Pairs sort by subject, so a limit smaller than one subject's run count silently
        # drops every later subject. Say so rather than letting the TSV imply coverage.
        dropped = {p.subject for p in pairs[args.limit :]} - {
            p.subject for p in pairs[: args.limit]
        }
        pairs = pairs[: args.limit]
        if dropped:
            print(
                f"warning: --limit {args.limit} excludes {', '.join(sorted(dropped))} entirely",
                flush=True,
            )
    print(
        f"selected {len(pairs)} recordings across "
        f"{len(sorted({p.subject for p in pairs}))} subjects",
        flush=True,
    )

    default_names = {
        "report": "recovery_report.tsv",
        "markers": "markers.tsv",
        "benchmark": "benchmark.tsv",
        "apply": "apply.tsv",
        "verify": "verify.tsv",
    }
    destination = args.output or report_dir / default_names[args.command]
    apply_settings = ApplySettings(method=args.method, n_components=args.n_components)

    rows: list[dict] = []
    for pair in pairs:
        try:
            if args.command == "report":
                rows.append(report_run(pair))
            elif args.command == "markers":
                rows.append(markers_run(pair, args.output_root))
            elif args.command == "benchmark":
                rows.extend(benchmark_run(pair, BenchmarkSettings()))
            elif args.command == "apply":
                rows.append(apply_run(pair, args.output_root, apply_settings))
            else:
                rows.append(verify_run(pair, args.output_root))
        except Exception as error:  # a failing run is a measurement, not a fault
            rows.append(
                {
                    "subject": pair.subject,
                    "run": pair.run,
                    "status": f"error: {type(error).__name__}: {error}",
                }
            )
        print(json.dumps(rows[-1]), flush=True)
    # Only now is there a second run of the same subject to compare a rate against, so the
    # per-subject check cannot live in the per-run status. The streamed lines above are
    # per-run and predate it; the written table carries the revised status.
    if args.command in {"report", "markers", "apply"}:
        flag_rates_against_subject(rows)
        revised = [r for r in rows if str(r.get("status", "")).startswith("rate_below_subject")]
        for row in revised:
            print(f"revised {row['subject']} run-{row['run']}: {row['status']}", flush=True)
    _write_tsv(rows, destination)
    print(f"wrote {len(rows)} rows to {destination}")


if __name__ == "__main__":
    main()
