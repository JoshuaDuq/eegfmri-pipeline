"""Fill Analyzer's pulse-marker gaps: recover beats, correct only there, and score it.

Analyzer's correction is kept wherever it marked a beat, because it measurably beats ours
on both arms there (0.16% residual against our 2.03%, alpha retained 0.54 against 0.34).
What it never marked is untouched artifact -- 6,954 s across the cohort, up to 77% of a
single run -- and that is all this stage changes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eeg_pipeline.preprocessing.bcg import correct as bcg_correct
from eeg_pipeline.preprocessing.bcg import detect as bcg_detect
from eeg_pipeline.preprocessing.bcg import metrics as bcg_metrics
from eeg_pipeline.preprocessing.bcg.sources import discover_run_pairs, validate_pair

DEFAULT_UNCORRECTED = Path(
    "/Volumes/KINGSTON/EEG_fMRI_data/source_data/"
    "processed_scanner_artifact_with_pulse_markers_no_bcg_correction"
)
DEFAULT_CORRECTED = Path("data/source_data/processed_trimmed_0-60s_30-115bpm_marker_template")
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
    return "ok"


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
    # `apply` corrects whatever the matcher found, so its floor is 1 beat rather than the
    # 8 the referee needs to score a run; the status still separates "nothing to correct"
    # from "gaps the matcher could not fill".
    status = recovery_status(recovery, minimum=1)
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

    info = mne.create_info(eeg_names, sfreq, ch_types="eeg")
    out_raw = mne.io.RawArray(merged * 1e-6, info, verbose="ERROR")
    destination = output_root / pair.corrected_vhdr.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(destination, out_raw, fmt="brainvision", overwrite=True, verbose="ERROR")

    check = mne.io.read_raw_brainvision(destination, preload=True, verbose="ERROR")
    deviation = float(np.max(np.abs(check.get_data() * 1e6 - merged)))
    scale = float(np.max(np.abs(merged)))
    if deviation > ROUNDTRIP_RELATIVE_TOLERANCE * scale:
        raise RuntimeError(
            f"{destination.name}: written data differs by {deviation:.3e} uV, "
            f"above the {ROUNDTRIP_RELATIVE_TOLERANCE * scale:.3e} uV round-trip tolerance."
        )

    return {
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
        "output": str(destination),
    }


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

    written = mne.io.read_raw_brainvision(destination, preload=True, verbose="ERROR")
    uncorrected, _ = _load_pair(pair)
    sfreq = written.info["sfreq"]
    data = written.get_data() * 1e6

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
        "recovered_removal_max": result.max_value,
        "recovered_null_max": result.null_max,
        "recovered_channels_above_null": result.channels_above_null,
        "analyzer_removal_max": analyzer_result.max_value,
        "analyzer_channels_above_null": analyzer_result.channels_above_null,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["benchmark", "apply", "verify"])
    parser.add_argument("--uncorrected-root", type=Path, default=DEFAULT_UNCORRECTED)
    parser.add_argument("--corrected-root", type=Path, default=DEFAULT_CORRECTED)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/cardiac_gap_fill/corrected")
    )
    parser.add_argument("--method", default="obs", choices=["obs", "aas"])
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

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
        "benchmark": "benchmark.tsv",
        "apply": "apply.tsv",
        "verify": "verify.tsv",
    }
    destination = args.output or Path("outputs/cardiac_gap_fill") / default_names[args.command]
    apply_settings = ApplySettings(method=args.method, n_components=args.n_components)

    rows: list[dict] = []
    for pair in pairs:
        try:
            if args.command == "benchmark":
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
    _write_tsv(rows, destination)
    print(f"wrote {len(rows)} rows to {destination}")


if __name__ == "__main__":
    main()
