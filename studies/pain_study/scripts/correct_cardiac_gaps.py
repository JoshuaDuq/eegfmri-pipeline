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
    if recovery.recovered_beats.size < 8:
        return [
            {
                "subject": pair.subject,
                "run": pair.run,
                "status": f"too_few_recovered ({recovery.recovered_beats.size})",
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["benchmark"])
    parser.add_argument("--uncorrected-root", type=Path, default=DEFAULT_UNCORRECTED)
    parser.add_argument("--corrected-root", type=Path, default=DEFAULT_CORRECTED)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/cardiac_gap_fill/benchmark.tsv")
    )
    args = parser.parse_args(argv)

    pairs = discover_run_pairs(args.uncorrected_root, args.corrected_root)
    if args.subjects:
        pairs = [p for p in pairs if p.subject in set(args.subjects)]
    if args.limit:
        pairs = pairs[: args.limit]

    settings = BenchmarkSettings()
    rows: list[dict] = []
    for pair in pairs:
        try:
            rows.extend(benchmark_run(pair, settings))
        except Exception as error:  # a failing run is a measurement, not a fault
            rows.append(
                {
                    "subject": pair.subject,
                    "run": pair.run,
                    "status": f"error: {type(error).__name__}: {error}",
                }
            )
        print(json.dumps(rows[-1]), flush=True)
    _write_tsv(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
