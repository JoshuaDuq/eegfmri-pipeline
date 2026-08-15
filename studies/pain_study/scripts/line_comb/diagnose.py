"""Diagnose narrowband scanner-linked contamination across the thermal-pain cohort.

Runs in two stages. ``cache`` reads the recordings once and writes per-participant
spectra to disk; ``analyse`` works only from that cache, so the statistics can be
reworked without touching the drive again. ``all`` does both.

    eeg-pipeline line-comb diagnose --stage all

Two sources are read. The final cleaned epochs define which lines reach the analyses.
The gradient-free head of every recording -- the stretch between the amplifier starting
and the scanner's first gradient pulse -- says which of those lines exist when the
gradients are off, which is what separates an imaging artifact from a room artifact.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from studies.pain_study.analysis.line_comb import cohort as hc
from studies.pain_study.analysis.line_comb import diagnosis as hd

WORKFLOW = "line_comb"

TASK = "thermalactive"
BACKGROUND_HALF_WIDTH_HZ = 100.0 / 21.6  # 4.6296 Hz: 100 bins high-res, 25 bins matched
DETECTION_LOW_HZ = 3.0
DETECTION_HIGH_HZ = 95.0
NOTCH_HZ = (59.5, 60.5)
CACHE_HIGH_HZ = 110.0
LINE_MASK_HALF_WIDTH_HZ = 0.15
NARROW_LINEWIDTH_RATIO = 3.0
"""Half-power width, in window widths, below which a detection counts as monochromatic.

This only decides detections that are *not* comb members; see :func:`classify_lines` for
why membership is the primary criterion. Among the non-members the widths fall into a
tight group at 1.4-2.2 -- at or near the 1.4382/T floor a Hann window imposes, so as
narrow as anything can be measured -- and a scattered tail from 4.9 upward. Any threshold
between 2.2 and 4.9 gives the same partition of the non-members."""
WIDE_MEMBER_RATIO = 10.0
"""Comb members wider than this are reported separately and never masked.

Six members measure 12-16 window widths against 1.1-8.6 for the rest, with the gap
between 8.6 and 12.1. All six are weak (1.3-2.6 dB) and found in only 2-8 participants,
so the half-power width is being set by the broadband activity they sit on rather than by
the line. Five of the six sit within 9 mHz of a harmonic, which is far too precise to be
coincidence, so they are treated as members; but because a rhythm coinciding with a comb
position cannot be excluded for any individual one, they are left out of the band-power
masking, where including a real rhythm would overstate contamination."""
CONTROL_TAIL_SECONDS = 3.0
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
FDR_ALPHA = 0.05

BANDS = {
    "delta": (1.0, 3.9),
    "theta": (4.0, 7.9),
    "alpha": (8.0, 12.9),
    "beta": (13.0, 30.0),
    "beta_low_clean": (13.0, 17.9),
    "beta_high_clean": (23.1, 30.0),
    "gamma": (30.1, 80.0),
    "gamma_low_clean": (30.1, 38.0),
    "gamma_mid_clean": (43.0, 56.0),
    "gamma_high_clean": (67.0, 77.0),
}

CLASSIC_REFERENCE_HARMONICS = (18, 37, 55, 74)


# --------------------------------------------------------------------------- caching


def discover_subjects(deriv_root: Path) -> list[str]:
    """Participants with a final cleaned epochs file."""
    found = sorted(
        path.name
        for path in deriv_root.glob("sub-*")
        if (path / "eeg" / f"{path.name}_task-{TASK}_epo.fif").exists()
    )
    if not found:
        raise FileNotFoundError(f"No cleaned epochs found under {deriv_root}")
    return found


def run_lengths(deriv_root: Path, subject: str) -> list[int]:
    """Sample count of each run's filtered continuous file, in run order."""
    import mne

    lengths = []
    for run in range(1, 100):
        path = deriv_root / subject / "eeg" / f"{subject}_task-{TASK}_run-{run}_proc-filt_raw.fif"
        if not path.exists():
            break
        raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
        lengths.append(int(raw.n_times))
    if not lengths:
        raise FileNotFoundError(f"No per-run filtered files for {subject}")
    return lengths


def heart_rate_bpm(ecg: np.ndarray, sfreq: float) -> float:
    """Median heart rate over a set of epochs, from R peaks in the retained ECG lead."""
    from mne.preprocessing.ecg import qrs_detector

    intervals: list[float] = []
    for trace in ecg:
        try:
            peaks = qrs_detector(sfreq, trace, verbose="ERROR")
        except Exception:  # noqa: BLE001 - a detector failing is a missing datum, not a fault
            continue
        if len(peaks) >= 3:
            intervals.extend(np.diff(np.asarray(peaks, dtype=float)) / sfreq)
    usable = [value for value in intervals if 0.3 <= value <= 2.0]
    return float(60.0 / np.median(usable)) if usable else float("nan")


def _coefficients(data: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed one-sided DFT of each epoch, keeping phase."""
    window = np.hanning(data.shape[-1])
    return (
        np.fft.rfftfreq(data.shape[-1], d=1.0 / sfreq),
        np.fft.rfft(data * window, axis=-1),
    )


def _psd_from_coefficients(coefficients: np.ndarray, sfreq: float, n_times: int) -> np.ndarray:
    window = np.hanning(n_times)
    psd = np.abs(coefficients) ** 2 / (sfreq * float(np.sum(window**2)))
    psd[..., 1:-1] *= 2.0
    return psd


def build_final_cache(deriv_root: Path, subject: str, cache_dir: Path) -> Path:
    """Compute and store one participant's final-epoch spectra and DFT coefficients."""
    import mne

    mne.set_log_level("ERROR")
    destination = cache_dir / f"{subject}_final.npz"
    epochs = mne.read_epochs(
        deriv_root / subject / "eeg" / f"{subject}_task-{TASK}_epo.fif", preload=True
    )
    sfreq = float(epochs.info["sfreq"])
    labels = hc.assign_runs(epochs.events[:, 0], run_lengths(deriv_root, subject))

    eeg_picks = mne.pick_types(epochs.info, eeg=True, exclude=())
    channel_names = [epochs.ch_names[pick] for pick in eeg_picks]
    data = epochs.get_data(picks=eeg_picks)

    n_high = hd.tr_commensurate_length(data.shape[-1], sfreq)
    freqs_high, coefficients = _coefficients(data[..., :n_high], sfreq)
    psd_high = _psd_from_coefficients(coefficients, sfreq, n_high)
    freqs_matched, psd_matched = hc.segment_periodograms(data, sfreq, tr_count=hc.MATCHED_TR_COUNT)

    keep_high = freqs_high <= CACHE_HIGH_HZ
    keep_matched = freqs_matched <= CACHE_HIGH_HZ
    coefficient_band = (freqs_high >= DETECTION_LOW_HZ) & (freqs_high <= DETECTION_HIGH_HZ)

    runs = sorted(set(labels.tolist()))
    run_psd_high = np.stack([psd_high[labels == run].mean(axis=0) for run in runs])
    run_psd_matched = np.stack([psd_matched[labels == run].mean(axis=0) for run in runs])

    if "ECG" in epochs.ch_names:
        ecg = epochs.get_data(picks=[epochs.ch_names.index("ECG")])[:, 0, :]
        rates = np.array([heart_rate_bpm(ecg[labels == run], sfreq) for run in runs])
    else:
        rates = np.full(len(runs), np.nan)

    volume_offsets = epochs.metadata["Volume/V  1"].to_numpy(dtype=float) - float(epochs.tmin)

    np.savez_compressed(
        destination,
        subject=subject,
        sfreq=sfreq,
        freqs_high=freqs_high[keep_high],
        freqs_matched=freqs_matched[keep_matched],
        freqs_coefficients=freqs_high[coefficient_band],
        psd_high=run_psd_high[..., keep_high].astype(np.float32),
        psd_matched=run_psd_matched[..., keep_matched].astype(np.float32),
        coefficients=coefficients[..., coefficient_band].astype(np.complex64),
        channel_names=np.array(channel_names),
        bads=np.array(epochs.info["bads"], dtype=object).astype(str),
        runs=np.array(runs),
        epoch_runs=labels,
        epochs_per_run=np.array([int((labels == run).sum()) for run in runs]),
        volume_offsets_s=volume_offsets,
        heart_rate_bpm=rates,
    )
    return destination


def build_control_cache(
    source_root: Path,
    reference_channels: list[str],
    cache_dir: Path,
) -> Path:
    """Compute and store gradient-free spectra from the head of every recording."""
    import mne

    mne.set_log_level("ERROR")
    records: list[dict] = []
    spectra: list[np.ndarray] = []
    freqs_matched: np.ndarray | None = None

    for subject_dir in sorted(source_root.glob("sub-0*")):
        directory = subject_dir / "eeg" / "original_untrimmed_5khz"
        for vmrk in sorted(directory.glob("*.vmrk")):
            markers = hc.read_volume_markers(vmrk)
            if not markers:
                continue  # an aborted acquisition with no imaging
            raw = mne.io.read_raw_brainvision(vmrk.with_suffix(".vhdr"), preload=False)
            sfreq = float(raw.info["sfreq"])
            order = hc.align_channels(raw.ch_names, reference_channels)
            # Read a little past the first marker so the profile always contains the
            # gradient state the quiet level is measured against.
            probe_stop = min(markers[0] + int(CONTROL_TAIL_SECONDS * sfreq), raw.n_times)
            probe = raw.get_data(start=0, stop=probe_stop)[order]

            profile = hc.block_standard_deviation(probe, sfreq)
            start, stop = hc.detect_quiet_interval(profile, sfreq=sfreq)
            block = int(round(hd.TR_SECONDS * sfreq)) * hc.MATCHED_TR_COUNT
            tiles = hc.tile_intervals(start, stop, block=block, overlap=0.5)

            block_samples = int(round(hc.QUIET_BLOCK_SECONDS * sfreq))
            quiet_blocks = profile[start // block_samples : max(stop // block_samples, 1)]
            records.append(
                {
                    "subject": subject_dir.name,
                    "recording": vmrk.stem,
                    "kind": "baseline" if vmrk.name.startswith("Baseline") else "task",
                    "quiet_seconds": (stop - start) / sfreq,
                    "n_blocks": len(tiles),
                    "first_volume_s": markers[0] / sfreq,
                    "gradient_onset_s": stop / sfreq + hc.QUIET_GUARD_SECONDS,
                    "quiet_sd_uv": (
                        float(np.median(quiet_blocks)) * 1e6 if quiet_blocks.size else float("nan")
                    ),
                }
            )
            if tiles:
                stack = np.stack([probe[:, begin:end] for begin, end in tiles])
                freqs, psd = hc.segment_periodograms(stack, sfreq, tr_count=hc.MATCHED_TR_COUNT)
                keep = freqs <= CACHE_HIGH_HZ
                if freqs_matched is None:
                    freqs_matched = freqs[keep]
                spectra.append(psd.mean(axis=0)[:, keep].astype(np.float32))
            else:
                spectra.append(np.zeros((0, 0), dtype=np.float32))
            print(
                f"  {records[-1]['subject']} {records[-1]['kind']:8s} "
                f"{records[-1]['recording'][:36]:36s} "
                f"quiet={records[-1]['quiet_seconds']:5.2f}s blocks={records[-1]['n_blocks']}"
            )

    if freqs_matched is None:
        raise RuntimeError("No recording yielded a usable gradient-free window.")
    usable = [index for index, record in enumerate(records) if record["n_blocks"] > 0]
    destination = cache_dir / "control.npz"
    np.savez_compressed(
        destination,
        freqs_matched=freqs_matched,
        psd=np.stack([spectra[index] for index in usable]),
        channel_names=np.array(reference_channels),
        subjects=np.array([records[index]["subject"] for index in usable]),
        recordings=np.array([records[index]["recording"] for index in usable]),
        kinds=np.array([records[index]["kind"] for index in usable]),
        quiet_seconds=np.array([records[index]["quiet_seconds"] for index in usable]),
        n_blocks=np.array([records[index]["n_blocks"] for index in usable]),
        all_records=np.array(json.dumps(records)),
    )
    return destination


# -------------------------------------------------------------------------- analysis


@dataclass
class Grid:
    """One frequency grid with the cohort's spectra and prominences on it."""

    freqs: np.ndarray
    subject_psd: np.ndarray  # (n_subjects, n_freqs)
    subject_prominence: np.ndarray  # (n_subjects, n_freqs)
    half_width_bins: int

    @property
    def bin_width_hz(self) -> float:
        return float(self.freqs[1] - self.freqs[0])


def half_width_bins(freqs: np.ndarray) -> int:
    return int(round(BACKGROUND_HALF_WIDTH_HZ / float(freqs[1] - freqs[0])))


def prominence_of(spectrum: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    return hd.prominence_db(hd.to_db(spectrum), half_width_bins=half_width_bins(freqs))


def channel_median(psd: np.ndarray, channel_names, bads) -> np.ndarray:
    """Median over good channels. ``psd`` is (n_channels, n_freqs)."""
    mask = hc.good_channel_mask(list(channel_names), list(bads))
    return np.median(psd[mask], axis=0)


def build_grid(freqs: np.ndarray, subject_psd: np.ndarray) -> Grid:
    prominence = np.stack([prominence_of(spectrum, freqs) for spectrum in subject_psd])
    return Grid(freqs, subject_psd, prominence, half_width_bins(freqs))


class NoLinesDetected(RuntimeError):
    """No line survived FDR control, which is a measurement rather than a fault.

    Separated from the other RuntimeErrors this module raises so a caller can report a
    genuinely clean cohort without also swallowing "no usable gradient-free window" and
    "no usable background estimate", which mean the analysis could not run. Recorded
    identically, those read as successful cleaning -- the most dangerous direction for a
    verification step to fail in.
    """


def detection_mask(freqs: np.ndarray) -> np.ndarray:
    inside = (freqs >= DETECTION_LOW_HZ) & (freqs <= DETECTION_HIGH_HZ)
    return inside & ~((freqs >= NOTCH_HZ[0]) & (freqs <= NOTCH_HZ[1]))


def detect_cohort_lines(grid: Grid) -> pd.DataFrame:
    """Find lines in the cohort-mean prominence spectrum under FDR control."""
    usable = detection_mask(grid.freqs) & np.all(np.isfinite(grid.subject_prominence), axis=0)
    candidates = np.flatnonzero(usable)
    if candidates.size == 0:
        raise RuntimeError("No frequency bin has a usable background estimate.")
    cohort = np.full(grid.freqs.size, np.nan)
    cohort[candidates] = grid.subject_prominence[:, candidates].mean(axis=0)

    pvalues = hd.upper_tail_pvalues(cohort[candidates])
    qvalues = hd.fdr_bh(pvalues)
    significant = np.zeros(grid.freqs.size, dtype=bool)
    significant[candidates] = qvalues < FDR_ALPHA
    q_by_bin = np.ones(grid.freqs.size)
    q_by_bin[candidates] = qvalues

    peaks = hd.cluster_peaks(significant, np.nan_to_num(cohort, nan=-np.inf))
    if not peaks:
        raise NoLinesDetected("No line survived FDR control; nothing to characterise.")

    # Each participant is scored against their own null so prevalence is not driven by
    # whichever participants happen to have the largest lines.
    subject_significant = np.zeros_like(grid.subject_prominence, dtype=bool)
    for index in range(grid.subject_prominence.shape[0]):
        values = grid.subject_prominence[index, candidates]
        subject_significant[index, candidates] = (
            hd.fdr_bh(hd.upper_tail_pvalues(values)) < FDR_ALPHA
        )

    # A percentile bootstrap resamples the sampling unit, so a single participant has no
    # interval to report -- and a lone continuous acquisition is a shape this workflow
    # accepts on purpose. The line is still detected and its position, width and prominence
    # are all still measured; only the between-participant interval is undefined, and it is
    # reported as undefined rather than aborting a verification of data already written.
    single_unit = grid.subject_prominence.shape[0] < 2

    records = []
    for index in peaks:
        values = grid.subject_prominence[:, index]
        if single_unit:
            point, low, high = float(np.median(values)), float("nan"), float("nan")
        else:
            point, low, high = hd.bootstrap_ci(
                values, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
            )
        refined = hd.refine_peak_frequency(grid.freqs, cohort, index)
        position = hd.comb_index(refined)
        neighbourhood = slice(max(index - 1, 0), index + 2)
        linewidth = hd.spectral_linewidth_hz(grid.freqs, cohort, index)
        records.append(
            {
                "bin": index,
                "frequency_hz": float(grid.freqs[index]),
                "refined_hz": refined,
                "linewidth_hz": linewidth,
                "linewidth_over_resolution": linewidth
                / hd.hann_resolution_hz(1.0 / grid.bin_width_hz),
                "is_narrow": bool(
                    linewidth / hd.hann_resolution_hz(1.0 / grid.bin_width_hz)
                    < NARROW_LINEWIDTH_RATIO
                ),
                "cohort_mean_prominence_db": float(cohort[index]),
                "cohort_median_prominence_db": point,
                "ci_low_db": low,
                "ci_high_db": high,
                "q_value": float(q_by_bin[index]),
                "n_subjects_detected": int(
                    np.sum(np.any(subject_significant[:, neighbourhood], axis=1))
                ),
                "n_subjects": int(grid.subject_prominence.shape[0]),
                "comb_harmonic": position.harmonic_index,
                "comb_offset_hz": position.offset_hz,
                "on_comb": bool(abs(position.offset_hz) < grid.bin_width_hz),
            }
        )
    return pd.DataFrame(records).sort_values("frequency_hz").reset_index(drop=True)


def per_subject_prominence(grid: Grid, subjects: list[str], lines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for position, subject in enumerate(subjects):
        for _, line in lines.iterrows():
            rows.append(
                {
                    "subject": subject,
                    "frequency_hz": line["frequency_hz"],
                    "prominence_db": float(grid.subject_prominence[position, int(line["bin"])]),
                }
            )
    return pd.DataFrame(rows)


def control_persistence(
    control: dict,
    matched: Grid,
    lines: pd.DataFrame,
    subjects: list[str],
    bads_by_subject: dict[str, list[str]],
) -> pd.DataFrame:
    """Compare each line's prominence with gradients on against with gradients off.

    Both sides are measured on 5.4 s segments. Line prominence grows with segment
    length, so comparing the 21.6 s catalogue against a 5.4 s control would report a
    difference that is an artifact of the window rather than of the gradients.
    """
    freqs = control["freqs_matched"]
    channel_names = list(control["channel_names"])
    recording_prominence = np.stack(
        [
            prominence_of(
                channel_median(psd, channel_names, bads_by_subject.get(str(subject), [])),
                freqs,
            )
            for psd, subject in zip(control["psd"], control["subjects"])
        ]
    )
    control_subjects = np.asarray(control["subjects"], dtype=str)

    rows = []
    for _, line in lines.iterrows():
        frequency = float(line["refined_hz"])
        control_bin = int(np.argmin(np.abs(freqs - frequency)))
        matched_bin = int(np.argmin(np.abs(matched.freqs - frequency)))
        control_values = np.nanmax(
            recording_prominence[:, control_bin - 1 : control_bin + 2], axis=1
        )
        by_subject = np.array(
            [
                np.nanmedian(control_values[control_subjects == subject])
                for subject in subjects
                if np.any(control_subjects == subject)
            ]
        )
        final_values = np.nanmax(
            matched.subject_prominence[:, matched_bin - 1 : matched_bin + 2], axis=1
        )
        control_point, control_low, control_high = hd.bootstrap_ci(
            by_subject, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
        )
        final_point, final_low, final_high = hd.bootstrap_ci(
            final_values, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
        )
        rows.append(
            {
                "frequency_hz": line["frequency_hz"],
                "refined_hz": frequency,
                "on_comb": bool(line["on_comb"]),
                "final_matched_prominence_db": final_point,
                "final_ci_low_db": final_low,
                "final_ci_high_db": final_high,
                "control_prominence_db": control_point,
                "control_ci_low_db": control_low,
                "control_ci_high_db": control_high,
                "control_minus_final_db": control_point - final_point,
                "control_ci_excludes_zero": bool(control_low > 0),
                "n_control_subjects": int(by_subject.size),
            }
        )
    return pd.DataFrame(rows)


def phase_locking(caches: dict[str, dict], lines: pd.DataFrame) -> pd.DataFrame:
    """Rayleigh test of phase concentration once each epoch is put in the scanner's frame.

    Only a frequency completing a whole number of cycles per TR has a phase that repeats
    from volume to volume, so a high resultant is interpretable as scanner locking only
    on the comb. Off-comb rows are reported as the negative control they are.

    Two resultants are reported. The pooled one uses every epoch, and assumes the
    residual keeps the same phase relative to the volume marker across runs. The
    within-run one drops that assumption: each run has its own sequence start, so a
    per-run phase offset would dilute the pooled estimate while leaving this one intact.
    Eleven epochs per run put the null expectation for the within-run resultant near
    0.27, so it is read against that floor rather than against zero.
    """
    probes = list(lines["refined_hz"]) + [k / hd.TR_SECONDS for k in CLASSIC_REFERENCE_HARMONICS]
    rows = []
    for frequency in probes:
        for subject, cache in caches.items():
            grid = cache["freqs_coefficients"]
            index = int(np.argmin(np.abs(grid - frequency)))
            mask = hc.good_channel_mask(list(cache["channel_names"]), list(cache["bads"]))
            coefficients = cache["coefficients"][:, mask, index]
            offsets = cache["volume_offsets_s"]
            epoch_runs = cache["epoch_runs"]
            pooled, pvalues, within_run = [], [], []
            for channel in range(coefficients.shape[1]):
                phases = hd.volume_locked_phases(
                    coefficients[:, channel], offsets, float(grid[index])
                )
                resultant, pvalue = hd.rayleigh_test(phases)
                pooled.append(resultant)
                pvalues.append(pvalue)
                within_run.append(
                    float(
                        np.mean(
                            [
                                hd.rayleigh_test(phases[epoch_runs == run])[0]
                                for run in np.unique(epoch_runs)
                            ]
                        )
                    )
                )
            position = hd.comb_index(float(grid[index]))
            rows.append(
                {
                    "subject": subject,
                    "probe_hz": float(frequency),
                    "grid_hz": float(grid[index]),
                    "on_comb": bool(abs(position.offset_hz) < float(grid[1] - grid[0])),
                    "median_resultant": float(np.median(pooled)),
                    "max_resultant": float(np.max(pooled)),
                    "median_within_run_resultant": float(np.median(within_run)),
                    "max_within_run_resultant": float(np.max(within_run)),
                    "min_p": float(np.min(pvalues)),
                    "n_channels_p_below_001": int(np.sum(np.asarray(pvalues) < 0.001)),
                    "n_epochs": int(coefficients.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def frequency_stability(
    subjects: list[str],
    lines: pd.DataFrame,
    grid: Grid,
    session_dates: dict[str, str],
) -> pd.DataFrame:
    """Per-participant centre frequency of each line, and its spread across sessions."""
    spectra_db = [hd.to_db(spectrum) for spectrum in grid.subject_psd]
    rows = []
    for _, line in lines.iterrows():
        index = int(line["bin"])
        estimates = []
        for spectrum in spectra_db:
            window = spectrum[index - 2 : index + 3]
            local = int(np.clip(index + int(np.argmax(window)) - 2, 1, spectrum.size - 2))
            estimates.append(hd.refine_peak_frequency(grid.freqs, spectrum, local))
        values = np.array(estimates)
        ordinals = np.array(
            [pd.Timestamp(session_dates[subject]).toordinal() for subject in subjects],
            dtype=float,
        )
        slope = float(np.polyfit(ordinals - ordinals.mean(), values, 1)[0])
        rows.append(
            {
                "frequency_hz": line["frequency_hz"],
                "refined_hz": line["refined_hz"],
                "mean_hz": float(np.mean(values)),
                "sd_hz": float(np.std(values, ddof=1)),
                "range_hz": float(np.ptp(values)),
                "drift_hz_per_day": slope,
                "drift_hz_over_study": slope * float(np.ptp(ordinals)),
                "n_subjects": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def channel_prominence_by_subject(
    caches: dict[str, dict], subjects: list[str], reference: list[str]
) -> np.ndarray:
    """Per-channel prominence spectra, computed once. Shape (n_subjects, n_channels, n_freqs)."""
    stack = []
    for subject in subjects:
        cache = caches[subject]
        order = hc.align_channels(list(cache["channel_names"]), reference)
        psd = cache["psd_high"].mean(axis=0)[order]
        maps = np.stack([prominence_of(channel, cache["freqs_high"]) for channel in psd])
        bad = ~hc.good_channel_mask(reference, list(cache["bads"]))
        maps[bad] = np.nan
        stack.append(maps)
    return np.stack(stack)


def topography_reproducibility(
    channel_prominence: np.ndarray, lines: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """How similar is each line's scalp distribution from one participant to the next?"""
    rows, topographies = [], []
    n_subjects = channel_prominence.shape[0]
    for _, line in lines.iterrows():
        maps = channel_prominence[:, :, int(line["bin"])]
        correlations = []
        for first in range(n_subjects):
            for second in range(first + 1, n_subjects):
                usable = np.isfinite(maps[first]) & np.isfinite(maps[second])
                if usable.sum() > 10:
                    correlations.append(
                        float(np.corrcoef(maps[first, usable], maps[second, usable])[0, 1])
                    )
        topographies.append(np.nanmean(maps, axis=0))
        rows.append(
            {
                "frequency_hz": line["frequency_hz"],
                "mean_pairwise_r": float(np.mean(correlations)) if correlations else np.nan,
                "median_pairwise_r": float(np.median(correlations)) if correlations else np.nan,
                "n_pairs": len(correlations),
            }
        )
    return pd.DataFrame(rows), np.stack(topographies)


def band_impact(grid: Grid, subjects: list[str], lines: pd.DataFrame) -> pd.DataFrame:
    """Fraction of each band's power that is artifact, per participant.

    Measured as excess over the local background at the line bins. Dropping the line bins
    and comparing band powers -- the first version of this -- also drops their background
    and so counts ordinary spectrum as contamination; it put gamma at 47% against a true
    35%. See :func:`harmonic_diagnosis.line_excess_fraction`.

    Only comb members and isolated narrow lines count. The remaining detections are broad,
    weak and present in a few participants; charging those to the artifact would be
    charging it for the brain rhythms the band exists to measure.
    """
    narrow = lines.loc[lines["kind"].isin(("comb", "isolated"))]
    artifact = list(narrow["refined_hz"])
    rows = []
    for position, subject in enumerate(subjects):
        spectrum = grid.subject_psd[position]
        for name, (low, high) in BANDS.items():
            fraction = hd.line_excess_fraction(
                grid.freqs,
                spectrum,
                low_hz=low,
                high_hz=high,
                line_freqs=artifact,
                half_width_bins=grid.half_width_bins,
                line_half_width_hz=LINE_MASK_HALF_WIDTH_HZ,
            )
            rows.append(
                {
                    "subject": subject,
                    "band": name,
                    "low_hz": low,
                    "high_hz": high,
                    "n_lines_inside": sum(1 for f in artifact if low <= f <= high),
                    "n_artifact_lines_total": int(len(narrow)),
                    "artifact_share": fraction,
                    "artifact_share_percent": 100.0 * fraction,
                    "line_contribution_db": float(-10.0 * np.log10(max(1.0 - fraction, 1e-12))),
                }
            )
    return pd.DataFrame(rows)


def run_level_prominence(
    caches: dict[str, dict], subjects: list[str], lines: pd.DataFrame
) -> pd.DataFrame:
    """Prominence of every line in every run, from one prominence pass per run."""
    rows = []
    for subject in subjects:
        cache = caches[subject]
        freqs = cache["freqs_high"]
        for position, run in enumerate(cache["runs"]):
            spectrum = channel_median(
                cache["psd_high"][position], cache["channel_names"], cache["bads"]
            )
            prominence = prominence_of(spectrum, freqs)
            for _, line in lines.iterrows():
                rows.append(
                    {
                        "subject": subject,
                        "run": int(run),
                        "frequency_hz": line["frequency_hz"],
                        "prominence_db": float(prominence[int(line["bin"])]),
                    }
                )
    return pd.DataFrame(rows)


def variance_components(run_level: pd.DataFrame) -> pd.DataFrame:
    """Split each line's run-level prominence into between- and within-participant parts."""
    rows = []
    for frequency, group in run_level.groupby("frequency_hz"):
        components = hd.one_way_variance_components(
            group["prominence_db"].to_numpy(), group["subject"].to_numpy()
        )
        components["frequency_hz"] = float(frequency)
        rows.append(components)
    return pd.DataFrame(rows).sort_values("frequency_hz").reset_index(drop=True)


def comb_structure(lines: pd.DataFrame, tolerance_hz: float = 0.06) -> pd.DataFrame:
    """Recover the comb the narrow lines belong to, and fit its fundamental.

    Only narrow lines take part. The dominant repeated spacing is found first, then each
    line is assigned the nearest harmonic index of that spacing and the fundamental is
    re-estimated by least squares over the assignments. The residuals are the evidence:
    a source with one fixed period leaves residuals at the millihertz level, while a set
    of unrelated peaks that merely happen to fall near a common spacing does not.
    """
    narrow = lines.loc[lines["is_narrow"], "refined_hz"].to_numpy()
    rows: list[dict] = []
    if narrow.size < 3:
        return pd.DataFrame(rows)

    try:
        gap, support = hd.dominant_spacing(
            narrow, max_difference_hz=12.0, tolerance_hz=tolerance_hz
        )
    except ValueError:
        # No repeated spacing to find. That is a result, not a failure, and it must not
        # take the rest of the cohort analysis down with it.
        spacing, support = float("nan"), 0
        members = np.zeros(narrow.size, dtype=bool)
    else:
        spacing, members = hd.refine_comb_fundamental(narrow, gap, tolerance_hz=tolerance_hz)

    if members.sum() >= 3:
        try:
            fit_intercept_hz = hd.fit_arithmetic_comb(narrow[members]).intercept_hz
        except ValueError:
            # Two members inside the tolerance of the *same* harmonic, so rounding puts
            # both on one index and the free-intercept fit cannot be posed. That costs the
            # intercept and nothing else: the fundamental below is fitted through the
            # origin, which is the physically right model anyway.
            #
            # Discarding the family instead was measured to be wrong -- on sub-0008 it
            # threw away 41 members at 1.20004 Hz because two lines sat 0.106 Hz apart, so
            # `verify` reported zero comb lines on *uncleaned* data.
            fit_intercept_hz = float("nan")

        # Re-fit through the origin: a comb generated by one periodic source has lines at
        # exact integer multiples, so the fundamental is the only free parameter.
        harmonics = np.rint(narrow[members] / spacing)
        fundamental = float(np.sum(harmonics * narrow[members]) / np.sum(harmonics**2))
        through_origin = narrow[members] - harmonics * fundamental
        rows.append(
            {
                "family": "narrow_comb",
                "fundamental_hz": fundamental,
                "spacing_from_pairs_hz": spacing,
                "supporting_pairs": support,
                "n_lines": int(members.sum()),
                "harmonic_min": int(harmonics.min()),
                "harmonic_max": int(harmonics.max()),
                "rmse_hz": float(np.sqrt(np.mean(through_origin**2))),
                "max_abs_residual_hz": float(np.max(np.abs(through_origin))),
                "free_intercept_hz": fit_intercept_hz,
                "volume_comb_uniformity_p": hd.comb_uniformity_pvalue(narrow[members]),
            }
        )
    remainder = narrow[~members]
    if remainder.size:
        rows.append(
            {
                "family": "narrow_off_comb",
                "fundamental_hz": np.nan,
                "spacing_from_pairs_hz": np.nan,
                "supporting_pairs": 0,
                "n_lines": int(remainder.size),
                "harmonic_min": -1,
                "harmonic_max": -1,
                "rmse_hz": np.nan,
                "max_abs_residual_hz": np.nan,
                "free_intercept_hz": np.nan,
                "volume_comb_uniformity_p": (
                    hd.comb_uniformity_pvalue(remainder) if remainder.size >= 2 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_lines(
    lines: pd.DataFrame,
    structure: pd.DataFrame,
    *,
    tolerance_hz: float = 0.06,
) -> pd.DataFrame:
    """Label every detection: comb member, isolated narrow line, or other.

    Comb membership comes first and is decided arithmetically, because it is the sharper
    criterion: a detection sitting within a few millihertz of an integer multiple is a
    member whatever its measured width. Width is used only for the detections that are
    not members.

    Deciding membership by width first would be wrong. Half-power width is measured
    against the peak's own height, so a weak line reaches the half-power point further
    out and measures wider than a strong one from the same source -- among comb members
    here the two are correlated at rho = -0.87. Width separates a monochromatic source
    from a brain rhythm only at comparable amplitude.
    """
    comb = (
        structure.loc[structure["family"] == "narrow_comb"]
        if "family" in structure.columns
        else structure
    )
    labelled = lines.copy()
    if not len(comb):
        labelled["comb_harmonic_1p2"] = -1
        labelled["comb_residual_hz"] = np.nan
        labelled["kind"] = np.where(labelled["is_narrow"], "isolated", "other")
        return labelled

    fundamental = float(comb["fundamental_hz"].iloc[0])
    harmonics = np.rint(labelled["refined_hz"] / fundamental)
    residual = labelled["refined_hz"] - harmonics * fundamental
    member = np.abs(residual) <= tolerance_hz

    labelled["comb_harmonic_1p2"] = np.where(member, harmonics, -1).astype(int)
    labelled["comb_residual_hz"] = np.where(member, residual, np.nan)
    wide = labelled["linewidth_over_resolution"] >= WIDE_MEMBER_RATIO
    labelled["kind"] = np.where(
        member,
        np.where(wide, "comb_wide", "comb"),
        np.where(labelled["is_narrow"], "isolated", "other"),
    )
    return labelled


def comb_enrichment(lines: pd.DataFrame, fundamental: float, tolerance_hz: float = 0.06) -> dict:
    """How improbable the comb membership count is if the frequencies were unrelated.

    A frequency drawn without regard to the comb falls within ``tolerance_hz`` of a
    multiple with probability ``2 * tolerance / fundamental``, so the count of members is
    binomial under the null of no comb.
    """
    from scipy.stats import binomtest

    members = int(lines["kind"].isin(("comb", "comb_wide")).sum())
    chance = 2.0 * tolerance_hz / fundamental
    result = binomtest(members, len(lines), chance, alternative="greater")
    return {
        "n_members": members,
        "n_detections": int(len(lines)),
        "chance_rate": float(chance),
        "binomial_p": float(result.pvalue),
    }


def per_subject_fundamental(
    grid: Grid,
    subjects: list[str],
    lines: pd.DataFrame,
    spacing: float,
    tolerance_hz: float = 0.06,
) -> pd.DataFrame:
    """Estimate the comb fundamental separately in every participant.

    One physical oscillator running through all fifteen sessions should give fifteen
    indistinguishable estimates. This is the strongest available statement that the comb
    comes from a single piece of hardware rather than from anything about the person.

    Only the well-resolved members contribute. The ``comb_wide`` detections are counted as
    members elsewhere, but their peaks are not localised well enough to sharpen a frequency
    estimate -- including them widens the between-participant scatter from 58 to 70 uHz
    without adding information.
    """
    members = lines.loc[lines["kind"] == "comb"].copy()
    members["harmonic"] = np.rint(members["refined_hz"] / spacing)

    rows = []
    for position, subject in enumerate(subjects):
        spectrum = hd.to_db(grid.subject_psd[position])
        estimates, harmonics = [], []
        for _, line in members.iterrows():
            index = int(line["bin"])
            window = spectrum[index - 2 : index + 3]
            local = int(np.clip(index + int(np.argmax(window)) - 2, 1, spectrum.size - 2))
            estimates.append(hd.refine_peak_frequency(grid.freqs, spectrum, local))
            harmonics.append(float(line["harmonic"]))
        estimate_array = np.asarray(estimates)
        harmonic_array = np.asarray(harmonics)
        rows.append(
            {
                "subject": subject,
                "fundamental_hz": float(
                    np.sum(harmonic_array * estimate_array) / np.sum(harmonic_array**2)
                ),
                "n_lines_used": int(estimate_array.size),
            }
        )
    return pd.DataFrame(rows)


def read_session_dates(bids_root: Path, subjects: list[str]) -> dict[str, str]:
    dates = {}
    for subject in subjects:
        scans = bids_root / subject / f"{subject}_scans.tsv"
        frame = pd.read_csv(scans, sep="\t")
        dates[subject] = str(frame["acq_time"].iloc[0])[:10]
    return dates


# ------------------------------------------------------------------------------ main


def load_caches(cache_dir: Path, subjects: list[str]) -> dict[str, dict]:
    caches = {}
    for subject in subjects:
        with np.load(cache_dir / f"{subject}_final.npz", allow_pickle=False) as handle:
            caches[subject] = {key: handle[key] for key in handle.files}
    return caches


def run_cache_stage(args: argparse.Namespace, cache_dir: Path) -> None:
    subjects = discover_subjects(args.deriv_root)
    print(f"Caching final epochs for {len(subjects)} participants")
    reference: list[str] = []
    for subject in subjects:
        path = build_final_cache(args.deriv_root, subject, cache_dir)
        with np.load(path, allow_pickle=False) as handle:
            names = list(handle["channel_names"])
            runs = handle["runs"]
            epochs_per_run = handle["epochs_per_run"]
        reference = reference or names
        print(f"  {subject}: runs {list(runs)} epochs {list(epochs_per_run)} ch {len(names)}")
    print("Caching gradient-free control windows")
    build_control_cache(args.source_root, reference, cache_dir)


def run_analysis_stage(args: argparse.Namespace, cache_dir: Path, output_dir: Path) -> None:
    subjects = sorted(path.name.split("_")[0] for path in cache_dir.glob("sub-*_final.npz"))
    caches = load_caches(cache_dir, subjects)
    with np.load(cache_dir / "control.npz", allow_pickle=False) as handle:
        control = {key: handle[key] for key in handle.files}

    reference = list(caches[subjects[0]]["channel_names"])
    bads_by_subject = {s: [str(b) for b in caches[s]["bads"]] for s in subjects}

    high = build_grid(
        caches[subjects[0]]["freqs_high"],
        np.stack(
            [
                channel_median(
                    caches[s]["psd_high"].mean(axis=0),
                    caches[s]["channel_names"],
                    caches[s]["bads"],
                )
                for s in subjects
            ]
        ),
    )
    matched = build_grid(
        caches[subjects[0]]["freqs_matched"],
        np.stack(
            [
                channel_median(
                    caches[s]["psd_matched"].mean(axis=0),
                    caches[s]["channel_names"],
                    caches[s]["bads"],
                )
                for s in subjects
            ]
        ),
    )

    lines = detect_cohort_lines(high)
    print(f"Detected {len(lines)} lines at FDR q < {FDR_ALPHA}")

    session_dates = read_session_dates(args.bids_root, subjects)
    channel_prominence = channel_prominence_by_subject(caches, subjects, reference)
    run_level = run_level_prominence(caches, subjects, lines)
    topography, topo_maps = topography_reproducibility(channel_prominence, lines)

    structure = comb_structure(lines)
    lines = classify_lines(lines, structure)
    comb_rows = structure.loc[structure["family"] == "narrow_comb"]
    if len(comb_rows):
        fundamental = float(comb_rows["fundamental_hz"].iloc[0])
        fundamentals = per_subject_fundamental(high, subjects, lines, fundamental)
        structure = structure.assign(
            **{
                key: [
                    value if family == "narrow_comb" else np.nan for family in structure["family"]
                ]
                for key, value in comb_enrichment(lines, fundamental).items()
            }
        )
    else:
        fundamentals = pd.DataFrame()
    print("  classified: " + ", ".join(f"{k} {v}" for k, v in lines["kind"].value_counts().items()))

    cardiac = pd.DataFrame(
        [
            {
                "subject": subject,
                "run": int(run),
                "heart_rate_bpm": float(rate),
                "heart_rate_hz": float(rate) / 60.0,
            }
            for subject in subjects
            for run, rate in zip(caches[subject]["runs"], caches[subject]["heart_rate_bpm"])
        ]
    )

    frames = {
        "cohort_line_catalog": lines,
        "comb_structure": structure,
        "per_subject_fundamental": fundamentals,
        "cardiac_rate": cardiac,
        "control_persistence": control_persistence(
            control, matched, lines, subjects, bads_by_subject
        ),
        "phase_locking": phase_locking(caches, lines),
        "frequency_stability": frequency_stability(subjects, lines, high, session_dates),
        "topography_reproducibility": topography,
        "band_impact": band_impact(high, subjects, lines),
        "variance_components": variance_components(run_level),
        "per_run_line_prominence": run_level,
        "per_subject_line_prominence": per_subject_prominence(high, subjects, lines),
        "control_windows": pd.DataFrame(json.loads(str(control["all_records"]))),
        "session_dates": pd.DataFrame(
            {"subject": subjects, "session_date": [session_dates[s] for s in subjects]}
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.tsv", sep="\t", index=False, float_format="%.6g")
        print(f"  wrote {name}.tsv ({len(frame)} rows)")

    np.savez_compressed(
        output_dir / "spectra.npz",
        freqs_high=high.freqs,
        subject_psd_high=high.subject_psd,
        subject_prominence_high=high.subject_prominence,
        freqs_matched=matched.freqs,
        subject_psd_matched=matched.subject_psd,
        subject_prominence_matched=matched.subject_prominence,
        control_freqs=control["freqs_matched"],
        control_psd=control["psd"],
        control_subjects=control["subjects"],
        control_kinds=control["kinds"],
        subjects=np.array(subjects),
        line_bins=lines["bin"].to_numpy(),
        line_freqs=lines["refined_hz"].to_numpy(),
        topography_maps=topo_maps,
        channel_names=np.array(reference),
        heart_rate_bpm=np.stack([caches[s]["heart_rate_bpm"] for s in subjects]),
    )
    print("  wrote spectra.npz")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deriv-root", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--bids-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--stage", choices=("cache", "analyse", "all"), default="all")
    args = parser.parse_args(argv)

    run(args)


def run(args: argparse.Namespace) -> None:
    """Execute one stage. Split from ``main`` so the CLI command can call it with its own args."""
    from studies.pain_study.scripts.workflow_config import load_workflow_config

    config = load_workflow_config(WORKFLOW, getattr(args, "config", None))
    args.deriv_root = config.path("preprocessed_eeg", override=args.deriv_root)
    args.source_root = config.path("source_data", override=args.source_root)
    args.bids_root = config.path("bids_root", override=args.bids_root)
    args.output_dir = config.path("diagnosis_dir", override=args.output_dir)

    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("cache", "all"):
        run_cache_stage(args, cache_dir)
    if args.stage in ("analyse", "all"):
        run_analysis_stage(args, cache_dir, args.output_dir)


if __name__ == "__main__":
    main()
