"""Remove the room's line comb from the BIDS EEG runs.

    python -m studies.pain_study.scripts.remove_line_comb --stage benchmark
    python -m studies.pain_study.scripts.remove_line_comb --stage apply

``benchmark`` injects known signals into a sample of runs, removes the lines, and reports
what survived against the criteria in :class:`PreservationGate`. ``apply`` writes a cleaned
copy of the BIDS dataset. Run the benchmark first; the criteria are stated before the
measurement, and a failure means the settings are wrong, not that the criteria should move.

The cleaned dataset keeps every sidecar byte-identical and rewrites only the ``.eeg``
binaries. Sampling rate, channel set, length and annotations are untouched, so the BIDS
contract downstream tooling relies on -- including the ``Volume`` and ``R`` marker names
the pipeline reads from ``events.tsv`` -- cannot drift.
"""

from __future__ import annotations

import argparse
import re
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from studies.pain_study.analysis import harmonic_diagnosis as hd
from studies.pain_study.analysis import line_removal as lr

DEFAULT_BIDS_ROOT = Path("/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg")
DEFAULT_OUTPUT_ROOT = Path("/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg_linecleaned")
DEFAULT_REPORT_DIR = Path("outputs/line_comb_removal")

TASK = "thermalactive"
ESTIMATION_TR_COUNT = 60  # 54 s segments -> 18.5 mHz bins
BACKGROUND_HALF_WIDTH_HZ = 100.0 / 21.6
FILTER_LENGTH = "20s"
ROUNDTRIP_RELATIVE_TOLERANCE = 1e-6
"""Largest round-trip error accepted when reading a written binary back.

The binaries are float32, whose 24-bit mantissa gives a relative precision near 6e-8 of
full scale. A decade of headroom above that distinguishes quantisation from corruption.
"""
NOTCH_WIDTH_RATIO = 450.0
NOTCH_WIDTH_MIN_HZ = 0.05
"""Width around each target within which bins are subtracted: ``freq / ratio``.

This, not ``mt_bandwidth``, decides how much spectrum the removal takes, because
spectrum_fit subtracts a sinusoid at every bin inside it. The width scales with frequency
because the uncertainty does: the comb is mains-locked, so harmonic *k* inherits *k*
times the fundamental's wander.

A sweep on sub-0009 run-1 fixed the constant. Reading the narrowest setting that still
pushes every line below its local background:

    f/200 (MNE default)  25.3% of band   worst line -6.6 dB
    f/300                17.1%           worst line -6.1 dB
    f/450                12.1%           worst line -4.3 dB   <- chosen
    f/600                 8.4%           worst line +4.0 dB   (a line survives)
    fixed 0.10 Hz         8.8%           worst line +4.1 dB   (a line survives)

Below f/450 the high harmonics escape the window their own wander needs. The minimum of
0.05 Hz is one bin at the 20 s working resolution, which the lowest harmonics need.
"""
MT_BANDWIDTH = 0.6
"""Multitaper bandwidth for the sinusoid estimate, in Hz.

At 0.6 Hz the estimation band reaches +/-0.3 Hz, half the distance to the neighbouring
comb line, so no line's amplitude is estimated from a band containing another. A sweep
over 4/10/20 s windows and 0.3/0.6/1.0 Hz bandwidths put every 10 s and 20 s setting
inside the gate; 4 s failed, leaving lines up to 11 dB above background because a 4 s
window cannot resolve a 1.2 Hz spacing.
"""


@dataclass(frozen=True)
class RemovalSettings:
    """Everything the removal needs, resolved from configuration."""

    nominal_fundamental_hz: float = lr.NOMINAL_FUNDAMENTAL_HZ
    harmonic_range: tuple[int, int] = lr.COMB_HARMONIC_RANGE
    isolated_hz: tuple[float, ...] = lr.ISOLATED_NOMINAL_HZ
    search_hz: float = 0.25
    isolated_search_hz: float = 0.15
    min_prominence_db: float = 1.0
    filter_length: str = FILTER_LENGTH
    mt_bandwidth: float = MT_BANDWIDTH
    notch_width_ratio: float = NOTCH_WIDTH_RATIO
    notch_width_min_hz: float = NOTCH_WIDTH_MIN_HZ
    low_hz: float = 3.0
    high_hz: float = 95.0

    @classmethod
    def from_config(cls, config) -> "RemovalSettings":
        """Read ``preprocessing.line_comb_removal``, falling back to this study's values."""
        defaults = cls()
        block = config.get("preprocessing.line_comb_removal") or {}
        harmonic_range = block.get("harmonic_range", list(defaults.harmonic_range))
        return cls(
            nominal_fundamental_hz=float(
                block.get("nominal_fundamental_hz", defaults.nominal_fundamental_hz)
            ),
            harmonic_range=(int(harmonic_range[0]), int(harmonic_range[1])),
            isolated_hz=tuple(float(f) for f in block.get("isolated_hz", defaults.isolated_hz)),
            search_hz=float(block.get("search_hz", defaults.search_hz)),
            isolated_search_hz=float(block.get("isolated_search_hz", defaults.isolated_search_hz)),
            min_prominence_db=float(block.get("min_prominence_db", defaults.min_prominence_db)),
            filter_length=str(block.get("filter_length", defaults.filter_length)),
            mt_bandwidth=float(block.get("mt_bandwidth", defaults.mt_bandwidth)),
            notch_width_ratio=float(block.get("notch_width_ratio", defaults.notch_width_ratio)),
            notch_width_min_hz=float(block.get("notch_width_min_hz", defaults.notch_width_min_hz)),
            low_hz=float(block.get("low_hz", defaults.low_hz)),
            high_hz=float(block.get("high_hz", defaults.high_hz)),
        )


def parse_channel_scaling(vhdr_path: Path) -> tuple[list[str], np.ndarray]:
    """Channel names and their binary resolution, in the file's own unit."""
    text = vhdr_path.read_text(encoding="utf-8", errors="replace")
    binary_format = re.search(r"BinaryFormat=(\S+)", text)
    orientation = re.search(r"DataOrientation=(\S+)", text)
    if binary_format is None or binary_format.group(1) != "IEEE_FLOAT_32":
        raise ValueError(f"{vhdr_path.name}: expected IEEE_FLOAT_32 binary data.")
    if orientation is None or orientation.group(1) != "MULTIPLEXED":
        raise ValueError(f"{vhdr_path.name}: expected MULTIPLEXED data orientation.")

    names, resolutions = [], []
    for match in re.finditer(r"^Ch(\d+)=([^,]*),([^,]*),([^,]*),", text, flags=re.MULTILINE):
        names.append(match.group(2))
        resolutions.append(float(match.group(4)))
    if not names:
        raise ValueError(f"{vhdr_path.name}: no channel definitions found.")
    return names, np.asarray(resolutions, dtype=float)


def write_eeg_binary(vhdr_path: Path, destination: Path, data_volts: np.ndarray) -> None:
    """Write one ``.eeg`` binary in the layout its existing header already describes."""
    names, resolutions = parse_channel_scaling(vhdr_path)
    array = np.asarray(data_volts, dtype=float)
    if array.shape[0] != len(names):
        raise ValueError(
            f"{vhdr_path.name}: header describes {len(names)} channels, got {array.shape[0]}."
        )
    scaled = (array * 1e6) / resolutions[:, None]
    scaled.T.astype("<f4").tofile(destination)


def mirror_sidecars(source_root: Path, output_root: Path) -> int:
    """Copy every BIDS file except the binaries, which get rewritten."""
    copied = 0
    for path in sorted(source_root.rglob("*")):
        if path.is_dir() or path.suffix in {".eeg", ".lock"}:
            continue
        target = output_root / path.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def run_spectrum(raw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Channel-median spectrum of the EEG, on a TR-commensurate grid."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    sfreq = float(raw.info["sfreq"])
    block = int(round(hd.TR_SECONDS * sfreq)) * ESTIMATION_TR_COUNT
    data = raw.get_data(picks=picks)
    n_blocks = data.shape[-1] // block
    if n_blocks < 1:
        raise ValueError("Recording is shorter than one estimation block.")
    blocks = data[..., : n_blocks * block].reshape(len(picks), n_blocks, block)
    freqs, psd = hd.hann_periodogram(blocks, sfreq)
    spectrum = np.median(psd.mean(axis=1), axis=0)
    spectrum_db = hd.to_db(spectrum)
    half_width = int(round(BACKGROUND_HALF_WIDTH_HZ / float(freqs[1])))
    return freqs, spectrum_db, hd.prominence_db(spectrum_db, half_width_bins=half_width)


def clean_raw(raw, targets, *, filter_length: str, mt_bandwidth: float, notch_widths):
    """Project the listed frequencies out of the EEG channels.

    ``notch_widths`` is always passed explicitly. Left to its default it becomes
    ``freq / 200``, which turns a line removal into a band removal.
    """
    import warnings

    with warnings.catch_warnings():
        # scipy's DPSS eigenvalue side-computation overflows on long windows. The tapers
        # themselves are finite and orthonormal to 5e-4, and spectrum_fit uses only the
        # tapers, never the eigenvalues.
        warnings.filterwarnings("ignore", message=".*matmul", category=RuntimeWarning)
        return raw.notch_filter(
            freqs=list(targets),
            picks="eeg",
            method="spectrum_fit",
            filter_length=filter_length,
            mt_bandwidth=mt_bandwidth,
            notch_widths=notch_widths,
            verbose="ERROR",
        )


def estimate_and_targets(
    raw, settings: RemovalSettings
) -> tuple[lr.CombEstimate, tuple[float, ...], np.ndarray]:
    freqs, spectrum_db, prominence = run_spectrum(raw)
    estimate = lr.estimate_comb(
        freqs,
        spectrum_db,
        prominence,
        nominal_hz=settings.nominal_fundamental_hz,
        harmonic_range=settings.harmonic_range,
        isolated_nominal_hz=settings.isolated_hz,
        search_hz=settings.search_hz,
        isolated_search_hz=settings.isolated_search_hz,
        min_prominence_db=settings.min_prominence_db,
    )
    targets = lr.removal_frequencies(
        estimate,
        harmonic_range=settings.harmonic_range,
        low_hz=settings.low_hz,
        high_hz=settings.high_hz,
    )
    return estimate, targets, prominence


def benchmark_run(vhdr: Path, settings: RemovalSettings) -> dict:
    """Inject probes, remove the lines, and measure what came back."""
    import mne

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_brainvision(vhdr, preload=True)
    estimate, targets, prominence_before = estimate_and_targets(raw, settings)
    probe = lr.Probe()
    lr.check_probe_clearance(probe, targets)

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    times = raw.times
    waveform = probe.waveform(times)

    with_probe = raw.copy()
    with_probe._data[picks] += waveform[None, :]
    probe_only = raw.copy()
    probe_only._data[picks] = np.tile(waveform, (len(picks), 1))

    passes = {
        "filter_length": settings.filter_length,
        "mt_bandwidth": settings.mt_bandwidth,
        "notch_widths": lr.notch_widths_for(
            targets, ratio=settings.notch_width_ratio, minimum_hz=settings.notch_width_min_hz
        ),
    }
    cleaned_with = clean_raw(with_probe, targets, **passes)
    cleaned_bare = clean_raw(raw.copy(), targets, **passes)
    cleaned_probe = clean_raw(probe_only, targets, **passes)

    freqs, _, prominence_after = run_spectrum(cleaned_bare)
    _, psd_before = _psd(with_probe, picks)
    _, psd_after = _psd(cleaned_with, picks)

    recovered = lr.recover_probe(
        cleaned_with.get_data(picks=picks), cleaned_bare.get_data(picks=picks)
    )
    metrics = {
        **lr.line_suppression(freqs, prominence_before, prominence_after, targets),
        **lr.probe_preservation(freqs, psd_before, psd_after, probe),
        "max_nonline_change_db": float(
            np.max(
                np.abs(
                    lr.nonline_change_db(
                        freqs,
                        psd_before,
                        psd_after,
                        targets,
                        guard_hz=float(np.max(passes["notch_widths"])),
                    )
                )
            )
        ),
        "removed_band_fraction": lr.removed_band_fraction(freqs, targets, passes["notch_widths"]),
        **lr.probe_recovery(recovered, cleaned_probe.get_data(picks=picks)[0], times, probe),
    }
    verdict = lr.PreservationGate().evaluate(metrics)
    return {
        "recording": vhdr.stem,
        "fundamental_hz": estimate.fundamental_hz,
        "n_harmonics": estimate.n_harmonics,
        "n_targets": len(targets),
        **metrics,
        **{f"gate_{name}": value for name, value in verdict.items()},
        "gate_passed": all(verdict.values()),
    }


def _psd(raw, picks):
    sfreq = float(raw.info["sfreq"])
    block = int(round(hd.TR_SECONDS * sfreq)) * ESTIMATION_TR_COUNT
    data = raw.get_data(picks=picks)
    n_blocks = data.shape[-1] // block
    blocks = data[..., : n_blocks * block].reshape(data.shape[0], n_blocks, block)
    freqs, psd = hd.hann_periodogram(blocks, sfreq)
    return freqs, psd.mean(axis=1)


def estimate_session(vhdrs, settings: RemovalSettings):
    """Per-run estimates for one session, and the pooled estimate used to clean it."""
    import mne

    mne.set_log_level("ERROR")
    per_run = []
    for vhdr in vhdrs:
        raw = mne.io.read_raw_brainvision(vhdr, preload=True)
        freqs, spectrum_db, prominence = run_spectrum(raw)
        per_run.append(
            lr.estimate_comb(
                freqs,
                spectrum_db,
                prominence,
                nominal_hz=settings.nominal_fundamental_hz,
                harmonic_range=settings.harmonic_range,
                isolated_nominal_hz=settings.isolated_hz,
                search_hz=settings.search_hz,
                isolated_search_hz=settings.isolated_search_hz,
                min_prominence_db=settings.min_prominence_db,
            )
        )
    return per_run, lr.combine_estimates(per_run)


def apply_run(
    vhdr: Path,
    output_root: Path,
    bids_root: Path,
    settings: RemovalSettings,
    estimate: lr.CombEstimate,
    per_run_estimate: lr.CombEstimate,
):
    """Clean one run with the session's frequencies and write its binary."""
    import mne

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_brainvision(vhdr, preload=True)
    _, _, prominence_before = run_spectrum(raw)
    targets = lr.removal_frequencies(
        estimate,
        harmonic_range=settings.harmonic_range,
        low_hz=settings.low_hz,
        high_hz=settings.high_hz,
    )
    widths = lr.notch_widths_for(
        targets, ratio=settings.notch_width_ratio, minimum_hz=settings.notch_width_min_hz
    )
    cleaned = clean_raw(
        raw.copy(),
        targets,
        filter_length=settings.filter_length,
        mt_bandwidth=settings.mt_bandwidth,
        notch_widths=widths,
    )

    destination = output_root / vhdr.relative_to(bids_root).with_suffix(".eeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_eeg_binary(vhdr, destination, cleaned.get_data())

    verify = mne.io.read_raw_brainvision(output_root / vhdr.relative_to(bids_root), preload=True)
    expected = cleaned.get_data()
    deviation = float(np.max(np.abs(verify.get_data() - expected)))
    scale = float(np.max(np.abs(expected)))
    # The binary is float32, so a round trip loses about 2^-24 of full scale. Anything
    # beyond a decade above that is corruption, not quantisation.
    tolerance = ROUNDTRIP_RELATIVE_TOLERANCE * scale
    if deviation > tolerance:
        raise RuntimeError(
            f"{vhdr.name}: written data differs by {deviation:.3e} V, "
            f"above the {tolerance:.3e} V float32 round-trip tolerance."
        )

    freqs, _, prominence_after = run_spectrum(cleaned)
    suppression = lr.line_suppression(freqs, prominence_before, prominence_after, targets)
    return {
        "removed_band_fraction": lr.removed_band_fraction(freqs, targets, widths),
        "recording": vhdr.stem,
        "fundamental_hz": estimate.fundamental_hz,
        "run_fundamental_hz": per_run_estimate.fundamental_hz,
        "residual_rms_hz": per_run_estimate.residual_rms_hz,
        "n_harmonics": per_run_estimate.n_harmonics,
        "n_targets": len(targets),
        "isolated_hz": ";".join(f"{f:.4f}" for f in estimate.isolated_hz),
        **suppression,
        "roundtrip_max_deviation_v": deviation,
        "roundtrip_relative": deviation / scale if scale else 0.0,
    }


def verify_cohort(bids_root: Path, cleaned_root: Path, settings: RemovalSettings, runs):
    """Run the diagnosis's own line detector over cleaned and original data alike.

    The manifest reports each run against its own targets. This asks the question the
    diagnosis asked: sweeping the whole band with FDR control and no knowledge of where
    the lines were, what is still detectable?
    """
    import mne

    mne.set_log_level("ERROR")
    from studies.pain_study.scripts import diagnose_scanner_harmonics as ds

    rows, spectra = [], {"original": [], "cleaned": []}
    for vhdr in runs:
        for label, root in (("original", bids_root), ("cleaned", cleaned_root)):
            raw = mne.io.read_raw_brainvision(root / vhdr.relative_to(bids_root), preload=True)
            freqs, spectrum_db, prominence = run_spectrum(raw)
            spectra[label].append(10 ** (spectrum_db / 10.0))
        rows.append(vhdr.stem)

    grids = {label: ds.build_grid(freqs, np.stack(values)) for label, values in spectra.items()}
    report = []
    for label, grid in grids.items():
        try:
            lines = ds.detect_cohort_lines(grid)
        except RuntimeError:
            report.append(
                {"stage": label, "n_lines": 0, "n_comb_lines": 0, "max_prominence_db": float("nan")}
            )
            continue
        classified = ds.classify_lines(lines, ds.comb_structure(lines))
        report.append(
            {
                "stage": label,
                "n_lines": int(len(classified)),
                "n_comb_lines": int(classified.kind.isin(("comb", "comb_wide")).sum()),
                "n_isolated": int((classified.kind == "isolated").sum()),
                "max_prominence_db": float(classified.cohort_median_prominence_db.max()),
                "median_prominence_db": float(classified.cohort_median_prominence_db.median()),
            }
        )
    return pd.DataFrame(report), grids


def discover_runs(bids_root: Path, subjects: list[str] | None) -> list[Path]:
    paths = sorted(bids_root.glob(f"sub-*/eeg/sub-*_task-{TASK}_run-*_eeg.vhdr"))
    if subjects:
        wanted = set(subjects)
        paths = [p for p in paths if p.parent.parent.name in wanted]
    if not paths:
        raise FileNotFoundError(f"No task runs found under {bids_root}")
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--stage", choices=("benchmark", "apply", "verify"), default="benchmark")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None, help="benchmark: runs to sample")
    parser.add_argument(
        "--fundamental-scope",
        choices=("session", "run"),
        default="session",
        help="pool the frequency estimate over a session (default) or use each run's own",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--filter-length", default=None)
    parser.add_argument("--mt-bandwidth", type=float, default=None)
    args = parser.parse_args(argv)

    from eeg_pipeline.utils.config.loader import load_config

    settings = RemovalSettings.from_config(
        load_config(args.config) if args.config else load_config()
    )
    overrides = {}
    if args.filter_length is not None:
        overrides["filter_length"] = args.filter_length
    if args.mt_bandwidth is not None:
        overrides["mt_bandwidth"] = args.mt_bandwidth
    if overrides:
        settings = replace(settings, **overrides)
    print(f"settings: {settings}")

    runs = discover_runs(args.bids_root, args.subjects)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "verify":
        # One run per participant keeps the participant the unit of inference.
        by_subject: dict[str, Path] = {}
        for vhdr in runs:
            by_subject.setdefault(vhdr.parent.parent.name, vhdr)
        sample = list(by_subject.values())
        print(f"Verifying on {len(sample)} runs, one per participant")
        report, grids = verify_cohort(args.bids_root, args.output_root, settings, sample)
        report.to_csv(
            args.report_dir / "verification.tsv", sep="\t", index=False, float_format="%.6g"
        )
        print(report.to_string(index=False))
        np.savez_compressed(
            args.report_dir / "verification_spectra.npz",
            freqs=grids["original"].freqs,
            original=grids["original"].subject_psd,
            cleaned=grids["cleaned"].subject_psd,
            recordings=np.array([p.stem for p in sample]),
        )
        print(f"  wrote {args.report_dir/'verification.tsv'}")
        return

    if args.stage == "benchmark":
        # One run per participant unless told otherwise, so the sample spans the cohort.
        sample = runs if args.limit is None else runs[:: max(len(runs) // args.limit, 1)]
        rows = []
        for index, vhdr in enumerate(sample, start=1):
            started = time.time()
            row = benchmark_run(vhdr, settings)
            rows.append(row)
            print(
                f"[{index}/{len(sample)}] {vhdr.stem[:44]:44s} "
                f"f0={row['fundamental_hz']:.6f} suppress={row['median_suppression_db']:5.1f} dB "
                f"probe={row['max_probe_deviation_db']:.3f} dB burst={row['burst_energy_ratio']:.3f} "
                f"{'PASS' if row['gate_passed'] else 'FAIL'} ({time.time()-started:.0f}s)"
            )
        frame = pd.DataFrame(rows)
        frame.to_csv(args.report_dir / "benchmark.tsv", sep="\t", index=False, float_format="%.6g")
        gate_columns = [c for c in frame.columns if c.startswith("gate_") and c != "gate_passed"]
        print(f"\npassed {int(frame.gate_passed.sum())}/{len(frame)} runs")
        for column in gate_columns:
            print(f"  {column:32s} {int(frame[column].sum())}/{len(frame)}")
        print(f"  wrote {args.report_dir/'benchmark.tsv'}")
        return

    print(f"Mirroring sidecars into {args.output_root}")
    print(f"  copied {mirror_sidecars(args.bids_root, args.output_root)} files")

    by_subject: dict[str, list[Path]] = {}
    for vhdr in runs:
        by_subject.setdefault(vhdr.parent.parent.name, []).append(vhdr)

    rows, index = [], 0
    for subject, vhdrs in by_subject.items():
        per_run, pooled = estimate_session(vhdrs, settings)
        session = pooled if args.fundamental_scope == "session" else None
        spread = (
            float(np.std([e.fundamental_hz for e in per_run], ddof=1)) if len(per_run) > 1 else 0.0
        )
        print(
            f"{subject}: session f0 = {session.fundamental_hz:.7f} Hz "
            f"(per-run spread {spread*1e6:.0f} uHz over {len(per_run)} runs)"
        )
        for vhdr, estimate in zip(vhdrs, per_run):
            index += 1
            started = time.time()
            rows.append(
                apply_run(
                    vhdr,
                    args.output_root,
                    args.bids_root,
                    settings,
                    session if session is not None else estimate,
                    estimate,
                )
            )
            print(
                f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} "
                f"suppress={rows[-1]['median_suppression_db']:5.1f} dB "
                f"max_resid={rows[-1]['max_residual_prominence_db']:6.2f} dB "
                f"({time.time()-started:.0f}s)"
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        args.report_dir / "removal_manifest.tsv", sep="\t", index=False, float_format="%.6g"
    )
    sessions = frame.groupby(frame.recording.str.split("_").str[0]).fundamental_hz.first()
    print(
        f"\nsession fundamental across {len(sessions)} participants: "
        f"mean {sessions.mean():.7f} Hz, SD {sessions.std(ddof=1)*1e6:.0f} uHz"
    )
    print(
        f"per-run estimates before pooling: SD {frame.run_fundamental_hz.std(ddof=1)*1e6:.0f} uHz"
    )
    print(
        f"median suppression {frame.median_suppression_db.median():.1f} dB; "
        f"worst residual line {frame.max_residual_prominence_db.max():.2f} dB"
    )
    print(f"  wrote {args.report_dir/'removal_manifest.tsv'}")


if __name__ == "__main__":
    main()
