"""Remove the room's line comb from the BIDS EEG runs.

    eeg-pipeline line-comb benchmark
    eeg-pipeline line-comb apply

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

from studies.pain_study.analysis.line_comb import diagnosis as hd
from studies.pain_study.analysis.line_comb import removal as lr

WORKFLOW = "line_comb"

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
    removal_harmonic_range: tuple[int, int] = lr.REMOVAL_HARMONIC_RANGE
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
    detect_isolated: bool = False
    """Find each run's isolated lines in its own spectrum instead of using ``isolated_hz``.

    A cohort-wide list is wrong for somebody by construction. Measured on the uncleaned
    root these lines scatter up to 0.595 Hz between participants while one seed window
    reaches 0.30 Hz, so the 94 Hz line was caught in 11 of 15 and left standing at +20 to
    +28 dB in the rest -- and in sub-0008 the delivered data ended up worse than before
    cleaning, because its neighbours were removed and it was not. Detection reaches it in
    14 of 15, including three of the four the list missed.

    ``isolated_hz`` stays meaningful when this is False, and stays in the config either
    way as the record of what the curated list contained.
    """
    detection_min_prominence_db: float = lr.LINE_PROMINENCE_FLOOR_DB
    detection_low_hz: float = 20.0
    detection_high_hz: float = 100.0
    max_isolated_lines: int = lr.MAX_ISOLATED_LINES
    detection_search_hz: float = 0.05
    """Refinement window for a nominal that came from detection.

    Detected nominals sit on the summit already, so they need only enough room to refine
    sub-bin -- and the window has to stay narrow, because ``estimate_comb`` refuses a
    nominal within ``isolated_search_hz`` of a comb position on the grounds that a search
    that wide would refine onto the harmonic instead. It is right to: at 0.15 Hz it would.
    sub-0001's 93.759 Hz line sits 0.137 Hz from harmonic 78, so the detector offered it
    and the estimator raised, stopping the benchmark.

    Kept below the detector's own floor of one line width, so a line the detector admits
    can never be one the estimator refuses.
    """
    min_runs_per_line: int = 2
    """Runs of a session a line must appear in before it is removed from any of them.

    The runs are replication already in hand -- one machine, minutes apart -- and they
    separate a line from a fluctuation cleanly. On sub-0000, fifteen of twenty candidates
    appeared in exactly one of six runs while the five that recurred are the known lines;
    on sub-0008 all seven appeared in all six. Clamped to the number of runs available, so
    a single-run session can still contribute.
    """
    exclude_mains: bool = True
    """Leave 59.5-60.5 Hz to the pipeline's own notch.

    False moves mains into this pass, which is the point of doing so: the pipeline's FIR
    notch measured 0.97 Hz wide on the delivered epochs (59.537-60.463 Hz) against
    0.133 Hz for spectrum_fit at freq/450. Exactly one of the two may remove mains --
    ``preprocessing.notch_freq`` has to be null when this is False, and
    tests/scripts/line_comb/test_config_pairing.py fails if the two ever disagree.
    """

    @classmethod
    def from_config(cls, config) -> "RemovalSettings":
        """Read ``line_comb_removal`` from the workflow config, falling back to the code's values."""
        defaults = cls()
        block = config.get("line_comb_removal") or {}
        harmonic_range = block.get("harmonic_range", list(defaults.harmonic_range))
        removal_range = block.get("removal_harmonic_range", list(defaults.removal_harmonic_range))
        return cls(
            nominal_fundamental_hz=float(
                block.get("nominal_fundamental_hz", defaults.nominal_fundamental_hz)
            ),
            harmonic_range=(int(harmonic_range[0]), int(harmonic_range[1])),
            removal_harmonic_range=(int(removal_range[0]), int(removal_range[1])),
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
            detect_isolated=bool(block.get("detect_isolated", defaults.detect_isolated)),
            detection_min_prominence_db=float(
                block.get("detection_min_prominence_db", defaults.detection_min_prominence_db)
            ),
            detection_low_hz=float(block.get("detection_low_hz", defaults.detection_low_hz)),
            detection_high_hz=float(block.get("detection_high_hz", defaults.detection_high_hz)),
            max_isolated_lines=int(
                block.get("max_isolated_lines", defaults.max_isolated_lines)
            ),
            min_runs_per_line=int(
                block.get("min_runs_per_line", defaults.min_runs_per_line)
            ),
            detection_search_hz=float(
                block.get("detection_search_hz", defaults.detection_search_hz)
            ),
            exclude_mains=bool(block.get("exclude_mains", defaults.exclude_mains)),
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

    # Channel definitions carry four comma-separated fields: name, reference, resolution,
    # unit. The classes exclude newlines so a `[Coordinates]` line, which holds only three
    # numbers, cannot be run into the one below it and parsed as `"-72\nCh2=1"`.
    names, resolutions = [], []
    for match in re.finditer(r"^Ch(\d+)=([^,\n]*),([^,\n]*),([^,\n]*),", text, flags=re.MULTILINE):
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


def settings_fingerprint(settings: RemovalSettings) -> str:
    """A short stable hash of every setting that changes what the removal does.

    Binds a benchmark to the configuration it measured, so a stale or mismatched
    benchmark.tsv cannot stand in for one describing the settings about to be applied.
    """
    import hashlib
    from dataclasses import asdict

    payload = repr(sorted(asdict(settings).items())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def require_passing_benchmark(path, settings: RemovalSettings) -> None:
    """Refuse to write derived data without a passing benchmark of these settings.

    Each clause here is a way this went wrong in practice rather than a hypothetical. A
    benchmark.tsv from an earlier configuration was read as though it described the current
    one. A benchmark that raised on its second recording left the previous run's file in
    place, so the gates appeared to pass. And an apply was started under gates that were
    later found unable to fail at all.
    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"Refusing to apply: no benchmark at {path}. Run `line-comb benchmark` first; "
            "the criteria are stated before the measurement for a reason."
        )
    frame = pd.read_csv(path, sep="	")
    expected = settings_fingerprint(settings)
    recorded = set(frame.get("settings_fingerprint", pd.Series(dtype=str)).dropna().unique())
    if recorded != {expected}:
        raise RuntimeError(
            f"Refusing to apply: {path} was produced under different settings "
            f"({recorded or 'none recorded'} against {expected}). Re-run the benchmark."
        )
    if not bool(frame["gate_passed"].all()):
        failed = int((~frame["gate_passed"].astype(bool)).sum())
        raise RuntimeError(
            f"Refusing to apply: {failed} of {len(frame)} benchmarked runs did not pass. "
            "A failure means the settings are wrong, not that the criteria should move."
        )


def search_for(settings: RemovalSettings) -> float:
    """The refinement window that matches where the nominals came from.

    Detected nominals are already on the summit and must stay clear of the comb; the
    curated fallback still needs the wide window, because a listed frequency has to find a
    line that has drifted since it was listed.
    """
    return settings.detection_search_hz if settings.detect_isolated else settings.isolated_search_hz


def isolated_nominals(
    freqs, spectrum_db, prominence, settings: RemovalSettings
) -> tuple[float, ...]:
    """The isolated lines to target: detected from this spectrum, or the configured list.

    Detection needs a fundamental to measure clearance against, and the fundamental comes
    from the comb fit, so this fits once with no isolated lines purely to obtain it. The
    detected positions then go back through ``estimate_comb`` like any nominal would, so
    the claim logic and the comb-collision guard apply to them unchanged rather than
    detection acquiring a second, softer path into the target list.
    """
    if not settings.detect_isolated:
        return settings.isolated_hz

    scaffold = lr.estimate_comb(
        freqs,
        spectrum_db,
        prominence,
        nominal_hz=settings.nominal_fundamental_hz,
        harmonic_range=settings.harmonic_range,
        isolated_nominal_hz=(),
        search_hz=settings.search_hz,
        isolated_search_hz=search_for(settings),
        min_prominence_db=settings.min_prominence_db,
    )
    return lr.detect_isolated_lines(
        freqs,
        spectrum_db,
        prominence,
        fundamental_hz=scaffold.fundamental_hz,
        harmonic_range=settings.removal_harmonic_range,
        min_prominence_db=settings.detection_min_prominence_db,
        low_hz=settings.detection_low_hz,
        high_hz=settings.detection_high_hz,
        max_lines=settings.max_isolated_lines,
    )


def estimate_and_targets(
    raw, settings: RemovalSettings, nominals: tuple[float, ...] | None = None
) -> tuple[lr.CombEstimate, tuple[float, ...], np.ndarray]:
    """Estimate and targets for one run, optionally with the session's nominal list.

    ``nominals`` exists so the benchmark can gate what the apply will actually do. Resolved
    per run instead, detection has no recurrence to lean on and offers every one-off peak
    that clears the floor: on sub-0000 that is sixteen candidates against the five its
    session keeps. Benchmarking the sixteen would measure a configuration that never ships.
    """
    freqs, spectrum_db, prominence = run_spectrum(raw)
    if nominals is None:
        nominals = isolated_nominals(freqs, spectrum_db, prominence, settings)
    estimate = lr.estimate_comb(
        freqs,
        spectrum_db,
        prominence,
        nominal_hz=settings.nominal_fundamental_hz,
        harmonic_range=settings.harmonic_range,
        isolated_nominal_hz=nominals,
        search_hz=settings.search_hz,
        isolated_search_hz=search_for(settings),
        min_prominence_db=settings.min_prominence_db,
    )
    targets = lr.removal_frequencies(
        estimate,
        harmonic_range=settings.removal_harmonic_range,
        low_hz=settings.low_hz,
        high_hz=settings.high_hz,
        excluded_hz=(lr.MAINS_NOTCH_HZ,) if settings.exclude_mains else (),
    )
    return estimate, targets, prominence


def benchmark_run(
    vhdr: Path, settings: RemovalSettings, nominals: tuple[float, ...] | None = None
) -> dict:
    """Inject probes, remove the lines, and measure what came back."""
    import mne

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_brainvision(vhdr, preload=True)
    estimate, targets, prominence_before = estimate_and_targets(raw, settings, nominals)
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


def session_nominals(spectra, settings: RemovalSettings) -> tuple[float, ...]:
    """One nominal list for the whole session, detected from every run in it.

    Detection has to be resolved per session rather than per run, because the pooling in
    ``combine_estimates`` lines estimates up position by position and refuses a session
    whose runs disagree on how many isolated lines they carry. Detecting separately in each
    run would produce exactly that disagreement whenever a line sits just either side of
    the prominence floor in different runs.

    A line found in any run is taken for the session: the runs are minutes apart on one
    machine, so a line present in one and marginal in another is the same line, and a
    nominal that resolves to nothing in a given run contributes nothing to it.
    """
    if not settings.detect_isolated:
        return settings.isolated_hz

    found: list[dict] = []
    for freqs, spectrum_db, prominence in spectra:
        frequency_array = np.asarray(freqs, dtype=float)
        prominence_array = np.asarray(prominence, dtype=float)
        for position in isolated_nominals(freqs, spectrum_db, prominence, settings):
            strength = float(prominence_array[int(np.argmin(np.abs(frequency_array - position)))])
            for entry in found:
                if abs(position - entry["hz"]) <= lr._LINE_CLAIM_HZ:
                    entry["runs"] += 1
                    # Keep the run that saw it most clearly, so the ranking below compares
                    # each line at its best rather than at whichever run came first.
                    if strength > entry["db"]:
                        entry["hz"], entry["db"] = float(position), strength
                    break
            else:
                found.append({"hz": float(position), "db": strength, "runs": 1})

    # A line has to show up in more than one run of the session. The runs are the
    # replication already in hand: one machine, minutes apart, so a real line does not come
    # and go. Measured on sub-0000, fifteen of its twenty candidates appeared in exactly one
    # of six runs while the five that recurred are the known lines -- 28.278, 57.296,
    # 58.185, 82.204 and 93.944 Hz. sub-0008 is the clean case, all seven of its lines in
    # all six runs. Without this the cap was arbitrating between noise entries.
    required = min(settings.min_runs_per_line, len(spectra))
    recurring = [entry for entry in found if entry["runs"] >= required]

    # Rank on strength before applying the budget. Truncating the frequency-ordered list
    # instead spends the budget on whatever sits lowest in the spectrum: on sub-0000 that
    # dropped 93.944 Hz -- the line this detection exists to catch -- to keep 20.037 Hz.
    strongest = sorted(recurring, key=lambda e: (-e["db"], e["hz"]))[: settings.max_isolated_lines]
    return tuple(sorted(entry["hz"] for entry in strongest))


def estimate_session(vhdrs, settings: RemovalSettings):
    """Per-run estimates for one session, and the pooled estimate used to clean it."""
    import mne

    mne.set_log_level("ERROR")
    spectra = []
    for vhdr in vhdrs:
        raw = mne.io.read_raw_brainvision(vhdr, preload=True)
        spectra.append(run_spectrum(raw))

    nominals = session_nominals(spectra, settings)
    per_run = [
        lr.estimate_comb(
            freqs,
            spectrum_db,
            prominence,
            nominal_hz=settings.nominal_fundamental_hz,
            harmonic_range=settings.harmonic_range,
            isolated_nominal_hz=nominals,
            search_hz=settings.search_hz,
            isolated_search_hz=search_for(settings),
            min_prominence_db=settings.min_prominence_db,
        )
        for freqs, spectrum_db, prominence in spectra
    ]
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
        harmonic_range=settings.removal_harmonic_range,
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
    from studies.pain_study.scripts.line_comb import diagnose as ds

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
        except ds.NoLinesDetected:
            # A clean stage. Anything else -- no usable window, no usable background --
            # is the analysis failing and must not be written here as zero lines.
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
    parser.add_argument("--bids-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
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

    run(args)


def run(args: argparse.Namespace) -> None:
    """Execute one stage. Split from ``main`` so the CLI command can call it with its own args."""
    from studies.pain_study.scripts.workflow_config import load_workflow_config

    config = load_workflow_config(WORKFLOW, getattr(args, "config", None))
    args.bids_root = config.path("bids_root", override=args.bids_root)
    args.output_root = config.path("output_root", override=args.output_root)
    args.report_dir = config.path("removal_dir", override=args.report_dir)

    settings = RemovalSettings.from_config(config)
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

        # Resolve each sampled run's session list first, so the gates are measured on the
        # targets the apply will use. Detection resolved per run has no recurrence to lean
        # on and offers every one-off peak clearing the floor -- sixteen candidates on
        # sub-0000 against the five its session keeps -- so benchmarking without this would
        # gate a configuration that never ships.
        session_lists: dict[str, tuple[float, ...]] = {}
        if settings.detect_isolated:
            import mne

            mne.set_log_level("ERROR")
            for vhdr in sample:
                subject = vhdr.stem.split("_")[0]
                if subject in session_lists:
                    continue
                siblings = sorted(vhdr.parent.glob(f"{subject}_task-*_run-*_eeg.vhdr"))
                spectra = [
                    run_spectrum(mne.io.read_raw_brainvision(s, preload=True))
                    for s in siblings
                ]
                session_lists[subject] = session_nominals(spectra, settings)
                print(
                    f"  {subject}: {len(siblings)} runs -> "
                    f"{len(session_lists[subject])} isolated lines",
                    flush=True,
                )

        rows = []
        for index, vhdr in enumerate(sample, start=1):
            started = time.time()
            row = benchmark_run(vhdr, settings, session_lists.get(vhdr.stem.split("_")[0]))
            rows.append(row)
            print(
                f"[{index}/{len(sample)}] {vhdr.stem[:44]:44s} "
                f"f0={row['fundamental_hz']:.6f} suppress={row['median_suppression_db']:5.1f} dB "
                f"probe={row['max_probe_deviation_db']:.3f} dB burst={row['burst_energy_ratio']:.3f} "
                f"{'PASS' if row['gate_passed'] else 'FAIL'} ({time.time()-started:.0f}s)"
            )
        frame = pd.DataFrame(rows)
        frame["settings_fingerprint"] = settings_fingerprint(settings)
        frame.to_csv(args.report_dir / "benchmark.tsv", sep="\t", index=False, float_format="%.6g")
        gate_columns = [c for c in frame.columns if c.startswith("gate_") and c != "gate_passed"]
        print(f"\npassed {int(frame.gate_passed.sum())}/{len(frame)} runs")
        for column in gate_columns:
            print(f"  {column:32s} {int(frame[column].sum())}/{len(frame)}")
        print(f"  wrote {args.report_dir/'benchmark.tsv'}")
        return

    # Nothing is written before this passes. The benchmark's criteria are stated ahead of
    # the measurement precisely so that they can refuse; letting apply run regardless made
    # them advisory.
    require_passing_benchmark(args.report_dir / "benchmark.tsv", settings)
    print(f"Benchmark {settings_fingerprint(settings)} passed; applying.")

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
