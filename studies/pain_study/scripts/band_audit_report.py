"""Per-band costs on a set of preprocessed epochs, for before/after comparison.

Reads the delivered epochs, takes the channel-median spectrum of 21.6 s TR-commensurate
segments, and reports the three costs from :mod:`band_audit` for each analysis band.

Run it once before a cleaning change and once after. The value is in the difference: a
band whose ``independent_pct`` falls is one the line removal reached, and a band whose
``hole_pct`` rises is one the removal has started eating.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from studies.pain_study.analysis import band_audit as ba
from studies.pain_study.analysis.line_comb import diagnosis as hd

TR = 0.9
KEEP_HIGH_HZ = 125.0

#: Narrowband features belonging to neither comb, measured on the delivered epochs.
INDEPENDENT_HZ = (23.7776, 29.6854, 46.5839, 57.1925, 59.0168, 61.0353, 61.4039, 99.5982)
#: Gradient volume comb. The centres are nulled by the volume-average subtraction; what is
#: measured here is the sideband energy beside them.
COMB_HZ = tuple(k / TR for k in range(1, 91))

BANDS = {
    "delta 1-4": (1.0, 4.0),
    "theta 4-8": (4.0, 8.0),
    "alpha 8-13": (8.0, 13.0),
    "beta 13-30": (13.0, 30.0),
    "gamma 30.1-45": (30.1, 45.0),
    "gamma 45-58": (45.0, 58.0),
    "notch gap 58-62": (58.0, 62.0),
    "gamma 62-95": (62.0, 95.0),
    "top 95-100": (95.0, 100.0),
}


def subject_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Channel-median spectrum over good EEG channels, on the TR-commensurate grid."""
    epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
    bads = set(epochs.info["bads"])
    names = [
        name
        for name, kind in zip(epochs.ch_names, epochs.get_channel_types())
        if kind == "eeg" and name not in bads
    ]
    if not names:
        raise ValueError(f"{path.name} has no good EEG channel.")

    data = epochs.copy().pick(names).get_data(copy=False)
    keep = hd.tr_commensurate_length(data.shape[-1], epochs.info["sfreq"])
    freqs, psd = hd.hann_periodogram(data[..., :keep], epochs.info["sfreq"])
    band = freqs <= KEEP_HIGH_HZ
    return freqs[band], np.median(psd.mean(axis=0)[:, band], axis=0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deriv-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude", nargs="*", default=["sub-0008"])
    args = parser.parse_args(argv)

    mne.set_log_level("ERROR")
    subjects = sorted(
        p.name
        for p in args.deriv_root.glob("sub-*")
        if p.is_dir() and not p.name.startswith("._") and p.name not in args.exclude
    )
    if not subjects:
        raise SystemExit(f"No subjects found under {args.deriv_root}")

    rows = []
    for subject in subjects:
        path = args.deriv_root / subject / "eeg" / f"{subject}_task-thermalactive_epo.fif"
        freqs, spectrum = subject_spectrum(path)
        for band, (low, high) in BANDS.items():
            costs = ba.band_costs(
                freqs,
                spectrum,
                low_hz=low,
                high_hz=high,
                independent_hz=INDEPENDENT_HZ,
                comb_hz=COMB_HZ,
            )
            rows.append({"subject": subject.replace("sub-", ""), "band": band, **costs})
        print(f"  {subject} done", flush=True)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    summary = frame.groupby("band", sort=False)[
        ["independent_pct", "sideband_pct", "hole_pct"]
    ].median()
    print()
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
