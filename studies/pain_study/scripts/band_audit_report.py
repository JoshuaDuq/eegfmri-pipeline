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
from studies.pain_study.analysis.line_comb import removal as lr
from studies.pain_study.scripts.workflow_config import load_workflow_config

TR = 0.9
KEEP_HIGH_HZ = 125.0


#: Where the removal records what it actually took, per recording.
MANIFEST = Path("outputs/line_comb_mains/removal_manifest.tsv")


def _audited_lines() -> tuple[float, ...]:
    """The isolated lines the removal acted on: the manifest first, the config as fallback.

    Not a stylistic preference: a copy drifts, and this one had. It carried 61.0353 Hz,
    dropped from the removal for sitting 0.128 Hz from comb harmonic 51, plus four more
    frequencies nothing targets -- and carried nothing near 94 Hz, where the strongest
    residual in the cohort sits. Since this report is what judges whether the line work
    helped, a drifted list charges excess where nothing was removed and leaves what was
    removed unscored.

    Reading the config fixed that drift but is no longer sufficient on its own: the lines
    are detected per session now, so ``isolated_hz`` names the fallback list rather than
    what was removed. The manifest is the only record of the latter.

    TR above is deliberately still a literal. The distinction is whether a constant varies
    between participants: the isolated lines scatter 0.19-0.595 Hz across the cohort and so
    must be read, while TR is 0.9 s by the Volume markers for everyone, and k/TR is the
    honest way to name the gradient comb.
    """
    workflow = load_workflow_config("line_comb")
    configured = tuple(float(f) for f in workflow.get("line_comb_removal.isolated_hz"))
    return lr.removed_isolated_lines(MANIFEST, fallback=configured)


#: Narrowband features belonging to neither comb, as the removal recorded them.
INDEPENDENT_HZ = _audited_lines()
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


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the report's arguments, kept separate so the defaults can be tested.

    ``--exclude`` defaults to nothing. It used to default to ``sub-0008``, which is
    excluded elsewhere because its BCG detection is unreliable -- it sits at 60.0 bpm, at
    the edge of the detector's rate window. That is a cardiac problem and says nothing
    about narrowband lines, so a spectral audit has no business inheriting it. It also hid
    the audit's own second-worst case: sub-0008 carries 10.1% of its 62-95 Hz power in the
    94 Hz line. A caller who needs an exclusion still passes one.
    """
    parser.add_argument("--deriv-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude", nargs="*", default=[])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
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
