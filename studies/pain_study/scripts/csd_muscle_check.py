"""Does CSD actually reduce the muscle confound on this cohort?

Measures, per participant, the across-channel correlation between a channel's muscle index
(62-95 Hz power over 8-30 Hz power, lines masked out of both) and its pain-minus-warm gamma
change, once in voltage space and once after CSD. If CSD does not lower it, the transform
has bought nothing here and that is the finding -- do not tune ``lambda2`` or ``stiffness``
to reach a wanted answer, because that makes the measurement meaningless.

CSD fits a spherical spline over the montage, so a missing channel distorts its
neighbourhood -- degrading exactly the participants that already have problems. Bad
channels are therefore interpolated inside the CSD arm only; the voltage arm keeps the
pipeline's no-interpolation rule, under which an interpolated channel is a weighted sum of
its neighbours.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from studies.pain_study.analysis.line_comb import diagnosis as hd
from studies.pain_study.analysis.line_comb import removal as hd_removal
from studies.pain_study.scripts.workflow_config import load_workflow_config


#: Where the configured removal workflow records what it actually took, per recording.
MANIFEST = load_workflow_config("line_comb").path("removal_dir") / "removal_manifest.tsv"


def _removal_targets(subject: str) -> tuple[float, ...]:
    """The isolated lines the removal acted on, read from its required manifest.

    This was a hardcoded tuple and it drifted from the removal it was meant to track. It
    masked 61.0353 Hz, which was dropped for sitting 0.128 Hz from comb harmonic 51, and
    masked nothing near 94 Hz -- where the strongest residual line in the cohort sits, at
    10.1% of sub-0008's 62-95 Hz power and 10.9% of sub-0001's. Since the muscle index
    below is a 62-95 Hz ratio, that line sat in the numerator of the measurement the mask
    exists to protect.

    The lines are detected per session, so the manifest is the only record of what was
    removed. Missing or malformed provenance is an error rather than permission to mask a
    historical frequency list.

    Only the isolated lines are masked, not the comb harmonics. The harmonics are removed
    and so read as holes rather than excess: they lower the index by a near-constant
    fraction in every participant, which the across-channel correlation is insensitive to,
    whereas a surviving line raises it in some participants only.
    """
    return hd_removal.removed_isolated_lines(MANIFEST, subject=subject)


BANDS = {"gamma_low": (30.1, 45.0), "gamma_mid": (45.0, 58.0), "gamma_high": (62.0, 95.0)}
CSD_LAMBDA2 = 1.0e-5
CSD_STIFFNESS = 4.0


def voltage_arm(epochs):
    """Voltage-space epochs, bad channels excluded as everywhere else in the pipeline."""
    return epochs.copy().pick("eeg", exclude="bads")


def csd_arm(epochs):
    """CSD epochs, bad channels interpolated first so the spline has no gap."""
    filled = epochs.copy().pick("eeg").interpolate_bads(reset_bads=True)
    return mne.preprocessing.compute_current_source_density(
        filled, lambda2=CSD_LAMBDA2, stiffness=CSD_STIFFNESS
    )


def _band_power(freqs, spectrum, low, high, mask):
    return 10.0 ** (
        hd.band_power_db(freqs, spectrum, low_hz=low, high_hz=high, excluded_hz=mask) / 10.0
    )


def _spectra(epochs, pain):
    data = epochs.get_data(copy=False)
    keep = hd.tr_commensurate_length(data.shape[-1], epochs.info["sfreq"])
    freqs, psd = hd.hann_periodogram(data[..., :keep], epochs.info["sfreq"])
    return freqs, psd[pain == 1].mean(axis=0), psd[pain == 0].mean(axis=0), psd.mean(axis=0)


def correlations(epochs, pain, isolated_hz) -> dict[str, float]:
    """Across-channel correlation of the muscle index with the pain-minus-warm change."""
    freqs, painful, warm, overall = _spectra(epochs, pain)
    mask = hd.line_exclusion_windows(isolated_hz, half_width_hz=0.25)
    index, contrast = [], {band: [] for band in BANDS}
    for channel in range(overall.shape[0]):
        index.append(
            _band_power(freqs, overall[channel], 62.0, 95.0, mask)
            / _band_power(freqs, overall[channel], 8.0, 30.0, mask)
        )
        for band, (low, high) in BANDS.items():
            contrast[band].append(
                10.0
                * np.log10(
                    _band_power(freqs, painful[channel], low, high, mask)
                    / _band_power(freqs, warm[channel], low, high, mask)
                )
            )
    muscle = np.asarray(index)
    return {band: float(pearsonr(muscle, np.asarray(v))[0]) for band, v in contrast.items()}


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
        folder = args.deriv_root / subject / "eeg"
        epochs = mne.read_epochs(
            folder / f"{subject}_task-thermalactive_epo.fif", preload=True, verbose="ERROR"
        )
        events = pd.read_csv(folder / f"{subject}_task-thermalactive_events.tsv", sep="\t")
        if len(events) != len(epochs):
            raise ValueError(f"{subject}: {len(events)} event rows vs {len(epochs)} epochs")
        pain = events["pain_binary_coded"].to_numpy()
        isolated_hz = _removal_targets(subject)

        entry = {"subject": subject.replace("sub-", "")}
        for arm, data in (("voltage", voltage_arm(epochs)), ("csd", csd_arm(epochs))):
            for band, value in correlations(data, pain, isolated_hz).items():
                entry[f"{arm}_{band}_r"] = value
        rows.append(entry)
        print(f"  {subject} done", flush=True)

    frame = pd.DataFrame(rows).set_index("subject")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out)
    print()
    print(frame.round(3).to_string())
    print()
    for band in BANDS:
        voltage, csd = frame[f"voltage_{band}_r"], frame[f"csd_{band}_r"]
        print(
            f"  {band:11s} voltage median r {voltage.median():+.3f} "
            f"({int((voltage > 0).sum())}/{len(voltage)} positive)   "
            f"CSD median r {csd.median():+.3f} ({int((csd > 0).sum())}/{len(csd)} positive)"
        )


if __name__ == "__main__":
    main()
