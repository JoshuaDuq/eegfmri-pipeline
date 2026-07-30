"""Summarise what the line removal achieved, band by band.

Reads only what the earlier stages wrote -- the diagnosis catalogue, the removal manifest
and the verification spectra -- and turns them into the outcome tables and figure that
back ``docs/scanner_harmonic_removal.md``.

    python -m studies.pain_study.scripts.report_line_removal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from studies.pain_study.analysis import harmonic_diagnosis as hd  # noqa: E402
from studies.pain_study.analysis import line_removal as lr  # noqa: E402

DEFAULT_REMOVAL_DIR = Path("outputs/line_comb_removal")
DEFAULT_DIAGNOSIS_DIR = Path("outputs/scanner_harmonic_diagnosis")

BANDS = {
    "delta": (1.0, 3.9),
    "theta": (4.0, 7.9),
    "alpha": (8.0, 12.9),
    "beta": (13.0, 30.0),
    "gamma_low": (30.1, 45.0),
    "gamma_mid": (45.0, 58.0),
    "gamma_high": (62.0, 95.0),
    "gamma_full_masked": (30.1, 95.0),
}
MAINS_NOTCH_HZ = (59.5, 60.5)
LINE_HALF_WIDTH_HZ = 0.15


def artifact_share(freqs, psd, band, lines, half_width_bins):
    """Fraction of a band's power sitting above background at the lines, per participant."""
    inside = [f for f in lines if band[0] <= f <= band[1]]
    if not inside:
        return np.zeros(psd.shape[0]), 0
    shares = [
        hd.line_excess_fraction(
            freqs,
            spectrum,
            low_hz=band[0],
            high_hz=band[1],
            line_freqs=lines,
            half_width_bins=half_width_bins,
            line_half_width_hz=LINE_HALF_WIDTH_HZ,
        )
        for spectrum in psd
    ]
    return np.asarray(shares), len(inside)


def residual_prominence(freqs, psd, lines, half_width_bins):
    """Local prominence at each line, per participant."""
    prominence = np.stack(
        [hd.prominence_db(hd.to_db(spectrum), half_width_bins=half_width_bins) for spectrum in psd]
    )
    index = [int(np.argmin(np.abs(freqs - f))) for f in lines]
    return prominence[:, index]


def build_report(removal_dir: Path, diagnosis_dir: Path) -> dict:
    with np.load(removal_dir / "verification_spectra.npz", allow_pickle=False) as handle:
        freqs = handle["freqs"]
        original = handle["original"]
        cleaned = handle["cleaned"]
    catalogue = pd.read_csv(diagnosis_dir / "cohort_line_catalog.tsv", sep="\t")
    artifact = sorted(catalogue.loc[catalogue["kind"].isin(("comb", "isolated")), "refined_hz"])
    manifest = pd.read_csv(removal_dir / "removal_manifest.tsv", sep="\t")

    half_width = int(round((100.0 / 21.6) / float(freqs[1] - freqs[0])))
    rows = []
    for name, band in BANDS.items():
        before, n_lines = artifact_share(freqs, original, band, artifact, half_width)
        after, _ = artifact_share(freqs, cleaned, band, artifact, half_width)
        removed = (
            lr.removed_band_fraction(
                freqs,
                [f for f in artifact if band[0] <= f <= band[1]] or [band[0]],
                lr.notch_widths_for(
                    np.array([f for f in artifact if band[0] <= f <= band[1]] or [band[0]]),
                    ratio=450.0,
                    minimum_hz=0.05,
                ),
                band_hz=band,
            )
            if n_lines
            else 0.0
        )
        rows.append(
            {
                "band": name,
                "low_hz": band[0],
                "high_hz": band[1],
                "n_artifact_lines": n_lines,
                "artifact_share_before": float(np.median(before)),
                "artifact_share_before_max": float(np.max(before)),
                "artifact_share_after": float(np.median(after)),
                "artifact_share_after_max": float(np.max(after)),
                "band_fraction_removed": float(removed),
            }
        )

    residual = residual_prominence(freqs, cleaned, artifact, half_width)
    per_line = pd.DataFrame(
        {
            "frequency_hz": artifact,
            "median_residual_db": np.median(residual, axis=0),
            "max_residual_db": np.max(residual, axis=0),
            "n_participants_above_1db": (residual > 1.0).sum(axis=0),
        }
    )
    return {
        "bands": pd.DataFrame(rows),
        "per_line": per_line,
        "manifest": manifest,
        "freqs": freqs,
        "original": original,
        "cleaned": cleaned,
        "artifact": artifact,
    }


def figure(report: dict, path: Path) -> None:
    freqs, original, cleaned = report["freqs"], report["original"], report["cleaned"]
    band = (freqs >= 3) & (freqs <= 100)
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[2, 1])

    axes[0].plot(
        freqs[band],
        10 * np.log10(np.median(original, axis=0))[band],
        color="#C1442E",
        lw=0.6,
        label="before removal",
    )
    axes[0].plot(
        freqs[band],
        10 * np.log10(np.median(cleaned, axis=0))[band],
        color="#111827",
        lw=0.6,
        label="after removal",
    )
    axes[0].axvspan(95, 100, color="#FEE2E2", zorder=0)
    axes[0].axvspan(*MAINS_NOTCH_HZ, color="#E5E7EB", zorder=0)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("cohort median PSD (dB re 1 V²/Hz)")
    axes[0].set_title(
        "Line removal, 15 runs one per participant. Grey band: mains notch applied later "
        "by the pipeline.\nRed band: above the 95 Hz removal ceiling, where the comb "
        "continues and is not removed."
    )

    half_width = int(round((100.0 / 21.6) / float(freqs[1] - freqs[0])))
    for data, colour, label in ((original, "#C1442E", "before"), (cleaned, "#111827", "after")):
        prom = np.median(
            np.stack([hd.prominence_db(hd.to_db(s), half_width_bins=half_width) for s in data]),
            axis=0,
        )
        axes[1].plot(freqs[band], prom[band], color=colour, lw=0.6, label=label)
    axes[1].axhline(0, color="#6B7280", lw=0.6, ls="--")
    axes[1].axvspan(95, 100, color="#FEE2E2", zorder=0)
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("local prominence (dB)")
    axes[1].set_xlim(3, 100)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--removal-dir", type=Path, default=DEFAULT_REMOVAL_DIR)
    parser.add_argument("--diagnosis-dir", type=Path, default=DEFAULT_DIAGNOSIS_DIR)
    args = parser.parse_args(argv)

    report = build_report(args.removal_dir, args.diagnosis_dir)
    report["bands"].to_csv(
        args.removal_dir / "band_outcomes.tsv", sep="\t", index=False, float_format="%.6g"
    )
    report["per_line"].to_csv(
        args.removal_dir / "per_line_residual.tsv", sep="\t", index=False, float_format="%.6g"
    )
    figure(report, args.removal_dir / "removal_before_after.png")

    frame = report["bands"]
    print(f"{'band':20s} {'lines':>6s} {'before':>9s} {'after':>9s} {'bins removed':>13s}")
    for _, row in frame.iterrows():
        print(
            f"{row['band']:20s} {int(row['n_artifact_lines']):6d} "
            f"{100*row['artifact_share_before']:8.2f}% {100*row['artifact_share_after']:8.2f}% "
            f"{100*row['band_fraction_removed']:12.1f}%"
        )
    per_line = report["per_line"]
    print(
        f"\nlines still above 1 dB in any participant: "
        f"{int((per_line.max_residual_db > 1).sum())}/{len(per_line)}"
    )
    print(per_line.nlargest(6, "max_residual_db").to_string(index=False))
    print(
        f"\n  wrote {args.removal_dir/'band_outcomes.tsv'}, per_line_residual.tsv, "
        f"removal_before_after.png"
    )


if __name__ == "__main__":
    main()
