"""Figures for the scanner-harmonic diagnosis.

Reads only what ``diagnose_scanner_harmonics.py --stage analyse`` wrote, so the plots can
be restyled without recomputing anything.

    python -m studies.pain_study.scripts.plot_scanner_harmonics
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

DEFAULT_DIR = Path("outputs/scanner_harmonic_diagnosis")
COMB_COLOUR = "#C1442E"  # narrow line on the 1.2 Hz comb
ISOLATED_COLOUR = "#2B6CB0"  # narrow line that does not join the comb
BROAD_COLOUR = "#9CA3AF"  # broad feature: a rhythm, not an instrument line
CONTROL_COLOUR = "#6B7280"
DPI = 200


def classify(lines, structure):
    """Read back the classification the analysis stage already made."""
    comb = structure.loc[structure["family"] == "narrow_comb"]
    fundamental = float(comb["fundamental_hz"].iloc[0]) if len(comb) else float("nan")
    return lines["kind"].to_numpy(), fundamental


def colour_of(kind):
    return {
        "comb": COMB_COLOUR,
        "comb_wide": COMB_COLOUR,
        "isolated": ISOLATED_COLOUR,
        "other": BROAD_COLOUR,
    }[kind]


def _load(directory: Path):
    with np.load(directory / "spectra.npz", allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    frames = {
        name: pd.read_csv(directory / f"{name}.tsv", sep="\t")
        for name in (
            "cohort_line_catalog",
            "comb_structure",
            "control_persistence",
            "band_impact",
            "topography_reproducibility",
            "frequency_stability",
            "phase_locking",
            "variance_components",
            "per_run_line_prominence",
        )
    }
    return arrays, frames


def figure_cohort_spectrum(arrays, frames, path: Path) -> None:
    freqs = arrays["freqs_high"]
    psd = arrays["subject_psd_high"]
    lines = frames["cohort_line_catalog"]
    kinds, fundamental = classify(lines, frames["comb_structure"])
    band = (freqs >= 3.0) & (freqs <= 95.0)

    cohort_db = 10 * np.log10(np.median(psd, axis=0))
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[2, 1])

    for harmonic in range(3, int(95 * 0.9) + 1):
        axes[0].axvline(harmonic / 0.9, color="#E5E7EB", lw=0.5, zorder=0)
    axes[0].plot(freqs[band], cohort_db[band], color="#111827", lw=0.6)
    for kind, frequency in zip(kinds, lines["refined_hz"]):
        axes[0].axvline(frequency, color=colour_of(kind), lw=0.7, alpha=0.55)
    axes[0].set_ylabel("cohort median PSD (dB re 1 V²/Hz)")
    axes[0].set_title(
        "Final cleaned epochs, cohort median across 15 participants "
        f"(Δf = {freqs[1]:.4f} Hz, 21.6 s segments)\n"
        "pale grey verticals: scanner volume comb k/TR, where no line is detected   |   "
        f"red: narrow line on the {fundamental:.4f} Hz comb   "
        "blue: isolated narrow line   grey: other"
    )

    prominence = np.median(arrays["subject_prominence_high"], axis=0)
    axes[1].plot(freqs[band], prominence[band], color="#111827", lw=0.6)
    axes[1].axhline(0, color=CONTROL_COLOUR, lw=0.6, ls="--")
    for kind, frequency, value in zip(
        kinds, lines["refined_hz"], lines["cohort_median_prominence_db"]
    ):
        axes[1].plot(frequency, value, "o", ms=3, color=colour_of(kind))
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("local prominence (dB)")
    axes[1].set_xlim(3, 95)
    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    plt.close(figure)


def figure_final_versus_control(arrays, frames, path: Path) -> None:
    freqs = arrays["freqs_matched"]
    control_freqs = arrays["control_freqs"]
    final_db = 10 * np.log10(np.median(arrays["subject_psd_matched"], axis=0))
    control_db = 10 * np.log10(np.median(np.median(arrays["control_psd"], axis=1), axis=0))
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    band = (freqs >= 3) & (freqs <= 95)
    control_band = (control_freqs >= 3) & (control_freqs <= 95)
    axes[0].plot(
        freqs[band], final_db[band], color="#111827", lw=0.8, label="final epochs (gradients on)"
    )
    axes[0].plot(
        control_freqs[control_band],
        control_db[control_band],
        color=CONTROL_COLOUR,
        lw=0.8,
        label="gradient-free head windows (5 kHz, uncorrected)",
    )
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("PSD (dB re 1 V²/Hz)")
    axes[0].set_title(
        "Matched 3.6 s segments. Absolute levels differ (different processing and "
        "reference); the comparison of interest is which narrow lines appear in both."
    )

    persistence = frames["control_persistence"]
    width = 0.4
    positions = np.arange(len(persistence))
    axes[1].bar(
        positions - width / 2,
        persistence["final_matched_prominence_db"],
        width,
        color="#111827",
        label="final epochs",
    )
    axes[1].bar(
        positions + width / 2,
        persistence["control_prominence_db"],
        width,
        color=CONTROL_COLOUR,
        label="gradients off",
    )
    axes[1].errorbar(
        positions + width / 2,
        persistence["control_prominence_db"],
        yerr=[
            persistence["control_prominence_db"] - persistence["control_ci_low_db"],
            persistence["control_ci_high_db"] - persistence["control_prominence_db"],
        ],
        fmt="none",
        ecolor="#111827",
        elinewidth=0.6,
        capsize=1.5,
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(
        [f"{f:.2f}" for f in persistence["refined_hz"]], rotation=90, fontsize=6
    )
    axes[1].axhline(0, color="#111827", lw=0.6)
    axes[1].set_ylabel("prominence (dB)")
    axes[1].set_xlabel("detected line (Hz)")
    axes[1].legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    plt.close(figure)


def figure_comb_structure(frames, path: Path) -> None:
    lines = frames["cohort_line_catalog"]
    kinds, fundamental = classify(lines, frames["comb_structure"])
    narrow = lines.loc[
        lines["kind"].isin(("comb", "comb_wide", "isolated")), "refined_hz"
    ].to_numpy()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    gaps = hd.pairwise_differences(narrow, max_difference_hz=12.0)
    axes[0].hist(gaps, bins=np.arange(0, 12.05, 0.02), color=COMB_COLOUR)
    for multiple in range(1, 11):
        axes[0].axvline(multiple * fundamental, color="#111827", lw=0.4, ls=":")
    axes[0].set_xlabel("pairwise frequency gap between narrow lines (Hz)")
    axes[0].set_ylabel("count of pairs")
    axes[0].set_title(f"Gaps pile up at multiples of {fundamental:.4f} Hz (dotted)")

    members = lines["refined_hz"].to_numpy()[kinds == "comb"]
    harmonics = np.rint(members / fundamental)
    residuals = members - harmonics * fundamental
    axes[1].plot(members, residuals * 1000, "o", ms=4, color=COMB_COLOUR)
    for frequency in lines["refined_hz"].to_numpy()[kinds == "isolated"]:
        axes[1].axvline(frequency, color=ISOLATED_COLOUR, lw=0.8, ls="--")
    axes[1].axhline(0, color="#111827", lw=0.6)
    axes[1].set_xlabel("line frequency (Hz)")
    axes[1].set_ylabel(f"residual from {fundamental:.5f} Hz × k (mHz)")
    axes[1].set_title(
        f"{len(members)} lines, harmonics {int(harmonics.min())}–{int(harmonics.max())}, "
        f"RMS residual {np.sqrt(np.mean(residuals**2))*1000:.1f} mHz\n"
        "dashed blue: narrow lines that do not join the comb"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    plt.close(figure)


def figure_subject_heatmap(arrays, frames, path: Path) -> None:
    lines = frames["cohort_line_catalog"]
    subjects = [str(s) for s in arrays["subjects"]]
    prominence = arrays["subject_prominence_high"][:, lines["bin"].to_numpy()]

    figure, axis = plt.subplots(figsize=(max(9, 0.34 * len(lines)), 6))
    image = axis.imshow(prominence, aspect="auto", cmap="magma", vmin=0)
    axis.set_yticks(range(len(subjects)))
    axis.set_yticklabels(subjects, fontsize=7)
    axis.set_xticks(range(len(lines)))
    axis.set_xticklabels([f"{f:.2f}" for f in lines["refined_hz"]], rotation=90, fontsize=6)
    kinds, fundamental = classify(lines, frames["comb_structure"])
    for position, kind in enumerate(kinds):
        axis.get_xticklabels()[position].set_color(colour_of(kind))
    axis.set_xlabel(
        f"detected line (Hz); red = on the {fundamental:.4f} Hz comb, "
        "blue = isolated line, grey = other"
    )
    axis.set_title("Line prominence per participant (dB)")
    figure.colorbar(image, ax=axis, label="prominence (dB)")
    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    plt.close(figure)


def figure_topographies(arrays, frames, path: Path, n_lines: int = 8) -> None:
    import mne

    lines = frames["cohort_line_catalog"].copy()
    strongest = lines.nlargest(n_lines, "cohort_median_prominence_db").sort_values("refined_hz")
    channel_names = [str(name) for name in arrays["channel_names"]]
    montage = mne.channels.make_standard_montage("standard_1005")
    info = mne.create_info(channel_names, sfreq=500.0, ch_types="eeg")
    info.set_montage(montage, match_case=False, on_missing="ignore")

    order = {float(f): i for i, f in enumerate(frames["cohort_line_catalog"]["frequency_hz"])}
    columns = min(4, len(strongest))
    rows = int(np.ceil(len(strongest) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    reproducibility = frames["topography_reproducibility"].set_index("frequency_hz")

    for position, (_, line) in enumerate(strongest.iterrows()):
        values = arrays["topography_maps"][order[float(line["frequency_hz"])]]
        finite = np.nan_to_num(values, nan=float(np.nanmedian(values)))
        mne.viz.plot_topomap(finite, info, axes=axes[position], show=False, cmap="magma")
        r = reproducibility.loc[float(line["frequency_hz"]), "mean_pairwise_r"]
        axes[position].set_title(
            f"{line['refined_hz']:.3f} Hz\n{line['cohort_median_prominence_db']:.1f} dB, r̄ = {r:.2f}",
            fontsize=8,
        )
    for axis in axes[len(strongest) :]:
        axis.axis("off")
    figure.suptitle("Mean prominence topography of the strongest lines", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    plt.close(figure)


def figure_band_impact(frames, path: Path) -> None:
    impact = frames["band_impact"]
    order = [
        "delta",
        "theta",
        "alpha",
        "beta",
        "beta_low_clean",
        "beta_high_clean",
        "gamma",
        "gamma_low_clean",
        "gamma_mid_clean",
        "gamma_high_clean",
    ]
    figure, axis = plt.subplots(figsize=(11, 5))
    data = [impact.loc[impact["band"] == band, "line_contribution_db"].to_numpy() for band in order]
    parts = axis.boxplot(data, tick_labels=order, showfliers=False, patch_artist=True)
    for patch in parts["boxes"]:
        patch.set_facecolor("#DBEAFE")
    for position, values in enumerate(data, start=1):
        axis.plot(
            np.full(values.size, position)
            + np.random.default_rng(0).uniform(-0.12, 0.12, values.size),
            values,
            "o",
            ms=2.5,
            color="#1F2937",
            alpha=0.7,
        )
    axis.axhline(0, color="#111827", lw=0.6)
    axis.set_ylabel("band power inflation from detected lines (dB)")
    axis.set_title("Power the detected lines add to each analysis band, per participant")
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args(argv)

    arrays, frames = _load(args.input_dir)
    figures_dir = args.input_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_cohort_spectrum(arrays, frames, figures_dir / "fig1_cohort_spectrum.png")
    figure_final_versus_control(arrays, frames, figures_dir / "fig2_final_vs_control.png")
    figure_comb_structure(frames, figures_dir / "fig3_comb_structure.png")
    figure_subject_heatmap(arrays, frames, figures_dir / "fig4_subject_lines.png")
    figure_topographies(arrays, frames, figures_dir / "fig5_topographies.png")
    figure_band_impact(frames, figures_dir / "fig6_band_impact.png")
    print(f"Wrote six figures to {figures_dir}")


if __name__ == "__main__":
    main()
