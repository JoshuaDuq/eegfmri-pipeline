"""Summarise what the line removal achieved, band by band.

Reads only what the removal wrote -- participant-specific transform provenance and
verification spectra -- and turns them into the outcome tables and figure that back
``docs/scanner_harmonic_removal.md``.

    eeg-pipeline line-comb report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from studies.pain_study.analysis.line_comb import diagnosis as hd  # noqa: E402
from studies.pain_study.analysis.line_comb import removal as lr  # noqa: E402
from studies.pain_study.scripts.line_comb.remove import RemovalSettings  # noqa: E402

WORKFLOW = "line_comb"

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


def _artifact_frequencies(cell) -> tuple[float, ...]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return ()
    values = tuple(float(piece) for piece in str(cell).split(";") if piece.strip())
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Removal manifest contains a non-finite artifact frequency.")
    return values


def subject_artifact_targets(
    manifest: pd.DataFrame,
    subjects: tuple[str, ...],
    settings: RemovalSettings,
) -> dict[str, tuple[float, ...]]:
    """Artifact frequencies actually authorised for each participant."""
    required = {"recording", "fundamental_hz", "isolated_hz", "adjacent_hz"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Removal manifest is missing columns: {sorted(required - set(manifest))}")

    targets = {}
    for subject in subjects:
        rows = manifest[manifest.recording.astype(str).str.startswith(f"{subject}_")]
        if rows.empty:
            raise ValueError(f"Removal manifest has no manifest rows for {subject}.")
        frequencies = []
        for row in rows.itertuples(index=False):
            fundamental_hz = float(row.fundamental_hz)
            if not np.isfinite(fundamental_hz) or fundamental_hz <= 0.0:
                raise ValueError(f"Removal manifest has an invalid fundamental for {subject}.")
            frequencies.extend(
                harmonic * fundamental_hz
                for harmonic in range(
                    settings.removal_harmonic_range[0],
                    settings.removal_harmonic_range[1] + 1,
                )
            )
            frequencies.extend(_artifact_frequencies(row.isolated_hz))
            frequencies.extend(_artifact_frequencies(row.adjacent_hz))
        targets[subject] = tuple(
            sorted(
                {
                    frequency
                    for frequency in frequencies
                    if settings.low_hz <= frequency <= settings.high_hz
                    and (
                        not settings.exclude_mains
                        or not MAINS_NOTCH_HZ[0] <= frequency <= MAINS_NOTCH_HZ[1]
                    )
                }
            )
        )
    return targets


def _line_sources(frequencies: tuple[float, ...]) -> tuple[float, ...]:
    """Collapse within-source drift while preserving neighbouring physical lines."""
    clusters: list[list[float]] = []
    for frequency in sorted(frequencies):
        eligible = [
            cluster
            for cluster in clusters
            if abs(frequency - float(np.median(cluster))) <= lr._LINE_CLAIM_HZ
        ]
        if eligible:
            nearest = min(
                eligible,
                key=lambda cluster: abs(frequency - float(np.median(cluster))),
            )
            nearest.append(frequency)
        else:
            clusters.append([frequency])
    return tuple(float(np.median(cluster)) for cluster in clusters)


def artifact_share_by_subject(
    freqs,
    psd,
    band,
    subject_targets: dict[str, tuple[float, ...]],
    subjects: tuple[str, ...],
    half_width_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Line-excess share using each participant's own detected target set."""
    if psd.shape[0] != len(subjects):
        raise ValueError("The spectrum rows must align one-to-one with subjects.")
    shares = []
    counts = []
    for subject, spectrum in zip(subjects, psd):
        lines = tuple(
            frequency for frequency in subject_targets[subject] if band[0] <= frequency <= band[1]
        )
        counts.append(len(_line_sources(lines)))
        shares.append(
            hd.line_excess_fraction(
                freqs,
                spectrum,
                low_hz=band[0],
                high_hz=band[1],
                line_freqs=lines,
                half_width_bins=half_width_bins,
                line_half_width_hz=LINE_HALF_WIDTH_HZ,
            )
            if lines
            else 0.0
        )
    return np.asarray(shares), np.asarray(counts, dtype=int)


def _per_subject_residuals(
    freqs: np.ndarray,
    cleaned: np.ndarray,
    subjects: tuple[str, ...],
    subject_targets: dict[str, tuple[float, ...]],
    half_width_bins: int,
) -> pd.DataFrame:
    rows = []
    for subject, spectrum in zip(subjects, cleaned):
        prominence = hd.prominence_db(hd.to_db(spectrum), half_width_bins=half_width_bins)
        narrow = lr._narrow_peak_mask(freqs, prominence)
        for frequency in _line_sources(subject_targets[subject]):
            inside = np.abs(freqs - frequency) <= lr.RESIDUAL_SEARCH_HZ
            candidates = np.flatnonzero(inside & narrow & np.isfinite(prominence))
            index = (
                int(candidates[np.argmax(prominence[candidates])])
                if candidates.size
                else int(np.argmin(np.abs(freqs - frequency)))
            )
            rows.append(
                {
                    "subject": subject,
                    "target_frequency_hz": frequency,
                    "residual_frequency_hz": float(freqs[index]),
                    "residual_prominence_db": float(prominence[index]),
                }
            )
    return pd.DataFrame(rows)


def build_report(removal_dir: Path, settings: RemovalSettings) -> dict:
    with np.load(removal_dir / "verification_spectra.npz", allow_pickle=False) as handle:
        freqs = handle["freqs"]
        original = handle["original"]
        cleaned = handle["cleaned"]
        subjects = tuple(str(value) for value in handle["subjects"])
    manifest = pd.read_csv(removal_dir / "removal_manifest.tsv", sep="\t")
    subject_targets = subject_artifact_targets(manifest, subjects, settings)

    half_width = int(round((100.0 / 21.6) / float(freqs[1] - freqs[0])))
    rows = []
    for name, band in BANDS.items():
        before, counts = artifact_share_by_subject(
            freqs, original, band, subject_targets, subjects, half_width
        )
        after, _ = artifact_share_by_subject(
            freqs, cleaned, band, subject_targets, subjects, half_width
        )
        rows.append(
            {
                "band": name,
                "low_hz": band[0],
                "high_hz": band[1],
                "median_artifact_sources": float(np.median(counts)),
                "min_artifact_sources": int(np.min(counts)),
                "max_artifact_sources": int(np.max(counts)),
                "artifact_share_before": float(np.median(before)),
                "artifact_share_before_max": float(np.max(before)),
                "artifact_share_after": float(np.median(after)),
                "artifact_share_after_max": float(np.max(after)),
            }
        )

    per_subject_lines = _per_subject_residuals(
        freqs,
        cleaned,
        subjects,
        subject_targets,
        half_width,
    )
    return {
        "bands": pd.DataFrame(rows),
        "per_subject_lines": per_subject_lines,
        "manifest": manifest,
        "freqs": freqs,
        "original": original,
        "cleaned": cleaned,
        "subjects": subjects,
        "subject_targets": subject_targets,
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
    axes[0].axvspan(*MAINS_NOTCH_HZ, color="#E5E7EB", zorder=0)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("cohort median PSD (dB re 1 V²/Hz)")
    axes[0].set_title(
        "Line removal across participant-median spectra. Grey band: mains notch applied "
        "later by the pipeline."
    )

    half_width = int(round((100.0 / 21.6) / float(freqs[1] - freqs[0])))
    for data, colour, label in ((original, "#C1442E", "before"), (cleaned, "#111827", "after")):
        prom = np.median(
            np.stack([hd.prominence_db(hd.to_db(s), half_width_bins=half_width) for s in data]),
            axis=0,
        )
        axes[1].plot(freqs[band], prom[band], color=colour, lw=0.6, label=label)
    axes[1].axhline(0, color="#6B7280", lw=0.6, ls="--")
    axes[1].axvspan(*MAINS_NOTCH_HZ, color="#E5E7EB", zorder=0)
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("local prominence (dB)")
    axes[1].set_xlim(3, 100)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--removal-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    run(args)


def run(args: argparse.Namespace) -> None:
    """Build the tables. Split from ``main`` so the CLI command can call it with its own args."""
    from studies.pain_study.scripts.workflow_config import load_workflow_config

    config = load_workflow_config(WORKFLOW, getattr(args, "config", None))
    args.removal_dir = config.path("removal_dir", override=args.removal_dir)

    settings = RemovalSettings.from_config(config)
    report = build_report(args.removal_dir, settings)
    report["bands"].to_csv(
        args.removal_dir / "band_outcomes.tsv", sep="\t", index=False, float_format="%.6g"
    )
    report["per_subject_lines"].to_csv(
        args.removal_dir / "per_subject_line_residual.tsv",
        sep="\t",
        index=False,
        float_format="%.6g",
    )
    figure(report, args.removal_dir / "removal_before_after.png")

    frame = report["bands"]
    print(f"{'band':20s} {'sources':>9s} {'before':>9s} {'after':>9s}")
    for _, row in frame.iterrows():
        print(
            f"{row['band']:20s} {row['median_artifact_sources']:8.1f} "
            f"{100*row['artifact_share_before']:8.2f}% "
            f"{100*row['artifact_share_after']:8.2f}%"
        )
    per_line = report["per_subject_lines"]
    print(
        f"\nparticipant-specific line positions still above 1 dB: "
        f"{int((per_line.residual_prominence_db > 1).sum())}/{len(per_line)}"
    )
    print(per_line.nlargest(10, "residual_prominence_db").to_string(index=False))
    print(
        f"\n  wrote {args.removal_dir/'band_outcomes.tsv'}, "
        "per_subject_line_residual.tsv, removal_before_after.png"
    )


if __name__ == "__main__":
    main()
