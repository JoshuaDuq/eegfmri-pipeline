"""Figures for the Analyzer correction and the beat-marker comparison.

Draws what ``studies.pain_study.analysis.bcg.report`` measured. Saved as PNG rather than
rendered into a report section; what the panels mean is in README.md.

    eeg-pipeline bcg plot
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from eeg_pipeline.preprocessing.report.style import (  # noqa: E402
    AFTER_COLOR,
    BEFORE_COLOR,
    FLAG_COLOR,
    GUIDE_COLOR,
    PRIMARY_COLOR,
    run_label,
)
from studies.pain_study.analysis.bcg.report import (  # noqa: E402
    AnalyzerCorrectionQc,
    MarkerAgreement,
)

DPI = 200


def save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return path


#: Width of the window each detector's beat count is expressed over, in seconds.
#:
#: Wide enough that a healthy train gives a steady rate rather than a count that swings
#: between eight and nine beats, and narrow enough to place the moment a train stops to
#: within a few seconds of it.
MARKER_RATE_BIN_S = 10.0


def _binned_rate(onsets_s: np.ndarray, edges_s: np.ndarray) -> np.ndarray:
    """Beats per minute in each window, as a rate rather than a count."""
    counts, _ = np.histogram(onsets_s, bins=edges_s)
    return counts * (60.0 / MARKER_RATE_BIN_S)


def plot_analyzer_qc(qc: AnalyzerCorrectionQc) -> plt.Figure:
    """Plot residual R-locked amplitude per run, and the attenuation achieved."""
    runs = qc.runs
    if not {"before_rms_uv", "after_rms_uv"}.issubset(runs.columns):
        raise ValueError("Analyzer QC figure requires before and after R-locked amplitudes.")
    labels = [f"run-{value}" for value in runs["run"]]
    positions = np.arange(len(runs))
    fallback = (
        runs["is_fallback"].fillna(False).astype(bool).to_numpy()
        if "is_fallback" in runs.columns
        else np.zeros(len(runs), dtype=bool)
    )

    figure, (amplitude_axis, attenuation_axis) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.0),
        layout="constrained",
    )
    # Residual amplitude spans more than an order of magnitude between runs, so the axis
    # is logarithmic and the two states are drawn as paired markers rather than bars.
    for position, before, after in zip(
        positions,
        runs["before_rms_uv"],
        runs["after_rms_uv"],
        strict=True,
    ):
        amplitude_axis.plot([position, position], [before, after], color="0.75", linewidth=1.2)
    amplitude_axis.scatter(
        positions,
        runs["before_rms_uv"],
        color=BEFORE_COLOR,
        s=34,
        label="Reaching this pipeline",
        zorder=3,
    )
    amplitude_axis.scatter(
        positions,
        runs["after_rms_uv"],
        color=AFTER_COLOR,
        s=34,
        label="After pipeline ICA",
        zorder=3,
    )
    amplitude_axis.set(
        title="R-locked EEG amplitude per run",
        ylabel="Median RMS (µV)",
        yscale="log",
        xticks=positions,
        xticklabels=labels,
    )
    amplitude_axis.grid(axis="y", alpha=0.2)
    amplitude_axis.spines[["top", "right"]].set_visible(False)
    amplitude_axis.legend(frameon=False, fontsize=8, loc="best")

    if "attenuation_db" in runs.columns:
        # The highlight is vermillion and the pre-ICA state above is orange. They were
        # the same colour until the shared palette split them: one figure using one ink
        # for "before correction" in its left panel and "used fallback detection" in its
        # right panel is a reading trap, not a convention.
        colors = [FLAG_COLOR if flag else "0.80" for flag in fallback]
        attenuation_axis.bar(positions, runs["attenuation_db"], color=colors)
        median_attenuation = float(np.nanmedian(runs["attenuation_db"]))
        attenuation_axis.axhline(
            median_attenuation,
            color=GUIDE_COLOR,
            linestyle="--",
            linewidth=1.0,
            label=f"median {median_attenuation:.1f} dB",
        )
        attenuation_axis.set(
            title="Cardiac attenuation achieved by pipeline ICA",
            ylabel="Attenuation (dB)",
            xticks=positions,
            xticklabels=labels,
        )
        handles, _ = attenuation_axis.get_legend_handles_labels()
        if fallback.any():
            # Explain the highlight in the legend, beside the bars it marks, rather than
            # in a second title line the eye has already left by the time it reaches them.
            handles.append(Patch(facecolor=FLAG_COLOR, label="fallback R-peak detection"))
        attenuation_axis.legend(handles=handles, frameon=False, fontsize=8)
        attenuation_axis.grid(axis="y", alpha=0.2)
        attenuation_axis.spines[["top", "right"]].set_visible(False)
    for axis in (amplitude_axis, attenuation_axis):
        axis.tick_params(axis="x", labelrotation=30, labelsize=8)
    figure.suptitle(
        f"sub-{html.unescape(qc.subject)} · BrainVision Analyzer correction quality",
        fontsize=10,
    )
    plt.close(figure)
    return figure


def plot_marker_agreement(agreements: Sequence[MarkerAgreement]) -> plt.Figure:
    """Draw both detectors' beat rate against time, one panel per run.

    A total says the marker train was short; only the time axis says *when* it stopped, and
    a train that never started and one that lost the trace partway are different faults
    with different implications for the correction either side of that moment.

    The rate is binned rather than drawn one tick per beat. A run holds several hundred
    beats over a few hundred seconds, which is more events than the axis has pixels: drawn
    individually they alias, and the interference banding reads as structure in the marker
    train that is not in the data. Binning also puts both detectors in the same unit, so
    the panel answers "were beats being marked at this moment, at the rate the ECG shows"
    rather than leaving two tick rows to be compared by eye.
    """
    if not agreements:
        raise ValueError("The marker agreement figure needs at least one run.")

    figure, axes = plt.subplots(
        len(agreements),
        1,
        figsize=(11.0, 1.9 * len(agreements) + 0.8),
        squeeze=False,
        sharex=True,
        layout="constrained",
    )
    drawn_rates: list[np.ndarray] = []
    for axis, agreement in zip(axes[:, 0], agreements, strict=True):
        onsets = np.concatenate([agreement.marker_onsets_s, agreement.detected_onsets_s])
        stop = float(onsets.max()) if onsets.size else MARKER_RATE_BIN_S
        # Whole windows only. A run rarely ends on a boundary, and counting the short
        # remainder at the full window's rate drove the trace to the floor at the right
        # edge of every panel — a collapse manufactured by where the recording stopped,
        # in the one figure whose subject is when a train really did stop.
        n_windows = max(int(stop // MARKER_RATE_BIN_S), 1)
        edges = np.arange(n_windows + 1) * MARKER_RATE_BIN_S
        centres = edges[:-1] + MARKER_RATE_BIN_S / 2.0
        rates = []
        for onsets_s, color, label in (
            (agreement.detected_onsets_s, PRIMARY_COLOR, "Detected from ECG"),
            (agreement.marker_onsets_s, BEFORE_COLOR, "Analyzer markers"),
        ):
            # Steps, not a smooth line: the rate is constant within each window by
            # construction, and interpolating between centres would draw a beat rate at
            # moments where none was measured.
            rate = _binned_rate(onsets_s, edges)
            rates.append(rate)
            axis.step(
                centres,
                rate,
                where="mid",
                color=color,
                linewidth=1.1,
                label=label,
            )
        # A run where Analyzer marked nothing draws one lone trace, and a reader who does
        # not check the legend reads a single-detector panel as agreement. Saying it on
        # the panel also states what the lone trace is evidence *of*: the ECG carried a
        # heartbeat that was there to be marked, so the gap is in the marker train and
        # not in the physiology.
        if agreement.n_markers == 0 and agreement.n_detected > 0:
            axis.annotate(
                "Analyzer marked no beats in this run — the ECG trace is what it missed",
                xy=(0.5, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=7.5,
                color=FLAG_COLOR,
            )
        fraction = agreement.matched_fraction
        share = "no beats detected" if fraction is None else f"{fraction:.1%} of beats marked"
        title = (
            f"{run_label(agreement.recording_id)} · "
            f"{agreement.n_markers} markers · {agreement.n_detected} detected · {share}"
        )
        # What a low share means, stated beside it. A tight lag says the two trains
        # describe the same heartbeat at an offset; a broad one says they disagree. On
        # sub-0012 run-5 the share was 0.0% and the lag was 303 ms with a 19 ms spread,
        # and the panel gave a reader no way to tell those apart.
        median_lag = agreement.median_lag_s
        if median_lag is not None and agreement.n_matched < agreement.n_detected:
            title += (
                f"\nnearest marker {median_lag * 1000:+.0f} ms away "
                f"(IQR {agreement.lag_iqr_s * 1000:.0f} ms)"
            )
        drawn_rates.extend(rates)
        axis.set(
            title=title,
            ylabel="Beats per min",
        )
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    # One range over every run, not one per panel. Panels that each chose their own ran
    # 0–80, 40–80, 50–78 and 50–90 bpm in the same figure, so a 6 bpm wobble on a healthy
    # run drew as much ink as a collapse and the eye recalibrated at every panel —
    # comparing runs is the only reason to stack them.
    #
    # Zero is on the axis only when some detector reached it, which is the rule the panels
    # already used, now applied to the runs together: anchoring there always spent half
    # the height on rates neither trace visits, while the comparison this figure exists
    # for is between two traces a few beats per minute apart.
    observed = np.concatenate(drawn_rates) if drawn_rates else np.zeros(1)
    finite = observed[np.isfinite(observed)]
    if finite.size:
        floor = 0.0 if finite.min() <= 0.0 else max(float(finite.min()) - 10.0, 0.0)
        ceiling = float(finite.max())
        headroom = max(0.05 * (ceiling - floor), 2.0)
        for axis in axes[:, 0]:
            axis.set_ylim(floor, ceiling + headroom)
    # Below the grid rather than inside the first panel. Placed in-axes it sat on top of
    # the marker trace of whichever run came first, which on sub-0012 was the run with the
    # sparsest markers — the one the panel most needed to show.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=8,
    )
    axes[-1, 0].set_xlabel(f"Time in run (s) · beats counted in {MARKER_RATE_BIN_S:.0f} s windows")
    plt.close(figure)
    return figure


__all__ = [
    "DPI",
    "plot_analyzer_qc",
    "plot_marker_agreement",
    "save",
]
