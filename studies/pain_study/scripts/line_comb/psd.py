"""Before-and-after power spectra of the removal, computed with MNE.

    eeg-pipeline line-comb psd

Answers the question a reader asks first: what did this actually take out, and did it take
out anything else? Three figures from Welch spectra computed by ``Raw.compute_psd`` on the
source and on each derivative in turn: an overview of the whole band, the same data tiled
into readable spans, and one panel per recording.

The tiled figure exists because the overview cannot answer the second question. Drawing
1-100 Hz on one axis puts several frequency bins in every pixel, so a comb line 1.2 Hz from
its neighbour is sub-pixel: the overview shows that the lines went, and cannot show whether
they went surgically or took their surroundings with them.

Runs on whatever exists. After ``apply`` it compares the source against the line-cleaned
copy; after ``notch`` it adds the notched copy as a third trace, so the two transforms can
be told apart rather than seen only in combination.

Unlike the stages that transform or certify, this one accepts ``--subjects``: a figure of
part of a cohort is a smaller figure, not a false claim about the whole one.

The comparison is only worth reading if both sides were measured identically, so the
spectra are computed with one set of parameters, on the same channels, over the same
samples, and the stage refuses when the recordings do not correspond.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from studies.pain_study.scripts.line_comb import remove as rm

WORKFLOW = rm.WORKFLOW

BAND_HZ = (1.0, 100.0)
"""Frequencies drawn. Wide enough to show that low frequencies were left alone."""


@dataclass(frozen=True)
class PsdSettings:
    """How the spectra are computed. One set of values for every arm of the comparison."""

    window_s: float = 54.0
    """Welch segment length. Sets the resolution: 54 s gives 18.5 mHz bins.

    Defaults to the removal's own estimation window, so the figure resolves what the fit
    resolved. A comb at 1.2 Hz spacing needs far less; a line pair 26 mHz apart needs this.
    """
    overlap: float = 0.5
    band_hz: tuple[float, float] = BAND_HZ
    panel_span_hz: float = 10.0
    """Width of each panel in the tiled figure.

    The overview draws the whole band on one axis, where 5347 bins share about 1900 pixels
    and a comb line 1.2 Hz from its neighbour is sub-pixel -- so it shows that lines went
    and cannot show whether they went surgically. Tiling the same data into consecutive
    spans of this width puts roughly one bin per pixel, which is what it takes to see the
    shape of an individual notch.
    """

    def __post_init__(self) -> None:
        if not np.isfinite(self.window_s) or self.window_s <= 0.0:
            raise ValueError("window_s must be finite and positive.")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must lie in [0, 1).")
        low_hz, high_hz = self.band_hz
        if not 0.0 <= low_hz < high_hz:
            raise ValueError("band_hz must be an increasing non-negative range.")
        if not np.isfinite(self.panel_span_hz) or self.panel_span_hz <= 0.0:
            raise ValueError("panel_span_hz must be finite and positive.")

    @classmethod
    def from_config(cls, config) -> "PsdSettings":
        block = config.get("psd") or {}
        defaults = cls()
        band = block.get("band_hz", defaults.band_hz)
        return cls(
            window_s=float(
                block.get(
                    "window_s",
                    config.get("line_comb_removal.estimation_window_s", defaults.window_s),
                )
            ),
            overlap=float(block.get("overlap", defaults.overlap)),
            band_hz=(float(band[0]), float(band[1])),
            panel_span_hz=float(block.get("panel_span_hz", defaults.panel_span_hz)),
        )


def channel_median_psd(raw, settings: PsdSettings) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of the EEG channels, and their median, via ``Raw.compute_psd``.

    EEG only: ECG and EOG carry different units and amplitudes, and averaging them in would
    put a millivolt trace on the same axis as a microvolt one.

    The median rather than the mean across channels, because one bad channel should not set
    the level of a cohort figure.
    """
    sampling_frequency_hz = float(raw.info["sfreq"])
    n_fft = int(round(settings.window_s * sampling_frequency_hz))
    if n_fft > raw.n_times:
        raise ValueError(
            f"The recording holds {raw.n_times} samples, fewer than the {n_fft} of one "
            f"{settings.window_s:g} s Welch segment. Lower `psd.window_s`."
        )
    spectrum = raw.compute_psd(
        method="welch",
        picks="eeg",
        fmin=settings.band_hz[0],
        fmax=settings.band_hz[1],
        n_fft=n_fft,
        n_overlap=int(round(n_fft * settings.overlap)),
        n_per_seg=n_fft,
        verbose="ERROR",
    )
    psd, freqs = spectrum.get_data(return_freqs=True)
    return freqs, np.median(psd, axis=0)


def to_db(psd: np.ndarray) -> np.ndarray:
    """Decibels relative to 1 V^2/Hz, with a floor so an empty bin cannot be -inf."""
    values = np.asarray(psd, dtype=float)
    return 10.0 * np.log10(np.maximum(values, np.finfo(float).tiny))


def compare_recording(
    source_vhdr: Path,
    derivative_vhdrs: Sequence[tuple[str, Path]],
    settings: PsdSettings,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One recording's spectrum before and after each derivative, on one frequency grid.

    Every arm is read through the same BIDS reader and measured with the same parameters,
    and the geometry is checked rather than assumed: a derivative that has drifted in
    channel set, length or sampling rate is not a comparison, it is two different
    recordings on one axis.
    """
    source = rm.read_bids_raw(source_vhdr)
    freqs, source_psd = channel_median_psd(source, settings)
    spectra = {"source": source_psd}
    for label, path in derivative_vhdrs:
        derivative = rm.read_bids_raw(path)
        if derivative.ch_names != source.ch_names:
            raise ValueError(f"{label} {path.name}: channel set differs from the source.")
        if derivative.n_times != source.n_times:
            raise ValueError(f"{label} {path.name}: length differs from the source.")
        if not np.isclose(derivative.info["sfreq"], source.info["sfreq"]):
            raise ValueError(f"{label} {path.name}: sampling rate differs from the source.")
        derivative_freqs, psd = channel_median_psd(derivative, settings)
        if not np.array_equal(derivative_freqs, freqs):
            raise ValueError(f"{label} {path.name}: spectra landed on a different grid.")
        spectra[label] = psd
        del derivative
    del source
    return freqs, spectra


ANALYSIS_BANDS = (
    ("delta", 1.0, 3.9),
    ("theta", 4.0, 7.9),
    ("alpha", 8.0, 12.9),
    ("beta", 13.0, 30.0),
    ("gamma", 30.1, 80.0),
)
"""Named bands, annotated on each panel so a span can be placed in what gets analysed."""


def analysis_bands_from_config(config) -> tuple[tuple[str, float, float], ...]:
    """The study's own bands where it defines them, so the labels match its analyses."""
    defined = config.get("frequency_bands") or {}
    if not isinstance(defined, dict):
        return ANALYSIS_BANDS
    named = [
        (name, float(defined[name][0]), float(defined[name][1]))
        for name, _, _ in ANALYSIS_BANDS
        if name in defined
    ]
    return tuple(named) or ANALYSIS_BANDS


def panel_edges(band_hz: tuple[float, float], span_hz: float) -> tuple[tuple[float, float], ...]:
    """Consecutive equal spans tiling the plotted band, the last one truncated."""
    low_hz, high_hz = band_hz
    starts = np.arange(low_hz, high_hz, span_hz)
    return tuple((float(start), float(min(start + span_hz, high_hz))) for start in starts)


def bands_covering(low_hz: float, high_hz: float, bands) -> str:
    """The named bands a span overlaps, for the panel label."""
    return ", ".join(
        name for name, band_low, band_high in bands if min(high_hz, band_high) > max(low_hz, band_low)
    )


def figure_band_panels(
    freqs,
    arms: dict[str, np.ndarray],
    path: Path,
    *,
    settings: PsdSettings,
    bands=ANALYSIS_BANDS,
) -> None:
    """The same spectra tiled into readable spans, one panel per range.

    The overview cannot show the shape of a notch: at 1.2 Hz line spacing and 18.5 mHz bins
    it draws several bins per pixel. Here each panel spans ``panel_span_hz``, which brings
    it to about one bin per pixel, and autoscales on its own contents so a deep notch in one
    span does not flatten the spectrum in another.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = panel_edges((float(freqs[0]), float(freqs[-1])), settings.panel_span_hz)
    figure, axes = plt.subplots(len(edges), 1, figsize=(13, 1.9 * len(edges)), squeeze=False)
    medians = {label: to_db(np.median(stack, axis=0)) for label, stack in arms.items()}
    for index, (low_hz, high_hz) in enumerate(edges):
        axis = axes[index][0]
        inside = (freqs >= low_hz) & (freqs <= high_hz)
        for label in _draw_order(medians):
            colour, name, width = _style(label)
            axis.plot(freqs[inside], medians[label][inside], color=colour, lw=width, label=name)
        axis.set_xlim(low_hz, high_hz)
        covered = bands_covering(low_hz, high_hz, bands)
        axis.set_ylabel(f"{low_hz:g}-{high_hz:g} Hz", fontsize=8)
        if covered:
            axis.text(
                0.995,
                0.93,
                covered,
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="#6B7280",
            )
        if index == 0:
            axis.legend(loc="upper left", fontsize=8)
        axis.tick_params(labelsize=8)
    axes[0][0].set_title(
        f"Power spectra before and after removal, {settings.panel_span_hz:g} Hz per panel. "
        "Welch via MNE."
    )
    axes[-1][0].set_xlabel("frequency (Hz)")
    figure.supylabel("median PSD (dB re 1 V²/Hz)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


ARM_STYLE = {
    # The source is drawn wide and pale beneath the derivatives, so where nothing changed
    # it survives as a halo instead of vanishing under the trace on top. Without that,
    # "this line was removed" and "this feature was already here" render identically --
    # which matters below 26 Hz, where the periodic dips belong to the gradient correction
    # and not to this pass at all.
    "source": ("#F3A28E", "before removal", 2.2),
    "line-cleaned": ("#111827", "after line removal", 0.7),
    "notched": ("#2563EB", "after line removal + notch", 0.7),
}
DEFAULT_STYLE = ("#6B7280", "", 0.7)


def _style(label: str) -> tuple[str, str, float]:
    colour, name, width = ARM_STYLE.get(label, DEFAULT_STYLE)
    return colour, name or label, width


def _draw_order(arms) -> tuple[str, ...]:
    """Source first so everything else lands on top of it."""
    return ("source", *(label for label in arms if label != "source"))


def figure_cohort(freqs, arms: dict[str, np.ndarray], path: Path, *, n_recordings: int) -> None:
    """Cohort median spectrum, and what the removal took, on one frequency axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[2, 1])
    for label in _draw_order(arms):
        colour, name, width = _style(label)
        axes[0].plot(
            freqs, to_db(np.median(arms[label], axis=0)), color=colour, lw=width, label=name
        )
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("median PSD (dB re 1 V²/Hz)")
    axes[0].set_title(
        f"Power spectra before and after removal, median of {n_recordings} recordings. "
        "Welch via MNE."
    )

    # What changed, which is the question the top panel only implies. Negative is removed.
    source = to_db(np.median(arms["source"], axis=0))
    for label in _draw_order(arms)[1:]:
        colour, name, _ = _style(label)
        stack = arms[label]
        change = to_db(np.median(stack, axis=0)) - source
        axes[1].plot(freqs, change, color=colour, lw=0.6, label=name)
        # "Added power here" and "removed power here" are different claims and should not
        # render identically. Over-subtraction against a cluster is what produces the first.
        gained = change > 0.0
        axes[1].plot(
            freqs[gained],
            change[gained],
            linestyle="none",
            marker="|",
            markersize=3,
            color="#C1442E",
        )
    axes[1].axhline(0.0, color="#6B7280", lw=0.6, ls="--")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("change (dB)")
    axes[1].set_xlim(freqs[0], freqs[-1])
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def figure_per_recording(freqs, per_recording: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    """One panel per recording, so a single bad one cannot hide inside a cohort median."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(per_recording)
    columns = min(3, len(names))
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.5 * columns, 2.8 * rows),
        squeeze=False,
        sharex=True,
    )
    for index, name in enumerate(names):
        axis = axes[index // columns][index % columns]
        for label in _draw_order(per_recording[name]):
            colour, _, width = _style(label)
            axis.plot(freqs, to_db(per_recording[name][label]), color=colour, lw=width * 0.7)
        axis.set_title(name, fontsize=8)
        axis.set_xlim(freqs[0], freqs[-1])
    for index in range(len(names), rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.supxlabel("frequency (Hz)")
    figure.supylabel("median PSD (dB re 1 V²/Hz)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    """Compare the source against every derivative that exists."""
    import time

    from studies.pain_study.scripts.workflow_config import load_workflow_config

    config = load_workflow_config(WORKFLOW, getattr(args, "config", None))
    settings = PsdSettings.from_config(config)
    removal = rm.RemovalSettings.from_config(config)
    source_root = config.path("bids_root", override=getattr(args, "bids_root", None))
    report_dir = config.path("removal_dir", override=getattr(args, "report_dir", None))

    candidates = [("line-cleaned", config.path("output_root"))]
    try:
        candidates.append(("notched", config.path("notched_root")))
    except KeyError:  # a config without the optional notch stage
        pass
    available = [(label, root) for label, root in candidates if root.is_dir()]
    if not available:
        raise FileNotFoundError(
            "Nothing to compare against: no cleaned dataset exists yet. Run "
            "`line-comb apply` first."
        )

    subjects = getattr(args, "subjects", None)
    runs = rm.discover_runs(source_root, subjects=subjects, task=removal.task)
    print(f"Measuring {len(runs)} recordings from {source_root}")
    for label, root in available:
        print(f"  against {label}: {root}")
    print(f"  Welch, {settings.window_s:g} s segments, {settings.overlap:.0%} overlap, EEG only")

    per_recording: dict[str, dict[str, np.ndarray]] = {}
    freqs = None
    for index, vhdr in enumerate(runs, start=1):
        started = time.time()
        derivatives = [(label, root / vhdr.relative_to(source_root)) for label, root in available]
        missing = [str(path) for _, path in derivatives if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{vhdr.stem}: derivative missing at {missing[0]}")
        run_freqs, spectra = compare_recording(vhdr, derivatives, settings)
        if freqs is None:
            freqs = run_freqs
        elif not np.array_equal(run_freqs, freqs):
            raise ValueError(f"{vhdr.stem}: spectra landed on a different grid from the first.")
        per_recording[vhdr.stem] = spectra
        print(f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} ({time.time() - started:.0f}s)")

    arms = {
        label: np.stack([per_recording[name][label] for name in sorted(per_recording)])
        for label in ("source", *(label for label, _ in available))
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_cohort(
        freqs,
        arms,
        report_dir / "psd_before_after.png",
        n_recordings=len(per_recording),
    )
    figure_band_panels(
        freqs,
        arms,
        report_dir / "psd_before_after_panels.png",
        settings=settings,
        bands=analysis_bands_from_config(config),
    )
    figure_per_recording(freqs, per_recording, report_dir / "psd_before_after_per_recording.png")
    np.savez_compressed(
        report_dir / "psd_before_after.npz",
        freqs=freqs,
        recordings=np.array(sorted(per_recording)),
        **arms,
    )

    source_db = to_db(np.median(arms["source"], axis=0))
    print("\nmedian change over the cohort:")
    for label, _ in available:
        change = to_db(np.median(arms[label], axis=0)) - source_db
        worst = int(np.argmin(change))
        print(
            f"  {label:14s} deepest {change[worst]:7.2f} dB at {freqs[worst]:7.3f} Hz; "
            f"bins changed by more than 1 dB: {np.mean(np.abs(change) > 1.0):.1%}"
        )
    print(f"\n  wrote {report_dir / 'psd_before_after.png'}")
    print(f"  wrote {report_dir / 'psd_before_after_panels.png'}")
    print(f"  wrote {report_dir / 'psd_before_after_per_recording.png'}")
    print(f"  wrote {report_dir / 'psd_before_after.npz'}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=None, help="Uncleaned source root")
    parser.add_argument("--subjects", nargs="*", default=None, help="Restrict the figure")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
