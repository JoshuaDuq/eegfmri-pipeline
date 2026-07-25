"""Exploratory band-specific ICA diagnostics for the MNE HTML report."""

from __future__ import annotations

import csv
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import mne
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.organize import remove_tagged_content
from eeg_pipeline.preprocessing.report.summary import (
    compute_removal_topography,
    decomposition_summary_html,
    plot_component_overview,
    plot_removal_topography,
    plot_run_component_variance,
    plot_variance_overview,
    removal_topography_html,
    summarize_decomposition,
)
from eeg_pipeline.preprocessing.report.style import (
    DIVERGING_POWER_COLORMAP,
    OKABE_ITO,
    PRIMARY_COLOR,
    REPORT_IMAGE_FORMAT,
    REPORT_RASTER_IMAGE_FORMAT,
    apply_report_style,
    power_colorbar_label,
    report_image_format,
    robust_symmetric_limit,
)


@dataclass(frozen=True)
class BandIcaDefinition:
    """One fixed frequency range in the exploratory ICA report."""

    slug: str
    title: str
    fmin: float
    fmax: float


@dataclass(frozen=True)
class ComponentLabel:
    """Exploratory ICLabel result for one component.

    The full class distribution is kept, not only the winning class. A component at
    brain 0.45 / muscle 0.42 is a different review decision from brain 0.98, and the
    winning label alone cannot distinguish them.
    """

    label: str
    probability: float
    #: Probability for each entry of ``_ICLABEL_CLASSES``; empty when ICLabel did not run.
    probabilities: tuple[float, ...] = ()

    @property
    def has_distribution(self) -> bool:
        return len(self.probabilities) == len(_ICLABEL_CLASSES)


@dataclass(frozen=True)
class TfrBandParameters:
    """FieldTrip-style DPSS parameters for one frequency interval."""

    fmin: float
    fmax: float
    window_seconds: float
    smoothing_hz: float


@dataclass(frozen=True)
class ConditionGroup:
    """One configured set of metadata values."""

    label: str
    values: tuple[Any, ...]


@dataclass(frozen=True)
class ConditionComparison:
    """A configured group-A minus group-B comparison."""

    name: str
    column: str
    group_a: ConditionGroup
    group_b: ConditionGroup


@dataclass(frozen=True)
class SourceDiagnostics:
    """Band-limited source spectrum and, for event-locked data, time-frequency power.

    The time-frequency fields are ``None`` for continuous recordings. Resting-state
    epochs are fixed-length segments with no event and no pre-stimulus interval, so a
    baseline-relative, event-locked TFR has nothing to be relative to; the spectrum is
    the interpretable component evidence there.
    """

    frequencies: np.ndarray
    power_db: np.ndarray
    tfr_frequencies: np.ndarray | None
    tfr_times: np.ndarray | None
    tfr: np.ndarray | None

    @property
    def has_tfr(self) -> bool:
        return self.tfr is not None


@dataclass(frozen=True)
class ConditionTfrResult:
    """Condition averages for one configured comparison."""

    comparison: ConditionComparison
    group_a_tfr: np.ndarray
    group_b_tfr: np.ndarray
    group_a_count: int
    group_b_count: int


@dataclass(frozen=True)
class BandReviewData:
    """All evidence displayed for one authoritative ICA frequency band."""

    band: BandIcaDefinition
    diagnostics: SourceDiagnostics
    comparisons: tuple[ConditionTfrResult, ...] = ()


@dataclass(frozen=True)
class BandIcaReportSettings:
    """Runtime controls for the computationally expensive report."""

    fit_decim: int = 2
    frequency_step_hz: float = 1.0
    time_min_s: float = -5.0
    time_max_s: float = 14.4
    time_step_s: float = 0.1
    baseline_tmin_s: float = -5.0
    #: Must clear the widest DPSS half-window (1.5 s) so that the centred tapers do
    #: not draw post-stimulus data into the pre-stimulus baseline.
    baseline_tmax_s: float = -1.5
    comparisons: tuple[ConditionComparison, ...] = ()
    run_iclabel: bool = False
    tfr_enabled: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> BandIcaReportSettings:
        tfr_values = values.get("tfr", {})
        if not isinstance(tfr_values, Mapping):
            raise TypeError("ica.band_specific_report.tfr must be a mapping.")
        settings = cls(
            fit_decim=int(values.get("fit_decim", cls.fit_decim)),
            frequency_step_hz=float(tfr_values.get("frequency_step_hz", cls.frequency_step_hz)),
            time_min_s=float(tfr_values.get("time_min_s", cls.time_min_s)),
            time_max_s=float(tfr_values.get("time_max_s", cls.time_max_s)),
            time_step_s=float(tfr_values.get("time_step_s", cls.time_step_s)),
            baseline_tmin_s=float(tfr_values.get("baseline_tmin_s", cls.baseline_tmin_s)),
            baseline_tmax_s=float(tfr_values.get("baseline_tmax_s", cls.baseline_tmax_s)),
            comparisons=_parse_comparisons(values.get("comparisons", [])),
            run_iclabel=bool(values.get("run_iclabel", cls.run_iclabel)),
            tfr_enabled=bool(tfr_values.get("enabled", cls.tfr_enabled)),
        )
        if settings.fit_decim < 1:
            raise ValueError("ica.band_specific_report.fit_decim must be at least 1.")
        if settings.frequency_step_hz <= 0:
            raise ValueError("ica.band_specific_report.tfr.frequency_step_hz must be positive.")
        if settings.time_step_s <= 0:
            raise ValueError("ica.band_specific_report.tfr.time_step_s must be positive.")
        if settings.time_min_s >= settings.time_max_s:
            raise ValueError("TFR time_min_s must be earlier than time_max_s.")
        if settings.baseline_tmin_s >= settings.baseline_tmax_s:
            raise ValueError("TFR baseline_tmin_s must be earlier than baseline_tmax_s.")
        if settings.tfr_enabled:
            required_margin = max(parameters.window_seconds for parameters in _TFR_PARAMETERS) / 2.0
            if settings.baseline_tmax_s > -required_margin:
                raise ValueError(
                    "ica.band_specific_report.tfr.baseline_tmax_s must be at most "
                    f"{-required_margin:g} s. The DPSS tapers are centred and up to "
                    f"{2 * required_margin:g} s wide, so a baseline ending at "
                    f"{settings.baseline_tmax_s:g} s estimates pre-stimulus power from "
                    f"data reaching {settings.baseline_tmax_s + required_margin:g} s "
                    "after the event. That leak scales with each condition's own "
                    "response, so it biases the condition differences rather than "
                    "cancelling in them."
                )
        if not settings.tfr_enabled and settings.comparisons:
            raise ValueError(
                "ica.band_specific_report.comparisons compare event-locked, "
                "baseline-relative TFRs and cannot be used with tfr.enabled=false. "
                "Resting-state recordings have no events to compare; remove the "
                "comparisons or enable the TFR."
            )
        return settings


BAND_ICA_DEFINITIONS = (
    BandIcaDefinition("deltatheta", "Delta + theta (1–8 Hz)", 1.0, 8.0),
    BandIcaDefinition("alpha", "Alpha (8–13 Hz)", 8.0, 13.0),
    BandIcaDefinition("beta", "Beta (13–30 Hz)", 13.0, 30.0),
    BandIcaDefinition("gamma", "Gamma (30–100 Hz)", 30.0, 100.0),
    BandIcaDefinition("broadband1to30", "Broadband 1–30 Hz", 1.0, 30.0),
)

_TFR_PARAMETERS = (
    TfrBandParameters(1.0, 8.0, 3.0, 1.0),
    TfrBandParameters(8.0, 13.0, 2.0, 1.5),
    TfrBandParameters(13.0, 30.0, 2.0, 2.5),
    TfrBandParameters(30.0, 100.0, 1.0, 5.0),
)

_ICLABEL_CLASSES = (
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
)


def _parse_group(values: Any, *, comparison_name: str, group_name: str) -> ConditionGroup:
    if not isinstance(values, Mapping):
        raise TypeError(f"Comparison {comparison_name!r} {group_name} must be a mapping.")
    label = str(values.get("label", "")).strip()
    group_values = values.get("values")
    if not label:
        raise ValueError(f"Comparison {comparison_name!r} {group_name}.label must not be empty.")
    if not isinstance(group_values, list) or not group_values:
        raise ValueError(
            f"Comparison {comparison_name!r} {group_name}.values must be a non-empty list."
        )
    return ConditionGroup(label=label, values=tuple(group_values))


def _parse_comparisons(values: Any) -> tuple[ConditionComparison, ...]:
    if not isinstance(values, list):
        raise TypeError("ica.band_specific_report.comparisons must be a list.")
    comparisons = []
    names = set()
    for value in values:
        if not isinstance(value, Mapping):
            raise TypeError("Each band-specific TFR comparison must be a mapping.")
        name = str(value.get("name", "")).strip()
        column = str(value.get("column", "")).strip()
        if not name or not column:
            raise ValueError("Each band-specific TFR comparison requires name and column.")
        if name in names:
            raise ValueError(f"Duplicate band-specific TFR comparison name: {name}")
        names.add(name)
        comparisons.append(
            ConditionComparison(
                name=name,
                column=column,
                group_a=_parse_group(
                    value.get("group_a"), comparison_name=name, group_name="group_a"
                ),
                group_b=_parse_group(
                    value.get("group_b"), comparison_name=name, group_name="group_b"
                ),
            )
        )
    return tuple(comparisons)


def _fit_band_ica(
    *,
    epochs: mne.BaseEpochs,
    random_state: int,
    fit_decim: int,
) -> mne.preprocessing.ICA:
    ica = mne.preprocessing.ICA(
        method="infomax",
        fit_params={"extended": True},
        n_components=None,
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(epochs, decim=fit_decim, verbose="ERROR")
    return ica


def _band_epochs(
    epochs: mne.BaseEpochs,
    band: BandIcaDefinition,
) -> mne.BaseEpochs:
    return epochs.copy().filter(
        l_freq=band.fmin,
        h_freq=band.fmax,
        picks="eeg",
        verbose="ERROR",
    )


def _label_components(
    *,
    epochs: mne.BaseEpochs,
    ica: mne.preprocessing.ICA,
) -> list[ComponentLabel]:
    from mne_icalabel.iclabel import iclabel_label_components

    probabilities = iclabel_label_components(inst=epochs, ica=ica, inplace=False)
    if probabilities.shape != (int(ica.n_components_), len(_ICLABEL_CLASSES)):
        raise ValueError("ICLabel probability matrix does not match the ICA components.")
    if not np.isfinite(probabilities).all():
        raise ValueError("ICLabel returned non-finite component probabilities.")
    return [
        ComponentLabel(
            label=_ICLABEL_CLASSES[int(np.argmax(component_probabilities))],
            probability=float(np.max(component_probabilities)),
            probabilities=tuple(float(value) for value in component_probabilities),
        )
        for component_probabilities in probabilities
    ]


def _component_spectrum(
    *,
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = (frequencies >= fmin) & (frequencies <= fmax)
    if not np.any(selected):
        raise ValueError(f"No spectrum frequencies fall within {fmin:g}–{fmax:g} Hz.")
    return frequencies[selected], spectrum[..., selected]


def _source_diagnostics(
    *,
    ica: mne.preprocessing.ICA,
    epochs: mne.BaseEpochs,
    band: BandIcaDefinition,
    settings: BandIcaReportSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sources = ica.get_sources(epochs)
    diagnostics = _source_diagnostics_from_sources(
        sources=sources,
        band=band,
        settings=settings,
    )
    return (
        diagnostics.frequencies,
        diagnostics.power_db,
        diagnostics.tfr_frequencies,
        diagnostics.tfr_times,
        diagnostics.tfr,
    )


def _source_diagnostics_from_sources(
    *,
    sources: mne.BaseEpochs,
    band: BandIcaDefinition,
    settings: BandIcaReportSettings,
) -> SourceDiagnostics:
    spectrum = sources.compute_psd(
        method="welch",
        fmin=band.fmin,
        fmax=band.fmax,
        picks="all",
        verbose="ERROR",
    )
    frequencies, power = _component_spectrum(
        frequencies=spectrum.freqs,
        spectrum=spectrum.get_data(picks=spectrum.ch_names).mean(axis=0),
        fmin=band.fmin,
        fmax=band.fmax,
    )
    power_db = 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny))

    if settings.tfr_enabled:
        tfr_frequencies, tfr_times, tfr = _fieldtrip_tfr(
            data=sources.get_data(copy=False),
            sfreq=float(sources.info["sfreq"]),
            times=sources.times,
            band=band,
            settings=settings,
        )
    else:
        tfr_frequencies, tfr_times, tfr = None, None, None
    return SourceDiagnostics(
        frequencies=frequencies,
        power_db=power_db,
        tfr_frequencies=tfr_frequencies,
        tfr_times=tfr_times,
        tfr=tfr,
    )


#: Display frequencies per DPSS smoothing half-width.
#:
#: Two samples per full smoothing width is the point beyond which extra frequencies add
#: no resolution, only compute: neighbouring bins spaced far below the smoothing are
#: near-duplicates. This value oversamples that sufficiency point so the rendered mesh
#: stays smooth, while ``frequency_step_hz`` remains the floor a caller can request.
#: At 1 Hz spacing the gamma band computed 71 bins against a ±5 Hz smoothing, ten per
#: smoothing width, which was over half of all time-frequency work in the report.
DISPLAY_SAMPLES_PER_SMOOTHING_HALF_WIDTH = 2.0

_NARROW_BAND_PARAMETERS = {
    "deltatheta": _TFR_PARAMETERS[0],
    "alpha": _TFR_PARAMETERS[1],
    "beta": _TFR_PARAMETERS[2],
    "gamma": _TFR_PARAMETERS[3],
}


def _tfr_parameter_groups(
    band: BandIcaDefinition,
    settings: BandIcaReportSettings,
) -> tuple[tuple[TfrBandParameters, np.ndarray], ...]:
    """Pair each DPSS parameter set with the display frequencies it produces.

    A wide band is covered by several parameter sets, so the spectral smoothing is not
    constant along the frequency axis. This is the single source of truth for that
    mapping, used both to compute the TFR and to annotate the resolution on the plot.
    """

    def grid(parameters: TfrBandParameters, low: float, high: float) -> np.ndarray:
        """Frequencies for one parameter set, spanning [low, high] inclusive.

        Both endpoints are included so that consecutive parameter sets meet exactly. A
        grid that stopped short of its upper edge would leave a sliver for the next set
        to claim, and that set would then smooth a single frequency by its own much
        wider kernel.
        """
        step = max(
            settings.frequency_step_hz,
            parameters.smoothing_hz / DISPLAY_SAMPLES_PER_SMOOTHING_HALF_WIDTH,
        )
        if high - low < step / 2.0:
            return np.array([low])
        count = max(2, int(round((high - low) / step)) + 1)
        return np.linspace(low, high, count)

    if band.slug in _NARROW_BAND_PARAMETERS:
        parameters = _NARROW_BAND_PARAMETERS[band.slug]
        return ((parameters, grid(parameters, band.fmin, band.fmax)),)

    # Parameter ranges are treated as closed and matched lowest-first, so a frequency
    # sitting exactly on a boundary belongs to the narrower-smoothing set below it.
    # Otherwise a band ending on a boundary hands that single frequency to the set above,
    # which would smooth a lone 30 Hz bin by ±5 Hz and draw half its estimate from
    # outside the band entirely.
    groups = []
    covered_above = band.fmin - 1.0
    for parameters in _TFR_PARAMETERS:
        low = max(band.fmin, parameters.fmin, covered_above)
        high = min(band.fmax, parameters.fmax)
        if high < low:
            continue
        selected = grid(parameters, low, high)
        if not selected.size:
            continue
        groups.append((parameters, selected))
        covered_above = float(selected[-1]) + 1e-9
    if not groups:
        raise ValueError(f"No DPSS parameters cover band {band.slug!r}.")
    return tuple(groups)


def _fieldtrip_tfr(
    *,
    data: np.ndarray,
    sfreq: float,
    times: np.ndarray,
    band: BandIcaDefinition,
    settings: BandIcaReportSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    decim = int(round(settings.time_step_s * sfreq))
    if decim < 1 or not np.isclose(decim / sfreq, settings.time_step_s):
        raise ValueError(
            f"TFR time step {settings.time_step_s:g} s is incompatible with {sfreq:g} Hz data."
        )

    frequency_parts = []
    power_parts = []
    for parameters, frequencies in _tfr_parameter_groups(band, settings):
        power = mne.time_frequency.tfr_array_multitaper(
            data,
            sfreq=sfreq,
            freqs=frequencies,
            n_cycles=frequencies * parameters.window_seconds,
            time_bandwidth=2.0 * parameters.window_seconds * parameters.smoothing_hz,
            output="avg_power",
            decim=decim,
            n_jobs=1,
            verbose="ERROR",
        )
        frequency_parts.append(frequencies)
        power_parts.append(power)

    frequencies = np.concatenate(frequency_parts)
    power = np.concatenate(power_parts, axis=1)
    decimated_times = times[::decim][: power.shape[-1]]

    time_tolerance = settings.time_step_s / 100.0
    baseline_mask = (decimated_times >= settings.baseline_tmin_s - time_tolerance) & (
        decimated_times <= settings.baseline_tmax_s + time_tolerance
    )
    if not np.any(baseline_mask):
        raise ValueError("No TFR samples fall inside the configured baseline window.")
    baseline_power = power[..., baseline_mask].mean(axis=-1, keepdims=True)
    power_db = 10.0 * np.log10(
        np.maximum(power, np.finfo(float).tiny) / np.maximum(baseline_power, np.finfo(float).tiny)
    )

    time_mask = (decimated_times >= settings.time_min_s - time_tolerance) & (
        decimated_times <= settings.time_max_s + time_tolerance
    )
    if not np.any(time_mask):
        raise ValueError("No TFR samples fall inside the configured display time range.")
    return frequencies, decimated_times[time_mask], power_db[..., time_mask]


def _tfr_configuration_title(
    band: BandIcaDefinition,
    settings: BandIcaReportSettings,
) -> str:
    if not settings.tfr_enabled:
        return (
            "Welch spectrum only · event-locked time-frequency power disabled "
            "(ica.band_specific_report.tfr.enabled=false)"
        )
    parameter_text = "; ".join(
        f"{parameters.window_seconds:g} s, ±{parameters.smoothing_hz:g} Hz"
        for parameters, _ in _tfr_parameter_groups(band, settings)
    )
    return (
        f"DPSS {parameter_text} · {settings.frequency_step_hz:g} Hz × "
        f"{settings.time_step_s:g} s grid · baseline "
        f"{settings.baseline_tmin_s:g}–{settings.baseline_tmax_s:g} s · relative dB"
    )


def _comparison_configuration_html(
    settings: BandIcaReportSettings,
    *,
    status: str = "Pending provisional task epochs",
) -> str:
    if not settings.comparisons:
        return (
            "<p><strong>No condition comparisons configured.</strong> Add entries under "
            "<code>ica.band_specific_report.comparisons</code> to compute provisional "
            "pre-review TFRs and finalized retained-epoch TFRs.</p>"
        )
    rows = []
    for comparison in settings.comparisons:
        rows.append(
            "<tr>"
            f"<td>{html.escape(comparison.name)}</td>"
            f"<td>{html.escape(comparison.column)}</td>"
            f"<td>{html.escape(comparison.group_a.label)}: "
            f"{html.escape(str(list(comparison.group_a.values)))}</td>"
            f"<td>{html.escape(comparison.group_b.label)}: "
            f"{html.escape(str(list(comparison.group_b.values)))}</td>"
            f"<td>{html.escape(status)}</td>"
            "</tr>"
        )
    return (
        "<p>Configured comparisons are first computed from all pre-ICA task epochs for manual "
        "component review, then replaced after rejection using retained epochs and aligned "
        "events metadata.</p>"
        "<table><thead><tr><th>Name</th><th>Column</th><th>Group A</th><th>Group B</th>"
        f"<th>Status</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _build_component_figures(
    *,
    ica: mne.preprocessing.ICA,
    epochs: mne.BaseEpochs,
    band: BandIcaDefinition,
    labels: Sequence[ComponentLabel],
    settings: BandIcaReportSettings,
) -> list[plt.Figure]:
    frequencies, power_db, tfr_frequencies, tfr_times, tfr = _source_diagnostics(
        ica=ica,
        epochs=epochs,
        band=band,
        settings=settings,
    )
    if len(labels) != int(ica.n_components_):
        raise ValueError("ICLabel result count does not match the band-specific ICA components.")

    figures = []
    panel_count = 3 if settings.tfr_enabled else 2
    color_limit = robust_symmetric_limit(tfr) if settings.tfr_enabled else None
    for component, label in enumerate(labels):
        figure, axes = plt.subplots(
            1,
            panel_count,
            figsize=(4.7 * panel_count, 4),
            layout="constrained",
        )
        ica.plot_components(
            picks=component,
            axes=[axes[0]],
            colorbar=False,
            show=False,
        )
        axes[0].set_title(f"ICA{component:03d} topomap")
        _add_iclabel_panel(axes[0], label)

        _plot_source_spectrum(
            axes[1],
            frequencies=frequencies,
            power_db=power_db,
            component=component,
            band=band,
            title="Source spectrum",
        )

        if settings.tfr_enabled:
            image = _plot_tfr(
                axis=axes[2],
                power=tfr[component],
                frequencies=tfr_frequencies,
                times=tfr_times,
                title="Source time-frequency power",
                color_limit=color_limit,
                resolution_groups=_tfr_parameter_groups(band, settings),
            )
            figure.colorbar(image, ax=axes[2], label=power_colorbar_label(color_limit))
        figure.suptitle(
            f"{band.title} · ICA{component:03d} · grand average · exploratory ICLabel: "
            f"{label.label} ({label.probability:.3f})\n"
            f"{_tfr_configuration_title(band, settings)}"
        )
        plt.close(figure)
        figures.append(figure)
    return figures


def _comparison_color_limits(
    group_a_tfr: np.ndarray,
    group_b_tfr: np.ndarray,
) -> tuple[float, float]:
    """Return robust symmetric limits for the two conditions and their difference."""
    condition_limit = robust_symmetric_limit(group_a_tfr, group_b_tfr)
    difference_limit = robust_symmetric_limit(group_a_tfr - group_b_tfr)
    return condition_limit, difference_limit


def _plot_source_spectrum(
    axis: plt.Axes,
    *,
    frequencies: np.ndarray,
    power_db: np.ndarray,
    component: int,
    band: BandIcaDefinition,
    title: str,
) -> None:
    """Plot one component's spectrum against the distribution across all components.

    A single spectrum in isolation is hard to judge: EEG power falls off as roughly 1/f,
    so every component looks similar in shape. What decides a review is whether this
    component departs from the others, so the interquartile range and median across the
    decomposition are drawn behind it as the reference.

    The frequency axis is logarithmic whenever the band spans more than an octave. On a
    linear axis a 1-30 Hz band gives delta a handful of pixels and beta several hundred,
    which hides exactly the low-frequency structure that distinguishes artifacts.
    """
    lower, median, upper = np.percentile(power_db, [25, 50, 75], axis=0)
    axis.fill_between(
        frequencies,
        lower,
        upper,
        color="0.85",
        linewidth=0,
        label="All components (IQR)",
    )
    axis.plot(frequencies, median, color="0.55", linewidth=1.0, label="Median")
    axis.plot(
        frequencies,
        power_db[component],
        color=PRIMARY_COLOR,
        linewidth=1.6,
        label=f"IC{component:03d}",
    )

    peak_index = int(np.argmax(power_db[component]))
    axis.annotate(
        f"peak {frequencies[peak_index]:.1f} Hz",
        xy=(frequencies[peak_index], power_db[component][peak_index]),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=6.5,
        color=PRIMARY_COLOR,
    )
    axis.set(title=title, xlabel="Frequency (Hz)", ylabel="Power (dB)")
    if band.fmax / max(band.fmin, 1e-9) >= 2.0:
        axis.set_xscale("log")
        axis.set_xticks(
            [tick for tick in (1, 2, 4, 8, 13, 20, 30, 50, 100) if band.fmin <= tick <= band.fmax]
        )
        axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.set_xlim(band.fmin, band.fmax)
    axis.legend(frameon=False, fontsize=6)


def _plot_tfr(
    *,
    axis: plt.Axes,
    power: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    title: str,
    color_limit: float,
    resolution_groups: Sequence[tuple[TfrBandParameters, np.ndarray]] | None = None,
):
    image = axis.pcolormesh(
        times,
        frequencies,
        power,
        shading="auto",
        cmap=DIVERGING_POWER_COLORMAP,
        vmin=-color_limit,
        vmax=color_limit,
        rasterized=True,
    )
    axis.axvline(0.0, color="black", linestyle="--", linewidth=0.75)
    axis.set(title=title, xlabel="Time (s)", ylabel="Frequency (Hz)")
    if resolution_groups is not None:
        _annotate_spectral_resolution(axis, resolution_groups)
    return image


def _annotate_spectral_resolution(
    axis: plt.Axes,
    groups: Sequence[tuple[TfrBandParameters, np.ndarray]],
) -> None:
    """Mark where the DPSS spectral smoothing changes along the frequency axis.

    The display grid is uniform, but the smoothing is not: a wide band is covered by
    several DPSS parameter sets, so the same pixel height means a different effective
    frequency resolution in delta than in gamma. Without this the plot would imply a
    resolution it does not have.

    A band covered by a single parameter set needs no annotation: the smoothing is
    constant, the figure title already states it, and drawing it over the data would
    add clutter that carries no information.
    """
    if len(groups) < 2:
        return
    for index, (parameters, frequencies) in enumerate(groups):
        if index > 0:
            axis.axhline(
                float(frequencies[0]),
                color="0.35",
                linestyle=":",
                linewidth=0.7,
            )
        axis.annotate(
            f"±{parameters.smoothing_hz:g} Hz",
            xy=(1.0, float(np.median(frequencies))),
            xycoords=("axes fraction", "data"),
            xytext=(-3, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=6,
            color="0.25",
            bbox={
                "boxstyle": "square,pad=0.15",
                "facecolor": "white",
                "alpha": 0.7,
                "edgecolor": "none",
            },
        )


#: One colour per ICLabel class, ordered as ``_ICLABEL_CLASSES``.
_ICLABEL_COLORS = (
    OKABE_ITO["bluish_green"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["orange"],
    OKABE_ITO["yellow"],
    "0.75",
)


def _plot_iclabel_distribution(axis: plt.Axes, label: ComponentLabel) -> None:
    """Draw the full ICLabel class distribution as one stacked probability bar.

    The winning class and its probability are already stated in the title. This shows
    how much of the remaining probability mass sits on competing classes, which is what
    separates a confident classification from a coin flip between two artifact types.
    """
    left = 0.0
    for probability, name, color in zip(
        label.probabilities,
        _ICLABEL_CLASSES,
        _ICLABEL_COLORS,
        strict=True,
    ):
        axis.barh(0, probability, left=left, color=color, height=1.0)
        if probability >= 0.12:
            axis.text(
                left + probability / 2.0,
                0,
                name.split()[0],
                ha="center",
                va="center",
                fontsize=6,
                color="black",
            )
        left += probability
    axis.set(xlim=(0.0, 1.0), ylim=(-0.5, 0.5), yticks=[])
    axis.tick_params(axis="x", labelsize=6, length=2)
    axis.set_xlabel("ICLabel probability", fontsize=6)
    axis.spines[["top", "right", "left"]].set_visible(False)


def _add_iclabel_panel(axis: plt.Axes, label: ComponentLabel) -> None:
    """Attach the ICLabel distribution beneath a component topography."""
    if not label.has_distribution:
        return
    _plot_iclabel_distribution(axis.inset_axes([0.0, -0.16, 1.0, 0.08]), label)


def _component_review_status(ica: mne.preprocessing.ICA, component: int) -> str:
    return "AUTO-MARKED BAD" if component in ica.exclude else "RETAINED"


def _plot_dossier_summary(
    *,
    figure: plt.Figure,
    axes: np.ndarray,
    ica: mne.preprocessing.ICA,
    review: BandReviewData,
    label: ComponentLabel,
    component: int,
    color_limit: float | None,
    settings: BandIcaReportSettings,
) -> None:
    ica.plot_components(
        picks=component,
        axes=[axes[0]],
        colorbar=False,
        show=False,
    )
    axes[0].set_title(f"ICA{component:03d} topomap")
    _add_iclabel_panel(axes[0], label)
    _plot_source_spectrum(
        axes[1],
        frequencies=review.diagnostics.frequencies,
        power_db=review.diagnostics.power_db,
        component=component,
        band=review.band,
        title="Band-limited source spectrum",
    )
    if not review.diagnostics.has_tfr:
        return
    image = _plot_tfr(
        axis=axes[2],
        power=review.diagnostics.tfr[component],
        frequencies=review.diagnostics.tfr_frequencies,
        times=review.diagnostics.tfr_times,
        title="Grand average",
        color_limit=color_limit,
        resolution_groups=_tfr_parameter_groups(review.band, settings),
    )
    figure.colorbar(image, ax=axes[2], label=power_colorbar_label(color_limit))


def _comparison_titles(result: ConditionTfrResult) -> tuple[str, str, str]:
    comparison = result.comparison
    return (
        f"{comparison.group_a.label} · n={result.group_a_count}\n"
        f"{comparison.column} ∈ {list(comparison.group_a.values)}",
        f"{comparison.group_b.label} · n={result.group_b_count}\n"
        f"{comparison.column} ∈ {list(comparison.group_b.values)}",
        f"{comparison.name}\n{comparison.group_a.label} − {comparison.group_b.label}",
    )


def _plot_dossier_comparison(
    *,
    figure: plt.Figure,
    axes: np.ndarray,
    result: ConditionTfrResult,
    review: BandReviewData,
    component: int,
    color_limits: tuple[float, float],
) -> None:
    difference = result.group_a_tfr - result.group_b_tfr
    images = [
        _plot_tfr(
            axis=axis,
            power=power,
            frequencies=review.diagnostics.tfr_frequencies,
            times=review.diagnostics.tfr_times,
            title=title,
            color_limit=color_limit,
        )
        for axis, title, power, color_limit in zip(
            axes,
            _comparison_titles(result),
            (
                result.group_a_tfr[component],
                result.group_b_tfr[component],
                difference[component],
            ),
            (color_limits[0], color_limits[0], color_limits[1]),
        )
    ]
    figure.colorbar(
        images[0],
        ax=axes[:2],
        label=power_colorbar_label(color_limits[0]),
    )
    figure.colorbar(
        images[2],
        ax=axes[2],
        label=power_colorbar_label(color_limits[1], quantity="Group difference"),
    )


def _create_component_dossier(
    *,
    ica: mne.preprocessing.ICA,
    review: BandReviewData,
    label: ComponentLabel,
    component: int,
    grand_average_limit: float | None,
    comparison_limits: Sequence[tuple[float, float]],
    settings: BandIcaReportSettings,
    analysis_status: str,
) -> plt.Figure:
    row_count = 1 + len(review.comparisons)
    column_count = 3 if review.diagnostics.has_tfr else 2
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.3 * column_count, 3.8 * row_count),
        squeeze=False,
        layout="constrained",
    )
    _plot_dossier_summary(
        figure=figure,
        axes=axes[0],
        ica=ica,
        review=review,
        label=label,
        component=component,
        color_limit=grand_average_limit,
        settings=settings,
    )
    for row, (result, color_limits) in enumerate(
        zip(review.comparisons, comparison_limits),
        start=1,
    ):
        _plot_dossier_comparison(
            figure=figure,
            axes=axes[row],
            result=result,
            review=review,
            component=component,
            color_limits=color_limits,
        )
    figure.suptitle(
        f"{review.band.title} · ICA{component:03d} · "
        f"{label.label} ({label.probability:.3f}) · "
        f"{_component_review_status(ica, component)}\n"
        f"{analysis_status} · {_tfr_configuration_title(review.band, settings)}"
    )
    plt.close(figure)
    return figure


def _build_standard_component_dossiers(
    *,
    ica: mne.preprocessing.ICA,
    review: BandReviewData,
    labels: Sequence[ComponentLabel],
    settings: BandIcaReportSettings,
    analysis_status: str,
) -> list[plt.Figure]:
    """Render one complete manual-review slide per authoritative ICA component."""
    if len(labels) != int(ica.n_components_):
        raise ValueError("ICLabel result count does not match the standard ICA components.")

    grand_average_limit = (
        robust_symmetric_limit(review.diagnostics.tfr) if review.diagnostics.has_tfr else None
    )
    comparison_limits = [
        _comparison_color_limits(result.group_a_tfr, result.group_b_tfr)
        for result in review.comparisons
    ]

    return [
        _create_component_dossier(
            ica=ica,
            review=review,
            label=label,
            component=component,
            grand_average_limit=grand_average_limit,
            comparison_limits=comparison_limits,
            settings=settings,
            analysis_status=analysis_status,
        )
        for component, label in enumerate(labels)
    ]


def _organize_component_review(report: mne.Report) -> None:
    """Keep authoritative review sections together before MNE's ICA components."""
    content = report._content
    review_indices = [
        index for index, element in enumerate(content) if "ica-component-review" in element.tags
    ]
    if not review_indices:
        raise ValueError("The report has no authoritative ICA component-review content.")
    manual_review_indices = [
        index for index, element in enumerate(content) if element.section == "ICA: components"
    ]
    if not manual_review_indices:
        raise ValueError("The report has no MNE 'ICA: components' section.")

    remaining_indices = [index for index in range(len(content)) if index not in review_indices]
    insertion_index = remaining_indices.index(manual_review_indices[0])
    order = (
        remaining_indices[:insertion_index] + review_indices + remaining_indices[insertion_index:]
    )
    report.reorder(order)


def _comparison_masks(
    metadata: pd.DataFrame,
    comparison: ConditionComparison,
) -> tuple[np.ndarray, np.ndarray]:
    if comparison.column not in metadata.columns:
        raise ValueError(
            f"Comparison {comparison.name!r} requires missing clean-events column "
            f"{comparison.column!r}."
        )
    values = metadata[comparison.column]
    group_a_mask = values.isin(comparison.group_a.values).to_numpy()
    group_b_mask = values.isin(comparison.group_b.values).to_numpy()
    if not group_a_mask.any():
        raise ValueError(
            f"Comparison {comparison.name!r} has no retained trials for "
            f"{comparison.group_a.label!r}."
        )
    if not group_b_mask.any():
        raise ValueError(
            f"Comparison {comparison.name!r} has no retained trials for "
            f"{comparison.group_b.label!r}."
        )
    if np.any(group_a_mask & group_b_mask):
        raise ValueError(f"Comparison {comparison.name!r} groups overlap.")
    return group_a_mask, group_b_mask


def _condition_tfr_results(
    *,
    source_data: np.ndarray,
    metadata: pd.DataFrame,
    sfreq: float,
    times: np.ndarray,
    band: BandIcaDefinition,
    settings: BandIcaReportSettings,
) -> tuple[ConditionTfrResult, ...]:
    results = []
    for comparison in settings.comparisons:
        group_a_mask, group_b_mask = _comparison_masks(metadata, comparison)
        _, _, group_a_tfr = _fieldtrip_tfr(
            data=source_data[group_a_mask],
            sfreq=sfreq,
            times=times,
            band=band,
            settings=settings,
        )
        _, _, group_b_tfr = _fieldtrip_tfr(
            data=source_data[group_b_mask],
            sfreq=sfreq,
            times=times,
            band=band,
            settings=settings,
        )
        results.append(
            ConditionTfrResult(
                comparison=comparison,
                group_a_tfr=group_a_tfr,
                group_b_tfr=group_b_tfr,
                group_a_count=int(group_a_mask.sum()),
                group_b_count=int(group_b_mask.sum()),
            )
        )
    return tuple(results)


def _build_band_review_data(
    *,
    ica: mne.preprocessing.ICA,
    epochs: mne.BaseEpochs,
    metadata: pd.DataFrame | None,
    band: BandIcaDefinition,
    settings: BandIcaReportSettings,
) -> BandReviewData:
    band_epochs = _band_epochs(epochs, band)
    sources = ica.get_sources(band_epochs)
    diagnostics = _source_diagnostics_from_sources(
        sources=sources,
        band=band,
        settings=settings,
    )
    if metadata is None:
        comparison_results = ()
    else:
        source_data = sources.get_data(copy=False)
        if len(metadata) != source_data.shape[0]:
            raise ValueError("ICA review epochs and events metadata must have identical lengths.")
        comparison_results = _condition_tfr_results(
            source_data=source_data,
            metadata=metadata,
            sfreq=float(sources.info["sfreq"]),
            times=sources.times,
            band=band,
            settings=settings,
        )
    return BandReviewData(
        band=band,
        diagnostics=diagnostics,
        comparisons=comparison_results,
    )


def _component_captions(
    ica: mne.preprocessing.ICA,
    labels: Sequence[ComponentLabel],
) -> list[str]:
    return [
        f"ICA{component:03d} · {label.label} ({label.probability:.3f}) · "
        f"{_component_review_status(ica, component)}"
        for component, label in enumerate(labels)
    ]


def _review_context_html(
    band: BandIcaDefinition,
    settings: BandIcaReportSettings,
    analysis_status: str,
) -> str:
    if not settings.tfr_enabled:
        return (
            f"<p><strong>{html.escape(analysis_status)}</strong>. Each slide keeps one "
            "standard ICA component's topography and band-limited Welch spectrum "
            "together.</p>"
            "<p>Event-locked time-frequency power is disabled. Continuous and "
            "resting-state recordings are segmented into fixed-length epochs with no "
            "event and no pre-stimulus interval, so a baseline-relative TFR would have "
            "no baseline to be relative to. The spectrum and the topography are the "
            "interpretable component evidence here.</p>"
        )
    return (
        f"<p><strong>{html.escape(analysis_status)}</strong>. Each slide keeps one "
        "standard ICA component's topography, band-limited spectrum, grand-average TFR, "
        "and configured condition comparisons together.</p>"
        f"<p>{html.escape(_tfr_configuration_title(band, settings))}. Condition A and B "
        "share one symmetric color scale; their difference uses a separate symmetric "
        "zero-centred scale.</p>"
    )


def _review_guide_html(
    settings: BandIcaReportSettings,
    analysis_status: str,
) -> str:
    return (
        "<p><strong>Authoritative manual-review components.</strong> Component numbers in "
        "these sections all refer to the same standard broadband ICA model used for "
        "artifact removal.</p>"
        "<p>The independently fitted band-specific ICAs remain exploratory: their component "
        "numbers do not correspond numerically across bands or to the standard ICA.</p>"
        "<p>The stacked bar under each topography is the full ICLabel class distribution, "
        "not only the winning class. A bar split between two classes marks a component "
        "whose classification is uncertain and that deserves a decision on the evidence "
        "in the other panels rather than on the label.</p>"
        + _comparison_configuration_html(settings, status=analysis_status)
    )


def _remove_legacy_condition_tfr_entries(report: mne.Report) -> None:
    legacy_titles = {
        element.name
        for element in report._content
        if "band-specific-ica" in element.tags
        and {"condition-tfr", "condition-tfr-configuration"}.intersection(element.tags)
    }
    for title in legacy_titles:
        report.remove(
            title=title,
            tags=("band-specific-ica",),
            remove_all=True,
        )


_DECOMPOSITION_SECTION = "ICA decomposition quality"


def _slider_title(title: str, slide_count: int) -> str:
    """Announce how many figures sit behind a slider.

    MNE renders a list of figures as a range slider showing one at a time. Without the
    count in the title a reviewer scrolling the report sees a single plot and has no
    reason to think the remaining ones exist.
    """
    return f"{title} — {slide_count} figures, use the slider"


def _add_decomposition_summary(
    *,
    report: mne.Report,
    ica: mne.preprocessing.ICA,
    epochs: mne.BaseEpochs,
    labels: Sequence[ComponentLabel],
    filtered_raw_paths: Sequence[Path] | None,
) -> None:
    """Add whole-decomposition evidence ahead of the per-component review."""
    summary = summarize_decomposition(ica=ica, epochs=epochs)
    report.add_figure(
        fig=plot_component_overview(ica=ica, labels=labels),
        title="All component topographies",
        section=_DECOMPOSITION_SECTION,
        tags=("ica", "ica-component-review", "ica-decomposition"),
        image_format=REPORT_RASTER_IMAGE_FORMAT,
        replace=True,
    )
    report.add_html(
        html=decomposition_summary_html(summary),
        title="Decomposition summary",
        section=_DECOMPOSITION_SECTION,
        tags=("ica", "ica-component-review", "ica-decomposition"),
        replace=True,
    )
    report.add_figure(
        fig=plot_variance_overview(summary),
        title="Sensor variance per component",
        section=_DECOMPOSITION_SECTION,
        tags=("ica", "ica-component-review", "ica-decomposition"),
        image_format=REPORT_IMAGE_FORMAT,
        replace=True,
    )
    # The variance figure says how much was removed; this says from where, which is what
    # separates focal artifact removal from uniform signal loss.
    topography = compute_removal_topography(ica=ica, epochs=epochs)
    report.add_html(
        html=removal_topography_html(topography),
        title="Spatial signature of the removal",
        section=_DECOMPOSITION_SECTION,
        tags=("ica", "ica-component-review", "ica-decomposition"),
        replace=True,
    )
    report.add_figure(
        fig=plot_removal_topography(topography),
        title="Per-channel amplitude removed by ICA",
        section=_DECOMPOSITION_SECTION,
        tags=("ica", "ica-component-review", "ica-decomposition"),
        image_format=REPORT_IMAGE_FORMAT,
        replace=True,
    )
    if filtered_raw_paths:
        report.add_figure(
            fig=plot_run_component_variance(
                ica=ica,
                filtered_raw_paths=list(filtered_raw_paths),
            ),
            title="Component variance by run",
            section=_DECOMPOSITION_SECTION,
            tags=("ica", "ica-component-review", "ica-decomposition"),
            image_format=REPORT_IMAGE_FORMAT,
            replace=True,
        )


def _add_component_properties(
    *,
    report: mne.Report,
    ica: mne.preprocessing.ICA,
    epochs: mne.BaseEpochs,
    labels: Sequence[ComponentLabel],
) -> None:
    """Add MNE's canonical per-component panel.

    The dossiers show topography, spectrum, and time-frequency power, but no view of the
    component's actual time course. ``plot_properties`` supplies the standard evidence
    for that: the epochs image, the ERP, and the per-epoch variance that reveals whether
    a component is driven by a handful of epochs.
    """
    figures = ica.plot_properties(
        epochs,
        picks=list(range(int(ica.n_components_))),
        show=False,
        verbose="ERROR",
    )
    for figure in figures:
        plt.close(figure)
    report.add_figure(
        fig=figures,
        title=_slider_title("Component properties", int(ica.n_components_)),
        caption=_component_captions(ica, labels),
        section=_DECOMPOSITION_SECTION,
        tags=("ica", "ica-component-review", "ica-decomposition", "ica-properties"),
        image_format=REPORT_RASTER_IMAGE_FORMAT,
        replace=True,
    )


def _add_standard_component_review(
    *,
    report: mne.Report,
    ica: mne.preprocessing.ICA,
    epochs: mne.BaseEpochs,
    metadata: pd.DataFrame | None,
    labels: Sequence[ComponentLabel],
    settings: BandIcaReportSettings,
    analysis_status: str,
    has_fallback_runs: bool = False,
    filtered_raw_paths: Sequence[Path] | None = None,
) -> None:
    """Add authoritative, component-centred evidence before MNE's ICA section."""
    _remove_legacy_condition_tfr_entries(report)
    # Every panel this function writes carries "ica-component-review". Clearing the tag
    # first makes the rebuild idempotent: renaming a panel cannot strand the previous
    # one, because removal is keyed to the tag rather than to the title.
    remove_tagged_content(report, tag="ica-component-review")
    _add_decomposition_summary(
        report=report,
        ica=ica,
        epochs=epochs,
        labels=labels,
        filtered_raw_paths=filtered_raw_paths,
    )
    guide_title = "How to review ICA component dossiers"
    report.remove(title=guide_title, remove_all=True)
    guide_html = _review_guide_html(settings, analysis_status)
    if has_fallback_runs:
        warning_html = (
            '<div style="background-color: #fff3cd; color: #856404; padding: 15px; '
            'margin-bottom: 20px; border: 1px solid #ffeeba; border-radius: 4px;">'
            "<strong>&#9888; DATA QUALITY WARNING:</strong> "
            "Some runs in this subject lacked manual BrainVision R-peak markers "
            "(Analyzer defaulted to a 0.21s delay). "
            "Cardiac QC metrics (CTPS and attenuation) for this subject relied on "
            "automated MNE fallback detection and may be noisier than standard."
            "</div>"
        )
        guide_html = warning_html + guide_html

    report.add_html(
        html=guide_html,
        title=guide_title,
        section="ICA component review guide",
        tags=("ica", "ica-component-review", "ica-review-guide"),
        replace=True,
    )

    _add_component_properties(report=report, ica=ica, epochs=epochs, labels=labels)

    captions = _component_captions(ica, labels)
    for band in BAND_ICA_DEFINITIONS:
        review = _build_band_review_data(
            ica=ica,
            epochs=epochs,
            metadata=metadata,
            band=band,
            settings=settings,
        )
        figures = _build_standard_component_dossiers(
            ica=ica,
            review=review,
            labels=labels,
            settings=settings,
            analysis_status=analysis_status,
        )
        section = f"ICA component review: {band.title}"
        report.add_html(
            html=_review_context_html(band, settings, analysis_status),
            title="Review context",
            section=section,
            tags=("ica", "ica-component-review", band.slug),
            replace=True,
        )
        report.add_figure(
            fig=figures,
            title=_slider_title("Component dossiers", len(figures)),
            caption=captions,
            section=section,
            tags=("ica", "ica-component-review", "condition-tfr", band.slug),
            image_format=report_image_format(
                has_dense_image=settings.tfr_enabled, is_figure_list=True
            ),
            replace=True,
        )
    _organize_component_review(report)


def _write_component_table(
    *,
    path: Path,
    labels: Sequence[ComponentLabel],
) -> None:
    """Write one row per component, keeping the full ICLabel class distribution.

    The winning class and its probability are retained as the first columns; the
    per-class columns let a reviewer see how close the runner-up was without
    re-running ICLabel.
    """
    distribution_columns = tuple(
        f"probability_{name.replace(' ', '_')}" for name in _ICLABEL_CLASSES
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("component", "iclabel", "probability", "interpretation")
            + distribution_columns,
            delimiter="\t",
        )
        writer.writeheader()
        for component, label in enumerate(labels):
            row = {
                "component": component,
                "iclabel": label.label,
                "probability": f"{label.probability:.6f}",
                "interpretation": "exploratory",
            }
            distribution = (
                label.probabilities if label.has_distribution else ("",) * len(_ICLABEL_CLASSES)
            )
            for column, value in zip(distribution_columns, distribution, strict=True):
                row[column] = f"{value:.6f}" if value != "" else ""
            writer.writerow(row)


def _select_retained_epochs(
    pre_ica_epochs: mne.BaseEpochs,
    clean_epochs: mne.BaseEpochs,
) -> mne.BaseEpochs:
    """Map retained original-event selections back to pre-ICA row positions."""
    pre_ica_selection = np.asarray(pre_ica_epochs.selection, dtype=int)
    clean_selection = np.asarray(clean_epochs.selection, dtype=int)
    if len(np.unique(pre_ica_selection)) != len(pre_ica_selection):
        raise ValueError("Pre-ICA epoch selection contains duplicate event indices.")
    selection_positions = {
        event_index: position for position, event_index in enumerate(pre_ica_selection.tolist())
    }
    missing = sorted(set(clean_selection.tolist()) - selection_positions.keys())
    if missing:
        raise ValueError(
            f"Clean epoch selection contains event indices absent from pre-ICA epochs: {missing}"
        )
    retained_positions = [selection_positions[index] for index in clean_selection.tolist()]
    return pre_ica_epochs[retained_positions]


def append_condition_tfr_report(
    *,
    ica_fit_epochs_path: Path,
    standard_ica_path: Path,
    pre_ica_epochs_path: Path,
    clean_epochs_path: Path,
    clean_events_path: Path,
    report_path: Path,
    settings: BandIcaReportSettings,
    analysis_status: str,
) -> None:
    """Replace authoritative component dossiers with condition-aware evidence."""
    if not settings.comparisons:
        return
    apply_report_style()
    ica_fit_epochs = mne.read_epochs(ica_fit_epochs_path, preload=True, verbose="ERROR")
    pre_ica_epochs = mne.read_epochs(pre_ica_epochs_path, preload=True, verbose="ERROR")
    clean_epochs = mne.read_epochs(clean_epochs_path, preload=False, verbose="ERROR")
    clean_events = pd.read_csv(clean_events_path, sep="\t")
    if len(clean_events) != len(clean_epochs):
        raise ValueError(
            "Clean events and clean epochs must have identical row counts for TFR comparisons."
        )
    expected_epoch_indices = np.arange(len(clean_events))
    if "epoch_index" not in clean_events.columns or not np.array_equal(
        clean_events["epoch_index"].to_numpy(), expected_epoch_indices
    ):
        raise ValueError("Clean events require contiguous zero-based epoch_index values.")
    retained_epochs = _select_retained_epochs(pre_ica_epochs, clean_epochs)
    if len(retained_epochs) != len(clean_events):
        raise ValueError("Retained pre-ICA epochs do not align with clean events.")

    standard_ica = mne.preprocessing.read_ica(standard_ica_path, verbose="ERROR")
    labels = _label_components(epochs=ica_fit_epochs, ica=standard_ica)
    report = mne.open_report(report_path)
    _add_standard_component_review(
        report=report,
        ica=standard_ica,
        epochs=retained_epochs,
        metadata=clean_events,
        labels=labels,
        settings=settings,
        analysis_status=analysis_status,
    )
    report.save(report_path, overwrite=True, open_browser=False)
    report.save(report_path.with_suffix(".html"), overwrite=True, open_browser=False)


def generate_band_ica_report(
    *,
    epochs_path: Path,
    report_path: Path,
    output_dir: Path,
    output_prefix: str,
    random_state: int,
    settings: BandIcaReportSettings,
    filtered_raw_paths: Sequence[Path] | None = None,
) -> list[Path]:
    """Fit exploratory band-specific ICAs and append diagnostics to an MNE report."""
    apply_report_style()
    epochs = mne.read_epochs(epochs_path, preload=True, verbose="ERROR")
    nyquist = float(epochs.info["sfreq"]) / 2.0
    maximum_frequency = max(band.fmax for band in BAND_ICA_DEFINITIONS)
    if maximum_frequency >= nyquist:
        raise ValueError(
            f"Band-specific ICA requires Nyquist above {maximum_frequency:g} Hz; "
            f"received {nyquist:g} Hz."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report = mne.open_report(report_path)
    standard_ica_path = epochs_path.with_name(f"{output_prefix}_proc-ica_ica.fif")
    if not standard_ica_path.is_file():
        raise FileNotFoundError(f"Standard ICA does not exist: {standard_ica_path}")
    standard_ica = mne.preprocessing.read_ica(standard_ica_path, verbose="ERROR")
    standard_labels = _label_components(epochs=epochs, ica=standard_ica)

    components_path = epochs_path.with_name(f"{output_prefix}_proc-ica_components.tsv")
    has_fallback_runs = False
    if components_path.is_file():
        components = pd.read_csv(components_path, sep="\t")
        if "analyzer_marker_ctps_fallback" in components.columns:
            has_fallback_runs = bool(components["analyzer_marker_ctps_fallback"].any())

    _add_standard_component_review(
        report=report,
        ica=standard_ica,
        epochs=epochs,
        metadata=None,
        labels=standard_labels,
        settings=settings,
        analysis_status="Pending provisional task epochs",
        has_fallback_runs=has_fallback_runs,
        filtered_raw_paths=filtered_raw_paths,
    )
    generated_paths = []
    for band in BAND_ICA_DEFINITIONS:
        filtered_epochs = _band_epochs(epochs, band)
        ica = _fit_band_ica(
            epochs=filtered_epochs,
            random_state=random_state,
            fit_decim=settings.fit_decim,
        )
        ica.exclude = []
        if settings.run_iclabel:
            labels = _label_components(epochs=filtered_epochs, ica=ica)
        else:
            labels = [
                ComponentLabel(label="unlabeled", probability=0.0)
                for _ in range(int(ica.n_components_))
            ]
        figures = _build_component_figures(
            ica=ica,
            epochs=filtered_epochs,
            band=band,
            labels=labels,
            settings=settings,
        )

        prefix = f"{output_prefix}_desc-{band.slug}"
        ica_path = output_dir / f"{prefix}_ica.fif"
        table_path = output_dir / f"{prefix}_components.tsv"
        ica.save(ica_path, overwrite=True)
        _write_component_table(path=table_path, labels=labels)
        generated_paths.extend((ica_path, table_path))

        section = f"Band-specific ICA: {band.title}"
        report.remove(
            title=f"{band.title}: component topomaps, spectra, and TFRs",
            remove_all=True,
        )
        report.add_html(
            title="Interpretation",
            html=(
                "<p><strong>Exploratory only.</strong> This ICA was fitted to "
                f"{band.fmin:g}–{band.fmax:g} Hz data. ICLabel was not validated "
                "for narrow-band decompositions. Component numbers do not correspond "
                "across bands or to the standard ICA, and these labels do not control "
                "artifact removal.</p>"
            ),
            section=section,
            tags=("ica", "band-specific-ica", band.slug),
            replace=True,
        )
        report.add_figure(
            fig=figures,
            title=(
                f"{band.title}: component topomaps, spectra, and grand-average "
                f"relative TFRs · {_tfr_configuration_title(band, settings)}"
            ),
            section=section,
            tags=("ica", "band-specific-ica", band.slug),
            image_format=report_image_format(
                has_dense_image=settings.tfr_enabled, is_figure_list=True
            ),
            replace=True,
        )
    report.save(report_path, overwrite=True, open_browser=False)
    report.save(report_path.with_suffix(".html"), overwrite=True, open_browser=False)
    return generated_paths
