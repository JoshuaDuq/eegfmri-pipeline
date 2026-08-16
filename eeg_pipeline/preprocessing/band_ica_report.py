"""Exploratory band-specific ICA diagnostics for the MNE HTML report."""

from __future__ import annotations

import csv
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import mne
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.ica_exclusions import read_ica_with_reviewed_exclusions
from eeg_pipeline.preprocessing.report.build_record import save_subject_report
from eeg_pipeline.preprocessing.report.organize import (
    drop_superseded_mne_ica_panels,
    open_subject_report,
    remove_tagged_content,
)
from eeg_pipeline.preprocessing.report.summary import (
    DecompositionSummary,
    compute_removal_topography,
    decomposition_measurements,
    decomposition_summary_html,
    exclusion_ledger_html,
    plot_component_overview,
    plot_removal_topography,
    plot_run_component_variance,
    plot_variance_overview,
    removal_topography_html,
    summarize_decomposition,
)
from eeg_pipeline.preprocessing.report.style import (
    DIVERGING_POWER_COLORMAP,
    FLAG_COLOR,
    OKABE_ITO,
    PRIMARY_COLOR,
    REPORT_IMAGE_FORMAT,
    REPORT_RASTER_IMAGE_FORMAT,
    apply_report_style,
    power_colorbar_label,
    report_image_format,
    robust_symmetric_limit,
)
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table


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
    #: Epochs contributing to each TFR frequency; ``None`` unless exclusions are active.
    tfr_eligible_counts: np.ndarray | None = None

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
    #: Whether the exploratory band decompositions get a report file of their own.
    #:
    #: They dominate the subject report while explicitly controlling nothing in it: on
    #: sub-0015 the five sliders were 30 MB of a 101 MB document, above a guide saying
    #: ICLabel was not validated for narrow-band decompositions and that artifact removal
    #: is decided on the standard ICA. Written beside the subject report and linked from
    #: it, the evidence is one click away for the reader who wants it and free for the
    #: many who do not. Set false for a single self-contained file.
    exploratory_separate_file: bool = True

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
            exploratory_separate_file=bool(
                values.get("exploratory_separate_file", cls.exploratory_separate_file)
            ),
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
    epoch_availability: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sources = ica.get_sources(epochs)
    diagnostics = _source_diagnostics_from_sources(
        sources=sources,
        band=band,
        settings=settings,
        epoch_availability=epoch_availability,
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
    epoch_availability: Any = None,
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
        tfr_frequencies, tfr_times, tfr, tfr_counts = _fieldtrip_tfr(
            data=sources.get_data(copy=False),
            sfreq=float(sources.info["sfreq"]),
            times=sources.times,
            band=band,
            settings=settings,
            epoch_availability=epoch_availability,
        )
    else:
        tfr_frequencies, tfr_times, tfr, tfr_counts = None, None, None, None
    return SourceDiagnostics(
        frequencies=frequencies,
        power_db=power_db,
        tfr_frequencies=tfr_frequencies,
        tfr_times=tfr_times,
        tfr=tfr,
        tfr_eligible_counts=tfr_counts,
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


def _subset_availability(epoch_availability: Any, mask: np.ndarray) -> Any:
    """Restrict epoch-aligned availability to a condition group's epochs."""
    if epoch_availability is None:
        return None

    from eeg_pipeline.spectral_availability import EpochSpectralAvailability

    selected = np.flatnonzero(np.asarray(mask, dtype=bool))
    return EpochSpectralAvailability(
        recording_keys=tuple(epoch_availability.recording_keys[index] for index in selected),
        exclusions_by_epoch=tuple(
            epoch_availability.exclusions_by_epoch[index] for index in selected
        ),
    )


def _recording_epoch_groups(epoch_availability: Any) -> tuple[tuple[Any, np.ndarray], ...]:
    """Group epoch indices by recording, since one recording has one geometry."""
    groups: dict[Any, list[int]] = {}
    for index, key in enumerate(epoch_availability.recording_keys):
        groups.setdefault(key, []).append(index)
    return tuple((key, np.asarray(indices, dtype=int)) for key, indices in groups.items())


def _multitaper_avg_power(
    data: np.ndarray,
    *,
    sfreq: float,
    frequencies: np.ndarray,
    parameters: TfrBandParameters,
    decim: int,
) -> np.ndarray:
    return mne.time_frequency.tfr_array_multitaper(
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


def _available_avg_power(
    data: np.ndarray,
    *,
    sfreq: float,
    frequencies: np.ndarray,
    parameters: TfrBandParameters,
    decim: int,
    epoch_availability: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Average each frequency over only the epochs whose recording still measured it.

    Averaging per recording and recombining by epoch count is exactly the overall
    mean restricted to eligible epochs, and it never materialises per-epoch power.
    ``time_bandwidth = 2 * window * smoothing`` with ``n_cycles = f * window`` makes
    the multitaper half-support exactly ``smoothing_hz`` at every frequency.
    """
    valid = epoch_availability.valid_frequency_mask(frequencies, parameters.smoothing_hz)

    numerator: np.ndarray | None = None
    weights = np.zeros(frequencies.size, dtype=float)
    for _key, indices in _recording_epoch_groups(epoch_availability):
        recording_valid = valid[indices[0]]
        if not np.any(recording_valid):
            continue
        power = _multitaper_avg_power(
            data[indices],
            sfreq=sfreq,
            frequencies=frequencies,
            parameters=parameters,
            decim=decim,
        )
        if numerator is None:
            numerator = np.zeros_like(power)
        count = float(indices.size)
        numerator[:, recording_valid, :] += count * power[:, recording_valid, :]
        weights[recording_valid] += count

    if numerator is None:
        reference = _multitaper_avg_power(
            data[:1],
            sfreq=sfreq,
            frequencies=frequencies,
            parameters=parameters,
            decim=decim,
        )
        numerator = np.zeros_like(reference)

    with np.errstate(invalid="ignore", divide="ignore"):
        averaged = numerator / weights[None, :, None]
    averaged[:, weights <= 0, :] = np.nan
    return averaged, weights


def _fieldtrip_tfr(
    *,
    data: np.ndarray,
    sfreq: float,
    times: np.ndarray,
    band: BandIcaDefinition,
    settings: BandIcaReportSettings,
    epoch_availability: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    decim = int(round(settings.time_step_s * sfreq))
    if decim < 1 or not np.isclose(decim / sfreq, settings.time_step_s):
        raise ValueError(
            f"TFR time step {settings.time_step_s:g} s is incompatible with {sfreq:g} Hz data."
        )

    if epoch_availability is not None:
        n_keys = len(epoch_availability.recording_keys)
        if n_keys != int(data.shape[0]):
            raise ValueError(
                f"spectral_availability covers {n_keys} epochs but the TFR input has "
                f"{int(data.shape[0])}."
            )

    frequency_parts = []
    power_parts = []
    weight_parts = []
    for parameters, frequencies in _tfr_parameter_groups(band, settings):
        if epoch_availability is None:
            power = _multitaper_avg_power(
                data,
                sfreq=sfreq,
                frequencies=frequencies,
                parameters=parameters,
                decim=decim,
            )
        else:
            power, weights = _available_avg_power(
                data,
                sfreq=sfreq,
                frequencies=frequencies,
                parameters=parameters,
                decim=decim,
                epoch_availability=epoch_availability,
            )
            weight_parts.append(weights)
        frequency_parts.append(frequencies)
        power_parts.append(power)

    frequencies = np.concatenate(frequency_parts)
    power = np.concatenate(power_parts, axis=1)
    eligible_counts = (
        np.concatenate(weight_parts).astype(int) if epoch_availability is not None else None
    )
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
    return frequencies, decimated_times[time_mask], power_db[..., time_mask], eligible_counts


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
    columns = (
        Column("Name", align=Align.TEXT, code=True),
        Column("Column", align=Align.TEXT, code=True),
        Column("Group A", align=Align.TEXT),
        Column("Group B", align=Align.TEXT),
        Column("Status", align=Align.TEXT),
    )
    rows = [
        [
            comparison.name,
            comparison.column,
            f"{comparison.group_a.label}: {list(comparison.group_a.values)}",
            f"{comparison.group_b.label}: {list(comparison.group_b.values)}",
            status,
        ]
        for comparison in settings.comparisons
    ]
    return (
        "<p>Configured comparisons are first computed from all pre-ICA task epochs for manual "
        "component review, then replaced after rejection using retained epochs and aligned "
        "events metadata.</p>"
        + grid_table(columns, rows)
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
        if panel_count == 3:
            figure, mosaic = plt.subplot_mosaic(
                [["topography", "spectrum"], ["tfr", "tfr"]],
                figsize=(11.0, 7.6),
                layout="constrained",
            )
            axes = [mosaic["topography"], mosaic["spectrum"], mosaic["tfr"]]
        else:
            figure, axis_array = plt.subplots(1, 2, figsize=(9.4, 4.2), layout="constrained")
            axes = list(axis_array)
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
            f"{label.label} ({label.probability:.3f})"
        )
        _mark_exploratory(figure)
        plt.close(figure)
        figures.append(figure)
    return figures


def _mark_exploratory(figure: plt.Figure) -> None:
    """Stamp a figure as belonging to a band-fitted, non-authoritative decomposition.

    These figures and the standard-ICA dossiers carry the same panels in the same
    layout, and their component numbers refer to different decompositions. Only the
    accordion heading told them apart, which put a reviewer one collapsed section away
    from acting on a band-fitted component index as though it were an authoritative one.
    The mark travels with the figure, including when it is exported on its own.
    """
    # Constrained layout does not reserve space for a bare figure text, so the axes are
    # pulled up off the bottom of the canvas first. Without that the stamp lands on the
    # time-axis label; moving it to a corner instead only trades that for the suptitle.
    engine = figure.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, 0.045, 1.0, 0.955))
    figure.text(
        0.5,
        0.008,
        "EXPLORATORY · band-fitted decomposition · component numbers do not match the standard ICA",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="white",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": FLAG_COLOR, "edgecolor": "none"},
    )


def _comparison_color_limits(
    group_a_tfr: np.ndarray,
    group_b_tfr: np.ndarray,
) -> tuple[float, float]:
    """Return robust symmetric limits for the two conditions and their difference."""
    condition_limit = robust_symmetric_limit(group_a_tfr, group_b_tfr)
    difference_limit = robust_symmetric_limit(group_a_tfr - group_b_tfr)
    return condition_limit, difference_limit


@dataclass(frozen=True)
class DossierColorLimits:
    """Colour limits for one dossier slide.

    Two scales, not one per panel. Every panel showing baseline-relative power — the
    grand average and each condition — shares :attr:`power`, because they are the same
    quantity and the layout puts them side by side for comparison. Scaling each to its
    own range makes a weak condition and a strong one look alike, which is the specific
    error this figure most invites.

    Differences keep their own limits: a difference of two dB-relative maps is centred on
    zero by construction and routinely spans a wider range than either map, so forcing it
    onto the power scale would flatten it to the neutral colour.
    """

    #: Shared limit for every absolute-power panel, or ``None`` without time-frequency
    #: data.
    power: float | None
    #: One limit per configured comparison, in the order the comparisons are rendered.
    differences: tuple[float, ...]


def dossier_color_limits(
    grand_average_tfr: np.ndarray | None,
    comparison_tfrs: Sequence[tuple[np.ndarray, np.ndarray]],
) -> DossierColorLimits:
    """Return the colour limits shared across one dossier slide.

    The power limit is pooled over the grand average and every condition so that one
    colour means one number everywhere on the slide.
    """
    if grand_average_tfr is None:
        return DossierColorLimits(power=None, differences=())
    pooled: list[np.ndarray] = [np.asarray(grand_average_tfr, dtype=float)]
    for group_a_tfr, group_b_tfr in comparison_tfrs:
        pooled.append(np.asarray(group_a_tfr, dtype=float))
        pooled.append(np.asarray(group_b_tfr, dtype=float))
    return DossierColorLimits(
        power=robust_symmetric_limit(*pooled),
        differences=tuple(
            robust_symmetric_limit(np.asarray(group_a_tfr) - np.asarray(group_b_tfr))
            for group_a_tfr, group_b_tfr in comparison_tfrs
        ),
    )


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
        # The log locator also labels its own minor ticks, which arrive formatted as
        # "3 × 10⁰" and sit between the plain "2" and "4" chosen above. Silencing them
        # leaves one labelling convention on the axis instead of two.
        axis.xaxis.set_minor_formatter(NullFormatter())
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
    # The colour scale is shared across the decomposition so that components can be
    # compared, which means a component whose modulation is small against the strongest
    # one renders almost entirely neutral. Stating this component's own peak turns that
    # near-blank map from "something is wrong with the figure" into the measurement it
    # is: the effect is this many decibels on a scale that reaches that many.
    peak = float(np.max(np.abs(power))) if np.isfinite(power).any() else 0.0
    axis.annotate(
        f"peak |{peak:.1f}| dB",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(3, -3),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=6,
        color="0.25",
        bbox={
            "boxstyle": "square,pad=0.15",
            "facecolor": "white",
            "alpha": 0.7,
            "edgecolor": "none",
        },
    )
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
    """Report the status the component table currently carries for one component."""
    return "MARKED BAD" if component in ica.exclude else "RETAINED"


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
    power_limit: float,
    difference_limit: float,
    draw_power_colorbar: bool,
) -> None:
    """Draw one comparison row: condition A, condition B, and their difference.

    ``draw_power_colorbar`` is false on every row whose power scale is already shown
    elsewhere on the slide. The scale is shared by construction — see
    :class:`DossierColorLimits` — so a dossier with three comparisons was drawing four
    identical power colourbars, one per row plus the grand average's. The difference
    colourbar stays on every row, because each comparison has its own difference limit.
    """
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
            (power_limit, power_limit, difference_limit),
            strict=True,
        )
    ]
    if draw_power_colorbar:
        figure.colorbar(
            images[0],
            ax=axes[:2],
            label=power_colorbar_label(power_limit),
        )
    figure.colorbar(
        images[2],
        ax=axes[2],
        label=power_colorbar_label(difference_limit, quantity="Group difference"),
    )


def _create_component_dossier(
    *,
    ica: mne.preprocessing.ICA,
    review: BandReviewData,
    label: ComponentLabel,
    component: int,
    color_limits: DossierColorLimits,
    settings: BandIcaReportSettings,
    analysis_status: str,
) -> plt.Figure:
    if review.comparisons:
        # Every comparison row is a triple — condition A, condition B, difference — so
        # the grid has to be three wide and the summary row runs across the top of it.
        row_count = 1 + len(review.comparisons)
        figure, axes = plt.subplots(
            row_count,
            3,
            figsize=(15.9, 3.8 * row_count),
            squeeze=False,
            layout="constrained",
        )
        summary_axes = axes[0]
        for row, (result, difference_limit) in enumerate(
            zip(review.comparisons, color_limits.differences, strict=True),
            start=1,
        ):
            _plot_dossier_comparison(
                figure=figure,
                axes=axes[row],
                result=result,
                review=review,
                component=component,
                power_limit=color_limits.power,
                difference_limit=difference_limit,
                # The summary row draws the shared power scale when it has a grand
                # average to draw it beside; the first comparison row picks it up when
                # there is none, so the slide always carries the scale exactly once.
                draw_power_colorbar=row == 1 and not review.diagnostics.has_tfr,
            )
    elif review.diagnostics.has_tfr:
        # Without comparison rows the dossier was a single row 15.9 inches wide and 3.8
        # tall. The browser fits that to the report column, which scales three panels
        # down to roughly a third of the column each and takes the 6 pt annotations with
        # them. Folding the same three panels into two rows trades width the reader does
        # not have for height they do, and gives the time axis the wider cell.
        figure, mosaic = plt.subplot_mosaic(
            [["topography", "spectrum"], ["tfr", "tfr"]],
            figsize=(11.0, 7.6),
            layout="constrained",
        )
        summary_axes = [mosaic["topography"], mosaic["spectrum"], mosaic["tfr"]]
    else:
        figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), squeeze=False, layout="constrained")
        summary_axes = axes[0]

    _plot_dossier_summary(
        figure=figure,
        axes=summary_axes,
        ica=ica,
        review=review,
        label=label,
        component=component,
        color_limit=color_limits.power,
        settings=settings,
    )
    # The title names the component and its status only. The taper, grid and baseline
    # are identical on all sixty-odd slides and are already stated once in the review
    # context above them, so repeating them here spent the second title line of every
    # slide on text no reader needs twice.
    figure.suptitle(
        f"{review.band.title} · ICA{component:03d} · "
        f"{label.label} ({label.probability:.3f}) · "
        f"{_component_review_status(ica, component)}\n"
        f"{analysis_status}"
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

    # Computed once for the whole set rather than per slide, so that one colour means one
    # number across every component as well as across the panels of any one of them.
    color_limits = dossier_color_limits(
        review.diagnostics.tfr if review.diagnostics.has_tfr else None,
        [(result.group_a_tfr, result.group_b_tfr) for result in review.comparisons],
    )

    return [
        _create_component_dossier(
            ica=ica,
            review=review,
            label=label,
            component=component,
            color_limits=color_limits,
            settings=settings,
            analysis_status=analysis_status,
        )
        for component, label in enumerate(labels)
    ]


def _organize_component_review(report: mne.Report) -> None:
    """Keep authoritative review sections together before MNE's ICA components.

    Decomposition evidence moves with the review even though it no longer carries the
    review tag: the tags separate *what a rebuild may clear*, not where content belongs.
    """
    content = report._content
    review_indices = [
        index
        for index, element in enumerate(content)
        if "ica-component-review" in element.tags or _DECOMPOSITION_TAG in element.tags
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
    epoch_availability: Any = None,
) -> tuple[ConditionTfrResult, ...]:
    results = []
    for comparison in settings.comparisons:
        group_a_mask, group_b_mask = _comparison_masks(metadata, comparison)
        _, _, group_a_tfr, _ = _fieldtrip_tfr(
            data=source_data[group_a_mask],
            sfreq=sfreq,
            times=times,
            band=band,
            settings=settings,
            epoch_availability=_subset_availability(epoch_availability, group_a_mask),
        )
        _, _, group_b_tfr, _ = _fieldtrip_tfr(
            data=source_data[group_b_mask],
            sfreq=sfreq,
            times=times,
            band=band,
            settings=settings,
            epoch_availability=_subset_availability(epoch_availability, group_b_mask),
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
    epoch_availability: Any = None,
) -> BandReviewData:
    band_epochs = _band_epochs(epochs, band)
    sources = ica.get_sources(band_epochs)
    diagnostics = _source_diagnostics_from_sources(
        sources=sources,
        band=band,
        settings=settings,
        epoch_availability=epoch_availability,
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
            epoch_availability=epoch_availability,
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


def _exploratory_component_captions(labels: Sequence[ComponentLabel]) -> list[str]:
    """Return one slider caption per component of an exploratory band-fitted ICA.

    Distinct from :func:`_component_captions`, which appends the exclusion status. An
    exploratory fit excludes nothing and its labels do not control artifact removal, so a
    status in the caption would assert a decision that was never taken. A component whose
    classifier did not run carries no label at all rather than ``unlabeled (0.000)``.
    """
    captions = []
    for component, label in enumerate(labels):
        caption = f"ICA{component:03d}"
        if getattr(label, "label", "") not in ("", "unlabeled"):
            caption += f" · {label.label} ({label.probability:.2f})"
        captions.append(caption)
    return captions


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
        f"<p>{html.escape(_tfr_configuration_title(band, settings))}. The grand average "
        "and every condition share one symmetric color scale, held fixed across all "
        "components, so a panel that looks stronger than its neighbour is stronger. Each "
        "difference uses its own symmetric zero-centred scale, because a difference of "
        "two baseline-relative maps routinely spans a wider range than either map.</p>"
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

#: Tag for evidence about the decomposition as a whole, rather than about a component.
#:
#: Deliberately *not* ``ica-component-review``. The condition-TFR stage reopens the
#: report, clears that tag, and rebuilds the per-component dossiers from clean epochs —
#: but it has no filtered run files and no Analyzer-marker fallback state, so rebuilding
#: the decomposition section from it produced a strictly poorer version: "Component
#: variance by run" and the fallback warning were cleared and never re-added. Keying the
#: two bodies of content to different tags makes that failure unrepresentable rather than
#: relying on every caller to thread the same inputs through.
_DECOMPOSITION_TAG = "ica-decomposition"
_DECOMPOSITION_TAGS = ("ica", _DECOMPOSITION_TAG)

#: Panel carrying the Analyzer-marker fallback warning, when one applies.
#:
#: Its own panel rather than a banner prepended to the review guide. The guide explains
#: how to read the dossiers and is rebuilt whenever those change; this is a statement
#: about the subject's data, and folding it into replaceable content is what let a
#: rebuild drop it. Separating them also puts it beside the other decomposition evidence
#: rather than inside a "how to" panel a reader may reasonably skip.
_FALLBACK_WARNING_TITLE = "Cardiac marker fallback"

_FALLBACK_WARNING_HTML = (
    '<div style="background-color: #fff3cd; color: #856404; padding: 15px; '
    'margin-bottom: 20px; border: 1px solid #ffeeba; border-radius: 4px;">'
    "<strong>&#9888; DATA QUALITY WARNING:</strong> "
    "Some runs in this subject lacked manual BrainVision R-peak markers "
    "(Analyzer defaulted to a 0.21s delay). "
    "Cardiac QC metrics (CTPS and attenuation) for this subject relied on "
    "automated MNE fallback detection and may be noisier than standard."
    "</div>"
)


def _should_rebuild_decomposition(
    report: mne.Report,
    *,
    filtered_raw_paths: Sequence[Path] | None,
) -> bool:
    """Whether this pass should rebuild the whole-decomposition evidence.

    Rebuild when there is something better to write — the filtered run files this pass
    carries — or when the report has no decomposition evidence yet. Otherwise leave what
    is already there: a pass without the run files can only produce a poorer version of
    panels that are already correct, which is exactly how the run-variance figure and the
    fallback warning disappeared from the delivered report.
    """
    if filtered_raw_paths:
        return True
    return not any(
        _DECOMPOSITION_TAG in element.tags for element in getattr(report, "_content", [])
    )


#: Section holding every exploratory band-fitted decomposition.
#:
#: One section rather than one per band. MNE builds the table of contents from section
#: names, so five bands here plus five authoritative ``ICA component review: <band>``
#: sections produced ten entries distinguishable only by their prefix — and the five that
#: matter for artifact removal were the ones a reader could not pick out. The band still
#: titles its own block inside the section, which is where it belongs: a band is a figure
#: within the exploratory analysis, not an analysis of its own.
EXPLORATORY_BAND_SECTION = "Exploratory band-fitted ICAs"


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
    status_descriptions: Sequence[str] | None = None,
    has_fallback_runs: bool = False,
) -> DecompositionSummary:
    """Add whole-decomposition evidence ahead of the per-component review.

    ``status_descriptions`` is the ``status_description`` column of the component table
    that decided the exclusions. Optional because a rebuild triggered by a condition-TFR
    pass reaches this function without the paths to read it; the ledger is then left as
    the previous pass wrote it rather than replaced by one that cannot name a detector.
    """
    remove_tagged_content(report, tag=_DECOMPOSITION_TAG)
    if has_fallback_runs:
        report.add_html(
            html=_FALLBACK_WARNING_HTML,
            title=_FALLBACK_WARNING_TITLE,
            section=_DECOMPOSITION_SECTION,
            tags=(*_DECOMPOSITION_TAGS, "ica-fallback-warning"),
            replace=True,
        )
    summary = summarize_decomposition(ica=ica, epochs=epochs)
    report.add_figure(
        fig=plot_component_overview(ica=ica, labels=labels),
        title="All component topographies",
        section=_DECOMPOSITION_SECTION,
        tags=_DECOMPOSITION_TAGS,
        image_format=REPORT_RASTER_IMAGE_FORMAT,
        replace=True,
    )
    report.add_html(
        html=decomposition_summary_html(summary),
        title="Decomposition summary",
        section=_DECOMPOSITION_SECTION,
        tags=_DECOMPOSITION_TAGS,
        replace=True,
    )
    report.add_figure(
        fig=plot_variance_overview(summary),
        title="Sensor variance per component",
        section=_DECOMPOSITION_SECTION,
        tags=_DECOMPOSITION_TAGS,
        image_format=REPORT_IMAGE_FORMAT,
        replace=True,
    )
    if status_descriptions is not None:
        report.add_html(
            html=exclusion_ledger_html(summary, status_descriptions=status_descriptions),
            title="Why each component was excluded",
            section=_DECOMPOSITION_SECTION,
            tags=_DECOMPOSITION_TAGS,
            replace=True,
        )
    # The variance figure says how much was removed; this says from where, which is what
    # separates focal artifact removal from uniform signal loss.
    topography = compute_removal_topography(ica=ica, epochs=epochs)
    report.add_html(
        html=removal_topography_html(topography),
        title="Spatial signature of the removal",
        section=_DECOMPOSITION_SECTION,
        tags=_DECOMPOSITION_TAGS,
        replace=True,
    )
    report.add_figure(
        fig=plot_removal_topography(topography),
        title="Per-channel amplitude removed by ICA",
        section=_DECOMPOSITION_SECTION,
        tags=_DECOMPOSITION_TAGS,
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
            tags=_DECOMPOSITION_TAGS,
            image_format=REPORT_IMAGE_FORMAT,
            replace=True,
        )
    return summary


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
    status_descriptions: Sequence[str] | None = None,
    spectral_availability: Any = None,
) -> DecompositionSummary | None:
    """Add authoritative, component-centred evidence before MNE's ICA section.

    Returns the decomposition it measured so the caller can record those numbers
    beside the report, or ``None`` when the rebuild was skipped and the numbers on
    the page are the previous pass's rather than this one's.
    """
    _remove_legacy_condition_tfr_entries(report)
    # Every panel this function writes carries "ica-component-review". Clearing the tag
    # first makes the rebuild idempotent: renaming a panel cannot strand the previous
    # one, because removal is keyed to the tag rather than to the title.
    remove_tagged_content(report, tag="ica-component-review")
    # Guarded rather than unconditional: a condition-TFR rebuild carries neither the run
    # files nor the fallback state, so rebuilding here would replace correct panels with
    # poorer ones. See :func:`_should_rebuild_decomposition`.
    summary = None
    if _should_rebuild_decomposition(report, filtered_raw_paths=filtered_raw_paths):
        summary = _add_decomposition_summary(
            report=report,
            ica=ica,
            epochs=epochs,
            labels=labels,
            filtered_raw_paths=filtered_raw_paths,
            status_descriptions=status_descriptions,
            has_fallback_runs=has_fallback_runs,
        )
    guide_title = "How to review ICA component dossiers"
    report.remove(title=guide_title, remove_all=True)
    report.add_html(
        html=_review_guide_html(settings, analysis_status),
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
            epoch_availability=spectral_availability,
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
    # After the replacements exist, not before: the drop is guarded on their presence, so
    # calling it at the top of this function would find nothing to stand in for MNE's
    # panels and leave them. ``open_subject_report`` re-applies it on every later reopen,
    # which is what catches the copies MNE-BIDS-Pipeline writes again after this stage.
    drop_superseded_mne_ica_panels(report)
    _organize_component_review(report)
    return summary


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
    spectral_availability: Any = None,
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

    standard_ica = read_ica_with_reviewed_exclusions(standard_ica_path)
    labels = _label_components(epochs=ica_fit_epochs, ica=standard_ica)
    report = open_subject_report(report_path)
    summary = _add_standard_component_review(
        spectral_availability=spectral_availability,
        report=report,
        ica=standard_ica,
        epochs=retained_epochs,
        metadata=clean_events,
        labels=labels,
        settings=settings,
        analysis_status=analysis_status,
    )
    save_subject_report(
        report,
        report_path,
        stage="ica-condition-tfr",
        measurements=decomposition_measurements(summary) if summary else None,
    )


def generate_band_ica_report(
    *,
    epochs_path: Path,
    report_path: Path,
    output_dir: Path,
    output_prefix: str,
    random_state: int,
    settings: BandIcaReportSettings,
    filtered_raw_paths: Sequence[Path] | None = None,
    spectral_availability: Any = None,
) -> list[Path]:
    """Fit exploratory band-specific ICAs and append diagnostics to an MNE report.

    ``spectral_availability`` is the epoch-aligned unavailable-frequency contract. When
    it is ``None`` every panel is computed exactly as before.
    """
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
    report = open_subject_report(report_path)
    standard_ica_path = epochs_path.with_name(f"{output_prefix}_proc-ica_ica.fif")
    if not standard_ica_path.is_file():
        raise FileNotFoundError(f"Standard ICA does not exist: {standard_ica_path}")
    standard_ica = read_ica_with_reviewed_exclusions(standard_ica_path)
    standard_labels = _label_components(epochs=epochs, ica=standard_ica)

    components_path = epochs_path.with_name(f"{output_prefix}_proc-ica_components.tsv")
    has_fallback_runs = False
    status_descriptions = None
    if components_path.is_file():
        components = pd.read_csv(components_path, sep="\t")
        if "analyzer_marker_ctps_fallback" in components.columns:
            has_fallback_runs = bool(components["analyzer_marker_ctps_fallback"].any())
        # Read from the table that decides the exclusions rather than from the labels, so
        # the panel naming the detector cannot disagree with the data the pipeline built.
        if "status_description" in components.columns:
            status_descriptions = tuple(components["status_description"].fillna("").astype(str))

    summary = _add_standard_component_review(
        spectral_availability=spectral_availability,
        report=report,
        ica=standard_ica,
        epochs=epochs,
        metadata=None,
        labels=standard_labels,
        settings=settings,
        analysis_status="Pending provisional task epochs",
        has_fallback_runs=has_fallback_runs,
        filtered_raw_paths=filtered_raw_paths,
        status_descriptions=status_descriptions,
    )
    # Previous runs wrote one section per band. Those section names no longer exist, so
    # without clearing by tag an incrementally updated report keeps both layouts at once.
    remove_tagged_content(report, tag="band-specific-ica")
    exploratory_path = output_dir / f"{output_prefix}_desc-exploratorybandica_report.html"
    # Where the exploratory figures are added. Either the subject report itself, or a
    # report of its own that is saved beside it and linked from the subject report.
    exploratory_report = report
    if settings.exploratory_separate_file:
        exploratory_report = mne.Report(
            title=f"{output_prefix} · exploratory band-fitted ICAs",
            verbose="ERROR",
        )
    report.add_html(
        title="How to read the exploratory band ICAs",
        html=(
            "<p><strong>Exploratory only.</strong> Each decomposition "
            f"{'in the linked file' if settings.exploratory_separate_file else 'below'} was "
            "fitted to one band of the same epochs. ICLabel was not validated for "
            "narrow-band decompositions, component numbers do not correspond across bands "
            "or to the standard ICA, and nothing here controls artifact removal. That is "
            "decided on the standard broadband ICA, in the authoritative component review "
            "sections.</p>"
            + (
                "<p>Because it decides nothing and is large, this evidence is written "
                "beside the subject report rather than inside it: "
                f'<a href="{html.escape(exploratory_path.name)}">'
                f"{html.escape(exploratory_path.name)}</a>. The two files live in the same "
                "directory, so keeping them together keeps the link working. Set "
                "<code>ica.band_specific_report.exploratory_separate_file</code> to false "
                "for a single self-contained report.</p>"
                if settings.exploratory_separate_file
                else ""
            )
        ),
        section=EXPLORATORY_BAND_SECTION,
        tags=("ica", "band-specific-ica"),
        replace=True,
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

        exploratory_report.add_figure(
            fig=figures,
            title=_slider_title(
                f"{band.title}: component topomaps, spectra, and grand-average "
                f"relative TFRs · {_tfr_configuration_title(band, settings)}",
                len(figures),
            ),
            caption=_exploratory_component_captions(labels),
            section=EXPLORATORY_BAND_SECTION,
            tags=("ica", "band-specific-ica", band.slug),
            image_format=report_image_format(
                has_dense_image=settings.tfr_enabled, is_figure_list=True
            ),
            replace=True,
        )
    if settings.exploratory_separate_file:
        # Saved as HTML alone. The .h5 archive exists so a later stage can reopen and
        # append to a report, and nothing appends to this one: it is rebuilt from scratch
        # whenever the band ICAs are refitted.
        exploratory_report.save(exploratory_path, overwrite=True, open_browser=False)
        generated_paths.append(exploratory_path)
    save_subject_report(
        report,
        report_path,
        stage="band-ica-report",
        measurements=decomposition_measurements(summary) if summary else None,
    )
    return generated_paths
