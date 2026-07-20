"""Exploratory band-specific ICA diagnostics for the MNE HTML report."""

from __future__ import annotations

import csv
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BandIcaDefinition:
    """One fixed frequency range in the exploratory ICA report."""

    slug: str
    title: str
    fmin: float
    fmax: float


@dataclass(frozen=True)
class ComponentLabel:
    """Exploratory ICLabel result for one band-specific component."""

    label: str
    probability: float


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
class BandIcaReportSettings:
    """Runtime controls for the computationally expensive report."""

    fit_decim: int = 2
    frequency_step_hz: float = 1.0
    time_min_s: float = -5.0
    time_max_s: float = 14.4
    time_step_s: float = 0.1
    baseline_tmin_s: float = -5.0
    baseline_tmax_s: float = -0.01
    comparisons: tuple[ConditionComparison, ...] = ()

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
        return settings


BAND_ICA_DEFINITIONS = (
    BandIcaDefinition("deltatheta", "Delta + theta (1–8 Hz)", 1.0, 8.0),
    BandIcaDefinition("alpha", "Alpha (8–13 Hz)", 8.0, 13.0),
    BandIcaDefinition("beta", "Beta (13–30 Hz)", 13.0, 30.0),
    BandIcaDefinition("gamma", "Gamma (30–100 Hz)", 30.0, 100.0),
    BandIcaDefinition("broadband1to30", "Broadband 1–30 Hz", 1.0, 30.0),
    BandIcaDefinition("broadband30to100", "Broadband 30–100 Hz", 30.0, 100.0),
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


def _label_band_components(
    *,
    epochs: mne.BaseEpochs,
    ica: mne.preprocessing.ICA,
) -> list[ComponentLabel]:
    from mne_icalabel.iclabel import iclabel_label_components

    probabilities = iclabel_label_components(inst=epochs, ica=ica, inplace=False)
    if probabilities.shape != (int(ica.n_components_), len(_ICLABEL_CLASSES)):
        raise ValueError(
            "ICLabel probability matrix does not match the band-specific ICA components."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("ICLabel returned non-finite band-specific component probabilities.")
    return [
        ComponentLabel(
            label=_ICLABEL_CLASSES[int(np.argmax(component_probabilities))],
            probability=float(np.max(component_probabilities)),
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
    spectrum = sources.compute_psd(
        method="welch",
        fmin=band.fmin,
        fmax=band.fmax,
        picks="all",
        verbose="ERROR",
    )
    frequencies, power = _component_spectrum(
        frequencies=spectrum.freqs,
        spectrum=spectrum.get_data().mean(axis=0),
        fmin=band.fmin,
        fmax=band.fmax,
    )
    power_db = 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny))

    tfr_frequencies, tfr_times, tfr = _fieldtrip_tfr(
        data=sources.get_data(copy=False),
        sfreq=float(sources.info["sfreq"]),
        times=sources.times,
        band=band,
        settings=settings,
    )
    return frequencies, power_db, tfr_frequencies, tfr_times, tfr


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
    band_frequencies = np.arange(
        band.fmin,
        band.fmax + settings.frequency_step_hz / 2.0,
        settings.frequency_step_hz,
    )
    narrow_band_parameters = {
        "deltatheta": _TFR_PARAMETERS[0],
        "alpha": _TFR_PARAMETERS[1],
        "beta": _TFR_PARAMETERS[2],
        "gamma": _TFR_PARAMETERS[3],
    }
    if band.slug in narrow_band_parameters:
        parameter_groups = ((narrow_band_parameters[band.slug], band_frequencies),)
    else:
        groups = []
        for index, parameters in enumerate(_TFR_PARAMETERS):
            upper_inclusive = index == len(_TFR_PARAMETERS) - 1
            frequency_mask = band_frequencies >= parameters.fmin
            if upper_inclusive:
                frequency_mask &= band_frequencies <= parameters.fmax
            else:
                frequency_mask &= band_frequencies < parameters.fmax
            groups.append((parameters, band_frequencies[frequency_mask]))
        parameter_groups = tuple(groups)
    for parameters, frequencies in parameter_groups:
        if not len(frequencies):
            continue
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
    narrow_band_indices = {"deltatheta": (0,), "alpha": (1,), "beta": (2,), "gamma": (3,)}
    indices = narrow_band_indices.get(
        band.slug,
        (0, 1, 2) if band.slug == "broadband1to30" else (3,),
    )
    parameter_text = "; ".join(
        f"{parameters.window_seconds:g} s, ±{parameters.smoothing_hz:g} Hz"
        for parameters in (_TFR_PARAMETERS[index] for index in indices)
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


def _add_comparison_configuration(
    report: mne.Report,
    settings: BandIcaReportSettings,
    *,
    status: str = "Pending provisional task epochs",
) -> None:
    title = "Condition comparison configuration"
    report.remove(title=title, remove_all=True)
    report.add_html(
        title=title,
        html=_comparison_configuration_html(settings, status=status),
        section="TFR comparisons",
        tags=("ica", "band-specific-ica", "condition-tfr-configuration"),
        replace=True,
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
    color_limit = float(np.nanmax(np.abs(tfr)))
    if not np.isfinite(color_limit) or color_limit <= 0:
        raise ValueError("Band-specific TFR has no finite non-zero values.")
    for component, label in enumerate(labels):
        figure, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        ica.plot_components(
            picks=component,
            axes=[axes[0]],
            colorbar=False,
            show=False,
        )
        axes[0].set_title(f"ICA{component:03d} topomap")

        axes[1].plot(frequencies, power_db[component], color="#276B8A", linewidth=1.5)
        axes[1].set(
            title="Source spectrum",
            xlabel="Frequency (Hz)",
            ylabel="Power (dB)",
            xlim=(band.fmin, band.fmax),
        )

        image = axes[2].pcolormesh(
            tfr_times,
            tfr_frequencies,
            tfr[component],
            shading="auto",
            cmap="turbo",
            vmin=-color_limit,
            vmax=color_limit,
        )
        axes[2].axvline(0.0, color="black", linestyle="--", linewidth=0.75)
        axes[2].set(
            title="Source time-frequency power",
            xlabel="Time (s)",
            ylabel="Frequency (Hz)",
        )
        figure.colorbar(image, ax=axes[2], label="Baseline-relative power (dB)")
        figure.suptitle(
            f"{band.title} · ICA{component:03d} · grand average · exploratory ICLabel: "
            f"{label.label} ({label.probability:.3f})\n"
            f"{_tfr_configuration_title(band, settings)}"
        )
        plt.close(figure)
        figures.append(figure)
    return figures


def _write_component_table(
    *,
    path: Path,
    labels: Sequence[ComponentLabel],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("component", "iclabel", "probability", "interpretation"),
            delimiter="\t",
        )
        writer.writeheader()
        for component, label in enumerate(labels):
            writer.writerow(
                {
                    "component": component,
                    "iclabel": label.label,
                    "probability": f"{label.probability:.6f}",
                    "interpretation": "exploratory",
                }
            )


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


def _build_comparison_figures(
    *,
    group_a_tfr: np.ndarray,
    group_b_tfr: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    comparison: ConditionComparison,
    band: BandIcaDefinition,
    settings: BandIcaReportSettings,
    group_a_count: int,
    group_b_count: int,
    analysis_status: str,
) -> list[plt.Figure]:
    contrast = group_a_tfr - group_b_tfr
    color_limits = tuple(
        float(np.nanmax(np.abs(power))) for power in (group_a_tfr, group_b_tfr, contrast)
    )
    if any(not np.isfinite(limit) or limit <= 0 for limit in color_limits):
        raise ValueError(f"Comparison {comparison.name!r} has no finite non-zero TFR values.")

    figures = []
    titles = (
        f"{comparison.group_a.label} · {comparison.column} ∈ "
        f"{list(comparison.group_a.values)} · n={group_a_count}",
        f"{comparison.group_b.label} · {comparison.column} ∈ "
        f"{list(comparison.group_b.values)} · n={group_b_count}",
        f"{comparison.group_a.label} − {comparison.group_b.label} · relative dB difference",
    )
    for component in range(group_a_tfr.shape[0]):
        figure, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
        for axis, title, power, color_limit in zip(
            axes,
            titles,
            (group_a_tfr[component], group_b_tfr[component], contrast[component]),
            color_limits,
        ):
            image = axis.pcolormesh(
                times,
                frequencies,
                power,
                shading="auto",
                cmap="turbo",
                vmin=-color_limit,
                vmax=color_limit,
            )
            axis.axvline(0.0, color="black", linestyle="--", linewidth=0.75)
            axis.set(title=title, xlabel="Time (s)", ylabel="Frequency (Hz)")
            figure.colorbar(image, ax=axis, label="Baseline-relative power (dB)")
        figure.suptitle(
            f"{band.title} · ICA{component:03d} · {comparison.name} · {analysis_status}\n"
            f"{_tfr_configuration_title(band, settings)} · each result scaled to ±max|dB|"
        )
        plt.close(figure)
        figures.append(figure)
    return figures


def append_condition_tfr_report(
    *,
    pre_ica_epochs_path: Path,
    clean_epochs_path: Path,
    clean_events_path: Path,
    report_path: Path,
    output_dir: Path,
    output_prefix: str,
    settings: BandIcaReportSettings,
    analysis_status: str,
) -> None:
    """Append FieldTrip-style clean-trial condition comparisons to a report."""
    if not settings.comparisons:
        return
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
    if np.max(clean_epochs.selection) >= len(pre_ica_epochs):
        raise ValueError("Clean epoch selection exceeds the saved pre-ICA task epochs.")
    retained_epochs = pre_ica_epochs[clean_epochs.selection]
    if len(retained_epochs) != len(clean_events):
        raise ValueError("Retained pre-ICA epochs do not align with clean events.")

    report = mne.open_report(report_path)
    for band in BAND_ICA_DEFINITIONS:
        ica_path = output_dir / f"{output_prefix}_desc-{band.slug}_ica.fif"
        if not ica_path.is_file():
            raise FileNotFoundError(f"Band-specific ICA does not exist: {ica_path}")
        ica = mne.preprocessing.read_ica(ica_path, verbose="ERROR")
        sources = ica.get_sources(_band_epochs(retained_epochs, band))
        source_data = sources.get_data(copy=False)
        for comparison in settings.comparisons:
            group_a_mask, group_b_mask = _comparison_masks(clean_events, comparison)
            frequencies, times, group_a_tfr = _fieldtrip_tfr(
                data=source_data[group_a_mask],
                sfreq=float(sources.info["sfreq"]),
                times=sources.times,
                band=band,
                settings=settings,
            )
            _, _, group_b_tfr = _fieldtrip_tfr(
                data=source_data[group_b_mask],
                sfreq=float(sources.info["sfreq"]),
                times=sources.times,
                band=band,
                settings=settings,
            )
            figures = _build_comparison_figures(
                group_a_tfr=group_a_tfr,
                group_b_tfr=group_b_tfr,
                frequencies=frequencies,
                times=times,
                comparison=comparison,
                band=band,
                settings=settings,
                group_a_count=int(group_a_mask.sum()),
                group_b_count=int(group_b_mask.sum()),
                analysis_status=analysis_status,
            )
            section = f"Band-specific ICA comparison: {band.title} · {comparison.name}"
            report.add_figure(
                fig=figures,
                title=(
                    f"{comparison.name} · {analysis_status} · column {comparison.column}: "
                    f"{comparison.group_a.label} {list(comparison.group_a.values)}, "
                    f"{comparison.group_b.label} {list(comparison.group_b.values)}, and relative "
                    f"dB difference · {_tfr_configuration_title(band, settings)}"
                ),
                section=section,
                tags=("ica", "band-specific-ica", "condition-tfr", band.slug),
                replace=True,
            )
            report.save(report_path, overwrite=True, open_browser=False)
            report.save(report_path.with_suffix(".html"), overwrite=True, open_browser=False)
    _add_comparison_configuration(report, settings, status=analysis_status)
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
) -> list[Path]:
    """Fit exploratory band-specific ICAs and append diagnostics to an MNE report."""
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
    _add_comparison_configuration(report, settings)
    generated_paths = []
    for band in BAND_ICA_DEFINITIONS:
        filtered_epochs = _band_epochs(epochs, band)
        ica = _fit_band_ica(
            epochs=filtered_epochs,
            random_state=random_state,
            fit_decim=settings.fit_decim,
        )
        ica.exclude = []
        labels = _label_band_components(epochs=filtered_epochs, ica=ica)
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
                "for narrow-band decompositions, and these labels do not control "
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
            replace=True,
        )
        report.save(report_path, overwrite=True, open_browser=False)
        report.save(report_path.with_suffix(".html"), overwrite=True, open_browser=False)
    return generated_paths
