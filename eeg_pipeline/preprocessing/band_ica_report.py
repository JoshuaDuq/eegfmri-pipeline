"""Exploratory band-specific ICA diagnostics for the MNE HTML report."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np


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
class BandIcaReportSettings:
    """Runtime controls for the computationally expensive report."""

    fit_decim: int = 2
    tfr_frequency_count: int = 24
    tfr_decim: int = 5

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> BandIcaReportSettings:
        settings = cls(
            fit_decim=int(values.get("fit_decim", cls.fit_decim)),
            tfr_frequency_count=int(values.get("tfr_frequency_count", cls.tfr_frequency_count)),
            tfr_decim=int(values.get("tfr_decim", cls.tfr_decim)),
        )
        if settings.fit_decim < 1:
            raise ValueError("ica.band_specific_report.fit_decim must be at least 1.")
        if settings.tfr_frequency_count < 2:
            raise ValueError("ica.band_specific_report.tfr_frequency_count must be at least 2.")
        if settings.tfr_decim < 1:
            raise ValueError("ica.band_specific_report.tfr_decim must be at least 1.")
        return settings


BAND_ICA_DEFINITIONS = (
    BandIcaDefinition("deltatheta", "Delta + theta (1–8 Hz)", 1.0, 8.0),
    BandIcaDefinition("alpha", "Alpha (8–13 Hz)", 8.0, 13.0),
    BandIcaDefinition("beta", "Beta (13–30 Hz)", 13.0, 30.0),
    BandIcaDefinition("gamma", "Gamma (30–100 Hz)", 30.0, 100.0),
    BandIcaDefinition("broadband1to30", "Broadband 1–30 Hz", 1.0, 30.0),
    BandIcaDefinition("broadband30to100", "Broadband 30–100 Hz", 30.0, 100.0),
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

    tfr_frequencies = np.linspace(
        band.fmin,
        band.fmax,
        settings.tfr_frequency_count,
    )
    n_cycles = np.clip(tfr_frequencies / 2.0, 2.0, 10.0)
    tfr = mne.time_frequency.tfr_array_morlet(
        sources.get_data(copy=False),
        sfreq=float(sources.info["sfreq"]),
        freqs=tfr_frequencies,
        n_cycles=n_cycles,
        output="avg_power",
        decim=settings.tfr_decim,
        n_jobs=1,
        verbose="ERROR",
    )
    tfr_times = sources.times[:: settings.tfr_decim][: tfr.shape[-1]]
    return frequencies, power_db, tfr_frequencies, tfr_times, tfr


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
            10.0 * np.log10(np.maximum(tfr[component], np.finfo(float).tiny)),
            shading="auto",
            cmap="magma",
        )
        axes[2].set(
            title="Source time-frequency power",
            xlabel="Time (s)",
            ylabel="Frequency (Hz)",
        )
        figure.colorbar(image, ax=axes[2], label="Power (dB)")
        figure.suptitle(
            f"{band.title} · ICA{component:03d} · exploratory ICLabel: "
            f"{label.label} ({label.probability:.3f})"
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
            title=f"{band.title}: component topomaps, spectra, and TFRs",
            section=section,
            tags=("ica", "band-specific-ica", band.slug),
            replace=True,
        )
        report.save(report_path, overwrite=True, open_browser=False)
        report.save(report_path.with_suffix(".html"), overwrite=True, open_browser=False)
    return generated_paths
