"""Overview plots for ICA-component time-frequency analysis."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from eeg_pipeline.analysis.component_tfr import ConditionTFR

#: Decibels per unit log10 power ratio. Power in dB is 10 * log10(power / reference).
_DECIBELS_PER_LOG10 = 10.0


def save_ica_topography_overview(
    ica: mne.preprocessing.ICA,
    epochs: mne.Epochs,
    output_dir: Path,
    file_prefix: str,
) -> list[Path]:
    """Save paginated topographies containing every fitted component."""
    figures = ica.plot_components(
        inst=epochs,
        show_names=False,
        colorbar=True,
        title="6-14 Hz analysis ICA topographies",
        show=False,
    )
    figure_list = figures if isinstance(figures, list) else [figures]
    if not figure_list:
        raise RuntimeError("MNE did not return an ICA topography figure.")

    pdf_path = output_dir / f"{file_prefix}_desc-ica-topographies.pdf"
    image_paths = []
    with PdfPages(pdf_path) as pdf:
        for page_number, figure in enumerate(figure_list, start=1):
            figure.suptitle(
                f"6-14 Hz analysis ICA topographies — page {page_number}",
                y=0.995,
            )
            image_path = output_dir / (
                f"{file_prefix}_desc-ica-topographies_page-{page_number:02d}.png"
            )
            figure.savefig(image_path, dpi=180, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            image_paths.append(image_path)
            plt.close(figure)
    return [pdf_path, *image_paths]


def save_component_tfr_overview(
    condition_tfrs: list[ConditionTFR],
    condition_column: str,
    output_dir: Path,
    file_prefix: str,
    components_per_page: int,
) -> list[Path]:
    """Save baseline-normalized TFR pages covering every component and condition."""
    if not condition_tfrs:
        raise ValueError("At least one condition TFR is required for plotting.")

    component_names = condition_tfrs[0].baseline_power.ch_names
    _validate_component_axes(condition_tfrs, component_names)
    color_limit = _shared_color_limit(condition_tfrs)
    pdf_path = output_dir / f"{file_prefix}_desc-component-tfr-overview.pdf"
    image_paths = []

    with PdfPages(pdf_path) as pdf:
        for condition in condition_tfrs:
            pages = math.ceil(len(component_names) / components_per_page)
            for page_index in range(pages):
                figure = _plot_tfr_page(
                    condition=condition,
                    condition_column=condition_column,
                    component_names=component_names,
                    page_index=page_index,
                    page_count=pages,
                    components_per_page=components_per_page,
                    color_limit=color_limit,
                )
                image_path = output_dir / (
                    f"{file_prefix}_cond-{condition.file_label}_desc-component-tfr_"
                    f"page-{page_index + 1:02d}.png"
                )
                figure.savefig(image_path, dpi=180, bbox_inches="tight")
                pdf.savefig(figure, bbox_inches="tight")
                image_paths.append(image_path)
                plt.close(figure)

    return [pdf_path, *image_paths]


def _plot_tfr_page(
    *,
    condition: ConditionTFR,
    condition_column: str,
    component_names: list[str],
    page_index: int,
    page_count: int,
    components_per_page: int,
    color_limit: float,
) -> plt.Figure:
    first_component = page_index * components_per_page
    last_component = min(first_component + components_per_page, len(component_names))
    page_components = list(range(first_component, last_component))
    columns = min(3, len(page_components))
    rows = math.ceil(len(page_components) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.0 * columns, 3.6 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    # MNE's "logratio" baseline is log10(power/baseline). The rest of the report states
    # baseline-relative power in decibels, so convert here rather than presenting the
    # same quantity in two units a factor of ten apart across adjacent sections.
    data = _DECIBELS_PER_LOG10 * condition.baseline_power.get_data()
    frequencies = condition.baseline_power.freqs
    times = condition.baseline_power.times
    image = None
    for axis, component_index in zip(axes.flat, page_components):
        image = axis.pcolormesh(
            times,
            frequencies,
            data[component_index],
            shading="auto",
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
            rasterized=True,
        )
        axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(component_names[component_index])
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Frequency (Hz)")
        axis.set_ylim(float(frequencies[0]), float(frequencies[-1]))

    for axis in axes.flat[len(page_components) :]:
        axis.remove()

    if image is None:
        raise RuntimeError("No ICA components were available for TFR plotting.")
    figure.colorbar(
        image,
        ax=[axis for axis in axes.flat if axis in figure.axes],
        label="Baseline-relative power (dB)",
        shrink=0.82,
    )
    figure.suptitle(
        f"{condition_column} = {condition.label} (n={condition.epoch_count}) — "
        f"components {first_component + 1}-{last_component} of {len(component_names)} "
        f"— page {page_index + 1}/{page_count}",
        fontsize=13,
    )
    return figure


def _validate_component_axes(
    condition_tfrs: list[ConditionTFR],
    component_names: list[str],
) -> None:
    if not component_names:
        raise ValueError("Condition TFRs contain no ICA components.")
    reference = condition_tfrs[0].baseline_power
    for condition in condition_tfrs[1:]:
        current = condition.baseline_power
        if current.ch_names != component_names:
            raise ValueError("Condition TFRs do not contain identical ICA component axes.")
        if not np.array_equal(current.freqs, reference.freqs):
            raise ValueError("Condition TFRs do not contain identical frequency axes.")
        if not np.array_equal(current.times, reference.times):
            raise ValueError("Condition TFRs do not contain identical time axes.")


def _shared_color_limit(condition_tfrs: list[ConditionTFR]) -> float:
    finite_values = []
    for condition in condition_tfrs:
        data = _DECIBELS_PER_LOG10 * condition.baseline_power.get_data()
        finite_values.append(np.abs(data[np.isfinite(data)]))
    combined = np.concatenate(finite_values)
    if combined.size == 0:
        raise ValueError("Component TFRs contain no finite baseline-normalized power values.")
    color_limit = float(np.percentile(combined, 98.0))
    if color_limit <= 0:
        raise ValueError("Component TFRs contain no non-zero baseline-normalized power values.")
    return color_limit
