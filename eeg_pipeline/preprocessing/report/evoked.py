"""Condition-wise evoked waveforms and scalp maps for task QC."""

from __future__ import annotations

import html
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mne
import numpy as np

from eeg_pipeline.preprocessing.report.organize import remove_tagged_content
from eeg_pipeline.preprocessing.report.style import report_image_format


@dataclass(frozen=True)
class ConditionEvoked:
    name: str
    event_code: int
    n_trials: int
    evoked: mne.Evoked


def condition_evokeds(epochs: mne.BaseEpochs) -> tuple[ConditionEvoked, ...]:
    """Average each exact event code without MNE's hierarchical partial matching."""
    names_by_code: dict[int, list[str]] = {}
    for name, code in epochs.event_id.items():
        names_by_code.setdefault(int(code), []).append(str(name))

    results = []
    for code in np.unique(epochs.events[:, 2]):
        names = names_by_code.get(int(code), [])
        if len(names) != 1:
            raise ValueError(
                f"Event code {int(code)} must have exactly one condition name, got {names}."
            )
        indices = np.flatnonzero(epochs.events[:, 2] == code)
        results.append(
            ConditionEvoked(
                name=names[0],
                event_code=int(code),
                n_trials=int(indices.size),
                evoked=epochs[indices].average(picks="eeg"),
            )
        )
    return tuple(results)


def _topography_times(
    epochs: mne.BaseEpochs,
    response_window_s: tuple[float, float],
    topomap_count: int,
) -> np.ndarray:
    if topomap_count < 1:
        raise ValueError("Evoked topomap count must be positive.")
    low = max(float(response_window_s[0]), float(epochs.times[0]))
    high = min(float(response_window_s[1]), float(epochs.times[-1]))
    if high <= low:
        raise ValueError("report.analysis.response_window_s does not overlap the retained epochs.")
    if topomap_count == 1:
        return np.asarray([(low + high) / 2.0])
    return np.linspace(low, high, topomap_count)


def evoked_html(
    *,
    response_window_s: tuple[float, float],
    analysis_status: str,
) -> str:
    return (
        f"<p><strong>{html.escape(analysis_status)}</strong>. Topographies are sampled "
        f"uniformly from {response_window_s[0]:g} to {response_window_s[1]:g} s, the "
        "configured response interval rather than peaks selected from these data.</p>"
    )


def add_evoked_response_review(
    *,
    report: mne.Report,
    epochs: mne.BaseEpochs,
    response_window_s: tuple[float, float],
    topomap_count: int,
    analysis_status: str,
    section: str = "Evoked responses",
) -> tuple[ConditionEvoked, ...]:
    """Append one MNE joint evoked plot per exact task condition."""
    conditions = condition_evokeds(epochs)
    if not conditions:
        raise ValueError("Evoked response review requires at least one retained condition.")
    times = _topography_times(epochs, response_window_s, topomap_count)
    figures = [
        condition.evoked.plot_joint(times=times, picks="eeg", show=False)
        for condition in conditions
    ]
    remove_tagged_content(report, tag="evoked-response")
    report.add_html(
        html=evoked_html(
            response_window_s=(float(times[0]), float(times[-1])),
            analysis_status=analysis_status,
        ),
        title="Condition averages and topography sampling",
        section=section,
        tags=("epochs", "evoked-response"),
        replace=True,
    )
    report.add_figure(
        fig=figures,
        title="Condition-averaged evoked responses",
        caption=[
            f"{condition.name} · code {condition.event_code} · "
            f"{condition.n_trials} retained trials"
            for condition in conditions
        ],
        section=section,
        tags=("epochs", "evoked-response"),
        image_format=report_image_format(is_figure_list=True),
        replace=True,
    )
    for figure in figures:
        plt.close(figure)
    return conditions


__all__ = [
    "ConditionEvoked",
    "add_evoked_response_review",
    "condition_evokeds",
    "evoked_html",
]
