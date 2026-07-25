"""Whole-decomposition evidence for the ICA section of the subject report.

Every other panel in the report judges one component at a time. This module answers the
questions that come before that: is the decomposition identifiable at all, and how much
of the recording does the current exclusion set remove?

Nothing here is a new metric. The rank, condition number, and explained-variance ratios
are properties of the fitted ICA that the pipeline already computes and then discards.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np

from eeg_pipeline.preprocessing.report.settings import ReportSettings
from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    FLAG_COLOR,
    GUIDE_COLOR,
    RUN_COLORS,
)

#: Samples per squared component below which an infomax fit is under-determined.
#: Onton and Makeig recommend roughly 20-30 times the squared channel count.
MINIMUM_SAMPLES_PER_SQUARED_COMPONENT = 20.0

#: Variance share below which excluding a component buys negligible cleaning while
#: still costing one dimension of the data.
LOW_VARIANCE_EXCLUSION_FLOOR = 0.01


@dataclass(frozen=True)
class DecompositionSummary:
    """Identifiability and impact of one fitted ICA."""

    n_channels: int
    n_components: int
    data_rank: int
    condition_number: float
    samples_per_squared_component: float
    #: Fraction of sensor variance carried by each component, in component order.
    explained_variance: np.ndarray
    excluded: tuple[int, ...]
    #: Channels dropped from the fit because the pipeline marked them bad. Recorded so
    #: the expected rank can be stated rather than inferred.
    n_bad_channels: int = 0
    #: Whether an average reference was applied, which costs exactly one dimension.
    uses_average_reference: bool = False

    @property
    def expected_rank(self) -> int:
        """Rank the data should have, from the montage and the reference alone.

        Reporting the measured rank without the rank it was supposed to have leaves a
        reviewer unable to tell a benign deficiency from a real one: an average
        reference always costs one dimension, so ``rank = channels - 1`` is correct
        rather than alarming. A measured rank below this is the case worth chasing.
        """
        return self.n_channels - (1 if self.uses_average_reference else 0)

    @property
    def rank_shortfall(self) -> int:
        """Dimensions missing beyond what the reference and montage explain."""
        return self.expected_rank - self.data_rank

    @property
    def rank_accounting(self) -> str:
        """Human-readable derivation of the expected rank."""
        terms = [f"{self.n_channels} good channel(s)"]
        if self.n_bad_channels:
            terms.append(f"{self.n_bad_channels} excluded as bad")
        if self.uses_average_reference:
            terms.append("−1 for the average reference")
        return ", ".join(terms)

    @property
    def variance_removed(self) -> float:
        if not self.excluded:
            return 0.0
        return float(self.explained_variance[list(self.excluded)].sum())

    @property
    def variance_retained(self) -> float:
        return 1.0 - self.variance_removed

    @property
    def is_rank_deficient(self) -> bool:
        """True when more components were fitted than the data can support.

        Fitting beyond the data rank splits real sources across several components and
        manufactures components that look physiological but are numerical artifacts.
        """
        return self.n_components > self.data_rank

    @property
    def is_under_determined(self) -> bool:
        return self.samples_per_squared_component < MINIMUM_SAMPLES_PER_SQUARED_COMPONENT

    @property
    def retained_dimensions(self) -> int:
        """Dimensions left for downstream analysis after the exclusions are applied."""
        return self.data_rank - len(self.excluded)

    def exclusion_cost(
        self,
        *,
        variance_floor: float = LOW_VARIANCE_EXCLUSION_FLOOR,
    ) -> tuple[int, float, int, float]:
        """Split the exclusion set by whether a component carries real variance.

        Every removal costs one dimension of the data, whatever the component's size.
        Splitting the set at a variance floor shows whether the rank is being spent on
        components that actually carry artifact, or on many small ones whose removal
        buys almost no cleaning while still reducing the rank available to covariance
        estimation, source localisation, and connectivity.

        Returns ``(n_above, variance_above, n_below, variance_below)``.
        """
        if not self.excluded:
            return 0, 0.0, 0, 0.0
        excluded = np.asarray(self.excluded, dtype=int)
        variance = self.explained_variance[excluded]
        above = variance >= variance_floor
        return (
            int(above.sum()),
            float(variance[above].sum()),
            int((~above).sum()),
            float(variance[~above].sum()),
        )


def pick_good_eeg(epochs: mne.BaseEpochs) -> mne.BaseEpochs:
    """Return a copy holding only the EEG channels the decomposition actually saw.

    ``Epochs.pick("eeg")`` keeps bad channels — unlike ``pick_types``, whose default is
    ``exclude="bads"`` — so a channel the pipeline already rejected would otherwise
    contribute a dimension to the measured rank and a channel to the reported count.
    """
    good = mne.pick_types(epochs.info, eeg=True, exclude="bads")
    return epochs.copy().pick([epochs.ch_names[index] for index in good])


def count_bad_eeg(epochs: mne.BaseEpochs) -> int:
    """Count EEG channels the pipeline marked bad."""
    all_eeg = mne.pick_types(epochs.info, eeg=True, exclude=())
    good_eeg = mne.pick_types(epochs.info, eeg=True, exclude="bads")
    return len(all_eeg) - len(good_eeg)


def summarize_decomposition(
    *,
    ica: mne.preprocessing.ICA,
    epochs: mne.BaseEpochs,
) -> DecompositionSummary:
    """Measure the rank, conditioning, and variance impact of a fitted ICA."""
    eeg = pick_good_eeg(epochs)
    data = eeg.get_data(copy=False)
    flattened = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
    centered = flattened - flattened.mean(axis=1, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    tolerance = singular_values.max() * max(centered.shape) * np.finfo(float).eps
    data_rank = int((singular_values > tolerance).sum())
    condition_number = float(singular_values[0] / singular_values[data_rank - 1])

    component_count = int(ica.n_components_)
    explained_variance = np.array(
        [
            float(ica.get_explained_variance_ratio(eeg, components=[index])["eeg"])
            for index in range(component_count)
        ]
    )
    # A projection is only counted when it is applied; an unapplied average-reference
    # projector has not yet cost the data a dimension.
    uses_average_reference = any(
        projection["desc"] == "Average EEG reference" and projection["active"]
        for projection in epochs.info["projs"]
    ) or bool(epochs.info.get("custom_ref_applied"))

    return DecompositionSummary(
        n_channels=len(eeg.ch_names),
        n_components=component_count,
        data_rank=data_rank,
        condition_number=condition_number,
        samples_per_squared_component=float(centered.shape[1]) / float(component_count**2),
        explained_variance=explained_variance,
        excluded=tuple(int(index) for index in sorted(ica.exclude)),
        n_bad_channels=count_bad_eeg(epochs),
        uses_average_reference=bool(uses_average_reference),
    )


def decomposition_summary_html(
    summary: DecompositionSummary,
    *,
    settings: ReportSettings | None = None,
) -> str:
    """Render the decomposition's measured properties, without grading them."""
    settings = settings or ReportSettings()
    rows = [
        ("EEG channels used", f"{summary.n_channels}"),
        ("Components fitted", f"{summary.n_components}"),
        (
            "Expected rank",
            f"{summary.expected_rank} ({summary.rank_accounting})",
        ),
        (
            "Numerical data rank",
            f"{summary.data_rank}"
            + (
                f" &mdash; {summary.rank_shortfall} below expected"
                if summary.rank_shortfall > 0
                else ""
            ),
        ),
        ("Condition number", f"{summary.condition_number:.1f}"),
        (
            "Samples per squared component",
            f"{summary.samples_per_squared_component:.0f} "
            f"(want &ge; {settings.min_samples_per_squared_component:.0f})",
        ),
        ("Components excluded", f"{len(summary.excluded)} of {summary.n_components}"),
        (
            "Dimensions left after removal",
            f"{summary.retained_dimensions} of {summary.data_rank}",
        ),
        (
            "<strong>Sensor variance removed</strong>",
            f"<strong>{summary.variance_removed:.1%}</strong>",
        ),
        (
            "<strong>Sensor variance retained</strong>",
            f"<strong>{summary.variance_retained:.1%}</strong>",
        ),
    ]
    body = "".join(f"<tr><td>{name}</td><td>{value}</td></tr>" for name, value in rows)
    document = (
        "<p>Properties of the decomposition as a whole, before judging any single "
        "component. Component numbers refer to the standard broadband ICA used for "
        "artifact removal.</p>"
        f"<table><tbody>{body}</tbody></table>"
        "<p>The expected rank follows from the montage and the reference alone, so a "
        "measured rank that matches it is not a deficiency: an average reference always "
        "costs one dimension. A measured rank <em>below</em> the expectation means "
        "something further reduced the data, most often interpolation or a duplicated "
        "channel, and that is the case worth chasing.</p>"
        "<p>Variance figures describe how much of the recorded sensor variance the "
        "current exclusion set removes. A high value is not wrong on its own &mdash; "
        "ocular and cardiac artifact genuinely dominate variance, especially inside the "
        "scanner &mdash; but it is the number that most needs to be defensible.</p>"
    )
    above_count, above_variance, below_count, below_variance = summary.exclusion_cost(
        variance_floor=settings.low_variance_exclusion_floor
    )
    if below_count:
        document += (
            "<p>Split at "
            f"{settings.low_variance_exclusion_floor:.0%} of variance "
            "(<code>report.thresholds.low_variance_exclusion_floor</code>): "
            f"{above_count} excluded component(s) above it account for "
            f"{above_variance:.1%} of sensor variance, and {below_count} below it for "
            f"{below_variance:.2%}. Each exclusion costs one dimension, so the rank "
            f"available downstream is {summary.retained_dimensions} of "
            f"{summary.data_rank}.</p>"
        )
    return document


def plot_variance_overview(
    summary: DecompositionSummary,
    *,
    settings: ReportSettings | None = None,
) -> plt.Figure:
    """Plot per-component variance, the exclusion set, and the cumulative total.

    Variance spans several orders of magnitude, so the axis is logarithmic. That rules
    out bars: a bar encodes its quantity by length measured from zero, and zero is at
    negative infinity on a log axis, so the length would be set by the arbitrary axis
    limit rather than by the data. Markers encode position only, which stays honest.
    """
    settings = settings or ReportSettings()
    components = np.arange(summary.n_components)
    excluded_mask = np.zeros(summary.n_components, dtype=bool)
    excluded_mask[list(summary.excluded)] = True
    share = summary.explained_variance * 100.0

    figure, (share_axis, cumulative_axis) = plt.subplots(
        2,
        1,
        figsize=(11.0, 5.8),
        height_ratios=(2, 1),
        sharex=True,
        layout="constrained",
    )
    # A stem from the axis floor would reintroduce the same false-length reading, so the
    # guide lines are drawn only as faint context for locating each marker.
    share_axis.vlines(
        components,
        share.min() * 0.7,
        share,
        color="0.90",
        linewidth=0.7,
        zorder=1,
    )
    for mask, color, label in (
        (~excluded_mask, "0.45", "Retained"),
        (excluded_mask, FLAG_COLOR, "Excluded"),
    ):
        share_axis.scatter(
            components[mask],
            share[mask],
            s=26,
            color=color,
            label=label,
            zorder=3,
            edgecolor="white",
            linewidth=0.5,
        )
    share_axis.axhline(
        settings.low_variance_exclusion_floor * 100.0,
        color=GUIDE_COLOR,
        linestyle=":",
        linewidth=1.0,
    )
    share_axis.annotate(
        f"{settings.low_variance_exclusion_floor:.0%} of variance",
        xy=(summary.n_components, settings.low_variance_exclusion_floor * 100.0),
        xytext=(-2, 3),
        textcoords="offset points",
        ha="right",
        fontsize=7,
        color=GUIDE_COLOR,
    )
    share_axis.set(
        title=(
            f"Sensor variance per component · {int(excluded_mask.sum())} excluded "
            f"components remove {summary.variance_removed:.1%} of variance"
        ),
        ylabel="Variance explained (%)",
        yscale="log",
    )
    share_axis.grid(axis="y", alpha=0.2)
    share_axis.spines[["top", "right"]].set_visible(False)
    share_axis.legend(frameon=False, fontsize=8)

    # Both panels share the component axis, so the cumulative curve follows component
    # order too. Mixing a variance-rank curve with a component-order axis would give the
    # same x position two different meanings.
    cumulative = np.cumsum(share)
    cumulative_axis.plot(components, cumulative, color=AFTER_COLOR)
    cumulative_axis.fill_between(
        components,
        0.0,
        cumulative,
        where=excluded_mask,
        color=FLAG_COLOR,
        alpha=0.12,
        step="mid",
    )
    cumulative_axis.axhline(90.0, color=GUIDE_COLOR, linestyle="--", linewidth=1.0)
    cumulative_axis.annotate(
        "90%",
        xy=(summary.n_components, 90.0),
        xytext=(-4, 3),
        textcoords="offset points",
        ha="right",
        fontsize=7,
        color=GUIDE_COLOR,
    )
    cumulative_axis.set(
        xlabel="ICA component (components are ordered by decreasing variance)",
        ylabel="Cumulative (%)",
        ylim=(0, 101),
    )
    cumulative_axis.grid(axis="y", alpha=0.2)
    cumulative_axis.spines[["top", "right"]].set_visible(False)
    plt.close(figure)
    return figure


@dataclass(frozen=True)
class RemovalTopography:
    """Where on the scalp the ICA exclusions actually took amplitude out."""

    channel_names: tuple[str, ...]
    #: Per-channel change in RMS amplitude across ICA, in decibels. Negative is removal.
    change_db: np.ndarray
    info: mne.Info

    @property
    def median_change_db(self) -> float:
        return float(np.median(self.change_db))

    @property
    def spatial_spread_db(self) -> float:
        """Spread of the removal across the montage.

        This is the number that separates the two readings of a large variance figure.
        Artifact is focal, so removing it leaves a wide spread: some channels lose a lot
        and others almost nothing. A removal that is nearly uniform across the scalp has
        no spatial signature of an artifact source and is the case where a large
        variance figure most likely means brain signal went with it.
        """
        tenth, ninetieth = np.percentile(self.change_db, [10.0, 90.0])
        return float(ninetieth - tenth)

    @property
    def worst_channel(self) -> str:
        return self.channel_names[int(np.argmin(self.change_db))]

    @property
    def worst_change_db(self) -> float:
        return float(np.min(self.change_db))


def compute_removal_topography(
    *,
    ica: mne.preprocessing.ICA,
    epochs: mne.BaseEpochs,
) -> RemovalTopography:
    """Measure the per-channel amplitude change produced by the ICA exclusions."""
    eeg = pick_good_eeg(epochs)
    before = eeg.get_data(copy=False)
    after = ica.apply(eeg.copy(), exclude=ica.exclude, verbose="ERROR").get_data(copy=False)

    # RMS pooled over epochs and time, per channel.
    before_rms = np.sqrt(np.mean(before**2, axis=(0, 2)))
    after_rms = np.sqrt(np.mean(after**2, axis=(0, 2)))
    tiny = np.finfo(float).tiny
    change_db = 20.0 * np.log10(np.maximum(after_rms, tiny) / np.maximum(before_rms, tiny))
    return RemovalTopography(
        channel_names=tuple(eeg.ch_names),
        change_db=change_db,
        info=eeg.info,
    )


def removal_topography_html(topography: RemovalTopography) -> str:
    """Render the spatial signature of the removal, without grading it."""
    return (
        "<p>Sensor variance removed is one number for the whole head, and the same "
        "number can mean two opposite things. This is where that amplitude came from.</p>"
        "<table><tbody>"
        f"<tr><td>Median change across channels</td>"
        f"<td>{topography.median_change_db:.1f} dB</td></tr>"
        f"<tr><td>Largest single-channel change</td>"
        f"<td>{topography.worst_change_db:.1f} dB at "
        f"{html.escape(topography.worst_channel)}</td></tr>"
        f"<tr><td><strong>Spatial spread (10th to 90th percentile)</strong></td>"
        f"<td><strong>{topography.spatial_spread_db:.1f} dB</strong></td></tr>"
        "</tbody></table>"
        "<p>Artifact sources are focal, so removing them leaves a wide spread: frontal "
        "or peripheral channels lose a great deal and central channels lose little. A "
        "removal that is close to uniform across the montage has no such spatial "
        "signature, which is the case in which a large variance figure most plausibly "
        "includes brain signal rather than artifact.</p>"
    )


def plot_removal_topography(topography: RemovalTopography) -> plt.Figure:
    """Plot per-channel amplitude change as a topography and as a ranked distribution."""
    figure, (map_axis, rank_axis) = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.2),
        width_ratios=(1, 1.5),
        layout="constrained",
    )
    # Removal is one-signed, so a diverging map centred on zero would spend half its
    # range on values that cannot occur. The limit is the largest observed removal.
    limit = float(np.max(np.abs(topography.change_db)))
    image, _ = mne.viz.plot_topomap(
        topography.change_db,
        topography.info,
        axes=map_axis,
        show=False,
        cmap="viridis",
        vlim=(-limit, 0.0),
        contours=0,
    )
    figure.colorbar(image, ax=map_axis, shrink=0.7, label="Amplitude change (dB)")
    map_axis.set_title("Where the amplitude was removed", fontsize=9)

    order = np.argsort(topography.change_db)
    positions = np.arange(len(order))
    rank_axis.bar(positions, topography.change_db[order], color="0.55", width=0.9)
    rank_axis.axhline(
        topography.median_change_db,
        color=GUIDE_COLOR,
        linestyle="--",
        linewidth=1.0,
        label=f"median ({topography.median_change_db:.1f} dB)",
    )
    rank_axis.set(
        title=(
            f"Per-channel change, ranked · spread "
            f"{topography.spatial_spread_db:.1f} dB between the 10th and 90th percentile"
        ),
        ylabel="Amplitude change (dB)",
        xlabel="EEG channel, ordered by how much was removed",
        xticks=positions,
        xticklabels=[topography.channel_names[index] for index in order],
    )
    rank_axis.tick_params(axis="x", labelrotation=90, labelsize=5)
    rank_axis.legend(frameon=False, fontsize=8)
    rank_axis.grid(axis="y", alpha=0.2)
    rank_axis.spines[["top", "right"]].set_visible(False)
    plt.close(figure)
    return figure


#: Short labels for the contact sheet, keyed by ICLabel class.
_SHORT_LABELS = {
    "brain": "brain",
    "muscle artifact": "muscle",
    "eye blink": "eye",
    "heart beat": "heart",
    "line noise": "line",
    "channel noise": "chan",
    "other": "other",
    "unlabeled": "",
}


def plot_component_overview(
    *,
    ica: mne.preprocessing.ICA,
    labels: Sequence[object],
    columns: int | None = None,
) -> plt.Figure:
    """Plot every component topography on one sheet, annotated with label and status.

    The per-component panels live behind range sliders that show one figure at a time,
    so a reviewer scrolling the report never sees the decomposition as a whole. This is
    the triage view: which components exist, what they look like, which are excluded,
    and where the classifier was unsure.
    """
    component_count = int(ica.n_components_)
    if len(labels) != component_count:
        raise ValueError("Component labels do not match the fitted ICA components.")
    columns = columns or ReportSettings().component_overview_columns
    excluded = set(int(index) for index in ica.exclude)
    rows = int(np.ceil(component_count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(1.55 * columns, 1.85 * rows),
        squeeze=False,
        layout="constrained",
    )
    components = ica.get_components()
    flat = axes.ravel()
    for index in range(component_count):
        axis = flat[index]
        mne.viz.plot_topomap(
            components[:, index],
            ica.info,
            axes=axis,
            show=False,
            contours=0,
            sensors=False,
        )
        label = labels[index]
        short = _SHORT_LABELS.get(getattr(label, "label", ""), getattr(label, "label", ""))
        probability = getattr(label, "probability", 0.0)
        is_excluded = index in excluded
        axis.set_title(
            f"IC{index:03d}{' ×' if is_excluded else ''}\n{short} {probability:.2f}".rstrip(),
            fontsize=6.5,
            color=FLAG_COLOR if is_excluded else "black",
            pad=2,
        )
        if is_excluded:
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color(FLAG_COLOR)
                spine.set_linewidth(1.4)
    for axis in flat[component_count:]:
        axis.remove()
    figure.suptitle(
        f"All {component_count} components · {len(excluded)} excluded (×, outlined) · "
        "label and ICLabel probability beneath each topography\n"
        "Each map is scaled to its own range and ICA signs are arbitrary, so compare "
        "spatial pattern, not colour or polarity",
        fontsize=9,
    )
    plt.close(figure)
    return figure


def plot_run_component_variance(
    *,
    ica: mne.preprocessing.ICA,
    filtered_raw_paths: list[Path],
    components: int = 16,
) -> plt.Figure:
    """Plot the components whose variance changes most from run to run.

    One ICA is fitted to every run concatenated, which assumes the mixing stays fixed
    across the session. A component confined to a single run breaks that assumption and
    usually means something changed in the recording rather than in the brain.

    Components are selected by how unstable they are, not by index. Selecting the first
    N by index would show whichever components happen to carry the most variance and
    could miss an unstable component further down the list entirely.
    """
    if not filtered_raw_paths:
        raise ValueError("Per-run component variance requires at least one filtered run.")
    run_labels = []
    variances = []
    for path in filtered_raw_paths:
        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        sources = ica.get_sources(raw).get_data()
        variances.append(sources.var(axis=1))
        run_labels.append(path.name.split("_run-")[-1].split("_")[0])
    matrix = np.stack(variances)

    # Normalize each component to its own across-run median so that components with
    # very different absolute scales can be compared on one axis.
    normalized_all = matrix / np.median(matrix, axis=0, keepdims=True)
    # Rank by the widest run-to-run swing, in log space so a 10x drop and a 10x rise
    # count equally.
    instability = np.ptp(np.log10(normalized_all), axis=0)
    shown = min(components, matrix.shape[1])
    selected = np.argsort(instability)[::-1][:shown]
    selected = selected[np.argsort(selected)]
    normalized = normalized_all[:, selected]

    figure, axis = plt.subplots(figsize=(max(7.0, 0.75 * shown), 4.4), layout="constrained")
    positions = np.arange(shown)
    for run_index, label in enumerate(run_labels):
        axis.scatter(
            positions,
            normalized[run_index],
            s=26,
            color=RUN_COLORS[run_index % len(RUN_COLORS)],
            label=f"run-{label}",
            edgecolor="white",
            linewidth=0.6,
        )
    axis.axhline(1.0, color=GUIDE_COLOR, linestyle="--", linewidth=1.0)
    axis.set(
        title=(
            f"Component variance by run, relative to each component's median · "
            f"{len(run_labels)} runs · the {shown} least stable of {matrix.shape[1]} components"
        ),
        xlabel="ICA component",
        ylabel="Variance / median across runs",
        xticks=positions,
        xticklabels=[f"{int(index):03d}" for index in selected],
        yscale="log",
    )
    axis.legend(frameon=False, fontsize=7, ncol=min(len(run_labels), 6))
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    plt.close(figure)
    return figure


__all__ = [
    "DecompositionSummary",
    "LOW_VARIANCE_EXCLUSION_FLOOR",
    "MINIMUM_SAMPLES_PER_SQUARED_COMPONENT",
    "RemovalTopography",
    "compute_removal_topography",
    "count_bad_eeg",
    "decomposition_summary_html",
    "pick_good_eeg",
    "plot_component_overview",
    "plot_removal_topography",
    "plot_run_component_variance",
    "plot_variance_overview",
    "removal_topography_html",
    "summarize_decomposition",
]
