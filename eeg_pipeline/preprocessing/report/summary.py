"""Whole-decomposition evidence for the ICA section of the subject report.

Every other panel in the report judges one component at a time. This module answers the
questions that come before that: is the decomposition identifiable at all, and how much
of the recording does the current exclusion set remove?

Nothing here is a new metric. The rank, condition number, and explained-variance ratios
are properties of the fitted ICA that the pipeline already computes and then discards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.patches import Rectangle

from eeg_pipeline.preprocessing.report.settings import ReportSettings
from eeg_pipeline.preprocessing.report.style import (
    EXCLUDED_COLOR,
    EXCLUDED_PANEL_FILL,
    GUIDE_COLOR,
    RUN_COLORS,
    draw_component_status_strip,
    draw_figure_footnote,
    run_label,
)
from eeg_pipeline.preprocessing.report.tables import (
    Align,
    Column,
    Metric,
    grid_table,
    metric_table,
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
    #: Sensor variance each component accounts for *alone*, in component order.
    #:
    #: These do not add up, and nothing may sum them. ICA components are not orthogonal,
    #: so the variance a set of them accounts for is not the sum of what each accounts
    #: for individually — MNE's ``get_explained_variance_ratio`` documents this, and the
    #: sum over all components is generally not 1. Summing them here reported 99.0% of
    #: sensor variance removed for sub-0015 where the measured joint value was 91.8%,
    #: understating retained variance roughly eightfold. Named ``individual_`` so that
    #: any future use has to acknowledge what it holds.
    individual_variance: np.ndarray
    excluded: tuple[int, ...]
    #: Sensor variance the whole exclusion set accounts for, measured jointly.
    #:
    #: A field rather than a property: it cannot be derived from
    #: :attr:`individual_variance` and has to be measured against the data, so every
    #: construction site is required to supply it.
    variance_removed: float
    #: Individual-share threshold that :attr:`variance_above_floor` split the set at.
    variance_floor: float
    #: Joint variance of the excluded components whose individual share reaches the
    #: floor, and of those below it. Each is its own joint measurement, so the two do
    #: not add to :attr:`variance_removed` and must never be presented as if they did.
    variance_above_floor: float
    variance_below_floor: float
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

    @property
    def unfitted_dimensions(self) -> int:
        """Dimensions of the data that never entered the decomposition.

        ``ICA.apply`` restores the PCA components between ``n_components_`` and the data
        rank unmodified, so whatever artifact lives in them survives every exclusion. The
        count matters because a variance criterion collapses precisely when artifact
        dominates variance: on sub-0015, blink and cardiac accounted for 85% of it and
        ``n_components=0.99`` fitted 22 components of a rank-62 recording, leaving 40
        dimensions no exclusion could reach.

        Clamped at zero. Fitting *beyond* the rank is a separate fault with its own
        measurement in :attr:`is_rank_deficient`, and reporting it here as a negative
        number of pass-through dimensions would describe something that cannot happen.
        """
        return max(0, self.data_rank - self.n_components)

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

        Both variances are joint measurements of their own subgroup, taken when the
        summary was built. They therefore do not add to :attr:`variance_removed`, and a
        caller that adds them has reintroduced the non-additivity bug this replaced.

        ``variance_floor`` must match the floor the summary was measured at: the split
        cannot be recomputed here without the data. Returns
        ``(n_above, variance_above, n_below, variance_below)``.
        """
        if not self.excluded:
            return 0, 0.0, 0, 0.0
        if not np.isclose(variance_floor, self.variance_floor):
            raise ValueError(
                f"This summary was measured at a variance floor of {self.variance_floor!r}; "
                f"the joint subgroup variances cannot be restated at {variance_floor!r}. "
                "Pass the floor to summarize_decomposition instead."
            )
        excluded = np.asarray(self.excluded, dtype=int)
        above = self.individual_variance[excluded] >= variance_floor
        return (
            int(above.sum()),
            self.variance_above_floor,
            int((~above).sum()),
            self.variance_below_floor,
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


def _joint_variance(
    ica: mne.preprocessing.ICA,
    eeg: mne.BaseEpochs,
    components: Sequence[int],
) -> float:
    """Return the sensor variance a set of components accounts for, measured together.

    ``get_explained_variance_ratio`` computes jointly when handed more than one
    component, which is the only correct way to score a set: the individual ratios are
    not additive because the components are not orthogonal.
    """
    if not len(components):
        return 0.0
    return float(ica.get_explained_variance_ratio(eeg, components=list(components))["eeg"])


def summarize_decomposition(
    *,
    ica: mne.preprocessing.ICA,
    epochs: mne.BaseEpochs,
    variance_floor: float = LOW_VARIANCE_EXCLUSION_FLOOR,
) -> DecompositionSummary:
    """Measure the rank, conditioning, and variance impact of a fitted ICA.

    ``variance_floor`` is fixed here rather than at render time because the subgroup
    variances either side of it are joint measurements against the data, and the data
    are only available at this point.
    """
    eeg = pick_good_eeg(epochs)
    data = eeg.get_data(copy=False)
    flattened = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
    centered = flattened - flattened.mean(axis=1, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    tolerance = singular_values.max() * max(centered.shape) * np.finfo(float).eps
    data_rank = int((singular_values > tolerance).sum())
    condition_number = float(singular_values[0] / singular_values[data_rank - 1])

    component_count = int(ica.n_components_)
    # Per-component shares, for the scatter that locates each component. Reported as
    # individual values only; the set-level numbers below are measured separately.
    individual_variance = np.array(
        [
            float(ica.get_explained_variance_ratio(eeg, components=[index])["eeg"])
            for index in range(component_count)
        ]
    )
    excluded = tuple(int(index) for index in sorted(ica.exclude))
    above_floor = [index for index in excluded if individual_variance[index] >= variance_floor]
    below_floor = [index for index in excluded if individual_variance[index] < variance_floor]
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
        individual_variance=individual_variance,
        excluded=excluded,
        variance_removed=_joint_variance(ica, eeg, excluded),
        variance_floor=float(variance_floor),
        variance_above_floor=_joint_variance(ica, eeg, above_floor),
        variance_below_floor=_joint_variance(ica, eeg, below_floor),
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
    rows: list[Metric | tuple[str, object]] = [
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
                f" — {summary.rank_shortfall} below expected"
                if summary.rank_shortfall > 0
                else ""
            ),
        ),
        ("Condition number", f"{summary.condition_number:.1f}"),
        (
            "Samples per squared component",
            f"{summary.samples_per_squared_component:.0f} "
            f"(want ≥ {settings.min_samples_per_squared_component:.0f})",
        ),
        ("Components excluded", f"{len(summary.excluded)} of {summary.n_components}"),
        (
            "Dimensions left after removal",
            f"{summary.retained_dimensions} of {summary.data_rank}",
        ),
        *(
            [
                (
                    "Dimensions outside the fit",
                    f"{summary.unfitted_dimensions} of {summary.data_rank}",
                )
            ]
            if summary.unfitted_dimensions
            else []
        ),
        Metric("Sensor variance removed", f"{summary.variance_removed:.1%}", emphasis=True),
        Metric("Sensor variance retained", f"{summary.variance_retained:.1%}", emphasis=True),
    ]
    document = (
        "<p>Properties of the decomposition as a whole, before judging any single "
        "component. Component numbers refer to the standard broadband ICA used for "
        "artifact removal.</p>"
        f"{metric_table(rows)}"
        "<p>The expected rank follows from the montage and the reference alone, so a "
        "measured rank that matches it is not a deficiency: an average reference always "
        "costs one dimension. A measured rank <em>below</em> the expectation means "
        "something further reduced the data, most often interpolation or a duplicated "
        "channel, and that is the case worth chasing.</p>"
        + (
            "<p>Fewer components were fitted than the data has dimensions, so "
            f"{summary.unfitted_dimensions} of {summary.data_rank} dimensions never "
            "entered the decomposition. Applying the ICA restores them "
            "<strong>unmodified</strong>, which means no exclusion can reach whatever "
            "they carry. This is worth reading beside the variance figures rather than "
            "after them: a criterion that selects components by variance stops early "
            "when artifact holds most of the variance, which is the usual case inside a "
            "scanner, so the components it did fit can explain almost all of the "
            "variance while leaving most of the dimensions untouched.</p>"
            if summary.unfitted_dimensions
            else ""
        )
        + "<p>Variance figures describe how much of the recorded sensor variance the "
        "current exclusion set removes. A high value is not wrong on its own &mdash; "
        "ocular and cardiac artifact genuinely dominate variance, especially inside the "
        "scanner &mdash; but it is the number that most needs to be defensible.</p>"
        "<p>Sensor variance removed is measured for the exclusion set as a whole, "
        "against the data. It is <strong>not</strong> the sum of the per-component "
        "shares plotted below: ICA components are not orthogonal, so their individual "
        "shares are <strong>not additive</strong> and generally do not total 100% even "
        "across every component. Adding them overstates removal, and correspondingly "
        "understates what was kept.</p>"
    )
    above_count, above_variance, below_count, below_variance = summary.exclusion_cost(
        variance_floor=summary.variance_floor
    )
    if below_count:
        document += (
            "<p>Split at "
            f"{summary.variance_floor:.0%} of variance "
            "(<code>report.thresholds.low_variance_exclusion_floor</code>): "
            f"{above_count} excluded component(s) sit above it and account for "
            f"{above_variance:.1%} of sensor variance together, and {below_count} below "
            f"it account for {below_variance:.2%} together. Each figure is measured "
            "jointly for its own group, so the two do not add to the total above. Each "
            "exclusion costs one dimension whatever its size, so the rank available "
            f"downstream is {summary.retained_dimensions} of {summary.data_rank}.</p>"
        )
    return document


def decomposition_measurements(summary: DecompositionSummary) -> dict[str, float | int]:
    """Return the decomposition's headline numbers, ready for the build record.

    These are the figures the decomposition panel renders as prose, and the ones a
    decision about whether to keep a subject actually turns on. Emitting them as data
    means such a decision reads a JSON file rather than parsing a paragraph, and reads the
    same numbers the report shows because both come from one measurement.

    Every value is a plain Python number: the record is serialised as JSON, and the numpy
    scalars these are computed as are not serialisable.
    """
    return {
        "n_channels": int(summary.n_channels),
        "n_components": int(summary.n_components),
        "n_excluded": int(len(summary.excluded)),
        "data_rank": int(summary.data_rank),
        "retained_dimensions": int(summary.retained_dimensions),
        "condition_number": float(summary.condition_number),
        "samples_per_squared_component": float(summary.samples_per_squared_component),
        "variance_removed": float(summary.variance_removed),
        "n_bad_channels": int(summary.n_bad_channels),
    }


def exclusion_ledger_html(
    summary: DecompositionSummary,
    *,
    status_descriptions: Sequence[str],
) -> str:
    """Render one row per component: the decision, what made it, and what it cost.

    The report already stated the decision, in MNE-BIDS-Pipeline's ICLabel table, and
    already stated the ICLabel class beside it. What it never stated is that those two
    columns answer different questions. ICLabel is one of several detectors that can mark
    a component — the ECG correlation and the ocular correlation mark their own — so a
    component reading "brain, 0.93" next to "excluded: yes" looked like the table
    contradicting itself rather than like two detectors disagreeing.

    ``status_descriptions`` is the ``status_description`` column of
    ``*_proc-ica_components.tsv``, which is where the pipeline already records which
    detector fired and is empty for a component nothing marked. That table is also what
    decides the exclusions applied to the data, so a ledger built from it cannot drift
    from the derivative the way one rebuilt from ``ICA.exclude`` could.

    Retained components are listed too. A reviewer asking whether a detector was too eager
    is asking about what it spared, and a table of removals alone cannot answer that.
    """
    if len(status_descriptions) != summary.n_components:
        raise ValueError(
            f"The ledger needs one status description per component: got "
            f"{len(status_descriptions)} for {summary.n_components} components."
        )
    excluded = set(summary.excluded)
    rows = []
    for component in range(summary.n_components):
        reason = str(status_descriptions[component]).strip()
        is_excluded = component in excluded
        # An exclusion with no recorded reason is worth naming as such rather than
        # leaving blank, which reads as "retained" at a glance.
        if is_excluded and not reason:
            reason = "excluded with no detector recorded"
        rows.append(
            [
                f"ICA{component:03d}",
                "excluded" if is_excluded else "retained",
                reason or None,
                f"{summary.individual_variance[component]:.1%}",
            ]
        )
    columns = (
        Column("Component", align=Align.TEXT, code=True),
        Column("Decision", align=Align.TEXT),
        Column("Marked by", align=Align.TEXT),
        Column("Variance share"),
    )
    return (
        "<p>Which detector marked each component, and what excluding it cost. The "
        "decision comes from <code>*_proc-ica_components.tsv</code>, the same table the "
        "pipeline reads when it applies the ICA, so this ledger and the cleaned data "
        "cannot disagree.</p>"
        f"{grid_table(columns, rows)}"
        "<p>Detectors are independent of each other. ICLabel classifies a component from "
        "its topography, spectrum, and time course; the cardiac and ocular detectors "
        "correlate it against a measured reference. A component ICLabel calls brain can "
        "therefore be excluded on a correlation it never saw, which is a disagreement "
        "between detectors to be reviewed rather than an inconsistency in the table. "
        "Variance share is the component's own, and shares are not additive across "
        "components.</p>"
    )


def plot_variance_overview(
    summary: DecompositionSummary,
    *,
    settings: ReportSettings | None = None,
) -> plt.Figure:
    """Plot the sensor variance each component accounts for on its own.

    Variance spans several orders of magnitude, so the axis is logarithmic. That rules
    out bars: a bar encodes its quantity by length measured from zero, and zero is at
    negative infinity on a log axis, so the length would be set by the arbitrary axis
    limit rather than by the data. Markers encode position only, which stays honest.

    There is deliberately no cumulative curve. It plotted a running total of these shares
    in ICA's own component order, and both halves of that are unsound: the shares are not
    additive because the components are not orthogonal, and the order is arbitrary, so
    the curve was neither a valid cumulative total nor a scree plot. Computing it jointly
    would not rescue it — the quantity it claimed to show is not defined. The set-level
    number lives in the summary table, measured against the data.
    """
    # The floor comes from the summary, not from settings: it is the floor the subgroup
    # variances were actually measured at, so taking it from anywhere else lets the guide
    # line and the table below the figure describe different splits.
    floor = summary.variance_floor
    components = np.arange(summary.n_components)
    excluded_mask = np.zeros(summary.n_components, dtype=bool)
    excluded_mask[list(summary.excluded)] = True
    share = summary.individual_variance * 100.0

    # The decision gets its own strip rather than only the marker's lightness. At the
    # marker size that fits sixty components, "excluded" and "retained" were two pale
    # greys a few pixels across, which made the panel's own subject the hardest thing on
    # it to read. Same strip the ocular correlation panel carries, so the two read alike.
    figure, (share_axis, status_axis) = plt.subplots(
        2,
        1,
        figsize=(11.0, 4.9),
        height_ratios=(12, 1),
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
        (excluded_mask, EXCLUDED_COLOR, "Excluded"),
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
        floor * 100.0,
        color=GUIDE_COLOR,
        linestyle=":",
        linewidth=1.0,
    )
    share_axis.annotate(
        f"{floor:.0%} of variance",
        xy=(summary.n_components, floor * 100.0),
        xytext=(-2, 3),
        textcoords="offset points",
        ha="right",
        fontsize=7,
        color=GUIDE_COLOR,
    )
    share_axis.set(
        title=(
            f"Sensor variance per component · {int(excluded_mask.sum())} excluded "
            f"components remove {summary.variance_removed:.1%} of variance, "
            "measured jointly"
        ),
        ylabel="Variance explained alone (%)",
        yscale="log",
    )
    # Stated on the figure, not only in the surrounding prose: a slide exported on its own
    # otherwise invites exactly the addition that produced the wrong headline number.
    #
    # The ordering caveat rides along here rather than in the axis label, where it used to
    # sit. Component order is ICA's own and carries no ranking, but the shares often fall
    # monotonically anyway, which is exactly when a reader mistakes the panel for a scree
    # plot — so the warning belongs with the other thing this panel is not.
    draw_figure_footnote(
        figure,
        "individual shares — not additive, they do not sum to the joint total, "
        "and component order carries no ranking",
    )
    share_axis.grid(axis="y", alpha=0.2)
    share_axis.spines[["top", "right"]].set_visible(False)
    share_axis.legend(frameon=False, fontsize=8)
    # The strip carries its own key, so the scatter no longer needs to explain the greys.
    draw_component_status_strip(
        status_axis,
        excluded=summary.excluded,
        component_count=summary.n_components,
        legend=False,
    )
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
        + metric_table(
            [
                ("Median change across channels", f"{topography.median_change_db:.1f} dB"),
                (
                    "Largest single-channel change",
                    f"{topography.worst_change_db:.1f} dB at {topography.worst_channel}",
                ),
                Metric(
                    "Spatial spread (10th to 90th percentile)",
                    f"{topography.spatial_spread_db:.1f} dB",
                    emphasis=True,
                ),
            ]
        )
        +
        "<p>Artifact sources are focal, so removing them leaves a wide spread: frontal "
        "or peripheral channels lose a great deal and central channels lose little. A "
        "removal that is close to uniform across the montage has no such spatial "
        "signature, which is the case in which a large variance figure most plausibly "
        "includes brain signal rather than artifact.</p>"
    )


#: Largest number of channel names drawn on the ranked panel before they are thinned.
#:
#: A 64-channel montage at every tick needs 5 pt type, which is a texture rather than a
#: set of names. Matches the component strip's own limit in
#: :mod:`eeg_pipeline.preprocessing.report.style`.
_MAX_CHANNEL_TICKS = 32


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
    # Horizontal, under the map it belongs to. A vertical colourbar between the two panels
    # sat hard against the ranked panel's y-axis label, which measures the same quantity
    # in the same units, so labelling both printed "Amplitude change (dB)" twice and
    # overlapping — and dropping the label left the topography, the panel a reader looks
    # at first, with a colour scale carrying no unit at all. Below the map the two labels
    # are far apart and each scale can name itself.
    figure.colorbar(
        image,
        ax=map_axis,
        orientation="horizontal",
        shrink=0.8,
        pad=0.04,
        label="Amplitude change (dB)",
    )
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
    # The percentiles the spread in the title is taken between. Quoting a spread while
    # drawing neither end of it asks the reader to take the panel's own headline on trust.
    low_percentile, high_percentile = (
        float(np.percentile(topography.change_db, percentile)) for percentile in (10.0, 90.0)
    )
    for level in (low_percentile, high_percentile):
        rank_axis.axhline(level, color=GUIDE_COLOR, linestyle=":", linewidth=0.9)
    rank_axis.annotate(
        "10th–90th",
        xy=(0.995, high_percentile),
        xycoords=("axes fraction", "data"),
        xytext=(0, 2),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7,
        color=GUIDE_COLOR,
    )
    # Short enough to stay over its own panel. The long form overflowed to the left and
    # printed across the colourbar's tick labels; the percentiles it described are now
    # drawn on the axis, so the title only has to name the number.
    rank_axis.set_title(
        f"Per-channel change, ranked · 10th–90th spread "
        f"{topography.spatial_spread_db:.1f} dB",
        fontsize=9,
    )
    rank_axis.set(
        ylabel="Amplitude change (dB)",
        xlabel="EEG channel, ordered by how much was removed",
    )
    # Named channels only where a name can be read. A full montage put sixty-odd labels at
    # 5 pt, which is a grey smear rather than an axis; the panel is about the shape of the
    # distribution and its extremes, so the ends are labelled and the middle is thinned.
    step = max(1, math.ceil(len(order) / _MAX_CHANNEL_TICKS))
    shown = positions[::step]
    rank_axis.set_xticks(shown)
    rank_axis.set_xticklabels([topography.channel_names[order[index]] for index in shown])
    rank_axis.tick_params(axis="x", labelrotation=90, labelsize=7)
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


def _runner_up_text(label: object) -> str:
    """Render the classifier's second choice for one tile, or nothing when it has none.

    The winning class alone cannot distinguish a confident call from a coin flip, and
    this sheet is where a reviewer decides which components to interrogate. Reading the
    runner-up off the component TSV instead means the decision about what to look at is
    made without the number that should drive it.

    Read through ``getattr`` like the label and probability beside it, so the sheet stays
    drawable from any label object -- including the placeholder the exploratory report
    uses when ICLabel did not run, which has no second choice to report.
    """
    runner_up = getattr(label, "runner_up", None)
    if runner_up is None:
        return ""
    name, probability = runner_up
    return f"{_SHORT_LABELS.get(name, name)} {probability:.2f}"


def _mark_excluded_panel(axis: plt.Axes) -> None:
    """Shade one topography panel to mark the component as excluded.

    ``mne.viz.plot_topomap`` calls ``set_axis_off`` on the axis it draws into, which stops
    the spines rendering no matter what visibility is set afterwards. An outline drawn
    through the spines is therefore silently dropped, which previously left the exclusion
    resting on a title in 0.25 grey against a black one — the weakest possible encoding
    for the most consequential read in the figure. Artists added to the axis are still
    drawn with the axis off, so the mark is a patch in axes coordinates.

    The fill stays on the neutral status ramp rather than taking a hue, because in this
    report hue always encodes a measured quantity and never a decision the pipeline made.
    """
    patch = Rectangle(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=axis.transAxes,
        facecolor=EXCLUDED_PANEL_FILL,
        edgecolor=EXCLUDED_COLOR,
        linewidth=1.0,
        zorder=-5,
    )
    # Tagged so a test can assert the mark exists without matching on colour or geometry.
    patch._is_exclusion_mark = True
    axis.add_patch(patch)


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
            f"IC{index:03d}{' ×' if is_excluded else ''}\n"
            f"{short} {probability:.2f}\n{_runner_up_text(label)}".rstrip(),
            fontsize=6.5,
            color=EXCLUDED_COLOR if is_excluded else "black",
            pad=2,
        )
        if is_excluded:
            _mark_excluded_panel(axis)
    for axis in flat[component_count:]:
        axis.remove()
    figure.suptitle(
        f"All {component_count} components · {len(excluded)} excluded (×, shaded panel) · "
        "ICLabel's first and second choice beneath each topography\n"
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
        run_labels.append(run_label(path.name, bare=True))
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
