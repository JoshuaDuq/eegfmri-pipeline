"""Whether an effect is carried by every run or by one of them.

An effect resting entirely on run 4 and an effect present in all six produce the same
map, the same z, and the same cluster table. This panel is the only thing in the report
that tells them apart, so what it draws has to be checked rather than assumed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.report.figures import run_consistency

SHAPE = (10, 10, 10)
N_RUNS = 6
CONSISTENT = (3.0, 3.0, 3.0)
ONE_RUN = (6.0, 6.0, 6.0)


@pytest.fixture()
def maps():
    """Two peaks: one present in every run, one carried by run 4 alone."""
    rng = np.random.default_rng(1)
    effect = rng.normal(0.0, 0.02, SHAPE + (N_RUNS,)).astype(np.float32)
    variance = np.full(SHAPE + (N_RUNS,), 0.0016, dtype=np.float32)

    effect[3, 3, 3, :] = [0.31, 0.28, 0.35, 0.26, 0.30, 0.33]
    effect[6, 6, 6, :] = [0.02, -0.03, 0.01, 0.62, 0.00, -0.02]

    return {
        "run_effect_img": nib.Nifti1Image(effect, np.eye(4)),
        "run_variance_img": nib.Nifti1Image(variance, np.eye(4)),
        "combined_effect_img": nib.Nifti1Image(effect.mean(axis=3), np.eye(4)),
        "combined_variance_img": nib.Nifti1Image(
            np.full(SHAPE, 0.0004, dtype=np.float32), np.eye(4)
        ),
    }


PEAKS = [("1", CONSISTENT), ("2", ONE_RUN)]


# --- reading the estimates -------------------------------------------------


def test_each_peak_gets_one_estimate_per_run(maps) -> None:
    estimates = run_consistency.collect_peak_estimates(PEAKS, **maps)
    assert len(estimates) == 2
    assert all(len(e.effects) == N_RUNS for e in estimates)
    assert all(len(e.errors) == N_RUNS for e in estimates)


def test_the_standard_error_is_the_root_of_the_variance(maps) -> None:
    estimates = run_consistency.collect_peak_estimates(PEAKS, **maps)
    assert estimates[0].errors[0] == pytest.approx(0.04)


def test_the_combined_estimate_is_read_alongside(maps) -> None:
    estimates = run_consistency.collect_peak_estimates(PEAKS, **maps)
    assert estimates[0].combined_effect == pytest.approx(0.305, abs=0.01)
    assert estimates[0].combined_error == pytest.approx(0.02)


def test_sign_agreement_separates_a_consistent_peak_from_a_one_run_peak(maps) -> None:
    consistent, one_run = run_consistency.collect_peak_estimates(PEAKS, **maps)
    assert consistent.sign_agreement == pytest.approx(1.0)
    # Three of six runs happen to share the combined sign at the one-run peak; the
    # point is only that it is well short of unanimous.
    assert one_run.sign_agreement < 0.75


def test_a_peak_outside_the_volume_yields_no_estimate(maps) -> None:
    estimates = run_consistency.collect_peak_estimates([("1", (900.0, 0.0, 0.0))], **maps)
    assert len(estimates) == 1
    assert all(not np.isfinite(value) for value in estimates[0].effects)


def test_the_estimates_are_read_without_a_combined_map(maps) -> None:
    # A contrast whose effect map was not written still gets per-run rows.
    estimates = run_consistency.collect_peak_estimates(
        PEAKS,
        run_effect_img=maps["run_effect_img"],
        run_variance_img=maps["run_variance_img"],
    )
    assert estimates[0].combined_effect is None


def test_run_effect_correlation_uses_only_the_fitted_mask() -> None:
    first = np.array([1.0, 2.0, 3.0, 100.0])
    second = np.array([2.0, 4.0, 6.0, -100.0])
    third = np.array([3.0, 2.0, 1.0, 50.0])
    effects = np.stack((first, second, third), axis=-1).reshape(4, 1, 1, 3)
    mask = np.array([1, 1, 1, 0], dtype=np.uint8).reshape(4, 1, 1)

    correlation = run_consistency.run_effect_correlation_matrix(
        nib.Nifti1Image(effects, np.eye(4)),
        nib.Nifti1Image(mask, np.eye(4)),
    )

    np.testing.assert_allclose(
        correlation,
        np.array(
            [
                [1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ]
        ),
    )


def test_run_effect_correlation_requires_spatial_variation() -> None:
    effects = np.ones((3, 1, 1, 2), dtype=np.float32)
    mask = np.ones((3, 1, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match="non-zero spatial variance"):
        run_consistency.run_effect_correlation_matrix(
            nib.Nifti1Image(effects, np.eye(4)),
            nib.Nifti1Image(mask, np.eye(4)),
        )


@pytest.mark.parametrize("invalid_value", [np.nan, 2.0])
def test_run_effect_correlation_rejects_a_non_binary_mask(
    invalid_value: float,
) -> None:
    effects = np.arange(12, dtype=np.float32).reshape(3, 2, 1, 2)
    mask = np.ones((3, 2, 1), dtype=np.float32)
    mask[0, 0, 0] = invalid_value

    with pytest.raises(ValueError, match="finite binary"):
        run_consistency.run_effect_correlation_matrix(
            nib.Nifti1Image(effects, np.eye(4)),
            nib.Nifti1Image(mask, np.eye(4)),
        )


def test_run_effect_correlation_figure_labels_every_run() -> None:
    correlation = np.array(
        [
            [1.0, 0.5, -0.25],
            [0.5, 1.0, 0.1],
            [-0.25, 0.1, 1.0],
        ]
    )
    labels = ("run-01", "run-02", "run-03")

    figure = run_consistency.run_effect_correlation_figure(
        correlation,
        run_labels=labels,
        title="Whole-mask run agreement",
    )
    figure.canvas.draw()

    tick_text = {
        tick.get_text()
        for axis in figure.axes
        for tick in (*axis.get_xticklabels(), *axis.get_yticklabels())
        if tick.get_text()
    }
    provenance = " ".join(artist.get_text() for artist in figure.texts)
    assert set(labels) <= tick_text
    assert "Pearson r" in provenance
    assert "fitted analysis mask" in provenance
    plt.close(figure)


# --- the figure ------------------------------------------------------------


def _labels() -> list[str]:
    return [f"run-{i:02d}" for i in range(1, N_RUNS + 1)]


def test_the_panel_draws_one_column_per_peak(maps) -> None:
    estimates = run_consistency.collect_peak_estimates(PEAKS, **maps)
    figure = run_consistency.peak_forest_figure(estimates, run_labels=_labels())
    # Two peak axes.
    assert len(figure.axes) == 2
    plt.close(figure)


def test_every_run_gets_a_labelled_row(maps) -> None:
    estimates = run_consistency.collect_peak_estimates(PEAKS, **maps)
    figure = run_consistency.peak_forest_figure(estimates, run_labels=_labels())
    figure.canvas.draw()
    ticks = [t.get_text() for t in figure.axes[0].get_yticklabels() if t.get_text()]
    assert ticks == _labels()
    plt.close(figure)


def test_the_panel_marks_the_combined_estimate(maps) -> None:
    # Every run is read against the combined estimate, not against zero alone.
    estimates = run_consistency.collect_peak_estimates(PEAKS, **maps)
    figure = run_consistency.peak_forest_figure(estimates, run_labels=_labels())
    marked = [line.get_xdata()[0] for line in figure.axes[0].lines]
    assert any(abs(float(x) - estimates[0].combined_effect) < 1e-6 for x in marked)
    plt.close(figure)


def test_the_panel_names_the_peak_and_its_coordinate(maps) -> None:
    # The columns key to the cluster table's rows.
    estimates = run_consistency.collect_peak_estimates(PEAKS, **maps)
    figure = run_consistency.peak_forest_figure(estimates, run_labels=_labels())
    title = figure.axes[1].get_title()
    assert "peak 2" in title and "+6" in title
    plt.close(figure)


def test_the_panel_reports_sign_agreement(maps) -> None:
    estimates = run_consistency.collect_peak_estimates(PEAKS, **maps)
    figure = run_consistency.peak_forest_figure(estimates, run_labels=_labels())
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "sign agreement" in text
    plt.close(figure)


def test_the_panel_scores_no_run(maps) -> None:
    # A task with habituation should show runs differing; that is not a fault.
    estimates = run_consistency.collect_peak_estimates(PEAKS, **maps)
    figure = run_consistency.peak_forest_figure(estimates, run_labels=_labels())
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "not a fault" in text
    plt.close(figure)


def test_too_many_peaks_are_capped_and_the_cap_is_stated(maps) -> None:
    # Past a handful the columns are narrower than their own labels, and the peaks
    # beyond the sixth are rarely what a result rests on.
    estimates = run_consistency.collect_peak_estimates(PEAKS, **maps) * 5
    figure = run_consistency.peak_forest_figure(estimates, run_labels=_labels(), max_peaks=3)
    assert len(figure.axes) == 3
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "3 of 10 peak(s) shown" in text
    plt.close(figure)


def test_no_peaks_is_refused_rather_than_drawn_empty() -> None:
    with pytest.raises(ValueError, match="at least one peak"):
        run_consistency.peak_forest_figure([], run_labels=_labels())
