"""The response shape at a cluster peak.

A cluster table says a voxel reached z = 6.3. It does not say whether the signal there
rose and fell like a haemodynamic response or whether a handful of frames moved
together -- and both reach the same z on a single-subject map at an uncorrected height.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.report.figures import timeseries


SHAPE = (6, 6, 6)
N_FRAMES = 170
TR = 1.0
PEAK_VOXEL = (2, 2, 2)
PEAK_WORLD = (2.0, 2.0, 2.0)
QUIET_WORLD = (5.0, 5.0, 5.0)


def _boxcar(n: int, onsets, width: int = 6) -> np.ndarray:
    values = np.zeros(n, dtype=float)
    for onset in onsets:
        values[onset : onset + width] = 1.0
    return values


def _run(tmp_path, name: str, *, response: float = 1.0, drop_first: int = 1):
    """One run whose peak voxel responds to condition A and not to condition B."""
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    # Spaced so no onset of one condition falls inside the other's 18 s epoch
    # window; overlapping them makes each condition's tail carry the other's response.
    a_onsets = [10, 60, 110]
    b_onsets = [35, 85, 135]
    cond_a = _boxcar(N_FRAMES, a_onsets)
    cond_b = _boxcar(N_FRAMES, b_onsets)

    data = (100 + rng.standard_normal(SHAPE + (N_FRAMES,))).astype(np.float32)
    data[PEAK_VOXEL] += (response * cond_a).astype(np.float32) * 6.0
    bold_path = tmp_path / f"{name}_bold.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(bold_path))

    rows = np.arange(drop_first, N_FRAMES)
    design = pd.DataFrame(
        {
            timeseries.FRAME_TIME_COLUMN: rows * TR,
            "cond_a": cond_a[rows],
            "cond_b": cond_b[rows],
            "drift_1": np.linspace(-1, 1, rows.size),
            "constant": np.ones(rows.size),
        }
    )
    design_path = tmp_path / f"{name}_design.tsv"
    design.to_csv(design_path, sep="\t", index=False)
    return bold_path, design_path


def _collect(tmp_path, n_runs: int = 2, **kwargs):
    bolds, designs = [], []
    for index in range(n_runs):
        bold, design = _run(tmp_path, f"run{index}")
        bolds.append(bold)
        designs.append(design)
    params = dict(
        bold_paths=bolds,
        design_paths=designs,
        contrast_columns=("cond_a", "cond_b"),
        contrast_vector=(1.0, -1.0),
        t_r=TR,
    )
    params.update(kwargs)
    return timeseries.collect_peak_responses([("1", PEAK_WORLD)], **params)


# --- onsets ----------------------------------------------------------------


def test_onsets_are_rising_crossings_only() -> None:
    # A regressor crosses its own level twice per event; counting both epochs each
    # event a second time from its falling edge, averaging the undershoot onto the
    # peak and flattening the shape the panel exists to show.
    column = _boxcar(60, [10, 30, 50])
    assert list(timeseries.onset_rows(column)) == [10, 30, 50]


def test_a_flat_regressor_has_no_onsets() -> None:
    assert timeseries.onset_rows(np.zeros(40)).size == 0


def test_onsets_scale_with_the_regressor_rather_than_an_absolute_level() -> None:
    # A convolved regressor is scaled by event duration, so a fixed cut would find
    # every onset in a block design and none in an event-related one.
    small = _boxcar(60, [10, 30]) * 0.01
    assert list(timeseries.onset_rows(small)) == [10, 30]


# --- the epoch average -----------------------------------------------------


def test_each_weighted_condition_gets_its_own_response(tmp_path) -> None:
    responses = _collect(tmp_path)
    assert len(responses) == 1
    names = {condition.name for condition in responses[0].conditions}
    assert names == {"cond_a", "cond_b"}


def test_the_responding_condition_rises_and_the_other_does_not(tmp_path) -> None:
    # The whole point of the panel. Compared over the response window rather than
    # the whole epoch: the tail is baseline for both conditions and its maximum is
    # noise.
    responses = _collect(tmp_path)
    by_name = {c.name: c for c in responses[0].conditions}

    def _response(condition):
        window = (condition.times >= 0) & (condition.times <= 10)
        return float(np.mean(condition.mean[window]))

    assert _response(by_name["cond_a"]) > 4 * abs(_response(by_name["cond_b"]))


def test_events_are_pooled_across_runs(tmp_path) -> None:
    one = _collect(tmp_path, n_runs=1)[0]
    two = _collect(tmp_path, n_runs=2)[0]
    a_one = next(c for c in one.conditions if c.name == "cond_a")
    a_two = next(c for c in two.conditions if c.name == "cond_a")
    assert a_two.n_events > a_one.n_events
    assert two.n_runs == 2


def test_epochs_are_baseline_corrected_on_their_own_pre_onset_frames(
    tmp_path,
) -> None:
    # Epochs beginning at different points of the residual drift must start from a
    # common zero, or the spread across events describes where they started.
    responses = _collect(tmp_path)
    condition = next(c for c in responses[0].conditions if c.name == "cond_a")
    pre = condition.mean[condition.times < 0]
    assert abs(float(np.mean(pre))) < 0.2


def test_the_window_spans_before_and_after_the_onset(tmp_path) -> None:
    responses = _collect(tmp_path)
    times = responses[0].conditions[0].times
    assert times.min() < 0 < times.max()


def test_the_conditions_are_ordered_by_their_contrast_weight(tmp_path) -> None:
    # The positively weighted condition reads first, matching the contrast's name.
    responses = _collect(tmp_path)
    assert [c.weight for c in responses[0].conditions] == [1.0, -1.0]


def test_the_spread_is_the_error_across_events(tmp_path) -> None:
    responses = _collect(tmp_path)
    condition = next(c for c in responses[0].conditions if c.name == "cond_a")
    assert condition.sem.shape == condition.mean.shape
    assert np.all(condition.sem >= 0)


# --- alignment -------------------------------------------------------------


def test_the_design_is_aligned_to_the_bold_by_its_frame_times(tmp_path) -> None:
    # The counts differ because the model drops non-steady-state and censored
    # volumes, and which ones is not recoverable from the counts alone. Assuming the
    # difference sits at the start shifts every sample.
    bold, design = _run(tmp_path, "a", drop_first=3)
    responses = timeseries.collect_peak_responses(
        [("1", PEAK_WORLD)],
        bold_paths=[bold],
        design_paths=[design],
        contrast_columns=("cond_a", "cond_b"),
        contrast_vector=(1.0, -1.0),
        t_r=TR,
    )
    condition = next(c for c in responses[0].conditions if c.name == "cond_a")
    # Still finds the response despite three frames missing from the design.
    assert condition.mean.max() > 0.3


def test_a_design_without_frame_times_is_skipped(tmp_path) -> None:
    bold, design = _run(tmp_path, "a")
    frame = pd.read_csv(design, sep="\t").drop(columns=[timeseries.FRAME_TIME_COLUMN])
    frame.to_csv(design, sep="\t", index=False)
    assert (
        timeseries.collect_peak_responses(
            [("1", PEAK_WORLD)],
            bold_paths=[bold],
            design_paths=[design],
            contrast_columns=("cond_a", "cond_b"),
            contrast_vector=(1.0, -1.0),
            t_r=TR,
        )
        == []
    )


# --- declining -------------------------------------------------------------


def test_no_recorded_tr_yields_nothing(tmp_path) -> None:
    assert _collect(tmp_path, t_r=None) == []


def test_an_all_zero_contrast_yields_nothing(tmp_path) -> None:
    assert _collect(tmp_path, contrast_vector=(0.0, 0.0)) == []


def test_no_peaks_yields_nothing(tmp_path) -> None:
    bold, design = _run(tmp_path, "a")
    assert (
        timeseries.collect_peak_responses(
            [],
            bold_paths=[bold],
            design_paths=[design],
            contrast_columns=("cond_a",),
            contrast_vector=(1.0,),
            t_r=TR,
        )
        == []
    )


# --- the figure ------------------------------------------------------------


def test_the_panel_draws_one_column_per_peak(tmp_path) -> None:
    responses = _collect(tmp_path)
    figure = timeseries.peak_response_figure(responses)
    assert len(figure.axes) == len(responses)
    plt.close(figure)


def test_the_panel_names_each_condition_and_its_event_count(tmp_path) -> None:
    responses = _collect(tmp_path)
    figure = timeseries.peak_response_figure(responses)
    legend = figure.axes[0].get_legend()
    labels = " ".join(text.get_text() for text in legend.get_texts())
    assert "cond_a" in labels and "events" in labels
    plt.close(figure)


def test_the_panel_says_it_is_descriptive(tmp_path) -> None:
    # A response shape is not a test, and the map's z already is one.
    responses = _collect(tmp_path)
    figure = timeseries.peak_response_figure(responses)
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "the map's z is the test" in text
    plt.close(figure)


def test_the_panel_says_nothing_was_refitted(tmp_path) -> None:
    responses = _collect(tmp_path)
    figure = timeseries.peak_response_figure(responses)
    text = " ".join(artist.get_text() for artist in figure.texts)
    assert "nothing is refitted" in text
    plt.close(figure)


def test_an_empty_response_list_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least one peak"):
        timeseries.peak_response_figure([])
