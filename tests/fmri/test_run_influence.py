"""Leave-one-run-out influence on the combined contrast.

A fixed-effects combination can rest on a single run. The forest panel shows each run's
estimate at the selected peaks; this measures what each run does to the whole map, which
is a different question and can give a different answer -- on this study's own data the
run with the largest peak estimates is not the run that moves the map most.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
import pandas as pd
import pytest

from fmri_pipeline.analysis import run_level

# Sized so the survivor count is driven by the effect rather than by noise. At
# 6x6x6 with 48 frames the whole map yields single-digit survivors, and a delta
# measured against 7 voxels cannot detect the behaviour under test.
SHAPE = (10, 10, 10)
N_FRAMES = 96
TR = 2.0
ONSETS = np.arange(6, 90, 14).astype(float)
EFFECT = 4.0


def _model(active_runs, n_runs=3, seed=0):
    from nilearn.glm.first_level import FirstLevelModel

    rng = np.random.default_rng(seed)
    mask = nib.Nifti1Image(np.ones(SHAPE, dtype=np.uint8), np.eye(4))
    bolds, events = [], []
    for run in range(n_runs):
        data = 100 + rng.standard_normal(SHAPE + (N_FRAMES,)).astype(np.float32)
        if run in active_runs:
            for onset in ONSETS:
                frame = int(onset / TR)
                # Offset by two frames: the response the SPM HRF predicts peaks a few
                # seconds after onset, and an effect written at the onset frame is
                # partly orthogonal to the regressor that has to detect it.
                data[2:7, 2:7, 2:7, frame + 2 : frame + 6] += EFFECT
        bolds.append(nib.Nifti1Image(data, np.eye(4)))
        events.append(
            pd.DataFrame({"onset": ONSETS, "duration": 6.0, "trial_type": "task"})
        )
    model = FirstLevelModel(
        t_r=TR,
        mask_img=mask,
        hrf_model="spm",
        drift_model=None,
        minimize_memory=False,
        standardize=False,
        signal_scaling=False,
    )
    model.fit(bolds, events=events)
    return model


@pytest.fixture(scope="module")
def one_active_run():
    """Three runs, the effect present in run-02 alone."""
    return _model(active_runs={1})


def test_one_row_per_run(one_active_run):
    rows = run_level.compute_run_influence(
        one_active_run, "task", run_labels=("run-01", "run-02", "run-03")
    )
    assert [row.dropped_run for row in rows] == ["run-01", "run-02", "run-03"]


def test_dropping_the_only_active_run_costs_the_most(one_active_run):
    """The effect lives in run-02 alone, so dropping it must cost the most survivors."""
    rows = run_level.compute_run_influence(
        one_active_run,
        "task",
        run_labels=("run-01", "run-02", "run-03"),
        threshold=2.3,
    )
    by_run = {row.dropped_run: row for row in rows}
    assert by_run["run-02"].delta < by_run["run-01"].delta
    assert by_run["run-02"].delta < by_run["run-03"].delta


def test_dropping_the_only_active_run_decorrelates_the_map(one_active_run):
    """Survivor count and map shape are separate measurements; both must respond."""
    rows = run_level.compute_run_influence(
        one_active_run, "task", run_labels=("run-01", "run-02", "run-03")
    )
    by_run = {row.dropped_run: row for row in rows}
    assert by_run["run-02"].correlation < by_run["run-01"].correlation
    assert by_run["run-02"].correlation < by_run["run-03"].correlation


def test_a_zero_vector_drops_the_run_rather_than_contributing_zero():
    """With two runs, dropping one must reproduce the survivor's own statistic exactly.

    This is the property the whole approach rests on. nilearn's
    ``compute_fixed_effect_contrast`` skips a null contrast vector and divides by the
    count of surviving contrasts, so ``[c, 0]`` returns run-01's own contrast untouched.
    If it ever stopped skipping them, the dropped run would instead contribute a zero
    effect and an inflated variance, and every delta here would be silently wrong.
    """
    model = _model(active_runs={0, 1}, n_runs=2)
    vectors = run_level._contrast_vectors(model, "task")

    dropped_second = model.masker_.transform(
        model.compute_contrast([vectors[0], np.zeros_like(vectors[1])], output_type="stat")
    ).ravel()

    per_run = run_level.compute_run_level_contrast(
        model, "task", run_labels=("run-01", "run-02")
    )
    effect = model.masker_.transform(per_run.effect)[0]
    variance = model.masker_.transform(per_run.variance)[0]
    run_one_alone = effect / np.sqrt(variance)

    np.testing.assert_allclose(dropped_second, run_one_alone, rtol=1e-5, atol=1e-5)


def test_baseline_is_the_stored_map(one_active_run):
    """The all-runs case must be the map the pipeline stores, so deltas reconcile."""
    stored = one_active_run.masker_.transform(
        one_active_run.compute_contrast("task", output_type="z_score")
    ).ravel()
    rows = run_level.compute_run_influence(
        one_active_run, "task", run_labels=("run-01", "run-02", "run-03"), threshold=2.3
    )
    baseline = int(np.sum(np.abs(stored) > 2.3))
    for row in rows:
        assert row.survivors - row.delta == baseline


def test_single_run_model_yields_none():
    assert (
        run_level.compute_run_influence(
            _model(active_runs={0}, n_runs=1), "task", run_labels=("run-01",)
        )
        is None
    )


def test_writer_columns(one_active_run, tmp_path):
    rows = run_level.compute_run_influence(
        one_active_run, "task", run_labels=("run-01", "run-02", "run-03")
    )
    path = run_level.write_run_influence(
        rows, out_dir=tmp_path, stem="sub-01_task-x_contrast-c", cfg_hash="abc123"
    )
    frame = pd.read_csv(path, sep="\t")
    assert list(frame.columns) == [
        "dropped_run",
        "survivors",
        "delta",
        "max_abs_z",
        "correlation",
    ]
    assert len(frame) == 3
