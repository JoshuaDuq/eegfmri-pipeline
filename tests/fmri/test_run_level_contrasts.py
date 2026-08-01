"""Per-run estimates read out of an already-fitted first-level model.

A fixed-effects combination across runs is weighted equally per run (pinned by
``tests/fmri/test_pooling_rule.py``), so an effect resting on one run and an effect
present in all of them produce the same map, the same z, and the same cluster table.
These estimates are what tell the two apart -- and they must come out of the fitted
model rather than a refit, or the report cannot afford them.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
import pandas as pd
import pytest

from fmri_pipeline.analysis import run_level


SHAPE = (8, 8, 8)
N_FRAMES = 60
TR = 2.0
ONSETS = np.arange(5, 55, 12).astype(float)
ACTIVE = (3, 3, 3)
NULL = (7, 7, 7)


def _fit(active_runs, n_runs=4, seed=0):
    """Fit a multi-run model where only ``active_runs`` carry the effect."""
    from nilearn.glm.first_level import FirstLevelModel

    rng = np.random.default_rng(seed)
    mask = nib.Nifti1Image(np.ones(SHAPE, dtype=np.uint8), np.eye(4))
    bolds, events = [], []
    for run in range(n_runs):
        data = 100 + rng.standard_normal(SHAPE + (N_FRAMES,)).astype(np.float32)
        if run in active_runs:
            for onset in ONSETS:
                frame = int(onset / TR)
                data[2:5, 2:5, 2:5, frame : frame + 3] += 8.0
        bolds.append(nib.Nifti1Image(data, np.eye(4)))
        events.append(
            pd.DataFrame({"onset": ONSETS, "duration": 4.0, "trial_type": "task"})
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
def split_model():
    """Four runs; the effect lives in the first two only."""
    return _fit(active_runs={0, 1})


# --- reading the estimates out --------------------------------------------


def test_one_estimate_per_run_is_returned(split_model) -> None:
    result = run_level.compute_run_level_contrast(
        split_model, "task", run_labels=["run-01", "run-02", "run-03", "run-04"]
    )
    assert result.n_runs == 4
    assert result.run_labels == ("run-01", "run-02", "run-03", "run-04")
    assert result.effect.shape == SHAPE + (4,)
    assert result.variance.shape == SHAPE + (4,)


def test_the_estimates_recover_which_runs_carry_the_effect(split_model) -> None:
    # The whole point. Runs 1 and 2 saw the stimulus; runs 3 and 4 did not.
    result = run_level.compute_run_level_contrast(split_model, "task")
    effects = np.abs(np.asarray(result.effect.get_fdata())[ACTIVE])
    assert effects[0] > 4.0 and effects[1] > 4.0
    assert effects[2] < 2.0 and effects[3] < 2.0


def test_a_null_voxel_shows_no_effect_in_any_run(split_model) -> None:
    result = run_level.compute_run_level_contrast(split_model, "task")
    effects = np.abs(np.asarray(result.effect.get_fdata())[NULL])
    assert np.all(effects < 4.0)


def test_the_variance_is_non_negative_everywhere(split_model) -> None:
    # It is square-rooted into a standard error downstream.
    result = run_level.compute_run_level_contrast(split_model, "task")
    variance = np.asarray(result.variance.get_fdata())
    assert np.all(variance[np.isfinite(variance)] >= 0.0)


def test_the_combined_estimate_lies_among_the_run_estimates(split_model) -> None:
    # A fixed-effects combination is a precision-weighted average, so it cannot fall
    # outside the range of what the runs themselves estimated.
    result = run_level.compute_run_level_contrast(split_model, "task")
    per_run = np.asarray(result.effect.get_fdata())[ACTIVE]
    combined = float(
        np.asarray(
            split_model.compute_contrast(
                ["task"] * 4, output_type="effect_size"
            ).get_fdata()
        )[ACTIVE]
    )
    assert per_run.min() <= combined <= per_run.max()


def test_labels_fall_back_to_position_when_not_supplied(split_model) -> None:
    result = run_level.compute_run_level_contrast(split_model, "task")
    assert result.run_labels == ("run-01", "run-02", "run-03", "run-04")


def test_a_contrast_vector_is_accepted_as_well_as_an_expression(split_model) -> None:
    columns = split_model.design_matrices_[0].columns.tolist()
    vector = [1.0 if name == "task" else 0.0 for name in columns]
    from_vector = run_level.compute_run_level_contrast(split_model, [vector] * 4)
    from_name = run_level.compute_run_level_contrast(split_model, "task")
    np.testing.assert_allclose(
        np.asarray(from_vector.effect.get_fdata()),
        np.asarray(from_name.effect.get_fdata()),
        rtol=1e-5,
    )


# --- declining -------------------------------------------------------------


def test_a_single_run_model_has_nothing_to_compare() -> None:
    model = _fit(active_runs={0}, n_runs=1)
    assert run_level.compute_run_level_contrast(model, "task") is None


def test_a_model_without_per_run_results_is_declined() -> None:
    # minimize_memory=True discards results_, which is nilearn's default.
    class Bare:
        labels_ = []
        results_ = []
        masker_ = None

    assert run_level.compute_run_level_contrast(Bare(), "task") is None


def test_an_unparseable_contrast_costs_the_diagnostic_and_not_the_run(
    split_model,
) -> None:
    # The contrast this describes is already on disk by the time this runs.
    assert run_level.compute_run_level_contrast(split_model, "no_such_column") is None


# --- writing ---------------------------------------------------------------


def test_the_maps_are_written_as_one_volume_per_quantity(split_model, tmp_path) -> None:
    # Twelve separate files per contrast is the same bytes and eleven more chances
    # for a partial write to leave a contrast half-described.
    result = run_level.compute_run_level_contrast(split_model, "task")
    effect_path, variance_path = run_level.write_run_level_maps(
        result, out_dir=tmp_path, stem="sub-01_task-heat_contrast-x", cfg_hash="abc123"
    )
    assert effect_path.exists() and variance_path.exists()
    assert "desc-perrun" in effect_path.name
    assert nib.load(str(effect_path)).shape == SHAPE + (4,)


def test_the_written_maps_carry_the_run_axis_last(split_model, tmp_path) -> None:
    result = run_level.compute_run_level_contrast(split_model, "task")
    effect_path, _variance = run_level.write_run_level_maps(
        result, out_dir=tmp_path, stem="s", cfg_hash="h"
    )
    written = np.asarray(nib.load(str(effect_path)).get_fdata())
    np.testing.assert_allclose(
        written, np.asarray(result.effect.get_fdata()), rtol=1e-5
    )


# --- the pipeline seam -----------------------------------------------------


def test_the_pipeline_writes_named_run_level_maps(split_model, tmp_path) -> None:
    # Only a real fit exercises this path, so it is checked against one here.
    import logging

    from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

    class _Stub:
        logger = logging.getLogger("test")

    class _GlmResult:
        flm = split_model

    run_meta = {
        "included_bold_paths": [
            f"/data/sub-01_task-heat_run-{i:02d}_bold.nii.gz" for i in (1, 2, 3, 4)
        ]
    }
    effect, variance, labels = FmriAnalysisPipeline._run_level_maps(
        _Stub(),
        glm_result=_GlmResult(),
        contrast_def="task",
        run_meta=run_meta,
        out_dir=tmp_path,
        stem="sub-01_task-heat_contrast-x",
        cfg_hash="abc123",
    )
    assert effect is not None and variance is not None
    assert "desc-perrun" in effect.name and "abc123" in effect.name
    # Run names come from the BIDS entities, so the forest plot's rows match the
    # motion table and the design section.
    assert labels == ["run-01", "run-02", "run-03", "run-04"]
    assert nib.load(str(effect)).shape[3] == 4


def test_a_pipeline_without_a_fitted_model_writes_nothing(tmp_path) -> None:
    import logging

    from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

    class _Stub:
        logger = logging.getLogger("test")

    class _GlmResult:
        flm = None

    assert FmriAnalysisPipeline._run_level_maps(
        _Stub(),
        glm_result=_GlmResult(),
        contrast_def="task",
        run_meta={},
        out_dir=tmp_path,
        stem="s",
        cfg_hash="h",
    ) == (None, None, [])
