"""The run-level sign-flip null.

Sign-flipping run-level contributions and recombining them through the same pooling
path gives a familywise height that respects the actual between-run variance, which
neither Bonferroni nor an FDR against N(0, 1) does.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
import pandas as pd
import pytest

from fmri_pipeline.analysis import run_level

SHAPE = (6, 6, 6)
N_FRAMES = 48
TR = 2.0
ONSETS = np.arange(4, 44, 10).astype(float)


def _model(n_runs: int, *, effect: float, seed: int = 0):
    from nilearn.glm.first_level import FirstLevelModel

    rng = np.random.default_rng(seed)
    mask = nib.Nifti1Image(np.ones(SHAPE, dtype=np.uint8), np.eye(4))
    bolds, events = [], []
    for _ in range(n_runs):
        data = 100 + rng.standard_normal(SHAPE + (N_FRAMES,)).astype(np.float32)
        if effect:
            for onset in ONSETS:
                frame = int(onset / TR)
                data[1:4, 1:4, 1:4, frame : frame + 3] += effect
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
def three_run_model():
    return _model(3, effect=6.0)


def test_pattern_count_is_two_to_the_n_minus_one(three_run_model):
    """The +/-global pair is redundant for a two-sided max statistic."""
    null = run_level.compute_sign_flip_null(three_run_model, "task")
    assert null.n_patterns == 2 ** (3 - 1) == 4
    assert len(null.null_max) == 4


def test_identity_pattern_reproduces_the_observed_maximum(three_run_model):
    """The all-positive pattern is the real contrast, so its max is the observed max."""
    null = run_level.compute_sign_flip_null(three_run_model, "task")
    assert null.null_max[0] == pytest.approx(null.observed_max)


def test_observed_maximum_matches_the_stored_contrast(three_run_model):
    """The identity must reproduce the map the pipeline stores, not approximate it."""
    stored = three_run_model.masker_.transform(
        three_run_model.compute_contrast("task", output_type="z_score")
    ).ravel()
    null = run_level.compute_sign_flip_null(three_run_model, "task")
    assert null.observed_max == pytest.approx(float(np.abs(stored).max()), rel=1e-9)


def test_p_floor_accounts_for_the_identity_tie(three_run_model):
    """The identity is always in the null set and always ties, so p >= 2/(n+1)."""
    null = run_level.compute_sign_flip_null(three_run_model, "task")
    assert null.p_floor == pytest.approx(2 / (2 ** (3 - 1) + 1))
    assert null.global_p >= null.p_floor


def test_strong_effect_puts_observed_at_the_top_of_the_null():
    null = run_level.compute_sign_flip_null(_model(4, effect=10.0), "task")
    assert null.observed_max >= max(null.null_max)
    assert null.global_p == pytest.approx(null.p_floor)


def test_survivors_are_counted_at_the_fwe_height(three_run_model):
    null = run_level.compute_sign_flip_null(three_run_model, "task")
    stored = three_run_model.masker_.transform(
        three_run_model.compute_contrast("task", output_type="z_score")
    ).ravel()
    assert null.fwe_survivors == int(np.sum(np.abs(stored) >= null.fwe_height))


def test_single_run_model_yields_none():
    """One run has no sign pattern to flip; a degenerate null would be worse than none."""
    assert run_level.compute_sign_flip_null(_model(1, effect=6.0), "task") is None


def test_writer_emits_one_row_per_pattern(three_run_model, tmp_path):
    null = run_level.compute_sign_flip_null(three_run_model, "task")
    path = run_level.write_sign_flip_null(
        null, out_dir=tmp_path, stem="sub-01_task-x_contrast-c", cfg_hash="abc123"
    )
    frame = pd.read_csv(path, sep="\t")
    assert list(frame.columns) == ["pattern", "signs", "max_abs_z"]
    assert len(frame) == null.n_patterns
    assert frame.loc[0, "signs"] == "+++"
