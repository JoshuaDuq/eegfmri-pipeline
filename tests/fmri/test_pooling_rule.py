"""Pins the rule nilearn uses to combine runs.

The rule is nilearn's, not ours. ``compute_fixed_effect_contrast`` sums ``Contrast``
objects and scales by 1/n; ``Contrast.__add__`` sums effects and variances, and
``__mul__ = __rmul__`` scales variance by the square of the scalar. The statistic is
therefore invariant to the 1/n and reduces to equal weight per run.

A nilearn release switching to precision weighting would silently invalidate the
sign-flip null, the leave-one-run-out deltas, and three corrected docstrings at once.
This test is what makes that a failure rather than a wrong number.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
import pandas as pd
import pytest

SHAPE = (6, 6, 6)
N_FRAMES = 48
TR = 2.0
ONSETS = np.arange(4, 44, 10).astype(float)


def _two_run_model(seed: int = 0):
    from nilearn.glm.first_level import FirstLevelModel

    rng = np.random.default_rng(seed)
    mask = nib.Nifti1Image(np.ones(SHAPE, dtype=np.uint8), np.eye(4))
    bolds, events = [], []
    for run in range(2):
        # Unequal noise between runs, so equal-weight and precision-weighted
        # combinations give visibly different answers.
        scale = 1.0 if run == 0 else 4.0
        data = 100 + (scale * rng.standard_normal(SHAPE + (N_FRAMES,))).astype(np.float32)
        for onset in ONSETS:
            frame = int(onset / TR)
            data[1:4, 1:4, 1:4, frame : frame + 3] += 6.0
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
def model():
    return _two_run_model()


def test_multirun_contrast_is_equal_weight(model):
    """t = sum(effect) / sqrt(sum(variance)), not the precision-weighted combination."""
    from fmri_pipeline.analysis.run_level import compute_run_level_contrast

    stored = model.compute_contrast("task", output_type="stat")
    stored_t = model.masker_.transform(stored).ravel()

    per_run = compute_run_level_contrast(model, "task", run_labels=("run-01", "run-02"))
    effect = model.masker_.transform(per_run.effect)  # (2, V)
    variance = model.masker_.transform(per_run.variance)

    equal_weight = effect.sum(0) / np.sqrt(variance.sum(0))
    np.testing.assert_allclose(equal_weight, stored_t, rtol=1e-5, atol=1e-5)


def test_precision_weighting_is_not_the_rule(model):
    """The two rules must be distinguishable on this fixture.

    Without this, the test above would also pass under precision weighting whenever
    the runs happen to have equal variance, and would stop being a pin.
    """
    from fmri_pipeline.analysis.run_level import compute_run_level_contrast

    stored_t = model.masker_.transform(
        model.compute_contrast("task", output_type="stat")
    ).ravel()
    per_run = compute_run_level_contrast(model, "task", run_labels=("run-01", "run-02"))
    effect = model.masker_.transform(per_run.effect)
    variance = model.masker_.transform(per_run.variance)

    weights = 1.0 / variance
    precision_weighted = (weights * effect).sum(0) / np.sqrt(weights.sum(0))
    assert not np.allclose(precision_weighted, stored_t, rtol=1e-3, atol=1e-3)
