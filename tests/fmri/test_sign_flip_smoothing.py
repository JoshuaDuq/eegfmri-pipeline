"""The sign-flip null must describe the map the report thresholds.

``FirstLevelModel(smoothing_fwhm=...)`` hands that smoothing to its own masker, so
``masker.transform`` applies it a second time to anything passed through it. The null
was enumerated that way, which put its height, its survivor count and its observed
maximum on a doubly-smoothed scale while every other row of the threshold table was on
the model's own.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.run_level import compute_sign_flip_null


@pytest.fixture(scope="module")
def smoothed_model():
    from nilearn.glm.first_level import FirstLevelModel

    rng = np.random.default_rng(0)
    shape, n_scans, t_r = (10, 10, 10), 60, 2.0
    mask = nib.Nifti1Image(np.ones(shape, dtype=np.uint8), np.eye(4))
    bolds, events = [], []
    for _ in range(3):
        data = rng.normal(100.0, 1.0, shape + (n_scans,))
        onsets = np.arange(6.0, n_scans * t_r - 10.0, 18.0)
        for onset in onsets:
            start = int(onset / t_r)
            data[3:6, 3:6, 3:6, start : start + 3] += 6.0
        bolds.append(nib.Nifti1Image(data, np.eye(4)))
        events.append(pd.DataFrame({"onset": onsets, "duration": 5.0, "trial_type": "task"}))

    model = FirstLevelModel(
        t_r=t_r, mask_img=mask, smoothing_fwhm=5.0, minimize_memory=False, standardize=False
    )
    model.fit(bolds, events=events)
    return model


def test_the_masker_would_smooth_a_second_time(smoothed_model) -> None:
    # The premise of the bug, asserted so the test says why it exists.
    assert smoothed_model.masker_.smoothing_fwhm == 5.0


def test_the_identity_pattern_reproduces_the_stored_map(smoothed_model) -> None:
    stored = smoothed_model.compute_contrast(["task"] * 3, output_type="z_score")
    mask = np.asanyarray(smoothed_model.masker_.mask_img_.dataobj).astype(bool)
    expected = float(np.abs(np.asanyarray(stored.dataobj)[mask]).max())

    null = compute_sign_flip_null(smoothed_model, "task")
    assert null is not None
    assert null.observed_max == pytest.approx(expected, rel=1e-6), (
        f"identity pattern gives {null.observed_max:.4f} against a stored map maximum "
        f"of {expected:.4f}"
    )


def test_survivors_are_counted_on_that_same_map(smoothed_model) -> None:
    stored = smoothed_model.compute_contrast(["task"] * 3, output_type="z_score")
    mask = np.asanyarray(smoothed_model.masker_.mask_img_.dataobj).astype(bool)
    values = np.asanyarray(stored.dataobj)[mask]

    null = compute_sign_flip_null(smoothed_model, "task")
    assert null is not None
    assert null.fwe_survivors == int(np.count_nonzero(np.abs(values) > null.fwe_height))
