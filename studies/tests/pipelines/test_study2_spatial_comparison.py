from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.study2.config import load_study2_config


def test_spatial_correspondence_tests_observed_r_against_surrogates() -> None:
    from studies.pain_study.study2.spatial_comparison import compute_spatial_correspondence

    fmri_map = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float)
    eeg_map = fmri_map.copy()
    surrogate_maps = np.asarray(
        [
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0, -1.0],
        ],
        dtype=float,
    )

    result = compute_spatial_correspondence(
        eeg_map=eeg_map,
        fmri_map=fmri_map,
        surrogate_maps=surrogate_maps,
        mask=np.asarray([True, True, True, True]),
        config=load_study2_config(),
    )

    assert result.spatial_r == pytest.approx(1.0)
    assert result.p_value == pytest.approx(0.25)
    assert result.meaningful is True


def test_spatial_correspondence_rejects_shape_mismatch() -> None:
    from studies.pain_study.study2.spatial_comparison import compute_spatial_correspondence

    with pytest.raises(ValueError, match="same shape"):
        compute_spatial_correspondence(
            eeg_map=np.ones(3, dtype=float),
            fmri_map=np.ones(4, dtype=float),
            surrogate_maps=np.ones((2, 3), dtype=float),
            mask=np.ones(3, dtype=bool),
            config=load_study2_config(),
        )


def test_spatial_correspondence_requires_analysis_mask() -> None:
    from studies.pain_study.study2.spatial_comparison import compute_spatial_correspondence

    with pytest.raises(ValueError, match="spatial mask"):
        compute_spatial_correspondence(
            eeg_map=np.ones(3, dtype=float),
            fmri_map=np.ones(3, dtype=float),
            surrogate_maps=np.ones((2, 3), dtype=float),
            mask=None,
            config=load_study2_config(),
        )
