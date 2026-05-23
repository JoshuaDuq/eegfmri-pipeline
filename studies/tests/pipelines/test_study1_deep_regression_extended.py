"""Tests for studies.pain_study.study1.deep_regression model, bands, and training edge cases."""

from __future__ import annotations


import numpy as np
import pytest

from studies.tests.test_support import DotConfig


###################################################################
# build_band_regressor architecture
###################################################################


def test_build_band_regressor_produces_correct_output_shape() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from studies.pain_study.study1.deep_regression.model import build_band_regressor

    cfg = DotConfig(
        {
            "study1": {
                "deep_regression": {
                    "temporal_filters": 8,
                    "dropout": 0.25,
                    "temporal_kernel_size": 15,
                }
            }
        }
    )

    model = build_band_regressor(nn=nn, input_shape=(2, 3, 50), config=cfg)
    x = torch.randn(4, 2, 3, 50)
    out = model(x)

    assert out.shape == (4,)


def test_build_band_regressor_clamps_kernel_to_odd() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from studies.pain_study.study1.deep_regression.model import build_band_regressor

    cfg = DotConfig(
        {
            "study1": {
                "deep_regression": {
                    "temporal_filters": 4,
                    "dropout": 0.1,
                    "temporal_kernel_size": 100,
                }
            }
        }
    )

    model = build_band_regressor(nn=nn, input_shape=(1, 2, 10), config=cfg)
    x = torch.randn(2, 1, 2, 10)
    out = model(x)

    assert out.shape == (2,)


###################################################################
# _safe_r and _safe_r2 edge cases
###################################################################


def test_safe_r_returns_nan_for_constant_prediction() -> None:
    from studies.pain_study.study1.deep_regression.training import _safe_r

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([5.0, 5.0, 5.0])

    assert np.isnan(_safe_r(y_true, y_pred))


def test_safe_r_returns_nan_for_single_sample() -> None:
    from studies.pain_study.study1.deep_regression.training import _safe_r

    assert np.isnan(_safe_r(np.array([1.0]), np.array([1.0])))


def test_safe_r2_returns_nan_for_zero_ss_tot() -> None:
    from studies.pain_study.study1.deep_regression.training import _safe_r2

    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([5.0, 5.0, 5.0])
    y_train_mean = 5.0

    assert np.isnan(_safe_r2(y_true, y_pred, y_train_mean))


def test_safe_r2_returns_valid_value_for_perfect_prediction() -> None:
    from studies.pain_study.study1.deep_regression.training import _safe_r2

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    y_train_mean = 2.0

    assert _safe_r2(y_true, y_pred, y_train_mean) == pytest.approx(1.0)


###################################################################
# _validation_indices edge cases
###################################################################


def test_validation_indices_returns_all_train_when_fraction_zero() -> None:
    from studies.pain_study.study1.deep_regression.training import _validation_indices

    groups = np.array(["sub-0001"] * 5 + ["sub-0002"] * 5, dtype=object)
    train_idx, val_idx = _validation_indices(groups, seed=42, fraction=0.0)

    assert len(train_idx) == 10
    assert len(val_idx) == 0


def test_validation_indices_returns_all_train_for_single_group() -> None:
    from studies.pain_study.study1.deep_regression.training import _validation_indices

    groups = np.array(["sub-0001"] * 5, dtype=object)
    train_idx, val_idx = _validation_indices(groups, seed=42, fraction=0.2)

    assert len(train_idx) == 5
    assert len(val_idx) == 0


def test_validation_indices_splits_multiple_groups() -> None:
    from studies.pain_study.study1.deep_regression.training import _validation_indices

    groups = np.array(
        ["sub-0001"] * 3 + ["sub-0002"] * 3 + ["sub-0003"] * 3,
        dtype=object,
    )
    train_idx, val_idx = _validation_indices(groups, seed=42, fraction=0.3)

    assert len(train_idx) + len(val_idx) == 9
    assert len(val_idx) > 0
    assert set(train_idx) & set(val_idx) == set()


###################################################################
# _validate_bands
###################################################################


def test_validate_bands_rejects_unknown_band() -> None:
    from studies.pain_study.study1.deep_regression.bands import _validate_bands

    cfg = DotConfig(
        {"time_frequency_analysis": {"bands": {"alpha": [8.0, 13.0]}}}
    )

    with pytest.raises(ValueError, match="Unknown frequency bands"):
        _validate_bands(cfg, ["alpha", "theta"])


def test_validate_bands_rejects_empty_list() -> None:
    from studies.pain_study.study1.deep_regression.bands import _validate_bands

    cfg = DotConfig(
        {"time_frequency_analysis": {"bands": {"alpha": [8.0, 13.0]}}}
    )

    with pytest.raises(ValueError, match="at least one"):
        _validate_bands(cfg, [])


###################################################################
# _deep_regression_time_window
###################################################################


def test_deep_regression_time_window_returns_none_when_not_configured() -> None:
    import mne

    from studies.pain_study.study1.deep_regression.bands import _deep_regression_time_window

    info = mne.create_info(["Cz"], sfreq=100.0, ch_types="eeg")
    epochs = mne.EpochsArray(
        np.ones((2, 1, 100), dtype=float), info, tmin=0.0, verbose=False
    )
    cfg = DotConfig({"study1": {"deep_regression": {}}})

    assert _deep_regression_time_window(epochs, cfg) is None


def test_deep_regression_time_window_rejects_inverted_bounds() -> None:
    import mne

    from studies.pain_study.study1.deep_regression.bands import _deep_regression_time_window

    info = mne.create_info(["Cz"], sfreq=100.0, ch_types="eeg")
    epochs = mne.EpochsArray(
        np.ones((2, 1, 100), dtype=float), info, tmin=0.0, verbose=False
    )
    cfg = DotConfig(
        {"study1": {"deep_regression": {"time_window": [0.5, 0.2]}}}
    )

    with pytest.raises(ValueError, match="start < end"):
        _deep_regression_time_window(epochs, cfg)


def test_deep_regression_time_window_rejects_non_overlapping_range() -> None:
    import mne

    from studies.pain_study.study1.deep_regression.bands import _deep_regression_time_window

    info = mne.create_info(["Cz"], sfreq=100.0, ch_types="eeg")
    epochs = mne.EpochsArray(
        np.ones((2, 1, 100), dtype=float), info, tmin=0.0, verbose=False
    )
    cfg = DotConfig(
        {"study1": {"deep_regression": {"time_window": [5.0, 10.0]}}}
    )

    with pytest.raises(ValueError, match="does not overlap"):
        _deep_regression_time_window(epochs, cfg)


###################################################################
# _standardize_train_test
###################################################################


def test_standardize_train_test_normalizes_to_zero_mean() -> None:
    from studies.pain_study.study1.deep_regression.training import _standardize_train_test

    X_train = np.array([[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]])
    X_test = np.array([[[[7.0, 8.0, 9.0]], [[10.0, 11.0, 12.0]]]])

    X_train_n, X_test_n = _standardize_train_test(X_train, X_test)

    assert np.allclose(X_train_n.mean(axis=(0, 3)), 0.0, atol=1e-6)
