from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd
import pytest

from studies.tests.test_support import DotConfig


def _config(root: Path) -> DotConfig:
    return DotConfig(
        {
            "paths": {"deriv_root": str(root / "derivatives")},
            "project": {"random_state": 11},
            "time_frequency_analysis": {
                "bands": {
                    "alpha": [8.0, 12.9],
                    "beta": [13.0, 30.0],
                    "gamma": [30.1, 80.0],
                }
            },
            "study1": {
                "outputs": {"root_name": "study1"},
                "cohort": {"min_subjects": 2},
                "targets": {"names": ["NPS", "SIIPS1"]},
                "deep_regression": {
                    "presets": {
                        "alpha": ["alpha"],
                        "alpha_beta": ["alpha", "beta"],
                    }
                },
            },
        }
    )


def _epochs_and_events() -> tuple[mne.EpochsArray, pd.DataFrame]:
    info = mne.create_info(["Cz", "Pz", "Fz"], sfreq=100.0, ch_types="eeg")
    data = np.arange(2 * 3 * 10, dtype=float).reshape(2, 3, 10)
    epochs = mne.EpochsArray(data, info, tmin=0.0, verbose=False)
    events = pd.DataFrame(
        {
            "run_id": [1, 1],
            "trial_number": [1, 2],
            "onset": [1.0, 2.0],
            "duration": [0.5, 0.5],
        }
    )
    return epochs, events


def _subject_targets(
    subject_id: str,
    target_name: str,
    *,
    finite: bool = True,
    nuisance: bool = False,
) -> pd.DataFrame:
    values = [1.0, 2.0] if target_name == "NPS" else [2.0, 3.0]
    if not finite:
        values[1] = np.nan
    frame = pd.DataFrame(
        {
            "subject_id": [subject_id, subject_id],
            "task": ["pain", "pain"],
            "block": [1, 1],
            "trial_index": [1, 2],
            "onset": [1.0, 2.0],
            "duration": [0.5, 0.5],
            target_name: values,
        }
    )
    if nuisance:
        frame["pain_binary_coded"] = [0, 1]
        frame["stimulus_temp"] = [44.0, 46.0]
    return frame


def test_load_band_tensor_matrix_builds_subject_grouped_tensor(tmp_path) -> None:
    from studies.pain_study.study1.deep_regression.dataset import load_band_tensor_matrix

    cfg = _config(tmp_path)
    epochs, events = _epochs_and_events()

    with (
        patch(
            "studies.pain_study.study1.deep_regression.dataset.resolve_primary_subjects",
            return_value=["sub-0001", "sub-0002"],
        ),
        patch(
            "studies.pain_study.study1.deep_regression.dataset.load_epochs_for_analysis",
            side_effect=[(epochs, events), (epochs, events)],
        ),
        patch(
            "studies.pain_study.study1.deep_regression.dataset.subject_target_rows",
            side_effect=[
                _subject_targets("sub-0001", "NPS"),
                _subject_targets("sub-0002", "NPS"),
            ],
        ),
        patch(
            "studies.pain_study.study1.deep_regression.dataset.build_band_tensor",
            side_effect=[
                np.ones((2, 2, 3, 10), dtype=float),
                np.full((2, 2, 3, 10), 2.0, dtype=float),
            ],
        ),
    ):
        X, y, groups, channels, meta = load_band_tensor_matrix(
            subjects=["0002", "0001"],
            task="pain",
            config=cfg,
            target_name="NPS",
            bands=["alpha", "beta"],
            logger=logging.getLogger(__name__),
        )

    assert X.shape == (4, 2, 3, 10)
    assert list(y) == [1.0, 2.0, 1.0, 2.0]
    assert list(groups) == ["sub-0001", "sub-0001", "sub-0002", "sub-0002"]
    assert channels == ["Cz", "Fz", "Pz"]
    assert list(meta["target_name"]) == ["NPS", "NPS", "NPS", "NPS"]


def test_load_band_tensor_matrix_carries_target_table_nuisance_columns(tmp_path) -> None:
    from studies.pain_study.study1.deep_regression.dataset import load_band_tensor_matrix

    cfg = _config(tmp_path)
    epochs, events = _epochs_and_events()

    with (
        patch(
            "studies.pain_study.study1.deep_regression.dataset.resolve_primary_subjects",
            return_value=["sub-0001", "sub-0002"],
        ),
        patch(
            "studies.pain_study.study1.deep_regression.dataset.load_epochs_for_analysis",
            side_effect=[(epochs, events), (epochs, events)],
        ),
        patch(
            "studies.pain_study.study1.deep_regression.dataset.subject_target_rows",
            side_effect=[
                _subject_targets("sub-0001", "NPS", nuisance=True),
                _subject_targets("sub-0002", "NPS", nuisance=True),
            ],
        ),
        patch(
            "studies.pain_study.study1.deep_regression.dataset.build_band_tensor",
            return_value=np.ones((2, 1, 3, 10), dtype=float),
        ),
    ):
        _X, _y, _groups, _channels, meta = load_band_tensor_matrix(
            subjects=["0001", "0002"],
            task="pain",
            config=cfg,
            target_name="NPS",
            bands=["alpha"],
            logger=logging.getLogger(__name__),
        )

    assert list(meta["pain_binary_coded"]) == [0, 1, 0, 1]
    assert list(meta["stimulus_temp"]) == [44.0, 46.0, 44.0, 46.0]


def test_deep_target_alignment_rejects_conflicting_duplicate_keys() -> None:
    from studies.pain_study.study1.deep_regression.dataset import _align_subject_targets

    aligned_events = pd.DataFrame(
        {
            "run_id": [1],
            "trial_number": [1],
            "onset": [1.0],
            "duration": [0.5],
        }
    )
    target_rows = pd.DataFrame(
        {
            "block": [1, 1],
            "trial_index": [1, 1],
            "onset": [1.0, 1.0],
            "duration": [0.5, 0.5],
            "NPS": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        _align_subject_targets(
            aligned_events=aligned_events,
            target_rows=target_rows,
            target_name="NPS",
        )


def test_load_band_tensor_matrix_rejects_non_finite_targets(tmp_path) -> None:
    from studies.pain_study.study1.deep_regression.dataset import load_band_tensor_matrix

    cfg = _config(tmp_path)
    epochs, events = _epochs_and_events()

    with (
        patch(
            "studies.pain_study.study1.deep_regression.dataset.resolve_primary_subjects",
            return_value=["sub-0001", "sub-0002"],
        ),
        patch(
            "studies.pain_study.study1.deep_regression.dataset.load_epochs_for_analysis",
            side_effect=[(epochs, events), (epochs, events)],
        ),
        patch(
            "studies.pain_study.study1.deep_regression.dataset.subject_target_rows",
            side_effect=[
                _subject_targets("sub-0001", "NPS", finite=False),
                _subject_targets("sub-0002", "NPS"),
            ],
        ),
        patch(
            "studies.pain_study.study1.deep_regression.dataset.build_band_tensor",
            return_value=np.ones((2, 1, 3, 10), dtype=float),
        ),
    ):
        with pytest.raises(ValueError, match="finite"):
            load_band_tensor_matrix(
                subjects=["0001", "0002"],
                task="pain",
                config=cfg,
                target_name="NPS",
                bands=["alpha"],
                logger=logging.getLogger(__name__),
            )


def test_run_loso_deep_regression_validates_shape_and_subject_folds(tmp_path) -> None:
    from studies.pain_study.study1.deep_regression.training import run_loso_deep_regression

    cfg = _config(tmp_path)
    meta = pd.DataFrame({"subject_id": ["sub-0001", "sub-0001", "sub-0002", "sub-0002"]})

    with pytest.raises(ValueError, match="4D"):
        run_loso_deep_regression(
            X=np.ones((4, 3, 10), dtype=float),
            y=np.ones(4, dtype=float),
            groups=np.asarray(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object),
            meta=meta,
            target_name="NPS",
            preset_name="alpha",
            bands=["alpha"],
            config=cfg,
            logger=logging.getLogger(__name__),
        )

    X = np.ones((4, 1, 3, 10), dtype=float)
    y = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float)
    groups = np.asarray(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
    meta = pd.DataFrame(
        {
            "subject_id": groups,
            "task": ["pain"] * 4,
            "block": [1, 1, 1, 1],
            "trial_index": [1, 2, 1, 2],
            "onset": [1.0, 2.0, 1.0, 2.0],
            "duration": [0.5, 0.5, 0.5, 0.5],
            "target_name": ["NPS"] * 4,
            "target_value": y,
        }
    )

    prediction_stream = [
        np.asarray([2.9, 4.1], dtype=float),
        np.asarray([0.9, 2.1], dtype=float),
    ]

    with patch(
        "studies.pain_study.study1.deep_regression.training._fit_regressor",
        side_effect=prediction_stream,
    ):
        result = run_loso_deep_regression(
            X=X,
            y=y,
            groups=groups,
            meta=meta,
            target_name="NPS",
            preset_name="alpha",
            bands=["alpha"],
            config=cfg,
            logger=logging.getLogger(__name__),
        )

    assert list(result.fold_metrics["test_subject"]) == ["sub-0001", "sub-0002"]
    assert list(result.predictions["y_pred"]) == [2.9, 4.1, 0.9, 2.1]
    assert result.summary["n_folds"] == 2


def test_run_loso_deep_regression_uses_foldwise_nuisance_residual_targets(tmp_path) -> None:
    from studies.pain_study.study1.deep_regression.training import run_loso_deep_regression

    cfg = _config(tmp_path)
    cfg["study1"]["targets"] = {
        "names": ["NPS", "SIIPS1"],
        "nuisance_regression": {
            "enabled": True,
            "continuous_columns": ["pain_binary_coded"],
            "categorical_columns": [],
        },
    }
    X = np.ones((6, 1, 3, 10), dtype=float)
    y = np.asarray([100.0, 110.0, 0.0, 10.0, 0.0, 10.0], dtype=float)
    groups = np.asarray(
        ["sub-0001", "sub-0001", "sub-0002", "sub-0002", "sub-0003", "sub-0003"],
        dtype=object,
    )
    meta = pd.DataFrame(
        {
            "subject_id": groups,
            "pain_binary_coded": [0, 1, 0, 1, 0, 1],
            "target_value": y,
        }
    )
    y_train_seen: list[np.ndarray] = []

    def _capture_fit(**kwargs):
        y_train_seen.append(np.asarray(kwargs["y_train"], dtype=float))
        return np.zeros(len(kwargs["X_test"]), dtype=float)

    with patch(
        "studies.pain_study.study1.deep_regression.training._fit_regressor",
        side_effect=_capture_fit,
    ):
        result = run_loso_deep_regression(
            X=X,
            y=y,
            groups=groups,
            meta=meta,
            target_name="NPS",
            preset_name="alpha",
            bands=["alpha"],
            config=cfg,
            logger=logging.getLogger(__name__),
        )

    assert np.allclose(y_train_seen[0], [0.0, 0.0, 0.0, 0.0])
    assert np.allclose(result.predictions.loc[:1, "y_true"], [100.0, 100.0])
    assert result.summary["target_residualization"]["columns"] == ["pain_binary_coded"]


def test_target_standardization_round_trips(tmp_path) -> None:
    from studies.pain_study.study1.deep_regression.training import (
        _apply_target_standardization,
        _invert_target_standardization,
        _standardize_train_targets,
    )

    values = np.asarray([100.0, 200.0, 300.0], dtype=float)
    standardized, mean, std = _standardize_train_targets(values)

    assert np.isclose(float(standardized.mean()), 0.0)
    assert np.isclose(float(standardized.std()), 1.0)
    assert np.allclose(
        _invert_target_standardization(standardized, mean=mean, std=std),
        values,
    )
    assert np.allclose(
        _apply_target_standardization(values, mean=mean, std=std),
        standardized,
    )


def test_target_standardization_handles_constant_targets(tmp_path) -> None:
    from studies.pain_study.study1.deep_regression.training import (
        _invert_target_standardization,
        _standardize_train_targets,
    )

    values = np.asarray([5.0, 5.0], dtype=float)
    standardized, mean, std = _standardize_train_targets(values)

    assert mean == 5.0
    assert std == 1.0
    assert np.allclose(standardized, np.zeros_like(values))
    assert np.allclose(
        _invert_target_standardization(standardized, mean=mean, std=std),
        values,
    )


def test_build_band_tensor_crops_to_configured_deep_time_window(tmp_path) -> None:
    from studies.pain_study.study1.deep_regression.bands import build_band_tensor

    info = mne.create_info(["Cz"], sfreq=10.0, ch_types="eeg")
    data = np.arange(10, dtype=float).reshape(1, 1, 10)
    epochs = mne.EpochsArray(data, info, tmin=0.0, verbose=False)
    cfg = DotConfig(
        {
            "time_frequency_analysis": {
                "bands": {"alpha": [1.0, 3.0]},
            },
            "study1": {
                "deep_regression": {
                    "time_window": [0.2, 0.5],
                }
            },
        }
    )

    tensor = build_band_tensor(
        epochs=epochs,
        config=cfg,
        bands=["alpha"],
        channels=["Cz"],
        logger=logging.getLogger(__name__),
    )

    assert tensor.shape == (1, 1, 1, 4)


def test_build_band_tensor_filters_before_cropping_to_configured_time_window(tmp_path) -> None:
    from studies.pain_study.study1.deep_regression.bands import build_band_tensor

    info = mne.create_info(["Cz"], sfreq=10.0, ch_types="eeg")
    data = np.arange(10, dtype=float).reshape(1, 1, 10)
    epochs = mne.EpochsArray(data, info, tmin=0.0, verbose=False)
    cfg = DotConfig(
        {
            "time_frequency_analysis": {"bands": {"alpha": [1.0, 3.0]}},
            "study1": {"deep_regression": {"time_window": [0.2, 0.5]}},
        }
    )
    filtered_time_counts: list[int] = []

    def _capture_filter(self, *args, **kwargs):
        filtered_time_counts.append(len(self.times))
        return self

    with patch("mne.epochs.BaseEpochs.filter", autospec=True, side_effect=_capture_filter):
        build_band_tensor(
            epochs=epochs,
            config=cfg,
            bands=["alpha"],
            channels=["Cz"],
            logger=logging.getLogger(__name__),
        )

    assert filtered_time_counts == [10]


def test_run_deep_regression_writes_one_output_per_target_and_preset(tmp_path) -> None:
    from studies.pain_study.study1.deep_regression.evaluation import run_deep_regression

    cfg = _config(tmp_path)

    fake_result = type(
        "_Result",
        (),
        {
            "predictions": pd.DataFrame({"y_true": [1.0], "y_pred": [1.1]}),
            "fold_metrics": pd.DataFrame({"fold_id": [0], "test_subject": ["sub-0001"]}),
            "summary": {
                "model_name": "band_temporal_regressor",
                "mean_r2": 0.5,
                "mean_mae": 0.1,
                "n_folds": 2,
            },
        },
    )()

    with (
        patch(
            "studies.pain_study.study1.deep_regression.evaluation.resolve_primary_subjects",
            return_value=["sub-0001", "sub-0002"],
        ),
        patch(
            "studies.pain_study.study1.deep_regression.evaluation.load_band_tensor_matrix",
            return_value=(
                np.ones((4, 1, 3, 10), dtype=float),
                np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float),
                np.asarray(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object),
                ["Cz", "Pz", "Fz"],
                pd.DataFrame({"subject_id": ["sub-0001"] * 4}),
            ),
        ),
        patch(
            "studies.pain_study.study1.deep_regression.evaluation.run_loso_deep_regression",
            return_value=fake_result,
        ),
    ):
        outputs = run_deep_regression(
            subjects=["0001", "0002"],
            task="pain",
            config=cfg,
            logger=logging.getLogger(__name__),
        )

    assert len(outputs) == 4
    assert outputs[0].parts[-3:] == ("NPS", "alpha", "summary.json")
