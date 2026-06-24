from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.types import PrecomputedData, TimeWindows


def _valid_precomputed(**overrides: object) -> PrecomputedData:
    times = np.arange(5, dtype=float) / 100.0
    defaults: dict[str, object] = {
        "data": np.zeros((2, 3, times.size), dtype=float),
        "times": times,
        "sfreq": 100.0,
        "ch_names": ["C3", "C4", "Pz"],
        "picks": np.arange(3),
        "windows": TimeWindows(
            masks={"active": np.ones(times.size, dtype=bool)},
            ranges={"active": (0.0, 0.05)},
            times=times,
            name="active",
        ),
    }
    defaults.update(overrides)
    return PrecomputedData(**defaults)


def test_precomputed_data_rejects_non_three_dimensional_data() -> None:
    with pytest.raises(ValueError, match="epochs, channels, times"):
        _valid_precomputed(data=np.zeros((3, 5), dtype=float))


def test_precomputed_data_rejects_time_axis_mismatch() -> None:
    with pytest.raises(ValueError, match="times length"):
        _valid_precomputed(times=np.arange(4, dtype=float) / 100.0)


def test_precomputed_data_rejects_non_increasing_times() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _valid_precomputed(times=np.array([0.0, 0.01, 0.01, 0.03, 0.04]))


def test_precomputed_data_rejects_nonfinite_times() -> None:
    with pytest.raises(ValueError, match="finite"):
        _valid_precomputed(times=np.array([0.0, 0.01, np.nan, 0.03, 0.04]))


def test_precomputed_data_rejects_nonpositive_sampling_rate() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        _valid_precomputed(sfreq=0.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ch_names": ["C3", "C4"]}, "channel names"),
        ({"picks": np.arange(2)}, "picks"),
    ],
)
def test_precomputed_data_rejects_channel_metadata_mismatch(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _valid_precomputed(**overrides)


def test_precomputed_data_rejects_metadata_trial_mismatch() -> None:
    with pytest.raises(ValueError, match="metadata rows"):
        _valid_precomputed(metadata=pd.DataFrame({"trial": [1]}))


def test_precomputed_data_rejects_condition_label_mismatch() -> None:
    with pytest.raises(ValueError, match="condition_labels length"):
        _valid_precomputed(condition_labels=np.array(["pain"], dtype=object))


def test_precomputed_data_rejects_train_mask_mismatch() -> None:
    with pytest.raises(ValueError, match="train_mask length"):
        _valid_precomputed(train_mask=np.array([True], dtype=bool))


def test_precomputed_data_rejects_window_mask_mismatch() -> None:
    times = np.arange(5, dtype=float) / 100.0
    windows = TimeWindows(
        masks={"active": np.ones(4, dtype=bool)},
        ranges={"active": (0.0, 0.05)},
        times=times,
        name="active",
    )

    with pytest.raises(ValueError, match="window mask 'active' length"):
        _valid_precomputed(windows=windows)


def test_precomputed_crop_rejects_nonoverlapping_range() -> None:
    precomputed = _valid_precomputed()

    with pytest.raises(ValueError, match="does not overlap"):
        precomputed.crop(10.0, 11.0)


def test_precomputed_crop_rejects_reversed_range() -> None:
    precomputed = _valid_precomputed()

    with pytest.raises(ValueError, match="tmax must be greater"):
        precomputed.crop(0.04, 0.01)
