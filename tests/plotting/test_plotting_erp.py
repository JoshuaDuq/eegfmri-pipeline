from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import mne
import numpy as np
import pytest

matplotlib.use("Agg")

import eeg_pipeline.plotting.erp.waveform as erp_waveform


def _make_epochs(
    data: np.ndarray,
    *,
    ch_names: list[str],
    ch_types: list[str],
    tmin: float = -0.2,
    sfreq: float = 10.0,
) -> mne.Epochs:
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return mne.EpochsArray(
        np.array(data, dtype=float, copy=True),
        info,
        tmin=tmin,
        baseline=None,
        verbose=False,
    )


def _erp_config(
    *,
    rois: dict[str, list[str]],
    baseline_window: list[float],
    baseline_correction: bool = True,
    allow_no_baseline: bool = False,
) -> dict:
    return {
        "rois": rois,
        "feature_engineering": {
            "erp": {
                "baseline_correction": baseline_correction,
                "baseline_window": list(baseline_window),
                "allow_no_baseline": allow_no_baseline,
            }
        },
    }


def test_get_baseline_window_rejects_post_stimulus_baseline() -> None:
    config = _erp_config(
        rois={"Frontal": ["Fz"]},
        baseline_window=[-0.2, 0.05],
    )

    with pytest.raises(ValueError, match="stimulus onset"):
        erp_waveform._get_baseline_window(config)


def test_plot_roi_erp_uses_full_epoch_channel_indices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    epochs = _make_epochs(
        np.array(
            [
                [
                    [10.0, 10.0, 10.0, 10.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0],
                ],
                [
                    [20.0, 20.0, 20.0, 20.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0],
                ],
            ]
        ),
        ch_names=["EOG1", "Fz", "Cz"],
        ch_types=["eog", "eeg", "eeg"],
    )
    config = _erp_config(
        rois={"Centro": ["Cz"]},
        baseline_window=[-0.2, 0.0],
        baseline_correction=False,
    )
    captured: dict[str, np.ndarray] = {}

    def _capture_statistics(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        captured["data"] = np.array(data, copy=True)
        n_times = data.shape[-1]
        return np.zeros(n_times), np.zeros(n_times)

    monkeypatch.setattr(erp_waveform, "_compute_roi_waveform_statistics", _capture_statistics)
    monkeypatch.setattr(erp_waveform, "save_fig", lambda *args, **kwargs: None)

    erp_waveform.plot_roi_erp(
        epochs=epochs,
        subject="0001",
        save_dir=tmp_path,
        config=config,
        logger=logging.getLogger("test.plotting.erp.roi_channels"),
    )

    assert "data" in captured
    assert captured["data"].shape == (2, 1, 4)
    assert np.allclose(captured["data"], 2.0)


def test_plot_roi_erp_applies_baseline_correction_before_statistics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    epochs = _make_epochs(
        np.array(
            [
                [[5.0, 5.0, 7.0, 7.0]],
                [[3.0, 3.0, 4.5, 4.5]],
            ]
        ),
        ch_names=["Fz"],
        ch_types=["eeg"],
    )
    config = _erp_config(
        rois={"Frontal": ["Fz"]},
        baseline_window=[-0.2, 0.0],
        baseline_correction=True,
        allow_no_baseline=False,
    )
    captured: dict[str, np.ndarray] = {}

    def _capture_statistics(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        captured["data"] = np.array(data, copy=True)
        n_times = data.shape[-1]
        return np.zeros(n_times), np.zeros(n_times)

    monkeypatch.setattr(erp_waveform, "_compute_roi_waveform_statistics", _capture_statistics)
    monkeypatch.setattr(erp_waveform, "save_fig", lambda *args, **kwargs: None)

    erp_waveform.plot_roi_erp(
        epochs=epochs,
        subject="0001",
        save_dir=tmp_path,
        config=config,
        logger=logging.getLogger("test.plotting.erp.baseline"),
    )

    assert "data" in captured
    baseline_corrected = captured["data"][:, 0, :]
    assert np.allclose(baseline_corrected[:, :2].mean(axis=1), 0.0)
    assert np.allclose(baseline_corrected[:, 2:], [[2.0, 2.0], [1.5, 1.5]])
