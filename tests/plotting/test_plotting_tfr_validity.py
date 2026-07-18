from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.plotting.core.annotations import get_sig_marker_text
from eeg_pipeline.plotting.core.statistics import build_statistical_title, compute_cluster_significance
from eeg_pipeline.plotting.io.figures import logratio_to_pct
from eeg_pipeline.plotting.orchestration import tfr as tfr_orchestration
from eeg_pipeline.plotting.tfr import band_evolution, topomaps
from eeg_pipeline.utils.analysis.stats.cluster import cluster_test_two_sample


matplotlib.use("Agg", force=True)


class _FakeEpochsTFR:
    def __init__(
        self,
        data: np.ndarray,
        *,
        times: np.ndarray,
        freqs: np.ndarray,
        ch_names: list[str],
    ) -> None:
        self.data = np.array(data, dtype=float, copy=True)
        self.times = np.array(times, dtype=float, copy=True)
        self.freqs = np.array(freqs, dtype=float, copy=True)
        self.ch_names = list(ch_names)

    def copy(self) -> "_FakeEpochsTFR":
        return _FakeEpochsTFR(
            self.data,
            times=self.times,
            freqs=self.freqs,
            ch_names=self.ch_names,
        )


def test_build_statistical_title_uses_cluster_alpha_source() -> None:
    config = {
        "plotting.plots.topomap": {"diff_annotation_enabled": True},
        "statistics": {"sig_alpha": 0.01, "fdr_alpha": 0.05, "cluster_n_perm": 2048},
    }

    title = build_statistical_title(
        config=config,
        baseline_used=(-0.5, 0.0),
        paired=False,
        n_trials_condition_2=10,
        n_trials_condition_1=12,
    )

    assert "alpha=0.050" in title


def test_sig_marker_text_uses_cluster_alpha_source() -> None:
    config = {
        "plotting.plots.topomap": {"diff_annotation_enabled": True},
        "statistics": {"sig_alpha": 0.01, "fdr_alpha": 0.05, "cluster_n_perm": 2048},
        "plotting": {"plots": {"topomap": {"diff_annotation_enabled": True}}},
    }

    marker_text = get_sig_marker_text(config)

    assert "p < 0.05" in marker_text
    assert "p < 0.01" not in marker_text


def test_build_statistical_title_describes_cluster_statistic_not_threshold() -> None:
    config = {
        "plotting.plots.topomap": {"diff_annotation_enabled": True},
        "statistics": {"sig_alpha": 0.05, "fdr_alpha": 0.05, "cluster_n_perm": 1024},
    }

    title = build_statistical_title(
        config=config,
        baseline_used=(-0.5, 0.0),
        paired=False,
        n_trials_condition_2=10,
        n_trials_condition_1=12,
    )

    assert "cluster statistic: mass" in title
    assert "cluster threshold: mass-based" not in title


def test_compute_cluster_significance_forwards_paired_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_cluster_test_epochs(*args, **kwargs):
        captured["paired"] = kwargs["paired"]
        return np.array([True, False]), 0.01, 1, 2.5

    monkeypatch.setattr(
        "eeg_pipeline.plotting.core.statistics.cluster_test_epochs",
        fake_cluster_test_epochs,
    )

    sig_mask, cluster_p_min, cluster_k, cluster_mass = compute_cluster_significance(
        tfr=object(),
        mask1=np.array([True, False, True]),
        mask2=np.array([False, True, False]),
        fmin=8.0,
        fmax_eff=12.0,
        tmin=0.0,
        tmax=0.5,
        config={"statistics": {"cluster_n_perm": 128, "sig_alpha": 0.05, "fdr_alpha": 0.05}},
        diff_data_len=2,
        paired=True,
    )

    assert bool(captured["paired"]) is True
    assert np.array_equal(sig_mask, np.array([True, False]))
    assert cluster_p_min == pytest.approx(0.01)
    assert cluster_k == 1
    assert cluster_mass == pytest.approx(2.5)


def test_cluster_test_two_sample_rejects_mismatched_paired_inputs() -> None:
    info = mne.create_info(ch_names=["Cz", "Pz"], sfreq=100.0, ch_types=["eeg", "eeg"])
    group_a = np.array([[1.0, 2.0], [1.5, 2.5], [1.2, 2.2]], dtype=float)
    group_b = np.array([[0.5, 1.5], [0.8, 1.8]], dtype=float)
    config = {"statistics": {"cluster_n_perm": 128, "cluster_n_jobs": 1, "sig_alpha": 0.05, "fdr_alpha": 0.05}}

    with pytest.raises(ValueError, match="paired cluster test requires equal sample counts"):
        cluster_test_two_sample(
            group_a=group_a,
            group_b=group_b,
            info=info,
            paired=True,
            config=config,
        )


def test_parallel_tfr_worker_propagates_subject_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_visualize_subject_tfr(*args, **kwargs) -> None:
        raise ValueError("baseline window missing")

    monkeypatch.setattr(
        tfr_orchestration,
        "visualize_subject_tfr",
        fake_visualize_subject_tfr,
    )

    with pytest.raises(ValueError, match="baseline window missing"):
        tfr_orchestration._visualize_single_subject(
            subject="0001",
            task="thermal",
            config={},
            tfr_roi_only=False,
            tfr_topomaps_only=False,
            plots=None,
            deriv_root=Path("."),
        )


def test_plot_single_topomap_window_passes_paired_to_statistical_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_compute_statistical_mask(*args, **kwargs):
        captured["paired"] = kwargs["paired"]
        return None, None, None, None

    monkeypatch.setattr(topomaps, "_compute_statistical_mask", fake_compute_statistical_mask)
    monkeypatch.setattr(topomaps, "plot_topomap_on_ax", lambda *args, **kwargs: None)
    monkeypatch.setattr(topomaps, "add_roi_annotations", lambda *args, **kwargs: None)

    fig, ax = plt.subplots()
    try:
        topomaps._plot_single_topomap_window(
            ax=ax,
            diff_data=np.array([0.1, -0.2]),
            info=object(),
            tfr_sub=None,
            condition_mask_a=np.array([True, False]),
            condition_mask_b=np.array([False, True]),
            fmin=8.0,
            fmax_eff=12.0,
            tmin_win=0.0,
            tmax_win=0.5,
            vabs_diff=1.0,
            config={"plotting": {"plots": {"topomap": {"diff_annotation_enabled": True}}}},
            viz_params={"diff_annotation_enabled": True, "sig_mask_params": {}},
            paired=True,
        )
    finally:
        plt.close(fig)

    assert bool(captured["paired"]) is True


def test_band_power_evolution_raises_when_shared_baseline_application_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_apply_baseline_and_crop(*args, **kwargs):
        raise ValueError("baseline window missing")

    monkeypatch.setattr(band_evolution, "apply_baseline_and_crop", fake_apply_baseline_and_crop)

    tfr = _FakeEpochsTFR(
        data=np.ones((2, 1, 2, 3), dtype=float),
        times=np.array([-0.5, 0.0, 0.5], dtype=float),
        freqs=np.array([8.0, 10.0], dtype=float),
        ch_names=["Cz"],
    )
    events_df = pd.DataFrame({"trial_type": ["A", "B"]})
    config = {
        "time_frequency_analysis": {"selected_bands": ["alpha"]},
        "plotting": {"formats": ["png"]},
        "feature_engineering": {"task_is_rest": True},
    }

    with pytest.raises(ValueError, match="baseline window missing"):
        band_evolution.plot_band_power_evolution_all_conditions(
            tfr=tfr,
            events_df=events_df,
            save_dir=tmp_path,
            config=config,
            logger=logging.getLogger("test.band_evolution.baseline"),
        )


def test_band_power_evolution_uses_shared_logratio_baseline_for_percent_display(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, np.ndarray] = {}

    def fake_apply_baseline_and_crop(tfr_obj, baseline, **kwargs):
        del baseline, kwargs
        tfr_obj.data = np.array(
            [[[[0.0, np.log10(2.0)], [0.0, np.log10(2.0)]]]],
            dtype=float,
        )
        return (-0.5, 0.0)

    def fake_plot_mean_sem(ax, times, mean_power, sem_power, color, linewidth=2.0, fill_alpha=0.3):
        del ax, times, color, linewidth, fill_alpha
        captured["mean_power"] = np.array(mean_power, dtype=float, copy=True)
        captured["sem_power"] = np.array(sem_power, dtype=float, copy=True)

    monkeypatch.setattr(band_evolution, "apply_baseline_and_crop", fake_apply_baseline_and_crop)
    monkeypatch.setattr(band_evolution, "_plot_mean_sem_with_reference_lines", fake_plot_mean_sem)
    monkeypatch.setattr(
        band_evolution,
        "save_fig",
        lambda *args, **kwargs: None,
    )

    tfr = _FakeEpochsTFR(
        data=np.ones((1, 1, 2, 2), dtype=float),
        times=np.array([-0.5, 0.5], dtype=float),
        freqs=np.array([8.0, 10.0], dtype=float),
        ch_names=["Cz"],
    )
    events_df = pd.DataFrame({"trial_id": [1]})
    config = {
        "time_frequency_analysis": {
            "selected_bands": ["alpha"],
            "tfr": {},
            "baseline_window": [-0.5, 0.0],
        },
        "plotting": {"formats": ["png"]},
        "feature_engineering": {"task_is_rest": True},
    }

    band_evolution.plot_band_power_evolution_all_conditions(
        tfr=tfr,
        events_df=events_df,
        save_dir=tmp_path,
        config=config,
        logger=logging.getLogger("test.band_evolution.logratio"),
    )

    expected = logratio_to_pct(np.array([0.0, np.log10(2.0)], dtype=float))
    assert np.allclose(captured["mean_power"], expected)
    assert np.allclose(captured["sem_power"], np.zeros_like(expected))


def test_band_power_evolution_compute_mean_sem_uses_sample_standard_deviation() -> None:
    power_data = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=float,
    )

    mean_power, sem_power = band_evolution._compute_mean_sem(power_data, n_trials=3)

    expected_sem = np.nanstd(power_data, axis=0, ddof=1) / np.sqrt(3.0)
    assert np.allclose(mean_power, np.array([3.0, 4.0], dtype=float))
    assert np.allclose(sem_power, expected_sem)


def test_band_power_summary_uses_sample_standard_deviation_for_sem(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, pd.DataFrame] = {}

    def fake_apply_baseline_and_crop(tfr_obj, baseline, **kwargs):
        del baseline, kwargs
        tfr_obj.data = np.array(
            [
                [[[0.0, np.log10(2.0)], [0.0, np.log10(2.0)]]],
                [[[0.0, np.log10(3.0)], [0.0, np.log10(3.0)]]],
                [[[0.0, np.log10(4.0)], [0.0, np.log10(4.0)]]],
            ],
            dtype=float,
        )
        return (-0.5, 0.0)

    def fake_plot_summary_bars(ax, df, condition_keys, condition_labels, title):
        del ax, condition_keys, condition_labels, title
        captured["df"] = df.copy()

    monkeypatch.setattr(band_evolution, "apply_baseline_and_crop", fake_apply_baseline_and_crop)
    monkeypatch.setattr(band_evolution, "_plot_summary_bars", fake_plot_summary_bars)
    monkeypatch.setattr(band_evolution, "save_fig", lambda *args, **kwargs: None)

    tfr = _FakeEpochsTFR(
        data=np.ones((3, 1, 2, 2), dtype=float),
        times=np.array([-0.5, 0.5], dtype=float),
        freqs=np.array([8.0, 10.0], dtype=float),
        ch_names=["Cz"],
    )
    events_df = pd.DataFrame({"condition": ["A", "A", "A"]})
    config = {
        "time_frequency_analysis": {
            "selected_bands": ["alpha"],
            "tfr": {},
            "baseline_window": [-0.5, 0.0],
        },
        "plotting": {
            "formats": ["png"],
            "comparisons": {
                "compare_columns": True,
                "comparison_column": "condition",
                "comparison_values": ["B", "A"],
                "comparison_labels": ["B", "A"],
            },
        },
    }

    band_evolution.plot_band_power_summary(
        tfr=tfr,
        events_df=events_df,
        save_dir=tmp_path,
        config=config,
        active_window=(0.0, 0.5),
        logger=logging.getLogger("test.band_evolution.summary_sem"),
    )

    sem_value = float(captured["df"].loc[captured["df"]["condition"] == "condition_2", "sem"].iloc[0])
    means = logratio_to_pct(np.array([np.log10(2.0), np.log10(3.0), np.log10(4.0)], dtype=float))
    expected_sem = float(np.nanstd(means, ddof=1) / np.sqrt(3.0))
    assert sem_value == pytest.approx(expected_sem)


def test_tfr_loading_defers_event_column_validation_to_requested_plotters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    epochs = object()
    events = pd.DataFrame({"trial_id": [1], "condition": ["A"]})

    def fake_load_epochs_for_analysis(*args, **kwargs):
        del args
        captured.update(kwargs)
        return epochs, events

    monkeypatch.setattr(
        tfr_orchestration,
        "load_epochs_for_analysis",
        fake_load_epochs_for_analysis,
    )

    loaded_epochs, loaded_events = tfr_orchestration._load_subject_data(
        "0001",
        "task",
        {},
        tmp_path,
        logging.getLogger("test.tfr.loading"),
    )

    assert loaded_epochs is epochs
    assert loaded_events is events
    assert captured["required_event_groups"] == []
