from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.plotting.features.power import (
    _build_topomap_panel,
    _compute_group_band_summary_stats,
    _compute_column_effect_summary,
    _compute_group_curve_significance_mask,
    _compute_group_paired_effect_forest_data,
    _compute_group_paired_effect_summary,
    _compute_group_paired_sample_count_summary,
    _compute_group_timecourse_significance_mask,
    _compute_paired_effect_matrix,
    _compute_shared_topomap_vlim,
    _compute_window_mean_series,
    _compute_window_effect_summary,
    _draw_curve_significance_strip,
    _draw_group_subject_traces,
    _format_condition_display_label,
    _format_triptych_condition_label,
    _format_topomap_condition_title_label,
)
from eeg_pipeline.plotting.features.utils import (
    _format_count_range,
    _compute_paired_wilcoxon_stats,
    _compute_paired_differences,
    _summarize_multi_window_sample_counts,
    _summarize_paired_sample_counts,
    _plot_single_band_comparison,
)
from eeg_pipeline.utils.analysis.stats.paired_comparisons import compute_paired_cohens_d


class _FakePlotTFR:
    def __init__(self, ch_names: list[str], n_epochs: int) -> None:
        self.ch_names = list(ch_names)
        self._n_epochs = int(n_epochs)

    def __len__(self) -> int:
        return self._n_epochs

    def copy(self) -> "_FakePlotTFR":
        return _FakePlotTFR(self.ch_names, self._n_epochs)

    def pick(self, channels: list[str]) -> "_FakePlotTFR":
        channel_set = set(channels)
        self.ch_names = [ch for ch in self.ch_names if ch in channel_set]
        return self


class _FakeAverageSpectralTFR:
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

    def copy(self) -> "_FakeAverageSpectralTFR":
        return _FakeAverageSpectralTFR(
            self.data,
            times=self.times,
            freqs=self.freqs,
            ch_names=self.ch_names,
        )

    def crop(self, tmin: float, tmax: float) -> "_FakeAverageSpectralTFR":
        time_mask = (self.times >= float(tmin)) & (self.times <= float(tmax))
        self.times = self.times[time_mask]
        self.data = self.data[:, :, time_mask]
        return self


class _FakeEpochsSpectralTFR:
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

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, item) -> "_FakeEpochsSpectralTFR":
        selected = np.array(self.data[item], dtype=float, copy=True)
        if selected.ndim == 3:
            selected = selected[np.newaxis, ...]
        return _FakeEpochsSpectralTFR(
            selected,
            times=self.times,
            freqs=self.freqs,
            ch_names=self.ch_names,
        )

    def copy(self) -> "_FakeEpochsSpectralTFR":
        return _FakeEpochsSpectralTFR(
            self.data,
            times=self.times,
            freqs=self.freqs,
            ch_names=self.ch_names,
        )

    def average(self) -> _FakeAverageSpectralTFR:
        return _FakeAverageSpectralTFR(
            np.nanmean(self.data, axis=0),
            times=self.times,
            freqs=self.freqs,
            ch_names=self.ch_names,
        )

    def crop(self, tmin: float, tmax: float) -> "_FakeEpochsSpectralTFR":
        time_mask = (self.times >= float(tmin)) & (self.times <= float(tmax))
        self.times = self.times[time_mask]
        self.data = self.data[:, :, :, time_mask]
        return self


def _fake_apply_baseline_and_crop(
    tfr_obj,
    baseline,
    crop_window=None,
    mode=None,
    logger=None,
    **kwargs,
):
    del mode, logger, kwargs

    baseline_start, baseline_end = float(baseline[0]), float(baseline[1])
    baseline_mask = (tfr_obj.times >= baseline_start) & (tfr_obj.times <= baseline_end)
    if not baseline_mask.any():
        raise ValueError("Baseline window does not overlap fake TFR times.")

    baseline_power = np.nanmean(tfr_obj.data[..., baseline_mask], axis=-1, keepdims=True)
    tfr_obj.data = np.log10(tfr_obj.data / baseline_power)

    if crop_window is not None:
        tfr_obj.crop(float(crop_window[0]), float(crop_window[1]))

    return baseline


def test_compute_window_effect_summary_returns_roi_band_matrices() -> None:
    power_df = pd.DataFrame(
        {
            "power_baseline_alpha_ch_Fz_logratio": [0.10, 0.12, 0.11, 0.09],
            "power_baseline_alpha_ch_Cz_logratio": [0.08, 0.07, 0.09, 0.08],
            "power_active_alpha_ch_Fz_logratio": [0.35, 0.33, 0.32, 0.31],
            "power_active_alpha_ch_Cz_logratio": [0.28, 0.27, 0.29, 0.26],
            "power_baseline_beta_ch_Fz_logratio": [0.02, 0.03, 0.01, 0.02],
            "power_active_beta_ch_Fz_logratio": [0.05, 0.06, 0.04, 0.05],
        }
    )
    config = {"statistics": {"fdr_alpha": 0.05}}
    bands = ["alpha", "beta"]
    roi_names = ["all", "Frontal"]
    rois = {"Frontal": [r"^Fz$"]}
    all_channels = ["Fz", "Cz"]

    effect_df, qvalue_df = _compute_window_effect_summary(
        power_df=power_df,
        bands=bands,
        segments=["baseline", "active"],
        roi_names=roi_names,
        rois=rois,
        all_channels=all_channels,
        config=config,
    )

    assert list(effect_df.index) == roi_names
    assert list(effect_df.columns) == bands
    assert effect_df.loc["all", "alpha"] > 0
    assert effect_df.loc["Frontal", "alpha"] > 0
    assert np.isfinite(qvalue_df.loc["all", "alpha"])


def test_resolve_power_plot_conditions_returns_rest_mask_without_comparison_config() -> None:
    from eeg_pipeline.plotting.features.power import _resolve_power_plot_conditions

    events_df = pd.DataFrame({"trial_id": [1, 2, 3, 4]})
    config = {"feature_engineering": {"task_is_rest": True}}

    conditions = _resolve_power_plot_conditions(
        events_df=events_df,
        config=config,
        context="test_rest_plot",
    )

    assert len(conditions) == 1
    assert conditions[0][0] == "Rest"
    assert np.array_equal(conditions[0][1], np.array([True, True, True, True]))


def test_compute_column_effect_summary_uses_configured_masks() -> None:
    power_df = pd.DataFrame(
        {
            "power_active_alpha_ch_Fz_logratio": [0.10, 0.11, 0.12, 0.34, 0.35, 0.36],
            "power_active_alpha_ch_Cz_logratio": [0.09, 0.10, 0.11, 0.30, 0.31, 0.32],
            "power_active_beta_ch_Fz_logratio": [0.03, 0.04, 0.03, 0.05, 0.06, 0.05],
        }
    )
    events_df = pd.DataFrame({"condition": [0, 0, 0, 1, 1, 1]})
    config = {
        "statistics": {"fdr_alpha": 0.05},
        "plotting": {
            "comparisons": {
                "compare_columns": True,
                "comparison_column": "condition",
                "comparison_values": [0, 1],
                "comparison_labels": ["Cool", "Hot"],
            }
        },
    }
    bands = ["alpha", "beta"]
    roi_names = ["all", "Frontal"]
    rois = {"Frontal": [r"^Fz$"]}
    all_channels = ["Fz", "Cz"]

    effect_df, qvalue_df, label1, label2 = _compute_column_effect_summary(
        power_df=power_df,
        events_df=events_df,
        bands=bands,
        seg_name="active",
        roi_names=roi_names,
        rois=rois,
        all_channels=all_channels,
        config=config,
    )

    assert (label1, label2) == ("Cool", "Hot")
    assert effect_df.loc["all", "alpha"] > 0
    assert np.isfinite(qvalue_df.loc["all", "alpha"])


def test_compute_group_paired_effect_summary_returns_roi_band_matrices() -> None:
    subject_values = {
        "all": {
            "alpha": {
                "sub-01": {"baseline": 1.0, "plateau": 1.3},
                "sub-02": {"baseline": 0.9, "plateau": 1.2},
                "sub-03": {"baseline": 1.1, "plateau": 1.4},
            },
            "beta": {
                "sub-01": {"baseline": 0.8, "plateau": 0.9},
                "sub-02": {"baseline": 0.7, "plateau": 0.8},
                "sub-03": {"baseline": 0.9, "plateau": 1.0},
            },
        },
        "Frontal": {
            "alpha": {
                "sub-01": {"baseline": 1.1, "plateau": 1.5},
                "sub-02": {"baseline": 1.0, "plateau": 1.4},
                "sub-03": {"baseline": 1.2, "plateau": 1.6},
            }
        },
    }

    effect_df, qvalue_df = _compute_group_paired_effect_summary(
        subject_values=subject_values,
        bands=["alpha", "beta"],
        roi_names=["all", "Frontal"],
        labels=("baseline", "plateau"),
        config={"statistics": {"fdr_alpha": 0.05}},
    )

    assert list(effect_df.index) == ["all", "Frontal"]
    assert list(effect_df.columns) == ["alpha", "beta"]
    assert effect_df.loc["all", "alpha"] > 0
    assert effect_df.loc["Frontal", "alpha"] > 0
    assert np.isfinite(qvalue_df.loc["all", "alpha"])


def test_compute_group_paired_effect_summary_handles_zero_differences() -> None:
    subject_values = {
        "all": {
            "alpha": {
                "sub-01": {"baseline": 1.0, "plateau": 1.0},
                "sub-02": {"baseline": 1.2, "plateau": 1.2},
                "sub-03": {"baseline": 0.8, "plateau": 0.8},
            }
        }
    }

    effect_df, qvalue_df = _compute_group_paired_effect_summary(
        subject_values=subject_values,
        bands=["alpha"],
        roi_names=["all"],
        labels=("baseline", "plateau"),
        config={"statistics": {"fdr_alpha": 0.05}},
    )

    assert effect_df.loc["all", "alpha"] == 0.0
    assert qvalue_df.loc["all", "alpha"] == 1.0


def test_compute_group_paired_sample_count_summary_counts_complete_pairs() -> None:
    subject_values = {
        "all": {
            "alpha": {
                "sub-01": {"baseline": 1.0, "plateau": 1.2},
                "sub-02": {"baseline": 0.9, "plateau": 1.1},
                "sub-03": {"baseline": 1.1},
            },
            "beta": {
                "sub-01": {"baseline": 0.8, "plateau": 0.9},
                "sub-02": {"baseline": 0.7, "plateau": 0.8},
            },
        }
    }

    count_df = _compute_group_paired_sample_count_summary(
        subject_values=subject_values,
        bands=["alpha", "beta"],
        roi_names=["all"],
        labels=("baseline", "plateau"),
    )

    assert count_df.loc["all", "alpha"] == 2
    assert count_df.loc["all", "beta"] == 2


def test_compute_group_paired_effect_forest_data_returns_ci_and_qvalues() -> None:
    subject_values = {
        "all": {
            "alpha": {
                "sub-01": {"baseline": 1.0, "plateau": 1.3},
                "sub-02": {"baseline": 0.9, "plateau": 1.2},
                "sub-03": {"baseline": 1.1, "plateau": 1.4},
            },
            "beta": {
                "sub-01": {"baseline": 0.8, "plateau": 0.9},
                "sub-02": {"baseline": 0.7, "plateau": 0.8},
                "sub-03": {"baseline": 0.9, "plateau": 1.0},
            },
        }
    }

    forest_df = _compute_group_paired_effect_forest_data(
        subject_values=subject_values,
        bands=["alpha", "beta"],
        roi_names=["all"],
        labels=("baseline", "plateau"),
        config={"statistics": {"fdr_alpha": 0.05}},
    )

    assert set(forest_df["band"]) == {"alpha", "beta"}
    assert set(forest_df["roi_name"]) == {"all"}
    assert np.all(np.isfinite(forest_df["effect_size"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(forest_df["q_value"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(forest_df["ci_low"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(forest_df["ci_high"].to_numpy(dtype=float)))


def test_compute_shared_topomap_vlim_uses_descriptive_range_for_positive_data() -> None:
    arrays = [
        np.array([0.08, 0.10, 0.12, 0.14]),
        np.array([0.09, 0.11, 0.13, 0.15]),
    ]

    config = {"visualization": {"robust_vlim": {"min_v": 1e-6}}}
    vmin, vmax = _compute_shared_topomap_vlim(arrays, config=config, symmetric=False)

    assert vmin >= 0
    assert vmax > vmin


def test_format_topomap_condition_title_label_expands_raw_value() -> None:
    config = {
        "plotting": {
            "comparisons": {
                "comparison_column": "pain_binary_coded",
                "comparison_values": [0.0, 1.0],
            }
        }
    }

    assert _format_topomap_condition_title_label("1.0", config) == "pain_binary_coded=1.0"


def test_plot_power_by_condition_allows_column_only_config_without_comparison_windows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from eeg_pipeline.plotting.features import power as power_module

    called: dict[str, object] = {}

    def fake_plot_column_comparison(*args, **kwargs) -> None:
        called["column"] = True

    def fake_compute_column_effect_summary(**kwargs):
        effect_df = pd.DataFrame([[0.5]], index=["all"], columns=["alpha"], dtype=float)
        qvalue_df = pd.DataFrame([[0.01]], index=["all"], columns=["alpha"], dtype=float)
        return effect_df, qvalue_df, "Cool", "Hot"

    def fake_plot_power_effect_summary_heatmap(**kwargs) -> None:
        called["heatmap"] = kwargs["title"]

    monkeypatch.setattr(power_module, "_plot_column_comparison", fake_plot_column_comparison)
    monkeypatch.setattr(power_module, "_compute_column_effect_summary", fake_compute_column_effect_summary)
    monkeypatch.setattr(power_module, "_plot_power_effect_summary_heatmap", fake_plot_power_effect_summary_heatmap)

    power_df = pd.DataFrame(
        {
            "power_active_alpha_ch_Fz_logratio": [0.1, 0.2, 0.3, 0.4],
            "power_active_alpha_ch_Cz_logratio": [0.2, 0.3, 0.4, 0.5],
        }
    )
    events_df = pd.DataFrame({"condition": [0, 0, 1, 1]})
    config = {
        "plotting": {
            "overwrite": True,
            "comparisons": {
                "compare_windows": False,
                "compare_columns": True,
                "comparison_segment": "active",
                "comparison_column": "condition",
                "comparison_values": [0, 1],
                "comparison_labels": ["Cool", "Hot"],
                "comparison_rois": ["all"],
            },
        },
        "feature_engineering": {"frequency_bands": {"alpha": [8.0, 12.0]}},
    }

    power_module.plot_power_by_condition(
        power_df=power_df,
        events_df=events_df,
        subject="01",
        save_dir=tmp_path,
        logger=power_module.logger,
        config=config,
        stats_dir=None,
    )

    assert called["column"] is True
    assert called["heatmap"] == "Power condition effects: Hot - Cool"


def test_plot_power_by_condition_rejects_rest_mode() -> None:
    from eeg_pipeline.plotting.features import power as power_module

    power_df = pd.DataFrame({"power_active_alpha_ch_Fz_logratio": [0.1, 0.2, 0.3]})
    events_df = pd.DataFrame({"trial_id": [1, 2, 3]})
    config = {"feature_engineering": {"task_is_rest": True}}

    with pytest.raises(ValueError, match="resting-state"):
        power_module.plot_power_by_condition(
            power_df=power_df,
            events_df=events_df,
            subject="01",
            save_dir=Path("."),
            logger=power_module.logger,
            config=config,
            stats_dir=None,
        )


def test_plot_band_power_topomaps_ignores_column_spec_when_compare_columns_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from eeg_pipeline.plotting.features import power as power_module

    captured: dict[str, object] = {}

    def fake_single_segment(*args, **kwargs) -> None:
        captured["conditions"] = args[8]
        captured["compare_columns"] = args[9]

    monkeypatch.setattr(power_module, "_plot_band_power_topomaps_single_segment", fake_single_segment)

    pow_df = pd.DataFrame(
        {
            "power_active_alpha_ch_Fz_logratio": [0.1, 0.2, 0.3],
            "power_active_beta_ch_Fz_logratio": [0.4, 0.5, 0.6],
            "power_active_alpha_ch_Cz_logratio": [0.2, 0.3, 0.4],
            "power_active_beta_ch_Cz_logratio": [0.5, 0.6, 0.7],
        }
    )
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq=250.0, ch_types="eeg")
    events_df = pd.DataFrame({"condition": [0, 1, 2]})
    config = {
        "plotting": {
            "overwrite": True,
            "comparisons": {
                "compare_columns": False,
                "comparison_column": "condition",
                "comparison_values": [0, 1, 2],
                "comparison_labels": ["Low", "Mid", "High"],
            },
        }
    }

    power_module.plot_band_power_topomaps(
        pow_df=pow_df,
        epochs_info=info,
        bands=["alpha", "beta"],
        subject="01",
        save_dir=tmp_path,
        logger=power_module.logger,
        config=config,
        segment="active",
        events_df=events_df,
    )

    assert captured["compare_columns"] is False
    assert captured["conditions"] is None


def test_plot_power_spectral_density_allows_rest_mode_without_comparison_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from eeg_pipeline.plotting.features import power as power_module

    captured: dict[str, object] = {}

    monkeypatch.setattr(power_module, "_validate_epochs_tfr", lambda *args, **kwargs: None)

    def fake_plot_psd_by_conditions(tfr_roi, conditions, subject, save_dir, logger, config, **kwargs) -> None:
        captured["labels"] = [label for label, _ in conditions]
        captured["channels"] = list(tfr_roi.ch_names)
        captured["subject"] = subject

    monkeypatch.setattr(power_module, "_plot_psd_by_conditions", fake_plot_psd_by_conditions)

    tfr = _FakePlotTFR(["Fz", "Cz"], n_epochs=4)
    events_df = pd.DataFrame({"trial_id": [1, 2, 3, 4]})
    config = {
        "feature_engineering": {"task_is_rest": True},
        "plotting": {"comparisons": {"comparison_rois": ["all"]}},
    }

    power_module.plot_power_spectral_density(
        tfr=tfr,
        subject="01",
        save_dir=tmp_path,
        logger=SimpleNamespace(debug=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
        events_df=events_df,
        config=config,
    )

    assert captured["labels"] == ["Rest"]
    assert captured["channels"] == ["Fz", "Cz"]
    assert captured["subject"] == "01"


def test_plot_cross_frequency_power_correlation_allows_rest_mode_without_comparison_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from eeg_pipeline.plotting.features import power as power_module

    captured: dict[str, object] = {}

    def fake_save_cross_frequency(*, matrices_by_label, save_path, logger, config, title, footer) -> None:
        captured["labels"] = list(matrices_by_label.keys())
        captured["title"] = title
        captured["save_path"] = save_path

    monkeypatch.setattr(
        power_module,
        "_save_cross_frequency_power_correlation_figure",
        fake_save_cross_frequency,
    )

    power_df = pd.DataFrame(
        {
            "power_active_alpha_ch_Fz_mean": [1.0, 1.2, 1.4, 1.6],
            "power_active_beta_ch_Fz_mean": [0.5, 0.7, 0.9, 1.1],
        }
    )
    events_df = pd.DataFrame({"trial_id": [1, 2, 3, 4]})
    config = {
        "feature_engineering": {"task_is_rest": True},
        "plotting": {"comparisons": {"comparison_segment": "active", "comparison_rois": ["all"]}},
    }

    saved_files = power_module.plot_cross_frequency_power_correlation(
        power_df=power_df,
        events_df=events_df,
        subject="01",
        save_dir=tmp_path,
        logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
        config=config,
    )

    assert captured["labels"] == ["Rest"]
    assert "cross_frequency_power_correlation_roi-all" in saved_files


def test_plot_cross_frequency_power_correlation_rejects_underpowered_condition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from eeg_pipeline.plotting.features import power as power_module

    monkeypatch.setattr(
        power_module,
        "_save_cross_frequency_power_correlation_figure",
        lambda **kwargs: pytest.fail("figure should not be saved for all-NaN matrices"),
    )

    power_df = pd.DataFrame(
        {
            "power_active_alpha_ch_Fz_mean": [1.0, 1.2],
            "power_active_beta_ch_Fz_mean": [0.5, 0.7],
        }
    )
    events_df = pd.DataFrame({"condition": [0, 0]})
    config = {
        "plotting": {
            "comparisons": {
                "comparison_segment": "active",
                "comparison_column": "condition",
                "comparison_values": [0],
                "comparison_rois": ["all"],
            }
        }
    }

    with pytest.raises(ValueError, match="generated no plots"):
        power_module.plot_cross_frequency_power_correlation(
            power_df=power_df,
            events_df=events_df,
            subject="01",
            save_dir=tmp_path,
            logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
            config=config,
        )


def test_group_cross_frequency_power_correlation_rejects_underpowered_subject_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import eeg_pipeline.plotting.orchestration.features as orchestration_module

    power_df = pd.DataFrame(
        {
            "power_active_alpha_ch_Fz_mean": [1.0, 1.2],
            "power_active_beta_ch_Fz_mean": [0.5, 0.7],
        }
    )
    events_df = pd.DataFrame({"condition": [0, 0]})
    config = {
        "plotting": {
            "comparisons": {
                "comparison_segment": "active",
                "comparison_column": "condition",
                "comparison_values": [0],
                "comparison_rois": ["all"],
            }
        }
    }

    monkeypatch.setattr(orchestration_module, "_load_config_if_needed", lambda cfg: cfg)
    monkeypatch.setattr(orchestration_module, "setup_matplotlib", lambda cfg: None)
    monkeypatch.setattr(orchestration_module, "_resolve_task", lambda task, cfg: "task")
    monkeypatch.setattr(orchestration_module, "resolve_deriv_root", lambda deriv_root, config: tmp_path)
    monkeypatch.setattr(orchestration_module, "deriv_features_path", lambda deriv_root, subject: tmp_path / subject)
    monkeypatch.setattr(orchestration_module, "_load_features_power_df", lambda **kwargs: power_df.copy())
    monkeypatch.setattr(
        orchestration_module,
        "load_epochs_for_analysis",
        lambda **kwargs: (None, events_df.copy()),
    )
    monkeypatch.setattr(orchestration_module, "deriv_plots_path", lambda *args, **kwargs: tmp_path / "plots")
    monkeypatch.setattr(orchestration_module, "ensure_dir", lambda path: path.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(orchestration_module, "_resolve_power_roi_names", lambda **kwargs: ["all"])
    monkeypatch.setattr(orchestration_module, "_save_plot_manifest", lambda **kwargs: None)

    monkeypatch.setattr(
        "eeg_pipeline.plotting.features.roi.get_roi_definitions",
        lambda config: {},
    )
    monkeypatch.setattr(
        "eeg_pipeline.plotting.features.power._save_cross_frequency_power_correlation_figure",
        lambda **kwargs: pytest.fail("group figure should not be saved for all-NaN matrices"),
    )

    with pytest.raises(ValueError, match="generated no plots"):
        orchestration_module.visualize_power_cross_frequency_correlation_for_group(
            subjects=["01", "02"],
            task="pain",
            deriv_root=tmp_path,
            config=config,
            logger=SimpleNamespace(
                warning=lambda *args, **kwargs: None,
                debug=lambda *args, **kwargs: None,
            ),
        )


def test_plot_power_timecourse_by_condition_rejects_rest_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eeg_pipeline.plotting.features import power as power_module

    monkeypatch.setattr(power_module, "_validate_epochs_tfr", lambda *args, **kwargs: None)
    tfr = _FakePlotTFR(["Fz"], n_epochs=3)
    events_df = pd.DataFrame({"trial_id": [1, 2, 3]})
    config = {"feature_engineering": {"task_is_rest": True}}

    with pytest.raises(ValueError, match="resting-state"):
        power_module.plot_power_timecourse_by_condition(
            tfr=tfr,
            subject="01",
            save_dir=Path("."),
            logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
            events_df=events_df,
            config=config,
        )


def test_format_condition_display_label_preserves_configured_label() -> None:
    config = {
        "plotting": {
            "comparisons": {
                "comparison_column": "pain_binary_coded",
                "comparison_values": [0.0, 1.0],
                "comparison_labels": ["Low pain", "High pain"],
            }
        }
    }

    assert _format_condition_display_label("High pain", config) == "High pain"


def test_format_triptych_condition_label_keeps_raw_comparison_value_compact() -> None:
    config = {
        "plotting": {
            "comparisons": {
                "comparison_column": "pain_binary_coded",
                "comparison_values": [0.0, 1.0],
            }
        }
    }

    assert _format_triptych_condition_label("1.0", config) == "1.0"


def test_compute_group_timecourse_significance_mask_flags_consistent_effect() -> None:
    condition1 = np.ones((8, 5), dtype=float)
    condition2 = np.ones((8, 5), dtype=float)
    condition2[:, 2:] += 0.5

    significant_mask = _compute_group_timecourse_significance_mask(condition1, condition2, config={})

    assert significant_mask.shape == (5,)
    assert not bool(significant_mask[0])
    assert bool(significant_mask[-1])


def test_compute_unpaired_curve_significance_mask_flags_consistent_effect() -> None:
    from eeg_pipeline.plotting.features.power import _compute_unpaired_curve_significance_mask

    condition1 = np.ones((7, 5), dtype=float)
    condition2 = np.ones((6, 5), dtype=float)
    condition2[:, 2:] += 0.6

    significant_mask = _compute_unpaired_curve_significance_mask(condition1, condition2, config={})

    assert significant_mask.shape == (5,)
    assert not bool(significant_mask[0])
    assert bool(significant_mask[-1])


def test_compute_group_curve_significance_mask_flags_consistent_effect() -> None:
    condition1 = np.ones((6, 4), dtype=float)
    condition2 = np.ones((6, 4), dtype=float)
    condition2[:, 1:] += 0.4

    significant_mask = _compute_group_curve_significance_mask(condition1, condition2, config={})

    assert significant_mask.shape == (4,)
    assert not bool(significant_mask[0])
    assert bool(significant_mask[-1])


def test_compute_group_band_summary_stats_flags_band_level_effect() -> None:
    freqs = np.array([2.0, 4.0, 8.0, 16.0], dtype=float)
    frequency_bands = {
        "delta": (1.0, 4.0),
        "alpha": (8.0, 12.0),
        "beta": (13.0, 30.0),
    }
    condition1 = np.ones((6, 4), dtype=float)
    condition2 = np.ones((6, 4), dtype=float)
    condition2[:, 2] += 0.5

    band_stats = _compute_group_band_summary_stats(
        condition1,
        condition2,
        freqs,
        frequency_bands,
        config={},
    )

    assert set(band_stats) == {"delta", "alpha", "beta"}
    assert not bool(band_stats["delta"]["significant"])
    assert bool(band_stats["alpha"]["significant"])
    assert band_stats["alpha"]["effect_size"] > 0


def test_compute_group_band_summary_stats_uses_frequency_width_weighting() -> None:
    freqs = np.array([1.0, 2.0, 10.0], dtype=float)
    frequency_bands = {"full": (1.0, 10.0)}
    condition1 = np.zeros((6, 3), dtype=float)
    condition2 = np.tile(np.array([1.0, 1.0, -0.6875], dtype=float), (6, 1))

    band_stats = _compute_group_band_summary_stats(
        condition1,
        condition2,
        freqs,
        frequency_bands,
        config={},
    )

    assert band_stats["full"]["effect_size"] == pytest.approx(0.0)
    assert band_stats["full"]["p_value"] == pytest.approx(1.0)
    assert not bool(band_stats["full"]["significant"])


def test_compute_unpaired_band_summary_stats_flags_band_level_effect() -> None:
    from eeg_pipeline.plotting.features.power import _compute_unpaired_band_summary_stats

    freqs = np.array([2.0, 4.0, 8.0, 16.0], dtype=float)
    frequency_bands = {
        "delta": (1.0, 4.0),
        "alpha": (8.0, 12.0),
        "beta": (13.0, 30.0),
    }
    condition1 = np.ones((8, 4), dtype=float)
    condition2 = np.ones((7, 4), dtype=float)
    condition2[:, 2] += 0.7

    band_stats = _compute_unpaired_band_summary_stats(
        condition1,
        condition2,
        freqs,
        frequency_bands,
        config={},
    )

    assert set(band_stats) == {"delta", "alpha", "beta"}
    assert not bool(band_stats["delta"]["significant"])
    assert bool(band_stats["alpha"]["significant"])
    assert band_stats["alpha"]["effect_size"] > 0


def test_compute_subject_condition_psd_averages_epochwise_logratios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eeg_pipeline.plotting.orchestration.features import _compute_subject_condition_psd
    import eeg_pipeline.utils.analysis.tfr as tfr_module

    monkeypatch.setattr(tfr_module, "apply_baseline_and_crop", _fake_apply_baseline_and_crop)

    tfr = _FakeEpochsSpectralTFR(
        np.array(
            [
                [[[1.0, 1.0, 4.0, 4.0], [1.0, 1.0, 4.0, 4.0]]],
                [[[4.0, 4.0, 8.0, 8.0], [4.0, 4.0, 8.0, 8.0]]],
            ],
            dtype=float,
        ),
        times=np.array([-1.0, -0.5, 0.5, 1.0], dtype=float),
        freqs=np.array([8.0, 12.0], dtype=float),
        ch_names=["Fz"],
    )

    freqs, psd_vector = _compute_subject_condition_psd(
        tfr_epochs=tfr,
        mask=np.array([True, True], dtype=bool),
        active_window=(0.5, 1.0),
        baseline_window=(-1.0, -0.5),
        logger=SimpleNamespace(),
    )

    expected_value = (np.log10(4.0) + np.log10(2.0)) / 2.0
    np.testing.assert_allclose(freqs, np.array([8.0, 12.0], dtype=float))
    np.testing.assert_allclose(psd_vector, np.full(2, expected_value, dtype=float))


def test_plot_psd_by_conditions_skips_trial_level_significance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from eeg_pipeline.plotting.features import power as power_module
    import eeg_pipeline.utils.analysis.tfr as tfr_module

    monkeypatch.setattr(tfr_module, "apply_baseline_and_crop", _fake_apply_baseline_and_crop)
    monkeypatch.setattr(power_module, "apply_baseline_and_crop", _fake_apply_baseline_and_crop)

    calls = {"curve_strip": 0, "band_strip": 0}

    def _count_curve_strip(*args, **kwargs) -> None:
        del args, kwargs
        calls["curve_strip"] += 1

    def _count_band_strip(*args, **kwargs) -> None:
        del args, kwargs
        calls["band_strip"] += 1

    monkeypatch.setattr(power_module, "_draw_curve_significance_strip", _count_curve_strip)
    monkeypatch.setattr(power_module, "_draw_psd_band_summary_strip", _count_band_strip)
    monkeypatch.setattr(power_module, "save_fig", lambda *args, **kwargs: None)

    tfr = _FakeEpochsSpectralTFR(
        np.array(
            [
                [[[1.0, 1.0, 4.0, 4.0], [1.0, 1.0, 4.0, 4.0]]],
                [[[1.0, 1.0, 4.0, 4.0], [1.0, 1.0, 4.0, 4.0]]],
                [[[1.0, 1.0, 4.0, 4.0], [1.0, 1.0, 4.0, 4.0]]],
                [[[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]],
                [[[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]],
                [[[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]],
            ],
            dtype=float,
        ),
        times=np.array([-1.0, -0.5, 0.5, 1.0], dtype=float),
        freqs=np.array([8.0, 12.0], dtype=float),
        ch_names=["Fz"],
    )

    config = {
        "time_frequency_analysis": {
            "baseline_window": [-1.0, -0.5],
            "active_window": [0.5, 1.0],
        },
        "feature_engineering": {
            "frequency_bands": {
                "alpha": [8.0, 12.0],
            }
        },
    }
    logger = SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    conditions = [
        ("Low", np.array([True, True, True, False, False, False], dtype=bool)),
        ("High", np.array([False, False, False, True, True, True], dtype=bool)),
    ]

    try:
        power_module._plot_psd_by_conditions(
            tfr,
            conditions,
            "01",
            tmp_path,
            logger,
            config,
            roi_name="all",
        )
    finally:
        plt.close("all")

    assert calls["curve_strip"] == 0
    assert calls["band_strip"] == 0


def test_compute_cross_frequency_power_matrix_returns_square_band_matrix() -> None:
    from eeg_pipeline.plotting.features.power import _compute_cross_frequency_power_matrix

    band_values = {
        "alpha": np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        "beta": np.array([2.0, 4.0, 6.0, 8.0], dtype=float),
        "theta": np.array([4.0, 3.0, 2.0, 1.0], dtype=float),
    }

    correlation_df = _compute_cross_frequency_power_matrix(band_values, band_order=["theta", "alpha", "beta"])

    assert list(correlation_df.index) == ["theta", "alpha", "beta"]
    assert list(correlation_df.columns) == ["theta", "alpha", "beta"]
    assert correlation_df.loc["alpha", "alpha"] == pytest.approx(1.0)
    assert correlation_df.loc["alpha", "beta"] == pytest.approx(1.0)
    assert correlation_df.loc["theta", "alpha"] < 0


def test_build_topomap_panel_requires_more_than_minimum_channels() -> None:
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq=250.0, ch_types="eeg")

    assert _build_topomap_panel({"Fz": 1.0, "Cz": 2.0, "Pz": 3.0}, info) is None

    panel = _build_topomap_panel({"Fz": 1.0, "Cz": 2.0, "Pz": 3.0, "Oz": 4.0}, info)

    assert panel is not None
    data, panel_info = panel
    assert data.shape == (4,)
    assert len(panel_info.ch_names) == 4


def test_draw_group_subject_traces_requires_matching_time_axis() -> None:
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="time axis length"):
            _draw_group_subject_traces(
                ax,
                np.array([0.0, 1.0, 2.0], dtype=float),
                np.ones((3, 2), dtype=float),
                color="tab:blue",
            )
    finally:
        plt.close(fig)


def test_draw_curve_significance_strip_requires_matching_axis_shape() -> None:
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="must match the axis shape"):
            _draw_curve_significance_strip(
                ax,
                np.array([1.0, 2.0, 4.0], dtype=float),
                np.array([True, False], dtype=bool),
            )
    finally:
        plt.close(fig)


def test_compute_window_mean_series_returns_one_value_per_sample() -> None:
    sample_matrix = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
        ],
        dtype=float,
    )
    times = np.array([-1.0, 0.0, 1.0, 2.0], dtype=float)

    window_mean = _compute_window_mean_series(
        sample_matrix,
        times,
        (0.0, 1.0),
        context="test window summary",
    )

    np.testing.assert_allclose(window_mean, np.array([2.5, 5.0], dtype=float))


def test_compute_paired_effect_matrix_uses_condition2_minus_condition1() -> None:
    condition1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    condition2 = np.array([[1.5, 1.0], [4.0, 6.0]], dtype=float)

    effect_matrix = _compute_paired_effect_matrix(
        condition1,
        condition2,
        context="test effect matrix",
    )

    np.testing.assert_allclose(
        effect_matrix,
        np.array([[0.5, -1.0], [1.0, 2.0]], dtype=float),
    )


def test_compute_paired_differences_uses_condition2_minus_condition1() -> None:
    condition1 = np.array([1.0, 2.0, 3.0], dtype=float)
    condition2 = np.array([1.5, 1.0, 4.0], dtype=float)

    differences = _compute_paired_differences(condition1, condition2)

    np.testing.assert_allclose(differences, np.array([0.5, -1.0, 1.0], dtype=float))


def test_compute_paired_cohens_d_returns_infinite_for_constant_nonzero_shift() -> None:
    before = np.array([1.0, 2.0, 3.0], dtype=float)
    after = np.array([1.5, 2.5, 3.5], dtype=float)

    effect_size = compute_paired_cohens_d(before, after)

    assert np.isposinf(effect_size)


def test_compute_paired_wilcoxon_stats_returns_rank_biserial_for_constant_shift() -> None:
    before = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    after = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=float)

    _, effect_size = _compute_paired_wilcoxon_stats(before, after)

    assert effect_size == pytest.approx(1.0)


def test_sample_count_summaries_report_exact_values_and_ranges() -> None:
    paired_summary = _summarize_paired_sample_counts(
        {
            "alpha": (np.array([1.0, 2.0]), np.array([1.1, 2.1])),
            "beta": (np.array([1.0, 2.0, 3.0]), np.array([1.2, 2.2, 3.2])),
        },
        "trials",
    )
    multi_window_summary = _summarize_multi_window_sample_counts(
        {
            "alpha": {
                "baseline": np.array([1.0, 2.0]),
                "plateau": np.array([1.1, 2.1, 3.1]),
            },
            "beta": {
                "baseline": np.array([0.5, 0.6, 0.7]),
            },
        },
        "trials",
    )

    assert _format_count_range([4, 4, 4]) == "4"
    assert _format_count_range([2, 4, 3]) == "2-4"
    assert paired_summary == "N: 2-3 trials"
    assert multi_window_summary == "N per window: 2-3 trials"


def test_plot_single_band_comparison_adds_delta_column_for_paired_data() -> None:
    fig, (ax, delta_ax) = plt.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [5, 1], "wspace": 0.1},
    )
    plot_cfg = SimpleNamespace(font=SimpleNamespace(title=12, small=8))

    try:
        _plot_single_band_comparison(
            ax=ax,
            delta_ax=delta_ax,
            condition1_values=np.array([1.0, 1.2, 1.4], dtype=float),
            condition2_values=np.array([1.3, 1.5, 1.8], dtype=float),
            band="alpha",
            label1="Low",
            label2="High",
            band_color="#ff7f0e",
            condition1_color="#1f77b4",
            condition2_color="#d62728",
            q_value=0.02,
            effect_size=0.8,
            is_significant=True,
            plot_cfg=plot_cfg,
            config={},
        )

        tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert tick_labels == ["Low", "High"]
        assert ax.get_xlim()[1] <= 1.4
        assert len(fig.axes) == 2
        delta_tick_labels = [tick.get_text() for tick in delta_ax.get_xticklabels()]
        assert delta_tick_labels == ["Δ"]
        assert delta_ax.get_position().x0 > ax.get_position().x1
        assert ax.get_ylim()[0] > 0.8
        assert delta_ax.get_ylim()[0] < 0.0 < delta_ax.get_ylim()[1]
    finally:
        plt.close(fig)
