from __future__ import annotations

import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.analysis.features.preparation import _determine_frequency_bands
from eeg_pipeline.analysis.features.phase import (
    _compute_itpc_map_by_method,
    extract_itpc_from_precomputed,
    extract_pac_from_precomputed,
    extract_phase_features,
)
from eeg_pipeline.analysis.features.precomputed.erds import extract_erds_from_precomputed
from eeg_pipeline.analysis.features.spectral import (
    extract_power_features,
    extract_spectral_features,
)
from eeg_pipeline.analysis.features.erp import extract_erp_features
from eeg_pipeline.analysis.features.quality import extract_quality_features
from eeg_pipeline.analysis.features.precomputed.extras import (
    extract_asymmetry_from_precomputed,
    extract_band_ratios_from_precomputed,
)
from eeg_pipeline.analysis.features.api import (
    _compute_tfr_for_features,
    _get_family_spatial_transform,
    _prepare_precomputed_data,
    _resolve_pac_segment_window,
    extract_precomputed_features,
)
from eeg_pipeline.analysis.features.selection import resolve_feature_categories
from eeg_pipeline.analysis.features.source_localization import (
    FMRIVoxelSelection,
    _append_source_band_family_features,
    _compute_eloreta_source_estimates,
    _compute_roi_envelope,
    _compute_roi_power,
    _compute_roi_timecourses_from_row_indices,
    _extract_roi_timecourses,
    _extract_roi_timecourses_from_vertex_indices,
    _load_source_contrast_config,
    _load_source_localization_config,
    extract_source_connectivity_features,
    extract_source_contrast_features,
    extract_source_localization_features,
)
from eeg_pipeline.context.features import FeatureContext
from eeg_pipeline.analysis.features.bursts import extract_burst_features
from eeg_pipeline.types import BandData, PrecomputedData, PrecomputedQC, TimeWindows
from eeg_pipeline.utils.analysis.tfr import compute_tfr_for_subject
from tests.pipelines_test_utils import DotConfig


class _EpochStub:
    def __init__(self, n_epochs: int, sfreq: float = 100.0, n_times: int = 80):
        self._n_epochs = int(n_epochs)
        self.info = {"sfreq": float(sfreq)}
        self.times = np.arange(int(n_times), dtype=float) / float(sfreq)

    def __len__(self):
        return self._n_epochs

    def copy(self):
        return _EpochStub(self._n_epochs, self.info["sfreq"], len(self.times))

    def filter(self, *_args, **_kwargs):
        return self

    def __getitem__(self, key):
        if isinstance(key, np.ndarray) and key.dtype == bool:
            return _EpochStub(int(np.sum(key)), self.info["sfreq"], len(self.times))
        return self


class _TFRStub:
    def __init__(self, n_epochs: int, times: np.ndarray):
        self._n_epochs = int(n_epochs)
        self.times = np.asarray(times, dtype=float)
        self.metadata = None
        self.comment = None

    def __len__(self):
        return self._n_epochs


class _ComplexTFRStub:
    def __init__(self, data: np.ndarray, times: np.ndarray, freqs: np.ndarray, sfreq: float):
        self.data = np.asarray(data)
        self.times = np.asarray(times, dtype=float)
        self.freqs = np.asarray(freqs, dtype=float)
        self.info = {
            "ch_names": [f"C{idx+1}" for idx in range(self.data.shape[1])],
            "sfreq": float(sfreq),
        }


class _StcStub:
    def __init__(self, data: np.ndarray):
        self.data = np.asarray(data, dtype=float)
        self.tmin = 0.0
        self.tstep = 1.0

    def copy(self):
        return _StcStub(self.data.copy())

    def save(self, path: str, overwrite: bool = False):
        self.saved_path = str(path)
        self.overwrite = bool(overwrite)


class TestScientificValidityGuards(unittest.TestCase):
    def test_source_roi_sign_alignment_rejects_nonfinite_source_values(self):
        stcs = [
            SimpleNamespace(
                data=np.array(
                    [
                        [1.0, np.nan, 2.0],
                        [0.5, 0.2, 0.1],
                    ],
                    dtype=float,
                )
            ),
            SimpleNamespace(
                data=np.array(
                    [
                        [1.2, 1.3, 1.4],
                        [0.4, 0.3, 0.2],
                    ],
                    dtype=float,
                )
            ),
        ]

        with self.assertRaisesRegex(ValueError, "non-finite source values"):
            _compute_roi_timecourses_from_row_indices(
                stcs=stcs,
                roi_row_indices={"roi": [0, 1]},
                roi_names=["roi"],
            )

    def test_feature_context_rejects_cross_trial_features_without_train_mask_in_trial_safe_mode(self):
        with self.assertRaisesRegex(ValueError, "train_mask"):
            FeatureContext(
                subject="0001",
                task="pain",
                config=DotConfig({"feature_engineering": {"analysis_mode": "trial_ml_safe"}}),
                deriv_root=Path(tempfile.mkdtemp()),
                logger=logging.getLogger("feature-context-trial-safe"),
                epochs=_EpochStub(n_epochs=4, sfreq=100.0, n_times=40),
                aligned_events=pd.DataFrame({"trial": [0, 1, 2, 3]}),
                feature_categories=["connectivity"],
                train_mask=None,
                analysis_mode="trial_ml_safe",
            )

    def test_feature_context_allows_trial_safe_bursts_without_train_mask(self):
        ctx = FeatureContext(
            subject="0001",
            task="pain",
            config=DotConfig(
                {
                    "feature_engineering": {
                        "analysis_mode": "trial_ml_safe",
                        "bursts": {"threshold_reference": "trial"},
                    }
                }
            ),
            deriv_root=Path(tempfile.mkdtemp()),
            logger=logging.getLogger("feature-context-bursts-trial-safe"),
            epochs=_EpochStub(n_epochs=4, sfreq=100.0, n_times=40),
            aligned_events=pd.DataFrame({"trial": [0, 1, 2, 3]}),
            feature_categories=["bursts"],
            train_mask=None,
            analysis_mode="trial_ml_safe",
        )

        self.assertEqual(ctx.analysis_mode, "trial_ml_safe")

    def test_rest_mode_rejects_event_locked_feature_categories(self):
        config = DotConfig({"feature_engineering": {"task_is_rest": True}})
        with self.assertRaisesRegex(ValueError, "event-locked categories: erp, itpc"):
            resolve_feature_categories(config, ["power", "erp", "itpc"])

    def test_erp_extractor_rejects_rest_mode(self):
        ctx = SimpleNamespace(
            config=DotConfig({"feature_engineering": {"task_is_rest": True}}),
            logger=logging.getLogger("erp-rest"),
        )
        with self.assertRaisesRegex(ValueError, "ERP is not scientifically valid"):
            extract_erp_features(ctx)

    def test_aperiodic_rest_target_window_must_match_requested_segment(self):
        from eeg_pipeline.analysis.features.aperiodic import _rebuild_window_masks

        times = np.array([0.0, 1.0, 2.0], dtype=float)
        windows = TimeWindows(
            masks={},
            ranges={"target": (10.0, 11.0), "available": (0.0, 2.0)},
            times=times,
        )

        with self.assertRaisesRegex(ValueError, "target window"):
            _rebuild_window_masks(
                windows=windows,
                times=times,
                target_name="target",
                config=DotConfig({"feature_engineering": {"task_is_rest": True}}),
                logger=logging.getLogger("aperiodic-rest-target"),
            )

    def test_itpc_extractors_reject_rest_mode(self):
        ctx = SimpleNamespace(
            config=DotConfig({"feature_engineering": {"task_is_rest": True}}),
            logger=logging.getLogger("itpc-rest"),
        )
        with self.assertRaisesRegex(ValueError, "ITPC is not scientifically valid"):
            extract_phase_features(ctx, ["alpha"])
        with self.assertRaisesRegex(ValueError, "ITPC is not scientifically valid"):
            extract_itpc_from_precomputed(SimpleNamespace(config=ctx.config))

    def test_erds_extractor_rejects_rest_mode(self):
        precomputed = SimpleNamespace(config=DotConfig({"feature_engineering": {"task_is_rest": True}}))
        with self.assertRaisesRegex(ValueError, "ERDS is not scientifically valid"):
            extract_erds_from_precomputed(precomputed, ["alpha"])

    def test_extract_precomputed_features_rejects_rest_incompatible_groups(self):
        with self.assertRaisesRegex(ValueError, "event-locked categories: itpc"):
            extract_precomputed_features(
                epochs=SimpleNamespace(),
                bands=["alpha"],
                config=DotConfig(
                    {
                        "preprocessing": {"task_is_rest": True},
                        "feature_engineering": {"task_is_rest": True},
                    }
                ),
                logger=logging.getLogger("precomputed-rest-groups"),
                feature_groups=["itpc"],
            )

    def test_extract_precomputed_features_rejects_trial_ml_safe_in_rest_mode(self):
        with self.assertRaisesRegex(ValueError, "analysis_mode=group_stats"):
            extract_precomputed_features(
                epochs=SimpleNamespace(),
                bands=["alpha"],
                config=DotConfig(
                    {
                        "preprocessing": {"task_is_rest": True},
                        "feature_engineering": {
                            "task_is_rest": True,
                            "analysis_mode": "trial_ml_safe",
                        }
                    }
                ),
                logger=logging.getLogger("precomputed-rest-analysis-mode"),
                feature_groups=["spectral"],
            )

    def test_extract_precomputed_features_rejects_mismatched_rest_configuration(self):
        with self.assertRaisesRegex(ValueError, "task_is_rest.*must match"):
            extract_precomputed_features(
                epochs=SimpleNamespace(),
                bands=["alpha"],
                config=DotConfig(
                    {
                        "preprocessing": {"task_is_rest": True},
                        "feature_engineering": {"task_is_rest": False},
                    }
                ),
                logger=logging.getLogger("precomputed-rest-mismatch"),
                feature_groups=["spectral"],
            )

    def test_epoch_validation_rejects_any_nonfinite_samples(self):
        from eeg_pipeline.utils.validation import validate_epochs

        data = np.ones((2, 2, 200), dtype=float)
        data[0, 0, 0] = np.nan
        info = mne.create_info(["C3", "C4"], sfreq=100.0, ch_types="eeg")
        epochs = mne.EpochsArray(data, info, verbose=False)

        result = validate_epochs(
            epochs,
            DotConfig({"validation": {"min_epochs": 1, "min_channels": 1}}),
            logger=logging.getLogger("validate-nonfinite"),
        )

        self.assertFalse(result.valid)
        self.assertTrue(any("NaN/Inf" in issue for issue in result.issues))

    def test_rest_target_window_rejects_segment_substitution(self):
        from eeg_pipeline.analysis.features.rest import select_single_rest_analysis_segment

        masks = {
            "analysis": np.ones(10, dtype=bool),
            "active": np.zeros(10, dtype=bool),
        }

        with self.assertRaisesRegex(ValueError, "target window 'active' does not contain valid samples"):
            select_single_rest_analysis_segment(
                masks,
                feature_name="Spectral",
                target_name="active",
            )

    def test_extract_precomputed_features_defaults_to_spectral_only(self):
        precomputed = SimpleNamespace(
            data=np.ones((2, 1, 4), dtype=float),
            metadata=None,
            condition_labels=None,
        )

        with patch(
            "eeg_pipeline.analysis.features.api.extract_power_from_precomputed",
            return_value=(pd.DataFrame({"power": [1.0, 2.0]}), ["power"]),
        ) as mock_spectral, patch(
            "eeg_pipeline.analysis.features.api.extract_erds_from_precomputed",
        ) as mock_erds:
            result = extract_precomputed_features(
                epochs=SimpleNamespace(),
                bands=["alpha"],
                config=DotConfig(
                    {
                        "preprocessing": {"task_is_rest": False},
                        "feature_engineering": {"task_is_rest": False},
                    }
                ),
                logger=logging.getLogger("precomputed-default-groups"),
                precomputed=precomputed,
            )

        self.assertIn("spectral", result.features)
        self.assertTrue(mock_spectral.called)
        self.assertFalse(mock_erds.called)

    def test_rest_mode_rejects_precomputed_subtract_evoked(self):
        ctx = SimpleNamespace(
            config=DotConfig(
                {
                    "preprocessing": {"task_is_rest": True},
                    "feature_engineering": {
                        "task_is_rest": True,
                        "precomputed": {"subtract_evoked": True},
                    },
                }
            ),
            feature_categories=["spectral"],
            logger=logging.getLogger("precomputed-subtract-evoked-rest"),
            aligned_events=pd.DataFrame({"trial_id": [1, 2]}),
            precomputed=None,
            windows=None,
            _original_epochs=None,
            train_mask=None,
            analysis_mode="group_stats",
            get_precomputed_for_family=lambda _family: None,
            set_precomputed=lambda _value: None,
            set_precomputed_for_family=lambda _family, _value: None,
        )

        with self.assertRaisesRegex(ValueError, "subtract_evoked is not scientifically valid"):
            _prepare_precomputed_data(
                ctx,
                working_epochs=SimpleNamespace(),
                power_bands=["alpha"],
                tmin=None,
                tmax=None,
            )

    def test_rest_mode_rejects_power_subtract_evoked(self):
        ctx = SimpleNamespace(
            config=DotConfig(
                {
                    "preprocessing": {"task_is_rest": True},
                    "feature_engineering": {
                        "task_is_rest": True,
                        "power": {"subtract_evoked": True},
                    },
                }
            ),
            feature_categories=["power"],
            logger=logging.getLogger("power-subtract-evoked-rest"),
            epochs=SimpleNamespace(
                info={"sfreq": 100.0},
                times=np.array([0.0, 0.1], dtype=float),
            ),
            aligned_events=pd.DataFrame({"trial_id": [1, 2]}),
            windows=None,
            tfr=None,
            analysis_mode="group_stats",
            train_mask=None,
        )

        with self.assertRaisesRegex(ValueError, "subtract_evoked is not scientifically valid"):
            _compute_tfr_for_features(ctx, tmin=None, tmax=None)

    def test_quality_resting_state_rejects_empty_target_window(self):
        n_epochs = 2
        n_channels = 2
        n_times = 50
        times = np.arange(n_times, dtype=float) / 100.0
        analysis_mask = np.ones(n_times, dtype=bool)
        empty_mask = np.zeros(n_times, dtype=bool)
        epochs = SimpleNamespace(
            info={"sfreq": 100.0, "ch_names": ["C3", "C4"]},
            times=times,
            get_data=lambda picks=None: np.ones((n_epochs, n_channels, n_times), dtype=float),
        )
        ctx = SimpleNamespace(
            epochs=epochs,
            windows=TimeWindows(
                masks={"analysis": analysis_mask, "active": empty_mask},
                ranges={"analysis": (0.0, 0.5), "active": (0.0, 0.5)},
                times=times,
                name="active",
            ),
            name="active",
            config=DotConfig({"feature_engineering": {"task_is_rest": True}}),
            logger=logging.getLogger("quality-rest"),
        )

        with patch(
            "eeg_pipeline.analysis.features.quality.pick_eeg_channels",
            return_value=(np.array([0, 1]), ["C3", "C4"]),
        ), patch(
            "eeg_pipeline.analysis.features.quality._compute_signal_metrics",
            return_value={"variance": np.array([1.0, 2.0], dtype=float)},
        ):
            with self.assertRaisesRegex(ValueError, "target window 'active' does not contain valid samples"):
                extract_quality_features(ctx)

    def test_quality_resting_state_rejects_empty_target_before_substitution(self):
        n_times = 50
        times = np.arange(n_times, dtype=float) / 100.0
        epochs = SimpleNamespace(
            info={"sfreq": 100.0, "ch_names": ["C3", "C4"]},
            times=times,
            get_data=lambda picks=None: np.ones((2, 2, n_times), dtype=float),
        )
        ctx = SimpleNamespace(
            epochs=epochs,
            windows=TimeWindows(
                masks={
                    "analysis_a": np.ones(n_times, dtype=bool),
                    "analysis_b": np.ones(n_times, dtype=bool),
                    "active": np.zeros(n_times, dtype=bool),
                },
                ranges={
                    "analysis_a": (0.0, 0.5),
                    "analysis_b": (0.0, 0.5),
                    "active": (0.0, 0.5),
                },
                times=times,
                name="active",
            ),
            name="active",
            config=DotConfig({"feature_engineering": {"task_is_rest": True}}),
            logger=logging.getLogger("quality-rest-ambiguous"),
        )

        with patch(
            "eeg_pipeline.analysis.features.quality.pick_eeg_channels",
            return_value=(np.array([0, 1]), ["C3", "C4"]),
        ):
            with self.assertRaisesRegex(ValueError, "target window 'active' does not contain valid samples"):
                extract_quality_features(ctx)

    def test_power_without_baseline_emits_log10raw_for_resting_state(self):
        config = DotConfig(
            {
                "feature_engineering": {
                    "task_is_rest": True,
                    "power": {"require_baseline": False, "emit_db": True},
                },
            }
        )
        tfr = SimpleNamespace(
            data=np.full((2, 1, 1, 2), 10.0, dtype=float),
            freqs=np.array([10.0], dtype=float),
            times=np.array([0.0, 0.5], dtype=float),
            info={"ch_names": ["Cz"]},
            comment=None,
        )
        ctx = SimpleNamespace(
            results={"tfr": tfr},
            config=config,
            frequency_bands={"alpha": (8.0, 12.0)},
            spatial_modes=["global"],
            windows=SimpleNamespace(ranges={"active": (0.0, 1.0)}),
            name="active",
            logger=logging.getLogger("power-rest"),
            baseline_df=None,
        )

        features_df, columns = extract_power_features(ctx, ["alpha"])

        self.assertEqual(columns, ["power_active_alpha_global_log10raw_mean"])
        self.assertTrue(np.allclose(features_df.iloc[:, 0].to_numpy(dtype=float), 1.0))

    def test_spectral_resting_state_uses_available_analysis_segments(self):
        class _EpochsStub:
            def __init__(self):
                self.info = mne.create_info(ch_names=["Cz"], sfreq=100.0, ch_types="eeg")
                self.times = np.linspace(0.0, 3.98, 400)

            def get_data(self, picks=None):
                data = np.ones((2, 1, self.times.size), dtype=float)
                return data if picks is None else data[:, picks, :]

        config = DotConfig(
            {
                "feature_engineering": {
                    "task_is_rest": True,
                    "spectral": {
                        "psd_method": "welch",
                        "min_segment_sec": 1.0,
                        "min_cycles_at_fmin": 0.0,
                        "exclude_line_noise": False,
                    }
                },
                "frequency_bands": {"alpha": [8.0, 12.0]},
                "rois": {},
            }
        )
        windows = SimpleNamespace(
            ranges={"analysis": (0.0, 4.0)},
            get_mask=lambda _name: np.zeros(400, dtype=bool),
        )
        ctx = SimpleNamespace(
            epochs=_EpochsStub(),
            config=config,
            logger=logging.getLogger("spectral-rest"),
            frequency_bands={"alpha": (8.0, 12.0)},
            spatial_modes=["global"],
            windows=windows,
            name=None,
        )

        features_df, columns, qc = extract_spectral_features(ctx, ["alpha"])

        self.assertFalse(features_df.empty)
        self.assertTrue(columns)
        self.assertIn("analysis", qc["segment_durations"])

    def test_power_uses_analysis_window_when_segment_name_is_missing(self):
        config = DotConfig(
            {
                "feature_engineering": {
                    "power": {
                        "require_baseline": False,
                        "exclude_line_noise": False,
                    }
                },
                "frequency_bands": {"alpha": [8.0, 12.0]},
                "rois": {},
            }
        )
        tfr = SimpleNamespace(
            data=np.full((2, 1, 1, 2), 10.0, dtype=float),
            freqs=np.array([10.0], dtype=float),
            times=np.array([0.0, 0.5], dtype=float),
            info={"ch_names": ["Cz"]},
            comment=None,
        )
        ctx = SimpleNamespace(
            results={"tfr": tfr},
            config=config,
            frequency_bands={"alpha": (8.0, 12.0)},
            spatial_modes=["global"],
            windows=SimpleNamespace(ranges={"analysis": (0.0, 1.0)}),
            name=None,
            logger=logging.getLogger("power-analysis-default"),
            baseline_df=None,
        )

        features_df, columns = extract_power_features(ctx, ["alpha"])

        self.assertEqual(columns, ["power_analysis_alpha_global_log10raw_mean"])
        self.assertTrue(np.allclose(features_df.iloc[:, 0].to_numpy(dtype=float), 1.0))

    def test_pac_api_path_rejects_empty_rest_target_window(self):
        epochs = _EpochStub(n_epochs=2, sfreq=100.0, n_times=80)
        ctx = SimpleNamespace(
            epochs=epochs,
            windows=TimeWindows(
                masks={
                    "analysis": np.concatenate([np.ones(40, dtype=bool), np.zeros(40, dtype=bool)]),
                    "active": np.zeros(80, dtype=bool),
                },
                ranges={"analysis": (0.0, 0.4), "active": (0.0, 0.8)},
                times=epochs.times,
                name="active",
            ),
            name="active",
            config=DotConfig({"feature_engineering": {"task_is_rest": True}}),
            logger=logging.getLogger("pac-api-rest"),
        )

        with self.assertRaisesRegex(ValueError, "target window 'active' does not contain valid samples"):
            _resolve_pac_segment_window(ctx, epochs.times)


    def test_ratios_resting_state_rejects_empty_target_window(self):
        sfreq = 100.0
        times = np.arange(200, dtype=float) / sfreq
        analysis_mask = np.ones(times.shape, dtype=bool)
        empty_mask = np.zeros(times.shape, dtype=bool)
        windows = TimeWindows(
            masks={"analysis": analysis_mask, "active": empty_mask},
            ranges={"analysis": (0.0, 2.0), "active": (0.0, 2.0)},
            times=times,
            name="active",
        )
        precomputed = PrecomputedData(
            data=np.ones((2, 1, times.size), dtype=float),
            times=times,
            sfreq=sfreq,
            ch_names=["Cz"],
            picks=np.array([0]),
            windows=windows,
            config=DotConfig(),
            logger=logging.getLogger("ratios-rest"),
            spatial_modes=["global"],
        )
        config = DotConfig(
            {
                "feature_engineering": {
                    "task_is_rest": True,
                    "spectral": {
                        "ratio_pairs": [["theta", "beta"]],
                        "include_log_ratios": False,
                        "psd_method": "welch",
                        "exclude_line_noise": False,
                    },
                    "ratios": {
                        "min_segment_sec": 0.0,
                        "min_cycles_at_fmin": 0.0,
                        "skip_invalid_segments": True,
                    },
                },
                "frequency_bands": {
                    "theta": [4.0, 8.0],
                    "beta": [13.0, 30.0],
                },
            }
        )

        with patch(
            "eeg_pipeline.analysis.features.precomputed.extras._compute_psd_band_power_for_segment",
            return_value={
                "theta": np.full((2, 1), 4.0, dtype=float),
                "beta": np.full((2, 1), 2.0, dtype=float),
            },
        ):
            with self.assertRaisesRegex(ValueError, "target window 'active' does not contain valid samples"):
                extract_band_ratios_from_precomputed(precomputed, config)

    def test_asymmetry_resting_state_rejects_empty_target_window(self):
        sfreq = 100.0
        times = np.arange(200, dtype=float) / sfreq
        analysis_mask = np.ones(times.shape, dtype=bool)
        empty_mask = np.zeros(times.shape, dtype=bool)
        windows = TimeWindows(
            masks={"analysis": analysis_mask, "active": empty_mask},
            ranges={"analysis": (0.0, 2.0), "active": (0.0, 2.0)},
            times=times,
            name="active",
        )
        config = DotConfig(
            {
                "feature_engineering": {
                    "task_is_rest": True,
                    "asymmetry": {
                        "channel_pairs": [["F3", "F4"]],
                        "min_segment_sec": 0.0,
                        "min_cycles_at_fmin": 0.0,
                        "skip_invalid_segments": True,
                    }
                },
                "frequency_bands": {"alpha": [8.0, 12.0]},
            }
        )
        precomputed = PrecomputedData(
            data=np.ones((2, 2, times.size), dtype=float),
            times=times,
            sfreq=sfreq,
            ch_names=["F3", "F4"],
            picks=np.array([0, 1]),
            windows=windows,
            config=config,
            logger=logging.getLogger("asymmetry-rest"),
        )
        with patch(
            "eeg_pipeline.analysis.features.precomputed.extras._compute_psd_band_power_for_segment",
            return_value={"alpha": np.array([[2.0, 4.0], [2.0, 4.0]], dtype=float)},
        ):
            with self.assertRaisesRegex(ValueError, "target window 'active' does not contain valid samples"):
                extract_asymmetry_from_precomputed(precomputed)

    def test_iaf_trial_ml_safe_requires_train_mask(self):
        data = np.random.default_rng(7).standard_normal((6, 2, 32))
        windows = TimeWindows(
            baseline_mask=np.ones(32, dtype=bool),
            times=np.linspace(-0.5, 0.5, 32),
        )
        qc = PrecomputedQC()
        cfg = DotConfig({"feature_engineering": {"bands": {"use_iaf": True}}})

        with self.assertRaisesRegex(ValueError, "train_mask"):
            _determine_frequency_bands(
                cfg,
                {"alpha": (8.0, 12.0)},
                data,
                100.0,
                ["Cz", "Pz"],
                windows,
                qc,
                logger=None,
                train_mask=None,
                analysis_mode="trial_ml_safe",
            )

    def test_iaf_trial_ml_safe_uses_training_trials_only(self):
        data = np.random.default_rng(11).standard_normal((8, 2, 64))
        train_mask = np.array([True, True, True, False, False, False, False, False], dtype=bool)
        windows = TimeWindows(
            baseline_mask=np.ones(64, dtype=bool),
            times=np.linspace(-0.8, 0.8, 64),
        )
        qc = PrecomputedQC()
        cfg = DotConfig({"feature_engineering": {"bands": {"use_iaf": True}}})
        seen = {"n_epochs": None}

        def _fake_iaf(data_arg, *_args, **_kwargs):
            seen["n_epochs"] = int(np.asarray(data_arg).shape[0])
            return None

        with patch(
            "eeg_pipeline.analysis.features.preparation._estimate_individual_alpha_frequency",
            side_effect=_fake_iaf,
        ):
            _determine_frequency_bands(
                cfg,
                {"alpha": (8.0, 12.0)},
                data,
                100.0,
                ["Cz", "Pz"],
                windows,
                qc,
                logger=None,
                train_mask=train_mask,
                analysis_mode="trial_ml_safe",
            )

        self.assertEqual(seen["n_epochs"], int(np.sum(train_mask)))

    def test_source_localization_trial_ml_safe_lcmv_requires_train_mask(self):
        fmri_cfg = SimpleNamespace(
            enabled=False,
            provenance="independent",
            require_provenance=False,
        )
        src_cfg = SimpleNamespace(method="lcmv", fmri_cfg=fmri_cfg)
        ctx = SimpleNamespace(
            epochs=_EpochStub(5),
            config=DotConfig({}),
            logger=logging.getLogger("src-loc-train-mask"),
            analysis_mode="trial_ml_safe",
            train_mask=None,
            frequency_bands={"alpha": (8.0, 12.0)},
            name="active",
        )

        with patch(
            "eeg_pipeline.analysis.features.source_localization._load_source_localization_config",
            return_value=src_cfg,
        ):
            with self.assertRaisesRegex(ValueError, "train_mask"):
                extract_source_localization_features(
                    ctx,
                    bands=["alpha"],
                    method="lcmv",
                )

    def test_source_connectivity_trial_ml_safe_lcmv_requires_train_mask(self):
        fmri_cfg = SimpleNamespace(
            enabled=False,
            provenance="independent",
            require_provenance=False,
        )
        src_cfg = SimpleNamespace(method="lcmv", fmri_cfg=fmri_cfg)
        ctx = SimpleNamespace(
            epochs=_EpochStub(5),
            config=DotConfig({}),
            logger=logging.getLogger("src-conn-train-mask"),
            analysis_mode="trial_ml_safe",
            train_mask=None,
            frequency_bands={"alpha": (8.0, 12.0)},
            name="active",
        )

        with patch(
            "eeg_pipeline.analysis.features.source_localization._load_source_localization_config",
            return_value=src_cfg,
        ):
            with self.assertRaisesRegex(ValueError, "train_mask"):
                extract_source_connectivity_features(
                    ctx,
                    bands=["alpha"],
                    method="lcmv",
                    connectivity_method="wpli",
                )

    def test_bursts_trial_ml_safe_subject_threshold_requires_train_mask(self):
        times = np.linspace(-0.5, 0.5, 20)
        mask = np.ones(times.size, dtype=bool)
        windows = TimeWindows(
            baseline_mask=mask,
            masks={"baseline": mask, "active": mask},
            ranges={"baseline": (-0.5, 0.5), "active": (-0.5, 0.5)},
            times=times,
        )
        data = np.ones((4, 2, times.size), dtype=float)
        band = BandData(
            band="alpha",
            fmin=8.0,
            fmax=12.0,
            filtered=data.copy(),
            analytic=data.astype(np.complex128),
            envelope=data.copy(),
            phase=np.zeros_like(data),
            power=data.copy(),
        )
        precomputed = PrecomputedData(
            data=data,
            times=times,
            sfreq=100.0,
            ch_names=["C3", "C4"],
            picks=np.arange(2),
            band_data={"alpha": band},
            windows=windows,
            logger=logging.getLogger("bursts-trial-safe"),
            train_mask=None,
        )
        ctx = SimpleNamespace(
            precomputed=precomputed,
            config=DotConfig(
                {
                    "feature_engineering": {
                        "bursts": {
                            "bands": ["alpha"],
                            "threshold_reference": "subject",
                        }
                    }
                }
            ),
            logger=logging.getLogger("bursts-trial-safe"),
            analysis_mode="trial_ml_safe",
            train_mask=None,
            spatial_modes=["global"],
        )

        with self.assertRaisesRegex(ValueError, "train_mask"):
            extract_burst_features(ctx, ["alpha"])

    def test_bursts_resting_state_rejects_empty_target_window(self):
        times = np.arange(200, dtype=float) / 100.0
        analysis_mask = np.ones(times.size, dtype=bool)
        empty_mask = np.zeros(times.size, dtype=bool)
        windows = TimeWindows(
            masks={"analysis": analysis_mask, "active": empty_mask},
            ranges={"analysis": (0.0, 2.0), "active": (0.0, 2.0)},
            times=times,
            name="active",
        )
        data = np.zeros((2, 2, times.size), dtype=float)
        data[:, :, 50:80] = 5.0
        band = BandData(
            band="alpha",
            fmin=8.0,
            fmax=12.0,
            filtered=data.copy(),
            analytic=data.astype(np.complex128),
            envelope=data.copy(),
            phase=np.zeros_like(data),
            power=data.copy(),
        )
        precomputed = PrecomputedData(
            data=data,
            times=times,
            sfreq=100.0,
            ch_names=["C3", "C4"],
            picks=np.arange(2),
            band_data={"alpha": band},
            windows=windows,
            logger=logging.getLogger("bursts-rest"),
            train_mask=None,
        )
        ctx = SimpleNamespace(
            precomputed=precomputed,
            config=DotConfig(
                {
                    "feature_engineering": {
                        "task_is_rest": True,
                        "bursts": {
                            "bands": ["alpha"],
                            "threshold_reference": "trial",
                            "threshold_method": "percentile",
                            "threshold_percentile": 50.0,
                            "min_duration_ms": 0.0,
                            "min_cycles": 0.0,
                        }
                    },
                }
            ),
            logger=logging.getLogger("bursts-rest"),
            analysis_mode="group_stats",
            train_mask=None,
            spatial_modes=["global"],
        )

        with self.assertRaisesRegex(ValueError, "target window 'active' does not contain valid samples"):
            extract_burst_features(ctx, ["alpha"])

    def test_condition_burst_thresholds_require_condition_support(self):
        from eeg_pipeline.analysis.features.bursts import _compute_thresholds_condition

        baseline = np.ones((4, 1, 8), dtype=float)
        condition_labels = np.array(["a", "a", "b", "b"], dtype=object)

        with self.assertRaisesRegex(ValueError, "condition-specific burst thresholds"):
            _compute_thresholds_condition(
                baseline,
                condition_labels,
                method="percentile",
                threshold_z=2.0,
                threshold_percentile=75.0,
                min_trials_per_condition=3,
            )

    def test_condition_burst_threshold_trainmask_requires_training_support(self):
        from eeg_pipeline.analysis.features.bursts import _compute_thresholds_condition_trainmask

        baseline = np.ones((4, 1, 8), dtype=float)
        condition_labels = np.array(["a", "a", "b", "b"], dtype=object)
        train_mask = np.zeros(4, dtype=bool)

        with self.assertRaisesRegex(ValueError, "training mask"):
            _compute_thresholds_condition_trainmask(
                baseline,
                condition_labels,
                train_mask,
                method="percentile",
                threshold_z=2.0,
                threshold_percentile=75.0,
                min_trials_per_condition=2,
            )

    def test_family_spatial_transform_resolution_surfaces_errors(self):
        with patch(
            "eeg_pipeline.analysis.features.preparation._get_spatial_transform_type",
            side_effect=ValueError("bad transform config"),
        ):
            with self.assertRaisesRegex(ValueError, "bad transform config"):
                _get_family_spatial_transform(DotConfig({}), "pac")

    def test_itpc_fold_global_requires_train_mask_for_tfr_path(self):
        data = np.ones((4, 2, 3, 10), dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, "train_mask"):
            _compute_itpc_map_by_method(
                data,
                "fold_global",
                train_mask=None,
                analysis_mode="group_stats",
                logger=logging.getLogger("itpc-fold-global"),
            )

    def test_itpc_fold_global_requires_train_mask_for_precomputed_path(self):
        times = np.linspace(-0.2, 0.3, 20)
        mask = np.ones(times.size, dtype=bool)
        windows = TimeWindows(
            baseline_mask=mask,
            masks={"baseline": mask, "active": mask},
            ranges={"baseline": (-0.2, 0.3), "active": (-0.2, 0.3)},
            times=times,
        )
        phase = np.zeros((4, 2, times.size), dtype=float)
        band = BandData(
            band="alpha",
            fmin=8.0,
            fmax=12.0,
            filtered=np.ones((4, 2, times.size), dtype=float),
            analytic=np.ones((4, 2, times.size), dtype=np.complex128),
            envelope=np.ones((4, 2, times.size), dtype=float),
            phase=phase,
            power=np.ones((4, 2, times.size), dtype=float),
        )
        precomputed = PrecomputedData(
            data=np.ones((4, 2, times.size), dtype=float),
            times=times,
            sfreq=100.0,
            ch_names=["C3", "C4"],
            picks=np.arange(2),
            band_data={"alpha": band},
            windows=windows,
            logger=logging.getLogger("itpc-precomputed-fold-global"),
            config=DotConfig(
                {"feature_engineering": {"itpc": {"method": "fold_global"}}}
            ),
            train_mask=None,
        )

        with self.assertRaisesRegex(ValueError, "train_mask"):
            extract_itpc_from_precomputed(precomputed)

    def test_source_localization_preflights_missing_trans_without_auto_create(self):
        fs_root = Path(tempfile.mkdtemp())
        (fs_root / "sub-0001" / "bem").mkdir(parents=True, exist_ok=True)

        config = DotConfig(
            {
                "project": {"task": "thermalactive"},
                "paths": {
                    "bids_root": str(fs_root / "bids"),
                    "deriv_root": str(fs_root / "derivatives"),
                },
                "feature_engineering": {
                    "sourcelocalization": {
                        "mode": "fmri_informed",
                        "method": "lcmv",
                        "subjects_dir": str(fs_root),
                        "fmri": {
                            "enabled": True,
                            "contrast": {"enabled": True},
                        },
                        "bem_generation": {
                            "create_trans": False,
                            "allow_identity_trans": False,
                        },
                    }
                },
            }
        )
        ctx = SimpleNamespace(subject="0001")

        with patch(
            "eeg_pipeline.analysis.features.source_localization._load_fmri_constraint_config"
        ) as mock_load_fmri_cfg:
            with self.assertRaisesRegex(ValueError, "requires feature_engineering.sourcelocalization.trans"):
                _load_source_localization_config(ctx, config, method="lcmv")

        mock_load_fmri_cfg.assert_not_called()

    def test_source_localization_skips_band_when_duration_too_short(self):
        n_epochs = 4
        roi_data = np.random.default_rng(23).standard_normal((n_epochs, 2, 20))  # 0.2 s @ 100 Hz

        fmri_cfg = SimpleNamespace(
            enabled=False,
            provenance="independent",
            require_provenance=False,
        )
        src_cfg = SimpleNamespace(
            method="lcmv",
            fmri_cfg=fmri_cfg,
            subjects_dir=None,
            trans_path=None,
            bem_path=None,
            parcellation="aparc",
            spacing="oct6",
            subject="fsaverage",
            mindist_mm=5.0,
            lcmv_reg=0.05,
            eloreta_loose=0.2,
            eloreta_depth=0.8,
            eloreta_snr=3.0,
            allow_template_fallback=True,
            save_stc=False,
        )

        ctx = SimpleNamespace(
            epochs=_EpochStub(n_epochs, sfreq=100.0),
            config=DotConfig(
                {
                    "feature_engineering": {
                        "sourcelocalization": {
                            "min_cycles_per_band": 3.0,
                        }
                    }
                }
            ),
            logger=logging.getLogger("src-loc-duration-guard"),
            analysis_mode="group_stats",
            train_mask=None,
            frequency_bands={"alpha": (8.0, 12.0)},
            name="active",
        )

        with patch(
            "eeg_pipeline.analysis.features.source_localization._load_source_localization_config",
            return_value=src_cfg,
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._setup_forward_model",
            return_value=("fwd", "src", None),
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._compute_lcmv_source_estimates",
            return_value=(["stc"] * n_epochs, None),
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._extract_roi_timecourses",
            return_value=roi_data,
        ), patch(
            "mne.read_labels_from_annot",
            return_value=[SimpleNamespace(name="roi1"), SimpleNamespace(name="roi2")],
        ):
            df, cols = extract_source_localization_features(
                ctx,
                bands=["alpha"],
                method="lcmv",
            )

        self.assertEqual(cols, [])
        self.assertTrue(df.empty)

    def test_source_localization_resting_state_rejects_empty_target_window(self):
        n_epochs = 4
        n_times = 80
        roi_data = np.random.default_rng(29).standard_normal((n_epochs, 2, n_times))
        analysis_mask = np.zeros(n_times, dtype=bool)
        analysis_mask[:40] = True
        empty_mask = np.zeros(n_times, dtype=bool)

        fmri_cfg = SimpleNamespace(
            enabled=False,
            provenance="independent",
            require_provenance=False,
        )
        src_cfg = SimpleNamespace(
            method="lcmv",
            fmri_cfg=fmri_cfg,
            subjects_dir=None,
            trans_path=None,
            bem_path=None,
            parcellation="aparc",
            spacing="oct6",
            subject="fsaverage",
            mindist_mm=5.0,
            lcmv_reg=0.05,
            eloreta_loose=0.2,
            eloreta_depth=0.8,
            eloreta_snr=3.0,
            allow_template_fallback=True,
            save_stc=False,
        )

        ctx = SimpleNamespace(
            epochs=_EpochStub(n_epochs, sfreq=100.0),
            windows=TimeWindows(
                masks={"analysis": analysis_mask, "active": empty_mask},
                ranges={"analysis": (0.0, 0.4), "active": (0.0, 0.8)},
                times=np.arange(n_times, dtype=float) / 100.0,
                name="active",
            ),
            config=DotConfig(
                {
                    "feature_engineering": {
                        "task_is_rest": True,
                        "sourcelocalization": {"min_cycles_per_band": 3.0},
                    }
                }
            ),
            logger=logging.getLogger("src-loc-rest"),
            analysis_mode="group_stats",
            train_mask=None,
            frequency_bands={"alpha": (8.0, 12.0)},
            name="active",
        )

        captured = {"n_times_power": None, "n_times_env": None}

        def _fake_compute_roi_power(data, *_args, **_kwargs):
            captured["n_times_power"] = int(np.asarray(data).shape[-1])
            return np.ones((n_epochs, 2), dtype=float)

        def _fake_compute_roi_envelope(data, *_args, **_kwargs):
            captured["n_times_env"] = int(np.asarray(data).shape[-1])
            return np.ones((n_epochs, 2, int(np.asarray(data).shape[-1])), dtype=float)
        with patch(
            "eeg_pipeline.analysis.features.source_localization._load_source_localization_config",
            return_value=src_cfg,
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._setup_forward_model",
            return_value=("fwd", "src", None),
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._compute_lcmv_source_estimates",
            return_value=(["stc"] * n_epochs, None),
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._extract_roi_timecourses",
            return_value=roi_data,
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._compute_roi_power",
            side_effect=_fake_compute_roi_power,
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._compute_roi_envelope",
            side_effect=_fake_compute_roi_envelope,
        ), patch(
            "mne.read_labels_from_annot",
            return_value=[SimpleNamespace(name="roi1"), SimpleNamespace(name="roi2")],
        ):
            with self.assertRaisesRegex(ValueError, "target window 'active' does not contain valid samples"):
                extract_source_localization_features(
                    ctx,
                    bands=["alpha"],
                    method="lcmv",
                )

        self.assertIsNone(captured["n_times_power"])
        self.assertIsNone(captured["n_times_env"])

    def test_source_localization_resting_state_save_stc_rejects_empty_target_window(self):
        n_epochs = 3
        n_times = 80
        analysis_mask = np.zeros(n_times, dtype=bool)
        analysis_mask[:40] = True
        empty_mask = np.zeros(n_times, dtype=bool)
        roi_data = np.random.default_rng(707).standard_normal((n_epochs, 2, n_times))
        stcs = [_StcStub(np.ones((3, n_times), dtype=float)) for _ in range(n_epochs)]

        fmri_cfg = SimpleNamespace(enabled=False, provenance="independent", require_provenance=False)
        src_cfg = SimpleNamespace(
            method="lcmv",
            fmri_cfg=fmri_cfg,
            subjects_dir=None,
            trans_path=None,
            bem_path=None,
            parcellation="aparc",
            spacing="oct6",
            subject="fsaverage",
            mindist_mm=5.0,
            lcmv_reg=0.05,
            eloreta_loose=0.2,
            eloreta_depth=0.8,
            eloreta_snr=3.0,
            allow_template_fallback=True,
            save_stc=True,
            mode="eeg_only",
        )
        ctx = SimpleNamespace(
            epochs=_EpochStub(n_epochs, sfreq=100.0, n_times=n_times),
            windows=TimeWindows(
                masks={"analysis": analysis_mask, "active": empty_mask},
                ranges={"analysis": (0.0, 0.4), "active": (0.0, 0.8)},
                times=np.arange(n_times, dtype=float) / 100.0,
                name="active",
            ),
            aligned_events=pd.DataFrame({"condition": ["A", "A", "B"]}),
            config=DotConfig(
                {
                    "feature_engineering": {
                        "task_is_rest": True,
                        "sourcelocalization": {"min_cycles_per_band": 3.0},
                    }
                }
            ),
            deriv_root=tempfile.mkdtemp(),
            subject="01",
            task="rest",
            logger=logging.getLogger("src-loc-rest-stc"),
            analysis_mode="group_stats",
            train_mask=None,
            frequency_bands={"alpha": (8.0, 12.0)},
            name="active",
        )

        captured_lengths: list[int] = []

        def _fake_compute_roi_power(data, *_args, **_kwargs):
            captured_lengths.append(int(np.asarray(data).shape[-1]))
            if np.asarray(data).ndim == 3 and np.asarray(data).shape[1] == 3:
                return np.ones((np.asarray(data).shape[0], 3), dtype=float)
            return np.ones((n_epochs, 2), dtype=float)

        with patch(
            "eeg_pipeline.analysis.features.source_localization._load_source_localization_config",
            return_value=src_cfg,
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._load_source_contrast_config",
            return_value=SimpleNamespace(condition_column="condition"),
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._setup_forward_model",
            return_value=("fwd", "src", None),
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._compute_lcmv_source_estimates",
            return_value=(stcs, None),
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._extract_roi_timecourses",
            return_value=roi_data,
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._compute_roi_power",
            side_effect=_fake_compute_roi_power,
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._compute_roi_envelope",
            return_value=np.ones((n_epochs, 2, 40), dtype=float),
        ), patch(
            "mne.write_source_spaces",
            return_value=None,
        ), patch(
            "mne.read_labels_from_annot",
            return_value=[SimpleNamespace(name="roi1"), SimpleNamespace(name="roi2")],
        ):
            with self.assertRaisesRegex(ValueError, "target window 'active' does not contain valid samples"):
                extract_source_localization_features(ctx, bands=["alpha"], method="lcmv")

        self.assertFalse(captured_lengths)

    def test_source_localization_rejects_non_finite_roi_power_aggregates(self):
        n_epochs = 3
        roi_data = np.ones((n_epochs, 2, 100), dtype=float)

        fmri_cfg = SimpleNamespace(
            enabled=False,
            provenance="independent",
            require_provenance=False,
        )
        src_cfg = SimpleNamespace(
            method="lcmv",
            fmri_cfg=fmri_cfg,
            subjects_dir=None,
            trans_path=None,
            bem_path=None,
            parcellation="aparc",
            spacing="oct6",
            subject="fsaverage",
            mindist_mm=5.0,
            lcmv_reg=0.05,
            eloreta_loose=0.2,
            eloreta_depth=0.8,
            eloreta_snr=3.0,
            allow_template_fallback=True,
            save_stc=False,
        )
        ctx = SimpleNamespace(
            epochs=_EpochStub(n_epochs, sfreq=100.0),
            config=DotConfig(
                {
                    "feature_engineering": {
                        "sourcelocalization": {"min_cycles_per_band": 3.0},
                    }
                }
            ),
            logger=logging.getLogger("src-loc-non-finite"),
            analysis_mode="group_stats",
            train_mask=None,
            frequency_bands={"alpha": (8.0, 12.0)},
            name="active",
        )

        with patch(
            "eeg_pipeline.analysis.features.source_localization._load_source_localization_config",
            return_value=src_cfg,
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._setup_forward_model",
            return_value=("fwd", "src", None),
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._compute_lcmv_source_estimates",
            return_value=(["stc"] * n_epochs, None),
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._extract_roi_timecourses",
            return_value=roi_data,
        ), patch(
            "mne.read_labels_from_annot",
            return_value=[SimpleNamespace(name="roi1"), SimpleNamespace(name="roi2")],
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._compute_roi_power",
            return_value=np.full((n_epochs, 2), np.nan, dtype=float),
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._compute_roi_envelope",
            return_value=np.ones((n_epochs, 2, 100), dtype=float),
        ):
            with self.assertRaisesRegex(ValueError, "non-finite source-localization power features"):
                extract_source_localization_features(
                    ctx,
                    bands=["alpha"],
                    method="lcmv",
                )

    def test_fmri_source_feature_helper_rejects_non_finite_family_aggregates(self):
        records = [{} for _ in range(2)]
        feature_cols: list[str] = []
        roi_data = np.ones((2, 2, 50), dtype=float)

        with patch(
            "eeg_pipeline.analysis.features.source_localization._compute_roi_power",
            return_value=np.full((2, 2), np.nan, dtype=float),
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._compute_roi_envelope",
            return_value=np.ones((2, 2, 50), dtype=float),
        ):
            with self.assertRaisesRegex(ValueError, "non-finite source-localization power features"):
                _append_source_band_family_features(
                    records=records,
                    feature_cols=feature_cols,
                    n_epochs=2,
                    roi_data=roi_data,
                    label_names=["roi1", "roi2"],
                    sfreq=100.0,
                    fmin=8.0,
                    fmax=12.0,
                    segment_label="active",
                    method="lcmv",
                    band="alpha",
                    family_prefix="atlas",
                )

    def test_fmri_output_space_controls_feature_families(self):
        n_epochs = 3
        n_times = 100
        stcs = [
            SimpleNamespace(
                data=np.ones((3, n_times), dtype=float),
                vertices=[np.array([0, 1, 2], dtype=int)],
            )
            for _ in range(n_epochs)
        ]

        voxel_selection = FMRIVoxelSelection(
            selected_voxels_ijk=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=int),
            selected_coords_m=np.array(
                [[0.000, 0.000, 0.000], [0.000, 0.000, 0.001], [0.000, 0.001, 0.000]],
                dtype=float,
            ),
            cluster_indices={"fmri_c01_peak5p00": [0, 1], "fmri_c02_peak4p00": [2]},
            cluster_voxel_counts={"fmri_c01_peak5p00": 2, "fmri_c02_peak4p00": 1},
            atlas_indices={"aparc_aseg_id17": [0, 2], "aparc_aseg_id53": [1]},
            atlas_voxel_counts={"aparc_aseg_id17": 2, "aparc_aseg_id53": 1},
            cluster_to_atlas_counts={
                "fmri_c01_peak5p00": {"aparc_aseg_id17": 1, "aparc_aseg_id53": 1},
                "fmri_c02_peak4p00": {"aparc_aseg_id17": 1},
            },
            dropped_unlabeled_voxels=0,
            matched_reference_path=Path("/tmp/ref.mgz"),
            voxel_volume_mm3=8.0,
        )

        def run_output_space(output_space: str) -> list[str]:
            fmri_cfg = SimpleNamespace(
                enabled=True,
                stats_map_path=Path(__file__),
                provenance="independent",
                require_provenance=False,
                allow_same_dataset_provenance=False,
                output_space=output_space,
            )
            src_cfg = SimpleNamespace(
                method="lcmv",
                fmri_cfg=fmri_cfg,
                subjects_dir="/tmp/freesurfer",
                subject="sub-0001",
                mindist_mm=5.0,
                lcmv_reg=0.05,
                eloreta_loose=1.0,
                eloreta_depth=0.8,
                eloreta_snr=3.0,
                save_stc=False,
                mode="fmri_informed",
            )
            ctx = SimpleNamespace(
                epochs=_EpochStub(n_epochs, sfreq=100.0),
                config=DotConfig(
                    {
                        "feature_engineering": {
                            "sourcelocalization": {"min_cycles_per_band": 3.0},
                        }
                    }
                ),
                logger=logging.getLogger(f"src-loc-output-space-{output_space}"),
                analysis_mode="group_stats",
                train_mask=None,
                frequency_bands={"alpha": (8.0, 12.0)},
                name="active",
                deriv_root="/tmp",
                subject="0001",
                task="task",
            )

            seen_families: list[str] = []

            def _fake_append(
                *,
                records,
                feature_cols,
                n_epochs,
                family_prefix,
                **_kwargs,
            ):
                seen_families.append(str(family_prefix))
                col_name = f"src_active_lcmv_alpha_{family_prefix}_dummy"
                feature_cols.append(col_name)
                for epoch_idx in range(n_epochs):
                    records[epoch_idx][col_name] = float(epoch_idx)

            with (
                patch(
                    "eeg_pipeline.analysis.features.source_localization._load_source_localization_config",
                    return_value=src_cfg,
                ),
                patch(
                    "fmri_pipeline.analysis.bem_generation.ensure_bem_and_trans_files",
                    return_value=(Path("/tmp/trans.fif"), Path("/tmp/bem.fif"), Path("/tmp/bem-sol.fif")),
                ),
                patch(
                    "eeg_pipeline.analysis.features.source_localization._select_fmri_constrained_voxels",
                    return_value=voxel_selection,
                ),
                patch(
                    "eeg_pipeline.analysis.features.source_localization._setup_volume_source_space_from_points_configured",
                    return_value=("fwd", "src"),
                ),
                patch(
                    "eeg_pipeline.analysis.features.source_localization._compute_lcmv_source_estimates",
                    return_value=(stcs, None),
                ),
                patch(
                    "eeg_pipeline.analysis.features.source_localization._compute_roi_timecourses_from_row_indices",
                    side_effect=lambda **kwargs: np.ones(
                        (n_epochs, len(kwargs["roi_names"]), n_times), dtype=float
                    ),
                ),
                patch(
                    "eeg_pipeline.analysis.features.source_localization._validate_source_localization_duration",
                    return_value=True,
                ),
                patch(
                    "eeg_pipeline.analysis.features.source_localization._append_source_band_family_features",
                    side_effect=_fake_append,
                ),
                patch(
                    "eeg_pipeline.analysis.features.source_localization._write_fmri_constraint_metadata_sidecar",
                    return_value=None,
                ),
            ):
                _df, cols = extract_source_localization_features(
                    ctx,
                    bands=["alpha"],
                    method="lcmv",
                )
            self.assertEqual(len(cols), len(set(cols)))
            return seen_families

        self.assertEqual(run_output_space("cluster"), ["fmri_cluster"])
        self.assertEqual(run_output_space("atlas"), ["atlas"])
        self.assertEqual(run_output_space("dual"), ["fmri_cluster", "atlas"])

    def test_fmri_atlas_columns_align_across_subjects(self):
        n_epochs = 3
        n_times = 100
        stcs = [
            SimpleNamespace(
                data=np.ones((3, n_times), dtype=float),
                vertices=[np.array([0, 1, 2], dtype=int)],
            )
            for _ in range(n_epochs)
        ]

        def make_selection(cluster_name: str) -> FMRIVoxelSelection:
            return FMRIVoxelSelection(
                selected_voxels_ijk=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=int),
                selected_coords_m=np.array(
                    [[0.000, 0.000, 0.000], [0.000, 0.000, 0.001], [0.000, 0.001, 0.000]],
                    dtype=float,
                ),
                cluster_indices={cluster_name: [0, 1, 2]},
                cluster_voxel_counts={cluster_name: 3},
                atlas_indices={"aparc_aseg_id17": [0, 2], "aparc_aseg_id53": [1]},
                atlas_voxel_counts={"aparc_aseg_id17": 2, "aparc_aseg_id53": 1},
                cluster_to_atlas_counts={
                    cluster_name: {"aparc_aseg_id17": 2, "aparc_aseg_id53": 1}
                },
                dropped_unlabeled_voxels=0,
                matched_reference_path=Path("/tmp/ref.mgz"),
                voxel_volume_mm3=8.0,
            )

        def extract_columns(subject_label: str, selection: FMRIVoxelSelection) -> list[str]:
            fmri_cfg = SimpleNamespace(
                enabled=True,
                stats_map_path=Path(__file__),
                provenance="independent",
                require_provenance=False,
                allow_same_dataset_provenance=False,
                output_space="atlas",
            )
            src_cfg = SimpleNamespace(
                method="lcmv",
                fmri_cfg=fmri_cfg,
                subjects_dir="/tmp/freesurfer",
                subject=subject_label,
                mindist_mm=5.0,
                lcmv_reg=0.05,
                eloreta_loose=1.0,
                eloreta_depth=0.8,
                eloreta_snr=3.0,
                save_stc=False,
                mode="fmri_informed",
            )
            ctx = SimpleNamespace(
                epochs=_EpochStub(n_epochs, sfreq=100.0),
                config=DotConfig(
                    {
                        "feature_engineering": {
                            "sourcelocalization": {"min_cycles_per_band": 3.0},
                        }
                    }
                ),
                logger=logging.getLogger(f"src-loc-atlas-align-{subject_label}"),
                analysis_mode="group_stats",
                train_mask=None,
                frequency_bands={"alpha": (8.0, 12.0)},
                name="active",
                deriv_root="/tmp",
                subject=subject_label.replace("sub-", ""),
                task="task",
            )

            with (
                patch(
                    "eeg_pipeline.analysis.features.source_localization._load_source_localization_config",
                    return_value=src_cfg,
                ),
                patch(
                    "fmri_pipeline.analysis.bem_generation.ensure_bem_and_trans_files",
                    return_value=(Path("/tmp/trans.fif"), Path("/tmp/bem.fif"), Path("/tmp/bem-sol.fif")),
                ),
                patch(
                    "eeg_pipeline.analysis.features.source_localization._select_fmri_constrained_voxels",
                    return_value=selection,
                ),
                patch(
                    "eeg_pipeline.analysis.features.source_localization._setup_volume_source_space_from_points_configured",
                    return_value=("fwd", "src"),
                ),
                patch(
                    "eeg_pipeline.analysis.features.source_localization._compute_lcmv_source_estimates",
                    return_value=(stcs, None),
                ),
                patch(
                    "eeg_pipeline.analysis.features.source_localization._compute_roi_timecourses_from_row_indices",
                    side_effect=lambda **kwargs: np.ones(
                        (n_epochs, len(kwargs["roi_names"]), n_times), dtype=float
                    ),
                ),
                patch(
                    "eeg_pipeline.analysis.features.source_localization._validate_source_localization_duration",
                    return_value=True,
                ),
                patch(
                    "eeg_pipeline.analysis.features.source_localization._compute_roi_power",
                    return_value=np.ones((n_epochs, 2), dtype=float),
                ),
                patch(
                    "eeg_pipeline.analysis.features.source_localization._compute_roi_envelope",
                    return_value=np.ones((n_epochs, 2, n_times), dtype=float),
                ),
                patch(
                    "eeg_pipeline.analysis.features.source_localization._write_fmri_constraint_metadata_sidecar",
                    return_value=None,
                ),
            ):
                _df, cols = extract_source_localization_features(
                    ctx,
                    bands=["alpha"],
                    method="lcmv",
                )
            return cols

        cols_sub_1 = extract_columns("sub-0001", make_selection("fmri_c01_peak5p00"))
        cols_sub_2 = extract_columns("sub-0002", make_selection("fmri_c03_peak6p20"))
        self.assertEqual(cols_sub_1, cols_sub_2)

    def test_fmri_metadata_records_surviving_stc_rows_per_cluster(self):
        n_epochs = 2
        n_times = 50
        stcs = [
            SimpleNamespace(
                data=np.ones((3, n_times), dtype=float),
                vertices=[np.array([0, 1, 2], dtype=int)],
            )
            for _ in range(n_epochs)
        ]
        voxel_selection = FMRIVoxelSelection(
            selected_voxels_ijk=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=int),
            selected_coords_m=np.array(
                [[0.000, 0.000, 0.000], [0.000, 0.000, 0.001], [0.000, 0.001, 0.000]],
                dtype=float,
            ),
            cluster_indices={"fmri_c01_peak5p00": [0, 1], "fmri_c02_peak4p00": [2]},
            cluster_voxel_counts={"fmri_c01_peak5p00": 2, "fmri_c02_peak4p00": 1},
            atlas_indices={},
            atlas_voxel_counts={},
            cluster_to_atlas_counts={},
            dropped_unlabeled_voxels=0,
            matched_reference_path=Path("/tmp/ref.mgz"),
            voxel_volume_mm3=8.0,
        )
        fmri_cfg = SimpleNamespace(
            enabled=True,
            stats_map_path=Path(__file__),
            provenance="independent",
            require_provenance=False,
            allow_same_dataset_provenance=False,
            output_space="cluster",
        )
        src_cfg = SimpleNamespace(
            method="lcmv",
            fmri_cfg=fmri_cfg,
            subjects_dir="/tmp/freesurfer",
            subject="sub-0001",
            mindist_mm=5.0,
            lcmv_reg=0.05,
            eloreta_loose=1.0,
            eloreta_depth=0.8,
            eloreta_snr=3.0,
            save_stc=False,
            mode="fmri_informed",
        )
        ctx = SimpleNamespace(
            epochs=_EpochStub(n_epochs, sfreq=100.0),
            config=DotConfig(
                {
                    "feature_engineering": {
                        "sourcelocalization": {"min_cycles_per_band": 3.0},
                    }
                }
            ),
            logger=logging.getLogger("src-loc-cluster-rows"),
            analysis_mode="group_stats",
            train_mask=None,
            frequency_bands={"alpha": (8.0, 12.0)},
            name="active",
            deriv_root="/tmp",
            subject="0001",
            task="task",
        )
        captured_payload: dict[str, object] = {}

        def _capture_metadata(**kwargs):
            captured_payload.update(kwargs["payload"])
            return None

        with (
            patch(
                "eeg_pipeline.analysis.features.source_localization._load_source_localization_config",
                return_value=src_cfg,
            ),
            patch(
                "fmri_pipeline.analysis.bem_generation.ensure_bem_and_trans_files",
                return_value=(Path("/tmp/trans.fif"), Path("/tmp/bem.fif"), Path("/tmp/bem-sol.fif")),
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._select_fmri_constrained_voxels",
                return_value=voxel_selection,
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._setup_volume_source_space_from_points_configured",
                return_value=("fwd", "src"),
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._compute_lcmv_source_estimates",
                return_value=(stcs, None),
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._compute_roi_timecourses_from_row_indices",
                side_effect=lambda **kwargs: np.ones(
                    (n_epochs, len(kwargs["roi_names"]), n_times), dtype=float
                ),
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._validate_source_localization_duration",
                return_value=True,
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._compute_roi_power",
                return_value=np.ones((n_epochs, 2), dtype=float),
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._compute_roi_envelope",
                return_value=np.ones((n_epochs, 2, n_times), dtype=float),
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._write_fmri_constraint_metadata_sidecar",
                side_effect=_capture_metadata,
            ),
        ):
            extract_source_localization_features(
                ctx,
                bands=["alpha"],
                method="lcmv",
            )

        roi_survival = captured_payload["roi_survival"]["fmri_cluster"]
        self.assertEqual(
            roi_survival["surviving_stc_rows_per_roi"],
            {
                "c01_peak5p00": [0, 1],
                "c02_peak4p00": [2],
            },
        )

    def test_source_localization_blocks_same_dataset_provenance_by_default(self):
        fmri_cfg = SimpleNamespace(
            enabled=True,
            provenance="same_dataset",
            require_provenance=False,
            allow_same_dataset_provenance=False,
        )
        src_cfg = SimpleNamespace(method="eloreta", fmri_cfg=fmri_cfg)
        ctx = SimpleNamespace(
            epochs=_EpochStub(5),
            config=DotConfig(
                {
                    "feature_engineering": {
                        "sourcelocalization": {"min_cycles_per_band": 3.0},
                    }
                }
            ),
            logger=logging.getLogger("src-loc-provenance"),
            analysis_mode="group_stats",
            train_mask=None,
            frequency_bands={"alpha": (8.0, 12.0)},
            name="active",
        )

        with patch(
            "eeg_pipeline.analysis.features.source_localization._load_source_localization_config",
            return_value=src_cfg,
        ):
            with self.assertRaisesRegex(ValueError, "same_dataset"):
                extract_source_localization_features(
                    ctx,
                    bands=["alpha"],
                    method="eloreta",
                )

    def test_source_connectivity_blocks_same_dataset_provenance_by_default(self):
        fmri_cfg = SimpleNamespace(
            enabled=True,
            provenance="same_dataset",
            require_provenance=False,
            allow_same_dataset_provenance=False,
        )
        src_cfg = SimpleNamespace(method="eloreta", fmri_cfg=fmri_cfg)
        ctx = SimpleNamespace(
            epochs=_EpochStub(5),
            config=DotConfig(
                {
                    "feature_engineering": {
                        "sourcelocalization": {"min_cycles_per_band": 3.0},
                    }
                }
            ),
            logger=logging.getLogger("src-conn-provenance"),
            analysis_mode="group_stats",
            train_mask=None,
            frequency_bands={"alpha": (8.0, 12.0)},
            name="active",
        )

        with patch(
            "eeg_pipeline.analysis.features.source_localization._load_source_localization_config",
            return_value=src_cfg,
        ):
            with self.assertRaisesRegex(ValueError, "same_dataset"):
                extract_source_connectivity_features(
                    ctx,
                    bands=["alpha"],
                    method="eloreta",
                    connectivity_method="wpli",
                )

    def test_source_connectivity_rejects_nonfinite_global_edges(self):
        fmri_cfg = SimpleNamespace(
            enabled=False,
            provenance="independent",
            require_provenance=False,
        )
        src_cfg = SimpleNamespace(
            method="eloreta",
            fmri_cfg=fmri_cfg,
            subjects_dir=None,
            trans_path=None,
            bem_path=None,
            subject="fsaverage",
            parcellation="aparc",
            allow_template_fallback=True,
            eloreta_loose=0.2,
            eloreta_depth=0.8,
            eloreta_snr=3.0,
        )
        ctx = SimpleNamespace(
            epochs=_EpochStub(4, sfreq=100.0, n_times=120),
            config=DotConfig(
                {
                    "feature_engineering": {
                        "sourcelocalization": {"min_cycles_per_band": 1.0},
                    }
                }
            ),
            logger=logging.getLogger("src-conn-nonfinite-edges"),
            analysis_mode="group_stats",
            train_mask=None,
            frequency_bands={"alpha": (8.0, 12.0)},
            name="active",
        )

        class _Connectivity:
            def get_data(self):
                return np.array([0.2, np.nan, 0.4], dtype=float)

        fake_connectivity = types.SimpleNamespace(
            spectral_connectivity_epochs=lambda *_args, **_kwargs: _Connectivity(),
            envelope_correlation=lambda *_args, **_kwargs: _Connectivity(),
        )

        with patch.dict(sys.modules, {"mne_connectivity": fake_connectivity}), patch(
            "eeg_pipeline.analysis.features.source_localization._load_source_localization_config",
            return_value=src_cfg,
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._setup_forward_model",
            return_value=(object(), object(), None),
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._compute_eloreta_source_estimates",
            return_value=([_StcStub(np.ones((2, 120), dtype=float)) for _ in range(4)], object()),
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._extract_roi_timecourses",
            return_value=np.ones((4, 3, 120), dtype=float),
        ), patch(
            "eeg_pipeline.analysis.features.source_localization._validate_source_connectivity_duration",
            return_value=True,
        ), patch(
            "mne.read_labels_from_annot",
            return_value=[SimpleNamespace(name="roi_a"), SimpleNamespace(name="roi_b"), SimpleNamespace(name="roi_c")],
        ):
            with self.assertRaisesRegex(ValueError, "non-finite source connectivity"):
                extract_source_connectivity_features(
                    ctx,
                    bands=["alpha"],
                    method="eloreta",
                    connectivity_method="wpli",
                )

    def test_source_contrast_config_rejects_non_boolean_emit_welch_stats(self):
        cfg = DotConfig(
            {
                "feature_engineering": {
                    "sourcelocalization": {
                        "contrast": {
                            "enabled": True,
                            "condition_column": "trial_type",
                            "condition_a": "A",
                            "condition_b": "B",
                            "emit_welch_stats": "false",
                        }
                    }
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "emit_welch_stats"):
            _load_source_contrast_config(cfg)

    def test_source_contrast_config_rejects_trial_level_welch_stats(self):
        cfg = DotConfig(
            {
                "feature_engineering": {
                    "sourcelocalization": {
                        "contrast": {
                            "enabled": True,
                            "condition_column": "trial_type",
                            "condition_a": "A",
                            "condition_b": "B",
                            "emit_welch_stats": True,
                        }
                    }
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "scientifically valid"):
            _load_source_contrast_config(cfg)

    def test_source_contrast_config_rejects_rest_mode(self):
        cfg = DotConfig(
            {
                "feature_engineering": {
                    "task_is_rest": True,
                    "sourcelocalization": {
                        "contrast": {
                            "enabled": True,
                            "condition_column": "trial_type",
                            "condition_a": "A",
                            "condition_b": "B",
                            "min_trials_per_condition": 2,
                            "emit_welch_stats": False,
                        }
                    },
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "not scientifically valid when feature_engineering.task_is_rest=true"):
            _load_source_contrast_config(cfg)

    def test_source_contrast_extracts_subject_level_row_without_welch(self):
        cfg = DotConfig(
            {
                "feature_engineering": {
                    "sourcelocalization": {
                        "contrast": {
                            "enabled": True,
                            "condition_column": "trial_type",
                            "condition_a": "A",
                            "condition_b": "B",
                            "min_trials_per_condition": 2,
                            "emit_welch_stats": False,
                        }
                    }
                }
            }
        )
        ctx = SimpleNamespace(
            config=cfg,
            aligned_events=pd.DataFrame({"trial_type": ["A", "A", "B", "B"]}),
            logger=logging.getLogger("src-contrast-no-welch"),
        )
        source_df = pd.DataFrame({"src_full_lcmv_alpha_global_power": [1.0, 2.0, 3.0, 4.0]})

        contrast_df, contrast_cols = extract_source_contrast_features(
            ctx,
            source_df,
            list(source_df.columns),
        )

        self.assertEqual(len(contrast_df), 1)
        self.assertIn(
            "sourcecontrast_src_full_lcmv_alpha_global_power_delta_A_minus_B",
            contrast_cols,
        )
        self.assertNotIn("sourcecontrast_src_full_lcmv_alpha_global_power_welch_t", contrast_cols)
        self.assertEqual(int(contrast_df["sourcecontrast_n_trials_A"].iloc[0]), 2)
        self.assertEqual(int(contrast_df["sourcecontrast_n_trials_B"].iloc[0]), 2)

    def test_source_contrast_errors_on_feature_token_collision(self):
        cfg = DotConfig(
            {
                "feature_engineering": {
                    "sourcelocalization": {
                        "contrast": {
                            "enabled": True,
                            "condition_column": "trial_type",
                            "condition_a": "A",
                            "condition_b": "B",
                            "min_trials_per_condition": 2,
                            "emit_welch_stats": False,
                        }
                    }
                }
            }
        )
        ctx = SimpleNamespace(
            config=cfg,
            aligned_events=pd.DataFrame({"trial_type": ["A", "A", "B", "B"]}),
            logger=logging.getLogger("src-contrast-token-collision"),
        )
        source_df = pd.DataFrame(
            {
                "src-a": [1.0, 1.0, 1.0, 1.0],
                "src a": [2.0, 2.0, 2.0, 2.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "naming collision"):
            extract_source_contrast_features(ctx, source_df, list(source_df.columns))

    def test_eloreta_rejects_normal_orientation_for_volume_sources(self):
        fwd = {"src": [{"type": "vol"}]}
        with self.assertRaisesRegex(ValueError, "pick_ori='normal'"):
            _compute_eloreta_source_estimates(
                epochs=object(),
                fwd=fwd,
                loose=1.0,
                pick_ori="normal",
            )

    def test_source_roi_power_rejects_band_edge_at_nyquist(self):
        roi_data = np.random.default_rng(9).standard_normal((2, 1, 200))
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            _compute_roi_power(
                roi_data=roi_data,
                sfreq=100.0,
                fmin=8.0,
                fmax=50.0,
            )

    def test_source_roi_envelope_rejects_band_edge_at_nyquist(self):
        roi_data = np.random.default_rng(10).standard_normal((2, 1, 200))
        with self.assertRaisesRegex(ValueError, "Invalid bandpass range"):
            _compute_roi_envelope(
                roi_data=roi_data,
                sfreq=100.0,
                fmin=8.0,
                fmax=50.0,
            )

    def test_source_label_extraction_errors_on_empty_labels(self):
        stc = SimpleNamespace(data=np.zeros((2, 20), dtype=float))
        with patch("mne.extract_label_time_course", side_effect=ValueError("empty label")):
            with self.assertRaisesRegex(ValueError, "labels have no vertices"):
                _extract_roi_timecourses(
                    stcs=[stc],
                    labels=[SimpleNamespace(name="roi1")],
                    src=object(),
                    mode="mean_flip",
                )

    def test_fmri_roi_vertex_mapping_errors_when_vertices_are_missing(self):
        stc = SimpleNamespace(
            data=np.zeros((2, 20), dtype=float),
            vertices=[np.array([0, 1], dtype=int)],
        )
        with self.assertRaisesRegex(ValueError, "no surviving vertices"):
            _extract_roi_timecourses_from_vertex_indices(
                stcs=[stc],
                roi_indices={"roi1": [2]},
            )

    def test_fmri_roi_vertex_mapping_drops_empty_rois_when_others_survive(self):
        stc_a = SimpleNamespace(
            data=np.array(
                [
                    [1.0, 3.0],
                    [2.0, 4.0],
                    [5.0, 7.0],
                ],
                dtype=float,
            ),
            vertices=[np.array([0, 1, 2], dtype=int)],
        )
        stc_b = SimpleNamespace(
            data=np.array(
                [
                    [2.0, 4.0],
                    [4.0, 6.0],
                    [6.0, 8.0],
                ],
                dtype=float,
            ),
            vertices=[np.array([0, 1, 2], dtype=int)],
        )

        roi_data, roi_names = _extract_roi_timecourses_from_vertex_indices(
            stcs=[stc_a, stc_b],
            roi_indices={
                "roi_keep": [1, 2],
                "roi_drop": [99],
            },
            logger=logging.getLogger("src-vertex-partial"),
        )

        self.assertEqual(roi_names, ["roi_keep"])
        self.assertEqual(tuple(roi_data.shape), (2, 1, 2))
        np.testing.assert_allclose(roi_data[0, 0, :], np.array([3.5, 5.5], dtype=float))
        np.testing.assert_allclose(roi_data[1, 0, :], np.array([5.0, 7.0], dtype=float))

    def test_tfr_baseline_validation_is_strict_by_default(self):
        tfr = _TFRStub(n_epochs=2, times=np.array([0.0, 0.1], dtype=float))
        aligned_events = pd.DataFrame({"trial": [1, 2]})
        cfg = DotConfig(
            {
                "time_frequency_analysis": {
                    "baseline_window": [-0.2, 0.05],
                    "constants": {"min_samples_for_baseline_validation": 1},
                }
            }
        )
        strict_seen = {"value": None}

        def _fake_validate(baseline_window, logger=None, *, strict=False):
            strict_seen["value"] = bool(strict)
            if strict and float(baseline_window[1]) > 0:
                raise ValueError("invalid baseline window")
            return (float(baseline_window[0]), float(baseline_window[1]))

        with patch(
            "eeg_pipeline.utils.analysis.tfr.validate_baseline_window_pre_stimulus",
            side_effect=_fake_validate,
        ), patch(
            "eeg_pipeline.utils.analysis.tfr.compute_adaptive_n_cycles",
            side_effect=lambda freqs, **_kwargs: np.ones_like(freqs, dtype=float),
        ):
            with self.assertRaisesRegex(ValueError, "invalid baseline window"):
                compute_tfr_for_subject(
                    epochs=object(),
                    aligned_events=aligned_events,
                    subject="sub-01",
                    task="task",
                    config=cfg,
                    deriv_root=Path("."),
                    logger=logging.getLogger("tfr-strict-default"),
                    tfr_computed=tfr,
                )

        self.assertTrue(bool(strict_seen["value"]))

    def test_tfr_baseline_validation_allows_opt_out(self):
        tfr = _TFRStub(n_epochs=2, times=np.array([0.0, 0.1], dtype=float))
        aligned_events = pd.DataFrame({"trial": [1, 2]})
        cfg = DotConfig(
            {
                "time_frequency_analysis": {
                    "baseline_window": [-0.2, 0.05],
                    "strict_baseline_validation": False,
                    "constants": {"min_samples_for_baseline_validation": 1},
                }
            }
        )
        strict_seen = {"value": None}

        def _fake_validate(baseline_window, logger=None, *, strict=False):
            strict_seen["value"] = bool(strict)
            return (float(baseline_window[0]), float(baseline_window[1]))

        with patch(
            "eeg_pipeline.utils.analysis.tfr.validate_baseline_window_pre_stimulus",
            side_effect=_fake_validate,
        ), patch(
            "eeg_pipeline.utils.analysis.tfr.compute_adaptive_n_cycles",
            side_effect=lambda freqs, **_kwargs: np.ones_like(freqs, dtype=float),
        ), patch(
            "eeg_pipeline.utils.analysis.tfr._extract_baseline_power_features",
            return_value=(pd.DataFrame(), []),
        ):
            tfr_out, baseline_df, baseline_cols, _b_start, _b_end = compute_tfr_for_subject(
                epochs=object(),
                aligned_events=aligned_events,
                subject="sub-01",
                task="task",
                config=cfg,
                deriv_root=Path("."),
                logger=logging.getLogger("tfr-strict-optout"),
                tfr_computed=tfr,
            )

        self.assertIs(tfr_out, tfr)
        self.assertTrue(baseline_df.empty)
        self.assertEqual(baseline_cols, [])
        self.assertFalse(bool(strict_seen["value"]))

    def test_tfr_baseline_uses_override_band_definitions(self):
        tfr = _TFRStub(n_epochs=2, times=np.linspace(-1.0, 1.0, 101, dtype=float))
        aligned_events = pd.DataFrame({"trial": [1, 2]})
        cfg = DotConfig(
            {
                "time_frequency_analysis": {
                    "baseline_window": [-0.5, -0.1],
                    "strict_baseline_validation": True,
                    "constants": {"min_samples_for_baseline_validation": 1},
                }
            }
        )
        override_bands = {
            "alpha": [9.0, 11.0],
            "beta": [14.0, 24.0],
        }
        seen = {"bands": None}

        def _fake_extract(_tfr_obj, bands, _baseline_idx, _logger):
            seen["bands"] = dict(bands)
            return pd.DataFrame(index=np.arange(2)), []

        with patch(
            "eeg_pipeline.utils.analysis.tfr._extract_baseline_power_features",
            side_effect=_fake_extract,
        ), patch(
            "eeg_pipeline.utils.analysis.tfr.compute_adaptive_n_cycles",
            side_effect=lambda freqs, **_kwargs: np.ones_like(freqs, dtype=float),
        ):
            _tfr_out, _baseline_df, _baseline_cols, _b_start, _b_end = compute_tfr_for_subject(
                epochs=object(),
                aligned_events=aligned_events,
                subject="sub-01",
                task="task",
                config=cfg,
                deriv_root=Path("."),
                logger=logging.getLogger("tfr-band-override"),
                tfr_computed=tfr,
                power_bands=override_bands,
            )

        self.assertEqual(seen["bands"], override_bands)

    def test_itpc_tfr_skips_short_segments_by_duration(self):
        n_epochs, n_ch, n_freqs, n_times = 4, 2, 3, 20  # 0.2 s at 100 Hz
        sfreq = 100.0
        times = np.arange(n_times, dtype=float) / sfreq
        freqs = np.array([8.0, 10.0, 12.0], dtype=float)
        tfr = _ComplexTFRStub(
            data=np.ones((n_epochs, n_ch, n_freqs, n_times), dtype=np.complex128),
            times=times,
            freqs=freqs,
            sfreq=sfreq,
        )
        mask = np.ones((n_times,), dtype=bool)
        windows = TimeWindows(
            masks={"active": mask},
            ranges={"active": (float(times[0]), float(times[-1]))},
            times=times,
            name="active",
        )
        cfg = DotConfig(
            {
                "feature_engineering": {
                    "itpc": {
                        "method": "global",
                        "min_segment_sec": 1.0,
                        "min_cycles_at_fmin": 3.0,
                    }
                }
            }
        )
        ctx = SimpleNamespace(
            config=cfg,
            epochs=SimpleNamespace(info={"sfreq": sfreq}, times=times),
            logger=logging.getLogger("itpc-short-segment"),
            tfr_complex=tfr,
            train_mask=None,
            analysis_mode="group_stats",
            frequency_bands={"alpha": [8.0, 12.0]},
            spatial_modes=["channels"],
            windows=windows,
            name="active",
            aligned_events=None,
        )

        df, cols = extract_phase_features(ctx, bands=["alpha"])
        self.assertTrue(df.empty)
        self.assertEqual(cols, [])

    def test_itpc_precomputed_skips_short_segments_by_cycles(self):
        n_epochs, n_ch, n_times = 4, 2, 20  # 0.2 s at 100 Hz
        sfreq = 100.0
        times = np.arange(n_times, dtype=float) / sfreq
        mask = np.ones((n_times,), dtype=bool)
        windows = TimeWindows(
            masks={"active": mask},
            ranges={"active": (float(times[0]), float(times[-1]))},
            times=times,
            name="active",
        )
        zeros = np.zeros((n_epochs, n_ch, n_times), dtype=float)
        phase = np.zeros((n_epochs, n_ch, n_times), dtype=float)
        analytic = np.exp(1j * phase)
        band = BandData(
            band="alpha",
            fmin=8.0,
            fmax=12.0,
            filtered=zeros.copy(),
            analytic=analytic,
            envelope=np.ones_like(zeros),
            phase=phase,
            power=np.ones_like(zeros),
        )
        cfg = DotConfig(
            {
                "feature_engineering": {
                    "analysis_mode": "group_stats",
                    "itpc": {
                        "method": "global",
                        "min_segment_sec": 0.0,
                        "min_cycles_at_fmin": 5.0,
                    },
                }
            }
        )
        precomputed = PrecomputedData(
            data=zeros.copy(),
            times=times,
            sfreq=sfreq,
            ch_names=["C1", "C2"],
            picks=np.arange(n_ch),
            windows=windows,
            band_data={"alpha": band},
            config=cfg,
            logger=logging.getLogger("itpc-precomputed-short"),
            frequency_bands={"alpha": [8.0, 12.0]},
        )

        df, cols = extract_itpc_from_precomputed(precomputed)
        self.assertTrue(df.empty)
        self.assertEqual(cols, [])

    def test_pac_precomputed_skips_short_segments(self):
        n_epochs, n_ch, n_times = 4, 2, 20  # 0.2 s at 100 Hz
        sfreq = 100.0
        times = np.arange(n_times, dtype=float) / sfreq
        mask = np.ones((n_times,), dtype=bool)
        windows = TimeWindows(
            masks={"active": mask},
            ranges={"active": (float(times[0]), float(times[-1]))},
            times=times,
            name="active",
        )

        phase = np.zeros((n_epochs, n_ch, n_times), dtype=float)
        analytic = np.exp(1j * phase)
        power = np.ones((n_epochs, n_ch, n_times), dtype=float)
        filtered = np.zeros((n_epochs, n_ch, n_times), dtype=float)

        theta = BandData(
            band="theta",
            fmin=4.0,
            fmax=8.0,
            filtered=filtered.copy(),
            analytic=analytic.copy(),
            envelope=np.sqrt(power),
            phase=phase.copy(),
            power=power.copy(),
        )
        gamma = BandData(
            band="gamma",
            fmin=30.0,
            fmax=80.0,
            filtered=filtered.copy(),
            analytic=analytic.copy(),
            envelope=np.sqrt(power),
            phase=phase.copy(),
            power=power.copy(),
        )

        cfg = DotConfig(
            {
                "feature_engineering": {
                    "analysis_mode": "group_stats",
                    "pac": {
                        "method": "mvl",
                        "pairs": [["theta", "gamma"]],
                        "n_surrogates": 0,
                        "min_segment_sec": 1.0,
                        "min_cycles_at_fmin": 3.0,
                    },
                },
                "time_frequency_analysis": {
                    "bands": {
                        "theta": [4.0, 8.0],
                        "gamma": [30.0, 80.0],
                    }
                },
            }
        )
        precomputed = PrecomputedData(
            data=np.zeros((n_epochs, n_ch, n_times), dtype=float),
            times=times,
            sfreq=sfreq,
            ch_names=["C1", "C2"],
            picks=np.arange(n_ch),
            windows=windows,
            band_data={"theta": theta, "gamma": gamma},
            config=cfg,
            logger=logging.getLogger("pac-precomputed-short"),
            spatial_modes=["channels"],
        )

        df, cols = extract_pac_from_precomputed(precomputed, cfg)
        self.assertTrue(df.empty)
        self.assertEqual(cols, [])

    def test_pac_precomputed_uses_precomputed_frequency_bands_when_config_bands_missing(self):
        n_epochs, n_ch, n_times = 4, 2, 300
        sfreq = 100.0
        times = np.arange(n_times, dtype=float) / sfreq
        mask = np.ones((n_times,), dtype=bool)
        windows = TimeWindows(
            masks={"active": mask},
            ranges={"active": (float(times[0]), float(times[-1]))},
            times=times,
            name="active",
        )

        rng = np.random.default_rng(17)
        phase = rng.uniform(-np.pi, np.pi, size=(n_epochs, n_ch, n_times))
        analytic = np.exp(1j * phase)
        power = 1.0 + rng.random((n_epochs, n_ch, n_times))
        filtered = np.real(analytic)

        theta = BandData(
            band="theta",
            fmin=4.0,
            fmax=8.0,
            filtered=filtered.copy(),
            analytic=analytic.copy(),
            envelope=np.sqrt(power),
            phase=phase.copy(),
            power=power.copy(),
        )
        gamma = BandData(
            band="gamma",
            fmin=30.0,
            fmax=80.0,
            filtered=filtered.copy(),
            analytic=analytic.copy(),
            envelope=np.sqrt(power),
            phase=phase.copy(),
            power=power.copy(),
        )

        cfg = DotConfig(
            {
                "feature_engineering": {
                    "analysis_mode": "group_stats",
                    "pac": {
                        "method": "mvl",
                        "pairs": [["theta", "gamma"]],
                        "n_surrogates": 0,
                        "min_segment_sec": 1.0,
                        "min_cycles_at_fmin": 3.0,
                        "allow_harmonic_overlap": True,
                    },
                    "spatial_modes": ["global"],
                }
            }
        )
        precomputed = PrecomputedData(
            data=np.zeros((n_epochs, n_ch, n_times), dtype=float),
            times=times,
            sfreq=sfreq,
            ch_names=["C1", "C2"],
            picks=np.arange(n_ch),
            windows=windows,
            band_data={"theta": theta, "gamma": gamma},
            config=cfg,
            logger=logging.getLogger("pac-precomputed-bands"),
            spatial_modes=["global"],
            frequency_bands={"theta": [4.0, 8.0], "gamma": [30.0, 80.0]},
        )

        df, cols = extract_pac_from_precomputed(precomputed, cfg)

        self.assertFalse(df.empty)
        self.assertTrue(cols)
        self.assertIn("pac_active_theta_gamma_global_val", df.columns)

    def test_pac_resting_state_rejects_empty_target_window(self):
        n_epochs, n_ch, n_times = 4, 2, 300
        sfreq = 100.0
        times = np.arange(n_times, dtype=float) / sfreq
        analysis_mask = np.ones((n_times,), dtype=bool)
        empty_mask = np.zeros((n_times,), dtype=bool)
        windows = TimeWindows(
            masks={"analysis": analysis_mask, "active": empty_mask},
            ranges={"analysis": (float(times[0]), float(times[-1])), "active": (float(times[0]), float(times[-1]))},
            times=times,
            name="active",
        )

        rng = np.random.default_rng(23)
        phase = rng.uniform(-np.pi, np.pi, size=(n_epochs, n_ch, n_times))
        analytic = np.exp(1j * phase)
        power = 1.0 + rng.random((n_epochs, n_ch, n_times))
        filtered = np.real(analytic)

        theta = BandData(
            band="theta",
            fmin=4.0,
            fmax=8.0,
            filtered=filtered.copy(),
            analytic=analytic.copy(),
            envelope=np.sqrt(power),
            phase=phase.copy(),
            power=power.copy(),
        )
        gamma = BandData(
            band="gamma",
            fmin=30.0,
            fmax=80.0,
            filtered=filtered.copy(),
            analytic=analytic.copy(),
            envelope=np.sqrt(power),
            phase=phase.copy(),
            power=power.copy(),
        )

        cfg = DotConfig(
            {
                "feature_engineering": {
                    "task_is_rest": True,
                    "analysis_mode": "group_stats",
                    "pac": {
                        "method": "mvl",
                        "pairs": [["theta", "gamma"]],
                        "n_surrogates": 0,
                        "min_segment_sec": 1.0,
                        "min_cycles_at_fmin": 3.0,
                        "allow_harmonic_overlap": True,
                    },
                    "spatial_modes": ["global"],
                }
            }
        )
        precomputed = PrecomputedData(
            data=np.zeros((n_epochs, n_ch, n_times), dtype=float),
            times=times,
            sfreq=sfreq,
            ch_names=["C1", "C2"],
            picks=np.arange(n_ch),
            windows=windows,
            band_data={"theta": theta, "gamma": gamma},
            config=cfg,
            logger=logging.getLogger("pac-precomputed-rest"),
            spatial_modes=["global"],
            frequency_bands={"theta": [4.0, 8.0], "gamma": [30.0, 80.0]},
        )

        with self.assertRaisesRegex(ValueError, "target window 'active' does not contain valid samples"):
            extract_pac_from_precomputed(precomputed, cfg)

    def test_pac_precomputed_without_normalization_is_not_divided_by_segment_length(self):
        n_epochs, n_ch, n_times = 2, 1, 100
        sfreq = 100.0
        times = np.arange(n_times, dtype=float) / sfreq
        mask = np.ones((n_times,), dtype=bool)
        windows = TimeWindows(
            masks={"active": mask},
            ranges={"active": (float(times[0]), float(times[-1]))},
            times=times,
            name="active",
        )

        phase = np.zeros((n_epochs, n_ch, n_times), dtype=float)
        analytic = np.exp(1j * phase)
        power = np.ones((n_epochs, n_ch, n_times), dtype=float)
        filtered = np.ones((n_epochs, n_ch, n_times), dtype=float)

        theta = BandData(
            band="theta",
            fmin=4.0,
            fmax=8.0,
            filtered=filtered.copy(),
            analytic=analytic.copy(),
            envelope=np.ones_like(power),
            phase=phase.copy(),
            power=power.copy(),
        )
        gamma = BandData(
            band="gamma",
            fmin=30.0,
            fmax=80.0,
            filtered=filtered.copy(),
            analytic=analytic.copy(),
            envelope=np.ones_like(power),
            phase=phase.copy(),
            power=power.copy(),
        )

        cfg = DotConfig(
            {
                "feature_engineering": {
                    "analysis_mode": "group_stats",
                    "pac": {
                        "method": "mvl",
                        "pairs": [["theta", "gamma"]],
                        "normalize": False,
                        "n_surrogates": 0,
                        "min_segment_sec": 0.0,
                        "min_cycles_at_fmin": 0.0,
                        "allow_harmonic_overlap": True,
                    },
                    "spatial_modes": ["global"],
                }
            }
        )
        precomputed = PrecomputedData(
            data=np.zeros((n_epochs, n_ch, n_times), dtype=float),
            times=times,
            sfreq=sfreq,
            ch_names=["C1"],
            picks=np.arange(n_ch),
            windows=windows,
            band_data={"theta": theta, "gamma": gamma},
            config=cfg,
            logger=logging.getLogger("pac-precomputed-normalize-false"),
            spatial_modes=["global"],
            frequency_bands={"theta": [4.0, 8.0], "gamma": [30.0, 80.0]},
        )

        df, cols = extract_pac_from_precomputed(precomputed, cfg)

        self.assertIn("pac_active_theta_gamma_global_val", cols)
        np.testing.assert_allclose(
            df["pac_active_theta_gamma_global_val"].to_numpy(dtype=float),
            np.ones((n_epochs,), dtype=float),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
