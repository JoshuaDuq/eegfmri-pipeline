from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from eeg_pipeline.analysis.features.preparation import _determine_frequency_bands
from eeg_pipeline.analysis.features.phase import (
    extract_itpc_from_precomputed,
    extract_pac_from_precomputed,
    extract_phase_features,
)
from eeg_pipeline.analysis.features.source_localization import (
    _compute_eloreta_source_estimates,
    _compute_roi_envelope,
    _compute_roi_power,
    _extract_roi_timecourses,
    _extract_roi_timecourses_from_vertex_indices,
    _load_source_contrast_config,
    _load_source_localization_config,
    extract_source_connectivity_features,
    extract_source_contrast_features,
    extract_source_localization_features,
)
from eeg_pipeline.types import BandData, PrecomputedData, PrecomputedQC, TimeWindows
from eeg_pipeline.utils.analysis.tfr import compute_tfr_for_subject
from tests.pipelines_test_utils import DotConfig


class _EpochStub:
    def __init__(self, n_epochs: int, sfreq: float = 100.0):
        self._n_epochs = int(n_epochs)
        self.info = {"sfreq": float(sfreq)}

    def __len__(self):
        return self._n_epochs

    def copy(self):
        return _EpochStub(self._n_epochs, self.info["sfreq"])

    def filter(self, *_args, **_kwargs):
        return self

    def __getitem__(self, key):
        if isinstance(key, np.ndarray) and key.dtype == bool:
            return _EpochStub(int(np.sum(key)), self.info["sfreq"])
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


class TestScientificValidityGuards(unittest.TestCase):
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
            config=DotConfig({}),
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
            config=DotConfig({}),
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


if __name__ == "__main__":
    unittest.main()
