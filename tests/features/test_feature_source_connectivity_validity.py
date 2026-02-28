from __future__ import annotations

import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from eeg_pipeline.analysis.features.source_localization import (
    FMRIConstraintConfig,
    _compute_fmri_analysis_voxel_mask,
    _fmri_roi_coords_from_stats_map,
    _load_fmri_constraint_config,
    _load_source_localization_config,
    _make_fmri_subsampling_rng,
    extract_source_connectivity_features,
)
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


class _TrackingEpochStub(_EpochStub):
    def __init__(self, n_epochs: int, sfreq: float = 100.0):
        super().__init__(n_epochs=n_epochs, sfreq=sfreq)
        self.filter_calls = 0

    def copy(self):
        return self

    def filter(self, *_args, **_kwargs):
        self.filter_calls += 1
        return self


class _FakeCon:
    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data, dtype=float)

    def get_data(self):
        return self._data


class TestSourceConnectivityValidity(unittest.TestCase):
    @staticmethod
    def _default_fmri_constraint() -> FMRIConstraintConfig:
        return FMRIConstraintConfig(
            enabled=True,
            stats_map_path=Path("/tmp/map.nii.gz"),
            provenance="independent",
            require_provenance=False,
            allow_same_dataset_provenance=False,
            threshold=3.1,
            tail="pos",
            threshold_mode="z",
            fdr_q=0.05,
            stat_type="z",
            cluster_min_voxels=1,
            cluster_min_volume_mm3=None,
            max_clusters=10,
            max_voxels_per_cluster=0,
            max_total_voxels=0,
            random_seed=0,
        )

    def test_make_fmri_rng_seed_zero_is_deterministic(self):
        rng_a = _make_fmri_subsampling_rng(0)
        rng_b = _make_fmri_subsampling_rng(0)
        self.assertEqual(int(rng_a.integers(0, 100000)), int(rng_b.integers(0, 100000)))

    def test_fmri_random_seed_must_be_non_negative(self):
        config = DotConfig(
            {
                "feature_engineering": {
                    "sourcelocalization": {
                        "fmri": {
                            "enabled": True,
                            "stats_map_path": "/tmp/map.nii.gz",
                            "provenance": "independent",
                            "random_seed": -2,
                        }
                    }
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "random_seed"):
            _load_fmri_constraint_config(config)

    def test_fmri_threshold_must_be_positive_finite(self):
        config = DotConfig(
            {
                "feature_engineering": {
                    "sourcelocalization": {
                        "fmri": {
                            "enabled": True,
                            "stats_map_path": "/tmp/map.nii.gz",
                            "provenance": "independent",
                            "threshold": 0,
                        }
                    }
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "threshold"):
            _load_fmri_constraint_config(config)

    def test_fmri_time_windows_are_rejected(self):
        config = DotConfig(
            {
                "feature_engineering": {
                    "sourcelocalization": {
                        "fmri": {
                            "enabled": True,
                            "stats_map_path": "/tmp/map.nii.gz",
                            "provenance": "independent",
                            "time_windows": {
                                "window_a": {"name": "active", "tmin": 1.0, "tmax": 2.0}
                            },
                        }
                    }
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "time_windows"):
            _load_fmri_constraint_config(config)

    def test_fmri_stats_map_must_be_3d(self):
        class _FakeHeader:
            @staticmethod
            def get_zooms():
                return (2.0, 2.0, 2.0)

        class _FakeImg:
            def __init__(self, data: np.ndarray, affine: np.ndarray):
                self._data = np.asarray(data, dtype=np.float32)
                self.affine = np.asarray(affine, dtype=float)
                self.header = _FakeHeader()
                self.shape = self._data.shape

            def get_fdata(self, dtype=None):
                if dtype is None:
                    return self._data
                return self._data.astype(dtype)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_path = root / "map.nii.gz"
            map_path.touch()
            orig_path = root / "sub-0001" / "mri" / "orig.mgz"
            orig_path.parent.mkdir(parents=True, exist_ok=True)
            orig_path.touch()

            img_map = _FakeImg(np.zeros((2, 2, 2, 2), dtype=float), np.eye(4))
            img_ref = _FakeImg(np.zeros((2, 2, 2), dtype=float), np.eye(4))
            fake_nib = types.ModuleType("nibabel")
            fake_nib.load = lambda p: img_map if Path(p) == map_path else img_ref

            cfg = self._default_fmri_constraint()
            with patch.dict(sys.modules, {"nibabel": fake_nib}):
                with self.assertRaisesRegex(ValueError, "single-volume 3D"):
                    _fmri_roi_coords_from_stats_map(
                        map_path,
                        cfg,
                        logger=logging.getLogger("fmri-3d-check"),
                        subjects_dir=root,
                        subject="sub-0001",
                    )

    def test_fmri_stats_map_grid_must_match_subject_mri(self):
        class _FakeHeader:
            @staticmethod
            def get_zooms():
                return (2.0, 2.0, 2.0)

        class _FakeImg:
            def __init__(self, data: np.ndarray, affine: np.ndarray):
                self._data = np.asarray(data, dtype=np.float32)
                self.affine = np.asarray(affine, dtype=float)
                self.header = _FakeHeader()
                self.shape = self._data.shape

            def get_fdata(self, dtype=None):
                if dtype is None:
                    return self._data
                return self._data.astype(dtype)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_path = root / "map.nii.gz"
            map_path.touch()
            orig_path = root / "sub-0001" / "mri" / "orig.mgz"
            orig_path.parent.mkdir(parents=True, exist_ok=True)
            orig_path.touch()

            map_data = np.zeros((2, 2, 2), dtype=float)
            map_data[0, 0, 0] = 5.0
            img_map = _FakeImg(map_data, np.eye(4))
            shifted_affine = np.array(
                [
                    [1.0, 0.0, 0.0, 10.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            img_ref = _FakeImg(np.zeros((2, 2, 2), dtype=float), shifted_affine)
            fake_nib = types.ModuleType("nibabel")
            fake_nib.load = lambda p: img_map if Path(p) == map_path else img_ref

            cfg = self._default_fmri_constraint()
            with patch.dict(sys.modules, {"nibabel": fake_nib}):
                with self.assertRaisesRegex(ValueError, "same voxel grid"):
                    _fmri_roi_coords_from_stats_map(
                        map_path,
                        cfg,
                        logger=logging.getLogger("fmri-grid-check"),
                        subjects_dir=root,
                        subject="sub-0001",
                    )

    def test_fmri_analysis_mask_includes_zero_stat_voxels_inside_subject_support(self):
        stats_data = np.array(
            [
                [[0.0, 2.0], [0.0, np.nan]],
                [[1.0, -3.0], [0.0, 0.0]],
            ],
            dtype=float,
        )
        subject_ref_data = np.array(
            [
                [[1.0, 1.0], [1.0, 0.0]],
                [[1.0, 1.0], [0.0, 0.0]],
            ],
            dtype=float,
        )
        mask = _compute_fmri_analysis_voxel_mask(stats_data, subject_ref_data)
        expected = np.array(
            [
                [[True, True], [True, False]],
                [[True, True], [False, False]],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(mask, expected)

    def test_fmri_fdr_requires_explicit_z_map_evidence(self):
        class _FakeHeader:
            @staticmethod
            def get_zooms():
                return (2.0, 2.0, 2.0)

            @staticmethod
            def get_intent():
                return (0, (), "")

        class _FakeImg:
            def __init__(self, data: np.ndarray, affine: np.ndarray):
                self._data = np.asarray(data, dtype=np.float32)
                self.affine = np.asarray(affine, dtype=float)
                self.header = _FakeHeader()
                self.shape = self._data.shape

            def get_fdata(self, dtype=None):
                if dtype is None:
                    return self._data
                return self._data.astype(dtype)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_path = root / "map.nii.gz"
            map_path.touch()
            orig_path = root / "sub-0001" / "mri" / "orig.mgz"
            orig_path.parent.mkdir(parents=True, exist_ok=True)
            orig_path.touch()

            map_data = np.zeros((2, 2, 2), dtype=float)
            map_data[0, 0, 0] = 5.0
            img_map = _FakeImg(map_data, np.eye(4))
            img_ref = _FakeImg(np.ones((2, 2, 2), dtype=float), np.eye(4))

            fake_nib = types.ModuleType("nibabel")
            fake_nib.load = lambda p: img_map if Path(p) == map_path else img_ref
            fake_nib.nifti1 = SimpleNamespace(
                intent_codes=SimpleNamespace(code={"z score": 5})
            )

            base = self._default_fmri_constraint()
            cfg = FMRIConstraintConfig(
                enabled=base.enabled,
                stats_map_path=base.stats_map_path,
                provenance=base.provenance,
                require_provenance=base.require_provenance,
                allow_same_dataset_provenance=base.allow_same_dataset_provenance,
                threshold=base.threshold,
                tail=base.tail,
                threshold_mode="fdr",
                fdr_q=base.fdr_q,
                stat_type=base.stat_type,
                cluster_min_voxels=base.cluster_min_voxels,
                cluster_min_volume_mm3=base.cluster_min_volume_mm3,
                max_clusters=base.max_clusters,
                max_voxels_per_cluster=base.max_voxels_per_cluster,
                max_total_voxels=base.max_total_voxels,
                random_seed=base.random_seed,
            )

            with patch.dict(sys.modules, {"nibabel": fake_nib}):
                with self.assertRaisesRegex(ValueError, "z-statistics map"):
                    _fmri_roi_coords_from_stats_map(
                        map_path,
                        cfg,
                        logger=logging.getLogger("fmri-fdr-z-check"),
                        subjects_dir=root,
                        subject="sub-0001",
                    )

    def test_template_fallback_is_opt_in(self):
        ctx = SimpleNamespace(subject="0001")
        config = DotConfig(
            {
                "feature_engineering": {
                    "sourcelocalization": {
                        "mode": "eeg_only",
                        "fmri": {"enabled": False, "provenance": "independent"},
                    }
                }
            }
        )
        src_cfg = _load_source_localization_config(ctx, config, method="lcmv")
        self.assertFalse(bool(src_cfg.allow_template_fallback))

    def test_wpli_uses_train_mask_and_marks_broadcast_attrs(self):
        n_epochs = 4
        train_mask = np.array([True, True, False, False], dtype=bool)
        roi_data = np.random.default_rng(3).standard_normal((n_epochs, 2, 80))

        captured = {"n_epochs_fit": None, "indices": None}

        def _fake_spectral_connectivity_epochs(data, **_kwargs):
            captured["n_epochs_fit"] = int(np.asarray(data).shape[0])
            captured["indices"] = _kwargs.get("indices")
            return _FakeCon(np.array([[0.5]], dtype=float))

        fake_conn_mod = types.ModuleType("mne_connectivity")
        fake_conn_mod.spectral_connectivity_epochs = _fake_spectral_connectivity_epochs
        fake_conn_mod.envelope_correlation = lambda *_args, **_kwargs: None

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
        )

        ctx = SimpleNamespace(
            epochs=_EpochStub(n_epochs),
            config=DotConfig({}),
            logger=logging.getLogger("source-connectivity-validity"),
            analysis_mode="trial_ml_safe",
            train_mask=train_mask,
            name="active",
            frequency_bands={"alpha": (8.0, 12.0)},
        )

        with (
            patch.dict(sys.modules, {"mne_connectivity": fake_conn_mod}),
            patch(
                "eeg_pipeline.analysis.features.source_localization._load_source_localization_config",
                return_value=src_cfg,
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._setup_forward_model",
                return_value=("fwd", "src", None),
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._compute_lcmv_source_estimates",
                return_value=(["stc"] * n_epochs, None),
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._extract_roi_timecourses",
                return_value=roi_data,
            ),
            patch(
                "mne.read_labels_from_annot",
                return_value=[SimpleNamespace(name="roi1"), SimpleNamespace(name="roi2")],
            ),
        ):
            df, cols = extract_source_connectivity_features(
                ctx,
                bands=["alpha"],
                method="lcmv",
                connectivity_method="wpli",
            )

        self.assertTrue(cols)
        self.assertEqual(len(df), n_epochs)
        self.assertEqual(captured["n_epochs_fit"], int(np.sum(train_mask)))
        self.assertIsNotNone(captured["indices"])
        row_idx, col_idx = captured["indices"]
        np.testing.assert_array_equal(row_idx, np.array([0], dtype=int))
        np.testing.assert_array_equal(col_idx, np.array([1], dtype=int))
        self.assertEqual(df.attrs.get("feature_granularity"), "subject")
        self.assertIn("broadcast_warning", df.attrs)
        self.assertTrue(bool(df.attrs.get("threshold_train_mask_used")))
        self.assertAlmostEqual(float(df["src_lcmv_alpha_wpli_global"].iloc[0]), 0.5, places=7)

    def test_wpli_does_not_prefilter_before_multitaper_connectivity(self):
        n_epochs = 4
        roi_data = np.random.default_rng(101).standard_normal((n_epochs, 2, 80))
        epochs = _TrackingEpochStub(n_epochs=n_epochs, sfreq=100.0)

        fake_conn_mod = types.ModuleType("mne_connectivity")
        fake_conn_mod.spectral_connectivity_epochs = lambda *_args, **_kwargs: _FakeCon(
            np.array([[0.5]], dtype=float)
        )
        fake_conn_mod.envelope_correlation = lambda *_args, **_kwargs: None

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
        )

        ctx = SimpleNamespace(
            epochs=epochs,
            config=DotConfig({}),
            logger=logging.getLogger("source-connectivity-no-prefilter"),
            analysis_mode="group_stats",
            train_mask=None,
            name="active",
            frequency_bands={"alpha": (8.0, 12.0)},
        )

        with (
            patch.dict(sys.modules, {"mne_connectivity": fake_conn_mod}),
            patch(
                "eeg_pipeline.analysis.features.source_localization._load_source_localization_config",
                return_value=src_cfg,
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._setup_forward_model",
                return_value=("fwd", "src", None),
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._compute_lcmv_source_estimates",
                return_value=(["stc"] * n_epochs, None),
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._extract_roi_timecourses",
                return_value=roi_data,
            ),
            patch(
                "mne.read_labels_from_annot",
                return_value=[SimpleNamespace(name="roi1"), SimpleNamespace(name="roi2")],
            ),
        ):
            _df, _cols = extract_source_connectivity_features(
                ctx,
                bands=["alpha"],
                method="lcmv",
                connectivity_method="wpli",
            )

        self.assertEqual(epochs.filter_calls, 0)

    def test_aec_prefilters_before_envelope_connectivity(self):
        n_epochs = 4
        roi_data = np.random.default_rng(202).standard_normal((n_epochs, 2, 80))
        epochs = _TrackingEpochStub(n_epochs=n_epochs, sfreq=100.0)

        class _FakeEnvelopeCon:
            def combine(self):
                return self

            def get_data(self, output="dense"):
                _ = output
                return np.array(
                    [
                        [[0.0], [0.2]],
                        [[0.2], [0.0]],
                    ],
                    dtype=float,
                )

        fake_conn_mod = types.ModuleType("mne_connectivity")
        fake_conn_mod.spectral_connectivity_epochs = lambda *_args, **_kwargs: None
        fake_conn_mod.envelope_correlation = lambda *_args, **_kwargs: _FakeEnvelopeCon()

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
        )

        ctx = SimpleNamespace(
            epochs=epochs,
            config=DotConfig({}),
            logger=logging.getLogger("source-connectivity-aec-prefilter"),
            analysis_mode="group_stats",
            train_mask=None,
            name="active",
            frequency_bands={"alpha": (8.0, 12.0)},
        )

        with (
            patch.dict(sys.modules, {"mne_connectivity": fake_conn_mod}),
            patch(
                "eeg_pipeline.analysis.features.source_localization._load_source_localization_config",
                return_value=src_cfg,
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._setup_forward_model",
                return_value=("fwd", "src", None),
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._compute_lcmv_source_estimates",
                return_value=(["stc"] * n_epochs, None),
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._extract_roi_timecourses",
                return_value=roi_data,
            ),
            patch(
                "mne.read_labels_from_annot",
                return_value=[SimpleNamespace(name="roi1"), SimpleNamespace(name="roi2")],
            ),
        ):
            _df, _cols = extract_source_connectivity_features(
                ctx,
                bands=["alpha"],
                method="lcmv",
                connectivity_method="aec",
            )

        self.assertGreater(epochs.filter_calls, 0)

    def test_fmri_informed_mode_requires_enabled_fmri_constraint(self):
        ctx = SimpleNamespace(subject="0001")
        config = DotConfig(
            {
                "feature_engineering": {
                    "sourcelocalization": {
                        "mode": "fmri_informed",
                        "fmri": {"enabled": False, "provenance": "independent"},
                    }
                }
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "fmri_informed",
        ):
            _load_source_localization_config(ctx, config, method="lcmv")

    def test_source_localization_config_rejects_invalid_method(self):
        ctx = SimpleNamespace(subject="0001")
        config = DotConfig(
            {
                "feature_engineering": {
                    "sourcelocalization": {
                        "method": "beamformerx",
                    }
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "sourcelocalization.method"):
            _load_source_localization_config(ctx, config, method="lcmv")

    def test_source_localization_config_rejects_invalid_spacing(self):
        ctx = SimpleNamespace(subject="0001")
        config = DotConfig(
            {
                "feature_engineering": {
                    "sourcelocalization": {
                        "spacing": "oct7",
                    }
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "sourcelocalization.spacing"):
            _load_source_localization_config(ctx, config, method="lcmv")

    def test_source_localization_config_rejects_invalid_parcellation(self):
        ctx = SimpleNamespace(subject="0001")
        config = DotConfig(
            {
                "feature_engineering": {
                    "sourcelocalization": {
                        "parcellation": "desikan_custom",
                    }
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "sourcelocalization.parcellation"):
            _load_source_localization_config(ctx, config, method="lcmv")

    def test_source_localization_config_rejects_invalid_lcmv_reg(self):
        ctx = SimpleNamespace(subject="0001")
        config = DotConfig(
            {
                "feature_engineering": {
                    "sourcelocalization": {
                        "reg": -0.1,
                    }
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "sourcelocalization.reg"):
            _load_source_localization_config(ctx, config, method="lcmv")

    def test_fmri_constrained_eloreta_requires_loose_one(self):
        ctx = SimpleNamespace(subject="0001")
        config = DotConfig(
            {
                "feature_engineering": {
                    "sourcelocalization": {
                        "method": "eloreta",
                        "loose": 0.2,
                        "fmri": {
                            "enabled": True,
                            "provenance": "independent",
                        },
                    }
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "loose=1.0"):
            _load_source_localization_config(ctx, config, method="eloreta")

    def test_wpli_skips_too_short_segments_by_cycle_guard(self):
        n_epochs = 4
        roi_data = np.random.default_rng(17).standard_normal((n_epochs, 2, 20))  # 0.2 s at 100 Hz

        spectral_connectivity_mock = Mock(return_value=_FakeCon(np.array([[0.5]], dtype=float)))
        fake_conn_mod = types.ModuleType("mne_connectivity")
        fake_conn_mod.spectral_connectivity_epochs = spectral_connectivity_mock
        fake_conn_mod.envelope_correlation = lambda *_args, **_kwargs: None

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
        )

        ctx = SimpleNamespace(
            epochs=_EpochStub(n_epochs, sfreq=100.0),
            config=DotConfig(
                {
                    "feature_engineering": {
                        "connectivity": {
                            "min_cycles_per_band": 3.0,
                        }
                    }
                }
            ),
            logger=logging.getLogger("source-connectivity-cycle-guard"),
            analysis_mode="group_stats",
            train_mask=None,
            name="active",
            frequency_bands={"alpha": (8.0, 12.0)},
        )

        with (
            patch.dict(sys.modules, {"mne_connectivity": fake_conn_mod}),
            patch(
                "eeg_pipeline.analysis.features.source_localization._load_source_localization_config",
                return_value=src_cfg,
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._setup_forward_model",
                return_value=("fwd", "src", None),
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._compute_lcmv_source_estimates",
                return_value=(["stc"] * n_epochs, None),
            ),
            patch(
                "eeg_pipeline.analysis.features.source_localization._extract_roi_timecourses",
                return_value=roi_data,
            ),
            patch(
                "mne.read_labels_from_annot",
                return_value=[SimpleNamespace(name="roi1"), SimpleNamespace(name="roi2")],
            ),
        ):
            df, cols = extract_source_connectivity_features(
                ctx,
                bands=["alpha"],
                method="lcmv",
                connectivity_method="wpli",
            )

        spectral_connectivity_mock.assert_not_called()
        self.assertEqual(cols, [])
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
