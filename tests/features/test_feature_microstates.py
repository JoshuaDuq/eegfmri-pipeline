import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mne
import numpy as np

from tests.pipelines_test_utils import DotConfig


class _WindowStub:
    def __init__(self, mask):
        self.masks = {"active": mask}
        self.name = None

    def get_mask(self, name):
        return self.masks.get(name)


class _NamedWindowStub:
    def __init__(self, name, masks):
        self.name = name
        self.masks = dict(masks)

    def get_mask(self, name):
        return self.masks.get(name)


class TestMicrostateFeatures(unittest.TestCase):
    def _build_epochs(self):
        rng = np.random.default_rng(7)
        ch_names = ["Fp1", "Fp2", "C3", "C4", "P3", "P4"]
        sfreq = 100.0
        n_times = 80

        templates = np.array(
            [
                [1.0, -1.0, 0.5, -0.5, 0.0, 0.0],    # A
                [0.0, 0.0, 1.0, -1.0, 0.6, -0.6],    # B
                [1.0, 1.0, 0.0, 0.0, -1.0, -1.0],    # C
                [1.0, 0.8, 0.6, 0.6, 0.2, 0.2],      # D
            ],
            dtype=float,
        )

        seq1 = np.repeat(np.array([0, 1, 2, 3]), n_times // 4)
        seq2 = np.repeat(np.array([3, 2, 1, 0]), n_times // 4)

        data = np.zeros((2, len(ch_names), n_times), dtype=float)
        for ti, seq in enumerate([seq1, seq2]):
            for t, state in enumerate(seq):
                data[ti, :, t] = templates[state] + 0.02 * rng.standard_normal(len(ch_names))

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        epochs = mne.EpochsArray(data, info=info, tmin=0.0, verbose=False)
        epochs.set_montage("standard_1020")
        epochs.set_eeg_reference("average", verbose=False)
        return epochs, templates, ch_names

    def _build_epochs_unreferenced(self):
        epochs, templates, ch_names = self._build_epochs()
        unreferenced = mne.EpochsArray(
            epochs.get_data().copy(),
            info=mne.create_info(ch_names=ch_names, sfreq=epochs.info["sfreq"], ch_types="eeg"),
            tmin=epochs.tmin,
            verbose=False,
        )
        unreferenced.set_montage("standard_1020")
        return unreferenced, templates, ch_names

    def test_extract_microstate_features_with_fixed_templates(self):
        from eeg_pipeline.analysis.features.microstates import extract_microstate_features

        epochs, templates, ch_names = self._build_epochs()
        mask = np.ones(epochs.get_data().shape[-1], dtype=bool)
        ctx = SimpleNamespace(
            epochs=epochs,
            windows=_WindowStub(mask),
            name="active",
            config=DotConfig(
                {
                    "feature_engineering": {
                        "microstates": {
                            "n_states": 4,
                            "min_duration_ms": 0.0,
                            "min_peak_distance_ms": 5.0,
                            "max_gfp_peaks_per_epoch": 200,
                        }
                    }
                }
            ),
            logger=logging.getLogger("microstate-test"),
            fixed_templates=templates,
            fixed_template_ch_names=ch_names,
            fixed_template_labels=["a", "b", "c", "d"],
        )

        df, cols = extract_microstate_features(ctx)

        self.assertEqual(len(df), 2)
        self.assertEqual(len(cols), len(df.columns))
        self.assertIn("microstates_active_broadband_global_coverage_a", df.columns)
        self.assertIn("microstates_active_broadband_global_duration_ms_a", df.columns)
        self.assertIn("microstates_active_broadband_global_occurrence_hz_a", df.columns)
        self.assertIn("microstates_active_broadband_global_trans_a_to_b_prob", df.columns)

        cov_cols = [
            "microstates_active_broadband_global_coverage_a",
            "microstates_active_broadband_global_coverage_b",
            "microstates_active_broadband_global_coverage_c",
            "microstates_active_broadband_global_coverage_d",
        ]
        coverage_sum = df[cov_cols].sum(axis=1).to_numpy()
        self.assertTrue(np.allclose(coverage_sum, 1.0, atol=0.1))
        self.assertAlmostEqual(
            float(df["microstates_active_broadband_global_trans_a_to_a_prob"].iloc[0]),
            0.0,
            places=7,
        )
        self.assertAlmostEqual(
            float(df["microstates_active_broadband_global_trans_a_to_b_prob"].iloc[0]),
            1.0,
            places=7,
        )
        self.assertTrue(bool(df.attrs.get("microstate_labels_canonical")))

    def test_microstates_category_registered(self):
        from eeg_pipeline.pipelines import constants

        self.assertIn("microstates", constants.FEATURE_CATEGORIES)

    def test_default_min_duration_is_20ms(self):
        from eeg_pipeline.analysis.features.microstates import _load_microstate_config

        cfg = _load_microstate_config(
            DotConfig({"feature_engineering": {"microstates": {"n_states": 4}}})
        )
        self.assertEqual(cfg.min_duration_ms, 20.0)

    def test_microstates_require_average_reference(self):
        from eeg_pipeline.analysis.features.microstates import extract_microstate_features

        epochs_unref, _, _ = self._build_epochs_unreferenced()
        mask = np.ones(epochs_unref.get_data().shape[-1], dtype=bool)
        ctx = SimpleNamespace(
            epochs=epochs_unref,
            windows=_WindowStub(mask),
            name="active",
            config=DotConfig({"feature_engineering": {"microstates": {"n_states": 2}}}),
            logger=logging.getLogger("microstate-ref-guard"),
            fixed_templates=np.array([[1.0] * 6, [0.0] * 6]),
            fixed_template_ch_names=["Fp1", "Fp2", "C3", "C4", "P3", "P4"],
            fixed_template_labels=["a", "b"],
        )

        with self.assertRaisesRegex(ValueError, "average-referenced EEG"):
            extract_microstate_features(ctx)

    def test_microstates_reject_non_average_custom_reference(self):
        from eeg_pipeline.analysis.features.microstates import extract_microstate_features

        altered, templates, ch_names = self._build_epochs_unreferenced()
        altered.set_eeg_reference(ref_channels=["Fp1"], verbose=False)
        mask = np.ones(altered.get_data().shape[-1], dtype=bool)
        ctx = SimpleNamespace(
            epochs=altered,
            windows=_WindowStub(mask),
            name="active",
            config=DotConfig({"feature_engineering": {"microstates": {"n_states": 4}}}),
            logger=logging.getLogger("microstate-non-average-custom-ref"),
            fixed_templates=templates,
            fixed_template_ch_names=ch_names,
            fixed_template_labels=["a", "b", "c", "d"],
        )

        with self.assertRaisesRegex(ValueError, "average-referenced EEG"):
            extract_microstate_features(ctx)

    def test_microstates_allow_active_average_reference_projection(self):
        from eeg_pipeline.analysis.features.microstates import extract_microstate_features

        epochs_proj, templates, ch_names = self._build_epochs_unreferenced()
        epochs_proj, _ = mne.set_eeg_reference(
            epochs_proj,
            ref_channels="average",
            projection=True,
            verbose=False,
        )
        epochs_proj.apply_proj(verbose=False)
        mask = np.ones(epochs_proj.get_data().shape[-1], dtype=bool)
        ctx = SimpleNamespace(
            epochs=epochs_proj,
            windows=_WindowStub(mask),
            name="active",
            config=DotConfig({"feature_engineering": {"microstates": {"n_states": 4}}}),
            logger=logging.getLogger("microstate-ref-proj"),
            fixed_templates=templates,
            fixed_template_ch_names=ch_names,
            fixed_template_labels=["a", "b", "c", "d"],
        )

        df, _ = extract_microstate_features(ctx)
        self.assertFalse(df.empty)

    def test_transition_probabilities_use_run_to_run_state_changes(self):
        from eeg_pipeline.analysis.features.microstates import _compute_epoch_metrics

        metrics = _compute_epoch_metrics(
            states=np.array([0, 0, 0, 0], dtype=int),
            sfreq=100.0,
            n_states=2,
        )
        self.assertTrue(np.isnan(metrics["transitions"]).all())

        metrics_seq = _compute_epoch_metrics(
            states=np.array([0, 0, 1, 1, 0, 0], dtype=int),
            sfreq=100.0,
            n_states=2,
        )
        self.assertTrue(np.allclose(metrics_seq["transitions"][0], np.array([0.0, 1.0])))
        self.assertTrue(np.allclose(metrics_seq["transitions"][1], np.array([1.0, 0.0])))

    def test_min_duration_prefers_longer_neighbor(self):
        from eeg_pipeline.analysis.features.microstates import _apply_min_duration

        states = np.array([1, 1, 0, 2, 2, 2], dtype=int)
        out = _apply_min_duration(states, min_samples=2)
        np.testing.assert_array_equal(out, np.array([1, 1, 2, 2, 2, 2], dtype=int))

    def test_backfit_peak_states_to_samplewise_segments(self):
        from eeg_pipeline.analysis.features.microstates import _backfit_peak_states_to_samples

        out = _backfit_peak_states_to_samples(
            n_samples=10,
            peak_indices=np.array([2, 7], dtype=int),
            peak_states=np.array([0, 1], dtype=int),
        )
        np.testing.assert_array_equal(
            out,
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int),
        )

    def test_template_fitting_uses_train_mask_only(self):
        from eeg_pipeline.analysis.features import microstates as mod

        epochs, templates, ch_names = self._build_epochs()
        epoch_data = epochs.get_data()
        extra = epoch_data[:1].copy()
        extra[:, :, :] = templates[1][:, None]
        data3 = np.concatenate([epoch_data, extra], axis=0)
        epochs3 = mne.EpochsArray(data3, info=epochs.info, tmin=epochs.tmin, verbose=False)
        epochs3.set_montage("standard_1020")

        mask = np.ones(epochs3.get_data().shape[-1], dtype=bool)
        train_mask = np.array([True, False, False], dtype=bool)
        captured: dict[str, np.ndarray] = {}

        def _fake_peak_maps(segment_epoch, _sfreq, _cfg):
            return segment_epoch[:, :1].T

        def _fake_fit(peak_maps, n_states, random_state):
            _ = random_state
            captured["peak_maps"] = peak_maps.copy()
            return np.tile(peak_maps[:1], (n_states, 1))

        ctx = SimpleNamespace(
            epochs=epochs3,
            windows=_WindowStub(mask),
            name="active",
            train_mask=train_mask,
            config=DotConfig(
                {
                    "feature_engineering": {
                        "microstates": {
                            "n_states": 2,
                            "min_duration_ms": 0.0,
                            "min_peak_distance_ms": 5.0,
                            "max_gfp_peaks_per_epoch": 200,
                        }
                    }
                }
            ),
            logger=logging.getLogger("microstate-train-mask"),
            fixed_templates=None,
            fixed_template_ch_names=ch_names,
        )

        with patch.object(mod, "_extract_peak_topographies", side_effect=_fake_peak_maps), patch.object(
            mod, "_fit_templates_kmeans", side_effect=_fake_fit
        ):
            mod.extract_microstate_features(ctx)

        self.assertIn("peak_maps", captured)
        self.assertEqual(captured["peak_maps"].shape[0], int(np.sum(train_mask)))

    def test_trial_ml_safe_requires_train_mask_or_fixed_templates(self):
        from eeg_pipeline.analysis.features.microstates import extract_microstate_features

        epochs, _, _ = self._build_epochs()
        mask = np.ones(epochs.get_data().shape[-1], dtype=bool)
        ctx = SimpleNamespace(
            epochs=epochs,
            windows=_WindowStub(mask),
            name="active",
            analysis_mode="trial_ml_safe",
            train_mask=None,
            config=DotConfig(
                {
                    "feature_engineering": {
                        "microstates": {
                            "n_states": 2,
                            "min_duration_ms": 0.0,
                            "min_peak_distance_ms": 5.0,
                            "max_gfp_peaks_per_epoch": 200,
                        }
                    }
                }
            ),
            logger=logging.getLogger("microstate-trial-safe"),
            fixed_templates=None,
            fixed_template_ch_names=None,
        )

        with self.assertRaisesRegex(ValueError, "fixed_templates|train_mask"):
            extract_microstate_features(ctx)

    def test_trial_ml_safe_invalid_fixed_templates_raise(self):
        from eeg_pipeline.analysis.features.microstates import extract_microstate_features

        epochs, templates, _ = self._build_epochs()
        mask = np.ones(epochs.get_data().shape[-1], dtype=bool)
        ctx = SimpleNamespace(
            epochs=epochs,
            windows=_WindowStub(mask),
            name="active",
            analysis_mode="trial_ml_safe",
            train_mask=None,
            config=DotConfig(
                {
                    "feature_engineering": {
                        "microstates": {
                            "n_states": 4,
                            "min_duration_ms": 0.0,
                            "min_peak_distance_ms": 5.0,
                            "max_gfp_peaks_per_epoch": 200,
                        }
                    }
                }
            ),
            logger=logging.getLogger("microstate-trial-safe-invalid-fixed"),
            fixed_templates=templates,
            fixed_template_ch_names=["Fp1", "Fp2"],
            fixed_template_labels=["a", "b", "c", "d"],
        )

        with self.assertRaisesRegex(ValueError, "fixed_templates channel"):
            extract_microstate_features(ctx)

    def test_explicit_windows_are_used_for_template_pooling(self):
        from eeg_pipeline.analysis.features import microstates as mod

        ch_names = ["Fp1", "Fp2", "C3"]
        sfreq = 100.0
        n_times = 10
        data = np.zeros((1, len(ch_names), n_times), dtype=float)
        data[0, :, :5] = np.array([[1.0], [0.0], [0.0]])
        data[0, :, 5:] = np.array([[0.0], [1.0], [0.0]])
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        epochs = mne.EpochsArray(data, info=info, tmin=0.0, verbose=False)
        epochs.set_montage("standard_1020")
        epochs.set_eeg_reference("average", verbose=False)

        explicit_windows = [
            {"name": "early", "tmin": 0.0, "tmax": 0.05},
            {"name": "late", "tmin": 0.05, "tmax": 0.10},
        ]
        early_mask = np.zeros(n_times, dtype=bool)
        early_mask[:5] = True
        late_mask = np.zeros(n_times, dtype=bool)
        late_mask[5:] = True

        calls: list[np.ndarray] = []

        def _fake_peak_maps(segment_epoch, _sfreq, _cfg):
            return segment_epoch[:, :1].T

        def _fake_fit(peak_maps, n_states, random_state):
            _ = random_state
            calls.append(peak_maps.copy())
            return np.tile(peak_maps[:1], (n_states, 1))

        base_cfg = DotConfig(
            {
                "feature_engineering": {
                    "microstates": {
                        "n_states": 2,
                        "min_duration_ms": 0.0,
                        "min_peak_distance_ms": 5.0,
                        "max_gfp_peaks_per_epoch": 200,
                    }
                }
            }
        )

        ctx_early = SimpleNamespace(
            epochs=epochs,
            windows=_NamedWindowStub("early", {"early": early_mask}),
            name="early",
            explicit_windows=explicit_windows,
            config=base_cfg,
            logger=logging.getLogger("microstate-explicit-early"),
            fixed_templates=None,
            fixed_template_ch_names=None,
        )
        ctx_late = SimpleNamespace(
            epochs=epochs,
            windows=_NamedWindowStub("late", {"late": late_mask}),
            name="late",
            explicit_windows=explicit_windows,
            config=base_cfg,
            logger=logging.getLogger("microstate-explicit-late"),
            fixed_templates=None,
            fixed_template_ch_names=None,
        )

        with patch.object(mod, "_extract_peak_topographies", side_effect=_fake_peak_maps), patch.object(
            mod, "_fit_templates_kmeans", side_effect=_fake_fit
        ):
            mod.extract_microstate_features(ctx_early)
            mod.extract_microstate_features(ctx_late)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].shape[0], 2)
        self.assertTrue(np.allclose(calls[0], calls[1]))

    def test_fitted_templates_use_neutral_labels(self):
        from eeg_pipeline.analysis.features import microstates as mod

        epochs, _, _ = self._build_epochs()
        mask = np.ones(epochs.get_data().shape[-1], dtype=bool)

        ctx = SimpleNamespace(
            epochs=epochs,
            windows=_WindowStub(mask),
            name="active",
            config=DotConfig(
                {
                    "feature_engineering": {
                        "microstates": {
                            "n_states": 2,
                            "min_duration_ms": 0.0,
                            "min_peak_distance_ms": 5.0,
                            "max_gfp_peaks_per_epoch": 200,
                        }
                    }
                }
            ),
            logger=logging.getLogger("microstate-neutral-labels"),
            fixed_templates=None,
            fixed_template_ch_names=None,
        )

        with patch.object(mod, "_fit_templates_kmeans", return_value=np.array([[1.0] * 6, [0.0] * 6])):
            df, _ = mod.extract_microstate_features(ctx)

        self.assertIn("microstates_active_broadband_global_coverage_state1", df.columns)
        self.assertNotIn("microstates_active_broadband_global_coverage_a", df.columns)

    def test_fixed_templates_without_labels_use_neutral_labels(self):
        from eeg_pipeline.analysis.features import microstates as mod

        epochs, templates, ch_names = self._build_epochs()
        mask = np.ones(epochs.get_data().shape[-1], dtype=bool)
        ctx = SimpleNamespace(
            epochs=epochs,
            windows=_WindowStub(mask),
            name="active",
            config=DotConfig(
                {
                    "feature_engineering": {
                        "microstates": {
                            "n_states": 4,
                            "min_duration_ms": 0.0,
                            "min_peak_distance_ms": 5.0,
                            "max_gfp_peaks_per_epoch": 200,
                        }
                    }
                }
            ),
            logger=logging.getLogger("microstate-fixed-neutral"),
            fixed_templates=templates,
            fixed_template_ch_names=ch_names,
            fixed_template_labels=None,
        )

        df, _ = mod.extract_microstate_features(ctx)
        self.assertIn("microstates_active_broadband_global_coverage_state1", df.columns)
        self.assertNotIn("microstates_active_broadband_global_coverage_a", df.columns)
        self.assertFalse(bool(df.attrs.get("microstate_labels_canonical")))

    def test_fixed_template_labels_reorder_to_canonical(self):
        from eeg_pipeline.analysis.features import microstates as mod

        epochs, templates, ch_names = self._build_epochs()
        reversed_templates = templates[::-1].copy()
        mask = np.ones(epochs.get_data().shape[-1], dtype=bool)
        ctx = SimpleNamespace(
            epochs=epochs,
            windows=_WindowStub(mask),
            name="active",
            config=DotConfig(
                {
                    "feature_engineering": {
                        "microstates": {
                            "n_states": 4,
                            "min_duration_ms": 0.0,
                            "min_peak_distance_ms": 5.0,
                            "max_gfp_peaks_per_epoch": 200,
                        }
                    }
                }
            ),
            logger=logging.getLogger("microstate-fixed-reorder"),
            fixed_templates=reversed_templates,
            fixed_template_ch_names=ch_names,
            fixed_template_labels=["d", "c", "b", "a"],
        )

        df, _ = mod.extract_microstate_features(ctx)
        self.assertIn("microstates_active_broadband_global_coverage_a", df.columns)
        self.assertNotIn("microstates_active_broadband_global_coverage_state1", df.columns)
        self.assertTrue(bool(df.attrs.get("microstate_labels_canonical")))
