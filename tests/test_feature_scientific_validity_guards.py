from __future__ import annotations

import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from eeg_pipeline.analysis.features.preparation import _determine_frequency_bands
from eeg_pipeline.analysis.features.source_localization import (
    extract_source_connectivity_features,
    extract_source_localization_features,
)
from eeg_pipeline.types import PrecomputedQC, TimeWindows
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


if __name__ == "__main__":
    unittest.main()
