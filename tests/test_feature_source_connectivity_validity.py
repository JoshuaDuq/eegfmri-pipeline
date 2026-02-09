from __future__ import annotations

import logging
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from eeg_pipeline.analysis.features.source_localization import (
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


class _FakeCon:
    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data, dtype=float)

    def get_data(self):
        return self._data


class TestSourceConnectivityValidity(unittest.TestCase):
    def test_wpli_uses_train_mask_and_marks_broadcast_attrs(self):
        n_epochs = 4
        train_mask = np.array([True, True, False, False], dtype=bool)
        roi_data = np.random.default_rng(3).standard_normal((n_epochs, 2, 80))

        captured = {"n_epochs_fit": None}

        def _fake_spectral_connectivity_epochs(data, **_kwargs):
            captured["n_epochs_fit"] = int(np.asarray(data).shape[0])
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
        self.assertEqual(df.attrs.get("feature_granularity"), "subject")
        self.assertIn("broadcast_warning", df.attrs)
        self.assertTrue(bool(df.attrs.get("threshold_train_mask_used")))


if __name__ == "__main__":
    unittest.main()
