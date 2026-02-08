from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from eeg_pipeline.analysis.features.connectivity import (
    extract_connectivity_features,
    extract_directed_connectivity_features,
)
from eeg_pipeline.types import BandData, PrecomputedData
from tests.pipelines_test_utils import DotConfig


class _EpochStub:
    preload = True

    def load_data(self):
        self.preload = True


class _ContextStub:
    def __init__(self, config, precomputed=None):
        self.config = config
        self.precomputed = precomputed
        self._by_family = {}
        self.epochs = _EpochStub()
        self.windows = None
        self.name = None
        self.train_mask = None
        self.aligned_events = None
        self.logger = logging.getLogger("test-connectivity-validity")

    def get_precomputed_for_family(self, family):
        return self._by_family.get(family)

    def set_precomputed_for_family(self, family, precomputed):
        self._by_family[family] = precomputed
        self.precomputed = precomputed

    def set_precomputed(self, precomputed):
        self.precomputed = precomputed


def _make_precomputed(*, transform: str, family: str) -> PrecomputedData:
    n_epochs = 4
    n_channels = 3
    sfreq = 100.0
    times = np.linspace(0.0, 0.59, 60)
    data = np.zeros((n_epochs, n_channels, times.size), dtype=float)
    analytic = data.astype(np.complex128)
    band_data = {
        "alpha": BandData(
            band="alpha",
            fmin=8.0,
            fmax=12.0,
            filtered=data.copy(),
            analytic=analytic,
            envelope=np.abs(analytic),
            phase=np.angle(analytic),
            power=np.abs(analytic) ** 2,
        )
    }
    return PrecomputedData(
        data=data,
        times=times,
        sfreq=sfreq,
        ch_names=["C3", "C4", "Pz"],
        picks=np.arange(n_channels),
        band_data=band_data,
        logger=logging.getLogger("test-connectivity-validity"),
        feature_family=family,
        spatial_transform=transform,
    )


class TestConnectivityValidityGuards(unittest.TestCase):
    def test_connectivity_recomputes_incompatible_precomputed(self):
        config = DotConfig(
            {
                "feature_engineering": {
                    "connectivity": {
                        "measures": ["wpli"],
                        "granularity": "trial",
                        "phase_estimator": "within_epoch",
                    },
                    "spatial_transform_per_family": {
                        "spectral": "none",
                        "connectivity": "csd",
                    },
                }
            }
        )
        stale = _make_precomputed(transform="none", family="spectral")
        fresh = _make_precomputed(transform="csd", family="connectivity")
        ctx = _ContextStub(config=config, precomputed=stale)

        out_df = pd.DataFrame({"conn_demo": [1.0, 2.0, 3.0, 4.0]})
        with (
            patch(
                "eeg_pipeline.analysis.features.connectivity.precompute_data",
                return_value=fresh,
            ) as mock_precompute,
            patch(
                "eeg_pipeline.analysis.features.connectivity.extract_connectivity_from_precomputed",
                return_value=(out_df, list(out_df.columns)),
            ),
        ):
            df, cols = extract_connectivity_features(ctx, ["alpha"])

        self.assertEqual(cols, ["conn_demo"])
        self.assertEqual(df.shape, (4, 1))
        mock_precompute.assert_called_once()
        self.assertEqual(mock_precompute.call_args.kwargs.get("feature_family"), "connectivity")

    def test_trial_granularity_marks_across_epochs_as_broadcast(self):
        config = DotConfig(
            {
                "feature_engineering": {
                    "connectivity": {
                        "measures": ["wpli"],
                        "granularity": "trial",
                        "phase_estimator": "across_epochs",
                        "force_within_epoch_for_ml": False,
                    }
                }
            }
        )
        precomputed = _make_precomputed(transform="none", family="connectivity")
        ctx = _ContextStub(config=config, precomputed=precomputed)
        ctx._by_family["connectivity"] = precomputed

        out_df = pd.DataFrame({"conn_demo": [1.0, 2.0, 3.0, 4.0]})
        with patch(
            "eeg_pipeline.analysis.features.connectivity.extract_connectivity_from_precomputed",
            return_value=(out_df, list(out_df.columns)),
        ):
            df, _cols = extract_connectivity_features(ctx, ["alpha"])

        self.assertEqual(df.attrs.get("feature_granularity"), "trial")
        self.assertIn("broadcast_warning", df.attrs)
        self.assertEqual(df.attrs.get("phase_estimator"), "across_epochs")

    def test_directed_connectivity_recomputes_incompatible_precomputed(self):
        config = DotConfig(
            {
                "feature_engineering": {
                    "connectivity": {
                        "measures": ["wpli"],
                    },
                    "spatial_transform_per_family": {
                        "connectivity": "laplacian",
                        "spectral": "none",
                    },
                }
            }
        )
        stale = _make_precomputed(transform="none", family="spectral")
        fresh = _make_precomputed(transform="laplacian", family="directedconnectivity")
        ctx = _ContextStub(config=config, precomputed=stale)

        out_df = pd.DataFrame({"dconn_demo": [0.1, 0.2, 0.3, 0.4]})
        with (
            patch(
                "eeg_pipeline.analysis.features.connectivity.precompute_data",
                return_value=fresh,
            ) as mock_precompute,
            patch(
                "eeg_pipeline.analysis.features.connectivity.extract_directed_connectivity_from_precomputed",
                return_value=(out_df, list(out_df.columns)),
            ),
        ):
            df, cols = extract_directed_connectivity_features(ctx, ["alpha"])

        self.assertEqual(cols, ["dconn_demo"])
        self.assertEqual(df.shape, (4, 1))
        mock_precompute.assert_called_once()
        self.assertEqual(
            mock_precompute.call_args.kwargs.get("feature_family"),
            "directedconnectivity",
        )


if __name__ == "__main__":
    unittest.main()
