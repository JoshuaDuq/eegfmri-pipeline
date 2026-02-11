from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from eeg_pipeline.analysis.features.connectivity import (
    _compute_psi_imaginary,
    extract_directed_connectivity_from_precomputed,
    extract_connectivity_features,
    extract_directed_connectivity_features,
)
from eeg_pipeline.domain.features.naming import NamingSchema
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
    def test_psi_is_zero_for_constant_phase_lag(self):
        """PSI should be ~0 when coherency phase is constant across frequency."""
        csd = np.zeros((1, 2, 2, 3), dtype=complex)

        # Auto-spectra (normalization denominator).
        csd[0, 0, 0, :] = 1.0 + 0.0j
        csd[0, 1, 1, :] = 1.0 + 0.0j

        # Cross-spectrum with constant imaginary coherency (no phase slope).
        csd[0, 0, 1, :] = 1.0j
        csd[0, 1, 0, :] = -1.0j

        psi = _compute_psi_imaginary(csd)
        self.assertEqual(psi.shape, (1, 2, 2))
        self.assertAlmostEqual(float(psi[0, 0, 1]), 0.0, places=7)

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

    def test_condition_granularity_auto_promotes_phase_estimator_without_train_mask(self):
        config = DotConfig(
            {
                "feature_engineering": {
                    "connectivity": {
                        "measures": ["wpli"],
                        "granularity": "condition",
                        "condition_column": "condition",
                        "phase_estimator": "within_epoch",
                        "min_epochs_per_group": 1,
                    }
                }
            }
        )
        precomputed = _make_precomputed(transform="none", family="connectivity")
        ctx = _ContextStub(config=config, precomputed=precomputed)
        ctx._by_family["connectivity"] = precomputed
        ctx.aligned_events = pd.DataFrame({"condition": ["a", "a", "b", "b"]})

        out_df = pd.DataFrame({"conn_demo": [1.0, 2.0, 3.0, 4.0]})
        with (
            patch(
                "eeg_pipeline.analysis.features.connectivity.extract_connectivity_from_precomputed",
                return_value=(out_df, list(out_df.columns)),
            ),
            patch(
                "eeg_pipeline.analysis.features.connectivity._apply_across_epochs_phase_estimates_inplace",
            ) as mock_apply,
        ):
            df, _cols = extract_connectivity_features(ctx, ["alpha"])

        mock_apply.assert_called_once()
        self.assertEqual(df.attrs.get("phase_estimator"), "across_epochs")

    def test_condition_granularity_keeps_within_epoch_with_train_mask(self):
        config = DotConfig(
            {
                "feature_engineering": {
                    "connectivity": {
                        "measures": ["wpli"],
                        "granularity": "condition",
                        "condition_column": "condition",
                        "phase_estimator": "within_epoch",
                        "min_epochs_per_group": 1,
                    }
                }
            }
        )
        precomputed = _make_precomputed(transform="none", family="connectivity")
        ctx = _ContextStub(config=config, precomputed=precomputed)
        ctx._by_family["connectivity"] = precomputed
        ctx.aligned_events = pd.DataFrame({"condition": ["a", "a", "b", "b"]})
        ctx.train_mask = np.array([True, True, False, False], dtype=bool)

        out_df = pd.DataFrame({"conn_demo": [1.0, 2.0, 3.0, 4.0]})
        with (
            patch(
                "eeg_pipeline.analysis.features.connectivity.extract_connectivity_from_precomputed",
                return_value=(out_df, list(out_df.columns)),
            ),
            patch(
                "eeg_pipeline.analysis.features.connectivity._apply_across_epochs_phase_estimates_inplace",
            ) as mock_apply,
        ):
            df, _cols = extract_connectivity_features(ctx, ["alpha"])

        mock_apply.assert_not_called()
        self.assertEqual(df.attrs.get("phase_estimator"), "within_epoch")

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

    def test_directed_dtf_fwd_bwd_match_channel_pair_direction(self):
        precomputed = _make_precomputed(transform="none", family="directedconnectivity")
        precomputed.ch_names = ["A", "B", "C"]
        precomputed.data = np.zeros((1, 3, 60), dtype=float)
        precomputed.frequency_bands = {"alpha": [8.0, 12.0]}
        precomputed.band_data = {"alpha": precomputed.band_data["alpha"]}

        config = DotConfig(
            {
                "feature_engineering": {
                    "directedconnectivity": {
                        "enable_psi": False,
                        "enable_dtf": True,
                        "enable_pdc": False,
                        "output_level": "full",
                        "min_segment_samples": 10,
                        "mvar_order": 2,
                        "n_freqs": 8,
                    }
                }
            }
        )

        dtf_matrix = np.array(
            [
                [0.0, 0.2, 0.8],
                [0.9, 0.0, 0.1],
                [0.3, 0.7, 0.0],
            ],
            dtype=float,
        )

        with patch(
            "eeg_pipeline.analysis.features.connectivity._compute_directed_connectivity_epoch",
            return_value={"dtf": dtf_matrix},
        ):
            df, cols = extract_directed_connectivity_from_precomputed(
                precomputed,
                bands=["alpha"],
                segments=["full"],
                config=config,
                logger=None,
            )

        col_ab_fwd = NamingSchema.build("dconn", "full", "alpha", "chpair", "dtf_fwd", channel_pair="A-B")
        col_ab_bwd = NamingSchema.build("dconn", "full", "alpha", "chpair", "dtf_bwd", channel_pair="A-B")
        col_ac_fwd = NamingSchema.build("dconn", "full", "alpha", "chpair", "dtf_fwd", channel_pair="A-C")
        col_ac_bwd = NamingSchema.build("dconn", "full", "alpha", "chpair", "dtf_bwd", channel_pair="A-C")
        col_bc_fwd = NamingSchema.build("dconn", "full", "alpha", "chpair", "dtf_fwd", channel_pair="B-C")
        col_bc_bwd = NamingSchema.build("dconn", "full", "alpha", "chpair", "dtf_bwd", channel_pair="B-C")
        col_fwd_mean = NamingSchema.build("dconn", "full", "alpha", "global", "dtf_fwd_mean")
        col_bwd_mean = NamingSchema.build("dconn", "full", "alpha", "global", "dtf_bwd_mean")
        col_asym = NamingSchema.build("dconn", "full", "alpha", "global", "dtf_asymmetry")

        for col in (
            col_ab_fwd,
            col_ab_bwd,
            col_ac_fwd,
            col_ac_bwd,
            col_bc_fwd,
            col_bc_bwd,
            col_fwd_mean,
            col_bwd_mean,
            col_asym,
        ):
            self.assertIn(col, cols)
            self.assertIn(col, df.columns)

        # For DTF/PDC, matrix entry [target, source] encodes source -> target.
        self.assertAlmostEqual(float(df[col_ab_fwd].iloc[0]), 0.9, places=8)  # A -> B
        self.assertAlmostEqual(float(df[col_ab_bwd].iloc[0]), 0.2, places=8)  # B -> A
        self.assertAlmostEqual(float(df[col_ac_fwd].iloc[0]), 0.3, places=8)  # A -> C
        self.assertAlmostEqual(float(df[col_ac_bwd].iloc[0]), 0.8, places=8)  # C -> A
        self.assertAlmostEqual(float(df[col_bc_fwd].iloc[0]), 0.7, places=8)  # B -> C
        self.assertAlmostEqual(float(df[col_bc_bwd].iloc[0]), 0.1, places=8)  # C -> B

        expected_fwd = np.mean([0.9, 0.3, 0.7])
        expected_bwd = np.mean([0.2, 0.8, 0.1])
        expected_asym = expected_fwd - expected_bwd
        self.assertAlmostEqual(float(df[col_fwd_mean].iloc[0]), expected_fwd, places=8)
        self.assertAlmostEqual(float(df[col_bwd_mean].iloc[0]), expected_bwd, places=8)
        self.assertAlmostEqual(float(df[col_asym].iloc[0]), expected_asym, places=8)


if __name__ == "__main__":
    unittest.main()
