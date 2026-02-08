from __future__ import annotations

import os
import logging
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from eeg_pipeline.analysis.features.connectivity import extract_connectivity_from_precomputed
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.types import BandData, PrecomputedData, TimeWindows
from tests.pipelines_test_utils import DotConfig


class _DummyConnectivity:
    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data)

    def get_data(self) -> np.ndarray:
        return self._data


def _fake_spectral_connectivity_time(
    seg_data: np.ndarray,
    *,
    indices,
    average: bool,
    **_kwargs,
):
    n_pairs = len(indices[0])
    n_epochs = int(seg_data.shape[0])
    if average:
        data = np.full((n_pairs, 1), 0.2, dtype=float)
    else:
        base = np.linspace(0.15, 0.35, n_pairs, dtype=float)
        data = np.tile(base[None, :, None], (n_epochs, 1, 1))
    return _DummyConnectivity(data)


def _fake_envelope_correlation(
    analytic_seg: np.ndarray,
    **_kwargs,
):
    n_epochs, n_channels, _ = analytic_seg.shape
    dense = np.zeros((n_epochs, n_channels, n_channels), dtype=float)
    for ep_idx in range(n_epochs):
        env = np.abs(analytic_seg[ep_idx])
        corr = np.corrcoef(env)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        dense[ep_idx] = corr
    return _DummyConnectivity(dense)


def _fake_state_labels(window_vectors: np.ndarray, *, n_states: int, random_state: int):
    del random_state
    n_epochs, n_windows, _ = window_vectors.shape
    seq = np.arange(n_windows, dtype=int) % max(2, int(n_states))
    return np.tile(seq[None, :], (n_epochs, 1))


class TestDynamicConnectivityFeatures(unittest.TestCase):
    def _build_precomputed(self) -> PrecomputedData:
        rng = np.random.default_rng(23)
        sfreq = 100.0
        times = np.arange(0.0, 3.0, 1.0 / sfreq)
        n_epochs = 6
        ch_names = ["C3", "C4", "P3", "P4"]
        n_channels = len(ch_names)
        n_times = len(times)

        analytic = np.zeros((n_epochs, n_channels, n_times), dtype=np.complex128)
        for ep_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                freq = 9.5 + 0.3 * ch_idx
                amp = 1.0 + 0.15 * np.sin(2 * np.pi * 0.6 * times + 0.15 * ep_idx)
                amp += 0.05 * rng.standard_normal(n_times)
                amp = np.clip(amp, 0.1, None)
                phase = 2 * np.pi * freq * times + 0.25 * ep_idx + 0.35 * ch_idx
                phase += 0.08 * rng.standard_normal(n_times)
                analytic[ep_idx, ch_idx, :] = amp * np.exp(1j * phase)

        filtered = np.real(analytic)
        envelope = np.abs(analytic)
        band_data = {
            "alpha": BandData(
                band="alpha",
                fmin=8.0,
                fmax=12.9,
                filtered=filtered,
                analytic=analytic,
                envelope=envelope,
                phase=np.angle(analytic),
                power=envelope**2,
            )
        }

        windows = TimeWindows(times=times, masks={}, ranges={})
        config = DotConfig(
            {
                "rois": {
                    "Left": ["C3", "P3"],
                    "Right": ["C4", "P4"],
                },
                "feature_engineering": {
                    "connectivity": {
                        "measures": ["wpli", "aec"],
                        "output_level": "full",
                        "enable_graph_metrics": False,
                        "aec_mode": "orth",
                        "aec_absolute": True,
                        "aec_output": ["r"],
                        "phase_estimator": "within_epoch",
                        "sliding_window_len": 1.0,
                        "sliding_window_step": 0.5,
                        "dynamic_enabled": True,
                        "dynamic_measures": ["wpli", "aec"],
                        "dynamic_autocorr_lag": 1,
                        "dynamic_min_windows": 3,
                        "dynamic_include_roi_pairs": True,
                        "dynamic_state_enabled": True,
                        "dynamic_state_n_states": 3,
                        "dynamic_state_min_windows": 4,
                        "dynamic_state_random_state": 7,
                    },
                    "parallel": {"n_jobs_connectivity": 1},
                },
            }
        )

        return PrecomputedData(
            data=filtered,
            times=times,
            sfreq=sfreq,
            ch_names=ch_names,
            picks=np.arange(n_channels),
            windows=windows,
            metadata=pd.DataFrame({"trial_type": ["pain"] * n_epochs}),
            band_data=band_data,
            config=config,
            logger=logging.getLogger("test-connectivity-dynamic"),
            spatial_modes=["global"],
            frequency_bands={"alpha": [8.0, 12.9]},
        )

    def test_extracts_dynamic_edge_roi_and_state_features(self):
        precomputed = self._build_precomputed()

        with (
            patch(
                "eeg_pipeline.analysis.features.connectivity.spectral_connectivity_time",
                new=_fake_spectral_connectivity_time,
            ),
            patch(
                "eeg_pipeline.analysis.features.connectivity.envelope_correlation",
                new=_fake_envelope_correlation,
            ),
            patch(
                "eeg_pipeline.analysis.features.connectivity._fit_dynamic_state_labels",
                new=_fake_state_labels,
            ),
        ):
            df, cols = extract_connectivity_from_precomputed(
                precomputed,
                bands=["alpha"],
                segments=["full"],
                config=precomputed.config,
                logger=precomputed.logger,
            )

        expected = [
            NamingSchema.build("conn", "full", "alpha", "chpair", "wpliswmean", channel_pair="C3-C4"),
            NamingSchema.build("conn", "full", "alpha", "chpair", "aecswstd", channel_pair="C3-C4"),
            NamingSchema.build("conn", "full", "alpha", "roi", "wpliswac1", channel="Left-Right"),
            NamingSchema.build("conn", "full", "alpha", "global", "wpliswtopostab"),
            NamingSchema.build("conn", "full", "alpha", "global", "wpliswswitch"),
            NamingSchema.build("conn", "full", "alpha", "global", "wpliswdwellsec"),
            NamingSchema.build("conn", "full", "alpha", "global", "wpliswstateent"),
            NamingSchema.build("conn", "full", "alpha", "global", "aecswmean"),
        ]
        for col in expected:
            self.assertIn(col, cols)
            self.assertIn(col, df.columns)
            self.assertTrue(np.isfinite(np.asarray(df[col], dtype=float)).any(), msg=col)

    def test_task_parallel_uses_thread_backend(self):
        precomputed = self._build_precomputed()
        precomputed.config["feature_engineering"]["parallel"]["n_jobs_connectivity"] = 2
        seg_name = "all"
        seg_mask = np.ones(precomputed.times.shape[0], dtype=bool)
        precomputed.windows.masks[seg_name] = seg_mask
        precomputed.windows.ranges[seg_name] = (
            float(precomputed.times[0]),
            float(precomputed.times[-1]),
        )

        backends_seen = []
        real_parallel = extract_connectivity_from_precomputed.__globals__["Parallel"]

        def _capture_parallel(*args, **kwargs):
            backends_seen.append(kwargs.get("backend"))
            return real_parallel(*args, **kwargs)

        with (
            patch(
                "eeg_pipeline.analysis.features.connectivity.spectral_connectivity_time",
                new=_fake_spectral_connectivity_time,
            ),
            patch(
                "eeg_pipeline.analysis.features.connectivity.envelope_correlation",
                new=_fake_envelope_correlation,
            ),
            patch(
                "eeg_pipeline.analysis.features.connectivity._fit_dynamic_state_labels",
                new=_fake_state_labels,
            ),
            patch(
                "eeg_pipeline.analysis.features.connectivity.Parallel",
                new=_capture_parallel,
            ),
            patch.dict(os.environ, {"EEG_PIPELINE_N_JOBS": "2"}),
        ):
            extract_connectivity_from_precomputed(
                precomputed,
                bands=["alpha"],
                segments=[seg_name],
                config=precomputed.config,
                logger=precomputed.logger,
            )

        self.assertTrue(backends_seen)
        self.assertIn("threading", backends_seen)


if __name__ == "__main__":
    unittest.main()
