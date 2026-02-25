from __future__ import annotations

import logging
import unittest

import numpy as np
import pandas as pd

from eeg_pipeline.analysis.features.precomputed.erds import extract_erds_from_precomputed
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.types import BandData, PrecomputedData, TimeWindows
from tests.pipelines_test_utils import DotConfig


class TestERDSPainMarkers(unittest.TestCase):
    def _build_precomputed(self) -> PrecomputedData:
        n_epochs = 12
        sfreq = 100.0
        times = np.arange(-0.5, 1.0, 1.0 / sfreq)
        n_times = len(times)

        ch_names = ["C3", "C4", "CP3", "CP4", "FC3", "FC4", "Cz", "Fz"]
        n_channels = len(ch_names)

        baseline_power = 10.0
        power = np.full((n_epochs, n_channels, n_times), baseline_power, dtype=float)

        left_hemi = [0, 2, 4]   # C3, CP3, FC3
        right_hemi = [1, 3, 5]  # C4, CP4, FC4

        def apply_pattern(channel_indices: list[int], erd_pct: float, rebound_pct: float | None) -> None:
            active = times >= 0.0
            erd_mask = (times >= 0.20) & (times < 0.45)
            rebound_mask = times >= 0.60
            for epoch_idx in range(n_epochs):
                for ch_idx in channel_indices:
                    trace_pct = np.zeros(n_times, dtype=float)
                    trace_pct[active & erd_mask] = erd_pct
                    if rebound_pct is not None:
                        trace_pct[active & rebound_mask] = rebound_pct
                    power[epoch_idx, ch_idx, :] = baseline_power * (1.0 + trace_pct / 100.0)

        # Base weak bilateral pattern.
        apply_pattern(left_hemi, erd_pct=-8.0, rebound_pct=4.0)
        apply_pattern(right_hemi, erd_pct=-8.0, rebound_pct=4.0)

        stimulated_side = []
        for ep in range(n_epochs):
            # contralateral-right with rebound
            if ep % 3 == 0:
                stimulated_side.append("left")
                for ch_idx in right_hemi:
                    erd_mask = (times >= 0.20) & (times < 0.45)
                    rebound_mask = times >= 0.60
                    power[ep, ch_idx, erd_mask] = baseline_power * 0.70
                    power[ep, ch_idx, rebound_mask] = baseline_power * 1.20
            # contralateral-left without rebound
            elif ep % 3 == 1:
                stimulated_side.append("right")
                for ch_idx in left_hemi:
                    erd_mask = (times >= 0.20) & (times < 0.45)
                    rebound_mask = times >= 0.60
                    power[ep, ch_idx, erd_mask] = baseline_power * 0.72
                    power[ep, ch_idx, rebound_mask] = baseline_power * 1.00
            # unknown side; right hemisphere has stronger ERD so inference should pick it
            else:
                stimulated_side.append("unknown")
                for ch_idx in right_hemi:
                    erd_mask = (times >= 0.20) & (times < 0.45)
                    rebound_mask = times >= 0.60
                    power[ep, ch_idx, erd_mask] = baseline_power * 0.65
                    power[ep, ch_idx, rebound_mask] = baseline_power * 1.22

        band_data = BandData(
            band="alpha",
            fmin=8.0,
            fmax=12.9,
            filtered=np.zeros((n_epochs, n_channels, n_times), dtype=float),
            analytic=np.zeros((n_epochs, n_channels, n_times), dtype=complex),
            envelope=np.zeros((n_epochs, n_channels, n_times), dtype=float),
            phase=np.zeros((n_epochs, n_channels, n_times), dtype=float),
            power=power,
        )

        baseline_mask = times < 0.0
        active_mask = times >= 0.0
        windows = TimeWindows(
            baseline_mask=baseline_mask,
            active_mask=active_mask,
            masks={"baseline": baseline_mask, "active": active_mask},
            ranges={"baseline": (-0.5, 0.0), "active": (0.0, 1.0)},
            times=times,
        )

        config = DotConfig(
            {
                "feature_engineering": {
                    "constants": {
                        "min_epochs_for_features": 10,
                        "min_valid_fraction": 0.5,
                    },
                    "erds": {
                        "bands": ["alpha"],
                        "enable_laterality_markers": True,
                        "laterality_marker_bands": ["alpha"],
                        "laterality_columns": ["stimulated_side"],
                        "infer_contralateral_when_missing": True,
                        "onset_threshold_sigma": 0.5,
                        "onset_min_threshold_percent": 5.0,
                        "onset_min_duration_ms": 20.0,
                        "rebound_threshold_sigma": 0.5,
                        "rebound_min_threshold_percent": 5.0,
                        "rebound_min_latency_ms": 80.0,
                    },
                }
            }
        )

        return PrecomputedData(
            data=np.copy(power),
            times=times,
            sfreq=sfreq,
            ch_names=ch_names,
            picks=np.arange(n_channels),
            windows=windows,
            metadata=pd.DataFrame({"stimulated_side": stimulated_side}),
            band_data={"alpha": band_data},
            config=config,
            logger=logging.getLogger("test-erds-pain"),
            spatial_modes=["channels", "roi", "global"],
        )

    def test_extracts_contralateral_alpha_erd_and_rebound_metrics(self):
        precomputed = self._build_precomputed()

        df, cols, _qc = extract_erds_from_precomputed(precomputed, ["alpha"])

        onset_col = NamingSchema.build(
            "erds", "active", "alpha", "roi", "onset_latency", channel="Somatosensory_Contralateral"
        )
        erd_peak_col = NamingSchema.build(
            "erds", "active", "alpha", "roi", "peak_latency", channel="Somatosensory_Contralateral"
        )
        erd_mag_col = NamingSchema.build(
            "erds", "active", "alpha", "roi", "erd_magnitude", channel="Somatosensory_Contralateral"
        )
        rebound_mag_col = NamingSchema.build(
            "erds", "active", "alpha", "roi", "rebound_magnitude", channel="Somatosensory_Contralateral"
        )
        rebound_lat_col = NamingSchema.build(
            "erds", "active", "alpha", "roi", "rebound_latency", channel="Somatosensory_Contralateral"
        )

        for col in (onset_col, erd_peak_col, erd_mag_col, rebound_mag_col, rebound_lat_col):
            self.assertIn(col, cols)
            self.assertIn(col, df.columns)

        row_left = df.iloc[0]
        self.assertAlmostEqual(float(row_left[onset_col]), 0.20, places=2)
        self.assertAlmostEqual(float(row_left[erd_peak_col]), 0.20, places=2)
        self.assertGreater(float(row_left[erd_mag_col]), 25.0)
        self.assertGreater(float(row_left[rebound_mag_col]), 15.0)
        self.assertAlmostEqual(float(row_left[rebound_lat_col]), 0.60, places=2)

        row_right_no_rebound = df.iloc[1]
        self.assertGreater(float(row_right_no_rebound[erd_mag_col]), 20.0)
        self.assertTrue(np.isnan(float(row_right_no_rebound[rebound_mag_col])))
        self.assertTrue(np.isnan(float(row_right_no_rebound[rebound_lat_col])))

        row_unknown_side = df.iloc[2]
        self.assertGreater(float(row_unknown_side[erd_mag_col]), 30.0)
        self.assertGreater(float(row_unknown_side[rebound_mag_col]), 15.0)


if __name__ == "__main__":
    unittest.main()
