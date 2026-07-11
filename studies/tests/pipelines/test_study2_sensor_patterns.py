from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from studies.pain_study.study2.sensor_patterns import (
    aggregate_fold_patterns,
    normalize_band_patterns,
    parse_channel_band,
)

BANDS = ("alpha", "beta")


def test_parse_channel_band_requires_exact_primary_feature() -> None:
    assert parse_channel_band(
        "power_active_gamma_low_clean_ch_FC1_logratio",
        ("alpha", "gamma_low_clean"),
    ) == ("FC1", "gamma_low_clean")
    with pytest.raises(ValueError, match="primary channel-power"):
        parse_channel_band("power_active_alpha_roi_Frontal_logratio", ("alpha",))


def test_normalize_band_patterns_gives_each_band_unit_norm() -> None:
    frame = pd.DataFrame(
        {
            "band": ["alpha", "alpha", "beta", "beta"],
            "channel": ["C3", "C4", "C3", "C4"],
            "haufe_pattern": [3.0, 4.0, -2.0, 0.0],
        }
    )

    normalized = normalize_band_patterns(frame, BANDS)

    norms = normalized.groupby("band", sort=False)["normalized_pattern"].apply(
        lambda values: np.linalg.norm(values.to_numpy())
    )
    assert np.allclose(norms.to_numpy(), 1.0)


def test_normalize_band_patterns_rejects_zero_norm() -> None:
    frame = pd.DataFrame(
        {"band": ["alpha", "alpha"], "channel": ["C3", "C4"], "haufe_pattern": 0.0}
    )
    with pytest.raises(ValueError, match="zero norm"):
        normalize_band_patterns(frame, ("alpha",))


def test_aggregate_fold_patterns_returns_median_and_pairwise_stability() -> None:
    rows = []
    fold_maps = (
        (0, [1.0, -1.0, 2.0, 0.0]),
        (1, [0.8, -1.2, 1.8, 0.2]),
        (2, [1.2, -0.8, 2.2, -0.2]),
    )
    for fold, values in fold_maps:
        for band_index, band in enumerate(BANDS):
            for channel_index, channel in enumerate(("C3", "C4")):
                rows.append(
                    {
                        "target": "NPS",
                        "fold": fold,
                        "test_subject": f"sub-{fold:04d}",
                        "band": band,
                        "channel": channel,
                        "normalized_pattern": values[2 * band_index + channel_index],
                    }
                )

    maps, stability = aggregate_fold_patterns(pd.DataFrame(rows), BANDS)

    assert len(maps) == 4
    assert maps.loc[maps["band"].eq("alpha"), "median_normalized_pattern"].tolist() == [
        1.0,
        -1.0,
    ]
    assert len(stability) == 6
    assert set(stability["comparison_index"]) == {0, 1, 2}
    assert stability["spatial_correlation"].between(-1.0, 1.0).all()


def test_aggregate_fold_patterns_rejects_incomplete_channel_maps() -> None:
    frame = pd.DataFrame(
        {
            "target": "NPS",
            "fold": [0, 0, 1],
            "test_subject": ["sub-0000", "sub-0000", "sub-0001"],
            "band": "alpha",
            "channel": ["C3", "C4", "C3"],
            "normalized_pattern": [1.0, -1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="identical channel sets"):
        aggregate_fold_patterns(frame, ("alpha",))
