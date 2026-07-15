from __future__ import annotations

from dataclasses import FrozenInstanceError

import mne
import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.utils.config.loader import ConfigDict
from studies.pain_study.study1.figures.sensor_topography_data import (
    SensorPowerData,
    build_sensor_power_data,
    load_sensor_montage,
    reconstruct_channel_power,
)
from studies.pain_study.study1.figures.validity_data import ValidityTrialData

BANDS = (
    "alpha",
    "beta",
    "gamma_low_clean",
    "gamma_mid_clean",
    "gamma_high_clean",
)
CHANNELS = ("Cz", "Fz")


def test_reconstruct_channel_power_converts_logratio_to_db_in_deterministic_order() -> None:
    table = _power_table(trial_ids=(2, 1), channels=("Fz", "Cz"))
    table.loc[table["trial_id"].eq(1), "power_active_alpha_ch_Cz_logratio"] = 0.2

    result = reconstruct_channel_power(table, subject_id="sub-02", bands=BANDS)

    assert result.loc[
        result[["trial_id", "band", "channel"]].eq([1, "alpha", "Cz"]).all(axis=1),
        "power_db",
    ].item() == pytest.approx(2.0)
    assert list(result[["subject_id", "trial_id", "band", "channel"]].itertuples(False, None)) == [
        ("sub-02", trial_id, band, channel)
        for trial_id in (1, 2)
        for band in BANDS
        for channel in CHANNELS
    ]


@pytest.mark.parametrize("bands", [BANDS[:-1], (BANDS[1], BANDS[0], *BANDS[2:])])
def test_reconstruct_channel_power_requires_exact_configured_bands(
    bands: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="exactly match"):
        reconstruct_channel_power(_power_table(), subject_id="sub-01", bands=bands)


def test_reconstruct_channel_power_rejects_duplicate_trial_ids() -> None:
    with pytest.raises(ValueError, match="duplicate trial IDs"):
        reconstruct_channel_power(
            _power_table(trial_ids=(1, 1)),
            subject_id="sub-01",
            bands=BANDS,
        )


def test_reconstruct_channel_power_rejects_missing_baseline_logratio_pair() -> None:
    table = _power_table().drop(columns="power_active_alpha_ch_Cz_logratio")

    with pytest.raises(ValueError, match="sets differ"):
        reconstruct_channel_power(table, subject_id="sub-01", bands=BANDS)


def test_reconstruct_channel_power_rejects_cross_band_channel_disagreement() -> None:
    table = _power_table().drop(
        columns=[
            "power_baseline_gamma_high_clean_ch_Cz_mean",
            "power_active_gamma_high_clean_ch_Cz_logratio",
        ]
    )

    with pytest.raises(ValueError, match="identical channel sets"):
        reconstruct_channel_power(table, subject_id="sub-01", bands=BANDS)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("power_baseline_alpha_ch_Cz_mean", np.nan, "Baseline channel power"),
        ("power_baseline_alpha_ch_Cz_mean", 0.0, "Baseline channel power"),
        ("power_active_alpha_ch_Cz_logratio", np.inf, "log-ratio"),
        ("power_active_alpha_ch_Cz_logratio", 400.0, "active power"),
    ],
)
def test_reconstruct_channel_power_rejects_invalid_linear_power(
    column: str,
    value: float,
    message: str,
) -> None:
    table = _power_table()
    table.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        reconstruct_channel_power(table, subject_id="sub-01", bands=BANDS)


@pytest.mark.parametrize(
    "column",
    [
        "power_bad",
        "power_active_alpha_ch_logratio",
    ],
)
def test_reconstruct_channel_power_rejects_malformed_power_feature_names(column: str) -> None:
    table = _power_table()
    table[column] = 1.0

    with pytest.raises(ValueError, match="Malformed power feature column"):
        reconstruct_channel_power(table, subject_id="sub-01", bands=BANDS)


def test_reconstruct_channel_power_rejects_unexpected_extra_band_columns() -> None:
    table = _power_table()
    table["power_baseline_delta_ch_Cz_mean"] = 1.0
    table["power_active_delta_ch_Cz_logratio"] = 0.1

    with pytest.raises(ValueError, match="unexpected band 'delta'"):
        reconstruct_channel_power(table, subject_id="sub-01", bands=BANDS)


def test_reconstruct_channel_power_rejects_unsupported_channel_statistic() -> None:
    table = _power_table()
    table["power_active_alpha_ch_Cz_mean"] = 1.0

    with pytest.raises(ValueError, match="Unsupported channel-power column"):
        reconstruct_channel_power(table, subject_id="sub-01", bands=BANDS)


def test_reconstruct_channel_power_allows_unrelated_metadata_and_global_power() -> None:
    table = _power_table()
    table["condition"] = "pain"
    table["power_active_alpha_global_mean"] = 1.0

    result = reconstruct_channel_power(table, subject_id="sub-01", bands=BANDS)

    assert len(result) == len(table) * len(BANDS) * len(CHANNELS)


def test_reconstruct_channel_power_validates_emitted_channel_db_columns() -> None:
    table = _power_table()
    for band in BANDS:
        for channel in CHANNELS:
            logratio = table[f"power_active_{band}_ch_{channel}_logratio"]
            table[f"power_active_{band}_ch_{channel}_db"] = 10.0 * logratio

    result = reconstruct_channel_power(table, subject_id="sub-01", bands=BANDS)

    assert len(result) == len(table) * len(BANDS) * len(CHANNELS)


def test_reconstruct_channel_power_rejects_inconsistent_emitted_channel_db() -> None:
    table = _power_table()
    for band in BANDS:
        for channel in CHANNELS:
            logratio = table[f"power_active_{band}_ch_{channel}_logratio"]
            table[f"power_active_{band}_ch_{channel}_db"] = 10.0 * logratio
    table.loc[0, "power_active_alpha_ch_Cz_db"] += 1.0

    with pytest.raises(ValueError, match="does not equal"):
        reconstruct_channel_power(table, subject_id="sub-01", bands=BANDS)


def test_build_sensor_power_data_returns_exact_aligned_tidy_boundary() -> None:
    validity = _validity_data()
    config = _config(nuisance_columns=("nuisance_value",))
    feature_tables = {
        subject_id: _power_table()
        for subject_id in reversed(validity.enriched_targets["subject_id"].unique())
    }

    data = build_sensor_power_data(
        validity=validity,
        feature_tables=feature_tables,
        config=config,
    )

    assert isinstance(data, SensorPowerData)
    assert data.subjects == ("sub-01", "sub-02")
    assert data.bands == BANDS
    assert data.channels == CHANNELS
    assert {
        "subject_id",
        "trial_id",
        "band",
        "channel",
        "power_db",
        "stimulus_temp",
        "run",
        "selected_surface",
        "within_run_trial",
        "residual_ecg_coupling",
        "fp1_fp2_high_frequency_power",
        "within_scale_intensity",
        "NPS",
        "SIIPS1",
        "nuisance_value",
    }.issubset(data.trials.columns)
    assert len(data.trials) == 2 * 2 * len(BANDS) * len(CHANNELS)
    with pytest.raises(FrozenInstanceError):
        data.channels = ("Cz",)  # type: ignore[misc]


def test_build_sensor_power_data_rejects_cross_subject_channel_disagreement() -> None:
    validity = _validity_data()
    feature_tables = {
        "sub-01": _power_table(),
        "sub-02": _power_table(channels=("Cz", "Pz")),
    }

    with pytest.raises(ValueError, match="identical channel sets across subjects"):
        build_sensor_power_data(
            validity=validity,
            feature_tables=feature_tables,
            config=_config(),
        )


def test_build_sensor_power_data_rejects_feature_trial_misalignment() -> None:
    validity = _validity_data()
    feature_tables = {
        "sub-01": _power_table(),
        "sub-02": _power_table(trial_ids=(1, 3)),
    }

    with pytest.raises(ValueError, match="trial IDs do not exactly match"):
        build_sensor_power_data(
            validity=validity,
            feature_tables=feature_tables,
            config=_config(),
        )


def test_build_sensor_power_data_rejects_retained_target_event_misalignment() -> None:
    validity = _validity_data()
    validity.clean_events.loc[
        validity.clean_events["subject_id"].eq("sub-02") & validity.clean_events["trial_id"].eq(2),
        "trial_number",
    ] = 3

    with pytest.raises(ValueError, match="matching feature trial IDs"):
        build_sensor_power_data(
            validity=validity,
            feature_tables={"sub-01": _power_table(), "sub-02": _power_table()},
            config=_config(),
        )


def test_build_sensor_power_data_rejects_feature_subject_disagreement() -> None:
    with pytest.raises(ValueError, match="subjects do not exactly match"):
        build_sensor_power_data(
            validity=_validity_data(),
            feature_tables={"sub-01": _power_table()},
            config=_config(),
        )


def test_load_sensor_montage_preserves_analyzed_channel_order() -> None:
    channels = ("Fz", "Cz", "Fp1")

    montage = load_sensor_montage(channels, _config(montage="easycap-M1"))
    expected = mne.channels.make_standard_montage("easycap-M1").get_positions()["ch_pos"]

    assert montage.name == "easycap-M1"
    assert montage.channels == channels
    assert np.allclose(montage.positions_3d, [expected[channel] for channel in channels])
    assert np.allclose(montage.positions_xy, np.asarray(montage.positions_3d)[:, :2])


@pytest.mark.parametrize("channels", [("Cz", "Cz"), (), ("Cz", "not-a-channel")])
def test_load_sensor_montage_rejects_invalid_analyzed_channels(
    channels: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="channels|montage"):
        load_sensor_montage(channels, _config())


@pytest.mark.parametrize(
    "positions",
    [
        {"Cz": [0.0, 0.0, 0.1], "Fz": [np.nan, 0.1, 0.1]},
        {"Cz": [0.0, 0.0, 0.1], "Fz": [0.0, 0.0, 0.1]},
        {"Cz": [0.0, 0.0, 0.1], "Fz": [0.0, 0.0, 0.2]},
    ],
)
def test_load_sensor_montage_rejects_invalid_or_duplicate_positions(
    positions: dict[str, list[float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_montage = mne.channels.make_dig_montage(ch_pos=positions, coord_frame="head")
    monkeypatch.setattr(mne.channels, "make_standard_montage", lambda _name: invalid_montage)

    with pytest.raises(ValueError, match="finite unique|top-view"):
        load_sensor_montage(("Cz", "Fz"), _config(montage="configured-montage"))


def _power_table(
    *,
    trial_ids: tuple[int, ...] = (1, 2),
    channels: tuple[str, ...] = CHANNELS,
) -> pd.DataFrame:
    frame = pd.DataFrame({"trial_id": trial_ids})
    for band_index, band in enumerate(BANDS):
        for channel_index, channel in enumerate(channels):
            frame[f"power_baseline_{band}_ch_{channel}_mean"] = 2.0 + band_index + channel_index
            frame[f"power_active_{band}_ch_{channel}_logratio"] = (
                0.01 * (band_index + 1) + 0.001 * channel_index
            )
    return frame


def _validity_data() -> ValidityTrialData:
    target_rows = []
    event_rows = []
    for subject_index, subject_id in enumerate(("sub-01", "sub-02"), start=1):
        for trial_id in (1, 2):
            target_rows.append(
                {
                    "subject_id": subject_id,
                    "run": 1,
                    "within_run_trial": trial_id,
                    "stimulus_temp": 44.0 + trial_id,
                    "selected_surface": trial_id,
                    "within_scale_intensity": 20.0 + trial_id,
                    "NPS": subject_index + trial_id / 10.0,
                    "SIIPS1": subject_index + trial_id / 5.0,
                    "nuisance_value": subject_index * 10.0 + trial_id,
                }
            )
            event_rows.append(
                {
                    "subject_id": subject_id,
                    "trial_id": trial_id,
                    "trial_number": trial_id,
                    "run": 1,
                    "stimulus_temp": 44.0 + trial_id,
                    "selected_surface": trial_id,
                    "residual_ecg_coupling": trial_id / 10.0,
                    "fp1_fp2_high_frequency_power": trial_id / 20.0,
                }
            )
    targets = pd.DataFrame(target_rows)
    return ValidityTrialData(
        targets=targets.copy(),
        clean_events=pd.DataFrame(event_rows),
        enriched_targets=targets,
    )


def _config(
    *,
    nuisance_columns: tuple[str, ...] = (),
    montage: str = "easycap-M1",
) -> ConfigDict:
    return ConfigDict(
        {
            "preprocessing": {"montage": montage},
            "study1": {
                "figures": {
                    "sensor_topographies": {
                        "bands": [{"name": band} for band in BANDS],
                    }
                },
                "targets": {
                    "nuisance_regression": {
                        "enabled": bool(nuisance_columns),
                        "continuous_columns": list(nuisance_columns),
                        "categorical_columns": [],
                    }
                },
            },
        }
    )
