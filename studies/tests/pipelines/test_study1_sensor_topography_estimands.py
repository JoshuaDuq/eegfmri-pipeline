from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from studies.pain_study.study1.config.loader import load_study1_config
from studies.pain_study.study1.figures.sensor_topography_data import SensorPowerData

BANDS = (
    "alpha",
    "beta",
    "gamma_low_clean",
    "gamma_mid_clean",
    "gamma_high_clean",
)
TEMPERATURES = (44.3, 45.3, 46.3, 47.3, 48.3, 49.3)
CHANNELS = ("Fp1", "Fp2", "Cz", "Pz")


def test_construct_effects_use_condition_means_and_existing_partial_design() -> None:
    from studies.pain_study.study1.figures.sensor_topography_estimands import (
        build_construct_effects,
    )

    data = construct_data(("sub-0002", "sub-0001"))
    result = build_construct_effects(data, construct_config())

    alpha = result.effects.loc[
        result.effects["band"].eq("alpha") & result.effects["subject_id"].eq("sub-0001")
    ]
    source = data.trials.loc[
        data.trials["band"].eq("alpha") & data.trials["subject_id"].eq("sub-0001")
    ]
    for channel in ("Cz", "Pz"):
        rows = source.loc[source["channel"].eq(channel)]
        means = rows.groupby("stimulus_temp", sort=True)["power_db"].mean()
        expected_slope = np.polyfit(means.index.to_numpy(), means.to_numpy(), 1)[0]
        observed = alpha.loc[
            alpha["estimand"].eq("temperature") & alpha["channel"].eq(channel),
            "effect_value",
        ].item()
        assert observed == pytest.approx(expected_slope)

    intensity = alpha.loc[alpha["estimand"].eq("intensity") & alpha["channel"].eq("Cz")].iloc[0]
    expected_r = construct_partial_r(source.loc[source["channel"].eq("Cz")])
    assert intensity["partial_r"] == pytest.approx(expected_r)
    assert intensity["fisher_z"] == pytest.approx(np.arctanh(expected_r))
    assert intensity["inference_value"] == pytest.approx(intensity["fisher_z"])


def test_construct_effects_preserve_scopes_order_and_fisher_display_contract() -> None:
    from studies.pain_study.study1.figures.sensor_topography_estimands import (
        build_construct_effects,
    )

    result = build_construct_effects(
        construct_data(("sub-0002", "sub-0001")),
        construct_config(),
    )

    assert result.map_order == tuple(
        (estimand, band) for estimand in ("temperature", "intensity") for band in BANDS
    )
    assert result.sensor_order == CHANNELS
    assert result.participant_order == ("sub-0001", "sub-0002")
    assert result.inference_value_column == "inference_value"
    tensor = result.inference_tensor()
    assert tensor.shape == (2, 10, 4)
    intensity = signature_cell(result.effects, "sub-0001", "intensity", "alpha", "Cz")
    assert tensor[0, 5, 2] == pytest.approx(intensity["fisher_z"])
    assert tensor[0, 5, 2] != pytest.approx(intensity["partial_r"])
    assert result.sensitivity_effects is not None
    assert result.sensitivity_summary is not None
    assert result.effects["include_fp1_fp2"].eq(True).all()
    assert result.sensitivity_effects["include_fp1_fp2"].eq(False).all()
    assert set(result.sensitivity_effects["channel"]) == {"Cz", "Pz"}

    row = result.summary.loc[
        result.summary["estimand"].eq("intensity")
        & result.summary["band"].eq("alpha")
        & result.summary["channel"].eq("Cz")
    ].iloc[0]
    participant_rows = result.effects.loc[
        result.effects["estimand"].eq("intensity")
        & result.effects["band"].eq("alpha")
        & result.effects["channel"].eq("Cz")
    ]
    expected_display = np.tanh(participant_rows["fisher_z"].mean())
    assert row["display_value"] == pytest.approx(expected_display)
    assert row["display_value"] != pytest.approx(participant_rows["partial_r"].mean())
    assert row["display_value"] != pytest.approx(participant_rows["fisher_z"].mean())

    flipped_config = construct_config()
    channels = flipped_config["study1"]["figures"]["power_construct_validity"]["channels"]
    channels["include_fp1_fp2"] = False
    flipped = build_construct_effects(construct_data(("sub-0001",)), flipped_config)
    assert flipped.sensor_order == ("Cz", "Pz")
    assert flipped.effects["include_fp1_fp2"].eq(False).all()
    assert flipped.sensitivity_effects["include_fp1_fp2"].eq(True).all()
    assert set(flipped.sensitivity_effects["channel"]) == set(CHANNELS)


def test_construct_missing_temperature_cell_excludes_whole_participant() -> None:
    from studies.pain_study.study1.figures.sensor_topography_estimands import (
        build_construct_effects,
    )

    data = construct_data(("sub-0001", "sub-0002"))
    trials = data.trials.loc[
        ~(
            data.trials["subject_id"].eq("sub-0002")
            & data.trials["band"].eq("alpha")
            & data.trials["channel"].eq("Cz")
            & data.trials["stimulus_temp"].eq(TEMPERATURES[0])
        )
    ].copy()
    result = build_construct_effects(replace_trials(data, trials), construct_config())

    assert result.participant_order == ("sub-0001",)
    assert set(result.effects["subject_id"]) == {"sub-0001"}
    exclusion = result.exclusions.iloc[0]
    assert exclusion.to_dict() == {
        "subject_id": "sub-0002",
        "reason": "missing_temperature_cell",
        "estimand": "temperature",
        "band": "alpha",
        "channel": "Cz",
    }


def test_construct_malformed_source_is_fatal() -> None:
    from studies.pain_study.study1.figures.sensor_topography_estimands import (
        build_construct_effects,
    )

    data = construct_data(("sub-0001",))
    malformed = data.trials.copy()
    malformed.loc[0, "power_db"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_construct_effects(replace_trials(data, malformed), construct_config())


def test_signature_effects_use_ordered_nuisance_design_and_excluded_channels() -> None:
    from studies.pain_study.study1.figures.sensor_topography_estimands import (
        build_signature_effects,
    )

    data = signature_data(("sub-0002", "sub-0001"))
    result = build_signature_effects(data, signature_config())

    assert result.map_order == tuple(
        (target, band) for target in ("NPS", "SIIPS1") for band in BANDS
    )
    assert result.sensor_order == ("Cz", "Pz")
    assert result.participant_order == ("sub-0001", "sub-0002")
    assert set(result.effects["channel"]) == {"Cz", "Pz"}
    assert result.sensitivity_effects is None
    assert result.sensitivity_summary is None

    nps = signature_cell(result.effects, "sub-0001", "NPS", "alpha", "Cz")
    expected_r = signature_partial_r(
        data.trials.loc[
            data.trials["subject_id"].eq("sub-0001")
            & data.trials["band"].eq("alpha")
            & data.trials["channel"].eq("Cz")
        ],
        target="NPS",
        nuisance_columns=("run", "nuisance"),
    )
    assert nps["partial_r"] == pytest.approx(expected_r)
    assert nps["inference_value"] == pytest.approx(np.arctanh(expected_r))
    assert nps["omitted_constant_nuisance_columns"] == "constant"
    assert nps["retained_nuisance_columns"] == "run,nuisance"
    assert nps["minimum_standardized_nuisance_norm"] == pytest.approx(1.0)
    assert nps["maximum_standardized_nuisance_norm"] == pytest.approx(1.0)

    siips1 = signature_cell(result.effects, "sub-0001", "SIIPS1", "alpha", "Cz")
    assert siips1["resolved_nuisance_columns"] == "run,nuisance,constant,NPS"
    assert siips1["retained_nuisance_columns"].endswith(",NPS")


def test_signature_summary_back_transforms_mean_fisher_z() -> None:
    from studies.pain_study.study1.figures.sensor_topography_estimands import (
        build_signature_effects,
    )

    result = build_signature_effects(
        signature_data(("sub-0001", "sub-0002")),
        signature_config(),
    )
    effects = result.effects.loc[
        result.effects["estimand"].eq("NPS")
        & result.effects["band"].eq("alpha")
        & result.effects["channel"].eq("Cz")
    ]
    summary = result.summary.loc[
        result.summary["estimand"].eq("NPS")
        & result.summary["band"].eq("alpha")
        & result.summary["channel"].eq("Cz")
    ].iloc[0]

    assert summary["display_value"] == pytest.approx(np.tanh(effects["fisher_z"].mean()))
    assert summary["display_value"] != pytest.approx(effects["partial_r"].mean())


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("rank", "rank_deficient_design"),
        ("condition", "excessive_condition_number"),
        ("residual_df", "nonpositive_residual_degrees_of_freedom"),
        ("zero_residual", "zero_residual_variance_power"),
    ),
)
def test_signature_design_failures_exclude_whole_participant(
    mutation: str,
    reason: str,
) -> None:
    from studies.pain_study.study1.figures.sensor_topography_estimands import (
        build_signature_effects,
    )

    data = signature_data(("sub-0001", "sub-0002"))
    config = signature_config()
    trials = data.trials.copy()
    failed = trials["subject_id"].eq("sub-0002")
    if mutation == "rank":
        trials.loc[failed, "near_duplicate"] = 2.0 * trials.loc[failed, "nuisance"]
        trials.loc[~failed, "near_duplicate"] = np.square(trials.loc[~failed, "nuisance"])
        config["study1"]["targets"]["nuisance_regression"]["continuous_columns"].insert(
            2, "near_duplicate"
        )
    elif mutation == "condition":
        trials.loc[failed, "near_duplicate"] = (
            trials.loc[failed, "nuisance"] + 1.0e-6 * trials.loc[failed, "trial_id"]
        )
        trials.loc[~failed, "near_duplicate"] = np.square(trials.loc[~failed, "nuisance"])
        config["study1"]["targets"]["nuisance_regression"]["continuous_columns"].insert(
            2, "near_duplicate"
        )
        model = config["study1"]["figures"]["sensor_topographies"]["signature_model"]
        model["rank_tolerance"] = 1.0e-14
    elif mutation == "residual_df":
        subject_rows = trials.loc[failed, "trial_id"].le(4)
        trials = trials.loc[~failed | subject_rows].copy()
        circular = config["study1"]["feature_benchmark"]["circular_shift"]
        circular["min_retained_trials_per_subject"] = 4
        circular["min_valid_runs_per_subject"] = 1
        for index in range(2):
            column = f"extra_{index}"
            trials[column] = np.power(trials["trial_id"], index + 2)
            config["study1"]["targets"]["nuisance_regression"]["continuous_columns"].insert(
                -1, column
            )
    else:
        rows = failed & trials["channel"].eq("Cz")
        trials.loc[rows, "power_db"] = trials.loc[rows, "nuisance"]

    result = build_signature_effects(replace_trials(data, trials), config)

    assert result.participant_order == ("sub-0001",)
    assert set(result.effects["subject_id"]) == {"sub-0001"}
    exclusion = result.exclusions.iloc[0]
    assert exclusion["subject_id"] == "sub-0002"
    assert exclusion["reason"] == reason
    assert (exclusion["estimand"], exclusion["band"], exclusion["channel"]) == (
        "NPS",
        "alpha",
        "Cz",
    )


def test_signature_exclusions_are_required_and_must_exist() -> None:
    from studies.pain_study.study1.figures.sensor_topography_estimands import (
        build_signature_effects,
    )

    data = signature_data(("sub-0001",))
    empty = signature_config()
    empty["study1"]["feature_benchmark"]["excluded_channels"] = []
    with pytest.raises(ValueError, match="non-empty"):
        build_signature_effects(data, empty)

    missing = signature_config()
    missing["study1"]["feature_benchmark"]["excluded_channels"] = ["Fp1", "Oz"]
    with pytest.raises(ValueError, match="absent"):
        build_signature_effects(data, missing)


def test_incomplete_signature_map_grid_excludes_whole_participant() -> None:
    from studies.pain_study.study1.figures.sensor_topography_estimands import (
        build_signature_effects,
    )

    data = signature_data(("sub-0001", "sub-0002"))
    incomplete = data.trials.loc[
        ~(
            data.trials["subject_id"].eq("sub-0002")
            & data.trials["band"].eq("alpha")
            & data.trials["channel"].eq("Cz")
        )
    ].copy()
    result = build_signature_effects(replace_trials(data, incomplete), signature_config())

    assert result.participant_order == ("sub-0001",)
    assert result.exclusions.iloc[0].to_dict() == {
        "subject_id": "sub-0002",
        "reason": "incomplete_effect_grid",
        "estimand": "NPS",
        "band": "alpha",
        "channel": "Cz",
    }


def test_complete_effect_frame_rejects_duplicate_and_missing_keys() -> None:
    from studies.pain_study.study1.figures.sensor_topography_estimands import (
        _complete_effect_frame,
    )

    duplicate = {
        "subject_id": "sub-0001",
        "estimand": "NPS",
        "band": "alpha",
        "channel": "Cz",
        "inference_value": 0.2,
    }
    with pytest.raises(ValueError, match="exact unique subject-map-sensor keys"):
        _complete_effect_frame(
            [duplicate, duplicate.copy()],
            ("sub-0001",),
            (("NPS", "alpha"), ("SIIPS1", "alpha")),
            ("Cz",),
        )


@pytest.mark.parametrize(
    ("family", "field", "value"),
    (
        ("signature", "min_retained_trials_per_subject", True),
        ("signature", "min_valid_runs_per_subject", 3.5),
        ("construct", "minimum_trials", True),
        ("construct", "minimum_runs", 3.5),
    ),
)
def test_sample_thresholds_require_positive_integers(
    family: str,
    field: str,
    value: object,
) -> None:
    from studies.pain_study.study1.figures.sensor_topography_estimands import (
        build_construct_effects,
        build_signature_effects,
    )

    if family == "signature":
        config = signature_config()
        config["study1"]["feature_benchmark"]["circular_shift"][field] = value
        with pytest.raises(ValueError, match="positive integer"):
            build_signature_effects(signature_data(("sub-0001",)), config)
        return

    config = construct_config()
    config["study1"]["figures"]["power_construct_validity"]["rating_model"][field] = value
    with pytest.raises(ValueError, match="positive integer"):
        build_construct_effects(construct_data(("sub-0001",)), config)


def test_construct_temperatures_must_be_finite() -> None:
    from studies.pain_study.study1.figures.sensor_topography_estimands import (
        build_construct_effects,
    )

    config = construct_config()
    config["study1"]["figures"]["validity"]["temperatures"][2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_construct_effects(construct_data(("sub-0001",)), config)


def construct_config() -> dict[str, object]:
    config = load_study1_config()
    rating = config["study1"]["figures"]["power_construct_validity"]["rating_model"]
    rating["minimum_trials"] = 12
    rating["minimum_runs"] = 3
    return config


def signature_config() -> dict[str, object]:
    config = load_study1_config()
    nuisance = config["study1"]["targets"]["nuisance_regression"]
    nuisance["continuous_columns"] = ["run", "nuisance", "constant"]
    nuisance["categorical_columns"] = []
    circular = config["study1"]["feature_benchmark"]["circular_shift"]
    circular["min_retained_trials_per_subject"] = 12
    circular["min_valid_runs_per_subject"] = 3
    return config


def construct_data(subjects: tuple[str, ...]) -> SensorPowerData:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260715)
    for subject_index, subject_id in enumerate(subjects):
        trial_id = 0
        subject_signal = 0.35 if subject_index == 0 else 1.1
        for temperature_index, temperature in enumerate(TEMPERATURES):
            for repetition in range(6):
                trial_id += 1
                run = repetition % 3 + 1
                latent = rng.normal()
                intensity = 45.0 + 3.0 * temperature_index + latent
                common = 0.2 * run + 0.04 * (temperature_index * 6 + repetition + 1)
                ecg = rng.normal()
                artifact = rng.normal()
                for band_index, band in enumerate(BANDS):
                    for channel_index, channel in enumerate(CHANNELS):
                        slope = 0.15 + 0.04 * channel_index
                        power = (
                            slope * temperature
                            + subject_signal * latent
                            + 0.05 * band_index
                            + common
                            + 0.11 * ecg
                            - 0.07 * artifact
                            + rng.normal(scale=0.08)
                        )
                        rows.append(
                            {
                                "subject_id": subject_id,
                                "trial_id": trial_id,
                                "band": band,
                                "channel": channel,
                                "power_db": power,
                                "stimulus_temp": temperature,
                                "run": run,
                                "selected_surface": (temperature_index + repetition) % 5 + 1,
                                "within_run_trial": temperature_index * 6 + repetition + 1,
                                "residual_ecg_coupling": ecg,
                                "fp1_fp2_high_frequency_power": artifact,
                                "within_scale_intensity": intensity,
                            }
                        )
    return SensorPowerData(
        trials=pd.DataFrame(rows),
        subjects=subjects,
        bands=BANDS,
        channels=CHANNELS,
    )


def signature_data(subjects: tuple[str, ...]) -> SensorPowerData:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(321)
    for subject_index, subject_id in enumerate(subjects):
        target_weight = 0.25 if subject_index == 0 else 1.4
        for trial_id in range(1, 19):
            run = (trial_id - 1) // 6 + 1
            nuisance = rng.normal()
            nps = 0.3 * nuisance + rng.normal()
            siips1 = 0.4 * nuisance + 0.5 * nps + rng.normal()
            for band_index, band in enumerate(BANDS):
                for channel_index, channel in enumerate(CHANNELS):
                    power = (
                        0.6 * nuisance
                        + target_weight * nps
                        + 0.15 * siips1
                        + 0.03 * band_index
                        + 0.02 * channel_index
                        + rng.normal(scale=0.35)
                    )
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "trial_id": trial_id,
                            "band": band,
                            "channel": channel,
                            "power_db": power,
                            "run": run,
                            "nuisance": nuisance,
                            "constant": 7.0,
                            "NPS": nps,
                            "SIIPS1": siips1,
                        }
                    )
    return SensorPowerData(
        trials=pd.DataFrame(rows),
        subjects=subjects,
        bands=BANDS,
        channels=CHANNELS,
    )


def construct_partial_r(rows: pd.DataFrame) -> float:
    temperature = pd.Categorical(rows["stimulus_temp"], categories=TEMPERATURES, ordered=True)
    columns = [np.ones(len(rows))]
    columns.extend(pd.get_dummies(temperature, drop_first=True, dtype=float).to_numpy().T)
    for column in ("run", "selected_surface"):
        categories = tuple(sorted(rows[column].unique()))
        categorical = pd.Categorical(rows[column], categories=categories, ordered=True)
        columns.extend(pd.get_dummies(categorical, drop_first=True, dtype=float).to_numpy().T)
    for column in (
        "within_run_trial",
        "residual_ecg_coupling",
        "fp1_fp2_high_frequency_power",
    ):
        values = rows[column].to_numpy(dtype=float)
        columns.append((values - values.mean()) / values.std(ddof=1))
    design = np.column_stack(columns)
    return residual_correlation(
        rows["power_db"].to_numpy(dtype=float),
        rows["within_scale_intensity"].to_numpy(dtype=float),
        design,
    )


def signature_partial_r(
    rows: pd.DataFrame,
    *,
    target: str,
    nuisance_columns: tuple[str, ...],
) -> float:
    columns = [np.ones(len(rows))]
    for column in nuisance_columns:
        values = rows[column].to_numpy(dtype=float)
        centered = values - values.mean()
        columns.append(centered / np.linalg.norm(centered))
    return residual_correlation(
        rows["power_db"].to_numpy(dtype=float),
        rows[target].to_numpy(dtype=float),
        np.column_stack(columns),
    )


def residual_correlation(left: np.ndarray, right: np.ndarray, design: np.ndarray) -> float:
    left_residual = left - design @ np.linalg.lstsq(design, left, rcond=None)[0]
    right_residual = right - design @ np.linalg.lstsq(design, right, rcond=None)[0]
    return float(np.corrcoef(left_residual, right_residual)[0, 1])


def signature_cell(
    effects: pd.DataFrame,
    subject_id: str,
    target: str,
    band: str,
    channel: str,
) -> pd.Series:
    return effects.loc[
        effects["subject_id"].eq(subject_id)
        & effects["estimand"].eq(target)
        & effects["band"].eq(band)
        & effects["channel"].eq(channel)
    ].iloc[0]


def replace_trials(data: SensorPowerData, trials: pd.DataFrame) -> SensorPowerData:
    return SensorPowerData(
        trials=trials,
        subjects=data.subjects,
        bands=data.bands,
        channels=data.channels,
    )
