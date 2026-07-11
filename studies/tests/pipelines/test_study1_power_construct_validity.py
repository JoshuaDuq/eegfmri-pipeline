from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from studies.pain_study.study1.config.loader import load_study1_config

BANDS = (
    "alpha",
    "beta",
    "gamma_low_clean",
    "gamma_mid_clean",
    "gamma_high_clean",
)
TEMPERATURES = (44.3, 45.3, 46.3, 47.3, 48.3, 49.3)


def test_reconstruct_global_power_averages_linear_power_before_log() -> None:
    from studies.pain_study.study1.figures.power_construct_data import (
        reconstruct_global_power,
    )

    table = power_feature_table(n_trials=3, bands=("alpha",), channels=("Fp1", "Cz"))
    result = reconstruct_global_power(
        table,
        subject_id="sub-0001",
        bands=("alpha",),
        include_fp1_fp2=True,
        channel_scope="primary",
    )

    baseline = table[
        ["power_baseline_alpha_ch_Fp1_mean", "power_baseline_alpha_ch_Cz_mean"]
    ].to_numpy()
    logratio = table[
        ["power_active_alpha_ch_Fp1_logratio", "power_active_alpha_ch_Cz_logratio"]
    ].to_numpy()
    active = baseline * np.power(10.0, logratio)
    expected = 10.0 * np.log10(active.mean(axis=1) / baseline.mean(axis=1))
    assert np.allclose(result["global_power_db"], expected)
    assert set(result["included_channels"]) == {"Cz,Fp1"}


def test_reconstruct_global_power_fp1_fp2_toggle_changes_scope() -> None:
    from studies.pain_study.study1.figures.power_construct_data import (
        reconstruct_global_power,
    )

    table = power_feature_table(
        n_trials=3,
        bands=("alpha",),
        channels=("Fp1", "Fp2", "Cz"),
    )
    included = reconstruct_global_power(
        table,
        subject_id="sub-0001",
        bands=("alpha",),
        include_fp1_fp2=True,
        channel_scope="primary",
    )
    excluded = reconstruct_global_power(
        table,
        subject_id="sub-0001",
        bands=("alpha",),
        include_fp1_fp2=False,
        channel_scope="complementary",
    )

    assert included["n_channels"].unique().tolist() == [3]
    assert excluded["n_channels"].unique().tolist() == [1]
    assert excluded["included_channels"].unique().tolist() == ["Cz"]
    assert not np.allclose(included["global_power_db"], excluded["global_power_db"])


def test_reconstruct_global_power_rejects_missing_clean_gamma() -> None:
    from studies.pain_study.study1.figures.power_construct_data import (
        reconstruct_global_power,
    )

    table = power_feature_table(n_trials=3, bands=("alpha",), channels=("Cz", "Pz"))
    with pytest.raises(ValueError, match="missing configured band"):
        reconstruct_global_power(
            table,
            subject_id="sub-0001",
            bands=BANDS,
            include_fp1_fp2=True,
            channel_scope="primary",
        )


def test_build_temperature_association_centers_each_participant_band() -> None:
    from studies.pain_study.study1.figures.power_construct_models import (
        build_temperature_association,
    )

    trials = association_trials(n_subjects=4)
    by_subject, cohort = build_temperature_association(
        trials,
        bands=BANDS,
        temperatures=TEMPERATURES,
        config=load_study1_config(),
    )

    centered_means = by_subject.groupby(["subject_id", "band"])["centered_power_db"].mean()
    assert np.allclose(centered_means, 0.0)
    assert len(by_subject) == 4 * len(BANDS) * len(TEMPERATURES)
    assert len(cohort) == len(BANDS) * len(TEMPERATURES)
    assert cohort["n_subjects"].eq(4).all()
    assert (cohort["ci_low"] <= cohort["mean"]).all()
    assert (cohort["mean"] <= cohort["ci_high"]).all()


def test_build_rating_association_recovers_partial_within_subject_signal() -> None:
    from studies.pain_study.study1.figures.power_construct_models import (
        build_rating_association,
    )

    participants, cohort = build_rating_association(
        association_trials(n_subjects=4),
        bands=BANDS,
        config=load_study1_config(),
    )

    assert participants["estimable"].all()
    assert participants["partial_r"].gt(0.75).all()
    assert cohort["mean_partial_r"].gt(0.75).all()
    assert set(cohort["band"]) == set(BANDS)


def test_build_rating_association_retains_nonestimability_reason() -> None:
    from studies.pain_study.study1.figures.power_construct_models import (
        build_rating_association,
    )

    config = load_study1_config()
    config["study1"]["figures"]["power_construct_validity"]["rating_model"]["minimum_trials"] = 100
    participants, _cohort = build_rating_association(
        association_trials(n_subjects=4),
        bands=BANDS,
        config=config,
    )

    assert not participants["estimable"].any()
    assert set(participants["non_estimability_reason"]) == {"too_few_trials"}


def test_build_power_construct_summary_computes_primary_and_complementary_scopes() -> None:
    from studies.pain_study.study1.figures.power_construct_validity import (
        build_power_construct_validity_summary,
    )
    from studies.pain_study.study1.figures.validity_data import ValidityTrialData

    enriched_rows = []
    event_rows = []
    feature_tables = {}
    for subject_index in range(4):
        subject_id = f"sub-{subject_index:04d}"
        feature_tables[subject_id] = power_feature_table(
            n_trials=36,
            bands=BANDS,
            channels=("Fp1", "Fp2", "Cz"),
        )
        for trial_index in range(36):
            run = trial_index // 12 + 1
            within_run_trial = trial_index % 12 + 1
            temperature = TEMPERATURES[(trial_index * 5 + subject_index) % 6]
            surface = (trial_index * 2 + subject_index) % 5 + 1
            within_scale = 35.0 + 5.0 * (temperature - 44.3) + (trial_index % 4)
            enriched_rows.append(
                {
                    "subject_id": subject_id,
                    "run": run,
                    "within_run_trial": within_run_trial,
                    "stimulus_temp": temperature,
                    "selected_surface": surface,
                    "within_scale_intensity": within_scale,
                }
            )
            event_rows.append(
                {
                    "subject_id": subject_id,
                    "trial_id": trial_index + 1,
                    "run": run,
                    "trial_number": within_run_trial,
                    "stimulus_temp": temperature,
                    "selected_surface": surface,
                    "residual_ecg_coupling": np.sin(trial_index + subject_index),
                    "fp1_fp2_high_frequency_power": np.cos(0.7 * trial_index),
                }
            )
    enriched = pd.DataFrame(enriched_rows)
    validity = ValidityTrialData(
        targets=enriched.copy(),
        clean_events=pd.DataFrame(event_rows),
        enriched_targets=enriched,
    )

    summary = build_power_construct_validity_summary(
        validity=validity,
        feature_tables=feature_tables,
        config=load_study1_config(),
    )

    assert len(summary.trials) == 4 * 36 * len(BANDS) * 2
    assert set(summary.trials["channel_scope"]) == {"primary", "complementary"}
    assert summary.primary_include_fp1_fp2 is True
    assert summary.n_subjects == 4
    assert summary.article_ready is False
    assert (
        summary.trials.loc[summary.trials["channel_scope"].eq("primary"), "n_channels"].eq(3).all()
    )
    assert (
        summary.trials.loc[summary.trials["channel_scope"].eq("complementary"), "n_channels"]
        .eq(1)
        .all()
    )
    assert not summary.sensitivity_by_subject.empty
    assert not summary.sensitivity_summary.empty

    excluded_config = load_study1_config()
    excluded_config["study1"]["figures"]["power_construct_validity"]["channels"][
        "include_fp1_fp2"
    ] = False
    excluded_summary = build_power_construct_validity_summary(
        validity=validity,
        feature_tables=feature_tables,
        config=excluded_config,
    )
    assert excluded_summary.primary_include_fp1_fp2 is False
    assert (
        excluded_summary.trials.loc[
            excluded_summary.trials["channel_scope"].eq("primary"), "n_channels"
        ]
        .eq(1)
        .all()
    )


def power_feature_table(
    *,
    n_trials: int,
    bands: tuple[str, ...],
    channels: tuple[str, ...],
) -> pd.DataFrame:
    frame = pd.DataFrame({"trial_id": np.arange(1, n_trials + 1)})
    for band_index, band in enumerate(bands):
        for channel_index, channel in enumerate(channels):
            frame[f"power_baseline_{band}_ch_{channel}_mean"] = (
                2.0 + 0.2 * channel_index + np.linspace(0.0, 0.1, n_trials)
            )
            logratio = 0.03 * (band_index + 1) + 0.02 * np.arange(n_trials)
            if channel in {"Fp1", "Fp2"}:
                logratio = logratio + 0.30
            frame[f"power_active_{band}_ch_{channel}_logratio"] = logratio
    return frame


def association_trials(*, n_subjects: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for subject_index in range(n_subjects):
        subject = f"sub-{subject_index:04d}"
        for trial_index in range(36):
            temperature = TEMPERATURES[(trial_index * 5 + subject_index) % len(TEMPERATURES)]
            run = trial_index // 12 + 1
            within_run_trial = trial_index % 12 + 1
            rating_residual = rng.normal()
            within_scale = 50.0 + 4.0 * (temperature - 46.8) + 8.0 * rating_residual
            for band_index, band in enumerate(BANDS):
                power = (
                    0.20 * (temperature - 46.8)
                    + 0.12 * run
                    + 0.7 * rating_residual
                    + rng.normal(scale=0.18)
                    + 0.04 * band_index
                    + 0.1 * subject_index
                )
                rows.append(
                    {
                        "subject_id": subject,
                        "trial_id": trial_index + 1,
                        "band": band,
                        "channel_scope": "primary",
                        "include_fp1_fp2": True,
                        "global_power_db": power,
                        "stimulus_temp": temperature,
                        "run": run,
                        "selected_surface": (trial_index * 2 + subject_index) % 5 + 1,
                        "within_run_trial": within_run_trial,
                        "residual_ecg_coupling": rng.normal(),
                        "fp1_fp2_high_frequency_power": rng.normal(),
                        "within_scale_intensity": within_scale,
                    }
                )
    return pd.DataFrame(rows)
