from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_compute_band_contribution_scores_decomposes_linear_predictor_by_band() -> None:
    from eeg_pipeline.domain.features.naming import NamingSchema
    from studies.pain_study.study2.contributions import compute_band_contribution_scores

    feature_names = [
        NamingSchema.build("power", "active", "alpha", "ch", "logratio_mean", channel="Cz"),
        NamingSchema.build("power", "active", "beta", "ch", "logratio_mean", channel="Cz"),
        NamingSchema.build("power", "active", "alpha", "ch", "logratio_mean", channel="Pz"),
    ]
    X = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=float,
    )
    coefficients = np.asarray([0.5, 2.0, -1.0], dtype=float)

    contributions = compute_band_contribution_scores(
        X=X,
        feature_names=feature_names,
        coefficients=coefficients,
        bands=("alpha", "beta"),
        band_members={"alpha": ("alpha",), "beta": ("beta",)},
        subject_ids=("sub-0001", "sub-0001"),
        trial_ids=(1, 2),
    )

    assert contributions.columns.tolist() == [
        "subject_id",
        "trial_id",
        "eta_combined",
        "eta_alpha",
        "eta_beta",
    ]
    np.testing.assert_allclose(contributions["eta_alpha"], [-2.5, -4.0])
    np.testing.assert_allclose(contributions["eta_beta"], [4.0, 10.0])
    np.testing.assert_allclose(contributions["eta_combined"], [1.5, 6.0])


def test_compute_band_contribution_scores_requires_each_requested_band() -> None:
    from eeg_pipeline.domain.features.naming import NamingSchema
    from studies.pain_study.study2.contributions import compute_band_contribution_scores

    feature_names = [
        NamingSchema.build("power", "active", "alpha", "ch", "logratio_mean", channel="Cz"),
    ]

    with pytest.raises(ValueError, match="No features found for requested band 'beta'"):
        compute_band_contribution_scores(
            X=np.ones((2, 1), dtype=float),
            feature_names=feature_names,
            coefficients=np.ones(1, dtype=float),
            bands=("alpha", "beta"),
            band_members={"alpha": ("alpha",), "beta": ("beta",)},
        )


def test_compute_band_contribution_scores_aggregates_clean_gamma_subbands() -> None:
    from eeg_pipeline.domain.features.naming import NamingSchema
    from studies.pain_study.study2.contributions import compute_band_contribution_scores

    feature_names = [
        NamingSchema.build("power", "active", "alpha", "ch", "logratio_mean", channel="Cz"),
        NamingSchema.build(
            "power", "active", "gamma_low_clean", "ch", "logratio_mean", channel="Cz"
        ),
        NamingSchema.build(
            "power", "active", "gamma_high_clean", "ch", "logratio_mean", channel="Cz"
        ),
    ]
    X = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)

    contributions = compute_band_contribution_scores(
        X=X,
        feature_names=feature_names,
        coefficients=np.ones(3, dtype=float),
        bands=("alpha", "gamma"),
        band_members={
            "alpha": ("alpha",),
            "gamma": (
                "gamma_low_clean",
                "gamma_mid_clean",
                "gamma_high_clean",
            ),
        },
    )

    np.testing.assert_allclose(contributions["eta_alpha"], [1.0, 4.0])
    np.testing.assert_allclose(contributions["eta_gamma"], [5.0, 11.0])


def test_compute_held_out_contribution_scores_uses_frozen_fold_space(monkeypatch) -> None:
    from eeg_pipeline.domain.features.naming import NamingSchema
    from studies.pain_study.study2 import contributions as contribution_module

    feature_names = (
        NamingSchema.build("power", "active", "alpha", "ch", "logratio_mean", channel="Cz"),
        NamingSchema.build(
            "power", "active", "gamma_low_clean", "ch", "logratio_mean", channel="Cz"
        ),
    )
    fold_fit = SimpleNamespace(
        transformed_test=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        transformed_feature_names=feature_names,
        coefficients=np.asarray([2.0, 3.0], dtype=float),
        residual_prediction=np.asarray([10.0, 20.0], dtype=float),
    )
    context = SimpleNamespace(
        outer_folds=((np.asarray([], dtype=int), np.asarray([0, 1], dtype=int)),),
        groups=np.asarray(["sub-0001", "sub-0001"], dtype=object),
    )
    monkeypatch.setattr(
        contribution_module,
        "fit_frozen_study1_fold",
        lambda _context, _fold: fold_fit,
        raising=False,
    )

    scores = contribution_module.compute_held_out_contribution_scores(
        context,
        bands=("alpha", "gamma"),
        band_members={"alpha": ("alpha",), "gamma": ("gamma_low_clean",)},
    )

    assert scores["subject_id"].tolist() == ["sub-0001", "sub-0001"]
    np.testing.assert_allclose(scores["eta_combined"], [10.0, 20.0])
    np.testing.assert_allclose(scores["eta_alpha"], [2.0, 6.0])
    np.testing.assert_allclose(scores["eta_gamma"], [6.0, 12.0])
    # The band columns decompose the linear predictor, not the raw-scale score.
    np.testing.assert_allclose(scores["eta_bands_linear_predictor"], [8.0, 18.0])


def test_compute_band_contribution_scores_rejects_invalid_feature_names() -> None:
    from studies.pain_study.study2.contributions import compute_band_contribution_scores

    with pytest.raises(ValueError, match="Cannot parse feature band"):
        compute_band_contribution_scores(
            X=np.ones((2, 1), dtype=float),
            feature_names=["unstructured_feature"],
            coefficients=np.ones(1, dtype=float),
            bands=("alpha",),
            band_members={"alpha": ("alpha",)},
        )


def test_standardize_contribution_scores_within_subject_and_excludes_zero_variance() -> None:
    from studies.pain_study.study2.contributions import standardize_contribution_scores

    frame = pd.DataFrame(
        {
            "subject_id": ["sub-0001"] * 3 + ["sub-0002"] * 3,
            "trial_id": [1, 2, 3, 1, 2, 3],
            "eta_alpha": [1.0, 2.0, 3.0, 4.0, 4.0, 4.0],
            "eta_beta": [2.0, 4.0, 6.0, 1.0, 2.0, 3.0],
        }
    )

    standardized, qc = standardize_contribution_scores(
        frame,
        subject_column="subject_id",
        score_columns=("eta_alpha", "eta_beta"),
    )

    assert standardized["subject_id"].unique().tolist() == ["sub-0001"]
    assert "eta_alpha_z" in standardized.columns
    assert "eta_beta_z" in standardized.columns
    assert np.mean(standardized["eta_alpha_z"]) == pytest.approx(0.0)
    assert np.std(standardized["eta_alpha_z"], ddof=0) == pytest.approx(1.0)
    assert np.mean(standardized["eta_beta_z"]) == pytest.approx(0.0)
    assert np.std(standardized["eta_beta_z"], ddof=0) == pytest.approx(1.0)
    assert qc.to_dict("records") == [
        {
            "subject_id": "sub-0001",
            "contribution_criteria_met": True,
            "unmet_criteria": "",
        },
        {
            "subject_id": "sub-0002",
            "contribution_criteria_met": False,
            "unmet_criteria": "zero_variance_score:eta_alpha",
        },
    ]


def test_standardize_contribution_scores_requires_finite_values() -> None:
    from studies.pain_study.study2.contributions import standardize_contribution_scores

    frame = pd.DataFrame(
        {
            "subject_id": ["sub-0001", "sub-0001"],
            "eta_alpha": [1.0, np.nan],
            "eta_beta": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="non-finite"):
        standardize_contribution_scores(
            frame,
            subject_column="subject_id",
            score_columns=("eta_alpha", "eta_beta"),
        )
