"""Study 2 contribution-score preparation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from studies.pain_study.study2.sensor_patterns import fit_frozen_study1_fold

CONTRIBUTION_BAND_MEMBERS = {
    "alpha": ("alpha",),
    "beta": ("beta",),
    "gamma": (
        "gamma_low_clean",
        "gamma_mid_clean",
        "gamma_high_clean",
    ),
}


def contribution_band_members(
    bands: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    unknown = sorted(set(bands) - set(CONTRIBUTION_BAND_MEMBERS))
    if unknown:
        raise ValueError(f"Study 2 contribution bands have no feature mapping: {unknown}.")
    return {band: CONTRIBUTION_BAND_MEMBERS[band] for band in bands}


def compute_band_contribution_scores(
    *,
    X: np.ndarray,
    feature_names: list[str],
    coefficients: np.ndarray,
    bands: tuple[str, ...],
    band_members: Mapping[str, tuple[str, ...]],
    subject_ids: tuple[str, ...] | list[str] | np.ndarray | None = None,
    trial_ids: tuple[int, ...] | list[int] | np.ndarray | None = None,
    combined_column: str = "eta_combined",
) -> pd.DataFrame:
    """Compute the combined NPS-predictive score and its band-specific decomposition.

    The combined score is the full frozen linear predictor (the primary Study 2
    score). The band columns decompose that predictor and support the secondary
    mutually-adjusted contribution analysis.
    """
    X_arr = np.asarray(X, dtype=float)
    coefficient_arr = np.asarray(coefficients, dtype=float)
    _validate_linear_contribution_inputs(
        X=X_arr,
        feature_names=feature_names,
        coefficients=coefficient_arr,
        bands=bands,
    )

    output = pd.DataFrame(index=np.arange(X_arr.shape[0]))
    if subject_ids is not None:
        subject_arr = np.asarray(subject_ids, dtype=object)
        if len(subject_arr) != X_arr.shape[0]:
            raise ValueError("subject_ids must have one value per row of X.")
        output["subject_id"] = subject_arr.astype(str)
    if trial_ids is not None:
        trial_arr = np.asarray(trial_ids)
        if len(trial_arr) != X_arr.shape[0]:
            raise ValueError("trial_ids must have one value per row of X.")
        output["trial_id"] = trial_arr

    output[combined_column] = X_arr @ coefficient_arr
    feature_bands = tuple(_feature_band(feature_name) for feature_name in feature_names)
    if set(band_members) != set(bands):
        raise ValueError("Study 2 band_members must define every requested band exactly once.")
    for band in bands:
        band_name = str(band).strip()
        if not band_name:
            raise ValueError("Requested contribution bands must be non-empty.")
        members = tuple(str(member).strip() for member in band_members[band_name])
        if not members or any(not member for member in members):
            raise ValueError(f"Study 2 band_members for '{band_name}' must be non-empty.")
        band_mask = np.asarray([feature_band in members for feature_band in feature_bands])
        if not np.any(band_mask):
            raise ValueError(f"No features found for requested band '{band_name}'.")
        output[f"eta_{band_name}"] = X_arr[:, band_mask] @ coefficient_arr[band_mask]
    return output.reset_index(drop=True)


def compute_held_out_contribution_scores(
    context,
    *,
    bands: tuple[str, ...],
    band_members: Mapping[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Compute frozen-fold combined and band-specific scores in canonical row order."""
    fold_scores: list[pd.DataFrame] = []
    assigned_rows = np.zeros(len(context.groups), dtype=bool)
    for fold, (_train_indices, test_indices) in enumerate(context.outer_folds):
        test_rows = np.asarray(test_indices, dtype=int)
        if np.any(assigned_rows[test_rows]):
            raise ValueError("Study 2 outer folds assign a trial more than once.")
        fit = fit_frozen_study1_fold(context, fold)
        residual_prediction = np.asarray(fit.residual_prediction, dtype=float)
        if residual_prediction.shape != (len(test_rows),):
            raise ValueError("Frozen-fold residual predictions do not match held-out rows.")
        scores = compute_band_contribution_scores(
            X=np.asarray(fit.transformed_test, dtype=float),
            feature_names=list(fit.transformed_feature_names),
            coefficients=np.asarray(fit.coefficients, dtype=float),
            bands=bands,
            band_members=band_members,
            subject_ids=np.asarray(context.groups, dtype=object)[test_rows],
            trial_ids=test_rows,
        )
        scores["eta_combined"] = residual_prediction
        fold_scores.append(scores)
        assigned_rows[test_rows] = True

    if not np.all(assigned_rows):
        missing_rows = np.flatnonzero(~assigned_rows).tolist()
        raise ValueError(f"Study 2 outer folds do not assign rows: {missing_rows}.")
    return pd.concat(fold_scores, ignore_index=True).sort_values("trial_id").reset_index(drop=True)


def standardize_contribution_scores(
    frame: pd.DataFrame,
    *,
    subject_column: str,
    score_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Center and scale frozen Study 1 contribution scores within subject."""
    _require_columns(frame, (subject_column, *score_columns))
    if not score_columns:
        raise ValueError("Study 2 contribution standardization requires at least one score column.")

    standardized_subjects: list[pd.DataFrame] = []
    qc_records: list[dict[str, object]] = []
    for subject_id, subject_frame in frame.groupby(subject_column, sort=True):
        subject_copy = subject_frame.copy()
        unmet_criteria = _unmet_subject_criteria(
            subject_copy,
            score_columns=score_columns,
        )
        if unmet_criteria:
            qc_records.append(
                {
                    "subject_id": str(subject_id),
                    "contribution_criteria_met": False,
                    "unmet_criteria": ";".join(unmet_criteria),
                }
            )
            continue

        for column in score_columns:
            values = pd.to_numeric(subject_copy[column], errors="raise").to_numpy(dtype=float)
            subject_copy[f"{column}_z"] = (values - float(np.mean(values))) / float(
                np.std(values, ddof=0)
            )
        standardized_subjects.append(subject_copy)
        qc_records.append(
            {
                "subject_id": str(subject_id),
                "contribution_criteria_met": True,
                "unmet_criteria": "",
            }
        )

    if standardized_subjects:
        standardized = pd.concat(standardized_subjects, axis=0, ignore_index=True)
    else:
        standardized = frame.iloc[0:0].copy()
        for column in score_columns:
            standardized[f"{column}_z"] = pd.Series(dtype=float)

    qc = pd.DataFrame(
        qc_records,
        columns=["subject_id", "contribution_criteria_met", "unmet_criteria"],
    )
    return standardized.reset_index(drop=True), qc


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Study 2 contribution table is missing columns: {missing}.")


def _validate_linear_contribution_inputs(
    *,
    X: np.ndarray,
    feature_names: list[str],
    coefficients: np.ndarray,
    bands: tuple[str, ...],
) -> None:
    if X.ndim != 2:
        raise ValueError(f"Study 2 contribution matrix must be 2D, got shape {X.shape}.")
    if coefficients.ndim != 1:
        raise ValueError(
            f"Study 2 contribution coefficients must be 1D, got shape {coefficients.shape}."
        )
    if X.shape[1] != len(feature_names):
        raise ValueError(
            "Study 2 contribution feature_names length must match X columns: "
            f"{len(feature_names)} != {X.shape[1]}."
        )
    if coefficients.shape[0] != X.shape[1]:
        raise ValueError(
            "Study 2 contribution coefficients length must match X columns: "
            f"{coefficients.shape[0]} != {X.shape[1]}."
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("Study 2 contribution matrix contains non-finite values.")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Study 2 contribution coefficients contain non-finite values.")
    if not bands:
        raise ValueError("Study 2 contribution decomposition requires at least one band.")


def _feature_band(feature_name: str) -> str:
    parsed = NamingSchema.parse(str(feature_name))
    if not parsed.get("valid", False) or not parsed.get("band"):
        raise ValueError(f"Cannot parse feature band from Study 2 feature name: {feature_name!r}.")
    return str(parsed["band"])


def _unmet_subject_criteria(
    frame: pd.DataFrame,
    *,
    score_columns: tuple[str, ...],
) -> tuple[str, ...]:
    for column in score_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Study 2 contribution score '{column}' contains non-finite values.")
        if float(np.std(values, ddof=0)) <= 0.0:
            return (f"zero_variance_score:{column}",)
    return ()


__all__ = [
    "contribution_band_members",
    "compute_band_contribution_scores",
    "compute_held_out_contribution_scores",
    "standardize_contribution_scores",
]
