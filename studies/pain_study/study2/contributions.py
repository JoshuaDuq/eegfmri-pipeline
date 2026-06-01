"""Study 2 contribution-score preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema


def compute_band_contribution_scores(
    *,
    X: np.ndarray,
    feature_names: list[str],
    coefficients: np.ndarray,
    bands: tuple[str, ...],
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
    for band in bands:
        band_name = str(band).strip()
        if not band_name:
            raise ValueError("Requested contribution bands must be non-empty.")
        band_mask = np.asarray([feature_band == band_name for feature_band in feature_bands])
        if not np.any(band_mask):
            raise ValueError(f"No features found for requested band '{band_name}'.")
        output[f"eta_{band_name}"] = X_arr[:, band_mask] @ coefficient_arr[band_mask]
    return output.reset_index(drop=True)


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
        ineligible_reason = _ineligible_subject_reason(
            subject_copy,
            score_columns=score_columns,
        )
        if ineligible_reason:
            qc_records.append(
                {
                    "subject_id": str(subject_id),
                    "eligible": False,
                    "reason": ineligible_reason,
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
                "eligible": True,
                "reason": "",
            }
        )

    if standardized_subjects:
        standardized = pd.concat(standardized_subjects, axis=0, ignore_index=True)
    else:
        standardized = frame.iloc[0:0].copy()
        for column in score_columns:
            standardized[f"{column}_z"] = pd.Series(dtype=float)

    qc = pd.DataFrame(qc_records, columns=["subject_id", "eligible", "reason"])
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


def _ineligible_subject_reason(
    frame: pd.DataFrame,
    *,
    score_columns: tuple[str, ...],
) -> str:
    for column in score_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Study 2 contribution score '{column}' contains non-finite values.")
        if float(np.std(values, ddof=0)) <= 0.0:
            return f"Zero-variance contribution score: {column}."
    return ""


__all__ = [
    "compute_band_contribution_scores",
    "standardize_contribution_scores",
]
