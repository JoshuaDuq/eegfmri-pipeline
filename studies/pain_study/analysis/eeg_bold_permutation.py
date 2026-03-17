"""Permutation-based sensitivity for the primary EEG-BOLD fixed effect."""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from studies.pain_study.analysis.eeg_bold_sensitivity import (
    PrimaryPermutationSensitivityConfig,
)


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p_arr = np.asarray(p_values, dtype=float)
    out = np.full_like(p_arr, np.nan, dtype=float)
    valid_mask = np.isfinite(p_arr)
    if not np.any(valid_mask):
        return out
    valid = p_arr[valid_mask]
    order = np.argsort(valid)
    sorted_p = valid[order]
    adjusted = np.empty_like(sorted_p)
    running_max = 0.0
    total = sorted_p.size
    for idx, p_value in enumerate(sorted_p):
        candidate = float((total - idx) * p_value)
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[valid_mask] = restored
    return out


def _as_float_array(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _standardize(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("Standardization expects a 1D array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Standardization expects finite values.")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Standardization requires non-zero variance.")
    return (values - mean) / std


def _design_matrix(
    *,
    table: pd.DataFrame,
    numeric_terms: Sequence[str],
    factor_terms: Sequence[str],
    include_run_fixed_effect: bool,
) -> np.ndarray:
    parts: List[np.ndarray] = [np.ones((len(table), 1), dtype=float)]
    for term in numeric_terms:
        if term not in table.columns:
            raise ValueError(f"Numeric nuisance term {term!r} is missing.")
        values = _standardize(_as_float_array(table[term]))
        parts.append(values.reshape(-1, 1))
    effective_factors = [str(term) for term in factor_terms]
    if include_run_fixed_effect:
        effective_factors.append("run_num")
    for term in effective_factors:
        if term not in table.columns:
            raise ValueError(f"Categorical nuisance term {term!r} is missing.")
        dummies = pd.get_dummies(table[term].astype(str), prefix=term, drop_first=True)
        if dummies.empty:
            continue
        parts.append(dummies.to_numpy(dtype=float))
    return np.concatenate(parts, axis=1)


def _predictor_beta(
    *,
    table: pd.DataFrame,
    predictor_column: str,
    outcome_column: str,
    numeric_terms: Sequence[str],
    factor_terms: Sequence[str],
    include_run_fixed_effect: bool,
) -> float:
    predictor = _standardize(_as_float_array(table[predictor_column]))
    outcome = _standardize(_as_float_array(table[outcome_column]))
    nuisance = _design_matrix(
        table=table,
        numeric_terms=numeric_terms,
        factor_terms=factor_terms,
        include_run_fixed_effect=include_run_fixed_effect,
    )
    design = np.concatenate([nuisance, predictor.reshape(-1, 1)], axis=1)
    coefficients, *_rest = np.linalg.lstsq(design, outcome, rcond=None)
    beta = float(coefficients[-1])
    if not np.isfinite(beta):
        raise ValueError("Subject-level predictor beta is not finite.")
    return beta


def _subject_status_map(analysis_cells: pd.DataFrame) -> Dict[str, str]:
    if analysis_cells.empty:
        return {}
    return {
        str(row.analysis_id): str(row.status)
        for row in analysis_cells.itertuples(index=False)
    }


def _subject_count_map(analysis_cells: pd.DataFrame) -> Dict[str, tuple[int, int]]:
    if analysis_cells.empty:
        return {}
    return {
        str(row.analysis_id): (int(row.n_trials), int(row.n_runs))
        for row in analysis_cells.itertuples(index=False)
    }


def _numeric_nuisance_terms(
    *,
    model_terms: Sequence[str],
    categorical_terms: Sequence[str],
) -> tuple[str, ...]:
    categorical = {str(term) for term in categorical_terms}
    return tuple(str(term) for term in model_terms if str(term) not in categorical)


def _analysis_row(
    *,
    subject: str,
    cell: Any,
    status: str,
    n_trials: int,
    n_runs: int,
    beta: float = np.nan,
    message: str = "",
) -> Dict[str, Any]:
    return {
        "subject": str(subject),
        "analysis_id": str(cell.analysis_id),
        "family": str(cell.family),
        "roi": str(cell.roi),
        "band": str(cell.band),
        "predictor_column": str(cell.predictor_column),
        "outcome_column": str(cell.outcome_column),
        "n_trials": int(n_trials),
        "n_runs": int(n_runs),
        "beta": float(beta),
        "status": str(status),
        "interpretable": str(status) == "ok",
        "message": str(message).strip(),
    }


def fit_subject_primary_permutation_effects(
    *,
    subject: str,
    merged_table: pd.DataFrame,
    cell_specs: Sequence[Any],
    analysis_cells: pd.DataFrame,
    include_run_fixed_effect: bool,
) -> pd.DataFrame:
    status_by_analysis = _subject_status_map(analysis_cells)
    counts_by_analysis = _subject_count_map(analysis_cells)
    rows: List[Dict[str, Any]] = []
    for cell in cell_specs:
        analysis_id = str(cell.analysis_id)
        status = status_by_analysis.get(analysis_id, "missing_analysis_cell")
        n_trials, n_runs = counts_by_analysis.get(analysis_id, (0, 0))
        if status != "ok":
            rows.append(
                _analysis_row(
                    subject=subject,
                    cell=cell,
                    status=status,
                    n_trials=n_trials,
                    n_runs=n_runs,
                )
            )
            continue
        required_columns = [
            cell.predictor_column,
            cell.outcome_column,
            *cell.model_terms,
            "run_num",
        ]
        subset = merged_table[required_columns].dropna().reset_index(drop=True)
        if subset.empty:
            rows.append(
                _analysis_row(
                    subject=subject,
                    cell=cell,
                    status="no_valid_rows",
                    n_trials=0,
                    n_runs=0,
                )
            )
            continue
        try:
            beta = _predictor_beta(
                table=subset,
                predictor_column=cell.predictor_column,
                outcome_column=cell.outcome_column,
                numeric_terms=_numeric_nuisance_terms(
                    model_terms=cell.model_terms,
                    categorical_terms=cell.categorical_terms,
                ),
                factor_terms=cell.categorical_terms,
                include_run_fixed_effect=include_run_fixed_effect,
            )
        except ValueError as exc:
            rows.append(
                _analysis_row(
                    subject=subject,
                    cell=cell,
                    status="insufficient_variance",
                    n_trials=int(len(subset)),
                    n_runs=int(subset["run_num"].nunique()),
                    message=str(exc),
                )
            )
            continue
        rows.append(
            _analysis_row(
                subject=subject,
                cell=cell,
                status="ok",
                n_trials=int(len(subset)),
                n_runs=int(subset["run_num"].nunique()),
                beta=beta,
            )
        )
    return pd.DataFrame(rows)


def _bootstrap_mean(
    *,
    betas: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_subjects = betas.size
    bootstrap = np.empty(iterations, dtype=float)
    for idx in range(iterations):
        sampled = rng.integers(0, n_subjects, size=n_subjects)
        bootstrap[idx] = float(np.mean(betas[sampled]))
    return bootstrap


def _exact_sign_matrix(n_subjects: int) -> np.ndarray:
    rows = list(itertools.product((-1.0, 1.0), repeat=n_subjects))
    return np.asarray(rows, dtype=float)


def _permutation_p_value(
    *,
    observed: float,
    betas: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> float:
    n_subjects = betas.size
    if n_subjects <= 12:
        sign_matrix = _exact_sign_matrix(n_subjects)
    else:
        sign_matrix = rng.choice(
            (-1.0, 1.0),
            size=(iterations, n_subjects),
            replace=True,
        )
    statistics = np.mean(sign_matrix * betas.reshape(1, -1), axis=1)
    extreme = float(np.sum(np.abs(statistics) >= abs(observed)))
    return float((extreme + 1.0) / (statistics.size + 1.0))


def aggregate_primary_permutation_effects(
    *,
    subject_effects: pd.DataFrame,
    cfg: PrimaryPermutationSensitivityConfig,
    alpha: float,
    random_state: int,
) -> pd.DataFrame:
    if subject_effects.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(int(random_state))
    rows: List[Dict[str, Any]] = []
    grouped = subject_effects.drop_duplicates(
        subset=[
            "analysis_id",
            "family",
            "roi",
            "band",
            "predictor_column",
            "outcome_column",
        ]
    )
    for cell in grouped.itertuples(index=False):
        subset = subject_effects.loc[
            (subject_effects["analysis_id"].astype(str) == str(cell.analysis_id))
            & (subject_effects["status"].astype(str) == "ok")
        ].reset_index(drop=True)
        if len(subset) < 2:
            rows.append(
                {
                    "analysis_id": str(cell.analysis_id),
                    "family": str(cell.family),
                    "roi": str(cell.roi),
                    "band": str(cell.band),
                    "predictor_column": str(cell.predictor_column),
                    "outcome_column": str(cell.outcome_column),
                    "n_subjects": int(len(subset)),
                    "n_trials": int(pd.to_numeric(subset["n_trials"], errors="coerce").sum()),
                    "beta": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "status": "insufficient_subjects",
                    "interpretable": False,
                    "message": "Permutation sensitivity requires at least two subjects.",
                }
            )
            continue
        betas = pd.to_numeric(subset["beta"], errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(betas)
        if int(np.sum(valid_mask)) < 2:
            rows.append(
                {
                    "analysis_id": str(cell.analysis_id),
                    "family": str(cell.family),
                    "roi": str(cell.roi),
                    "band": str(cell.band),
                    "predictor_column": str(cell.predictor_column),
                    "outcome_column": str(cell.outcome_column),
                    "n_subjects": int(np.sum(valid_mask)),
                    "n_trials": int(pd.to_numeric(subset["n_trials"], errors="coerce").sum()),
                    "beta": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "status": "insufficient_subjects",
                    "interpretable": False,
                    "message": "Permutation sensitivity retained fewer than two valid subjects.",
                }
            )
            continue
        betas = betas[valid_mask]
        observed = float(np.mean(betas))
        bootstrap = _bootstrap_mean(
            betas=betas,
            iterations=cfg.bootstrap_iterations,
            rng=rng,
        )
        ci_low = float(np.quantile(bootstrap, 0.025))
        ci_high = float(np.quantile(bootstrap, 0.975))
        p_value = _permutation_p_value(
            observed=observed,
            betas=betas,
            iterations=cfg.permutation_iterations,
            rng=rng,
        )
        rows.append(
            {
                "analysis_id": str(cell.analysis_id),
                "family": str(cell.family),
                "roi": str(cell.roi),
                "band": str(cell.band),
                "predictor_column": str(cell.predictor_column),
                "outcome_column": str(cell.outcome_column),
                "n_subjects": int(betas.size),
                "n_trials": int(pd.to_numeric(subset["n_trials"], errors="coerce").sum()),
                "beta": observed,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
                "status": "ok",
                "interpretable": True,
                "message": "",
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["family", "roi", "band", "analysis_id"]
    ).reset_index(drop=True)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    valid_mask = (
        out["interpretable"].astype(bool)
        & np.isfinite(pd.to_numeric(out["p_value"], errors="coerce").to_numpy(dtype=float))
    )
    if bool(valid_mask.any()):
        out.loc[valid_mask, "p_holm"] = _holm_adjust(
            pd.to_numeric(out.loc[valid_mask, "p_value"], errors="coerce").to_numpy(dtype=float)
        )
    out["significant_holm"] = (
        out["interpretable"].astype(bool)
        & (pd.to_numeric(out["p_holm"], errors="coerce") < float(alpha))
    )
    return out


__all__ = [
    "aggregate_primary_permutation_effects",
    "fit_subject_primary_permutation_effects",
]
