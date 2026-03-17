"""Model-light residualized EEG-BOLD coupling sensitivity."""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from studies.pain_study.analysis.eeg_bold_sensitivity import (
    ResidualizedCorrelationSensitivityConfig,
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
    m = sorted_p.size
    adjusted = np.empty_like(sorted_p)
    running_max = 0.0
    for idx, p_val in enumerate(sorted_p):
        holm = float((m - idx) * p_val)
        running_max = max(running_max, holm)
        adjusted[idx] = min(running_max, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[valid_mask] = restored
    return out


def _as_float_array(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _rank_array(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("Rank transform expects a 1D array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Rank transform expects only finite values.")
    return np.asarray(rankdata(values, method="average"), dtype=float)


def _numeric_nuisance_terms(
    *,
    model_terms: Sequence[str],
    categorical_terms: Sequence[str],
) -> Tuple[str, ...]:
    categorical = {str(term) for term in categorical_terms}
    return tuple(term for term in model_terms if str(term) not in categorical)


def _design_matrix(
    *,
    table: pd.DataFrame,
    numeric_terms: Sequence[str],
    factor_terms: Sequence[str],
    include_run_fixed_effect: bool,
) -> np.ndarray:
    parts: List[np.ndarray] = [np.ones((len(table), 1), dtype=float)]
    for term in numeric_terms:
        values = _rank_array(_as_float_array(table[term]))
        parts.append(values.reshape(-1, 1))
    effective_factors = list(str(term) for term in factor_terms)
    if include_run_fixed_effect:
        effective_factors.append("run_num")
    for term in effective_factors:
        if term not in table.columns:
            raise ValueError(f"Residualized-correlation factor {term!r} is missing.")
        factor = pd.get_dummies(table[term].astype(str), prefix=term, drop_first=True)
        if factor.empty:
            continue
        parts.append(factor.to_numpy(dtype=float))
    return np.concatenate(parts, axis=1)


def _residualize(
    *,
    values: np.ndarray,
    design: np.ndarray,
) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("Residualization expects a 1D response vector.")
    if design.ndim != 2 or design.shape[0] != values.shape[0]:
        raise ValueError("Residualization design has incompatible shape.")
    coefficients, *_rest = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    return np.asarray(values - fitted, dtype=float)


def _pearson_r(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Correlation expects matching 1D vectors.")
    left_centered = np.asarray(left, dtype=float) - float(np.mean(left))
    right_centered = np.asarray(right, dtype=float) - float(np.mean(right))
    denom = float(
        np.sqrt(np.sum(np.square(left_centered)) * np.sum(np.square(right_centered)))
    )
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("Residual vectors have zero variance.")
    return float(np.sum(left_centered * right_centered) / denom)


def _subject_status_map(analysis_cells: pd.DataFrame) -> Dict[str, str]:
    if analysis_cells.empty:
        return {}
    return {
        str(row.analysis_id): str(row.status)
        for row in analysis_cells.itertuples(index=False)
    }


def _subject_count_map(analysis_cells: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
    if analysis_cells.empty:
        return {}
    return {
        str(row.analysis_id): (int(row.n_trials), int(row.n_runs))
        for row in analysis_cells.itertuples(index=False)
    }


def _analysis_row(
    *,
    subject: str,
    cell: Any,
    status: str,
    n_trials: int,
    n_runs: int,
    rho: float = np.nan,
    fisher_z: float = np.nan,
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
        "rho": float(rho),
        "fisher_z": float(fisher_z),
        "weight": 1.0,
        "status": str(status),
        "interpretable": str(status) == "ok",
        "message": str(message).strip(),
    }


def fit_subject_residualized_correlations(
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
        predictor = _rank_array(_as_float_array(subset[cell.predictor_column]))
        outcome = _rank_array(_as_float_array(subset[cell.outcome_column]))
        design = _design_matrix(
            table=subset,
            numeric_terms=_numeric_nuisance_terms(
                model_terms=cell.model_terms,
                categorical_terms=cell.categorical_terms,
            ),
            factor_terms=cell.categorical_terms,
            include_run_fixed_effect=include_run_fixed_effect,
        )
        predictor_residual = _residualize(values=predictor, design=design)
        outcome_residual = _residualize(values=outcome, design=design)
        try:
            rho = _pearson_r(predictor_residual, outcome_residual)
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
        clipped = float(np.clip(rho, -0.999999, 0.999999))
        rows.append(
            _analysis_row(
                subject=subject,
                cell=cell,
                status="ok",
                n_trials=int(len(subset)),
                n_runs=int(subset["run_num"].nunique()),
                rho=rho,
                fisher_z=float(np.arctanh(clipped)),
            )
        )
    return pd.DataFrame(rows)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total_weight = float(np.sum(weights))
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("Weighted mean requires positive total weight.")
    return float(np.sum(values * weights) / total_weight)


def _bootstrap_fisher_z(
    *,
    fisher_z: np.ndarray,
    weights: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_subjects = fisher_z.size
    bootstrap = np.empty(iterations, dtype=float)
    for idx in range(iterations):
        sample_idx = rng.integers(0, n_subjects, size=n_subjects)
        bootstrap[idx] = _weighted_mean(fisher_z[sample_idx], weights[sample_idx])
    return bootstrap


def _exact_sign_matrix(n_subjects: int) -> np.ndarray:
    rows = list(itertools.product((-1.0, 1.0), repeat=n_subjects))
    return np.asarray(rows, dtype=float)


def _permutation_p_value(
    *,
    observed: float,
    fisher_z: np.ndarray,
    weights: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> float:
    n_subjects = fisher_z.size
    if n_subjects <= 12:
        sign_matrix = _exact_sign_matrix(n_subjects)
    else:
        sign_matrix = rng.choice(
            (-1.0, 1.0),
            size=(iterations, n_subjects),
            replace=True,
        )
    permuted = sign_matrix * fisher_z.reshape(1, -1)
    statistics = np.asarray(
        [
            _weighted_mean(row, weights)
            for row in permuted
        ],
        dtype=float,
    )
    extreme = float(np.sum(np.abs(statistics) >= abs(observed)))
    return float((extreme + 1.0) / (statistics.size + 1.0))


def _group_result_row(
    *,
    analysis_id: str,
    family: str,
    roi: str,
    band: str,
    predictor_column: str,
    outcome_column: str,
    n_subjects: int,
    n_trials: int,
    rho: float = np.nan,
    fisher_z: float = np.nan,
    ci_low: float = np.nan,
    ci_high: float = np.nan,
    p_value: float = np.nan,
    status: str,
    message: str = "",
) -> Dict[str, Any]:
    interpretable = str(status) == "ok"
    return {
        "analysis_id": str(analysis_id),
        "family": str(family),
        "roi": str(roi),
        "band": str(band),
        "predictor_column": str(predictor_column),
        "outcome_column": str(outcome_column),
        "n_subjects": int(n_subjects),
        "n_trials": int(n_trials),
        "rho": float(rho),
        "fisher_z": float(fisher_z),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "status": str(status),
        "interpretable": interpretable,
        "message": str(message).strip(),
    }


def aggregate_residualized_correlations(
    *,
    subject_effects: pd.DataFrame,
    cfg: ResidualizedCorrelationSensitivityConfig,
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
                _group_result_row(
                    analysis_id=str(cell.analysis_id),
                    family=str(cell.family),
                    roi=str(cell.roi),
                    band=str(cell.band),
                    predictor_column=str(cell.predictor_column),
                    outcome_column=str(cell.outcome_column),
                    n_subjects=int(len(subset)),
                    n_trials=int(pd.to_numeric(subset["n_trials"], errors="coerce").sum()),
                    status="insufficient_subjects",
                    message="Residualized-correlation sensitivity requires at least two subjects.",
                )
            )
            continue
        fisher_z = pd.to_numeric(subset["fisher_z"], errors="coerce").to_numpy(dtype=float)
        weights = pd.to_numeric(subset["weight"], errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(fisher_z) & np.isfinite(weights) & (weights > 0)
        if int(np.sum(valid_mask)) < 2:
            rows.append(
                _group_result_row(
                    analysis_id=str(cell.analysis_id),
                    family=str(cell.family),
                    roi=str(cell.roi),
                    band=str(cell.band),
                    predictor_column=str(cell.predictor_column),
                    outcome_column=str(cell.outcome_column),
                    n_subjects=int(np.sum(valid_mask)),
                    n_trials=int(pd.to_numeric(subset["n_trials"], errors="coerce").sum()),
                    status="insufficient_subjects",
                    message="Residualized-correlation sensitivity retained fewer than two valid subjects.",
                )
            )
            continue
        fisher_z = fisher_z[valid_mask]
        weights = weights[valid_mask]
        observed = _weighted_mean(fisher_z, weights)
        bootstrap = _bootstrap_fisher_z(
            fisher_z=fisher_z,
            weights=weights,
            iterations=cfg.bootstrap_iterations,
            rng=rng,
        )
        ci_low_z = float(np.quantile(bootstrap, 0.025))
        ci_high_z = float(np.quantile(bootstrap, 0.975))
        p_value = _permutation_p_value(
            observed=observed,
            fisher_z=fisher_z,
            weights=weights,
            iterations=cfg.permutation_iterations,
            rng=rng,
        )
        rows.append(
            _group_result_row(
                analysis_id=str(cell.analysis_id),
                family=str(cell.family),
                roi=str(cell.roi),
                band=str(cell.band),
                predictor_column=str(cell.predictor_column),
                outcome_column=str(cell.outcome_column),
                n_subjects=int(fisher_z.size),
                n_trials=int(pd.to_numeric(subset["n_trials"], errors="coerce").sum()),
                rho=float(np.tanh(observed)),
                fisher_z=float(observed),
                ci_low=float(np.tanh(ci_low_z)),
                ci_high=float(np.tanh(ci_high_z)),
                p_value=float(p_value),
                status="ok",
            )
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
    "aggregate_residualized_correlations",
    "fit_subject_residualized_correlations",
]
