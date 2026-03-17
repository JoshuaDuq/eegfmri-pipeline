"""Statistical modeling utilities for pain EEG-BOLD coupling."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from eeg_pipeline.utils.config.loader import get_config_value


_SINGULAR_TOL = 1.0e-8


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return value


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


def _subject_zscore(values: pd.Series, subjects: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    subject_labels = subjects.astype(str)
    for subject in subject_labels.unique():
        mask = subject_labels == subject
        numeric = pd.to_numeric(values.loc[mask], errors="coerce")
        arr = numeric.to_numpy(dtype=float)
        finite = np.isfinite(arr)
        if not np.any(finite):
            continue
        mean = float(np.mean(arr[finite]))
        std = float(np.std(arr[finite], ddof=0))
        if not np.isfinite(std) or std <= 0:
            continue
        out.loc[mask] = (arr - mean) / std
    return out


def _subject_standardize(
    values: pd.Series,
    subjects: pd.Series,
) -> Tuple[pd.Series, Dict[str, float]]:
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    scale_by_subject: Dict[str, float] = {}
    subject_labels = subjects.astype(str)
    for subject in subject_labels.unique():
        mask = subject_labels == subject
        numeric = pd.to_numeric(values.loc[mask], errors="coerce")
        arr = numeric.to_numpy(dtype=float)
        finite = np.isfinite(arr)
        if not np.any(finite):
            continue
        mean = float(np.mean(arr[finite]))
        std = float(np.std(arr[finite], ddof=0))
        if not np.isfinite(std) or std <= 0:
            continue
        out.loc[mask] = (arr - mean) / std
        scale_by_subject[str(subject)] = std
    return out, scale_by_subject


def _is_binary_column(values: pd.Series) -> bool:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric.to_numpy(dtype=float))]
    if finite.empty:
        return False
    unique = set(float(v) for v in finite.unique().tolist())
    return unique.issubset({0.0, 1.0})


def _serialize_terms(terms: Sequence[str]) -> str:
    return "|".join(str(term) for term in terms)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes"}


def _r_backend_script_path() -> Path:
    script_path = Path(__file__).with_name("eeg_bold_nlme_backend.R")
    if not script_path.exists():
        raise FileNotFoundError(f"Missing R backend script: {script_path}")
    return script_path


def _resolve_rscript_path(configured: str) -> str:
    if not configured:
        raise ValueError(
            "eeg_bold_coupling.statistics.rscript_path must not be blank."
        )
    if Path(configured).is_absolute():
        if not Path(configured).exists():
            raise FileNotFoundError(f"Rscript executable not found: {configured}")
        return configured
    resolved = shutil.which(configured)
    if resolved is None:
        raise FileNotFoundError(
            f"Rscript executable {configured!r} is not available on PATH."
        )
    return resolved


def _invalid_fit_row(
    *,
    cell: "CellSpec",
    model_table: pd.DataFrame,
    status: str,
    message: str,
) -> Dict[str, Any]:
    return {
        "analysis_id": cell.analysis_id,
        "family": cell.family,
        "roi": cell.roi,
        "band": cell.band,
        "predictor_column": cell.predictor_column,
        "outcome_column": cell.outcome_column,
        "n_trials": int(len(model_table)),
        "n_subjects": int(model_table["subject"].nunique()) if not model_table.empty else 0,
        "n_runs": int(model_table[["subject", "run_num"]].drop_duplicates().shape[0])
        if not model_table.empty
        else 0,
        "beta": np.nan,
        "se": np.nan,
        "z_value": np.nan,
        "p_value": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "rho": np.nan,
        "converged": False,
        "singular": False,
        "loglik": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "status": str(status),
        "interpretable": False,
        "message": str(message).strip(),
    }


@dataclass(frozen=True)
class CouplingStatisticsConfig:
    backend: str
    min_trials_per_subject: int
    min_runs_per_subject: int
    alpha: float
    include_run_fixed_effect: bool
    use_outcome_variance: bool
    rscript_path: str
    fit_method: str
    max_iterations: int
    em_iterations: int
    singular_tolerance: float

    @classmethod
    def from_config(cls, config: Any) -> "CouplingStatisticsConfig":
        raw = _require_mapping(
            get_config_value(config, "eeg_bold_coupling.statistics", {}),
            path="eeg_bold_coupling.statistics",
        )
        backend = str(raw.get("backend", "nlme_lme_ar1")).strip().lower()
        if backend != "nlme_lme_ar1":
            raise ValueError(
                "eeg_bold_coupling.statistics.backend must be 'nlme_lme_ar1'."
            )
        fit_method = str(raw.get("fit_method", "reml")).strip().lower()
        if fit_method != "reml":
            raise ValueError(
                "eeg_bold_coupling.statistics.fit_method must be 'reml'."
            )
        return cls(
            backend=backend,
            min_trials_per_subject=int(raw.get("min_trials_per_subject", 20)),
            min_runs_per_subject=int(raw.get("min_runs_per_subject", 2)),
            alpha=float(raw.get("alpha", 0.05)),
            include_run_fixed_effect=bool(raw.get("include_run_fixed_effect", True)),
            use_outcome_variance=bool(raw.get("use_outcome_variance", True)),
            rscript_path=str(raw.get("rscript_path", "Rscript")).strip(),
            fit_method=fit_method,
            max_iterations=int(raw.get("max_iterations", 200)),
            em_iterations=int(raw.get("em_iterations", 50)),
            singular_tolerance=float(
                raw.get("singular_tolerance", _SINGULAR_TOL)
            ),
        )


@dataclass(frozen=True)
class CellSpec:
    analysis_id: str
    family: str
    roi: str
    band: str
    predictor_column: str
    outcome_column: str
    outcome_variance_column: Optional[str]
    model_terms: Tuple[str, ...]
    categorical_terms: Tuple[str, ...]
    nonstandardized_terms: Tuple[str, ...] = ()


def summarize_subject_cells(
    *,
    subject: str,
    merged_table: pd.DataFrame,
    cell_specs: Sequence[CellSpec],
    stats_cfg: CouplingStatisticsConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cell in cell_specs:
        required_columns = [cell.predictor_column, cell.outcome_column, *cell.model_terms]
        if (
            stats_cfg.use_outcome_variance
            and cell.outcome_variance_column is not None
        ):
            required_columns.append(cell.outcome_variance_column)
        subset = merged_table[required_columns + ["run_num"]].copy()
        subset = subset.dropna().reset_index(drop=True)
        n_trials = int(len(subset))
        n_runs = int(subset["run_num"].nunique()) if not subset.empty else 0
        status = "ok"
        if n_trials < stats_cfg.min_trials_per_subject:
            status = "insufficient_trials"
        elif n_runs < stats_cfg.min_runs_per_subject:
            status = "insufficient_runs"
        else:
            predictor_values = pd.to_numeric(
                subset[cell.predictor_column],
                errors="coerce",
            ).to_numpy(dtype=float)
            outcome_values = pd.to_numeric(
                subset[cell.outcome_column],
                errors="coerce",
            ).to_numpy(dtype=float)
            if (
                np.nanstd(predictor_values, ddof=0) <= 0
                or np.nanstd(outcome_values, ddof=0) <= 0
            ):
                status = "insufficient_variance"
        rows.append(
            {
                "subject": str(subject),
                "analysis_id": cell.analysis_id,
                "family": cell.family,
                "roi": cell.roi,
                "band": cell.band,
                "predictor_column": cell.predictor_column,
                "outcome_column": cell.outcome_column,
                "outcome_variance_column": (
                    "" if cell.outcome_variance_column is None else cell.outcome_variance_column
                ),
                "model_terms": json.dumps(list(cell.model_terms)),
                "categorical_terms": json.dumps(list(cell.categorical_terms)),
                "nonstandardized_terms": json.dumps(list(cell.nonstandardized_terms)),
                "n_trials": n_trials,
                "n_runs": n_runs,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def _prepare_model_table(
    *,
    table: pd.DataFrame,
    predictor_column: str,
    outcome_column: str,
    outcome_variance_column: Optional[str],
    model_terms: Sequence[str],
    categorical_terms: Sequence[str],
    nonstandardized_terms: Sequence[str],
    include_run_fixed_effect: bool,
    use_outcome_variance: bool,
) -> Tuple[pd.DataFrame, List[str], List[str], Optional[str]]:
    out = table.copy()
    out["subject"] = out["subject"].astype(str)
    if "trial_position" not in out.columns:
        raise ValueError(
            "Pooled table is missing trial_position required for AR(1) spacing."
        )
    out["trial_position"] = pd.to_numeric(
        out["trial_position"],
        errors="coerce",
    )
    if not np.all(np.isfinite(out["trial_position"].to_numpy(dtype=float))):
        raise ValueError("Pooled table trial_position contains non-finite values.")
    out["trial_time"] = pd.to_numeric(out["onset"], errors="coerce")
    if not np.all(np.isfinite(out["trial_time"].to_numpy(dtype=float))):
        raise ValueError("Pooled table onset contains non-finite values for continuous-time correlation.")
    duplicate_time = out.duplicated(subset=["subject", "run_num", "trial_time"])
    if bool(duplicate_time.any()):
        raise ValueError(
            "Within-run trial onsets must be unique for continuous-time correlation."
        )
    out = out.sort_values(
        ["subject", "run_num", "trial_time", "duration"]
    ).reset_index(drop=True)

    categorical = set(str(term) for term in categorical_terms)
    nonstandardized = set(str(term) for term in nonstandardized_terms)
    if include_run_fixed_effect:
        categorical.add("run_num")

    continuous_columns = [predictor_column, outcome_column]
    numeric_terms: List[str] = []
    factor_terms: List[str] = []
    for term in model_terms:
        if term in categorical:
            factor_terms.append(term)
            continue
        if term not in out.columns:
            raise ValueError(f"Model term {term!r} is missing from pooled table.")
        numeric_terms.append(term)
        if term not in nonstandardized and not _is_binary_column(out[term]):
            continuous_columns.append(term)

    outcome_scale_by_subject: Dict[str, float] = {}
    for column in continuous_columns:
        if column not in out.columns:
            raise ValueError(f"Required model column {column!r} is missing.")
        standardized, scale_by_subject = _subject_standardize(
            pd.to_numeric(out[column], errors="coerce"),
            out["subject"],
        )
        out[column] = standardized
        if column == outcome_column:
            outcome_scale_by_subject = scale_by_subject

    for term in numeric_terms:
        out[term] = pd.to_numeric(out[term], errors="coerce")
    for term in factor_terms:
        if term not in out.columns:
            raise ValueError(f"Categorical term {term!r} is missing from pooled table.")
        out[term] = out[term].astype(str)

    variance_column: Optional[str] = None
    if use_outcome_variance and outcome_variance_column is not None:
        if outcome_variance_column not in out.columns:
            raise ValueError(
                f"Outcome variance column {outcome_variance_column!r} is missing."
            )
        out[outcome_variance_column] = pd.to_numeric(
            out[outcome_variance_column],
            errors="coerce",
        )
        scale_values = out["subject"].astype(str).map(outcome_scale_by_subject)
        out[outcome_variance_column] = (
            out[outcome_variance_column].to_numpy(dtype=float)
            / np.square(pd.to_numeric(scale_values, errors="coerce").to_numpy(dtype=float))
        )
        positive_mask = (
            np.isfinite(out[outcome_variance_column].to_numpy(dtype=float))
            & (out[outcome_variance_column].to_numpy(dtype=float) > 0)
        )
        out = out.loc[positive_mask].reset_index(drop=True)
        variance_column = outcome_variance_column

    keep_columns = [
        "subject",
        "run_num",
        "trial_position",
        "trial_time",
        "onset",
        "duration",
        predictor_column,
        outcome_column,
        *numeric_terms,
        *factor_terms,
    ]
    if variance_column is not None:
        keep_columns.append(variance_column)
    keep_columns = list(dict.fromkeys(keep_columns))
    out = out[keep_columns].dropna().reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid rows remained after model-table preparation.")
    return out, numeric_terms, factor_terms, variance_column


def _run_nlme_backend(
    *,
    model_table: pd.DataFrame,
    cell: CellSpec,
    numeric_terms: Sequence[str],
    factor_terms: Sequence[str],
    outcome_variance_column: Optional[str],
    stats_cfg: CouplingStatisticsConfig,
) -> Dict[str, Any]:
    rscript_path = _resolve_rscript_path(stats_cfg.rscript_path)
    script_path = _r_backend_script_path()
    with tempfile.TemporaryDirectory(prefix="eeg_bold_nlme_") as tmpdir:
        tmp_path = Path(tmpdir)
        data_path = tmp_path / "model_table.tsv"
        spec_path = tmp_path / "model_spec.tsv"
        output_path = tmp_path / "result.tsv"
        model_table.to_csv(data_path, sep="\t", index=False)
        pd.DataFrame(
            [
                {
                    "predictor_column": cell.predictor_column,
                    "outcome_column": cell.outcome_column,
                    "outcome_variance_column": (
                        "" if outcome_variance_column is None else outcome_variance_column
                    ),
                    "numeric_terms": _serialize_terms(numeric_terms),
                    "factor_terms": _serialize_terms(factor_terms),
                    "fit_method": stats_cfg.fit_method,
                    "max_iterations": int(stats_cfg.max_iterations),
                    "em_iterations": int(stats_cfg.em_iterations),
                    "singular_tolerance": float(stats_cfg.singular_tolerance),
                }
            ]
        ).to_csv(spec_path, sep="\t", index=False)
        completed = subprocess.run(
            [rscript_path, str(script_path), str(data_path), str(spec_path), str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
            detail = stderr or stdout or f"exit code {completed.returncode}"
            raise RuntimeError(f"R nlme backend failed: {detail}")
        if not output_path.exists():
            raise RuntimeError(
                "R nlme backend did not produce an output file."
            )
        out = pd.read_csv(output_path, sep="\t")
        if out.empty:
            raise RuntimeError("R nlme backend returned an empty result.")
        row = out.iloc[0].to_dict()
    return {
        "status": str(row.get("status", "")).strip(),
        "message": str(row.get("message", "")).strip(),
        "converged": _parse_bool(row.get("converged", False)),
        "singular": _parse_bool(row.get("singular", False)),
        "beta": float(pd.to_numeric(pd.Series([row.get("beta")]), errors="coerce").iloc[0]),
        "se": float(pd.to_numeric(pd.Series([row.get("se")]), errors="coerce").iloc[0]),
        "p_value": float(pd.to_numeric(pd.Series([row.get("p_value")]), errors="coerce").iloc[0]),
        "ci_low": float(pd.to_numeric(pd.Series([row.get("ci_low")]), errors="coerce").iloc[0]),
        "ci_high": float(pd.to_numeric(pd.Series([row.get("ci_high")]), errors="coerce").iloc[0]),
        "rho": float(pd.to_numeric(pd.Series([row.get("rho")]), errors="coerce").iloc[0]),
        "loglik": float(pd.to_numeric(pd.Series([row.get("loglik")]), errors="coerce").iloc[0]),
        "aic": float(pd.to_numeric(pd.Series([row.get("aic")]), errors="coerce").iloc[0]),
        "bic": float(pd.to_numeric(pd.Series([row.get("bic")]), errors="coerce").iloc[0]),
    }


def fit_mixedlm_cell(
    *,
    pooled_table: pd.DataFrame,
    cell: CellSpec,
    stats_cfg: CouplingStatisticsConfig,
) -> Dict[str, Any]:
    model_table, numeric_terms, factor_terms, outcome_variance_column = _prepare_model_table(
        table=pooled_table,
        predictor_column=cell.predictor_column,
        outcome_column=cell.outcome_column,
        outcome_variance_column=cell.outcome_variance_column,
        model_terms=cell.model_terms,
        categorical_terms=cell.categorical_terms,
        nonstandardized_terms=cell.nonstandardized_terms,
        include_run_fixed_effect=stats_cfg.include_run_fixed_effect,
        use_outcome_variance=stats_cfg.use_outcome_variance,
    )
    if int(model_table["subject"].nunique()) < 2:
        return _invalid_fit_row(
            cell=cell,
            model_table=model_table,
            status="insufficient_subjects",
            message="Exact mixed model requires at least two subjects.",
        )

    result = _run_nlme_backend(
        model_table=model_table,
        cell=cell,
        numeric_terms=numeric_terms,
        factor_terms=factor_terms,
        outcome_variance_column=outcome_variance_column,
        stats_cfg=stats_cfg,
    )
    beta = result["beta"]
    se = result["se"]
    z_value = beta / se if np.isfinite(beta) and np.isfinite(se) and se > 0 else np.nan
    status = result["status"] or "model_not_interpretable"
    interpretable = (
        status == "ok"
        and bool(result["converged"])
        and not bool(result["singular"])
    )
    return {
        "analysis_id": cell.analysis_id,
        "family": cell.family,
        "roi": cell.roi,
        "band": cell.band,
        "predictor_column": cell.predictor_column,
        "outcome_column": cell.outcome_column,
        "outcome_variance_column": (
            "" if outcome_variance_column is None else outcome_variance_column
        ),
        "n_trials": int(len(model_table)),
        "n_subjects": int(model_table["subject"].nunique()),
        "n_runs": int(model_table[["subject", "run_num"]].drop_duplicates().shape[0]),
        "beta": beta,
        "se": se,
        "z_value": float(z_value),
        "p_value": result["p_value"],
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
        "rho": result["rho"],
        "converged": bool(result["converged"]),
        "singular": bool(result["singular"]),
        "loglik": result["loglik"],
        "aic": result["aic"],
        "bic": result["bic"],
        "status": status,
        "interpretable": bool(interpretable),
        "message": result["message"],
    }


def finalize_group_results(
    *,
    rows: Sequence[Mapping[str, Any]],
    alpha: float,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(
        ["family", "roi", "band", "analysis_id"]
    ).reset_index(drop=True)
    out["p_holm"] = np.nan
    valid_mask = (
        (out["family"].astype(str) == "confirmatory")
        & out["interpretable"].astype(bool)
        & np.isfinite(pd.to_numeric(out["p_value"], errors="coerce").to_numpy(dtype=float))
    )
    if bool(valid_mask.any()):
        adjusted = _holm_adjust(
            pd.to_numeric(out.loc[valid_mask, "p_value"], errors="coerce").to_numpy(dtype=float)
        )
        out.loc[valid_mask, "p_holm"] = adjusted
    out["significant_holm"] = (
        out["interpretable"].astype(bool)
        & (pd.to_numeric(out["p_holm"], errors="coerce") < float(alpha))
    )
    return out


__all__ = [
    "CellSpec",
    "CouplingStatisticsConfig",
    "finalize_group_results",
    "fit_mixedlm_cell",
    "summarize_subject_cells",
]
