from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.behavior_loader import ensure_behavior_config
from eeg_pipeline.utils.config.loader import get_config_value, require_config_value


def _normalize_alignment_values(values: pd.Series) -> pd.Series:
    """Normalize task-identity values before cross-run consistency checks."""
    numeric_values = pd.to_numeric(values, errors="coerce")
    if numeric_values.notna().sum() == values.notna().sum():
        return pd.Series(
            np.round(numeric_values.to_numpy(dtype=float), decimals=8),
            index=values.index,
            dtype=float,
        )
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.where(values.notna(), pd.NA)


def _resolve_icc_alignment_columns(df_trials: pd.DataFrame, config: Any) -> List[str]:
    """Resolve task-identity columns used to validate trial-position ICC."""
    from eeg_pipeline.utils.data.columns import (
        get_binary_outcome_column_from_config,
        get_condition_column_from_config,
        resolve_predictor_column,
    )

    compare_column_value = get_config_value(
        config, "behavior_analysis.condition.compare_column", None
    )
    compare_column = str(compare_column_value or "").strip()
    if not compare_column:
        compare_column = str(get_condition_column_from_config(config, df_trials) or "").strip()
    if not compare_column:
        compare_column = str(get_binary_outcome_column_from_config(config, df_trials) or "").strip()

    predictor_column = str(resolve_predictor_column(df_trials, config) or "").strip()

    candidates = [compare_column, predictor_column]
    resolved: List[str] = []
    for column in candidates:
        if column and column in df_trials.columns and column not in resolved:
            resolved.append(column)
    return resolved


def _resolve_configured_icc_unit_columns(
    df_trials: pd.DataFrame,
    *,
    config: Any,
    trial_col: Optional[str],
) -> List[str]:
    """Resolve configured ICC unit columns from aliases or explicit column names."""
    from eeg_pipeline.utils.data.columns import (
        get_binary_outcome_column_from_config,
        get_condition_column_from_config,
        resolve_predictor_column,
    )

    config = ensure_behavior_config(config)
    compare_column_value = get_config_value(
        config, "behavior_analysis.condition.compare_column", None
    )
    compare_column = str(compare_column_value or "").strip()
    if not compare_column:
        compare_column = str(
            get_condition_column_from_config(config, df_trials) or ""
        ).strip()
    if not compare_column:
        compare_column = str(
            get_binary_outcome_column_from_config(config, df_trials) or ""
        ).strip()

    predictor_column = str(resolve_predictor_column(df_trials, config) or "").strip()
    configured_specs = require_config_value(
        config,
        "behavior_analysis.icc.unit_columns",
    )
    unit_specs = [str(spec).strip() for spec in (configured_specs or [])]
    if not unit_specs:
        raise ValueError("behavior_analysis.icc.unit_columns must not be empty.")

    alias_map = {
        "predictor": predictor_column,
        "condition": compare_column,
        "trial_type": "trial_type",
        "trial_position": trial_col or "",
    }
    resolved: List[str] = []
    unresolved: List[str] = []
    for spec in unit_specs:
        token = spec.strip()
        if not token:
            raise ValueError("behavior_analysis.icc.unit_columns contains a blank entry.")

        actual_column = alias_map.get(token.lower(), token)
        if not actual_column or actual_column not in df_trials.columns:
            unresolved.append(token)
            continue
        if actual_column not in resolved:
            resolved.append(actual_column)

    if unresolved:
        raise ValueError(
            "ICC unit column resolution failed for: "
            f"{', '.join(unresolved)}. Configure behavior_analysis.icc.unit_columns "
            "with resolvable columns or aliases."
        )
    return resolved


def _select_icc_unit_columns(
    df_trials: pd.DataFrame,
    *,
    config: Any,
    trial_col: Optional[str],
) -> Tuple[List[str], List[str], str]:
    """Choose the repeated-measures unit used to align runs for ICC."""
    unit_columns = _resolve_configured_icc_unit_columns(
        df_trials,
        config=config,
        trial_col=trial_col,
    )
    if trial_col and unit_columns == [trial_col]:
        alignment_columns = [
            column
            for column in _resolve_icc_alignment_columns(df_trials, config)
            if column not in unit_columns
        ]
        return unit_columns, alignment_columns, "trial_position"
    return unit_columns, [], "task_identity"


def _validate_icc_alignment_design(
    df_trials: pd.DataFrame,
    *,
    run_col: str,
    trial_col: str,
    alignment_columns: List[str],
) -> None:
    """Reject ICC designs where trial positions do not represent the same task unit across runs."""
    for column in alignment_columns:
        observed = df_trials[[run_col, trial_col, column]].dropna()
        if observed.empty:
            continue

        within_cell_unique = observed.groupby(
            [run_col, trial_col],
            dropna=True,
        )[column].nunique(dropna=True)
        ambiguous_cells = within_cell_unique[within_cell_unique > 1]
        if not ambiguous_cells.empty:
            raise ValueError(
                f"ICC alignment is invalid because '{column}' is not unique within "
                f"({run_col}, {trial_col}) for {int(len(ambiguous_cells))} cells."
            )

        normalized = observed.groupby([trial_col, run_col], dropna=True)[column].first().reset_index()
        normalized["__value__"] = _normalize_alignment_values(normalized[column])
        across_run_unique = normalized.groupby(trial_col, dropna=True)["__value__"].nunique(dropna=True)
        inconsistent_trials = across_run_unique[across_run_unique > 1]
        if inconsistent_trials.empty:
            continue

        example_trial = inconsistent_trials.index[0]
        example_rows = normalized.loc[normalized[trial_col] == example_trial, [run_col, column]]
        example_pairs = ", ".join(
            f"{run_col}={row[run_col]} -> {column}={row[column]}"
            for _, row in example_rows.iterrows()
        )
        raise ValueError(
            f"ICC alignment is invalid because '{column}' changes across runs for the same '{trial_col}' "
            f"({int(len(inconsistent_trials))} inconsistent trial positions; example {trial_col}={example_trial}: "
            f"{example_pairs}). Use a stable repeated-measures key before computing ICC."
        )


def stage_icc_impl(
    ctx: Any,
    config: Any,
    *,
    build_output_filename_fn: Callable[[Any, Any, str], str],
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    get_feature_columns_fn: Callable[[pd.DataFrame, Any], List[str]],
    check_early_exit_conditions_fn: Callable[..., Tuple[bool, Optional[str]]],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    write_stats_table_fn: Callable[[Any, pd.DataFrame, Path], Path],
    write_metadata_file_fn: Callable[[Path, Dict[str, Any]], None],
) -> pd.DataFrame:
    """Assess within-subject run-to-run reliability (ICC) of EEG features."""
    from eeg_pipeline.utils.analysis.stats.reliability import compute_icc

    ctx.config = ensure_behavior_config(ctx.config)
    filename = build_output_filename_fn(ctx, config, "icc_reliability")

    df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.warning("ICC: trial table missing; skipping.")
        return pd.DataFrame()

    run_col = str(
        require_config_value(ctx.config, "behavior_analysis.run_adjustment.column")
    ).strip()
    if run_col not in df_trials.columns:
        if "run" in df_trials.columns:
            run_col = "run"
        elif "block" in df_trials.columns:
            run_col = "block"
        else:
            ctx.logger.info("ICC: no run/block column available; skipping.")
            return pd.DataFrame()

    feature_cols = get_feature_columns_fn(df_trials, ctx)

    should_skip, skip_reason = check_early_exit_conditions_fn(
        df_trials,
        feature_cols,
        min_features=1,
        min_trials=1,
    )
    if should_skip:
        ctx.logger.info(f"ICC: skipping due to {skip_reason}")
        return pd.DataFrame()

    trial_col = None
    for cand in (
        "trial_index_within_group",
        "trial_in_run",
        "trial",
        "trial_number",
        "trial_index",
    ):
        if cand in df_trials.columns:
            trial_col = cand
            break

    unit_columns, alignment_columns, alignment_strategy = _select_icc_unit_columns(
        df_trials,
        config=ctx.config,
        trial_col=trial_col,
    )
    if alignment_strategy == "trial_position" and alignment_columns:
        _validate_icc_alignment_design(
            df_trials,
            run_col=run_col,
            trial_col=unit_columns[0],
            alignment_columns=alignment_columns,
        )
        ctx.logger.info("ICC: aligning runs by trial position '%s'.", unit_columns[0])
    elif alignment_strategy == "task_identity":
        ctx.logger.info(
            "ICC: aligning runs by task identity columns: %s",
            ", ".join(unit_columns),
        )

    records = []
    
    unique_runs = df_trials[run_col].dropna().unique()
    if len(unique_runs) < 2:
        ctx.logger.info("ICC: only one run available; skipping.")
        return pd.DataFrame()
        
    for feat in feature_cols:
        if feat not in df_trials.columns:
            continue

        feat_df = df_trials[[*unit_columns, run_col, feat]].dropna()
        if feat_df.empty:
            continue

        pivoted = feat_df.pivot_table(
            index=unit_columns,
            columns=run_col,
            values=feat,
            aggfunc="mean",
        )
        valid_pivoted = pivoted.dropna()

        if len(valid_pivoted) < 2:
            continue

        data = valid_pivoted.to_numpy()
        icc_val, ci_low, ci_high = compute_icc(data, icc_type="ICC(3,1)")

        records.append({
            "feature": feat,
            "icc": float(icc_val),
            "ci_lower_95": float(ci_low) if not pd.isna(ci_low) else float("nan"),
            "ci_upper_95": float(ci_high) if not pd.isna(ci_high) else float("nan"),
            "n_trials_aligned": int(len(valid_pivoted)),
            "n_runs": int(data.shape[1]),
        })

    out_df = pd.DataFrame(records)
    meta = {
        "status": "ok" if not out_df.empty else "empty",
        "icc_type": "ICC(3,1)",
        "run_col": run_col,
        "trial_col": trial_col,
        "unit_columns": list(unit_columns),
        "alignment_strategy": alignment_strategy,
    }

    ctx.data_qc["icc_reliability"] = meta

    out_dir = get_stats_subfolder_fn(ctx, "icc_reliability")
    out_path = out_dir / f"{filename}.parquet"
    if not out_df.empty:
        actual_path = write_stats_table_fn(ctx, out_df, out_path)
        ctx.logger.info("ICC reliability saved: %s (%d features)", actual_path.name, len(out_df))
    write_metadata_file_fn(out_dir / f"{filename}.metadata.json", meta)
    return out_df
