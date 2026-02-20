from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd


def is_dataframe_valid_impl(df: Optional[pd.DataFrame]) -> bool:
    """Check if DataFrame is not None and not empty."""
    return df is not None and not df.empty


def check_early_exit_conditions_impl(
    df: Optional[pd.DataFrame],
    *,
    feature_cols: Optional[List[str]] = None,
    min_features: int = 1,
    min_trials: int = 1,
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool] = is_dataframe_valid_impl,
) -> Tuple[bool, Optional[str]]:
    """Check if analysis should skip due to insufficient data."""
    if not is_dataframe_valid_fn(df):
        return True, "no_dataframe"

    if len(df) < min_trials:
        return True, f"insufficient_trials ({len(df)} < {min_trials})"

    if feature_cols is not None:
        if len(feature_cols) < min_features:
            return True, f"insufficient_features ({len(feature_cols)} < {min_features})"

        has_valid_features = False
        for col in feature_cols:
            if col in df.columns:
                valid_count = pd.to_numeric(df[col], errors="coerce").notna().sum()
                if valid_count >= min_trials:
                    has_valid_features = True
                    break

        if not has_valid_features:
            return True, "no_features_with_valid_data"

    return False, None


def write_stats_table_impl(
    ctx: Any,
    df: pd.DataFrame,
    path: Path,
    *,
    force_tsv: bool = False,
) -> Path:
    """Write stats table and optional CSV sidecar."""
    from eeg_pipeline.infra.tsv import write_stats_table

    actual_path = write_stats_table(df, path, force_tsv=force_tsv)

    if ctx.also_save_csv:
        from eeg_pipeline.infra.tsv import write_csv

        csv_path = actual_path.with_suffix(".csv")
        write_csv(df, csv_path, index=False)
        ctx.logger.info("Also saved stats table as CSV: %s", csv_path.name)

    return actual_path


def attach_temperature_metadata_impl(
    df: pd.DataFrame,
    metadata_dict: Dict[str, Any],
    *,
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """Attach temperature control metadata to result DataFrame."""
    if df is None or df.empty:
        return df

    if "temperature_control" not in df.columns:
        df = df.copy()
        df["temperature_control"] = metadata_dict.get("temperature_control", None)
        df["temperature_control_used"] = metadata_dict.get("temperature_control_used", None)

        spline_meta = metadata_dict.get("temperature_spline", None)
        if isinstance(spline_meta, dict):
            df["temperature_spline_status"] = spline_meta.get("status", None)
            df["temperature_spline_n_knots"] = spline_meta.get("n_knots", None)
            df["temperature_spline_quantile_low"] = spline_meta.get("quantile_low", None)
            df["temperature_spline_quantile_high"] = spline_meta.get("quantile_high", None)

    if target_col and target_col in df.columns:
        ctrl_by_out = metadata_dict.get("temperature_control_by_outcome", None)
        if isinstance(ctrl_by_out, dict):
            used_map = {str(k): (v or {}).get("temperature_control_used", None) for k, v in ctrl_by_out.items()}
            status_map = {}
            nknots_map = {}
            for k, v in ctrl_by_out.items():
                s = (v or {}).get("temperature_spline", None)
                if isinstance(s, dict):
                    status_map[str(k)] = s.get("status", None)
                    nknots_map[str(k)] = s.get("n_knots", None)
            df["temperature_control_used"] = df[target_col].astype(str).map(used_map)
            df["temperature_spline_status"] = df[target_col].astype(str).map(status_map)
            df["temperature_spline_n_knots"] = df[target_col].astype(str).map(nknots_map)

    return df


def write_metadata_file_impl(path: Path, metadata: Dict[str, Any]) -> None:
    """Write metadata JSON file (fail-fast)."""
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
