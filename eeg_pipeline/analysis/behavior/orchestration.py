"""Behavior pipeline orchestration.

This module contains the implementation of the behavior pipeline stages.
The pipeline layer (`eeg_pipeline.pipelines.behavior`) should remain a thin wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.context.behavior import BehaviorContext
from eeg_pipeline.utils.analysis.stats.correlation import compute_correlation
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.infra.paths import ensure_dir


# Centralized feature column prefixes - matches FEATURE_TYPES from domain.features
FEATURE_COLUMN_PREFIXES = (
    "power_",
    "connectivity_",
    "aperiodic_",
    "erp_",
    "itpc_",
    "pac_",
    "complexity_",
    "bursts_",
    "quality_",
    "erds_",
    "spectral_",
    "ratios_",
    "asymmetry_",
    "temporal_",
)

CATEGORY_PREFIX_MAP = {
    "power": "power_",
    "connectivity": "connectivity_",
    "aperiodic": "aperiodic_",
    "erp": "erp_",
    "itpc": "itpc_",
    "pac": "pac_",
    "complexity": "complexity_",
    "bursts": "bursts_",
    "quality": "quality_",
    "erds": "erds_",
    "spectral": "spectral_",
    "ratios": "ratios_",
    "asymmetry": "asymmetry_",
    "temporal": "temporal_",
}


def _get_stats_subfolder(ctx: BehaviorContext, kind: str) -> Path:
    """Helper to get a subfolder within stats_dir and ensure it exists."""
    path = ctx.stats_dir / kind
    ensure_dir(path)
    return path


def _find_stats_path(ctx: BehaviorContext, filename: str) -> Optional[Path]:
    """Helper to find a file in stats_dir or its subfolders."""
    # 1. Check in specific subfolder if we can infer kind
    kind = _infer_output_kind(filename)
    if kind != "unknown":
        p = ctx.stats_dir / kind / filename
        if p.exists():
            return p
    
    # 2. Check in root stats_dir
    p = ctx.stats_dir / filename
    if p.exists():
        return p
    
    # 3. Search all subfolders
    for p in ctx.stats_dir.rglob(filename):
        if p.is_file():
            return p
            
    return None


def combine_features(ctx: BehaviorContext) -> pd.DataFrame:
    def _signature() -> str:
        parts = []
        for name, df in ctx.iter_feature_tables():
            if df is None or df.empty:
                continue
            col_blob = ",".join(str(c) for c in df.columns)
            parts.append(f"{name}:{df.shape[0]}:{df.shape[1]}:{col_blob}")
        payload = "|".join(parts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    signature = _signature()
    if ctx._combined_features_df is not None and ctx._combined_features_signature == signature:
        return ctx._combined_features_df

    dfs = []
    base_index = None
    for name, df in ctx.iter_feature_tables():
        if df is not None and not df.empty:
            if base_index is None:
                base_index = df.index
            elif not df.index.equals(base_index):
                msg = (
                    f"Feature index mismatch for {name}: expected alignment of "
                    f"{len(base_index)} rows."
                )
                ctx.logger.error(msg)
                raise ValueError(msg)
            prefix = f"{name}_"
            df = df.rename(columns={c: c if str(c).startswith(prefix) else f"{prefix}{c}" for c in df.columns})

            if df.columns.duplicated().any():
                dupes = [str(c) for c in df.columns[df.columns.duplicated()].unique()]
                msg = f"Duplicate feature columns within {name}: {dupes}"
                ctx.logger.error(msg)
                raise ValueError(msg)

            if dfs:
                existing_cols = pd.Index([])
                for prev in dfs:
                    existing_cols = existing_cols.append(prev.columns)
                overlap = existing_cols.intersection(df.columns)
                if not overlap.empty:
                    msg = f"Duplicate feature columns across tables: {[str(c) for c in overlap.tolist()]}"
                    ctx.logger.error(msg)
                    raise ValueError(msg)
            dfs.append(df)

    combined = pd.concat(dfs, axis=1) if dfs else pd.DataFrame()
    if not combined.empty and combined.columns.duplicated().any():
        dupes = [str(c) for c in combined.columns[combined.columns.duplicated()].unique()]
        msg = f"Duplicate feature columns after combining: {dupes}"
        ctx.logger.error(msg)
        raise ValueError(msg)
    ctx._combined_features_df = combined
    ctx._combined_features_signature = signature
    return combined


def add_change_scores(ctx: BehaviorContext) -> None:
    """Compute and append change scores (active-baseline) once per context."""
    if ctx._change_scores_added or not ctx.compute_change_scores:
        return

    from eeg_pipeline.utils.analysis.stats.correlation import compute_change_features

    def _augment(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return df
        change_df = compute_change_features(df)
        if change_df is None or change_df.empty:
            return df
        new_cols = [c for c in change_df.columns if c not in df.columns]
        return pd.concat([df, change_df[new_cols]], axis=1) if new_cols else df

    ctx.power_df = _augment(ctx.power_df)
    ctx.connectivity_df = _augment(ctx.connectivity_df)
    ctx.aperiodic_df = _augment(ctx.aperiodic_df)
    ctx.itpc_df = _augment(ctx.itpc_df)
    ctx.pac_df = _augment(ctx.pac_df)
    ctx.complexity_df = _augment(ctx.complexity_df)
    ctx._change_scores_added = True


def stage_load(ctx: BehaviorContext) -> bool:
    if not ctx.load_data():
        ctx.logger.warning("Failed to load data")
        return False

    ctx.logger.info(f"Loaded {ctx.n_trials} trials")
    return True


def stage_correlate(ctx: BehaviorContext, config: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Correlations + pain sensitivity from canonical trial table (trial-table-only)."""
    from eeg_pipeline.utils.analysis.stats.correlation import safe_correlation
    from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh
    from eeg_pipeline.analysis.behavior.api import run_pain_sensitivity_correlations
    from eeg_pipeline.utils.analysis.stats import compute_partial_correlations_with_cov_temp

    suffix = _feature_suffix_from_context(ctx)

    # Load trial table with all features
    df_trials = _load_trial_table_df(ctx)
    if df_trials is not None:
        ctx.logger.info("Correlations: loaded trial table (%d features)", df_trials.shape[1])

    if df_trials is None or df_trials.empty:
        ctx.logger.warning("Correlations: trial table missing; skipping.")
        return pd.DataFrame(), pd.DataFrame()

    feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    
    # Apply band filtering if user selected specific bands
    feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)
    feature_cols = _filter_feature_cols_for_computation(feature_cols, "correlations", ctx)
    
    if not feature_cols:
        return pd.DataFrame(), pd.DataFrame()

    method = getattr(config, "method", "spearman")
    robust_method = getattr(config, "robust_method", None)
    method_label = getattr(config, "method_label", "")

    # Targets: configurable; defaults cover pain + stimulus.
    default_targets = ["rating", "temperature", "pain_residual"]
    targets_cfg = get_config_value(ctx.config, "behavior_analysis.correlations.targets", None)
    if isinstance(targets_cfg, (list, tuple)) and targets_cfg:
        targets = [str(t).strip().lower() for t in targets_cfg]
    else:
        targets = default_targets
    targets = [t for t in targets if t in df_trials.columns]

    # Covariates for "trial-order" control: use trial index column only (lightweight + stable).
    cov_df = None
    if bool(getattr(config, "control_trial_order", True)):
        for c in ["trial_index_within_group", "trial_index"]:
            if c in df_trials.columns:
                cov_df = pd.DataFrame({c: pd.to_numeric(df_trials[c], errors="coerce")})
                break

    temperature_series = None
    if bool(getattr(config, "control_temperature", True)) and "temperature" in df_trials.columns:
        temperature_series = pd.to_numeric(df_trials["temperature"], errors="coerce")

    records: List[Dict[str, Any]] = []
    for target in targets:
        y = pd.to_numeric(df_trials[target], errors="coerce")
        for feat in feature_cols:
            x = pd.to_numeric(df_trials[feat], errors="coerce")

            # Raw correlation
            r_raw, p_raw, n = safe_correlation(
                x.to_numpy(dtype=float),
                y.to_numpy(dtype=float),
                method,
                int(getattr(config, "min_samples", 10)),
                robust_method=robust_method,
            )

            rec: Dict[str, Any] = {
                "feature": str(feat),
                "feature_type": _infer_feature_type(str(feat), ctx.config),
                "target": str(target),
                "method": method,
                "robust_method": robust_method,
                "method_label": method_label,
                "n": int(n),
                "r_raw": float(r_raw) if np.isfinite(r_raw) else np.nan,
                "p_raw": float(p_raw) if np.isfinite(p_raw) else np.nan,
                "r": float(r_raw) if np.isfinite(r_raw) else np.nan,
                "p": float(p_raw) if np.isfinite(p_raw) else np.nan,
                "p_value": float(p_raw) if np.isfinite(p_raw) else np.nan,
            }

            # Partial correlations (primary selection mirrors control flags).
            # Skip temperature control when the target itself is temperature.
            temp_for_partial = temperature_series if (temperature_series is not None and target != "temperature") else None
            (
                r_pc,
                p_pc,
                n_pc,
                r_pt,
                p_pt,
                n_pt,
                r_pct,
                p_pct,
                n_pct,
            ) = compute_partial_correlations_with_cov_temp(
                roi_values=x,
                target_values=y,
                covariates_df=cov_df,
                temperature_series=temp_for_partial,
                method=method,
                context="trial_table",
                logger=ctx.logger,
                min_samples=int(getattr(config, "min_samples", 10)),
                config=ctx.config,
            )
            rec.update(
                {
                    "r_partial_cov": r_pc,
                    "p_partial_cov": p_pc,
                    "n_partial_cov": n_pc,
                    "r_partial_temp": r_pt,
                    "p_partial_temp": p_pt,
                    "n_partial_temp": n_pt,
                    "r_partial_cov_temp": r_pct,
                    "p_partial_cov_temp": p_pct,
                    "n_partial_cov_temp": n_pct,
                }
            )

            # Primary selection
            p_kind = "p_raw"
            p_primary = rec["p_raw"]
            r_primary = rec["r_raw"]
            src = "raw"
            if bool(getattr(config, "control_temperature", True)) and bool(getattr(config, "control_trial_order", True)) and target != "temperature":
                if pd.notna(rec.get("p_partial_cov_temp", np.nan)):
                    p_kind = "p_partial_cov_temp"
                    p_primary = rec.get("p_partial_cov_temp", np.nan)
                    r_primary = rec.get("r_partial_cov_temp", np.nan)
                    src = "partial_cov_temp"
            elif bool(getattr(config, "control_temperature", True)) and target != "temperature":
                if pd.notna(rec.get("p_partial_temp", np.nan)):
                    p_kind = "p_partial_temp"
                    p_primary = rec.get("p_partial_temp", np.nan)
                    r_primary = rec.get("r_partial_temp", np.nan)
                    src = "partial_temp"
            elif bool(getattr(config, "control_trial_order", True)):
                if pd.notna(rec.get("p_partial_cov", np.nan)):
                    p_kind = "p_partial_cov"
                    p_primary = rec.get("p_partial_cov", np.nan)
                    r_primary = rec.get("r_partial_cov", np.nan)
                    src = "partial_cov"

            rec["p_kind_primary"] = p_kind
            rec["p_primary"] = p_primary
            rec["r_primary"] = r_primary
            rec["p_primary_source"] = src
            records.append(rec)

    corr_df = pd.DataFrame(records) if records else pd.DataFrame()
    if not corr_df.empty:
        p_for_fdr = pd.to_numeric(corr_df["p_primary"], errors="coerce").to_numpy()
        corr_df["p_fdr"] = fdr_bh(p_for_fdr, alpha=float(getattr(config, "fdr_alpha", 0.05)), config=ctx.config)

    psi_df = pd.DataFrame()
    if bool(getattr(config, "compute_pain_sensitivity", False)) and "temperature" in df_trials.columns and "rating" in df_trials.columns:
        robust_method_cfg = get_config_value(ctx.config, "behavior_analysis.robust_correlation", None)
        if robust_method_cfg is not None:
            robust_method_cfg = str(robust_method_cfg).strip().lower() or None
        
        psi_feature_cols = [c for c in df_trials.columns if str(c).startswith(prefixes)]
        psi_feature_cols = _filter_feature_cols_by_band(psi_feature_cols, ctx)
        psi_feature_cols = _filter_feature_cols_for_computation(psi_feature_cols, "pain_sensitivity", ctx)
        ctx.logger.info("Pain sensitivity: analyzing %d features", len(psi_feature_cols))
        psi_features = df_trials[psi_feature_cols].copy()
        psi_df = run_pain_sensitivity_correlations(
            features_df=psi_features,
            ratings=pd.to_numeric(df_trials["rating"], errors="coerce"),
            temperatures=pd.to_numeric(df_trials["temperature"], errors="coerce"),
            method=method,
            robust_method=robust_method_cfg,
            min_samples=int(getattr(config, "min_samples", 10)),
            logger=ctx.logger,
            config=ctx.config,
        )

    return corr_df, psi_df


def _feature_suffix_from_context(ctx: BehaviorContext) -> str:
    feature_files = ctx.selected_feature_files or ctx.feature_categories or []
    return "_" + "_".join(sorted(str(x) for x in feature_files)) if feature_files else ""


def _filter_feature_cols_by_band(
    feature_cols: List[str],
    ctx: BehaviorContext,
) -> List[str]:
    """Filter feature columns to only include user-selected bands.
    
    If ctx.selected_bands is None or empty, returns all feature_cols unchanged.
    Otherwise, filters to keep only columns whose parsed 'band' matches one of
    the selected bands. Columns that don't have a band (e.g., ERP, complexity)
    are kept by default.
    """
    if not ctx.selected_bands:
        return feature_cols
    
    from eeg_pipeline.domain.features.naming import NamingSchema
    
    selected = set(b.lower() for b in ctx.selected_bands)
    filtered: List[str] = []
    
    for col in feature_cols:
        try:
            parsed = NamingSchema.parse(str(col))
            if not parsed.get("valid"):
                # Keep unparseable columns (might be summary or derived features)
                filtered.append(col)
                continue
            
            band = parsed.get("band")
            if not band:
                # Features without bands (ERP, complexity, etc.) - keep them
                filtered.append(col)
            elif str(band).lower() in selected:
                # Band matches user selection
                filtered.append(col)
            # else: band doesn't match, exclude
        except Exception:
            # On parse error, keep the column
            filtered.append(col)
    
    if len(filtered) < len(feature_cols):
        ctx.logger.info(
            "Band filter: kept %d/%d features for bands: %s",
            len(filtered),
            len(feature_cols),
            ", ".join(sorted(selected)),
        )
    
    return filtered


def _filter_feature_cols_for_computation(
    feature_cols: List[str],
    computation_name: str,
    ctx: BehaviorContext,
) -> List[str]:
    """Filter feature columns based on per-computation feature selection."""
    if not ctx.computation_features or computation_name not in ctx.computation_features:
        return feature_cols
    
    selected_features = ctx.computation_features[computation_name]
    if not selected_features:
        return feature_cols

    allowed_prefixes = tuple(
        CATEGORY_PREFIX_MAP[cat] for cat in selected_features if cat in CATEGORY_PREFIX_MAP
    )
    if not allowed_prefixes:
        ctx.logger.warning(
            f"Computation '{computation_name}' has feature filter {selected_features} but no matching prefixes found. Using all features."
        )
        return feature_cols

    filtered = [c for c in feature_cols if str(c).startswith(allowed_prefixes)]
    
    if len(filtered) < len(feature_cols):
        ctx.logger.info(
            "%s: filtered features to %s (%d/%d kept)",
            computation_name, selected_features, len(filtered), len(feature_cols)
        )
        
    return filtered


def stage_trial_table(ctx: BehaviorContext, config: Any) -> Optional[Path]:
    """Build and save the canonical per-trial analysis table for this subject."""
    from eeg_pipeline.utils.data.trial_table import (
        build_subject_trial_table,
        save_trial_table,
        add_lag_and_delta_features,
        add_pain_residual,
    )
    from eeg_pipeline.infra.tsv import write_tsv

    include_features = bool(get_config_value(ctx.config, "behavior_analysis.trial_table.include_features", True))
    include_covariates = bool(get_config_value(ctx.config, "behavior_analysis.trial_table.include_covariates", True))
    include_events = bool(get_config_value(ctx.config, "behavior_analysis.trial_table.include_events", True))
    fmt = str(get_config_value(ctx.config, "behavior_analysis.trial_table.format", "tsv")).strip().lower()

    extra_cols = get_config_value(ctx.config, "behavior_analysis.trial_table.extra_event_columns", None)
    extra_cols_list = [str(c) for c in extra_cols] if isinstance(extra_cols, (list, tuple)) else None

    result = build_subject_trial_table(
        ctx,
        include_features=include_features,
        include_covariates=include_covariates,
        include_events=include_events,
        extra_event_columns=extra_cols_list,
    )

    # Augment with lag/delta covariates for habituation/dynamics analyses.
    lag_enabled = bool(get_config_value(ctx.config, "behavior_analysis.trial_table.add_lag_features", True))
    if lag_enabled:
        result.df, lag_meta = add_lag_and_delta_features(result.df)
        result.metadata["lag_features"] = lag_meta

    # Add pain residual = rating - f(temperature) for "pain beyond stimulus intensity".
    result.df, resid_meta = add_pain_residual(result.df, ctx.config)
    result.metadata["pain_residual"] = resid_meta

    suffix = _feature_suffix_from_context(ctx)
    fname = f"trials{suffix}"
    out_dir = _get_stats_subfolder(ctx, "trial_table")
    out_path = out_dir / f"{fname}.tsv"
    save_trial_table(result, out_path, format=fmt)

    meta_path = out_dir / f"{fname}.metadata.json"
    meta_path.write_text(json.dumps(result.metadata, indent=2, default=str))
    ctx.logger.info("Saved trial table: %s/%s (%s rows, %s cols)", out_dir.name, out_path.name, len(result.df), result.df.shape[1])

    # Save temperature→rating nonlinearity diagnostics (subject-level; non-gating).
    try:
        mc_enabled = bool(get_config_value(ctx.config, "behavior_analysis.pain_residual.model_comparison.enabled", True))
        bp_enabled = bool(get_config_value(ctx.config, "behavior_analysis.pain_residual.breakpoint_test.enabled", True))
        if mc_enabled and "temperature" in result.df.columns and "rating" in result.df.columns:
            from eeg_pipeline.utils.analysis.stats.temperature_models import compare_temperature_rating_models

            df_cmp, meta_cmp = compare_temperature_rating_models(
                result.df["temperature"], result.df["rating"], config=ctx.config
            )
            if df_cmp is not None and not df_cmp.empty:
                write_tsv(df_cmp, out_dir / f"temperature_model_comparison{suffix}.tsv")
            (out_dir / f"temperature_model_comparison{suffix}.metadata.json").write_text(
                json.dumps(meta_cmp, indent=2, default=str)
            )
            ctx.data_qc["temperature_model_comparison"] = meta_cmp

        if bp_enabled and "temperature" in result.df.columns and "rating" in result.df.columns:
            from eeg_pipeline.utils.analysis.stats.temperature_models import fit_temperature_breakpoint_test

            df_bp, meta_bp = fit_temperature_breakpoint_test(
                result.df["temperature"], result.df["rating"], config=ctx.config
            )
            if df_bp is not None and not df_bp.empty:
                write_tsv(df_bp, out_dir / f"temperature_breakpoint_candidates{suffix}.tsv")
            (out_dir / f"temperature_breakpoint_test{suffix}.metadata.json").write_text(
                json.dumps(meta_bp, indent=2, default=str)
            )
            ctx.data_qc["temperature_breakpoint_test"] = meta_bp
    except Exception as exc:
        ctx.logger.debug("Temperature model diagnostics failed: %s", exc)

    return out_path


def _load_trial_table_df(ctx: BehaviorContext) -> Optional[pd.DataFrame]:
    from eeg_pipeline.infra.tsv import read_table
    suffix = _feature_suffix_from_context(ctx)
    # Prefer subfolder, fallback to root
    fnames = [f"trials{suffix}.tsv", f"trials{suffix}.parquet"]
    for fname in fnames:
        p = _find_stats_path(ctx, fname)
        if p:
            return read_table(p)

    # Fallback: any trials tsv/parquet
    candidates = sorted(ctx.stats_dir.rglob("trials*.tsv")) or sorted(ctx.stats_dir.rglob("trials*.parquet"))
    if candidates:
        return read_table(candidates[0])
    return None


def stage_trial_table_validate(ctx: BehaviorContext, config: Any) -> Dict[str, Any]:
    """Validate the canonical trial table (non-gating) and write a contract report."""
    from eeg_pipeline.utils.data.trial_table_validation import validate_trial_table
    from eeg_pipeline.infra.tsv import write_tsv

    enabled = bool(get_config_value(ctx.config, "behavior_analysis.trial_table.validate.enabled", True))
    if not enabled:
        return {"enabled": False, "status": "disabled"}

    df = _load_trial_table_df(ctx)
    if df is None or df.empty:
        return {"enabled": True, "status": "missing_trial_table"}

    suffix = _feature_suffix_from_context(ctx)
    result = validate_trial_table(df, config=ctx.config)

    out_dir = _get_stats_subfolder(ctx, "trial_table_validation")
    summary_path = out_dir / f"trial_table_validation_summary{suffix}.tsv"
    report_path = out_dir / f"trial_table_validation{suffix}.json"
    try:
        if result.summary_df is not None and not result.summary_df.empty:
            write_tsv(result.summary_df, summary_path)
        report_path.write_text(json.dumps(result.report, indent=2, default=str))
    except Exception as exc:
        ctx.logger.warning("Trial table validation write failed: %s", exc)

    ctx.data_qc["trial_table_validation"] = {
        "enabled": True,
        "status": result.report.get("status", "unknown"),
        "warnings": result.report.get("warnings", []),
        "summary_tsv": summary_path.name,
        "report_json": report_path.name,
    }
    if result.report.get("warnings"):
        ctx.logger.info("Trial table validation warnings: %s", "; ".join(result.report["warnings"][:10]))
    return result.report


def stage_confounds(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Audit QC confounds; optionally add selected QC covariates for downstream analyses."""
    from eeg_pipeline.utils.analysis.stats.confounds import (
        audit_qc_confounds,
        select_significant_qc_covariates,
    )
    from eeg_pipeline.infra.tsv import write_tsv
    from eeg_pipeline.utils.formatting import sanitize_label

    df_trials = _load_trial_table_df(ctx)
    if df_trials is None or df_trials.empty:
        ctx.logger.warning("Confounds: trial table missing; skipping.")
        return pd.DataFrame()

    # Use the correlation method label for file naming consistency.
    method = config.method
    robust_method = config.robust_method
    method_label = config.method_label

    min_samples = int(get_config_value(ctx.config, "behavior_analysis.min_samples.default", 10))
    audit_df, audit_meta = audit_qc_confounds(
        df_trials,
        config=ctx.config,
        targets=["rating", "temperature"],
        method=method,
        robust_method=robust_method,
        min_samples=min_samples,
    )

    suffix = _feature_suffix_from_context(ctx)
    method_suffix = f"_{method_label}" if method_label else ""
    out_dir = _get_stats_subfolder(ctx, "confounds_audit")
    out_path = out_dir / f"confounds_audit{suffix}{method_suffix}.tsv"
    if not audit_df.empty:
        write_tsv(audit_df, out_path)
        ctx.logger.info("Confounds audit saved: %s/%s (%d rows)", out_dir.name, out_path.name, len(audit_df))

    ctx.data_qc["confounds_audit"] = audit_meta

    # Optionally add QC covariates (selected by FDR) to ctx.covariates_df.
    add_cov = bool(get_config_value(ctx.config, "behavior_analysis.confounds.add_as_covariates_if_significant", False))
    if not add_cov or audit_df.empty:
        return audit_df

    alpha = float(get_config_value(ctx.config, "behavior_analysis.statistics.fdr_alpha", 0.05))
    max_cov = int(get_config_value(ctx.config, "behavior_analysis.confounds.max_qc_covariates", 3))
    qc_cols = select_significant_qc_covariates(
        audit_df,
        config=ctx.config,
        alpha=alpha,
        max_covariates=max_cov,
        prefer_target="rating",
    )
    if not qc_cols:
        return audit_df

    cov_add = pd.DataFrame(index=np.arange(len(df_trials)))
    for col in qc_cols:
        safe_name = sanitize_label(f"qc_{col}")
        if safe_name in cov_add.columns:
            continue
        cov_add[safe_name] = pd.to_numeric(df_trials[col], errors="coerce")

    if cov_add.empty:
        return audit_df

    # Align to ctx indices (0..n-1), then attach to ctx.covariates_df.
    if ctx.covariates_df is None or ctx.covariates_df.empty:
        ctx.covariates_df = pd.DataFrame(index=ctx.aligned_events.index if ctx.aligned_events is not None else None)

    cov_add.index = ctx.covariates_df.index
    for c in cov_add.columns:
        if c not in ctx.covariates_df.columns:
            ctx.covariates_df[c] = cov_add[c].to_numpy()

    from eeg_pipeline.utils.data.covariates import build_covariates_without_temp

    ctx.covariates_without_temp_df = build_covariates_without_temp(ctx.covariates_df, ctx.temperature_column)
    ctx.data_qc["confounds_qc_covariates_added"] = {
        "columns": list(cov_add.columns),
        "source_metrics": qc_cols,
        "alpha": alpha,
    }
    ctx.logger.info("Added QC covariates: %s", ", ".join(cov_add.columns))

    return audit_df


def stage_regression(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Trialwise regression: rating (or pain_residual) ~ temperature + trial order + feature (+ interaction)."""
    from eeg_pipeline.utils.analysis.stats.trialwise_regression import run_trialwise_feature_regressions
    from eeg_pipeline.infra.tsv import write_tsv

    suffix = _feature_suffix_from_context(ctx)
    method_label = config.method_label
    method_suffix = f"_{method_label}" if method_label else ""

    df_trials = _load_trial_table_df(ctx)
    if df_trials is None or df_trials.empty:
        ctx.logger.warning("Regression: trial table missing; skipping.")
        return pd.DataFrame()

    # Feature columns: use naming schema prefixes for safety.
    feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)

    groups = None
    if getattr(ctx, "group_ids", None) is not None:
        try:
            groups = np.asarray(ctx.group_ids)
        except Exception:
            groups = None

    reg_df, reg_meta = run_trialwise_feature_regressions(
        df_trials,
        feature_cols=feature_cols,
        config=ctx.config,
        groups_for_permutation=groups,
    )
    ctx.data_qc["trialwise_regression"] = reg_meta
    if reg_df is not None and not reg_df.empty and "temperature_control" not in reg_df.columns:
        reg_df = reg_df.copy()
        reg_df["temperature_control"] = reg_meta.get("temperature_control", None)
        reg_df["temperature_control_used"] = reg_meta.get("temperature_control_used", None)
        spline_meta = reg_meta.get("temperature_spline", None)
        if isinstance(spline_meta, dict):
            reg_df["temperature_spline_status"] = spline_meta.get("status", None)
            reg_df["temperature_spline_n_knots"] = spline_meta.get("n_knots", None)
            reg_df["temperature_spline_quantile_low"] = spline_meta.get("quantile_low", None)
            reg_df["temperature_spline_quantile_high"] = spline_meta.get("quantile_high", None)

    out_dir = _get_stats_subfolder(ctx, "trialwise_regression")
    out_path = out_dir / f"regression_feature_effects{suffix}{method_suffix}.tsv"
    if not reg_df.empty:
        write_tsv(reg_df, out_path)
        ctx.logger.info("Regression results saved: %s/%s (%d features)", out_dir.name, out_path.name, len(reg_df))
    return reg_df


def stage_models(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Fit multiple model families per feature (OLS-HC3 / robust / quantile / logistic)."""
    from eeg_pipeline.utils.analysis.stats.feature_models import run_feature_model_families
    from eeg_pipeline.infra.tsv import write_tsv

    suffix = _feature_suffix_from_context(ctx)
    method_label = getattr(config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    df_trials = _load_trial_table_df(ctx)
    if df_trials is None or df_trials.empty:
        ctx.logger.warning("Models: trial table missing; skipping.")
        return pd.DataFrame()

    feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)

    model_df, model_meta = run_feature_model_families(
        df_trials,
        feature_cols=feature_cols,
        config=ctx.config,
    )
    ctx.data_qc["feature_models"] = model_meta
    if model_df is not None and not model_df.empty and "temperature_control" not in model_df.columns:
        model_df = model_df.copy()
        model_df["temperature_control"] = model_meta.get("temperature_control", None)
        # Optional per-target (outcome) temperature control diagnostics.
        ctrl_by_out = model_meta.get("temperature_control_by_outcome", None)
        if isinstance(ctrl_by_out, dict) and "target" in model_df.columns:
            used_map = {str(k): (v or {}).get("temperature_control_used", None) for k, v in ctrl_by_out.items()}
            status_map = {}
            nknots_map = {}
            for k, v in ctrl_by_out.items():
                s = (v or {}).get("temperature_spline", None)
                if isinstance(s, dict):
                    status_map[str(k)] = s.get("status", None)
                    nknots_map[str(k)] = s.get("n_knots", None)
            model_df["temperature_control_used"] = model_df["target"].astype(str).map(used_map)
            model_df["temperature_spline_status"] = model_df["target"].astype(str).map(status_map)
            model_df["temperature_spline_n_knots"] = model_df["target"].astype(str).map(nknots_map)

    out_dir = _get_stats_subfolder(ctx, "feature_models")
    out_path = out_dir / f"models_feature_effects{suffix}{method_suffix}.tsv"
    if model_df is not None and not model_df.empty:
        write_tsv(model_df, out_path)
        ctx.logger.info("Model families results saved: %s/%s (%d rows)", out_dir.name, out_path.name, len(model_df))
    return model_df if model_df is not None else pd.DataFrame()


def stage_stability(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Assess within-subject run/block stability of feature→outcome associations (non-gating)."""
    from eeg_pipeline.utils.analysis.stats.stability import compute_groupwise_stability
    from eeg_pipeline.infra.tsv import write_tsv

    suffix = _feature_suffix_from_context(ctx)
    method_label = getattr(config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    df_trials = _load_trial_table_df(ctx)
    if df_trials is None or df_trials.empty:
        ctx.logger.warning("Stability: trial table missing; skipping.")
        return pd.DataFrame()

    group_col = str(get_config_value(ctx.config, "behavior_analysis.stability.group_column", "")).strip()
    if not group_col:
        group_col = "run" if "run" in df_trials.columns else ("block" if "block" in df_trials.columns else "")
    if not group_col:
        ctx.logger.info("Stability: no run/block column available; skipping.")
        return pd.DataFrame()

    outcome = str(get_config_value(ctx.config, "behavior_analysis.stability.outcome", "")).strip().lower()
    if not outcome:
        outcome = "pain_residual" if "pain_residual" in df_trials.columns else "rating"

    feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)

    stab_df, stab_meta = compute_groupwise_stability(
        df_trials,
        feature_cols=feature_cols,
        outcome=outcome,
        group_col=group_col,
        config=ctx.config,
    )
    ctx.data_qc["stability_groupwise"] = stab_meta

    out_dir = _get_stats_subfolder(ctx, "stability_groupwise")
    out_path = out_dir / f"stability_groupwise{suffix}{method_suffix}.tsv"
    if stab_df is not None and not stab_df.empty:
        write_tsv(stab_df, out_path)
        ctx.logger.info("Stability results saved: %s/%s (%d features)", out_dir.name, out_path.name, len(stab_df))
    try:
        (out_dir / f"stability_groupwise{suffix}{method_suffix}.metadata.json").write_text(
            json.dumps(stab_meta, indent=2, default=str)
        )
    except Exception:
        pass
    return stab_df if stab_df is not None else pd.DataFrame()


def stage_consistency(ctx: BehaviorContext, config: Any, results: Any) -> pd.DataFrame:
    """Merge correlations/regression/models and flag effect-direction contradictions (non-gating)."""
    from eeg_pipeline.utils.analysis.stats.consistency import build_effect_direction_consistency_summary
    from eeg_pipeline.infra.tsv import write_tsv

    suffix = _feature_suffix_from_context(ctx)
    method_label = getattr(config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    corr_df = getattr(results, "correlations", None)
    reg_df = getattr(results, "regression", None)
    models_df = getattr(results, "models", None)
    out_df, meta = build_effect_direction_consistency_summary(
        corr_df=corr_df,
        regression_df=reg_df,
        models_df=models_df,
        config=ctx.config,
    )
    ctx.data_qc["effect_direction_consistency"] = meta
    if out_df is None or out_df.empty:
        return pd.DataFrame()

    out_dir = _get_stats_subfolder(ctx, "consistency_summary")
    out_path = out_dir / f"consistency_summary{suffix}{method_suffix}.tsv"
    write_tsv(out_df, out_path)
    try:
        (out_dir / f"consistency_summary{suffix}{method_suffix}.metadata.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )
    except Exception:
        pass
    ctx.logger.info("Consistency summary saved: %s (%d features)", out_path.name, len(out_df))
    return out_df


def stage_influence(ctx: BehaviorContext, config: Any, results: Any) -> pd.DataFrame:
    """Compute leverage/Cook's summaries for top effects (non-gating)."""
    from eeg_pipeline.utils.analysis.stats.influence import compute_influence_diagnostics
    from eeg_pipeline.infra.tsv import write_tsv

    suffix = _feature_suffix_from_context(ctx)
    method_label = getattr(config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    df_trials = _load_trial_table_df(ctx)
    if df_trials is None or df_trials.empty:
        ctx.logger.info("Influence: trial table missing; skipping.")
        return pd.DataFrame()

    feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)

    out_df, meta = compute_influence_diagnostics(
        df_trials,
        feature_cols=feature_cols,
        corr_df=getattr(results, "correlations", None),
        regression_df=getattr(results, "regression", None),
        models_df=getattr(results, "models", None),
        config=ctx.config,
    )
    ctx.data_qc["influence_diagnostics"] = meta
    if out_df is None or out_df.empty:
        return pd.DataFrame()
    # Attach high-level temperature-control info (useful when spline/rating_hat are enabled).
    try:
        if "temperature_control" not in out_df.columns:
            out_df = out_df.copy()
            out_df["temperature_control"] = get_config_value(ctx.config, "behavior_analysis.influence.temperature_control", None)
        ctrl_by_out = meta.get("temperature_control_by_outcome", None) if isinstance(meta, dict) else None
        if isinstance(ctrl_by_out, dict) and "outcome" in out_df.columns:
            used_map = {str(k): (v or {}).get("temperature_control_used", None) for k, v in ctrl_by_out.items()}
            out_df["temperature_control_used"] = out_df["outcome"].astype(str).map(used_map)
    except Exception:
        pass

    out_dir = _get_stats_subfolder(ctx, "influence_diagnostics")
    out_path = out_dir / f"influence_diagnostics{suffix}{method_suffix}.tsv"
    write_tsv(out_df, out_path)
    try:
        (out_dir / f"influence_diagnostics{suffix}{method_suffix}.metadata.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )
    except Exception:
        pass
    ctx.logger.info("Influence diagnostics saved: %s/%s (%d rows)", out_dir.name, out_path.name, len(out_df))
    return out_df


def build_behavior_qc(ctx: BehaviorContext) -> Dict[str, Any]:
    qc: Dict[str, Any] = {
        "subject": ctx.subject,
        "task": ctx.task,
        "n_trials": ctx.n_trials,
        "has_temperature": ctx.has_temperature,
        "temperature_column": ctx.temperature_column,
        "group_column": ctx.group_column,
    }

    if ctx.data_qc:
        qc["data_qc"] = ctx.data_qc

    if ctx.targets is not None:
        s = pd.to_numeric(ctx.targets, errors="coerce")
        qc["rating"] = {
            "n_non_nan": int(s.notna().sum()),
            "min": float(s.min()) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan,
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "std": float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan,
        }

    if ctx.temperature is not None:
        t = pd.to_numeric(ctx.temperature, errors="coerce")
        qc["temperature"] = {
            "n_non_nan": int(t.notna().sum()),
            "min": float(t.min()) if t.notna().any() else np.nan,
            "max": float(t.max()) if t.notna().any() else np.nan,
            "mean": float(t.mean()) if t.notna().any() else np.nan,
            "std": float(t.std(ddof=1)) if t.notna().sum() > 1 else np.nan,
        }

    if ctx.targets is not None and ctx.temperature is not None:
        s = pd.to_numeric(ctx.targets, errors="coerce")
        t = pd.to_numeric(ctx.temperature, errors="coerce")
        valid = s.notna() & t.notna()
        if int(valid.sum()) >= 3:
            r, p = compute_correlation(s[valid].values, t[valid].values, method="spearman")
            qc["rating_temperature_sanity"] = {
                "method": "spearman",
                "n": int(valid.sum()),
                "r": float(r) if np.isfinite(r) else np.nan,
                "p": float(p) if np.isfinite(p) else np.nan,
            }

    if ctx.aligned_events is not None and ctx.targets is not None:
        from eeg_pipeline.analysis.behavior.api import split_by_condition

        try:
            pain_mask, nonpain_mask, n_pain, n_nonpain = split_by_condition(
                ctx.aligned_events, ctx.config, ctx.logger
            )
        except Exception as exc:
            qc["pain_vs_nonpain"] = {
                "status": "failed",
                "error": str(exc),
            }
        else:
            if int(n_pain) > 0 or int(n_nonpain) > 0:
                s = pd.to_numeric(ctx.targets, errors="coerce")
                pain_ratings = s[pain_mask] if len(pain_mask) == len(s) else pd.Series(dtype=float)
                nonpain_ratings = s[nonpain_mask] if len(nonpain_mask) == len(s) else pd.Series(dtype=float)
                qc["pain_vs_nonpain"] = {
                    "status": "ok",
                    "n_pain": int(n_pain),
                    "n_nonpain": int(n_nonpain),
                    "mean_rating_pain": float(pain_ratings.mean()) if pain_ratings.notna().any() else np.nan,
                    "mean_rating_nonpain": float(nonpain_ratings.mean()) if nonpain_ratings.notna().any() else np.nan,
                    "mean_rating_difference_pain_minus_nonpain": (
                        float(pain_ratings.mean() - nonpain_ratings.mean())
                        if pain_ratings.notna().any() and nonpain_ratings.notna().any()
                        else np.nan
                    ),
                }

    if ctx.covariates_df is not None and not ctx.covariates_df.empty:
        cov = ctx.covariates_df
        qc["covariates"] = {
            "n_covariates": int(cov.shape[1]),
            "columns": [str(c) for c in cov.columns],
            "missing_fraction_by_column": {
                str(c): float(pd.to_numeric(cov[c], errors="coerce").isna().mean()) for c in cov.columns
            },
        }

    feature_counts: Dict[str, int] = {}
    for name, df in ctx.iter_feature_tables():
        if df is None or df.empty:
            continue
        feature_counts[name] = int(df.shape[1])
    qc["feature_counts"] = feature_counts

    return qc


def _ensure_method_columns(
    df: pd.DataFrame,
    method: str,
    robust_method: Optional[str],
    method_label: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    if "method" not in df.columns:
        df["method"] = method
    if "robust_method" not in df.columns:
        df["robust_method"] = robust_method
    if "method_label" not in df.columns:
        df["method_label"] = method_label
    return df


_STANDARD_COLUMNS = (
    "analysis_type",
    "feature_id",
    "feature_type",
    "target",
    "method",
    "robust_method",
    "method_label",
    "n",
    "r",
    "p_raw",
    "p_primary",
    "p_fdr",
    "notes",
)


def _infer_feature_type(feature: str, config: Any) -> str:
    from eeg_pipeline.domain.features.registry import classify_feature, get_feature_registry
    try:
        registry = get_feature_registry(config)
        ftype, _, _ = classify_feature(feature, include_subtype=False, registry=registry)
        return ftype
    except Exception:
        return "unknown"


def _build_normalized_records(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Any,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    method = pipeline_config.method
    robust_method = pipeline_config.robust_method
    method_label = pipeline_config.method_label

    corr_df = getattr(results, "correlations", None)
    if corr_df is not None and not corr_df.empty:
        for _, row in corr_df.iterrows():
            feature = row.get("feature")
            feature_type = row.get("feature_type") or _infer_feature_type(str(feature), ctx.config)
            target = row.get("target", "rating")
            p_raw = row.get("p_raw", row.get("p_value", row.get("p")))
            p_primary = row.get("p_primary", p_raw)
            p_fdr = row.get("p_fdr", row.get("q_value"))
            r_val = row.get("r_primary", row.get("r"))
            records.append({
                "analysis_type": "correlations",
                "feature_id": feature,
                "feature_type": feature_type,
                "target": target,
                "method": row.get("method", method),
                "robust_method": row.get("robust_method", robust_method),
                "method_label": row.get("method_label", method_label),
                "n": row.get("n"),
                "r": r_val,
                "p_raw": p_raw,
                "p_primary": p_primary,
                "p_fdr": p_fdr,
                "notes": None,
            })

    psi_df = getattr(results, "pain_sensitivity", None)
    if psi_df is not None and not psi_df.empty:
        for _, row in psi_df.iterrows():
            feature = row.get("feature")
            feature_type = row.get("feature_type") or _infer_feature_type(str(feature), ctx.config)
            p_raw = row.get("p_psi", row.get("p_value"))
            r_val = row.get("r_psi", row.get("r"))
            records.append({
                "analysis_type": "pain_sensitivity",
                "feature_id": feature,
                "feature_type": feature_type,
                "target": row.get("target", "pain_sensitivity"),
                "method": row.get("method", method),
                "robust_method": row.get("robust_method", robust_method),
                "method_label": row.get("method_label", method_label),
                "n": row.get("n"),
                "r": r_val,
                "p_raw": p_raw,
                "p_primary": p_raw,
                "p_fdr": row.get("p_fdr", row.get("q_value")),
                "notes": None,
            })

    cond_df = getattr(results, "condition_effects", None)
    if cond_df is not None and not cond_df.empty:
        for _, row in cond_df.iterrows():
            feature = row.get("feature")
            feature_type = row.get("feature_type") or _infer_feature_type(str(feature), ctx.config)
            n_pain = row.get("n_pain")
            n_nonpain = row.get("n_nonpain")
            n_total = None
            if pd.notna(n_pain) and pd.notna(n_nonpain):
                n_total = int(n_pain) + int(n_nonpain)
            records.append({
                "analysis_type": "condition_effects",
                "feature_id": feature,
                "feature_type": feature_type,
                "target": "pain_vs_nonpain",
                "method": row.get("method", method),
                "robust_method": row.get("robust_method", robust_method),
                "method_label": row.get("method_label", method_label),
                "n": n_total,
                "r": row.get("hedges_g"),
                "p_raw": row.get("p_value"),
                "p_primary": row.get("p_value"),
                "p_fdr": row.get("q_value"),
                "notes": "hedges_g",
            })

    med_df = getattr(results, "mediation", None)
    if med_df is not None and not med_df.empty:
        for _, row in med_df.iterrows():
            feature = row.get("mediator")
            records.append({
                "analysis_type": "mediation",
                "feature_id": feature,
                "feature_type": "mediator",
                "target": "rating",
                "method": method,
                "robust_method": robust_method,
                "method_label": method_label,
                "n": None,
                "r": row.get("indirect_effect"),
                "p_raw": row.get("sobel_p"),
                "p_primary": row.get("sobel_p"),
                "p_fdr": None,
                "notes": "indirect_effect",
            })

    mixed_df = getattr(results, "mixed_effects", None)
    if mixed_df is not None and not mixed_df.empty:
        for _, row in mixed_df.iterrows():
            feature = row.get("feature")
            behavior = row.get("behavior", "rating")
            records.append({
                "analysis_type": "mixed_effects",
                "feature_id": feature,
                "feature_type": row.get("feature_type", "mixed_effects"),
                "target": behavior,
                "method": method,
                "robust_method": robust_method,
                "method_label": method_label,
                "n": row.get("n_observations"),
                "r": row.get("fixed_effect"),
                "p_raw": row.get("fixed_p"),
                "p_primary": row.get("fixed_p"),
                "p_fdr": row.get("fixed_p_fdr"),
                "notes": "fixed_effect",
            })

    return records


def _write_normalized_results(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Any,
) -> Optional[Path]:
    records = _build_normalized_records(ctx, pipeline_config, results)
    if not records:
        return None
    df = pd.DataFrame(records)
    df = df.reindex(columns=_STANDARD_COLUMNS)
    
    # Build feature suffix from selected feature files or categories
    feature_files = ctx.selected_feature_files or ctx.feature_categories or []
    feature_suffix = "_" + "_".join(sorted(feature_files)) if feature_files else ""
    
    path = ctx.stats_dir / f"normalized_results{feature_suffix}.tsv"
    from eeg_pipeline.infra.tsv import write_tsv
    write_tsv(df, path)
    return path


def _summarize_covariates_qc(ctx: BehaviorContext) -> Dict[str, Any]:
    cov = ctx.covariates_df
    if cov is None or cov.empty:
        return {"status": "empty"}
    return {
        "status": "ok",
        "columns": [str(c) for c in cov.columns],
        "missing_fraction_by_column": {
            str(c): float(pd.to_numeric(cov[c], errors="coerce").isna().mean()) for c in cov.columns
        },
    }


def write_analysis_metadata(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Any,
    stage_metrics: Optional[Dict[str, Any]] = None,
    outputs_manifest: Optional[Path] = None,
) -> None:
    robust_method = pipeline_config.robust_method
    method_label = pipeline_config.method_label
    payload: Dict[str, Any] = {
        "subject": ctx.subject,
        "task": ctx.task,
        "method": pipeline_config.method,
        "method_label": method_label,
        "robust_method": robust_method,
        "min_samples": pipeline_config.min_samples,
        "control_temperature": pipeline_config.control_temperature,
        "control_trial_order": pipeline_config.control_trial_order,
        "compute_change_scores": pipeline_config.compute_change_scores,
        "compute_pain_sensitivity": pipeline_config.compute_pain_sensitivity,
        "compute_reliability": pipeline_config.compute_reliability,
        "n_permutations": pipeline_config.n_permutations,
        "fdr_alpha": pipeline_config.fdr_alpha,
        "n_trials": ctx.n_trials,
        "statistics_config": {
            "method": pipeline_config.method,
            "robust_method": robust_method,
            "method_label": method_label,
            "min_samples": pipeline_config.min_samples,
            "bootstrap": pipeline_config.bootstrap,
            "n_permutations": pipeline_config.n_permutations,
            "fdr_alpha": pipeline_config.fdr_alpha,
            "control_temperature": pipeline_config.control_temperature,
            "control_trial_order": pipeline_config.control_trial_order,
            "compute_change_scores": pipeline_config.compute_change_scores,
            "compute_reliability": pipeline_config.compute_reliability,
            "compute_bayes_factors": getattr(pipeline_config, "compute_bayes_factors", False),
            "compute_loso_stability": getattr(pipeline_config, "compute_loso_stability", False),
        },
        "outputs": {
            "has_trial_table": bool(getattr(results, "trial_table_path", None)),
            "has_trial_table_validation": bool(getattr(results, "trial_table_validation", None)),
            "has_confounds_audit": bool(getattr(results, "confounds", None) is not None and not results.confounds.empty),
            "has_regression": bool(getattr(results, "regression", None) is not None and not results.regression.empty),
            "has_stability": bool(getattr(results, "stability", None) is not None and not getattr(results, "stability").empty) if getattr(results, "stability", None) is not None else False,
            "has_models": bool(getattr(results, "models", None) is not None and not getattr(results, "models").empty) if getattr(results, "models", None) is not None else False,
            "has_correlations": bool(getattr(results, "correlations", None) is not None and not results.correlations.empty),
            "has_pain_sensitivity": bool(getattr(results, "pain_sensitivity", None) is not None and not results.pain_sensitivity.empty),
            "has_condition_effects": bool(getattr(results, "condition_effects", None) is not None and not results.condition_effects.empty),
        },
        "qc": build_behavior_qc(ctx),
    }

    payload["temperature_status"] = {
        "available": bool(ctx.temperature is not None and ctx.temperature.notna().any()) if ctx.temperature is not None else False,
        "control_enabled": bool(ctx.control_temperature),
    }
    if not payload["temperature_status"]["available"]:
        payload["temperature_status"]["reason"] = "missing_temperature"

    if not pipeline_config.compute_pain_sensitivity:
        payload["pain_sensitivity_status"] = "disabled"
    elif payload["temperature_status"]["available"]:
        payload["pain_sensitivity_status"] = "computed" if payload["outputs"]["has_pain_sensitivity"] else "skipped"
    else:
        payload["pain_sensitivity_status"] = "skipped_no_temperature"

    payload["covariates_qc"] = _summarize_covariates_qc(ctx)

    if stage_metrics:
        payload["stage_metrics"] = stage_metrics

    if outputs_manifest is not None:
        payload["outputs_manifest"] = str(outputs_manifest)

    if ctx.data_qc:
        payload["data_qc"] = ctx.data_qc

    corr_df = getattr(results, "correlations", None)
    if corr_df is not None and not corr_df.empty:
        df = corr_df
        partial_cols = [
            ("p_partial_cov", "partial_cov"),
            ("p_partial_temp", "partial_temp"),
            ("p_partial_cov_temp", "partial_cov_temp"),
        ]
        partial_ok: Dict[str, Any] = {}
        for col, label in partial_cols:
            if col not in df.columns:
                continue
            pvals = pd.to_numeric(df[col], errors="coerce")
            partial_ok[label] = {
                "n_non_nan": int(pvals.notna().sum()),
                "fraction_non_nan": float(pvals.notna().mean()) if len(pvals) else np.nan,
            }
        if partial_ok:
            payload["partial_correlation_feasibility"] = partial_ok

        if "p_primary_source" in df.columns and df["p_primary_source"].notna().any():
            payload["primary_test_source_counts"] = df["p_primary_source"].fillna("unknown").value_counts().to_dict()

        if "within_family_p_kind" in df.columns and df["within_family_p_kind"].notna().any():
            payload["within_family_p_kind_counts"] = df["within_family_p_kind"].fillna("unknown").value_counts().to_dict()

    (ctx.stats_dir / "analysis_metadata.json").write_text(json.dumps(payload, indent=2, default=str))


def stage_condition(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    from eeg_pipeline.analysis.behavior.api import split_by_condition, compute_condition_effects

    # Trial-table-only: derive condition split from canonical trial table.
    df_trials = _load_trial_table_df(ctx)
    if df_trials is None or df_trials.empty:
        ctx.logger.warning("Condition: trial table missing; skipping.")
        return pd.DataFrame()

    fail_fast = get_config_value(ctx.config, "behavior_analysis.condition.fail_fast", True)

    try:
        pain_mask, nonpain_mask, n_pain, n_nonpain = split_by_condition(df_trials, ctx.config, ctx.logger)
    except Exception as exc:
        if fail_fast:
            raise
        ctx.logger.warning(f"Condition split failed; skipping condition effects: {exc}")
        return pd.DataFrame()

    if n_pain == 0 and n_nonpain == 0:
        msg = "Condition split produced zero trials; check pain coding and config event_columns.pain_binary"
        if fail_fast:
            raise ValueError(msg)
        ctx.logger.warning(msg)
        return pd.DataFrame()

    min_samples = max(config.min_samples, ctx.get_min_samples("default"))
    if n_pain < min_samples or n_nonpain < min_samples:
        ctx.logger.warning(
            f"Insufficient trials: {n_pain} pain, {n_nonpain} non-pain (need >= {min_samples})"
        )
        return pd.DataFrame()

    feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)
    feature_cols = _filter_feature_cols_for_computation(feature_cols, "condition", ctx)
    if not feature_cols:
        ctx.logger.info("Condition: no feature columns found in trial table; skipping.")
        return pd.DataFrame()
    features = df_trials[feature_cols].copy()

    out = compute_condition_effects(
        features,
        pain_mask,
        nonpain_mask,
        min_samples=min_samples,
        fdr_alpha=config.fdr_alpha,
        logger=ctx.logger,
        n_jobs=config.n_jobs,
        config=ctx.config,
    )
    # Standardize p-value columns for downstream validation/global FDR.
    if out is not None and not out.empty:
        out = out.copy()
        if "p_value" in out.columns and "p_raw" not in out.columns:
            out["p_raw"] = pd.to_numeric(out["p_value"], errors="coerce")
        if "p_value" in out.columns and "p_primary" not in out.columns:
            out["p_primary"] = pd.to_numeric(out["p_value"], errors="coerce")
        if "q_value" in out.columns and "p_fdr" not in out.columns:
            out["p_fdr"] = pd.to_numeric(out["q_value"], errors="coerce")
    return out if out is not None else pd.DataFrame()


def stage_temporal(ctx: BehaviorContext) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    from eeg_pipeline.analysis.behavior.api import (
        compute_time_frequency_from_context,
        compute_temporal_from_context,
        run_power_topomap_correlations,
    )

    status = {"time_frequency": "success", "temporal": "success", "topomap": "success"}
    tf_results = None
    temporal_results = None

    ctx.logger.info("Computing time-frequency correlations...")
    try:
        tf_results = compute_time_frequency_from_context(ctx)
    except Exception as exc:
        status["time_frequency"] = f"failed: {exc}"
        ctx.logger.error(f"Time-frequency correlations failed: {exc}")

    ctx.logger.info("Computing temporal correlations by condition...")
    try:
        temporal_results = compute_temporal_from_context(ctx)
    except Exception as exc:
        status["temporal"] = f"failed: {exc}"
        ctx.logger.error(f"Temporal correlations failed: {exc}")

    try:
        run_power_topomap_correlations(
            subject=ctx.subject,
            task=ctx.task,
            power_df=ctx.power_df,
            temperature=ctx.temperature,
            epochs_info=ctx.epochs_info,
            stats_dir=ctx.stats_dir,
            config=ctx.config,
            logger=ctx.logger,
            use_spearman=ctx.use_spearman,
            rng=ctx.rng,
            bootstrap=ctx.bootstrap,
            n_perm=ctx.n_perm,
        )
    except Exception as exc:
        status["topomap"] = f"failed: {exc}"
        ctx.logger.error(f"Topomap correlations failed: {exc}")

    failed = {k: v for k, v in status.items() if v != "success"}
    if failed:
        ctx.logger.info(f"Temporal stage status: {failed}")
    
    return tf_results, temporal_results


def stage_cluster(ctx: BehaviorContext, config: Any) -> Dict[str, Any]:
    from eeg_pipeline.analysis.behavior.api import run_cluster_test_from_context

    ctx.logger.info("Running cluster permutation tests...")
    ctx.n_perm = config.n_permutations

    try:
        results = run_cluster_test_from_context(ctx)
        return results if results else {"status": "completed"}
    except Exception as exc:
        ctx.logger.warning(f"Cluster tests failed: {exc}")
        return {"status": "failed", "error": str(exc)}


_METADATA_COLUMNS = frozenset(["subject", "epoch", "trial", "condition", "temperature", "rating"])


def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract feature columns excluding metadata columns."""
    return [c for c in df.columns if c not in _METADATA_COLUMNS]


def stage_advanced(ctx: BehaviorContext, config: Any, results: Any) -> None:
    from eeg_pipeline.analysis.behavior.api import run_mediation_analysis

    # Trial-table-only: mediation uses the canonical trial table.
    df_trials = _load_trial_table_df(ctx)
    if df_trials is None or df_trials.empty:
        ctx.logger.info("Advanced: trial table missing; skipping.")
        return

    feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)
    feature_cols = _filter_feature_cols_for_computation(feature_cols, "mediation", ctx)
    if not feature_cols:
        return
    features = df_trials.copy()

    if config.run_mediation:
        if "temperature" in features.columns and "rating" in features.columns:
            ctx.logger.info("Running mediation analysis...")
            n_bootstrap = int(get_config_value(ctx.config, "behavior_analysis.mediation.n_bootstrap", 1000))
            min_effect_size = float(get_config_value(ctx.config, "behavior_analysis.mediation.min_effect_size", 0.05))
            max_mediators = int(get_config_value(ctx.config, "behavior_analysis.mediation.max_mediators", 20))

            variances = features[feature_cols].var()
            mediators = variances.nlargest(max(1, max_mediators)).index.tolist()
            results.mediation = run_mediation_analysis(
                features,
                "temperature",
                mediators,
                "rating",
                n_bootstrap=n_bootstrap,
                min_effect_size=min_effect_size,
            )
        else:
            ctx.logger.warning("Skipping mediation: 'temperature' or 'rating' missing.")

    if config.run_mixed_effects:
        # Mixed-effects models require multiple subjects with repeated measures.
        # Subject-level behavior analysis should skip this stage to avoid misleading results.
        ctx.logger.warning(
            "Skipping mixed-effects analysis in subject-level mode; run this at group level with multiple subjects."
        )


def stage_validate(ctx: BehaviorContext, config: Any, results: Optional[Any] = None) -> None:
    from eeg_pipeline.utils.analysis.stats.fdr import apply_global_fdr
    from eeg_pipeline.utils.analysis.stats.reliability import compute_hierarchical_fdr_summary

    fdr_alpha = config.fdr_alpha

    # Determine which analysis files to include in FDR based on current computation flags
    patterns = []
    
    # 1. Condition effects
    if getattr(config, "run_condition_comparison", False):
        patterns.append("condition_effects*.tsv")

    # 1b. Confounds + regression
    if getattr(config, "run_confounds", False):
        patterns.append("confounds_audit*.tsv")
    if getattr(config, "run_regression", False):
        patterns.append("regression_feature_effects*.tsv")
    if getattr(config, "run_models", False):
        patterns.append("models_feature_effects*.tsv")
        
    # 2. Mediation and Mixed Effects
    if getattr(config, "run_mediation", False):
        patterns.append("mediation*.tsv")
    if getattr(config, "run_mixed_effects", False):
        patterns.append("mixed_effects*.tsv")

    # 3. Correlations and Pain Sensitivity (trial-table-only TSVs)
    run_corrs = getattr(config, "run_correlations", True)
    run_psi = getattr(config, "compute_pain_sensitivity", False)
    if run_corrs:
        patterns.append("correlations*.tsv")
    if run_psi:
        patterns.append("pain_sensitivity*.tsv")

    # 4. Temporal and Time-Frequency
    if getattr(config, "run_temporal_correlations", False):
        patterns.extend([
            "corr_stats_tf_*.tsv",
            "corr_stats_temporal_*.tsv",
            "*_topomap_*_correlations_*.tsv"
        ])

    # 5. Cluster Tests
    if getattr(config, "run_cluster_tests", False):
        patterns.append("cluster_results_*.tsv")

    # Remove duplicates
    patterns = list(set(patterns))
    
    # If no specific patterns were identified but we're in validation, 
    # it means all specific modules were disabled. 
    # We only use the broad fallback if absolutely no patterns were added.
    if not patterns:
        patterns = ["correlations*.tsv", "condition_effects*.tsv", "pain_sensitivity*.tsv"]

    ctx.logger.info("Applying global FDR correction...")
    summary = apply_global_fdr(
        ctx.stats_dir,
        alpha=fdr_alpha,
        logger=ctx.logger,
        include_glob=patterns,
    )

    # 4. If results object is provided, update its dataframes with global FDR info
    if results is not None:
        from eeg_pipeline.infra.tsv import read_tsv
        from eeg_pipeline.utils.analysis.stats.fdr import select_p_column_for_fdr
        
        # Mapping of result types to files
        mapping = {
            "confounds": "confounds_audit*.tsv",
            "regression": "regression_feature_effects*.tsv",
            "models": "models_feature_effects*.tsv",
            "correlations": "correlations*.tsv",
            "pain_sensitivity": "pain_sensitivity*.tsv",
            "condition_effects": "condition_effects*.tsv",
            "tf": "corr_stats_tf_*.tsv",
            "temporal": "corr_stats_temporal_*.tsv",
            "cluster": "cluster_results_*.tsv",
            "mediation": "mediation*.tsv",
            "mixed_effects": "mixed_effects*.tsv",
        }
        
        for attr, pattern in mapping.items():
            current_value = getattr(results, attr, None)
            if current_value is None:
                continue
                
            match_files = list(ctx.stats_dir.rglob(pattern))
            if not match_files:
                continue
            
            for fpath in match_files:
                df_disk = read_tsv(fpath)
                if df_disk is None or "q_global" not in df_disk.columns:
                    continue

                if isinstance(current_value, pd.DataFrame):
                    if current_value.empty: continue
                    merge_cols = [c for c in ["feature", "feature_id", "target", "mediator"] if c in df_disk.columns]
                    if merge_cols:
                        if "q_global" in current_value.columns:
                            current_value = current_value.drop(columns=["q_global"])
                        current_value = pd.merge(current_value, df_disk[merge_cols + ["q_global"]], on=merge_cols, how="left")
                        setattr(results, attr, current_value)
                    elif len(current_value) == len(df_disk):
                        current_value["q_global"] = df_disk["q_global"].values
                        setattr(results, attr, current_value)
                elif isinstance(current_value, dict):
                    # For cluster, find the band from filename if possible
                    if attr == "cluster":
                        band_name = fpath.stem.replace("cluster_results_", "")
                        if band_name in current_value and "cluster_records" in current_value[band_name]:
                            recs = current_value[band_name]["cluster_records"]
                            if len(recs) == len(df_disk):
                                for r, q in zip(recs, df_disk["q_global"]):
                                    r["q_global"] = q
                    else:
                        # For temporal/tf summary dicts
                        current_value["n_sig_fdr"] = int((df_disk["q_global"] < fdr_alpha).sum())
                        setattr(results, attr, current_value)

    ctx.logger.info("Computing hierarchical FDR summary...")
    try:
        hier_summary = compute_hierarchical_fdr_summary(
            ctx.stats_dir,
            alpha=fdr_alpha,
            config=ctx.config,
            include_glob=patterns,
        )
        if not hier_summary.empty:
            hier_path = ctx.stats_dir / "hierarchical_fdr_summary.tsv"
            hier_summary.to_csv(hier_path, sep="\t", index=False)
            ctx.logger.info(f"Hierarchical FDR summary saved to {hier_path}")
            
            for _, row in hier_summary.iterrows():
                ctx.logger.info(
                    f"  {row['analysis_type']}: {row['n_reject_within']}/{row['n_tests']} "
                    f"({row['pct_reject_within']:.1f}%) reject within-family"
                )
    except Exception as exc:
        ctx.logger.warning(f"Hierarchical FDR failed: {exc}")


def stage_paired_comparisons(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Compute all paired comparisons for time window and condition-based analyses.
    
    This stage pre-computes all paired comparisons that the plotting pipeline
    previously computed on-the-fly, enabling faster plotting and consistent statistics.
    
    Comparisons include:
    - Window comparisons (paired): Baseline vs Active using Wilcoxon signed-rank
    - Condition comparisons (unpaired): Pain vs Non-Pain using Mann-Whitney U
    
    All tests include FDR correction, effect sizes (Cohen's d, Hedges' g),
    and bootstrap confidence intervals.
    """
    from eeg_pipeline.utils.analysis.stats.paired_comparisons import (
        compute_all_paired_comparisons,
        save_paired_comparisons,
    )
    
    feature_dfs = {}
    for name, df in ctx.iter_feature_tables():
        if df is not None and not df.empty:
            feature_dfs[name] = df
    
    if not feature_dfs:
        ctx.logger.warning("Paired comparisons: no feature tables available; skipping.")
        return pd.DataFrame()
    
    min_samples = int(get_config_value(ctx.config, "behavior_analysis.min_samples.default", 5))
    n_boot = int(get_config_value(ctx.config, "behavior_analysis.paired_comparisons.n_bootstrap", 1000))
    fdr_alpha = float(get_config_value(ctx.config, "behavior_analysis.statistics.fdr_alpha", 0.05))
    
    summary = compute_all_paired_comparisons(
        feature_dfs=feature_dfs,
        events_df=ctx.aligned_events,
        config=ctx.config,
        logger=ctx.logger,
        min_samples=min_samples,
        n_boot=n_boot,
        fdr_alpha=fdr_alpha,
        rng=ctx.rng,
    )
    
    suffix = _feature_suffix_from_context(ctx)
    out_dir = _get_stats_subfolder(ctx, "paired_comparisons")
    out_path = save_paired_comparisons(summary, out_dir, suffix=suffix)
    
    ctx.logger.info(
        f"Paired comparisons: {summary.n_tests} tests, "
        f"{summary.n_significant_raw} raw significant, "
        f"{summary.n_significant_fdr} FDR significant"
    )
    
    return summary.to_dataframe()


def stage_report(ctx: BehaviorContext, pipeline_config: Any) -> Optional[Path]:
    """Write a single-subject, self-diagnosing Markdown report (non-gating)."""
    suffix = _feature_suffix_from_context(ctx)
    method_label = getattr(pipeline_config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    top_n = int(get_config_value(ctx.config, "behavior_analysis.report.top_n", 15))
    alpha = float(getattr(pipeline_config, "fdr_alpha", 0.05))

    # Trial-table summary
    df_trials = _load_trial_table_df(ctx)
    n_trials = int(len(df_trials)) if df_trials is not None else 0
    n_features = 0
    if df_trials is not None and not df_trials.empty:
        n_features = int(sum(1 for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)))

    # Trial-table validation (if present)
    val_status = None
    val_warnings: List[str] = []
    try:
        vpath = _find_stats_path(ctx, f"trial_table_validation{suffix}.json")
        if vpath and vpath.exists():
            payload = json.loads(vpath.read_text())
            val_status = payload.get("status")
            val_warnings = [str(x) for x in (payload.get("warnings") or [])]
    except Exception:
        pass

    def _read_tsv(path: Path) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return None

    def _sig_counts(df: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {"n": int(len(df))}
        for col in ["q_global", "p_fdr", "p_primary"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                out[f"n_sig_{col}"] = int((vals < alpha).sum()) if vals.notna().any() else 0
        return out

    def _top_rows(df: pd.DataFrame) -> pd.DataFrame:
        pcols = [c for c in ["q_global", "p_fdr", "p_primary"] if c in df.columns]
        if not pcols:
            return df.head(0)
        pcol = pcols[0]
        out = df.copy()
        out[pcol] = pd.to_numeric(out[pcol], errors="coerce")
        out = out.sort_values(pcol, ascending=True)
        keep = [c for c in ["feature", "target", "r_primary", "beta_feature", "hedges_g", "p_primary", "p_fdr", "q_global"] if c in out.columns]
        return out[keep].head(max(1, int(top_n))) if keep else out.head(0)

    def _to_md_table(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return ""
        df2 = df.copy()
        for c in df2.columns:
            df2[c] = df2[c].apply(lambda x: "" if pd.isna(x) else str(x))
        cols = [str(c) for c in df2.columns]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in df2.to_numpy(dtype=str).tolist()]
        return "\n".join([header, sep, *rows])

    patterns = [
        "correlations*.tsv",
        "pain_sensitivity*.tsv",
        "regression_feature_effects*.tsv",
        "models_feature_effects*.tsv",
        "condition_effects*.tsv",
        "consistency_summary*.tsv",
        "influence_diagnostics*.tsv",
        "trial_table_validation_summary*.tsv",
        "temperature_model_comparison*.tsv",
        "temperature_breakpoint_candidates*.tsv",
        "hierarchical_fdr_summary.tsv",
        "normalized_results*.tsv",
        "summary.json",
        "analysis_metadata.json",
        "outputs_manifest.json",
    ]

    files: List[Path] = []
    for pat in patterns:
        found = sorted(ctx.stats_dir.rglob(pat))
        files.extend(found)
    # Include any other TSV outputs not covered above.
    extra = sorted(p for p in ctx.stats_dir.rglob("*.tsv") if p not in files)
    files.extend(extra)
    files = sorted({p.resolve() for p in files if p.exists()})

    lines: List[str] = []
    lines.append(f"# Subject Report: sub-{ctx.subject}")
    lines.append("")
    lines.append(f"- Task: `{ctx.task}`")
    lines.append(f"- Trials: `{n_trials}`")
    lines.append(f"- Features in trial table: `{n_features}`")
    lines.append(f"- Method: `{getattr(pipeline_config, 'method', '')}` (`{method_label}`)")
    lines.append(f"- Controls: temperature=`{bool(getattr(pipeline_config, 'control_temperature', True))}`, trial_order=`{bool(getattr(pipeline_config, 'control_trial_order', True))}`")
    lines.append(f"- Global FDR alpha: `{alpha}`")
    if val_status is not None:
        lines.append(f"- Trial table validation: `{val_status}`")
    if val_warnings:
        lines.append("")
        lines.append("## Validation Warnings")
        for w in val_warnings[: min(20, len(val_warnings))]:
            lines.append(f"- {w}")
        if len(val_warnings) > 20:
            lines.append(f"- ... {len(val_warnings) - 20} more")

    # Summaries per output file (TSVs)
    tsvs = [p for p in files if p.suffix == ".tsv"]
    if tsvs:
        lines.append("")
        lines.append("## Outputs")
        for p in tsvs:
            df = _read_tsv(p)
            if df is None or df.empty:
                lines.append(f"- `{p.name}`: (empty or unreadable)")
                continue
            counts = _sig_counts(df)
            sig_bits = []
            for k in ["n_sig_q_global", "n_sig_p_fdr", "n_sig_p_primary"]:
                if k in counts:
                    sig_bits.append(f"{k}={counts[k]}")
            sig_str = ", ".join(sig_bits) if sig_bits else "no p-columns"
            lines.append(f"- `{p.name}`: n={counts['n']}, {sig_str}")

            top = _top_rows(df)
            if not top.empty and ("feature" in top.columns or "target" in top.columns):
                lines.append("")
                lines.append(f"### Top ({min(len(top), top_n)}) — `{p.name}`")
                lines.append("")
                lines.append(_to_md_table(top))
                lines.append("")

    report_dir = _get_stats_subfolder(ctx, "subject_report")
    out_path = report_dir / f"subject_report{suffix}{method_suffix}.md"
    try:
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ctx.logger.info("Subject report saved: %s/%s", report_dir.name, out_path.name)
        return out_path
    except Exception as exc:
        ctx.logger.warning("Failed to write subject report: %s", exc)
        return None

def stage_export(ctx: BehaviorContext, pipeline_config: Any, results: Any) -> List[Path]:
    from eeg_pipeline.infra.paths import ensure_dir
    from eeg_pipeline.infra.tsv import write_tsv

    ensure_dir(ctx.stats_dir)
    saved: List[Path] = []
    robust_method = pipeline_config.robust_method
    method_label = pipeline_config.method_label
    method_suffix = f"_{method_label}" if method_label else ""
    
    # Build feature suffix from selected feature files or categories
    feature_files = ctx.selected_feature_files or ctx.feature_categories or []
    feature_suffix = "_" + "_".join(sorted(feature_files)) if feature_files else ""

    if getattr(results, "correlations", None) is not None:
        results.correlations = _ensure_method_columns(
            results.correlations, pipeline_config.method, robust_method, method_label
        )
    if getattr(results, "pain_sensitivity", None) is not None:
        results.pain_sensitivity = _ensure_method_columns(
            results.pain_sensitivity, pipeline_config.method, robust_method, method_label
        )
    if getattr(results, "condition_effects", None) is not None:
        results.condition_effects = _ensure_method_columns(
            results.condition_effects, pipeline_config.method, robust_method, method_label
        )
    if getattr(results, "mediation", None) is not None:
        results.mediation = _ensure_method_columns(
            results.mediation, pipeline_config.method, robust_method, method_label
        )
    if getattr(results, "mixed_effects", None) is not None:
        results.mixed_effects = _ensure_method_columns(
            results.mixed_effects, pipeline_config.method, robust_method, method_label
        )
    if getattr(results, "confounds", None) is not None:
        results.confounds = _ensure_method_columns(
            results.confounds, pipeline_config.method, robust_method, method_label
        )
    if getattr(results, "regression", None) is not None:
        results.regression = _ensure_method_columns(
            results.regression, pipeline_config.method, robust_method, method_label
        )
    if getattr(results, "models", None) is not None:
        results.models = _ensure_method_columns(
            results.models, pipeline_config.method, robust_method, method_label
        )

    if getattr(results, "correlations", None) is not None and not results.correlations.empty:
        out_dir = _get_stats_subfolder(ctx, "correlations")
        path = out_dir / f"correlations{feature_suffix}{method_suffix}.tsv"
        write_tsv(results.correlations, path)
        saved.append(path)

    if getattr(results, "pain_sensitivity", None) is not None and not results.pain_sensitivity.empty:
        out_dir = _get_stats_subfolder(ctx, "pain_sensitivity")
        path = out_dir / f"pain_sensitivity{feature_suffix}{method_suffix}.tsv"
        write_tsv(results.pain_sensitivity, path)
        saved.append(path)

    if getattr(results, "condition_effects", None) is not None and not results.condition_effects.empty:
        out_dir = _get_stats_subfolder(ctx, "condition_effects")
        path = out_dir / f"condition_effects{feature_suffix}.tsv"
        write_tsv(results.condition_effects, path)
        saved.append(path)

    if getattr(results, "mediation", None) is not None and not results.mediation.empty:
        out_dir = _get_stats_subfolder(ctx, "mediation")
        path = out_dir / f"mediation{feature_suffix}.tsv"
        write_tsv(results.mediation, path)
        saved.append(path)

    if getattr(results, "mixed_effects", None) is not None and not results.mixed_effects.empty:
        out_dir = _get_stats_subfolder(ctx, "mixed_effects")
        path = out_dir / f"mixed_effects{feature_suffix}.tsv"
        write_tsv(results.mixed_effects, path)
        saved.append(path)

    if getattr(results, "confounds", None) is not None and not results.confounds.empty:
        out_dir = _get_stats_subfolder(ctx, "confounds_audit")
        path = out_dir / f"confounds_audit{feature_suffix}{method_suffix}.tsv"
        write_tsv(results.confounds, path)
        saved.append(path)

    if getattr(results, "regression", None) is not None and not results.regression.empty:
        out_dir = _get_stats_subfolder(ctx, "trialwise_regression")
        path = out_dir / f"regression_feature_effects{feature_suffix}{method_suffix}.tsv"
        write_tsv(results.regression, path)
        saved.append(path)

    if getattr(results, "models", None) is not None and not results.models.empty:
        out_dir = _get_stats_subfolder(ctx, "feature_models")
        path = out_dir / f"models_feature_effects{feature_suffix}{method_suffix}.tsv"
        write_tsv(results.models, path)
        saved.append(path)

    normalized_path = _write_normalized_results(ctx, pipeline_config, results)
    if normalized_path is not None:
        saved.append(normalized_path)

    summary = results.to_summary()
    (ctx.stats_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    saved.append(ctx.stats_dir / "summary.json")

    ctx.logger.info(f"Saved {len(saved)} output files")
    return saved


def _infer_output_kind(name: str) -> str:
    if name.startswith("corr_stats_"):
        return "correlation_stats"
    if name.startswith("correlations"):
        return "correlations"
    if name.startswith("pain_sensitivity"):
        return "pain_sensitivity"
    if name.startswith("condition_effects"):
        return "condition_effects"
    if name.startswith("mediation"):
        return "mediation"
    if name.startswith("mixed_effects"):
        return "mixed_effects"
    if name.startswith("confounds_audit"):
        return "confounds_audit"
    if name.startswith("regression_feature_effects"):
        return "trialwise_regression"
    if name.startswith("models_feature_effects"):
        return "feature_models"
    if name.startswith("trials"):
        return "trial_table"
    if name.startswith("trial_table_validation"):
        return "trial_table_validation"
    if name.startswith("temperature_model_comparison"):
        return "temperature_model_comparison"
    if name.startswith("temperature_breakpoint"):
        return "temperature_breakpoint_test"
    if name.startswith("stability_groupwise"):
        return "stability_groupwise"
    if name.startswith("consistency_summary"):
        return "consistency_summary"
    if name.startswith("influence_diagnostics"):
        return "influence_diagnostics"
    if name.startswith("normalized_results"):
        return "normalized_results"
    if name.startswith("feature_screening"):
        return "feature_screening"
    if name.startswith("paired_comparisons"):
        return "paired_comparisons"
    if name.startswith("summary"):
        return "summary"
    if name.startswith("analysis_metadata"):
        return "analysis_metadata"
    if name.startswith("subject_report"):
        return "subject_report"
    if name.startswith("time_frequency_correlation_data"):
        return "time_frequency"
    if name.startswith("temporal_correlations"):
        return "temporal_correlations"
    if name.startswith("hierarchical_fdr_summary"):
        return "hierarchical_fdr_summary"
    return "unknown"


def _count_rows(path: Path) -> Optional[int]:
    if path.suffix not in {".tsv", ".csv"}:
        return None
    try:
        with path.open("r") as f:
            header = f.readline()
            if not header:
                return 0
            return sum(1 for _ in f)
    except Exception:
        return None


def write_outputs_manifest(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Any,
    stage_metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    from datetime import datetime

    outputs = []
    for path in sorted(p for p in ctx.stats_dir.rglob("*") if p.is_file() and p.name != "outputs_manifest.json"):
        # Skip hidden files or logs
        if path.name.startswith(".") or path.suffix == ".log":
            continue
        outputs.append({
            "name": path.name,
            "path": str(path),
            "kind": _infer_output_kind(path.name),
            "subfolder": path.parent.name if path.parent != ctx.stats_dir else None,
            "rows": _count_rows(path),
            "size_bytes": int(path.stat().st_size),
            "method_label": pipeline_config.method_label,
        })

    feature_types = [name for name, df in ctx.iter_feature_tables() if df is not None and not df.empty]

    payload = {
        "subject": ctx.subject,
        "task": ctx.task,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "method": pipeline_config.method,
        "robust_method": pipeline_config.robust_method,
        "method_label": pipeline_config.method_label,
        "n_trials": ctx.n_trials,
        "feature_types": feature_types,
        "feature_categories": ctx.feature_categories or [],
        "feature_files": ctx.selected_feature_files or [],
        "targets": {
            "rating": bool(ctx.targets is not None and ctx.targets.notna().any()) if ctx.targets is not None else False,
            "temperature": bool(ctx.temperature is not None and ctx.temperature.notna().any()) if ctx.temperature is not None else False,
        },
        "covariates_qc": ctx.data_qc.get("covariates_qc", {}),
        "outputs": outputs,
    }

    if stage_metrics:
        payload["stage_metrics"] = stage_metrics

    path = ctx.stats_dir / "outputs_manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
