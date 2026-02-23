"""Behavior pipeline orchestration.

This module contains the implementation of the behavior pipeline stages.
The pipeline layer (`eeg_pipeline.pipelines.behavior`) should remain a thin wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.context.behavior import BehaviorContext
from eeg_pipeline.analysis.behavior.result_types import (
    FeatureQCResult,
    GroupLevelResult,
    MixedEffectsResult,
    TempBreakpointResult,
    TempModelComparisonResult,
    TrialTableResult,
)
from eeg_pipeline.analysis.behavior.result_cache import BehaviorResultCache
from eeg_pipeline.analysis.behavior import (
    change_scores as _change_scores,
    common_helpers as _common_helpers,
    feature_filters as _feature_filters,
    feature_inference as _feature_inference,
    group_level as _group_level,
    stage_execution as _stage_execution,
    stage_registry as _stage_registry,
    stage_runners as _stage_runners,
)
from eeg_pipeline.analysis.behavior.stages import (
    advanced as _stages_advanced,
    condition as _stages_condition,
    correlate as _stages_correlate,
    diagnostics as _stages_diagnostics,
    export as _stages_export,
    fdr as _stages_fdr,
    feature_qc as _stages_feature_qc,
    metadata as _stages_metadata,
    models as _stages_models,
    report as _stages_report,
    temporal as _stages_temporal,
    trial_table as _stages_trial_table,
)
from eeg_pipeline.analysis.behavior.trial_table_helpers import (
    compute_trial_table_input_hash as _compute_trial_table_input_hash,
    feature_folder_from_context as _feature_folder_from_context,
    find_trial_table_path as _find_trial_table_path,
    trial_table_feature_folder_from_features as _trial_table_feature_folder_from_features,
    trial_table_metadata_path as _trial_table_metadata_path,
    trial_table_suffix_from_context as _trial_table_suffix_from_context,
    validate_trial_table_contract_metadata as _validate_trial_table_contract_metadata,
)
from eeg_pipeline.utils.analysis.stats.correlation import (
    compute_correlation,
    format_correlation_method_label,
)
from eeg_pipeline.utils.config.loader import get_config_int, get_config_bool
from eeg_pipeline.infra.paths import ensure_dir


StageRegistry = _stage_registry.StageRegistry
config_to_stage_names = _stage_registry.config_to_stage_names


UNIFIED_FDR_FAMILY_COLUMNS = ["feature_type", "band", "target", "analysis_kind"]


def _write_parquet_with_optional_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    also_save_csv: bool,
) -> None:
    from eeg_pipeline.infra.tsv import write_parquet, write_csv

    write_parquet(df, path)
    if also_save_csv:
        write_csv(df, path.with_suffix(".csv"), index=False)


def _also_save_csv_from_config(config: Any) -> bool:
    """Resolve whether parquet outputs should also be emitted as CSV."""
    return bool(get_config_bool(config, "behavior_analysis.output.also_save_csv", False))


###################################################################
# Result Caching Layer
###################################################################


@dataclass
class BehaviorOrchestrationRuntime:
    """Run-scoped mutable orchestration state."""

    cache: BehaviorResultCache
    stage_runners: Dict[str, callable]


def _build_stage_runners() -> Dict[str, callable]:
    return _stage_runners.build_stage_runners_from_namespace_impl(
        globals(),
        build_results_from_outputs_fn=_stage_registry.build_results_from_outputs,
    )


def create_behavior_runtime() -> BehaviorOrchestrationRuntime:
    """Create an isolated runtime for one behavior pipeline execution."""
    return BehaviorOrchestrationRuntime(
        cache=_build_result_cache(),
        stage_runners=_build_stage_runners(),
    )


def _get_runtime(ctx: Any) -> BehaviorOrchestrationRuntime:
    runtime = getattr(ctx, "_behavior_runtime", None)
    if runtime is None:
        runtime = create_behavior_runtime()
        setattr(ctx, "_behavior_runtime", runtime)
    return runtime


def _get_cache(ctx: Any) -> BehaviorResultCache:
    return _get_runtime(ctx).cache


###################################################################
# Feature QC Screen Stage
###################################################################


def stage_feature_qc_screen(
    ctx: BehaviorContext,
    config: Any,
) -> FeatureQCResult:
    return _stages_feature_qc.stage_feature_qc_screen_impl(
        ctx,
        config,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        feature_column_prefixes=FEATURE_COLUMN_PREFIXES,
        feature_suffix_from_context_fn=_feature_suffix_from_context,
        get_stats_subfolder_fn=_get_stats_subfolder,
        write_parquet_with_optional_csv_fn=_write_parquet_with_optional_csv,
        max_missing_pct_default=MAX_MISSING_PCT_DEFAULT,
        min_variance_threshold=MIN_VARIANCE_THRESHOLD,
    )


###################################################################
# Stage Executor - Run Selected Stages from Registry
###################################################################


def run_selected_stages(
    ctx: BehaviorContext,
    config: Any,
    selected_stages: List[str],
    results: Optional[Any] = None,
    progress: Optional[Any] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    runtime = _get_runtime(ctx)
    return _stage_execution.run_selected_stages_impl(
        ctx=ctx,
        config=config,
        selected_stages=selected_stages,
        stage_registry=StageRegistry,
        stage_runners=runtime.stage_runners,
        is_stage_enabled_by_config_fn=_stage_registry.is_stage_enabled_by_config,
        update_results_from_stage_fn=_stage_execution.update_results_from_stage_impl,
        log_stage_outcome_fn=_stage_execution.log_stage_outcome_impl,
        results=results,
        progress=progress,
        dry_run=dry_run,
    )


def run_behavior_stages(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Optional[Any] = None,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    return _stage_execution.run_behavior_stages_impl(
        ctx=ctx,
        pipeline_config=pipeline_config,
        config_to_stage_names_fn=config_to_stage_names,
        run_selected_stages_fn=run_selected_stages,
        results=results,
        progress=progress,
    )


# Centralized feature column prefixes - single source of truth
FEATURE_COLUMN_PREFIXES = (
    "power_",
    "connectivity_",
    "directedconnectivity_",
    "sourcelocalization_",
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
    "microstates_",
    "temporal_",
)

CATEGORY_PREFIX_MAP = {prefix.rstrip("_"): prefix for prefix in FEATURE_COLUMN_PREFIXES}

# Constants for validation thresholds
MIN_SAMPLES_DEFAULT = 10
MIN_SAMPLES_RUN_LEVEL = 3
MIN_VARIANCE_THRESHOLD = 1e-10
CONSTANT_VARIANCE_THRESHOLD = 1e-12
MAX_MISSING_PCT_DEFAULT = 0.2
FDR_ALPHA_DEFAULT = 0.05
MIN_FEATURES_FOR_ANALYSIS = 1
MIN_TRIALS_FOR_ANALYSIS = 1


def _check_early_exit_conditions(
    df: Optional[pd.DataFrame],
    feature_cols: Optional[List[str]] = None,
    min_features: int = MIN_FEATURES_FOR_ANALYSIS,
    min_trials: int = MIN_TRIALS_FOR_ANALYSIS,
) -> Tuple[bool, Optional[str]]:
    return _common_helpers.check_early_exit_conditions_impl(
        df,
        feature_cols=feature_cols,
        min_features=min_features,
        min_trials=min_trials,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
    )


def _get_stats_subfolder(ctx: BehaviorContext, kind: str) -> Path:
    """Helper to get a subfolder within stats_dir and ensure it exists.
    
    If ctx.overwrite is False, appends a timestamp to the folder name
    (e.g., 'trial_table_20260120_143022') to preserve previous outputs.
    """
    return get_behavior_output_dir(ctx, kind, ensure=True)


def _get_stats_subfolder_with_overwrite(
    ctx: BehaviorContext,
    stats_dir: Path,
    kind: str,
    overwrite: bool,
    *,
    ensure: bool = True,
) -> Path:
    """Helper to get a subfolder within stats_dir with overwrite control.
    
    If overwrite is False, appends a timestamp to the folder name
    (e.g., 'trial_table_20260120_143022') to preserve previous outputs.
    """
    return _get_cache(ctx).get_stats_subfolder(stats_dir, kind, overwrite, ensure=ensure)


def _trial_table_output_dir(ctx: BehaviorContext, *, ensure: bool = True) -> Path:
    """Return output directory for trial-table artifacts."""
    kind_dir = _get_stats_subfolder_with_overwrite(
        ctx,
        ctx.stats_dir,
        "trial_table",
        ctx.overwrite,
        ensure=ensure,
    )
    feature_dir = kind_dir / _trial_table_feature_folder_from_features(
        ctx.selected_feature_files or ctx.feature_categories or []
    )
    if ensure:
        ensure_dir(feature_dir)
    return feature_dir


def get_behavior_output_dir(ctx: BehaviorContext, kind: str, *, ensure: bool = True) -> Path:
    """Return `stats_dir/<kind>/<feature_folder>` (and optionally create it)."""
    kind_dir = _get_stats_subfolder_with_overwrite(
        ctx,
        ctx.stats_dir,
        kind,
        ctx.overwrite,
        ensure=ensure,
    )
    feature_dir = kind_dir / _feature_folder_from_context(ctx)
    if ensure:
        ensure_dir(feature_dir)
    return feature_dir


def _write_stats_table(
    ctx: BehaviorContext,
    df: pd.DataFrame,
    path: Path,
    force_tsv: bool = False,
) -> Path:
    return _common_helpers.write_stats_table_impl(ctx, df, path, force_tsv=force_tsv)


def _get_feature_columns(
    df: pd.DataFrame,
    ctx: BehaviorContext,
    computation_name: Optional[str] = None,
) -> List[str]:
    """Extract and filter feature columns from DataFrame.
    
    Centralizes the pattern of extracting feature columns and applying
    band and computation-specific filters.
    
    Args:
        df: DataFrame containing feature columns
        ctx: BehaviorContext with filtering preferences
        computation_name: Optional computation name for feature filtering
        
    Returns:
        List of filtered feature column names
    """
    feature_cols = [c for c in df.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    return _get_cache(ctx).get_filtered_feature_cols(feature_cols, ctx, computation_name)


def _attach_temperature_metadata(
    df: pd.DataFrame,
    metadata_dict: Dict[str, Any],
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    return _common_helpers.attach_temperature_metadata_impl(df, metadata_dict, target_col=target_col)


def _has_precomputed_change_scores(df: Optional[pd.DataFrame]) -> bool:
    return _change_scores.has_precomputed_change_scores_impl(df)


def _augment_dataframe_with_change_scores(df: Optional[pd.DataFrame], config: Any) -> Optional[pd.DataFrame]:
    return _change_scores.augment_dataframe_with_change_scores_impl(
        df,
        config,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
    )


def add_change_scores(ctx: BehaviorContext) -> None:
    _change_scores.add_change_scores_impl(
        ctx,
        augment_dataframe_with_change_scores_fn=_augment_dataframe_with_change_scores,
        has_precomputed_change_scores_fn=_has_precomputed_change_scores,
    )


def stage_load(ctx: BehaviorContext) -> bool:
    if not ctx.load_data():
        ctx.logger.warning("Failed to load data")
        return False

    _get_cache(ctx).load_manifest(ctx)

    ctx.logger.info(f"Loaded {ctx.n_trials} trials")
    return True


###################################################################
# Correlate Stage - Single Responsibility Components
###################################################################


CorrelateDesign = _stages_correlate.CorrelateDesign


def stage_correlate_design(ctx: BehaviorContext, config: Any) -> Optional[CorrelateDesign]:
    return _stages_correlate.stage_correlate_design_impl(
        ctx,
        config,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        get_feature_columns_fn=_get_feature_columns,
        sanitize_permutation_groups_fn=_sanitize_permutation_groups,
    )


def _compute_single_effect_size(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    kwargs.setdefault("feature_type_resolver_fn", _infer_feature_type)
    kwargs.setdefault("feature_band_resolver_fn", _infer_feature_band)
    return _stages_correlate._compute_single_effect_size(*args, **kwargs)




def stage_correlate_effect_sizes(
    ctx: BehaviorContext,
    config: Any,
    design: CorrelateDesign,
) -> List[Dict[str, Any]]:
    cache = _get_cache(ctx)
    return _stages_correlate.stage_correlate_effect_sizes_impl(
        ctx,
        config,
        design,
        feature_type_resolver_fn=lambda feature_name, cfg: cache.get_feature_type(str(feature_name), cfg),
        feature_band_resolver_fn=lambda feature_name, cfg: cache.get_feature_band(str(feature_name), cfg),
    )




def stage_correlate_pvalues(
    ctx: BehaviorContext,
    config: Any,
    design: CorrelateDesign,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return _stages_correlate.stage_correlate_pvalues_impl(ctx, config, design, records)


def stage_correlate_primary_selection(
    ctx: BehaviorContext,
    config: Any,
    design: CorrelateDesign,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return _stages_correlate.stage_correlate_primary_selection_impl(ctx, config, design, records)


def _compute_unified_fdr(
    ctx: BehaviorContext,
    config: Any,
    df: pd.DataFrame,
    p_col: str = "p_primary",
    family_cols: Optional[List[str]] = None,
    analysis_type: str = "correlations",
) -> pd.DataFrame:
    cache = _get_cache(ctx)
    return _stages_fdr.compute_unified_fdr_impl(
        ctx,
        config,
        df,
        p_col=p_col,
        family_cols=family_cols,
        analysis_type=analysis_type,
        get_cached_fdr_fn=cache.get_fdr_results,
        set_cached_fdr_fn=cache.set_fdr_results,
    )


def stage_correlate_fdr(
    ctx: BehaviorContext,
    config: Any,
    records: List[Dict[str, Any]],
) -> pd.DataFrame:
    return _stages_correlate.stage_correlate_fdr_impl(
        ctx,
        config,
        records,
        compute_unified_fdr_fn=_compute_unified_fdr,
        unified_fdr_family_columns=UNIFIED_FDR_FAMILY_COLUMNS,
    )


def stage_correlate(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    return _stages_correlate.stage_correlate_impl(
        ctx,
        config,
        stage_correlate_design_fn=stage_correlate_design,
        stage_correlate_effect_sizes_fn=stage_correlate_effect_sizes,
        stage_correlate_pvalues_fn=stage_correlate_pvalues,
        stage_correlate_primary_selection_fn=stage_correlate_primary_selection,
        stage_correlate_fdr_fn=stage_correlate_fdr,
    )


def _sanitize_permutation_groups(
    groups: Optional[np.ndarray],
    logger: Any,
    context: str,
    *,
    min_group_size: int = 2,
) -> Optional[np.ndarray]:
    """Return groups if valid for grouped permutation, otherwise None."""
    if groups is None:
        return None
    groups_array = np.asarray(groups)
    if groups_array.size == 0:
        return None
    unique_groups, counts = np.unique(groups_array[pd.notna(groups_array)], return_counts=True)
    small_group_count = int((counts < int(min_group_size)).sum())
    if small_group_count > 0:
        logger.warning(
            "%s: %d/%d groups have fewer than %d samples. Permutation will fail for subsets containing these groups unless strict=False.",
            context,
            small_group_count,
            len(unique_groups),
            int(min_group_size),
        )
    return groups_array


def stage_predictor_sensitivity(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    return _stages_correlate.stage_predictor_sensitivity_impl(
        ctx,
        config,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        get_feature_columns_fn=_get_feature_columns,
        sanitize_permutation_groups_fn=_sanitize_permutation_groups,
        compute_unified_fdr_fn=_compute_unified_fdr,
        unified_fdr_family_columns=UNIFIED_FDR_FAMILY_COLUMNS,
    )



def _feature_suffix_from_context(ctx: BehaviorContext) -> str:
    feature_files = ctx.selected_feature_files or ctx.feature_categories or []
    return "_" + "_".join(sorted(str(x) for x in feature_files)) if feature_files else ""


def _filter_feature_cols_by_band(
    feature_cols: List[str],
    ctx: BehaviorContext,
) -> List[str]:
    return _feature_filters.filter_feature_cols_by_band_impl(
        feature_cols,
        ctx,
        feature_column_prefixes=FEATURE_COLUMN_PREFIXES,
    )


def _filter_feature_cols_for_computation(
    feature_cols: List[str],
    computation_name: str,
    ctx: BehaviorContext,
) -> List[str]:
    return _feature_filters.filter_feature_cols_for_computation_impl(
        feature_cols,
        computation_name,
        ctx,
        category_prefix_map=CATEGORY_PREFIX_MAP,
    )


def _filter_feature_cols_by_provenance(
    feature_cols: List[str],
    ctx: BehaviorContext,
    computation_name: Optional[str] = None,
) -> List[str]:
    return _feature_filters.filter_feature_cols_by_provenance_impl(
        feature_cols,
        ctx,
        computation_name,
        feature_column_prefixes=FEATURE_COLUMN_PREFIXES,
    )


def compute_trial_table(ctx: BehaviorContext, config: Any) -> Optional[TrialTableResult]:
    return _stages_trial_table.compute_trial_table_impl(ctx, config)


def write_trial_table(ctx: BehaviorContext, result: TrialTableResult) -> Path:
    out_path = _stages_trial_table.write_trial_table_impl(
        ctx,
        result,
        trial_table_suffix_from_context_fn=_trial_table_suffix_from_context,
        trial_table_output_dir_fn=lambda context: _trial_table_output_dir(context, ensure=True),
        write_metadata_file_fn=_write_metadata_file,
    )
    cache = _get_cache(ctx)
    cache._trial_table_df = result.df
    cache._trial_table_path = out_path
    return out_path


def _try_reuse_cached_trial_table(
    ctx: BehaviorContext,
    *,
    input_hash: str,
) -> Optional[Path]:
    reused = _stages_trial_table.try_reuse_cached_trial_table_impl(
        ctx,
        input_hash=input_hash,
        trial_table_suffix_from_context_fn=_trial_table_suffix_from_context,
        trial_table_output_dir_fn=lambda context: _trial_table_output_dir(context, ensure=True),
        trial_table_metadata_path_fn=_trial_table_metadata_path,
        validate_trial_table_contract_metadata_fn=_validate_trial_table_contract_metadata,
    )
    if reused is None:
        return None
    out_path, df_cached = reused
    cache = _get_cache(ctx)
    cache._trial_table_df = df_cached
    cache._trial_table_path = out_path
    return out_path


def stage_trial_table(ctx: BehaviorContext, config: Any) -> Optional[Path]:
    return _stages_trial_table.stage_trial_table_impl(
        ctx,
        config,
        compute_trial_table_input_hash_fn=_compute_trial_table_input_hash,
        try_reuse_cached_trial_table_fn=lambda context, input_hash: _try_reuse_cached_trial_table(
            context,
            input_hash=input_hash,
        ),
        compute_trial_table_fn=compute_trial_table,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        write_trial_table_fn=write_trial_table,
    )


def stage_lag_features(ctx: BehaviorContext, config: Any) -> Optional[Path]:
    cache = _get_cache(ctx)
    out_path = _stages_trial_table.stage_lag_features_impl(
        ctx,
        config,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        feature_suffix_from_context_fn=_feature_suffix_from_context,
        get_stats_subfolder_fn=_get_stats_subfolder,
        write_parquet_with_optional_csv_fn=_write_parquet_with_optional_csv,
        write_metadata_file_fn=_write_metadata_file,
        set_trial_table_cache_fn=lambda df: setattr(cache, "_trial_table_df", df),
    )
    if out_path is not None and out_path.exists():
        from eeg_pipeline.infra.tsv import read_table

        cache._trial_table_df = read_table(out_path)
    return out_path


def stage_predictor_residual(ctx: BehaviorContext, config: Any) -> Optional[Path]:
    cache = _get_cache(ctx)
    out_path = _stages_trial_table.stage_predictor_residual_impl(
        ctx,
        config,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        feature_suffix_from_context_fn=_feature_suffix_from_context,
        get_stats_subfolder_fn=_get_stats_subfolder,
        write_parquet_with_optional_csv_fn=_write_parquet_with_optional_csv,
        write_metadata_file_fn=_write_metadata_file,
        set_trial_table_cache_fn=lambda df: setattr(cache, "_trial_table_df", df),
    )
    if out_path is not None and out_path.exists():
        from eeg_pipeline.infra.tsv import read_table

        cache._trial_table_df = read_table(out_path)
    return out_path


def compute_temp_model_comparison(
    temperature: pd.Series,
    rating: pd.Series,
    config: Any,
) -> TempModelComparisonResult:
    """Compare temperature→rating model fits (linear vs polynomial vs spline)."""
    return _stages_models.compute_temp_model_comparison_impl(temperature, rating, config)


def compute_temp_breakpoints(
    temperature: pd.Series,
    rating: pd.Series,
    config: Any,
) -> TempBreakpointResult:
    """Detect threshold temperatures where sensitivity changes."""
    return _stages_models.compute_temp_breakpoints_impl(temperature, rating, config)


def write_temperature_models(
    ctx: BehaviorContext,
    model_comparison: Optional[TempModelComparisonResult],
    breakpoint: Optional[TempBreakpointResult],
) -> Path:
    """Write temperature model results to disk."""
    return _stages_models.write_temperature_models_impl(
        ctx,
        model_comparison,
        breakpoint,
        feature_suffix_from_context_fn=_feature_suffix_from_context,
        get_stats_subfolder_fn=_get_stats_subfolder,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        write_parquet_with_optional_csv_fn=_write_parquet_with_optional_csv,
    )


def stage_temperature_models(ctx: BehaviorContext, config: Any) -> Dict[str, Any]:
    """Compare temperature→rating model fits and test for breakpoints (composed)."""
    return _stages_models.stage_temperature_models_impl(
        ctx,
        config,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        compute_temp_model_comparison_fn=compute_temp_model_comparison,
        compute_temp_breakpoints_fn=compute_temp_breakpoints,
        write_temperature_models_fn=write_temperature_models,
    )


def _load_trial_table_df(ctx: BehaviorContext) -> Optional[pd.DataFrame]:
    """Load trial table with caching."""
    return _get_cache(ctx).get_trial_table(ctx)


def stage_regression(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    return _stages_models.stage_regression_impl(
        ctx,
        config,
        feature_suffix_from_context_fn=_feature_suffix_from_context,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        get_feature_columns_fn=lambda df, context: _get_feature_columns(df, context),
        check_early_exit_conditions_fn=_check_early_exit_conditions,
        sanitize_permutation_groups_fn=_sanitize_permutation_groups,
        attach_temperature_metadata_fn=_attach_temperature_metadata,
        get_stats_subfolder_fn=_get_stats_subfolder,
        write_stats_table_fn=_write_stats_table,
    )


def stage_models(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    return _stages_models.stage_models_impl(
        ctx,
        config,
        feature_suffix_from_context_fn=_feature_suffix_from_context,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        get_feature_columns_fn=lambda df, context: _get_feature_columns(df, context),
        check_early_exit_conditions_fn=_check_early_exit_conditions,
        attach_temperature_metadata_fn=_attach_temperature_metadata,
        get_stats_subfolder_fn=_get_stats_subfolder,
        write_stats_table_fn=_write_stats_table,
    )


def stage_stability(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Assess within-subject run/block stability of feature→outcome associations (non-gating)."""
    return _stages_diagnostics.stage_stability_impl(
        ctx,
        config,
        build_output_filename_fn=_build_output_filename,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        get_feature_columns_fn=lambda df, context: _get_feature_columns(df, context),
        check_early_exit_conditions_fn=_check_early_exit_conditions,
        get_stats_subfolder_fn=_get_stats_subfolder,
        write_stats_table_fn=_write_stats_table,
    )


def stage_icc(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Assess within-subject run-to-run reliability (ICC) of EEG features."""
    return _stages_diagnostics.stage_icc_impl(
        ctx,
        config,
        build_output_filename_fn=_build_output_filename,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        get_feature_columns_fn=lambda df, context: _get_feature_columns(df, context),
        check_early_exit_conditions_fn=_check_early_exit_conditions,
        get_stats_subfolder_fn=_get_stats_subfolder,
        write_stats_table_fn=_write_stats_table,
        write_metadata_file_fn=_write_metadata_file,
    )


def stage_consistency(ctx: BehaviorContext, config: Any, results: Any) -> pd.DataFrame:
    """Merge correlations/regression/models and flag effect-direction contradictions (non-gating)."""
    return _stages_diagnostics.stage_consistency_impl(
        ctx,
        config,
        results,
        build_output_filename_fn=_build_output_filename,
        get_stats_subfolder_fn=_get_stats_subfolder,
        write_stats_table_fn=_write_stats_table,
        write_metadata_file_fn=_write_metadata_file,
    )


def stage_influence(ctx: BehaviorContext, config: Any, results: Any) -> pd.DataFrame:
    """Compute leverage/Cook's summaries for top effects (non-gating)."""
    return _stages_diagnostics.stage_influence_impl(
        ctx,
        config,
        results,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        get_feature_columns_fn=lambda df, context: _get_feature_columns(df, context),
        check_early_exit_conditions_fn=_check_early_exit_conditions,
        attach_temperature_metadata_fn=_attach_temperature_metadata,
        get_stats_subfolder_fn=_get_stats_subfolder,
        build_output_filename_fn=_build_output_filename,
        write_stats_table_fn=_write_stats_table,
        write_metadata_file_fn=_write_metadata_file,
    )


def _compute_series_statistics(series: pd.Series) -> Dict[str, Any]:
    """Compute basic statistics for a numeric series."""
    return _stages_metadata.compute_series_statistics(series)


def build_behavior_qc(ctx: BehaviorContext) -> Dict[str, Any]:
    """Build behavior quality control summary."""
    return _stages_metadata.build_behavior_qc_impl(
        ctx,
        compute_series_statistics_fn=_compute_series_statistics,
        compute_correlation_fn=compute_correlation,
    )


def _infer_feature_type(feature: str, config: Any) -> str:
    return _feature_inference.infer_feature_type_impl(
        feature,
        config,
        feature_column_prefixes=FEATURE_COLUMN_PREFIXES,
    )


def _infer_feature_band(feature: str, config: Any) -> str:
    return _feature_inference.infer_feature_band_impl(feature, config)


def _build_result_cache() -> BehaviorResultCache:
    return BehaviorResultCache(
        feature_column_prefixes=FEATURE_COLUMN_PREFIXES,
        ensure_dir_fn=ensure_dir,
        trial_table_suffix_from_context_fn=_trial_table_suffix_from_context,
        trial_table_output_dir_fn=lambda ctx, ensure: _trial_table_output_dir(ctx, ensure=ensure),
        find_trial_table_path_fn=lambda stats_dir, feature_files: _find_trial_table_path(
            stats_dir, feature_files=feature_files
        ),
        validate_trial_table_contract_metadata_fn=_validate_trial_table_contract_metadata,
        filter_feature_cols_by_band_fn=_filter_feature_cols_by_band,
        filter_feature_cols_for_computation_fn=_filter_feature_cols_for_computation,
        filter_feature_cols_by_provenance_fn=_filter_feature_cols_by_provenance,
        infer_feature_type_fn=_infer_feature_type,
        infer_feature_band_fn=_infer_feature_band,
    )


def _summarize_covariates_qc(ctx: BehaviorContext) -> Dict[str, Any]:
    return _stages_metadata.summarize_covariates_qc_impl(ctx)


def write_analysis_metadata(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Any,
    stage_metrics: Optional[Dict[str, Any]] = None,
    outputs_manifest: Optional[Path] = None,
) -> Path:
    return _stages_metadata.write_analysis_metadata_impl(
        ctx,
        pipeline_config,
        results,
        stage_metrics=stage_metrics,
        outputs_manifest=outputs_manifest,
        build_behavior_qc_fn=build_behavior_qc,
        summarize_covariates_qc_fn=_summarize_covariates_qc,
        get_stats_subfolder_fn=_get_stats_subfolder,
    )


###################################################################
# Condition Stage - Single Responsibility Components
###################################################################


def _resolve_condition_compare_column(df_trials: pd.DataFrame, config: Any) -> str:
    """Resolve configured condition column, falling back to configured pain column."""
    return _stages_condition.resolve_condition_compare_column(df_trials, config)


def stage_condition_column(
    ctx: BehaviorContext,
    config: Any,
    df_trials: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run column-based condition comparison (e.g., pain vs non-pain).
    
    Single responsibility: Column contrast comparison.
    Supports primary_unit=trial|run to control unit of analysis.
    
    When overwrite=false, includes compare_column name in output filename to allow
    multiple comparisons without overwriting previous results.
    
    If compare_values has 3+ values, delegates to multigroup comparison instead.
    """
    return _stages_condition.stage_condition_column_impl(
        ctx,
        config,
        df_trials=df_trials,
        feature_cols=feature_cols,
        stage_condition_multigroup_fn=stage_condition_multigroup,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        get_feature_columns_fn=_get_feature_columns,
        check_early_exit_conditions_fn=_check_early_exit_conditions,
        feature_suffix_from_context_fn=_feature_suffix_from_context,
        get_stats_subfolder_fn=_get_stats_subfolder,
        sanitize_permutation_groups_fn=_sanitize_permutation_groups,
        compute_unified_fdr_fn=_compute_unified_fdr,
        write_parquet_with_optional_csv_fn=_write_parquet_with_optional_csv,
        resolve_condition_compare_column_fn=_resolve_condition_compare_column,
        unified_fdr_family_columns=UNIFIED_FDR_FAMILY_COLUMNS,
    )



def _compute_pairwise_effect_sizes(
    v1: np.ndarray,
    v2: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """Compute paired (within-subject) effect sizes (Cohen's dz and Hedge's gz).

    Returns:
        (mean_diff, std_diff, cohens_d, hedges_g, hedges_correction)
    """
    diff = v2 - v1
    mean_diff = float(np.nanmean(diff))
    std_diff = float(np.nanstd(diff, ddof=1))
    cohens_d = mean_diff / std_diff if np.isfinite(std_diff) and std_diff > 0 else np.nan

    # Hedge's g correction
    n = int(np.sum(np.isfinite(v1) & np.isfinite(v2)))
    hedges_correction = 1 - (3 / (4 * n - 1)) if n > 1 else 1.0
    hedges_g = cohens_d * hedges_correction if np.isfinite(cohens_d) else np.nan

    return mean_diff, std_diff, cohens_d, hedges_g, hedges_correction


def stage_condition_window(
    ctx: BehaviorContext,
    config: Any,
    df_trials: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    compare_windows: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run window-based condition comparison (e.g., baseline vs active).
    
    Single responsibility: Window contrast comparison.
    """

    return _stages_condition.stage_condition_window_impl(
        ctx,
        config,
        df_trials=df_trials,
        feature_cols=feature_cols,
        compare_windows=compare_windows,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        get_feature_columns_fn=_get_feature_columns,
        check_early_exit_conditions_fn=_check_early_exit_conditions,
        feature_suffix_from_context_fn=_feature_suffix_from_context,
        resolve_condition_compare_column_fn=_resolve_condition_compare_column,
        get_stats_subfolder_fn=_get_stats_subfolder,
        run_window_comparison_fn=_run_window_comparison,
        write_parquet_with_optional_csv_fn=_write_parquet_with_optional_csv,
    )


def stage_condition(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Backward-compatible condition stage (column + optional window + optional multigroup).
    
    The pipeline wrapper historically called a single stage and expected a DataFrame.
    Internally, we keep single-responsibility sub-stages:
    - stage_condition_column (2-group comparison)
    - stage_condition_window (paired window comparison)
    - stage_condition_multigroup (3+ group comparison)
    """
    cache = _get_cache(ctx)
    return _stages_condition.stage_condition_impl(
        ctx,
        config,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        get_filtered_feature_cols_fn=lambda df, context: cache.get_filtered_feature_cols(
            [c for c in df.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)],
            context,
            "condition",
        ),
        stage_condition_multigroup_fn=stage_condition_multigroup,
        stage_condition_column_fn=stage_condition_column,
        stage_condition_window_fn=stage_condition_window,
    )


def stage_condition_multigroup(
    ctx: BehaviorContext,
    config: Any,
    df_trials: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run multi-group condition comparison (3+ groups).
    
    Computes all pairwise Mann-Whitney U tests between groups with FDR correction.
    Results are saved to condition_effects_multigroup*.tsv.
    
    When overwrite=false, includes compare_column name in output filename to allow
    multiple comparisons without overwriting previous results.
    """
    return _stages_condition.stage_condition_multigroup_impl(
        ctx,
        config,
        df_trials=df_trials,
        feature_cols=feature_cols,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        get_feature_columns_fn=_get_feature_columns,
        resolve_condition_compare_column_fn=_resolve_condition_compare_column,
        compute_unified_fdr_fn=_compute_unified_fdr,
        feature_suffix_from_context_fn=_feature_suffix_from_context,
        get_stats_subfolder_fn=_get_stats_subfolder,
        write_stats_table_fn=_write_stats_table,
        unified_fdr_family_columns=UNIFIED_FDR_FAMILY_COLUMNS,
    )


def _run_window_comparison(
    ctx: BehaviorContext,
    df_trials: pd.DataFrame,
    feature_cols: List[str],
    windows: List[str],
    min_samples: int,
    fdr_alpha: float,
    suffix: str,
) -> pd.DataFrame:
    """Run paired window comparison on feature columns."""
    cache = _get_cache(ctx)
    return _stages_condition.run_window_comparison_impl(
        ctx,
        df_trials,
        feature_cols,
        windows,
        min_samples,
        fdr_alpha,
        suffix,
        feature_prefixes=FEATURE_COLUMN_PREFIXES,
        compute_pairwise_effect_sizes_fn=_compute_pairwise_effect_sizes,
        feature_type_resolver_fn=lambda feature_name, cfg: cache.get_feature_type(feature_name, cfg),
        compute_unified_fdr_fn=_compute_unified_fdr,
        unified_fdr_family_columns=UNIFIED_FDR_FAMILY_COLUMNS,
    )


###################################################################
# Temporal Stage - Single Responsibility Components
###################################################################


def stage_temporal_tfr(ctx: BehaviorContext) -> Optional[Dict[str, Any]]:
    """Compute time-frequency representation correlations."""
    return _stages_temporal.stage_temporal_tfr_impl(ctx)


def _resolve_temporal_feature_selection(
    ctx: BehaviorContext,
    selected_features: Optional[List[str]] = None,
) -> List[str]:
    """Resolve effective temporal features from config toggles and user filters."""
    return _stages_temporal.resolve_temporal_feature_selection_impl(ctx, selected_features)


def stage_temporal_stats(
    ctx: BehaviorContext,
    selected_features: Optional[List[str]] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Compute temporal statistics (power, ITPC, ERDS correlations)."""
    return _stages_temporal.stage_temporal_stats_impl(
        ctx,
        selected_features=selected_features,
        resolve_temporal_feature_selection_fn=_resolve_temporal_feature_selection,
        sanitize_permutation_groups_fn=_sanitize_permutation_groups,
        get_stats_subfolder_fn=_get_stats_subfolder,
        write_stats_table_fn=_write_stats_table,
        format_correlation_method_label_fn=format_correlation_method_label,
    )




def stage_cluster(ctx: BehaviorContext, config: Any) -> Dict[str, Any]:
    return _stages_temporal.stage_cluster_impl(ctx, config)


###################################################################
# Advanced Stage - Single Responsibility Components
###################################################################


def stage_mediation(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Run mediation analysis: test if neural features mediate the temperature→rating relationship."""
    cache = _get_cache(ctx)
    return _stages_advanced.stage_mediation_impl(
        ctx,
        config,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        get_feature_columns_fn=_get_feature_columns,
        check_early_exit_conditions_fn=_check_early_exit_conditions,
        sanitize_permutation_groups_fn=_sanitize_permutation_groups,
        feature_type_resolver_fn=lambda feature_name, cfg: cache.get_feature_type(str(feature_name), cfg),
        compute_unified_fdr_fn=_compute_unified_fdr,
        unified_fdr_family_columns=UNIFIED_FDR_FAMILY_COLUMNS,
    )


def stage_mixed_effects(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Run mixed-effects analysis."""
    return _stages_advanced.stage_mixed_effects_impl(ctx, config)


###################################################################
# Group-Level Analysis (Multi-Subject)
###################################################################


def run_group_level_mixed_effects(
    subjects: List[str],
    deriv_root: Path,
    config: Any,
    logger: Any,
    random_effects: str = "intercept",
    max_features: int = 50,
    fdr_alpha: float = 0.05,
) -> MixedEffectsResult:
    """Run proper mixed-effects models across all subjects."""
    return _group_level.run_group_level_mixed_effects_impl(
        subjects=subjects,
        deriv_root=deriv_root,
        config=config,
        logger=logger,
        random_effects=random_effects,
        max_features=max_features,
        fdr_alpha=fdr_alpha,
        find_trial_table_path_fn=_find_trial_table_path,
        feature_prefixes=FEATURE_COLUMN_PREFIXES,
        feature_type_resolver=lambda feature_name, cfg: _infer_feature_type(str(feature_name), cfg),
    )


def run_group_level_correlations(
    subjects: List[str],
    deriv_root: Path,
    config: Any,
    logger: Any,
    use_block_permutation: bool = True,
    n_perm: int = 1000,
    fdr_alpha: float = 0.05,
    target_col: str = "rating",
    control_temperature: bool = False,
    control_trial_order: bool = False,
    control_run_effects: bool = False,
    max_run_dummies: int = 20,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Run multilevel correlations across subjects with block-aware permutations."""
    return _group_level.run_group_level_correlations_impl(
        subjects=subjects,
        deriv_root=deriv_root,
        config=config,
        logger=logger,
        use_block_permutation=use_block_permutation,
        n_perm=n_perm,
        fdr_alpha=fdr_alpha,
        target_col=target_col,
        control_temperature=control_temperature,
        control_trial_order=control_trial_order,
        control_run_effects=control_run_effects,
        max_run_dummies=max_run_dummies,
        random_state=random_state,
        find_trial_table_path_fn=_find_trial_table_path,
        feature_prefixes=FEATURE_COLUMN_PREFIXES,
        feature_type_resolver=lambda feature_name, cfg: _infer_feature_type(str(feature_name), cfg),
        constant_variance_threshold=CONSTANT_VARIANCE_THRESHOLD,
    )


def run_group_level_analysis(
    subjects: List[str],
    deriv_root: Path,
    config: Any,
    logger: Any,
    run_mixed_effects: bool = False,
    run_multilevel_correlations: bool = False,
    output_dir: Optional[Path] = None,
) -> GroupLevelResult:
    """Run all group-level analyses."""
    return _group_level.run_group_level_analysis_impl(
        subjects=subjects,
        deriv_root=deriv_root,
        config=config,
        logger=logger,
        run_mixed_effects=run_mixed_effects,
        run_multilevel_correlations=run_multilevel_correlations,
        output_dir=output_dir,
        run_mixed_effects_fn=run_group_level_mixed_effects,
        run_multilevel_correlations_fn=run_group_level_correlations,
        write_parquet_with_optional_csv_fn=_write_parquet_with_optional_csv,
        also_save_csv_from_config_fn=_also_save_csv_from_config,
    )


def stage_moderation(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Run moderation analysis: test if neural features moderate the temperature→rating relationship.

    Model: rating = b0 + b1*temperature + b2*feature + b3*(temperature*feature) + error

    If b3 is significant, the feature moderates how temperature affects pain rating.
    """
    cache = _get_cache(ctx)
    return _stages_advanced.stage_moderation_impl(
        ctx,
        config,
        load_trial_table_df_fn=_load_trial_table_df,
        is_dataframe_valid_fn=_common_helpers.is_dataframe_valid_impl,
        get_feature_columns_fn=_get_feature_columns,
        check_early_exit_conditions_fn=_check_early_exit_conditions,
        sanitize_permutation_groups_fn=_sanitize_permutation_groups,
        feature_type_resolver_fn=lambda feature_name, cfg: cache.get_feature_type(str(feature_name), cfg),
        compute_unified_fdr_fn=_compute_unified_fdr,
        get_stats_subfolder_fn=_get_stats_subfolder,
        feature_suffix_from_context_fn=_feature_suffix_from_context,
        write_parquet_with_optional_csv_fn=_write_parquet_with_optional_csv,
        unified_fdr_family_columns=UNIFIED_FDR_FAMILY_COLUMNS,
    )


def stage_hierarchical_fdr_summary(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Compute hierarchical FDR summary across analysis types from cached FDR results."""
    cache = _get_cache(ctx)
    return _stages_fdr.stage_hierarchical_fdr_summary_impl(
        ctx,
        config,
        get_cached_fdr_fn=cache.get_fdr_results,
        get_stats_subfolder_fn=_get_stats_subfolder,
        write_parquet_with_optional_csv_fn=_write_parquet_with_optional_csv,
    )




def stage_report(ctx: BehaviorContext, pipeline_config: Any) -> Path:
    """Write a single-subject, self-diagnosing Markdown report (fail-fast)."""
    return _stages_report.stage_report_impl(
        ctx,
        pipeline_config,
        feature_suffix_from_context_fn=_feature_suffix_from_context,
        get_config_int_fn=get_config_int,
        load_trial_table_df_fn=_load_trial_table_df,
        get_stats_subfolder_fn=_get_stats_subfolder,
        feature_prefixes=FEATURE_COLUMN_PREFIXES,
    )


def _build_output_filename(
    ctx: BehaviorContext,
    pipeline_config: Any,
    base_name: str,
) -> str:
    """Build standardized output filename with feature and method suffixes."""
    feature_suffix = _feature_suffix_from_context(ctx)
    method_label = getattr(pipeline_config, "method_label", "")
    return _stages_export.build_output_filename(feature_suffix, method_label, base_name)


def _write_metadata_file(path: Path, metadata: Dict[str, Any]) -> None:
    _common_helpers.write_metadata_file_impl(path, metadata)


def stage_export(ctx: BehaviorContext, pipeline_config: Any, results: Any) -> List[Path]:
    """Export all analysis results to disk with normalization."""
    return _stages_export.stage_export_impl(
        ctx,
        pipeline_config,
        results,
        get_stats_subfolder_fn=_get_stats_subfolder,
        write_stats_table_fn=_write_stats_table,
        build_output_filename_fn=_build_output_filename,
    )


def write_outputs_manifest(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Any,
    stage_metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    _ = results  # kept for backwards-compatible signature
    return _stages_export.write_outputs_manifest_impl(
        ctx,
        pipeline_config,
        stage_metrics=stage_metrics,
        get_stats_subfolder_fn=_get_stats_subfolder,
        feature_folder_from_context_fn=_feature_folder_from_context,
    )
