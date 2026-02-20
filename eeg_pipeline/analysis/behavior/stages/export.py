from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def build_output_filename(
    feature_suffix: str,
    method_label: str,
    base_name: str,
) -> str:
    """Build standardized output filename with feature and method suffixes."""
    method_suffix = f"_{method_label}" if method_label else ""
    return f"{base_name}{feature_suffix}{method_suffix}"


def is_valid_df(obj: Any) -> bool:
    """Check if obj is a valid non-empty DataFrame (not a failed stage result dict)."""
    if obj is None:
        return False
    if isinstance(obj, dict):
        return False
    if not hasattr(obj, "empty"):
        return False
    return not obj.empty


OUTPUT_KIND_PATTERNS = [
    ("corr_stats_", "correlations"),
    ("correlations", "correlations"),
    ("pain_sensitivity", "pain_sensitivity"),
    ("condition_effects", "condition_effects"),
    ("mediation", "mediation"),
    ("moderation", "moderation"),
    ("mixed_effects", "mixed_effects"),
    ("regression_feature_effects", "trialwise_regression"),
    ("models_feature_effects", "feature_models"),
    ("trials_with_lags", "lag_features"),
    ("trials_with_residual", "pain_residual"),
    ("lag_features", "lag_features"),
    ("pain_residual", "pain_residual"),
    ("model_comparison", "temperature_models"),
    ("breakpoint_candidates", "temperature_models"),
    ("breakpoint_test", "temperature_models"),
    ("trials", "trial_table"),
    ("temperature_model_comparison", "temperature_model_comparison"),
    ("temperature_breakpoint", "temperature_breakpoint_test"),
    ("stability_groupwise", "stability_groupwise"),
    ("consistency_summary", "consistency_summary"),
    ("influence_diagnostics", "influence_diagnostics"),
    ("normalized_results", "normalized"),
    ("feature_screening", "feature_screening"),
    ("paired_comparisons", "paired_comparisons"),
    ("summary", "summary"),
    ("analysis_metadata", "analysis_metadata"),
    ("subject_report", "subject_report"),
    ("tf_grid", "time_frequency"),
    ("temporal_correlations", "temporal_correlations"),
    ("hierarchical_fdr_summary", "fdr"),
]


def infer_output_kind(name: str) -> str:
    for prefix, kind in OUTPUT_KIND_PATTERNS:
        if name.startswith(prefix):
            return kind
    return "unknown"


def count_rows(path: Path) -> Optional[int]:
    if path.suffix not in {".tsv", ".csv"}:
        return None
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            return 0
        return sum(1 for _ in f)


def stage_export_impl(
    ctx: Any,
    pipeline_config: Any,
    results: Any,
    *,
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    write_stats_table_fn: Callable[[Any, Any, Path], Path],
    build_output_filename_fn: Callable[[Any, Any, str], str],
) -> List[Path]:
    """Export all analysis results to disk with normalization."""
    from eeg_pipeline.infra.paths import ensure_dir

    ensure_dir(ctx.stats_dir)
    saved: List[Path] = []

    if is_valid_df(getattr(results, "correlations", None)):
        out_dir = get_stats_subfolder_fn(ctx, "correlations")
        filename = build_output_filename_fn(ctx, pipeline_config, "correlations")
        saved.append(write_stats_table_fn(ctx, results.correlations, out_dir / f"{filename}.tsv"))

    if is_valid_df(getattr(results, "pain_sensitivity", None)):
        out_dir = get_stats_subfolder_fn(ctx, "pain_sensitivity")
        filename = build_output_filename_fn(ctx, pipeline_config, "pain_sensitivity")
        saved.append(write_stats_table_fn(ctx, results.pain_sensitivity, out_dir / f"{filename}.tsv"))

    if is_valid_df(getattr(results, "condition_effects", None)):
        out_dir = get_stats_subfolder_fn(ctx, "condition_effects")
        filename = build_output_filename_fn(ctx, pipeline_config, "condition_effects")
        saved.append(write_stats_table_fn(ctx, results.condition_effects, out_dir / f"{filename}.tsv"))

    if is_valid_df(getattr(results, "mediation", None)):
        out_dir = get_stats_subfolder_fn(ctx, "mediation")
        filename = build_output_filename_fn(ctx, pipeline_config, "mediation")
        saved.append(write_stats_table_fn(ctx, results.mediation, out_dir / f"{filename}.tsv"))

    if is_valid_df(getattr(results, "mixed_effects", None)):
        out_dir = get_stats_subfolder_fn(ctx, "mixed_effects")
        filename = build_output_filename_fn(ctx, pipeline_config, "mixed_effects")
        saved.append(write_stats_table_fn(ctx, results.mixed_effects, out_dir / f"{filename}.tsv"))

    if is_valid_df(getattr(results, "regression", None)):
        out_dir = get_stats_subfolder_fn(ctx, "trialwise_regression")
        filename = build_output_filename_fn(ctx, pipeline_config, "regression_feature_effects")
        path = out_dir / f"{filename}.tsv"
        if path.exists():
            saved.append(path)

    if is_valid_df(getattr(results, "models", None)):
        out_dir = get_stats_subfolder_fn(ctx, "feature_models")
        filename = build_output_filename_fn(ctx, pipeline_config, "models_feature_effects")
        path = out_dir / f"{filename}.tsv"
        if path.exists():
            saved.append(path)

    return saved


def write_outputs_manifest_impl(
    ctx: Any,
    pipeline_config: Any,
    stage_metrics: Optional[Dict[str, Any]] = None,
    *,
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    feature_folder_from_context_fn: Callable[[Any], str],
) -> Path:
    feature_folder = feature_folder_from_context_fn(ctx)
    out_dir = get_stats_subfolder_fn(ctx, "summary")
    manifest_path = out_dir / "outputs_manifest.json"

    outputs = []
    for path in sorted(p for p in ctx.stats_dir.rglob("*") if p.is_file()):
        if path.name.startswith(".") or path.suffix == ".log" or path.name == "outputs_manifest.json":
            continue
        rel = path.relative_to(ctx.stats_dir)
        parts = rel.parts
        if len(parts) < 2 or parts[1] != feature_folder:
            continue
        outputs.append(
            {
                "name": path.name,
                "path": str(path),
                "kind": infer_output_kind(path.name),
                "subfolder": str(path.parent.relative_to(ctx.stats_dir)),
                "rows": count_rows(path),
                "size_bytes": int(path.stat().st_size),
                "method_label": pipeline_config.method_label,
            }
        )

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
            "rating": bool(ctx._find_rating_column() is not None) if hasattr(ctx, "_find_rating_column") else False,
            "temperature": bool(ctx.temperature is not None and ctx.temperature.notna().any()) if ctx.temperature is not None else False,
        },
        "covariates_qc": ctx.data_qc.get("covariates_qc", {}),
        "outputs": outputs,
    }

    if stage_metrics:
        payload["stage_metrics"] = stage_metrics

    manifest_path.write_text(json.dumps(payload, indent=2, default=str))
    return manifest_path
