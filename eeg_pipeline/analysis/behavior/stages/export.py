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
    ("condition_effects", "condition_effects"),
    ("icc_reliability", "icc_reliability"),
    ("regression_feature_effects", "trialwise_regression"),
    ("trials_with_residual", "predictor_residual"),
    ("predictor_residual", "predictor_residual"),
    ("trials", "trial_table"),
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


def _append_existing_output(saved: List[Path], out_dir: Path, filename: str) -> None:
    for suffix in (".parquet", ".tsv", ".csv"):
        path = out_dir / f"{filename}{suffix}"
        if path.exists():
            saved.append(path)
            return


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

    if is_valid_df(getattr(results, "condition_effects", None)):
        out_dir = get_stats_subfolder_fn(ctx, "condition_effects")
        filename = build_output_filename_fn(ctx, pipeline_config, "condition_effects")
        saved.append(write_stats_table_fn(ctx, results.condition_effects, out_dir / f"{filename}.tsv"))

    if is_valid_df(getattr(results, "icc", None)):
        out_dir = get_stats_subfolder_fn(ctx, "icc_reliability")
        filename = build_output_filename_fn(ctx, pipeline_config, "icc_reliability")
        _append_existing_output(saved, out_dir, filename)

    if is_valid_df(getattr(results, "regression", None)):
        out_dir = get_stats_subfolder_fn(ctx, "trialwise_regression")
        filename = build_output_filename_fn(ctx, pipeline_config, "regression_feature_effects")
        _append_existing_output(saved, out_dir, filename)

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
            "outcome": bool(ctx._find_outcome_column() is not None) if hasattr(ctx, "_find_outcome_column") else False,
            "predictor": bool(ctx.predictor_series is not None and ctx.predictor_series.notna().any()) if ctx.predictor_series is not None else False,
        },
        "covariates_qc": ctx.data_qc.get("covariates_qc", {}),
        "outputs": outputs,
    }

    if stage_metrics:
        payload["stage_metrics"] = stage_metrics

    manifest_path.write_text(json.dumps(payload, indent=2, default=str))
    return manifest_path
