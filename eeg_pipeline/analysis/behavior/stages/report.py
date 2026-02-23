from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd


def stage_report_impl(
    ctx: Any,
    pipeline_config: Any,
    *,
    feature_suffix_from_context_fn: Callable[[Any], str],
    get_config_int_fn: Callable[[Any, str, int], int],
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    feature_prefixes: Sequence[str],
) -> Path:
    """Write a single-subject, self-diagnosing Markdown report (fail-fast)."""
    suffix = feature_suffix_from_context_fn(ctx)
    method_label = getattr(pipeline_config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    top_n = get_config_int_fn(ctx.config, "behavior_analysis.report.top_n", 15)
    alpha = float(getattr(pipeline_config, "fdr_alpha", 0.05))

    df_trials = load_trial_table_df_fn(ctx)
    n_trials = int(len(df_trials)) if df_trials is not None else 0
    n_features = 0
    if df_trials is not None and not df_trials.empty:
        n_features = int(sum(1 for c in df_trials.columns if str(c).startswith(tuple(feature_prefixes))))

    def _read_tsv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, sep="\t")

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
        keep = [
            c
            for c in ["feature", "target", "r_primary", "beta_feature", "hedges_g", "p_primary", "p_fdr", "q_global"]
            if c in out.columns
        ]
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
        "correlations*.parquet",
        "predictor_sensitivity*.parquet",
        "regression_feature_effects*.parquet",
        "models_feature_effects*.parquet",
        "condition_effects*.parquet",
        "consistency_summary*.parquet",
        "influence_diagnostics*.parquet",
        "temperature_model_comparison*.parquet",
        "temperature_breakpoint_candidates*.parquet",
        "hierarchical_fdr_summary.parquet",
        "normalized_results*.parquet",
        "summary.json",
        "analysis_metadata.json",
        "outputs_manifest.json",
    ]

    files: List[Path] = []
    for pat in patterns:
        found = sorted(ctx.stats_dir.rglob(pat))
        files.extend(found)
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
    lines.append(
        f"- Controls: predictor=`{bool(getattr(pipeline_config, 'control_temperature', True))}`, trial_order=`{bool(getattr(pipeline_config, 'control_trial_order', True))}`"
    )
    outcome_col = ctx._find_rating_column() if hasattr(ctx, "_find_rating_column") else None
    predictor_col = getattr(ctx, "temperature_column", None)
    lines.append(f"- Outcome column: `{outcome_col or 'auto'}`")
    lines.append(f"- Predictor column: `{predictor_col or 'auto'}`")
    lines.append(f"- Global FDR alpha: `{alpha}`")

    tsvs = [p for p in files if p.suffix == ".tsv"]
    if tsvs:
        lines.append("")
        lines.append("## Outputs")
        for p in tsvs:
            df = _read_tsv(p)
            if df.empty:
                lines.append(f"- `{p.name}`: (empty)")
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

    report_dir = get_stats_subfolder_fn(ctx, "subject_report")
    out_path = report_dir / f"subject_report{suffix}{method_suffix}.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ctx.logger.info("Subject report saved: %s/%s", report_dir.name, out_path.name)
    return out_path
