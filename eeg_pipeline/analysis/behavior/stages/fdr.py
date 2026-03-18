from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import pandas as pd


def compute_unified_fdr_impl(
    ctx: Any,
    config: Any,
    df: pd.DataFrame,
    *,
    p_col: str = "p_primary",
    family_cols: Optional[Sequence[str]] = None,
    analysis_type: str = "correlations",
    get_cached_fdr_fn: Callable[[], Any],
    set_cached_fdr_fn: Callable[[Any], None],
) -> pd.DataFrame:
    """Compute unified FDR corrections (within-family, hierarchical, global) in one pass."""
    from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh, hierarchical_fdr
    from eeg_pipeline.utils.config.loader import require_config_value

    if df.empty or p_col not in df.columns:
        return df

    df = df.copy()
    fdr_alpha_value = getattr(config, "fdr_alpha", None)
    if fdr_alpha_value is None:
        raise ValueError("Missing fdr_alpha in behavior pipeline configuration.")
    fdr_alpha = float(fdr_alpha_value)
    use_hierarchical_fdr = bool(
        require_config_value(ctx.config, "behavior_analysis.statistics.hierarchical_fdr")
    )

    if family_cols is None:
        family_cols_list: List[str] = ["feature_type", "band", "target", "analysis_kind"]
    else:
        family_cols_list = [str(col) for col in family_cols]
    family_cols_list = [col for col in family_cols_list if col in df.columns]

    if use_hierarchical_fdr and family_cols_list:
        df["fdr_family"] = df[family_cols_list].astype(str).agg("_".join, axis=1)
        df = hierarchical_fdr(
            df,
            p_col=p_col,
            family_col="fdr_family",
            alpha=fdr_alpha,
            config=ctx.config,
        )

        q_within = pd.to_numeric(
            df.get("q_within_family", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        gate_pass = df.get("family_reject_gate", pd.Series(True, index=df.index)).fillna(False).astype(bool)
        df["p_fdr"] = np.where(gate_pass.to_numpy(), q_within.to_numpy(), 1.0)

        family_stats = []
        for family_id, family_group in df.groupby("fdr_family"):
            fam_reject = family_group.get("reject_within_family", pd.Series(False, index=family_group.index))
            fam_gate = family_group.get("family_reject_gate", pd.Series(False, index=family_group.index))
            family_stats.append(
                {
                    "family": family_id,
                    "n_tests": int(len(family_group)),
                    "n_significant": int(pd.Series(fam_reject).fillna(False).astype(bool).sum()),
                    "gate_rejected": bool(pd.Series(fam_gate).fillna(False).astype(bool).iloc[0]),
                }
            )

        reject_within = df.get("reject_within_family", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        q_global = pd.to_numeric(
            df.get("q_global", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        fdr_metadata = {
            "family_columns": family_cols_list,
            "n_families": len(family_stats),
            "family_stats": family_stats,
            "hierarchical": True,
            "analysis_type": analysis_type,
            "n_total_tests": len(df),
            "n_sig_global": int((q_global < fdr_alpha).sum()),
            "n_sig_within": int(reject_within.sum()),
        }
        ctx.data_qc["fdr_family_structure"] = fdr_metadata

        cached_fdr = get_cached_fdr_fn() or {}
        cached_fdr[analysis_type] = {"df": df, "metadata": fdr_metadata}
        set_cached_fdr_fn(cached_fdr)

        n_sig_total = int(reject_within.sum())
        ctx.logger.info(
            "Unified FDR [%s]: %d/%d significant within families, %d/%d globally (alpha=%.2f)",
            analysis_type,
            n_sig_total,
            len(df),
            fdr_metadata["n_sig_global"],
            len(df),
            fdr_alpha,
        )
    else:
        p_vals = pd.to_numeric(df[p_col], errors="coerce").to_numpy()
        df["q_global"] = fdr_bh(p_vals, alpha=fdr_alpha, config=ctx.config)
        df["p_fdr"] = df["q_global"]
        df["fdr_family"] = "all"

        fdr_metadata = {
            "family_columns": [],
            "n_families": 1,
            "hierarchical": False,
            "analysis_type": analysis_type,
            "n_total_tests": len(df),
            "n_sig_global": int((df["q_global"] < fdr_alpha).sum()),
            "n_sig_within": int((df["p_fdr"] < fdr_alpha).sum()),
        }
        ctx.data_qc["fdr_family_structure"] = fdr_metadata

        cached_fdr = get_cached_fdr_fn() or {}
        cached_fdr[analysis_type] = {"df": df, "metadata": fdr_metadata}
        set_cached_fdr_fn(cached_fdr)

        n_sig = int((df["p_fdr"] < fdr_alpha).sum())
        ctx.logger.info(
            "Unified FDR [%s]: %d/%d significant (flat, alpha=%.2f)",
            analysis_type,
            n_sig,
            len(df),
            fdr_alpha,
        )

    return df


def stage_hierarchical_fdr_summary_impl(
    ctx: Any,
    config: Any,
    *,
    get_cached_fdr_fn: Callable[[], Any],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    write_parquet_with_optional_csv_fn: Callable[[pd.DataFrame, Path, bool], None],
) -> pd.DataFrame:
    """Compute hierarchical FDR summary across analysis types from cached FDR results."""
    _ = config

    cached_fdr = get_cached_fdr_fn()
    if not cached_fdr:
        ctx.logger.info("No cached FDR results found; skipping hierarchical FDR summary.")
        return pd.DataFrame()

    ctx.logger.info("Computing hierarchical FDR summary from cached results...")

    summary_records = []
    for analysis_type, fdr_data in cached_fdr.items():
        metadata = fdr_data.get("metadata", {})
        if metadata:
            summary_records.append(
                {
                    "analysis_type": analysis_type,
                    "n_tests": metadata.get("n_total_tests", 0),
                    "n_reject_within": metadata.get("n_sig_within", 0),
                    "n_reject_global": metadata.get("n_sig_global", 0),
                    "pct_reject_within": (
                        100 * metadata.get("n_sig_within", 0) / metadata.get("n_total_tests", 1)
                        if metadata.get("n_total_tests", 0) > 0
                        else 0
                    ),
                    "pct_reject_global": (
                        100 * metadata.get("n_sig_global", 0) / metadata.get("n_total_tests", 1)
                        if metadata.get("n_total_tests", 0) > 0
                        else 0
                    ),
                    "n_families": metadata.get("n_families", 0),
                    "hierarchical": metadata.get("hierarchical", False),
                }
            )

    if not summary_records:
        ctx.logger.warning("No FDR metadata found in cache.")
        return pd.DataFrame()

    hier_summary = pd.DataFrame(summary_records)

    if not hier_summary.empty:
        hier_dir = get_stats_subfolder_fn(ctx, "fdr")
        hier_path = hier_dir / "hierarchical_fdr_summary.parquet"
        write_parquet_with_optional_csv_fn(hier_summary, hier_path, also_save_csv=ctx.also_save_csv)
        ctx.logger.info("Hierarchical FDR summary saved to %s", hier_path)

        for _, row in hier_summary.iterrows():
            ctx.logger.info(
                "  %s: %s/%s (%.1f%%) reject within-family",
                row["analysis_type"],
                row["n_reject_within"],
                row["n_tests"],
                row["pct_reject_within"],
            )

    return hier_summary
