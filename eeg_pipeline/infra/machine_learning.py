from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mne
import numpy as np
import pandas as pd
from scipy.stats import norm

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.infra.logging import get_logger
from eeg_pipeline.infra.paths import (
    deriv_stats_path,
    ensure_dir,
    find_clean_epochs_path,
)
from eeg_pipeline.infra.tsv import read_parquet, write_parquet
from eeg_pipeline.infra.tsv import write_tsv
from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh

logger = get_logger(__name__)


###################################################################
# Run Manifest and Logging
###################################################################

def _get_git_commit_hash() -> Optional[str]:
    """Retrieve the current git commit hash."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent.parent,
            text=True,
        )
        return output.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _get_package_versions() -> Dict[str, Optional[str]]:
    """Collect versions of key scientific computing packages."""
    versions = {
        "sklearn": None,
        "mne": mne.__version__,
        "pyriemann": None,
    }

    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except ImportError:
        pass

    try:
        import pyriemann

        versions["pyriemann"] = pyriemann.__version__
    except ImportError:
        pass

    return versions


def _get_environment_info() -> Dict[str, Any]:
    """Collect system environment information."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }


def _get_thread_limits() -> Dict[str, Optional[str]]:
    """Collect thread limit environment variables."""
    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
    }


def create_run_manifest(
    results_dir: Path,
    cli_args: dict,
    config: dict,
    run_id: Optional[str] = None,
) -> None:
    """Create a manifest file documenting the run configuration and environment."""
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "cli_args": cli_args,
        "resolved_config": config,
        "environment": _get_environment_info(),
        "thread_limits": _get_thread_limits(),
        "git_commit": _get_git_commit_hash(),
        "package_versions": _get_package_versions(),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = results_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


###################################################################
# Best Parameters I/O
###################################################################

def prepare_best_params_path(
    base_path: Path,
    mode: str,
    run_id: Optional[str] = None,
) -> Path:
    """Prepare the path for best parameters output based on mode."""
    base_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "run_scoped":
        resolved_run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
        output_path = base_path.with_name(
            f"{base_path.stem}_{resolved_run_id}{base_path.suffix}"
        )
    else:
        if mode == "truncate":
            base_path.unlink(missing_ok=True)
        output_path = base_path

    logger.info(f"Best-params mode='{mode}'; resolved path: {output_path}")
    return output_path


###################################################################
# Predictions and Indices Export
###################################################################

def export_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups_ordered: List[str],
    test_indices: List[int],
    fold_ids: List[int],
    model_name: str,
    meta: pd.DataFrame,
    save_path: Path,
) -> pd.DataFrame:
    extra_cols = {}
    if "run_id" in meta.columns:
        extra_cols["run_id"] = meta.loc[test_indices, "run_id"].tolist()
    if "block" in meta.columns:
        extra_cols["block"] = meta.loc[test_indices, "block"].tolist()

    pred_df = pd.DataFrame(
        {
            "subject_id": groups_ordered,
            "trial_id": meta.loc[test_indices, "trial_id"].values,
            "y_true": y_true,
            "y_pred": y_pred,
            "fold": fold_ids,
            "model": model_name,
            **extra_cols,
        }
    )

    # Write both TSV (for plotting/back-compat) and parquet (for fast downstream use).
    write_tsv(pred_df, save_path.with_suffix(".tsv"))
    write_parquet(pred_df, save_path.with_suffix(".parquet"))
    return pred_df


def export_indices(
    groups_ordered: List[str],
    test_indices: List[int],
    fold_ids: List[int],
    meta: pd.DataFrame,
    save_path: Path,
    blocks_source: Optional[str] = None,
    add_heldout_subject_id: bool = False,
) -> None:
    extra = {}
    if blocks_source is not None:
        extra["blocks_source"] = blocks_source
    if "run_id" in meta.columns:
        extra["run_id"] = meta.loc[test_indices, "run_id"].tolist()
    if "block" in meta.columns:
        extra["block"] = meta.loc[test_indices, "block"].tolist()

    idx_df = pd.DataFrame(
        {
            "subject_id": groups_ordered,
            "trial_id": meta.loc[test_indices, "trial_id"].values,
            "fold": fold_ids,
            **extra,
        }
    )

    if add_heldout_subject_id:
        idx_df["heldout_subject_id"] = idx_df["subject_id"].astype(str)

    write_tsv(idx_df, save_path.with_suffix(".tsv"))
    write_parquet(idx_df, save_path.with_suffix(".parquet"))


def _is_power_channel_feature(parsed: dict) -> bool:
    """Check if parsed feature is a power channel feature."""
    return (
        parsed.get("valid", False)
        and parsed.get("group") == "power"
        and parsed.get("scope") == "ch"
    )


def _extract_power_channel_features(
    feature_names: List[str],
) -> tuple[set[str], dict[str, dict[str, list[int]]]]:
    """Extract power channel features and build band/channel to index mapping."""
    bands_set = set()
    band_channel_to_indices: dict[str, dict[str, list[int]]] = {}

    for index, feature_name in enumerate(feature_names):
        parsed = NamingSchema.parse(str(feature_name))
        if not _is_power_channel_feature(parsed):
            continue

        band = parsed.get("band")
        channel = parsed.get("identifier")
        if band and channel:
            band_str = str(band)
            channel_str = str(channel)
            bands_set.add(band_str)
            band_channel_to_indices.setdefault(band_str, {}).setdefault(
                channel_str, []
            ).append(index)

    bands = sorted(bands_set)
    return bands, band_channel_to_indices


def _aggregate_coefficients(
    coefficient_matrix: np.ndarray,
    aggregate_method: str,
) -> np.ndarray:
    """Aggregate coefficients across samples."""
    if aggregate_method == "signed":
        return np.nanmean(coefficient_matrix, axis=0)
    return np.nanmean(np.abs(coefficient_matrix), axis=0)


def write_feature_importance(
    subject: str,
    coef_matrix: np.ndarray,
    feature_names: List[str],
    stats_dir: Path,
    method: str = "elasticnet",
    aggregate: str = "abs",
    mode: str = "regression",
    target: str = "auto",
    extra_columns: Optional[dict] = None,
) -> Optional[Path]:
    """Write feature importance parquet for topomap visualization."""
    ensure_dir(stats_dir)

    bands, band_channel_to_indices = _extract_power_channel_features(
        feature_names
    )
    aggregated_coefficients = _aggregate_coefficients(coef_matrix, aggregate)

    rows = []
    for band in bands:
        channel_map = band_channel_to_indices.get(band, {})
        for channel, indices in channel_map.items():
            feature_weights = aggregated_coefficients[indices]
            mean_weight = float(np.nanmean(feature_weights))

            row = {
                "subject": subject,
                "band": band,
                "channel": channel,
                "weight": mean_weight,
                "mode": mode,
                "target": target,
                "method": method,
                "aggregate": aggregate,
            }
            if extra_columns:
                row.update(extra_columns)
            rows.append(row)

    if not rows:
        return None

    filename = f"feature_topomap_{method}_{aggregate}_{mode}_{target}.parquet"
    output_path = stats_dir / filename
    write_parquet(pd.DataFrame(rows), output_path)
    logger.info(f"Saved feature importance: {output_path}")
    return output_path


def aggregate_group_feature_topomaps(
    subjects: List[str],
    deriv_root: Path,
    task: str,
    method: str = "elasticnet",
    aggregate: str = "abs",
    target: Optional[str] = None,
    min_subjects: Optional[int] = None,
    stats_dir: Optional[Path] = None,
    config: Optional[Any] = None,
) -> Optional[Path]:
    if min_subjects is None:
        if config is None:
            raise ValueError("config is required when min_subjects is None")
        min_subjects = int(config.get("analysis.min_subjects_for_topomaps"))
    
    if stats_dir is None:
        stats_dir = deriv_root / "group" / "eeg" / "stats"

    ensure_dir(stats_dir)

    tables: List[pd.DataFrame] = []
    for subject in subjects:
        subj_stats_dir = deriv_stats_path(deriv_root, subject) / "05_decode_pain_experience"
        if method == "elasticnet":
            pattern = f"feature_topomap_elasticnet_{aggregate}_*.parquet"
        else:
            pattern = "feature_topomap_rfperm_*.parquet"

        matches = list(subj_stats_dir.glob(pattern))
        if target is not None:
            matches = [m for m in matches if m.stem.endswith(f"_{target}")]

        if matches:
            try:
                df = read_parquet(matches[0])
                df["subject"] = subject
                tables.append(df)
            except (OSError, pd.errors.ParserError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to load {matches[0]}: {e}")

    if len(tables) < min_subjects:
        logger.info(f"Insufficient subjects ({len(tables)}) for group aggregation (min={min_subjects})")
        return None

    df_all = pd.concat(tables, axis=0, ignore_index=True)
    bands = sorted(df_all["band"].astype(str).unique())
    channels = sorted(df_all["channel"].astype(str).unique())

    keep_channels = [
        ch for ch in channels if df_all.loc[df_all["channel"] == ch, "subject"].nunique() >= min_subjects
    ]

    if not keep_channels:
        logger.info("No channels met the minimum subject threshold")
        return None

    info_ref = None
    for subject in subjects:
        epo_path = find_clean_epochs_path(subject, task, deriv_root=deriv_root, config=config)
        if epo_path and Path(epo_path).exists():
            try:
                epochs = mne.read_epochs(epo_path, preload=False, verbose=False)
                info_ref = epochs.info
                break
            except Exception:
                continue

    if info_ref is None:
        logger.error("Could not locate reference epochs info for group aggregation")
        return None

    picks = mne.pick_channels(info_ref["ch_names"], include=keep_channels)
    ordered_channels = [info_ref["ch_names"][idx] for idx in picks]

    if config is None:
        raise ValueError("config is required for FDR correction")
    fdr_alpha = float(config.get("statistics.fdr_alpha"))

    rows: List[dict] = []

    for band in bands:
        matrices: List[np.ndarray] = []
        for subject in subjects:
            df_sub = df_all[(df_all["subject"] == subject) & (df_all["band"].astype(str) == str(band))]
            if df_sub.empty:
                continue
            lookup = {row["channel"]: float(row["weight"]) for _, row in df_sub.iterrows()}
            vec = np.array([lookup.get(ch, np.nan) for ch in ordered_channels], dtype=float)
            matrices.append(vec)

        if not matrices:
            continue

        W = np.vstack(matrices)
        mean = np.nanmean(W, axis=0)
        sd = np.nanstd(W, axis=0, ddof=1)
        n = np.sum(np.isfinite(W), axis=0)
        valid = (n >= 2) & np.isfinite(sd) & (sd > 0)
        t = np.full_like(mean, np.nan, dtype=float)
        p = np.full_like(mean, np.nan, dtype=float)
        if np.any(valid):
            denom_valid = sd[valid] / np.sqrt(n[valid])
            t[valid] = mean[valid] / denom_valid
            p[valid] = 2 * (1 - norm.cdf(np.abs(t[valid])))
        
        q = fdr_bh(p, alpha=fdr_alpha)
        sig = q < fdr_alpha

        for ch, m, s, nn, tt, pp, qq, sg in zip(ordered_channels, mean, sd, n, t, p, q, sig):
            rows.append(
                {
                    "band": band,
                    "channel": ch,
                    "mean": float(m),
                    "sd": float(s),
                    "n": int(nn),
                    "t": float(tt),
                    "p": float(pp),
                    "q": float(qq),
                    "significant": bool(sg),
                }
            )

    if not rows:
        logger.info("No group rows generated")
        return None

    suffix = f"{method}_{aggregate}" if method == "elasticnet" else method
    if target:
        suffix += f"_{target}"

    out_path = stats_dir / f"feature_topomap_group_{suffix}.parquet"
    write_parquet(pd.DataFrame(rows), out_path)
    logger.info(f"Saved group feature importance: {out_path}")

    return out_path


__all__ = [
    "create_run_manifest",
    "prepare_best_params_path",
    "export_predictions",
    "export_indices",
    "write_feature_importance",
    "aggregate_group_feature_topomaps",
]
