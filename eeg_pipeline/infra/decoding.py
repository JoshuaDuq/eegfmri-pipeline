from __future__ import annotations

import os
import time
import json
import logging
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import mne
from scipy.stats import norm

from eeg_pipeline.infra.paths import ensure_dir, deriv_stats_path, find_clean_epochs_path
from eeg_pipeline.infra.tsv import read_tsv
from eeg_pipeline.infra.logging import get_logger
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh

logger = get_logger(__name__)

_BEST_PARAMS_LOGGED: set = set()
_handler_lock = threading.Lock()
_FILE_LOG_HANDLER: Optional[logging.Handler] = None


###################################################################
# Run Manifest and Logging
###################################################################

def create_run_manifest(results_dir: Path, cli_args: dict, config: dict, run_id: Optional[str] = None) -> None:
    import platform
    import subprocess

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "cli_args": cli_args,
        "resolved_config": config,
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
        },
        "thread_limits": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
        },
    }

    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent.parent,
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = None
    manifest["git_commit"] = git_hash

    import sklearn as _sk

    manifest["package_versions"] = {"sklearn": _sk.__version__, "mne": mne.__version__}
    try:
        import pyriemann as _pr

        manifest["package_versions"]["pyriemann"] = _pr.__version__
    except ImportError:
        manifest["package_versions"]["pyriemann"] = None

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def setup_file_logging(results_dir: Path, run_id: Optional[str] = None, logger_name: str = "decode_pain") -> Path:
    global _FILE_LOG_HANDLER
    logger_instance = get_logger(logger_name)

    with _handler_lock:
        log_dir = results_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"_{run_id}" if run_id else ""
        log_path = log_dir / f"{logger_name}_{ts}{suffix}.log"

        if _FILE_LOG_HANDLER is not None:
            logger_instance.removeHandler(_FILE_LOG_HANDLER)
            _FILE_LOG_HANDLER.close()

        for h in logger_instance.handlers:
            if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path):
                return log_path

        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger_instance.addHandler(fh)
        _FILE_LOG_HANDLER = fh
        return log_path


###################################################################
# Best Parameters I/O
###################################################################

def prepare_best_params_path(base_path: Path, mode: str, run_id: Optional[str]) -> Path:
    base_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "run_scoped":
        rid = run_id or time.strftime("%Y%m%d_%H%M%S")
        out_path = base_path.with_name(f"{base_path.stem}_{rid}{base_path.suffix}")
    else:
        if mode == "truncate":
            base_path.open("w", encoding="utf-8").close()
        out_path = base_path

    if base_path not in _BEST_PARAMS_LOGGED:
        logger.info(f"Best-params mode='{mode}'; resolved path: {out_path}")
        _BEST_PARAMS_LOGGED.add(base_path)

    return out_path


def read_best_params_jsonl(path: Path) -> dict:
    best = {}
    if not path or not path.exists():
        return best

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            fold = rec.get("fold")
            if fold is not None:
                best[int(fold)] = rec.get("best_params", {})

    return best


def read_best_params_jsonl_combined(path: Path) -> dict:
    combined: dict = {}
    if not path or not path.exists():
        return combined

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            params = rec.get("best_params") or rec.get("best_params_by_r") or {}
            if not isinstance(params, dict):
                continue

            fold = rec.get("fold")
            subj = rec.get("subject") or rec.get("heldout_subject") or rec.get("heldout_subject_id")

            if fold is not None:
                combined[int(fold)] = params
            if subj is not None:
                combined[str(subj)] = params

    return combined


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

    ensure_dir(save_path.parent)
    pred_df.to_csv(save_path, sep="\t", index=False)
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

    ensure_dir(save_path.parent)
    idx_df.to_csv(save_path, sep="\t", index=False)


###################################################################
# Feature Importance and Topomaps
###################################################################



def write_feature_importance_tsv(
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
    ensure_dir(stats_dir)

    bands_set = set()
    for feat in feature_names:
        parsed = NamingSchema.parse(str(feat))
        if not (parsed.get("valid") and parsed.get("group") == "power"):
            continue
        if parsed.get("scope") != "ch":
            continue
        band = parsed.get("band")
        identifier = parsed.get("identifier")
        if band and identifier:
            bands_set.add(str(band))
    bands = sorted(bands_set)

    band_ch_to_idx: dict = {}
    for idx, feat in enumerate(feature_names):
        parsed = NamingSchema.parse(str(feat))
        if not (parsed.get("valid") and parsed.get("group") == "power"):
            continue
        if parsed.get("scope") != "ch":
            continue
        band = parsed.get("band")
        channel = parsed.get("identifier")
        if band and channel:
            band_ch_to_idx.setdefault(str(band), {}).setdefault(str(channel), []).append(idx)

    coef_agg = np.nanmean(coef_matrix, axis=0) if aggregate == "signed" else np.nanmean(np.abs(coef_matrix), axis=0)

    rows = []
    for b in bands:
        ch_map = band_ch_to_idx.get(b, {})
        for ch, idxs in ch_map.items():
            weight = float(np.nanmean(coef_agg[idxs]))
            rows.append(
                {
                    "subject": subject,
                    "band": b,
                    "channel": ch,
                    "weight": weight,
                    "mode": mode,
                    "target": target,
                    "method": method,
                    "aggregate": aggregate,
                    **(extra_columns or {}),
                }
            )

    if not rows:
        return None

    tsv_name = f"feature_topomap_{method}_{aggregate}_{mode}_{target}.tsv"
    out_path = stats_dir / tsv_name
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    logger.info(f"Saved feature importance TSV: {out_path}")
    return out_path


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
        if config is not None:
            min_subjects = int(config.get("analysis.min_subjects_for_topomaps"))
        else:
            raise ValueError("config is required when min_subjects is None")
    if stats_dir is None:
        stats_dir = deriv_root / "group" / "eeg" / "stats"

    ensure_dir(stats_dir)

    tables: List[pd.DataFrame] = []
    for subject in subjects:
        subj_stats_dir = deriv_stats_path(deriv_root, subject) / "05_decode_pain_experience"
        if method == "elasticnet":
            pattern = f"feature_topomap_elasticnet_{aggregate}_*.tsv"
        else:
            pattern = "feature_topomap_rfperm_*.tsv"

        matches = list(subj_stats_dir.glob(pattern))
        if target is not None:
            matches = [m for m in matches if m.stem.endswith(f"_{target}")]

        if matches:
            try:
                df = read_tsv(matches[0])
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
        if config is None:
            raise ValueError("config is required for FDR correction")
        fdr_alpha = float(config.get("statistics.fdr_alpha"))
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

    out_tsv = stats_dir / f"feature_topomap_group_{suffix}.tsv"
    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)
    logger.info(f"Saved group feature importance TSV: {out_tsv}")

    return out_tsv


__all__ = [
    "create_run_manifest",
    "setup_file_logging",
    "prepare_best_params_path",
    "read_best_params_jsonl",
    "read_best_params_jsonl_combined",
    "export_predictions",
    "export_indices",
    "write_feature_importance_tsv",
    "aggregate_group_feature_topomaps",
]
