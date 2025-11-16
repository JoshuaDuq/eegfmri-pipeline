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

from eeg_pipeline.utils.io_utils import (
    ensure_dir,
    deriv_stats_path,
    _find_clean_epochs_path,
    fdr_bh,
)
from eeg_pipeline.utils.config_loader import load_settings
from eeg_pipeline.utils.data_loading import resolve_columns

logger = logging.getLogger(__name__)

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
        }
    }
    
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent.parent.parent,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = None
    manifest["git_commit"] = git_hash
    
    import sklearn as _sk
    import mne as _mne
    manifest["package_versions"] = {"sklearn": _sk.__version__, "mne": _mne.__version__}
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
    logger_instance = logging.getLogger(logger_name)
    
    with _handler_lock:
        log_dir = results_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"_{run_id}" if run_id else ""
        log_path = log_dir / f"{logger_name}_{ts}{suffix}.log"
        
        if _FILE_LOG_HANDLER is not None:
            logger_instance.removeHandler(_FILE_LOG_HANDLER)
            _FILE_LOG_HANDLER.close()
        
        existing_file_handlers = [
            h for h in logger_instance.handlers
            if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
        ]
        if existing_file_handlers:
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
    combined = {}
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

def _extract_subject_from_record(record: Dict[str, Any]) -> Optional[str]:
    return (
        record.get("subject") or
        record.get("heldout_subject") or
        record.get("heldout_subject_id")
    )

def _add_params_to_records(
    records: list,
    params_dict: dict,
    model: Optional[str],
    fold: Optional[int],
    subject: Optional[str],
    criterion: str,
    score: Optional[float] = None,
) -> None:
    for param_key, param_value in params_dict.items():
        records.append({
            "model": model,
            "fold": fold,
            "subject": subject,
            "criterion": criterion,
            "param_key": param_key,
            "param_value": param_value,
            "score": score,
        })

def export_best_params_long_table(jsonl_path: Path, save_path: Path) -> None:
    records = []
    if not jsonl_path or not jsonl_path.exists():
        return
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            
            model = rec.get("model")
            fold = rec.get("fold")
            fold_int = int(fold) if fold is not None else None
            subject = _extract_subject_from_record(rec)
            subject_str = str(subject) if subject is not None else None
            
            best_params = rec.get("best_params")
            if isinstance(best_params, dict):
                _add_params_to_records(records, best_params, model, fold_int, subject_str, "refit_r")
            
            best_params_by_r = rec.get("best_params_by_r")
            if isinstance(best_params_by_r, dict):
                score_r = rec.get("best_score_r")
                score_float = float(score_r) if score_r is not None else None
                _add_params_to_records(
                    records, best_params_by_r, model, fold_int, subject_str, "grid_best_r", score_float
                )
            
            best_params_by_neg_mse = rec.get("best_params_by_neg_mse")
            if isinstance(best_params_by_neg_mse, dict):
                score_neg_mse = rec.get("best_score_neg_mse")
                score_float = float(score_neg_mse) if score_neg_mse is not None else None
                _add_params_to_records(
                    records, best_params_by_neg_mse, model, fold_int, subject_str, "grid_best_neg_mse", score_float
                )
    
    if records:
        pd.DataFrame(records).to_csv(save_path, sep="\t", index=False)


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
    
    pred_df = pd.DataFrame({
        "subject_id": groups_ordered,
        "trial_id": meta.loc[test_indices, "trial_id"].values,
        "y_true": y_true,
        "y_pred": y_pred,
        "fold": fold_ids,
        "model": model_name,
        **extra_cols,
    })
    
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
    
    idx_df = pd.DataFrame({
        "subject_id": groups_ordered,
        "trial_id": meta.loc[test_indices, "trial_id"].values,
        "fold": fold_ids,
        **extra,
    })
    
    if add_heldout_subject_id:
        idx_df["heldout_subject_id"] = idx_df["subject_id"].astype(str)
    
    ensure_dir(save_path.parent)
    idx_df.to_csv(save_path, sep="\t", index=False)


###################################################################
# Feature Importance and Topomaps
###################################################################

def parse_pow_feature(feat: str) -> Optional[Tuple[str, str]]:
    if not isinstance(feat, str) or not feat.startswith("pow_"):
        return None
    parts = feat[4:].split("_")
    if len(parts) < 2:
        return None
    return "_".join(parts[:-1]), parts[-1]

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
        parsed = parse_pow_feature(feat)
        if parsed:
            bands_set.add(parsed[0])
    bands = sorted(bands_set)
    
    band_ch_to_idx = {}
    for idx, feat in enumerate(feature_names):
        parsed = parse_pow_feature(feat)
        if parsed:
            b, ch = parsed
            band_ch_to_idx.setdefault(b, {}).setdefault(ch, []).append(idx)
    
    coef_agg = np.nanmean(coef_matrix, axis=0) if aggregate == "signed" else np.nanmean(np.abs(coef_matrix), axis=0)
    
    rows = []
    for b in bands:
        ch_map = band_ch_to_idx.get(b, {})
        for ch, idxs in ch_map.items():
            weight = float(np.nanmean(coef_agg[idxs]))
            rows.append({
                "subject": subject,
                "band": b,
                "channel": ch,
                "weight": weight,
                "mode": mode,
                "target": target,
                "method": method,
                "aggregate": aggregate,
                **(extra_columns or {}),
            })
    
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
                df = pd.read_csv(matches[0], sep="\t")
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
        ch for ch in channels
        if df_all.loc[df_all["channel"] == ch, "subject"].nunique() >= min_subjects
    ]
    
    if not keep_channels:
        logger.info("No channels met the minimum subject threshold")
        return None
    
    info_ref = None
    for subject in subjects:
        epo_path = _find_clean_epochs_path(subject, task)
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
    maps_mean: dict = {}
    maps_sig: dict = {}
    
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
        denom = np.divide(sd, np.sqrt(np.maximum(n, 1)), out=np.zeros_like(sd), where=n > 0)
        t = np.divide(mean, denom, out=np.zeros_like(mean), where=denom > 0)
        p = 2 * (1 - norm.cdf(np.abs(t)))
        if config is None:
            raise ValueError("config is required for FDR correction")
        fdr_alpha = float(config.get("statistics.fdr_alpha"))
        q = fdr_bh(p, alpha=fdr_alpha)
        sig = q < fdr_alpha
        
        for ch, m, s, nn, tt, pp, qq, sg in zip(ordered_channels, mean, sd, n, t, p, q, sig):
            rows.append({
                "band": band,
                "channel": ch,
                "mean": float(m),
                "sd": float(s),
                "n": int(nn),
                "t": float(tt),
                "p": float(pp),
                "q": float(qq),
                "significant": bool(sg),
            })
        
        maps_mean[band] = mean
        maps_sig[band] = mean * sig.astype(float)
    
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


###################################################################
# Metrics Aggregation
###################################################################

def build_all_metrics_wide(results_dir: Path, save_path: Path) -> None:
    files = list(results_dir.glob("*per_subject_metrics.tsv"))
    if not files:
        logger.warning("No per-subject metrics found to build wide TSV.")
        return
    
    merged = None
    for fp in files:
        name = fp.name.replace("_per_subject_metrics.tsv", "")
        if "subject_test" in name:
            continue
        
        try:
            df = pd.read_csv(fp, sep="\t")
        except (FileNotFoundError, pd.errors.ParserError, UnicodeDecodeError) as e:
            logger.warning(f"Failed reading {fp}: {e}")
            continue
        
        subj_col = "group" if "group" in df.columns else ("subject_id" if "subject_id" in df.columns else None)
        if subj_col is None:
            continue
        
        df = df.rename(columns={subj_col: "subject_id"})
        keep = [c for c in ["pearson_r", "r2", "explained_variance", "n_trials"] if c in df.columns]
        df = df[["subject_id"] + keep].copy()
        df = df.rename(columns={c: f"{c}__{name}" for c in keep})
        
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="subject_id", how="outer")
    
    if merged is not None:
        merged.sort_values("subject_id").to_csv(save_path, sep="\t", index=False)


###################################################################
# Montage Resolution
###################################################################

def find_bids_electrodes_tsv(bids_root: Path, subjects: Optional[List[str]] = None) -> Optional[Path]:
    if subjects not in (None, [], ["all"]):
        for s in subjects:
            sub = f"sub-{s}"
            cand = list((bids_root / sub / "eeg").glob("*_electrodes.tsv"))
            if cand:
                return cand[0]
    
    for sub_dir in sorted((bids_root).glob("sub-*/eeg")):
        cand = list(sub_dir.glob("*_electrodes.tsv"))
        if cand:
            return cand[0]
    return None

def make_montage_from_bids_electrodes(electrodes_tsv: Path) -> mne.channels.DigMontage:
    df = pd.read_csv(electrodes_tsv, sep="\t")

    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("name") or cols.get("label") or cols.get("electrode")
    x_col = cols.get("x")
    y_col = cols.get("y")
    z_col = cols.get("z")
    if not all([name_col, x_col, y_col, z_col]):
        raise RuntimeError("electrodes.tsv must contain columns for name/label and x,y,z coordinates")

    df = df[[name_col, x_col, y_col, z_col]].dropna()
    df = df.rename(columns={name_col: "name", x_col: "x", y_col: "y", z_col: "z"})

    df[["x", "y", "z"]] = df[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["x", "y", "z"]).copy()

    coords = df[["x", "y", "z"]].values
    max_coord = np.nanmax(np.abs(coords)) if len(coords) > 0 else 0.0
    if max_coord > 2.0:
        logger.info("Electrode coordinates appear to be in mm; converting to meters.")
        df[["x", "y", "z"]] *= 0.001

    ch_pos = {str(row["name"]): np.array([row["x"], row["y"], row["z"]], dtype=float) for _, row in df.iterrows()}
    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")

def resolve_montage(montage_opt: Optional[str], deriv_root: Path, subjects: Optional[List[str]]) -> mne.channels.DigMontage:
    montage_opt = montage_opt or "standard_1005"
    
    if montage_opt.startswith("bids:"):
        return make_montage_from_bids_electrodes(Path(montage_opt.split(":", 1)[1]))
    
    if montage_opt == "bids_auto":
        tsv_path = find_bids_electrodes_tsv(deriv_root.parent, subjects)
        if tsv_path is None:
            raise RuntimeError("bids_auto montage requested but no electrodes.tsv found")
        return make_montage_from_bids_electrodes(tsv_path)
    
    return mne.channels.make_standard_montage(montage_opt)


###################################################################
# Config Preparation
###################################################################

def prepare_config_dict(cfg) -> tuple:
    config_dict = cfg.to_legacy_dict()
    
    results_subdir = config_dict.get("decoding", {}).get("paths", {}).get("results_subdir")
    if results_subdir is None:
        raise ValueError("decoding.paths.results_subdir not found in config")
    
    _paths = config_dict.setdefault("paths", {})
    
    _best = _paths.setdefault("best_params", {})
    _best.setdefault("elasticnet_loso", f"{results_subdir}/best_params/elasticnet_loso.jsonl")
    _best.setdefault("rf_loso", f"{results_subdir}/best_params/rf_loso.jsonl")
    _best.setdefault("elasticnet_within", f"{results_subdir}/best_params/elasticnet_within.jsonl")
    _best.setdefault("rf_within", f"{results_subdir}/best_params/rf_within.jsonl")
    _best.setdefault("riemann_loso", f"{results_subdir}/best_params/riemann_loso.jsonl")
    _best.setdefault("riemann_band_template", f"{results_subdir}/best_params/riemann_{{label}}_loso.jsonl")
    _best.setdefault("temperature_only", f"{results_subdir}/best_params/temperature_only.jsonl")
    
    _pred = _paths.setdefault("predictions", {})
    _pred.setdefault("elasticnet_loso", f"{results_subdir}/predictions/elasticnet_loso.tsv")
    _pred.setdefault("elasticnet_within", f"{results_subdir}/predictions/elasticnet_within.tsv")
    _pred.setdefault("rf_loso", f"{results_subdir}/predictions/rf_loso.tsv")
    _pred.setdefault("rf_within", f"{results_subdir}/predictions/rf_within.tsv")
    _pred.setdefault("riemann_loso", f"{results_subdir}/predictions/riemann_loso.tsv")
    _pred.setdefault("riemann_band_template", f"{results_subdir}/predictions/riemann_{{label}}_loso.tsv")
    _pred.setdefault("temperature_only", f"{results_subdir}/predictions/temperature_only.tsv")
    _pred.setdefault("baseline_global", f"{results_subdir}/predictions/baseline_global.tsv")
    
    _metrics = _paths.setdefault("per_subject_metrics", {})
    _metrics.setdefault("elasticnet_loso", f"{results_subdir}/metrics/elasticnet_loso_per_subject.tsv")
    _metrics.setdefault("elasticnet_within", f"{results_subdir}/metrics/elasticnet_within_per_subject.tsv")
    _metrics.setdefault("rf_loso", f"{results_subdir}/metrics/rf_loso_per_subject.tsv")
    _metrics.setdefault("rf_within", f"{results_subdir}/metrics/rf_within_per_subject.tsv")
    _metrics.setdefault("riemann_loso", f"{results_subdir}/metrics/riemann_loso_per_subject.tsv")
    _metrics.setdefault("temperature_only", f"{results_subdir}/metrics/temperature_only_per_subject.tsv")
    _metrics.setdefault("baseline_global", f"{results_subdir}/metrics/baseline_global_per_subject.tsv")
    
    _indices = _paths.setdefault("indices", {})
    _indices.setdefault("elasticnet_loso", f"{results_subdir}/indices/elasticnet_loso_indices.tsv")
    _indices.setdefault("elasticnet_within", f"{results_subdir}/indices/elasticnet_within_indices.tsv")
    _indices.setdefault("rf_loso", f"{results_subdir}/indices/rf_loso_indices.tsv")
    _indices.setdefault("rf_within", f"{results_subdir}/indices/rf_within_indices.tsv")
    _indices.setdefault("riemann_loso", f"{results_subdir}/indices/riemann_loso_indices.tsv")
    _indices.setdefault("riemann_band_template", f"{results_subdir}/indices/riemann_{{label}}_loso_indices.tsv")
    _indices.setdefault("temperature_only", f"{results_subdir}/indices/temperature_only_indices.tsv")
    _indices.setdefault("baseline_global", f"{results_subdir}/indices/baseline_global_indices.tsv")
    
    _summaries = _paths.setdefault("summaries", {})
    _summaries.setdefault("riemann_bands", f"{results_subdir}/summaries/riemann_bands.json")
    _summaries.setdefault("riemann_sliding_window", f"{results_subdir}/summaries/riemann_sliding_window.json")
    _summaries.setdefault("permutation_refit_null_rs", f"{results_subdir}/summaries/permutation_refit_null_rs.txt")
    _summaries.setdefault("permutation_refit_summary", f"{results_subdir}/summaries/permutation_refit_summary.json")
    _summaries.setdefault("all_metrics_wide", f"{results_subdir}/summaries/all_metrics_wide.tsv")
    _summaries.setdefault("summary", f"{results_subdir}/summaries/summary.json")
    _summaries.setdefault("incremental", f"{results_subdir}/summaries/incremental_validity.json")
    
    random_state = cfg.get("project.random_state")
    if random_state is None:
        raise ValueError(
            "project.random_state not found in config. "
            "Add 'random_state: 42' (or desired seed) under 'project:' section in eeg_config.yaml"
        )
    
    legacy = {
        "TASK": cfg.task,
        "RANDOM_STATE": random_state,
        "DERIV_ROOT": str(cfg.deriv_root),
    }
    
    return config_dict, legacy

