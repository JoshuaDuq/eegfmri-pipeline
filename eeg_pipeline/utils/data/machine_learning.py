from __future__ import annotations

import logging
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.infra.tsv import read_table, read_tsv
from eeg_pipeline.infra.paths import _find_clean_events_path, deriv_features_path, load_events_df
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.utils.data.columns import (
    find_pain_column_in_events,
    find_temperature_column_in_events,
    pick_target_column,
)
from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from ..config.loader import ConfigDict

EEGConfig = ConfigDict
logger = logging.getLogger(__name__)

MLTargetKind = Literal["continuous", "binary"]
MLFeatureHarmonization = Literal["intersection", "union_impute"]


def _filter_finite_targets(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    meta: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Filter out samples with non-finite target values."""
    finite_mask = np.isfinite(y)
    if np.all(finite_mask):
        return X, y, groups, meta

    X_filtered = X[finite_mask]
    y_filtered = y[finite_mask]
    groups_filtered = groups[finite_mask]
    meta_filtered = meta.loc[finite_mask].reset_index(drop=True)
    return X_filtered, y_filtered, groups_filtered, meta_filtered


def _find_block_column(aligned_events: pd.DataFrame) -> Optional[pd.Series]:
    """Find block/run identifier column from aligned events."""
    for candidate in ("block", "run_id", "run", "session"):
        if candidate in aligned_events.columns:
            series = aligned_events[candidate]
            numeric = pd.to_numeric(series, errors="coerce")
            if np.any(np.isfinite(numeric.to_numpy(dtype=float))):
                return numeric
            parsed = pd.Series([_parse_run_label_to_int(v) for v in series], index=series.index, dtype="float64")
            if np.any(np.isfinite(parsed.to_numpy(dtype=float))):
                return parsed
            return numeric
    return None


def _normalize_subject(subject: str) -> Tuple[str, str]:
    """Return (subject_raw, subject_bids) where subject_bids is 'sub-XXXX'."""
    subject_raw = str(subject).strip()
    if subject_raw.startswith("sub-"):
        return subject_raw.replace("sub-", "", 1), subject_raw
    return subject_raw, f"sub-{subject_raw}"


def _as_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip() != ""]
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else None
    return None


def _resolve_feature_families(
    *,
    feature_families: Optional[List[str]],
    feature_set: Optional[str],
    config: Any,
) -> List[str]:
    """Resolve which feature families to use for ML."""
    resolved = _as_list(feature_families)
    if resolved:
        return resolved

    if feature_set and feature_set not in {"combined", "channels_mean"}:
        return [str(feature_set)]

    from_config = _as_list(get_config_value(config, "machine_learning.data.feature_families", None))
    if from_config:
        return from_config

    # If feature_set is "combined" and no explicit list is provided, prefer a broad
    # default that matches the feature pipeline's standard outputs.
    if (feature_set or "").strip().lower() in {"combined", ""}:
        try:
            from eeg_pipeline.utils.data.feature_discovery import STANDARD_FEATURE_FILES

            return list(STANDARD_FEATURE_FILES.keys())
        except Exception as exc:
            logger.warning(
                "Failed to import STANDARD_FEATURE_FILES; falling back to conservative ML feature family defaults: %s",
                exc,
            )

    # Conservative fallback.
    return ["power"]


def _resolve_feature_filename(
    family: str,
    config: Any,
) -> str:
    # Allow overriding filenames via config if needed.
    override = get_config_value(config, f"machine_learning.data.feature_files.{family}", None)
    if isinstance(override, str) and override.strip():
        return override.strip()
    return f"features_{family}.parquet"


def _resolve_feature_path(features_dir: Path, family: str, filename: str) -> Path:
    """Resolve a feature table path (handles sourcelocalization subfolders)."""
    if family.startswith("pac"):
        # pac variants (pac_trials, pac_time) live in the shared "pac/" folder.
        return features_dir / "pac" / filename
    if family in {"sourcelocalization", "source_localization"}:
        candidates = [
            features_dir / "sourcelocalization" / "fmri_informed" / filename,
            features_dir / "sourcelocalization" / "eeg_only" / filename,
            features_dir / "sourcelocalization" / filename,
        ]
        for p in candidates:
            if p.exists():
                return p
        return candidates[0]
    return features_dir / family / filename


def _prefix_feature_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Prefix columns (avoids collisions when combining families)."""
    df = df.copy()
    renamed = {}
    for col in df.columns:
        col_str = str(col)
        renamed[col] = col_str if col_str.startswith(prefix) else f"{prefix}{col_str}"
    df.rename(columns=renamed, inplace=True)
    return df


def _resolve_target_series(
    events_df: pd.DataFrame,
    *,
    target: Optional[str],
    target_kind: MLTargetKind,
    binary_threshold: Optional[float],
    config: Any,
) -> Tuple[pd.Series, str]:
    """
    Resolve target vector from events.

    `target` can be:
    - logical names: "rating", "temperature", "pain_binary"
    - a literal column name in events_df
    """
    target_key = (target or "").strip()

    if not target_key:
        target_key = "rating"

    if target_key.lower() in {"rating", "pain_rating", "vas"}:
        rating_columns = config.get("event_columns.rating", [])
        tgt_col = pick_target_column(events_df, target_columns=list(rating_columns) if rating_columns else [])
        if tgt_col is None:
            raise ValueError(
                "No rating column found in events.tsv. "
                f"Tried config event_columns.rating={rating_columns}; available={list(events_df.columns)}"
            )
        series = pd.to_numeric(events_df[tgt_col], errors="coerce")
        return series, str(tgt_col)

    if target_key.lower() in {"temperature", "temp"}:
        temp_col = find_temperature_column_in_events(events_df, config)
        if temp_col is None:
            raise ValueError(
                "No temperature column found in events.tsv. "
                f"Tried config event_columns.temperature={config.get('event_columns.temperature', [])}; available={list(events_df.columns)}"
            )
        series = pd.to_numeric(events_df[temp_col], errors="coerce")
        return series, str(temp_col)

    if target_key.lower() in {"pain", "pain_binary", "binary"}:
        pain_col = find_pain_column_in_events(events_df, config)
        if pain_col is None:
            raise ValueError(
                "No pain-binary column found in events.tsv. "
                f"Tried config event_columns.pain_binary={config.get('event_columns.pain_binary', [])}; available={list(events_df.columns)}"
            )
        series = pd.to_numeric(events_df[pain_col], errors="coerce")
        return series, str(pain_col)

    if target_key in events_df.columns:
        series = pd.to_numeric(events_df[target_key], errors="coerce")
        return series, target_key

    # Try exact match ignoring case.
    lower_map = {str(c).lower(): str(c) for c in events_df.columns}
    if target_key.lower() in lower_map:
        col = lower_map[target_key.lower()]
        series = pd.to_numeric(events_df[col], errors="coerce")
        return series, col

    raise ValueError(
        f"Target '{target_key}' not found in events.tsv. Available columns: {list(events_df.columns)}"
    )


def _target_covariate_aliases(target: Optional[str], config: Optional[Any] = None) -> set[str]:
    """Return standardized target aliases for covariate leakage checks."""
    target_raw = str(target or "").strip()
    target_key = target_raw.lower()
    aliases: set[str] = set()

    if target_key in {"", "rating", "pain_rating", "vas"}:
        aliases.add("rating")
    elif target_key in {"temperature", "temp"}:
        aliases.add("temperature")
    elif target_key in {"pain", "pain_binary", "binary"}:
        aliases.add("pain_binary")
    elif target_key in {"fmri_signature", "fmri-signature"}:
        aliases.add("fmri_signature")

    # If target is provided as an explicit events column, map it back to canonical
    # aliases so leakage checks still block semantically identical predictors.
    if config is not None:
        event_alias_map = {
            "rating": _as_list(get_config_value(config, "event_columns.rating", [])) or [],
            "temperature": _as_list(get_config_value(config, "event_columns.temperature", [])) or [],
            "pain_binary": _as_list(get_config_value(config, "event_columns.pain_binary", [])) or [],
        }
        for canonical, column_aliases in event_alias_map.items():
            normalized = {str(c).strip().lower() for c in column_aliases if str(c).strip()}
            if target_key and target_key in normalized:
                aliases.add(canonical)

    if target_raw:
        aliases.add(target_raw)
    return aliases


def _fmri_signature_defaults(config: Any) -> dict:
    return {
        "method": str(get_config_value(config, "machine_learning.fmri_signature.method", "beta-series")).strip().lower(),
        "contrast_name": str(get_config_value(config, "machine_learning.fmri_signature.contrast_name", "pain_vs_nonpain")).strip(),
        "signature_name": str(get_config_value(config, "machine_learning.fmri_signature.signature_name", "NPS")).strip(),
        "metric": str(get_config_value(config, "machine_learning.fmri_signature.metric", "dot")).strip().lower(),
        "normalization": str(get_config_value(config, "machine_learning.fmri_signature.normalization", "none")).strip().lower(),
        "round_decimals": int(get_config_value(config, "machine_learning.fmri_signature.round_decimals", 3)),
    }


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    vals = values.astype(float)
    finite = np.isfinite(vals)
    out = np.full_like(vals, np.nan, dtype=float)
    if not np.any(finite):
        return out
    med = np.nanmedian(vals[finite])
    mad = np.nanmedian(np.abs(vals[finite] - med))
    denom = 1.4826 * mad
    if not np.isfinite(denom) or denom == 0:
        out[finite] = 0.0
        return out
    out[finite] = (vals[finite] - med) / denom
    return out


def _zscore(values: np.ndarray) -> np.ndarray:
    vals = values.astype(float)
    finite = np.isfinite(vals)
    out = np.full_like(vals, np.nan, dtype=float)
    if not np.any(finite):
        return out
    mu = float(np.nanmean(vals[finite]))
    sd = float(np.nanstd(vals[finite]))
    if not np.isfinite(sd) or sd == 0:
        out[finite] = 0.0
        return out
    out[finite] = (vals[finite] - mu) / sd
    return out


def _parse_run_label_to_int(run_value: Any) -> Optional[int]:
    if run_value is None:
        return None
    s = str(run_value).strip()
    if s == "":
        return None
    # Accept "run-01" or "01"
    m = re.match(r"^run-(\d+)$", s)
    if m:
        return int(m.group(1))
    # Accept generic labels with trailing digits (e.g., "session2", "block_03").
    m = re.search(r"(\d+)$", s)
    if m:
        return int(m.group(1))
    try:
        return int(float(s))
    except Exception:
        return None


def _load_fmri_signature_target_for_subject(
    *,
    subject_raw: str,
    task: str,
    deriv_root: Path,
    config: Any,
    events_df: pd.DataFrame,
    logger: logging.Logger,
) -> Tuple[pd.Series, str, pd.DataFrame]:
    """
    Load trial-wise fMRI signature targets and align to clean events by (run, onset, duration).

    Returns (y_series, y_label, extra_meta_df) where extra_meta_df can be merged into meta.
    """
    cfg = _fmri_signature_defaults(config)
    method = cfg["method"]
    if method not in {"beta-series", "lss"}:
        method = "beta-series"

    # Discover run column and onset/duration in clean events
    if "onset" not in events_df.columns or "duration" not in events_df.columns:
        raise ValueError("Clean events.tsv must contain onset and duration to align fMRI trial signatures.")

    run_series = _find_block_column(events_df)
    if run_series is None or not np.any(np.isfinite(run_series.to_numpy(dtype=float))):
        raise ValueError("Clean events.tsv is missing a usable run/block column (expected block/run_id/run/session).")

    round_decimals = int(cfg["round_decimals"])
    onset = pd.to_numeric(events_df["onset"], errors="coerce").to_numpy(dtype=float)
    duration = pd.to_numeric(events_df["duration"], errors="coerce").to_numpy(dtype=float)
    run_int = pd.to_numeric(run_series, errors="coerce").to_numpy(dtype=float)

    def _mk_key(run_num: float, on: float, dur: float) -> Optional[str]:
        if not (np.isfinite(run_num) and np.isfinite(on) and np.isfinite(dur)):
            return None
        return f"{int(run_num)}|{round(float(on), round_decimals):.{round_decimals}f}|{round(float(dur), round_decimals):.{round_decimals}f}"

    eeg_keys = [_mk_key(r, o, d) for r, o, d in zip(run_int, onset, duration)]

    def _mk_trial_key(run_num: float, trial_num: float) -> Optional[str]:
        if not (np.isfinite(run_num) and np.isfinite(trial_num)):
            return None
        return f"{int(run_num)}|{int(round(float(trial_num)))}"

    def _first_finite_numeric(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
        for col in candidates:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            if np.any(np.isfinite(vals.to_numpy(dtype=float))):
                return vals
        return None

    subject_bids = f"sub-{subject_raw}" if not str(subject_raw).startswith("sub-") else str(subject_raw)
    base = deriv_root / subject_bids / "fmri" / ("beta_series" if method == "beta-series" else "lss")
    contrast = str(cfg["contrast_name"]).strip() or "contrast"
    sig_dir = base / f"task-{task}" / f"contrast-{contrast}" / "signatures"
    sig_path = sig_dir / "trial_signature_expression.tsv"
    trials_path = base / f"task-{task}" / f"contrast-{contrast}" / "trials.tsv"

    if not sig_path.exists():
        candidates = sorted(sig_dir.glob("trial_signature_expression*.tsv"))
        if not candidates:
            raise FileNotFoundError(f"Missing fMRI trial signature table: {sig_path}")
        subject_tokens = {
            str(subject_bids).lower(),
            str(subject_raw).lower(),
            str(subject_bids).replace("sub-", "", 1).lower(),
        }
        preferred = [p for p in candidates if any(tok and tok in p.stem.lower() for tok in subject_tokens)]
        sig_path = preferred[0] if preferred else candidates[0]

    sig_df = read_tsv(sig_path)
    if sig_df is None or sig_df.empty:
        raise ValueError(f"Empty fMRI trial signature table: {sig_path}")

    sig_name = str(cfg["signature_name"]).strip().upper()
    metric = str(cfg["metric"]).strip().lower()
    if metric not in {"dot", "cosine", "pearson_r"}:
        metric = "dot"

    if "signature" not in sig_df.columns:
        raise ValueError(f"fMRI signature table missing required column 'signature': {sig_path}")
    if metric not in sig_df.columns:
        # Backward-compatible handling: some older outputs might not include cosine/pearson.
        raise ValueError(f"fMRI signature table missing requested metric '{metric}': {sig_path}")

    if trials_path.exists():
        trials_df = read_tsv(trials_path)
    else:
        trials_df = None
    if (
        trials_df is not None
        and not trials_df.empty
        and "run" in sig_df.columns
        and "trial_index" in sig_df.columns
        and "run" in trials_df.columns
        and "trial_index" in trials_df.columns
    ):
        enrich_cols = [
            c
            for c in ("onset", "duration", "run_num", "trial_number", "events_trial_number")
            if c in trials_df.columns
        ]
        if enrich_cols:
            sig_df = sig_df.merge(
                trials_df[["run", "trial_index", *enrich_cols]],
                on=["run", "trial_index"],
                how="left",
                suffixes=("", "_trial"),
            )
            for col in enrich_cols:
                trial_col = f"{col}_trial"
                if trial_col not in sig_df.columns:
                    continue
                left = pd.to_numeric(sig_df[col], errors="coerce") if col in sig_df.columns else pd.Series(np.nan, index=sig_df.index)
                right = pd.to_numeric(sig_df[trial_col], errors="coerce")
                sig_df[col] = left.where(np.isfinite(left.to_numpy(dtype=float)), right)

    # Parse run number
    if "run_num" not in sig_df.columns:
        if "run" in sig_df.columns:
            sig_df["run_num"] = sig_df["run"].map(_parse_run_label_to_int)
        else:
            sig_df["run_num"] = np.nan

    sig_df["run_num"] = pd.to_numeric(sig_df["run_num"], errors="coerce")
    if "onset" in sig_df.columns:
        sig_df["onset"] = pd.to_numeric(sig_df["onset"], errors="coerce")
    if "duration" in sig_df.columns:
        sig_df["duration"] = pd.to_numeric(sig_df["duration"], errors="coerce")

    sig_df = sig_df.loc[sig_df["signature"].astype(str).str.upper() == sig_name].copy()
    if sig_df.empty:
        raise ValueError(f"No rows found for signature={sig_name} in {sig_path}")

    sig_df[metric] = pd.to_numeric(sig_df[metric], errors="coerce")
    sig_df = sig_df.loc[np.isfinite(sig_df[metric].to_numpy(dtype=float))].copy()
    if sig_df.empty:
        raise ValueError(f"All fMRI signature values are non-finite for signature={sig_name}, metric={metric}")

    sig_df["__key__"] = None
    if {"onset", "duration"}.issubset(sig_df.columns):
        sig_df["__key__"] = [
            _mk_key(r, o, d)
            for r, o, d in zip(
                sig_df["run_num"].to_numpy(dtype=float),
                sig_df["onset"].to_numpy(dtype=float),
                sig_df["duration"].to_numpy(dtype=float),
            )
        ]

    events_trial = _first_finite_numeric(events_df, ["trial_number", "trial_index", "epoch"])
    sig_trial = _first_finite_numeric(sig_df, ["events_trial_number", "trial_number", "trial_index"])
    eeg_trial_keys: List[Optional[str]] = [None] * len(events_df)
    if events_trial is not None:
        eeg_trial_keys = [_mk_trial_key(r, t) for r, t in zip(run_int, events_trial.to_numpy(dtype=float))]
    sig_df["__trial_key__"] = None
    if sig_trial is not None:
        sig_df["__trial_key__"] = [
            _mk_trial_key(r, t)
            for r, t in zip(
                sig_df["run_num"].to_numpy(dtype=float),
                sig_trial.to_numpy(dtype=float),
            )
        ]

    agg_onset = sig_df.loc[sig_df["__key__"].notna()].groupby("__key__", dropna=True)[metric].mean()
    agg_trial = sig_df.loc[sig_df["__trial_key__"].notna()].groupby("__trial_key__", dropna=True)[metric].mean()
    onset_matches = int(sum(1 for k in eeg_keys if k is not None and k in agg_onset.index))
    trial_matches = int(sum(1 for k in eeg_trial_keys if k is not None and k in agg_trial.index))
    if onset_matches == 0 and trial_matches == 0:
        raise ValueError(
            "fMRI signature alignment failed: no matching trials via (run,onset,duration) "
            "or (run,trial_number/trial_index)."
        )

    use_trial_keys = trial_matches >= onset_matches and trial_matches > 0
    active_keys = eeg_trial_keys if use_trial_keys else eeg_keys
    active_agg = agg_trial if use_trial_keys else agg_onset
    active_sig_key_col = "__trial_key__" if use_trial_keys else "__key__"
    y = pd.Series([float(active_agg.get(k)) if k is not None and k in active_agg.index else np.nan for k in active_keys])

    # Optional target normalization
    norm = str(cfg["normalization"]).strip().lower()
    if norm != "none":
        y_arr = y.to_numpy(dtype=float)
        if norm in {"zscore_within_run", "robust_zscore_within_run"}:
            out = np.full_like(y_arr, np.nan, dtype=float)
            for run in sorted({int(r) for r in run_int[np.isfinite(run_int)]}):
                idx = np.where((np.isfinite(y_arr)) & (np.isfinite(run_int)) & (run_int.astype(int) == run))[0]
                if idx.size == 0:
                    continue
                vals = y_arr[idx]
                out[idx] = _robust_zscore(vals) if norm.startswith("robust") else _zscore(vals)
            y = pd.Series(out)
        elif norm in {"zscore_within_subject", "robust_zscore_within_subject"}:
            y = pd.Series(_robust_zscore(y_arr) if norm.startswith("robust") else _zscore(y_arr))

    # Extra meta columns: provide run-level QC covariates when present
    extra_meta = pd.DataFrame(index=np.arange(len(events_df)))
    for col in ("fd_mean", "dvars_mean", "n_motion_outliers", "confounds_n_cols"):
        if col in sig_df.columns:
            # Map per key (assumed constant within run; aggregate by mean)
            m = sig_df.loc[sig_df[active_sig_key_col].notna()].groupby(active_sig_key_col, dropna=True)[col].mean()
            extra_meta[f"fmri_{col}"] = [float(m.get(k)) if k is not None and k in m.index else np.nan for k in active_keys]

    y_label = f"fmri_signature.{method}.{contrast}.{sig_name}.{metric}"
    logger.info(
        "Loaded fMRI signature target for %s: %s (norm=%s, mode=%s, matches=%d/%d, onset_matches=%d, trial_matches=%d)",
        subject_bids,
        y_label,
        norm,
        "trial" if use_trial_keys else "onset",
        int(np.isfinite(y.to_numpy(dtype=float)).sum()),
        len(y),
        onset_matches,
        trial_matches,
    )

    return y, y_label, extra_meta


def _binarize_target(
    y_continuous: pd.Series,
    *,
    threshold: Optional[float],
    positive_rule: Literal[">", ">="] = ">",
) -> pd.Series:
    if threshold is None or not np.isfinite(threshold):
        raise ValueError(
            "Binary target requested but no fixed threshold provided. "
            "Pass --binary-threshold (or config machine_learning.targets.binary_threshold). "
            "This pipeline intentionally forbids median-split binarization to avoid leakage."
        )
    y = pd.to_numeric(y_continuous, errors="coerce")
    out = pd.Series(np.full(len(y), np.nan), index=y.index, dtype=float)
    finite = np.isfinite(y.to_numpy(dtype=float))
    if positive_rule == ">=":
        out.loc[finite] = (y.loc[finite] >= float(threshold)).astype(int).to_numpy(dtype=float)
    else:
        out.loc[finite] = (y.loc[finite] > float(threshold)).astype(int).to_numpy(dtype=float)
    return out


def _ensure_binary_target(
    y_series: pd.Series,
    *,
    binary_threshold: Optional[float],
    positive_rule: Literal[">", ">="] = ">",
) -> pd.Series:
    """Return a 0/1 float series. If finite values are already in {0,1}, use as-is; else binarize with threshold."""
    y_num = pd.to_numeric(y_series, errors="coerce")
    finite = np.isfinite(y_num.to_numpy(dtype=float))
    if np.any(finite):
        uniq = set(np.unique(y_num.loc[finite].to_numpy(dtype=float)).tolist())
        if uniq.issubset({0.0, 1.0}):
            out = pd.Series(np.full(len(y_num), np.nan), index=y_num.index, dtype=float)
            out.loc[finite] = (y_num.loc[finite] > 0).astype(int).to_numpy(dtype=float)
            return out
    return _binarize_target(y_series, threshold=binary_threshold, positive_rule=positive_rule)


def _load_subject_feature_table(
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    *,
    feature_families: List[str],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and concatenate requested feature tables for one subject."""
    subject_raw, subject_bids = _normalize_subject(subject)
    features_dir = deriv_features_path(deriv_root, subject_raw)

    def _load_extraction_config(feature_path: Path) -> Optional[Dict[str, Any]]:
        meta_dir = feature_path.parent / "metadata"
        if not meta_dir.exists():
            return None
        primary = meta_dir / "extraction_config.json"
        candidates = [primary] if primary.exists() else sorted(meta_dir.glob("extraction_config*.json"))
        for p in candidates:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    payload["_config_path"] = str(p)
                    return payload
            except Exception:
                continue
        return None

    def _extract_flag_value(cli: str, flag: str) -> Optional[str]:
        # Very small, robust-enough tokenizer for "--flag value" patterns.
        m = re.search(rf"(?:^|\s){re.escape(flag)}\s+([^\s]+)", cli)
        if not m:
            return None
        return str(m.group(1)).strip()

    def _warn_or_raise_if_feature_tables_not_ml_safe(feature_path: Path, family: str) -> None:
        require_safe = bool(get_config_value(config, "machine_learning.data.require_trial_ml_safe", True))
        meta = _load_extraction_config(feature_path)
        if not meta:
            if require_safe:
                raise ValueError(
                    f"Feature table for family '{family}' is missing extraction metadata "
                    f"(expected metadata/extraction_config*.json near {feature_path}). "
                    "Cannot verify trial_ml_safe provenance while "
                    "machine_learning.data.require_trial_ml_safe=true."
                )
            return
        cli = str(meta.get("cli_command") or meta.get("command") or "")
        cfg_path = str(meta.get("_config_path") or "")

        analysis_mode_meta = str(meta.get("analysis_mode") or "").strip()
        analysis_mode_cli = str(_extract_flag_value(cli, "--analysis-mode") or "").strip()
        analysis_mode = analysis_mode_meta or analysis_mode_cli
        if analysis_mode and analysis_mode.strip().lower() != "trial_ml_safe":
            msg = (
                f"Feature table for family '{family}' appears to have been extracted with "
                f"analysis_mode {analysis_mode!s} (from {cfg_path}). "
                "This can introduce CV leakage for ML. Re-run feature extraction with "
                "--analysis-mode trial_ml_safe (recommended) and then rerun ML."
            )
            if require_safe:
                raise ValueError(msg)
            logger.warning(msg)
        elif not analysis_mode:
            msg = (
                f"Feature table for family '{family}' is missing an explicit analysis_mode "
                f"in extraction metadata ({cfg_path}). Cannot verify trial_ml_safe provenance."
            )
            if require_safe:
                raise ValueError(msg)
            logger.warning(msg)

        if family in {"connectivity", "directedconnectivity", "directed_connectivity", "dconn"}:
            granularity_meta = str(meta.get("connectivity_granularity") or "").strip()
            granularity_cli = str(_extract_flag_value(cli, "--conn-granularity") or "").strip()
            granularity = granularity_meta or granularity_cli
            if granularity and granularity.strip().lower() in {"condition", "subject"}:
                msg = (
                    f"Connectivity features were extracted with --conn-granularity {granularity!s} "
                    f"(from {cfg_path}). This produces condition/subject-aggregated features that are "
                    "broadcast to all trials and will leak condition labels into ML. "
                    "Re-run features with --conn-granularity trial (and ideally --analysis-mode trial_ml_safe)."
                )
                if require_safe:
                    raise ValueError(msg)
                logger.warning(msg)
            elif granularity and granularity.strip().lower() not in {"trial"}:
                msg = (
                    f"Connectivity features use unrecognized granularity={granularity!s} "
                    f"(from {cfg_path}); cannot verify trial-safe provenance."
                )
                if require_safe:
                    raise ValueError(msg)
                logger.warning(msg)
            elif not granularity:
                msg = (
                    f"Feature table for family '{family}' is missing explicit connectivity granularity "
                    f"in extraction metadata ({cfg_path}). Cannot verify connectivity granularity."
                )
                if require_safe:
                    raise ValueError(msg)
                logger.warning(msg)

    dfs: List[pd.DataFrame] = []
    feature_names: List[str] = []
    expected_n_rows: Optional[int] = None

    for family in feature_families:
        fam = str(family).strip()
        if not fam:
            continue

        filename = _resolve_feature_filename(fam, config)
        path = _resolve_feature_path(features_dir, fam, filename)
        if not path.exists():
            logger.warning("Missing feature table for %s (%s): %s", subject_bids, fam, path)
            continue

        _warn_or_raise_if_feature_tables_not_ml_safe(path, family=fam)

        df = read_table(path)
        if df is None or df.empty:
            logger.warning("Empty feature table for %s (%s): %s", subject_bids, fam, path)
            continue

        if expected_n_rows is None:
            expected_n_rows = len(df)
        elif len(df) != expected_n_rows:
            raise ValueError(
                f"Feature length mismatch for {subject_bids}: family '{fam}' has {len(df)} rows "
                f"but expected {expected_n_rows} (cannot safely concatenate)."
            )

        prefixed = _prefix_feature_columns(df, prefix=f"{fam}_")
        dfs.append(prefixed)
        feature_names.extend(list(prefixed.columns))

    if not dfs:
        raise FileNotFoundError(
            f"No requested feature tables found for {subject_bids}. Requested families={feature_families}"
        )

    combined = pd.concat(dfs, axis=1).reset_index(drop=True)
    return combined, feature_names


def _standardize_meta_columns(
    events_df: pd.DataFrame,
    config: Any,
) -> Dict[str, pd.Series]:
    """Extract commonly used covariates into standardized meta column names."""
    meta_cols: Dict[str, pd.Series] = {}

    block_col = _find_block_column(events_df)
    if block_col is not None:
        meta_cols["block"] = block_col

    temp_col = find_temperature_column_in_events(events_df, config)
    if temp_col is not None:
        meta_cols["temperature"] = pd.to_numeric(events_df[temp_col], errors="coerce")

    pain_col = find_pain_column_in_events(events_df, config)
    if pain_col is not None:
        meta_cols["pain_binary"] = pd.to_numeric(events_df[pain_col], errors="coerce")

    rating_columns = config.get("event_columns.rating", [])
    rating_col = pick_target_column(events_df, target_columns=list(rating_columns) if rating_columns else [])
    if rating_col is not None:
        meta_cols["rating"] = pd.to_numeric(events_df[rating_col], errors="coerce")

    if "trial_index" in events_df.columns:
        meta_cols["trial_index"] = pd.to_numeric(events_df["trial_index"], errors="coerce")
    elif "epoch" in events_df.columns:
        meta_cols["trial_index"] = pd.to_numeric(events_df["epoch"], errors="coerce")
    else:
        meta_cols["trial_index"] = pd.Series(np.arange(len(events_df), dtype=int))

    # Helpful for cross-modality alignment (EEG ↔ fMRI).
    if "onset" in events_df.columns:
        meta_cols["onset"] = pd.to_numeric(events_df["onset"], errors="coerce")
    if "duration" in events_df.columns:
        meta_cols["duration"] = pd.to_numeric(events_df["duration"], errors="coerce")

    return meta_cols


def _load_subject_ml_from_features(
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    *,
    feature_families: List[str],
    target: Optional[str],
    target_kind: MLTargetKind,
    binary_threshold: Optional[float],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, np.ndarray, str, pd.DataFrame]:
    """Load (X_df, y, y_col, meta_df) for one subject from feature tables + clean events."""
    subject_raw, subject_bids = _normalize_subject(subject)

    events_df = load_events_df(subject_raw, task, config=config, prefer_clean=True)
    if events_df is None or events_df.empty:
        raise FileNotFoundError(f"Events.tsv not found (or empty) for {subject_bids}, task-{task}.")
    events_df = events_df.reset_index(drop=True)

    X_df, _ = _load_subject_feature_table(
        subject_raw,
        task,
        deriv_root,
        config,
        feature_families=feature_families,
        logger=logger,
    )

    if len(X_df) != len(events_df):
        raise ValueError(
            f"Feature/events count mismatch for {subject_bids}, task-{task}: "
            f"features={len(X_df)} rows, events={len(events_df)} rows. "
            "Re-run feature extraction and/or preprocessing to regenerate aligned clean events."
        )

    extra_meta_df: Optional[pd.DataFrame] = None
    if (target or "").strip().lower() in {"fmri_signature", "fmri-signature"}:
        y_series, y_col, extra_meta_df = _load_fmri_signature_target_for_subject(
            subject_raw=subject_raw,
            task=task,
            deriv_root=deriv_root,
            config=config,
            events_df=events_df,
            logger=logger,
        )
    else:
        y_series, y_col = _resolve_target_series(
            events_df,
            target=target,
            target_kind=target_kind,
            binary_threshold=binary_threshold,
            config=config,
        )

    if target_kind == "binary":
        y_series = _ensure_binary_target(
            y_series, binary_threshold=binary_threshold, positive_rule=">"
        )

    y = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=float)

    meta_base = _standardize_meta_columns(events_df, config)
    meta = pd.DataFrame({k: v.reset_index(drop=True) for k, v in meta_base.items()})
    meta.insert(0, "subject_id", subject_bids)
    if extra_meta_df is not None and not extra_meta_df.empty:
        for c in extra_meta_df.columns:
            if c in meta.columns:
                continue
            meta[c] = pd.to_numeric(extra_meta_df[c], errors="coerce")

    valid_mask = np.isfinite(y)
    if not np.all(valid_mask):
        dropped = int((~valid_mask).sum())
        logger.warning("Dropping %d trials with non-finite targets for %s", dropped, subject_bids)
        X_df = X_df.loc[valid_mask].reset_index(drop=True)
        y = y[valid_mask]
        meta = meta.loc[valid_mask].reset_index(drop=True)

    if target_kind == "binary":
        unique = set(np.unique(y).tolist())
        if not unique.issubset({0.0, 1.0}):
            raise ValueError(f"Binary target contains values outside {{0,1}} for {subject_bids}: {sorted(unique)}")
        y = y.astype(int)

    return X_df, y, y_col, meta




def load_epochs_with_targets(
    deriv_root: Path,
    config: Optional[EEGConfig] = None,
    subjects: Optional[List[str]] = None,
    subject_discovery_policy: Literal["intersection", "union", "config_only"] = "intersection",
    task: str = "",
    target: Optional[str] = None,
    target_kind: MLTargetKind = "continuous",
    binary_threshold: Optional[float] = None,
    bids_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Tuple[str, mne.Epochs, pd.Series]], List[str]]:
    from eeg_pipeline.utils.config.loader import load_config
    
    if config is None:
        config = load_config()
    if logger is None:
        logger = logging.getLogger(__name__)

    if task == "":
        task = config.get("project.task")
    if bids_root is None:
        bids_root = config.bids_root

    if subjects is None or subjects == ["all"]:
        from .subjects import get_available_subjects

        subjects = get_available_subjects(
            config=config,
            deriv_root=deriv_root,
            bids_root=bids_root,
            task=task,
            discovery_sources=["features"],
            subject_discovery_policy=subject_discovery_policy,
            logger=logger,
        )


    out: List[Tuple[str, mne.Epochs, pd.Series]] = []
    ch_sets: List[set] = []

    for s in subjects:
        sub = f"sub-{s}" if not str(s).startswith("sub-") else str(s)

        try:
            from eeg_pipeline.infra.paths import find_clean_epochs_path

            epochs_path = find_clean_epochs_path(s, task, deriv_root=deriv_root, config=config)
        except (FileNotFoundError, ValueError, KeyError):
            epochs_path = None

        if epochs_path is None or not Path(epochs_path).exists():
            logger.warning(f"Clean epochs not found for {sub}; skipping.")
            continue

        epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
        epochs.set_montage(mne.channels.make_standard_montage("standard_1005"))
        bad_channels = epochs.info.get("bads", [])
        if bad_channels:
            epochs.interpolate_bads(reset_bads=True)

        clean_events_path = _find_clean_events_path(
            subject=str(s),
            task=task,
            deriv_root=deriv_root,
            config=config,
            constants=None,
        )
        
        if clean_events_path is None or not clean_events_path.exists():
            logger.warning(f"Clean events.tsv not found for {sub}; skipping.")
            continue
        
        aligned = read_tsv(clean_events_path)
        
        if len(epochs) != len(aligned):
            raise ValueError(
                f"Epoch/events count mismatch for subject {sub}, task {task}: "
                f"epochs have {len(epochs)} trials but clean events.tsv has {len(aligned)} rows. "
                "Re-run preprocessing to regenerate clean events with the current epochs."
            )
        
        if len(aligned) == 0:
            logger.warning(f"Clean events.tsv is empty for {sub}; skipping.")
            continue

        try:
            y_series, _y_col = _resolve_target_series(
                aligned,
                target=target,
                target_kind=target_kind,
                binary_threshold=binary_threshold,
                config=config,
            )
            if target_kind == "binary":
                y_series = _ensure_binary_target(
                    y_series, binary_threshold=binary_threshold, positive_rule=">"
                )
            y = pd.to_numeric(y_series, errors="coerce")
        except ValueError as exc:
            logger.warning("No suitable target column for %s; skipping (%s).", sub, exc)
            continue
        if len(epochs) != len(y):
            logger.error(
                f"Epochs-target length mismatch for subject {sub}, task {task}: "
                f"epochs={len(epochs)}, y={len(y)}. Skipping subject."
            )
            continue

        if len(epochs) == 0:
            logger.warning(f"No trials for {sub}; skipping.")
            continue

        out.append((sub, epochs, y))
        eeg_channels = [
            ch
            for ch in epochs.info["ch_names"]
            if epochs.get_channel_types(picks=[ch])[0] == "eeg"
        ]
        ch_sets.append(set(eeg_channels))

    if not out:
        raise RuntimeError("No epochs + targets could be loaded for any subject.")

    if not ch_sets:
        return out, []

    if len(ch_sets) > 1:
        common_channels = sorted(set.intersection(*ch_sets))
    else:
        common_channels = sorted(ch_sets[0]) if ch_sets else []
    return out, common_channels


def load_active_matrix(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    log: Optional[logging.Logger] = None,
    *,
    feature_families: Optional[List[str]] = None,
    target: Optional[str] = None,
    target_kind: MLTargetKind = "continuous",
    binary_threshold: Optional[float] = None,
    covariates: Optional[List[str]] = None,
    feature_harmonization: Optional[MLFeatureHarmonization] = None,
    feature_bands: Optional[List[str]] = None,
    feature_segments: Optional[List[str]] = None,
    feature_scopes: Optional[List[str]] = None,
    feature_stats: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Canonical ML matrix loader.

    Loads per-trial feature tables produced by the feature pipeline and aligns them to
    the (clean) events.tsv for targets/covariates.

    Notes
    -----
    - This function is intentionally feature-table based (not epochs-based) so it
      stays synchronized with the feature pipeline.
    - For time-domain/ERP-only decoding, use `load_channels_mean_matrix`.
    """
    if log is None:
        log = logging.getLogger(__name__)

    feature_set_cfg = str(get_config_value(config, "machine_learning.data.feature_set", "combined")).strip().lower()
    if feature_set_cfg == "channels_mean" and feature_families is None:
        return load_channels_mean_matrix(
            subjects,
            task,
            deriv_root,
            config,
            log,
            target=target,
            target_kind=target_kind,
            binary_threshold=binary_threshold,
        )

    resolved_families = _resolve_feature_families(
        feature_families=feature_families,
        feature_set=feature_set_cfg,
        config=config,
    )

    harmonization = (
        feature_harmonization
        or str(get_config_value(config, "machine_learning.data.feature_harmonization", "intersection")).strip().lower()
    )
    if harmonization not in {"intersection", "union_impute"}:
        harmonization = "intersection"

    analysis_mode = str(get_config_value(config, "feature_engineering.analysis_mode", "group_stats")).strip()
    if analysis_mode != "trial_ml_safe":
        require_safe = bool(get_config_value(config, "machine_learning.data.require_trial_ml_safe", True))
        msg = (
            "ML loading features while feature_engineering.analysis_mode='%s'. "
            "If any extracted features use cross-trial estimates, this can leak information across CV folds. "
            "For ML, prefer feature_engineering.analysis_mode='trial_ml_safe'."
        ) % analysis_mode
        if require_safe:
            raise ValueError(msg + " (Blocked by machine_learning.data.require_trial_ml_safe=true)")
        log.warning(msg)

    requested_subject_ids = [_normalize_subject(str(s))[1] for s in subjects]
    excluded_subjects: List[Dict[str, str]] = []

    X_dfs: List[pd.DataFrame] = []
    y_list: List[np.ndarray] = []
    groups_list: List[str] = []
    meta_list: List[pd.DataFrame] = []
    col_sets: List[set] = []
    first_cols: Optional[List[str]] = None

    for sub in subjects:
        try:
            X_df, y_sub, _y_col, meta_sub = _load_subject_ml_from_features(
                sub,
                task,
                deriv_root,
                config,
                feature_families=resolved_families,
                target=target,
                target_kind=target_kind,
                binary_threshold=binary_threshold,
                logger=log,
            )
        except (FileNotFoundError, ValueError) as exc:
            log.warning("Skipping %s: %s", str(sub), exc)
            excluded_subjects.append(
                {
                    "subject_id": _normalize_subject(str(sub))[1],
                    "reason": str(exc),
                }
            )
            continue

        if len(y_sub) == 0:
            subject_bids = _normalize_subject(str(sub))[1]
            msg = "No valid trials after target filtering."
            log.warning("Skipping %s: %s", subject_bids, msg)
            excluded_subjects.append({"subject_id": subject_bids, "reason": msg})
            continue

        if first_cols is None:
            first_cols = list(X_df.columns)
        col_sets.append(set(X_df.columns))

        subject_raw, subject_bids = _normalize_subject(sub)
        X_dfs.append(X_df)
        y_list.append(np.asarray(y_sub))
        groups_list.extend([subject_bids] * len(y_sub))
        meta_list.append(meta_sub.reset_index(drop=True))

    if not X_dfs:
        raise RuntimeError(
            f"No subjects produced a usable ML matrix. Requested families={resolved_families}, target={target}"
        )

    if first_cols is None:
        raise RuntimeError("No feature columns detected.")

    all_cols = sorted(set.union(*col_sets)) if col_sets else first_cols
    X_all_df = pd.concat([df.reindex(columns=all_cols) for df in X_dfs], axis=0, ignore_index=True)
    feature_names = all_cols
    if harmonization == "intersection":
        log.info(
            "Using fold-specific intersection harmonization: preserving union feature space at load time "
            "and applying train-only intersection within CV folds."
        )

    # Optional filters by parsed feature metadata (band/segment/scope).
    # These filters are applied after harmonization so "intersection" behaves predictably.
    from eeg_pipeline.domain.features.naming import NamingSchema

    def _sanitize_list(values: Optional[List[str]]) -> Optional[List[str]]:
        if values is None:
            return None
        out = [str(v).strip() for v in values if str(v).strip() != ""]
        return out or None

    bands = _sanitize_list(
        feature_bands if feature_bands is not None else _as_list(get_config_value(config, "machine_learning.data.feature_bands", None))
    )
    segments = _sanitize_list(
        feature_segments if feature_segments is not None else _as_list(get_config_value(config, "machine_learning.data.feature_segments", None))
    )
    scopes = _sanitize_list(
        feature_scopes if feature_scopes is not None else _as_list(get_config_value(config, "machine_learning.data.feature_scopes", None))
    )
    stats = _sanitize_list(
        feature_stats if feature_stats is not None else _as_list(get_config_value(config, "machine_learning.data.feature_stats", None))
    )

    if bands or segments or scopes or stats:
        keep: List[str] = []
        for col in feature_names:
            parsed = NamingSchema.parse(str(col))
            if not parsed.get("valid", False):
                continue
            if bands is not None and parsed.get("band") not in set(bands):
                continue
            if segments is not None and parsed.get("segment") not in set(segments):
                continue
            if scopes is not None and parsed.get("scope") not in set(scopes):
                continue
            if stats is not None and parsed.get("stat") not in set(stats):
                continue
            keep.append(col)

        if not keep:
            raise ValueError(
                "No ML feature columns match requested filters. "
                f"bands={bands}, segments={segments}, scopes={scopes}, stats={stats}. "
                "Tip: run `eeg-pipeline info ml-feature-space --json` to see available values."
            )

        X_all_df = X_all_df.loc[:, keep].copy()
        feature_names = keep

    y_all = np.concatenate(y_list, axis=0)
    groups_arr = np.asarray(groups_list)
    meta = pd.concat(meta_list, axis=0, ignore_index=True)

    meta = meta.reset_index(drop=True)
    meta["trial_id"] = np.arange(len(meta), dtype=int)

    X = X_all_df.to_numpy(dtype=float)

    y_all = y_all.astype(float)

    # Optional covariates appended to X (from standardized meta column names).
    cov_cfg = covariates if covariates is not None else _as_list(get_config_value(config, "machine_learning.data.covariates", []))
    if cov_cfg:
        forbidden_covariates = {v.lower() for v in _target_covariate_aliases(target, config=config)}
        leaking = [c for c in cov_cfg if str(c).strip().lower() in forbidden_covariates]
        if leaking:
            raise ValueError(
                "Covariates include the selected target, which would leak labels into predictors: "
                f"{leaking}. Target={target!r}."
            )

        strict = bool(get_config_value(config, "machine_learning.data.covariates_strict", False))
        present = [c for c in cov_cfg if c in meta.columns]
        missing = [c for c in cov_cfg if c not in meta.columns]
        if missing:
            msg = f"Requested covariates missing from meta: {missing}. Available meta columns={list(meta.columns)}"
            if strict:
                raise ValueError(msg)
            log.warning(msg + " (dropping missing covariates)")
        if present:
            cov_df = meta[present].apply(pd.to_numeric, errors="coerce")
            X = np.concatenate([X, cov_df.to_numpy(dtype=float)], axis=1)
            feature_names = list(feature_names) + [f"cov_{c}" for c in present]

    # Always filter non-finite targets (classification must not silently coerce NaN→0).
    X, y_all, groups_arr, meta = _filter_finite_targets(X, y_all, groups_arr, meta)

    if target_kind == "binary":
        unique = set(np.unique(y_all).tolist())
        if not unique.issubset({0.0, 1.0}):
            raise ValueError(f"Binary target contains values outside {{0,1}} after filtering: {sorted(unique)}")
        y_all = y_all.astype(int)

    included_subject_ids = sorted(set(groups_arr.astype(str).tolist()))
    excluded_by_difference = sorted(set(requested_subject_ids) - set(included_subject_ids))
    excluded_seen = {str(r.get("subject_id", "")) for r in excluded_subjects}
    for subject_id in excluded_by_difference:
        if subject_id not in excluded_seen:
            excluded_subjects.append(
                {
                    "subject_id": subject_id,
                    "reason": "Excluded during matrix assembly (no usable trials/features).",
                }
            )

    n_requested = len(requested_subject_ids)
    n_excluded = len(excluded_by_difference)
    excluded_fraction = float(n_excluded / n_requested) if n_requested > 0 else 0.0
    max_excluded_fraction = float(
        get_config_value(config, "machine_learning.data.max_excluded_subject_fraction", 1.0)
    )

    meta.attrs["requested_subjects"] = requested_subject_ids
    meta.attrs["included_subjects"] = included_subject_ids
    meta.attrs["excluded_subjects"] = excluded_subjects
    meta.attrs["excluded_fraction"] = excluded_fraction
    meta.attrs["max_excluded_subject_fraction"] = max_excluded_fraction
    meta.attrs["feature_harmonization_mode"] = harmonization

    if n_requested > 0 and excluded_fraction > max_excluded_fraction:
        details = "; ".join(
            f"{rec.get('subject_id', '?')}: {rec.get('reason', 'unknown')}"
            for rec in excluded_subjects
            if rec.get("subject_id") in set(excluded_by_difference)
        )
        raise RuntimeError(
            "Too many requested subjects were excluded while building the ML matrix: "
            f"excluded={n_excluded}/{n_requested} ({excluded_fraction:.3f}) exceeds "
            f"machine_learning.data.max_excluded_subject_fraction={max_excluded_fraction:.3f}. "
            f"Excluded subjects: {details}"
        )

    log.info(
        "Built ML matrix from feature tables: X=%s, n_subjects=%d, families=%s, target=%s",
        tuple(X.shape),
        len(np.unique(groups_arr)),
        ",".join(resolved_families),
        str(target or "rating"),
    )

    return X, y_all, groups_arr, feature_names, meta


def load_channels_mean_matrix(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    log: Optional[logging.Logger] = None,
    *,
    target: Optional[str] = None,
    target_kind: MLTargetKind = "continuous",
    binary_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """Legacy ML features: baseline-corrected mean amplitude per EEG channel."""
    if log is None:
        log = logging.getLogger(__name__)

    X_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    groups: List[str] = []
    meta_blocks: List[pd.DataFrame] = []
    feature_cols: Optional[List[str]] = None

    from .epochs import load_epochs_for_analysis

    for sub in subjects:
        epochs, aligned_events = load_epochs_for_analysis(
            sub,
            task,
            align="strict",
            preload=True,
            deriv_root=deriv_root,
            config=config,
            logger=log,
        )
        if epochs is None or aligned_events is None:
            log.warning("No epochs for sub-%s; skipping", str(sub))
            continue

        y_series, _y_col = _resolve_target_series(
            aligned_events,
            target=target,
            target_kind=target_kind,
            binary_threshold=binary_threshold,
            config=config,
        )
        if target_kind == "binary":
            y_series = _ensure_binary_target(
                y_series, binary_threshold=binary_threshold, positive_rule=">"
            )
        y_sub = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=float)

        picks = mne.pick_types(
            epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads"
        )
        if len(picks) == 0:
            log.warning("No EEG channels for sub-%s; skipping", str(sub))
            continue

        data = epochs.get_data(picks=picks)
        times = np.asarray(epochs.times, dtype=float)

        baseline_window = get_config_value(config, "time_frequency_analysis.baseline_window", [-3.0, -0.5])
        active_window = get_config_value(config, "time_frequency_analysis.active_window", [3.0, 10.5])
        try:
            b0, b1 = float(baseline_window[0]), float(baseline_window[1])
            a0, a1 = float(active_window[0]), float(active_window[1])
        except (ValueError, TypeError, IndexError):
            b0, b1 = -3.0, -0.5
            a0, a1 = 3.0, 10.5

        bmask = (times >= b0) & (times < b1)
        amask = (times >= a0) & (times < a1)
        if not np.any(amask):
            log.warning("Active window empty for sub-%s; using full epoch mean.", str(sub))
            amask = np.ones_like(times, dtype=bool)
        if not np.any(bmask):
            log.warning("Baseline window empty for sub-%s; using zero baseline.", str(sub))

        active_mean = np.nanmean(data[..., amask], axis=2)
        epochs_baseline = getattr(epochs, "baseline", None)
        epochs_already_baselined = epochs_baseline not in (None, (None, None))

        if epochs_already_baselined:
            baseline_mean = np.zeros_like(active_mean)
        elif np.any(bmask):
            baseline_mean = np.nanmean(data[..., bmask], axis=2)
        else:
            baseline_mean = np.zeros_like(active_mean)

        X_sub = (active_mean - baseline_mean).astype(float)

        if feature_cols is None:
            feature_cols = list(np.asarray(epochs.ch_names)[picks])
        if X_sub.shape[0] != len(y_sub):
            log.warning("Mismatch X/y for sub-%s; skipping", str(sub))
            continue

        _subject_raw, subject_bids = _normalize_subject(sub)
        X_blocks.append(X_sub)
        y_blocks.append(y_sub)
        groups.extend([subject_bids] * len(y_sub))
        meta_sub = pd.DataFrame(_standardize_meta_columns(aligned_events, config))
        meta_sub.insert(0, "subject_id", subject_bids)
        meta_blocks.append(meta_sub)

    if not X_blocks:
        raise RuntimeError("No subjects with usable epoch data")

    X = np.vstack(X_blocks)
    y_all = np.concatenate(y_blocks)
    groups_arr = np.asarray(groups)
    meta = pd.concat(meta_blocks, axis=0, ignore_index=True)

    X, y_all, groups_arr, meta = _filter_finite_targets(X, y_all, groups_arr, meta)
    meta = meta.reset_index(drop=True)
    meta["trial_id"] = np.arange(len(meta), dtype=int)

    if target_kind == "binary":
        unique = set(np.unique(y_all).tolist())
        if not unique.issubset({0.0, 1.0}):
            raise ValueError(f"Binary target contains values outside {{0,1}} after filtering: {sorted(unique)}")
        y_all = y_all.astype(int)

    return X, y_all, groups_arr, feature_cols or [], meta


def load_epoch_tensor_matrix(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    log: Optional[logging.Logger] = None,
    *,
    target: Optional[str] = None,
    target_kind: MLTargetKind = "continuous",
    binary_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """ML epochs loader for CNN models: returns X as (n_trials, n_channels, n_timepoints)."""
    if log is None:
        log = logging.getLogger(__name__)

    payloads: List[Tuple[str, mne.Epochs, pd.DataFrame, np.ndarray, List[str]]] = []
    ch_sets: List[set] = []

    for sub in subjects:
        epochs, aligned_events = load_epochs_for_analysis(
            sub,
            task,
            align="strict",
            preload=True,
            deriv_root=deriv_root,
            config=config,
            logger=log,
        )
        if epochs is None or aligned_events is None:
            log.warning("No epochs for sub-%s; skipping", str(sub))
            continue

        y_series, _y_col = _resolve_target_series(
            aligned_events,
            target=target,
            target_kind=target_kind,
            binary_threshold=binary_threshold,
            config=config,
        )
        if target_kind == "binary":
            y_series = _ensure_binary_target(
                y_series, binary_threshold=binary_threshold, positive_rule=">"
            )
        y_sub = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=float)

        n_trials = min(len(epochs), len(y_sub), len(aligned_events))
        if n_trials < 1:
            continue
        epochs = epochs[:n_trials]
        aligned_events = aligned_events.iloc[:n_trials].reset_index(drop=True)
        y_sub = y_sub[:n_trials]

        picks = mne.pick_types(
            epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads"
        )
        if len(picks) == 0:
            log.warning("No EEG channels available for sub-%s; skipping", str(sub))
            continue

        ch_names = [str(epochs.ch_names[p]) for p in picks]
        _subject_raw, subject_bids = _normalize_subject(sub)
        payloads.append((subject_bids, epochs, aligned_events, y_sub, ch_names))
        ch_sets.append(set(ch_names))

    if not payloads:
        raise RuntimeError("No subjects with usable epoch tensors for CNN ML")

    common_channels = sorted(set.intersection(*ch_sets)) if len(ch_sets) > 1 else sorted(ch_sets[0])
    if len(common_channels) == 0:
        raise RuntimeError("No common EEG channels across selected subjects for CNN ML")

    X_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    groups: List[str] = []
    meta_blocks: List[pd.DataFrame] = []

    for subject_bids, epochs, aligned_events, y_sub, _ch_names in payloads:
        epochs_common = epochs.copy().pick(common_channels)
        X_sub = epochs_common.get_data(picks="eeg", reject_by_annotation=None).astype(float)
        if X_sub.shape[0] != len(y_sub):
            log.warning("Mismatch X/y for sub-%s; skipping", str(subject_bids))
            continue

        X_blocks.append(X_sub)
        y_blocks.append(y_sub)
        groups.extend([subject_bids] * len(y_sub))
        meta_sub = pd.DataFrame(_standardize_meta_columns(aligned_events, config))
        meta_sub.insert(0, "subject_id", subject_bids)
        meta_blocks.append(meta_sub)

    if not X_blocks:
        raise RuntimeError("No subjects with valid epoch tensor data")

    X = np.concatenate(X_blocks, axis=0)
    y_all = np.concatenate(y_blocks)
    groups_arr = np.asarray(groups)
    meta = pd.concat(meta_blocks, axis=0, ignore_index=True)

    X, y_all, groups_arr, meta = _filter_finite_targets(X, y_all, groups_arr, meta)
    meta = meta.reset_index(drop=True)
    meta["trial_id"] = np.arange(len(meta), dtype=int)

    if target_kind == "binary":
        unique = set(np.unique(y_all).tolist())
        if not unique.issubset({0.0, 1.0}):
            raise ValueError(f"Binary target contains values outside {{0,1}} after filtering: {sorted(unique)}")
        y_all = y_all.astype(int)

    return X, y_all, groups_arr, common_channels, meta


def filter_finite_targets(
    indices: np.ndarray,
    targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    target_values = targets[indices]
    finite_mask = np.isfinite(target_values)
    filtered_indices = indices[finite_mask]
    filtered_targets = target_values[finite_mask]
    return filtered_indices, filtered_targets


def extract_epoch_data_block(
    indices: np.ndarray,
    trial_records: List[Tuple[str, int]],
    aligned_epochs: Dict[str, mne.Epochs],
) -> np.ndarray:
    """Extract epoch data for specified trial indices."""
    X_list = []
    for idx in indices:
        subject_id, trial_idx = trial_records[int(idx)]
        epochs = aligned_epochs[subject_id]
        X_i = epochs.get_data(picks="eeg", reject_by_annotation=None)[trial_idx]
        X_list.append(X_i)
    return np.stack(X_list, axis=0)


def prepare_trial_records_from_epochs(
    tuples: List[Tuple[str, mne.Epochs, pd.Series]],
) -> Tuple[
    List[Tuple[str, int]],
    np.ndarray,
    np.ndarray,
    Dict[str, mne.Epochs],
    Dict[str, pd.Series],
]:
    trial_records: List[Tuple[str, int]] = []
    y_all_list: List[float] = []
    groups_list: List[str] = []
    subj_to_epochs: Dict[str, mne.Epochs] = {}
    subj_to_y: Dict[str, pd.Series] = {}

    for subject_id, epochs, y in tuples:
        n_trials = min(len(epochs), len(y))
        if n_trials == 0:
            continue
        subj_to_epochs[subject_id] = epochs
        y_numeric = pd.to_numeric(y.iloc[:n_trials], errors="coerce")
        subj_to_y[subject_id] = y_numeric
        for trial_idx in range(n_trials):
            trial_records.append((subject_id, trial_idx))
            y_all_list.append(float(y_numeric.iloc[trial_idx]))
            groups_list.append(subject_id)

    if len(trial_records) == 0:
        raise RuntimeError("No trial data available.")

    y_all_arr = np.asarray(y_all_list)
    groups_arr = np.asarray(groups_list)
    return trial_records, y_all_arr, groups_arr, subj_to_epochs, subj_to_y
