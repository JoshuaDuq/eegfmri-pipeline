from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.infra.tsv import read_tsv
from eeg_pipeline.utils.config.loader import require_config_value


def parse_run_label_to_int(run_value: Any) -> Optional[int]:
    if run_value is None:
        return None
    s = str(run_value).strip()
    if s == "":
        return None
    match = re.match(r"^run-(\d+)$", s)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)$", s)
    if match:
        return int(match.group(1))
    try:
        return int(float(s))
    except Exception:
        return None


def find_block_column(aligned_events: pd.DataFrame) -> Optional[pd.Series]:
    """Find block/run identifier column from aligned events."""
    for candidate in ("block", "run_id", "run", "session"):
        if candidate not in aligned_events.columns:
            continue
        series = aligned_events[candidate]
        numeric = pd.to_numeric(series, errors="coerce")
        if np.any(np.isfinite(numeric.to_numpy(dtype=float))):
            return numeric
        parsed = pd.Series(
            [parse_run_label_to_int(value) for value in series],
            index=series.index,
            dtype="float64",
        )
        if np.any(np.isfinite(parsed.to_numpy(dtype=float))):
            return parsed
        return numeric
    return None


def _signature_target_defaults(config: Any, *, config_path: str) -> dict[str, Any]:
    return {
        "method": str(require_config_value(config, f"{config_path}.method")).strip().lower(),
        "contrast_name": str(
            require_config_value(config, f"{config_path}.contrast_name")
        ).strip(),
        "signature_name": str(
            require_config_value(config, f"{config_path}.signature_name")
        ).strip(),
        "metric": str(require_config_value(config, f"{config_path}.metric")).strip().lower(),
        "normalization": str(
            require_config_value(config, f"{config_path}.normalization")
        ).strip().lower(),
        "round_decimals": int(require_config_value(config, f"{config_path}.round_decimals")),
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


def _first_finite_numeric(
    frame: pd.DataFrame,
    candidates: List[str],
) -> Optional[pd.Series]:
    for col in candidates:
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        if np.any(np.isfinite(vals.to_numpy(dtype=float))):
            return vals
    return None


def load_fmri_signature_target_for_subject(
    *,
    subject_raw: str,
    task: str,
    deriv_root: Path,
    config: Any,
    events_df: pd.DataFrame,
    logger: logging.Logger,
    config_path: str = "machine_learning.fmri_signature",
) -> Tuple[pd.Series, str, pd.DataFrame]:
    """
    Load trial-wise fMRI signature targets and align to clean events by run/onset/duration
    or run/trial number.
    """
    cfg = _signature_target_defaults(config, config_path=config_path)
    method = cfg["method"]
    if method not in {"beta-series", "lss"}:
        method = "beta-series"

    if "onset" not in events_df.columns or "duration" not in events_df.columns:
        raise ValueError("Clean events.tsv must contain onset and duration to align fMRI trial signatures.")

    run_series = find_block_column(events_df)
    if run_series is None or not np.any(np.isfinite(run_series.to_numpy(dtype=float))):
        raise ValueError("Clean events.tsv is missing a usable run/block column (expected block/run_id/run/session).")

    round_decimals = int(cfg["round_decimals"])
    onset = pd.to_numeric(events_df["onset"], errors="coerce").to_numpy(dtype=float)
    duration = pd.to_numeric(events_df["duration"], errors="coerce").to_numpy(dtype=float)
    run_int = pd.to_numeric(run_series, errors="coerce").to_numpy(dtype=float)

    def _mk_key(run_num: float, on: float, dur: float) -> Optional[str]:
        if not (np.isfinite(run_num) and np.isfinite(on) and np.isfinite(dur)):
            return None
        return (
            f"{int(run_num)}|"
            f"{round(float(on), round_decimals):.{round_decimals}f}|"
            f"{round(float(dur), round_decimals):.{round_decimals}f}"
        )

    def _mk_trial_key(run_num: float, trial_num: float) -> Optional[str]:
        if not (np.isfinite(run_num) and np.isfinite(trial_num)):
            return None
        return f"{int(run_num)}|{int(round(float(trial_num)))}"

    eeg_keys = [_mk_key(r, o, d) for r, o, d in zip(run_int, onset, duration)]

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
        preferred = [
            path
            for path in candidates
            if any(token and token in path.stem.lower() for token in subject_tokens)
        ]
        sig_path = preferred[0] if preferred else candidates[0]

    sig_df = read_tsv(sig_path)
    if sig_df is None or sig_df.empty:
        raise ValueError(f"Empty fMRI trial signature table: {sig_path}")

    requested_signature = str(cfg["signature_name"]).strip()
    metric = str(cfg["metric"]).strip().lower()
    if metric not in {"dot", "cosine", "pearson_r"}:
        metric = "dot"

    if "signature" not in sig_df.columns:
        raise ValueError(f"fMRI signature table missing required column 'signature': {sig_path}")
    if metric not in sig_df.columns:
        raise ValueError(f"fMRI signature table missing requested metric '{metric}': {sig_path}")

    trials_df = read_tsv(trials_path) if trials_path.exists() else None
    if (
        trials_df is not None
        and not trials_df.empty
        and "run" in sig_df.columns
        and "trial_index" in sig_df.columns
        and "run" in trials_df.columns
        and "trial_index" in trials_df.columns
    ):
        enrich_cols = [
            col
            for col in ("onset", "duration", "run_num", "trial_number", "events_trial_number")
            if col in trials_df.columns
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
                left = (
                    pd.to_numeric(sig_df[col], errors="coerce")
                    if col in sig_df.columns
                    else pd.Series(np.nan, index=sig_df.index)
                )
                right = pd.to_numeric(sig_df[trial_col], errors="coerce")
                sig_df[col] = left.where(np.isfinite(left.to_numpy(dtype=float)), right)

    if "run_num" not in sig_df.columns:
        if "run" in sig_df.columns:
            sig_df["run_num"] = sig_df["run"].map(parse_run_label_to_int)
        else:
            sig_df["run_num"] = np.nan

    sig_df["run_num"] = pd.to_numeric(sig_df["run_num"], errors="coerce")
    if "onset" in sig_df.columns:
        sig_df["onset"] = pd.to_numeric(sig_df["onset"], errors="coerce")
    if "duration" in sig_df.columns:
        sig_df["duration"] = pd.to_numeric(sig_df["duration"], errors="coerce")

    sig_df[metric] = pd.to_numeric(sig_df[metric], errors="coerce")
    signature_values = sig_df["signature"].astype(str).str.strip()
    finite_metric = np.isfinite(sig_df[metric].to_numpy(dtype=float))
    available_signatures = sorted({val for val in signature_values if val})
    available_signatures_with_metric = sorted(
        {
            val
            for val in signature_values[finite_metric]
            if isinstance(val, str) and val.strip()
        }
    )
    if not available_signatures:
        raise ValueError(f"No signature names found in {sig_path}")

    if requested_signature.lower() in {"", "auto"}:
        if not available_signatures_with_metric:
            raise ValueError(
                f"No finite values found for metric={metric} in {sig_path}. "
                f"Available signatures: {available_signatures}"
            )
        sig_name = available_signatures_with_metric[0]
    else:
        sig_name = requested_signature

    sig_df = sig_df.loc[
        (signature_values.str.casefold() == sig_name.casefold()) & finite_metric
    ].copy()
    if sig_df.empty:
        raise ValueError(
            f"No finite values found for signature={sig_name}, metric={metric} in {sig_path}. "
            f"Available signatures: {available_signatures_with_metric or available_signatures}"
        )

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
        eeg_trial_keys = [
            _mk_trial_key(r, t) for r, t in zip(run_int, events_trial.to_numpy(dtype=float))
        ]
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
    onset_matches = sum(1 for key in eeg_keys if key is not None and key in agg_onset.index)
    trial_matches = sum(1 for key in eeg_trial_keys if key is not None and key in agg_trial.index)
    if onset_matches == 0 and trial_matches == 0:
        raise ValueError(
            "fMRI signature alignment failed: no matching trials via (run,onset,duration) "
            "or (run,trial_number/trial_index)."
        )

    use_trial_keys = trial_matches >= onset_matches and trial_matches > 0
    active_keys = eeg_trial_keys if use_trial_keys else eeg_keys
    active_agg = agg_trial if use_trial_keys else agg_onset
    active_sig_key_col = "__trial_key__" if use_trial_keys else "__key__"
    y = pd.Series(
        [
            float(active_agg.get(key))
            if key is not None and key in active_agg.index
            else np.nan
            for key in active_keys
        ]
    )

    norm = str(cfg["normalization"]).strip().lower()
    if norm != "none":
        y_arr = y.to_numpy(dtype=float)
        if norm in {"zscore_within_run", "robust_zscore_within_run"}:
            out = np.full_like(y_arr, np.nan, dtype=float)
            for run in sorted({int(r) for r in run_int[np.isfinite(run_int)]}):
                idx = np.where(
                    (np.isfinite(y_arr))
                    & (np.isfinite(run_int))
                    & (run_int.astype(int) == run)
                )[0]
                if idx.size == 0:
                    continue
                vals = y_arr[idx]
                out[idx] = _robust_zscore(vals) if norm.startswith("robust") else _zscore(vals)
            y = pd.Series(out)
        elif norm in {"zscore_within_subject", "robust_zscore_within_subject"}:
            y = pd.Series(_robust_zscore(y_arr) if norm.startswith("robust") else _zscore(y_arr))

    extra_meta = pd.DataFrame(index=np.arange(len(events_df)))
    for col in ("fd_mean", "dvars_mean", "n_motion_outliers", "confounds_n_cols"):
        if col not in sig_df.columns:
            continue
        mapped = sig_df.loc[sig_df[active_sig_key_col].notna()].groupby(active_sig_key_col, dropna=True)[
            col
        ].mean()
        extra_meta[f"fmri_{col}"] = [
            float(mapped.get(key)) if key is not None and key in mapped.index else np.nan
            for key in active_keys
        ]

    y_label = f"fmri_signature.{method}.{contrast}.{sig_name}.{metric}"
    logger.info(
        "Loaded fMRI signature target for %s: %s (norm=%s, mode=%s, matches=%d/%d, onset_matches=%d, trial_matches=%d)",
        subject_bids,
        y_label,
        norm,
        "trial" if use_trial_keys else "onset",
        int(np.isfinite(y.to_numpy(dtype=float)).sum()),
        len(y),
        int(onset_matches),
        int(trial_matches),
    )
    return y, y_label, extra_meta


__all__ = [
    "find_block_column",
    "load_fmri_signature_target_for_subject",
    "parse_run_label_to_int",
]
