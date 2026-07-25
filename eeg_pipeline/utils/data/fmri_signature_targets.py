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


def find_run_column(aligned_events: pd.DataFrame) -> Optional[pd.Series]:
    """Return the required run identifier column from aligned events."""
    if "run" not in aligned_events.columns:
        return None
    series = aligned_events["run"]
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


def _signature_target_defaults(config: Any, *, config_path: str) -> dict[str, Any]:
    return {
        "method": str(require_config_value(config, f"{config_path}.method")).strip().lower(),
        "contrast_name": str(require_config_value(config, f"{config_path}.contrast_name")).strip(),
        "signature_name": str(
            require_config_value(config, f"{config_path}.signature_name")
        ).strip(),
        "metric": str(require_config_value(config, f"{config_path}.metric")).strip().lower(),
        "normalization": str(require_config_value(config, f"{config_path}.normalization"))
        .strip()
        .lower(),
        "round_decimals": int(require_config_value(config, f"{config_path}.round_decimals")),
    }


def _optional_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        value = config.get(key, default)
        if value is not None:
            return value
    if isinstance(config, dict):
        current: Any = config
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current
    return default


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


def _raise_on_duplicate_keys(
    frame: pd.DataFrame,
    key_col: str,
    label: str,
) -> None:
    keyed = frame.loc[frame[key_col].notna(), key_col]
    if keyed.empty:
        return
    counts = keyed.value_counts()
    duplicates = counts[counts > 1]
    if duplicates.empty:
        return
    examples = ", ".join(str(key) for key in duplicates.index[:5])
    raise ValueError(
        f"ambiguous fMRI signature alignment: duplicate {label} keys are not allowed "
        f"({examples})."
    )


def _raise_on_missing_or_duplicate_trial_keys(
    keys: List[Optional[str]],
    label: str,
) -> None:
    missing = [idx for idx, key in enumerate(keys) if key is None]
    if missing:
        raise ValueError(
            f"fMRI signature alignment requires finite trial identifiers for every {label} row; "
            f"missing at rows {missing[:5]}."
        )
    frame = pd.DataFrame({"__trial_key__": keys})
    _raise_on_duplicate_keys(frame, "__trial_key__", f"{label} (run,trial)")


def _unique_values_by_key(
    frame: pd.DataFrame,
    key_col: str,
    value_col: str,
    label: str,
) -> pd.Series:
    _raise_on_duplicate_keys(frame, key_col, label)
    keyed = frame.loc[frame[key_col].notna(), [key_col, value_col]].copy()
    if keyed.empty:
        return pd.Series(dtype=float)
    return keyed.set_index(key_col)[value_col]


def _values_for_keys(keys: List[Optional[str]], values: pd.Series) -> pd.Series:
    return pd.Series(
        [
            float(values.get(key)) if key is not None and key in values.index else np.nan
            for key in keys
        ]
    )


def _metadata_values_for_keys(keys: List[Optional[str]], values: pd.Series) -> pd.Series:
    return pd.Series(
        [values.get(key) if key is not None and key in values.index else np.nan for key in keys]
    )


def _read_target_table(table_path: Path) -> pd.DataFrame:
    if not table_path.exists():
        raise FileNotFoundError(f"Configured fMRI signature target table not found: {table_path}")
    suffix = table_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    if suffix == ".tsv":
        return pd.read_csv(table_path, sep="\t")
    raise ValueError(
        "Configured fMRI signature target table must be .parquet or .tsv, " f"got: {table_path}"
    )


def _load_target_table_values_for_subject(
    *,
    table_path: Path,
    subject_bids: str,
    task: str,
    target_column: str,
    events_df: pd.DataFrame,
    run_int: np.ndarray,
    eeg_time_keys: List[Optional[str]],
    eeg_trial_keys: List[Optional[str]],
    round_decimals: int,
) -> tuple[pd.Series, pd.DataFrame]:
    target_column = str(target_column).strip()
    if not target_column:
        raise ValueError("machine_learning.fmri_signature.target_column must be non-empty.")

    table = _read_target_table(table_path)
    required = {"subject_id", "task", "run", "trial_index", "onset", "duration", target_column}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(
            "Configured fMRI signature target table is missing required columns: "
            f"{missing}. Path: {table_path}"
        )

    subject_rows = table.loc[
        (table["subject_id"].astype(str) == subject_bids) & (table["task"].astype(str) == str(task))
    ].copy()
    if subject_rows.empty:
        raise ValueError(
            f"Configured fMRI signature target table has no rows for {subject_bids}, task-{task}: "
            f"{table_path}"
        )

    target_runs = pd.to_numeric(subject_rows["run"], errors="coerce").to_numpy(dtype=float)
    target_trials = pd.to_numeric(subject_rows["trial_index"], errors="coerce").to_numpy(
        dtype=float
    )
    target_onset = pd.to_numeric(subject_rows["onset"], errors="coerce").to_numpy(dtype=float)
    target_duration = pd.to_numeric(subject_rows["duration"], errors="coerce").to_numpy(dtype=float)
    target_values = pd.to_numeric(subject_rows[target_column], errors="coerce")

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

    target_time_keys = [
        _mk_key(run_num, onset, duration)
        for run_num, onset, duration in zip(target_runs, target_onset, target_duration)
    ]
    target_trial_keys = [
        _mk_trial_key(run_num, trial_num) for run_num, trial_num in zip(target_runs, target_trials)
    ]

    target_frame = pd.DataFrame(
        {
            "__key__": target_time_keys,
            "__trial_key__": target_trial_keys,
            target_column: target_values,
        }
    )
    _raise_on_missing_or_duplicate_trial_keys(eeg_trial_keys, "EEG event")
    trial_values = _unique_values_by_key(
        target_frame,
        "__trial_key__",
        target_column,
        "(run,trial)",
    )
    time_values = _unique_values_by_key(
        target_frame,
        "__key__",
        target_column,
        "(run,onset,duration)",
    )

    trial_y = _values_for_keys(eeg_trial_keys, trial_values)
    time_y = _values_for_keys(eeg_time_keys, time_values)
    trial_matches = int(np.isfinite(trial_y.to_numpy(dtype=float)).sum())
    time_matches = int(np.isfinite(time_y.to_numpy(dtype=float)).sum())
    if time_matches > 0:
        time_matched = np.isfinite(time_y.to_numpy(dtype=float))
        trial_matched = np.isfinite(trial_y.to_numpy(dtype=float))
        if not np.array_equal(time_matched, time_matched & trial_matched):
            raise ValueError(
                "Configured fMRI signature target table alignment temporal audit is ambiguous: "
                "onset/duration keys match rows that are not matched by trial identifiers."
            )
        both_matched = time_matched & trial_matched
        if np.any(both_matched) and not np.allclose(
            time_y.to_numpy(dtype=float)[both_matched],
            trial_y.to_numpy(dtype=float)[both_matched],
            equal_nan=True,
        ):
            raise ValueError(
                "Configured fMRI signature target table alignment temporal audit is ambiguous: "
                "trial-number and onset/duration keys match different target values."
            )

    active_keys = eeg_trial_keys
    active_key_column = "__trial_key__"
    y = trial_y
    y_arr = y.to_numpy(dtype=float)
    if len(y_arr) != len(events_df) or trial_matches == 0:
        raise ValueError(
            "Configured fMRI signature target table must align finite values to at least one "
            f"clean EEG event by unique trial identifiers for {subject_bids}, "
            f"task-{task}, target={target_column}; matched {trial_matches}/{len(events_df)}."
        )

    extra = _aligned_numeric_target_table_columns(
        subject_rows=subject_rows,
        active_key_column=active_key_column,
        active_keys=active_keys,
        target_column=target_column,
        round_decimals=round_decimals,
    )
    return y, extra


def _aligned_numeric_target_table_columns(
    *,
    subject_rows: pd.DataFrame,
    active_key_column: str,
    active_keys: List[Optional[str]],
    target_column: str,
    round_decimals: int,
) -> pd.DataFrame:
    excluded = {
        "subject_id",
        "task",
        "run",
        "trial_index",
        "onset",
        "duration",
        "NPS",
        "SIIPS1",
        "NPS_nuisance_residual",
        "SIIPS1_nuisance_residual",
        target_column,
    }
    extra = pd.DataFrame(index=np.arange(len(active_keys)))
    source = subject_rows.copy()
    if active_key_column == "__trial_key__":
        source[active_key_column] = [
            (
                f"{int(run_num)}|{int(round(float(trial_num)))}"
                if np.isfinite(run_num) and np.isfinite(trial_num)
                else None
            )
            for run_num, trial_num in zip(
                pd.to_numeric(source["run"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(source["trial_index"], errors="coerce").to_numpy(dtype=float),
            )
        ]
    else:
        source[active_key_column] = [
            (
                f"{int(run_num)}|"
                f"{round(float(onset), round_decimals):.{round_decimals}f}|"
                f"{round(float(duration), round_decimals):.{round_decimals}f}"
                if np.isfinite(run_num) and np.isfinite(onset) and np.isfinite(duration)
                else None
            )
            for run_num, onset, duration in zip(
                pd.to_numeric(source["run"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(source["onset"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(source["duration"], errors="coerce").to_numpy(dtype=float),
            )
        ]

    for column in source.columns:
        if column in excluded or column.startswith("__"):
            continue
        numeric = pd.to_numeric(source[column], errors="coerce")
        if not np.any(np.isfinite(numeric.to_numpy(dtype=float))):
            continue
        mapped = _unique_values_by_key(
            pd.DataFrame({"key": source[active_key_column], "value": numeric}),
            "key",
            "value",
            "(run,trial)",
        )
        extra[column] = [
            float(mapped.get(key)) if key is not None and key in mapped.index else np.nan
            for key in active_keys
        ]
    return extra


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
    """Load trial-wise fMRI signature targets and align to clean events by run/trial id."""
    cfg = _signature_target_defaults(config, config_path=config_path)
    method = cfg["method"]
    if method not in {"beta-series", "lss"}:
        method = "beta-series"

    if "onset" not in events_df.columns or "duration" not in events_df.columns:
        raise ValueError(
            "Clean events.tsv must contain onset and duration to align fMRI trial signatures."
        )

    run_series = find_run_column(events_df)
    if run_series is None or not np.any(np.isfinite(run_series.to_numpy(dtype=float))):
        raise ValueError("Clean events.tsv is missing a usable 'run' column.")

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
    events_trial = _first_finite_numeric(events_df, ["trial_number", "trial_index", "epoch"])
    eeg_trial_keys: List[Optional[str]] = [None] * len(events_df)
    if events_trial is not None:
        eeg_trial_keys = [
            _mk_trial_key(r, t) for r, t in zip(run_int, events_trial.to_numpy(dtype=float))
        ]
    _raise_on_missing_or_duplicate_trial_keys(eeg_trial_keys, "EEG event")

    subject_bids = (
        f"sub-{subject_raw}" if not str(subject_raw).startswith("sub-") else str(subject_raw)
    )
    target_table_raw = _optional_config_value(config, f"{config_path}.target_table_path")
    if target_table_raw:
        norm = str(cfg["normalization"]).strip().lower()
        if norm != "none":
            raise ValueError(
                "Configured fMRI signature target tables must contain final target values; "
                "set normalization='none' to avoid applying a second target normalization."
            )
        target_column = str(
            _optional_config_value(config, f"{config_path}.target_column", cfg["signature_name"])
        ).strip()
        y, extra_meta = _load_target_table_values_for_subject(
            table_path=Path(str(target_table_raw)).expanduser(),
            subject_bids=subject_bids,
            task=task,
            target_column=target_column,
            events_df=events_df,
            run_int=run_int,
            eeg_time_keys=eeg_keys,
            eeg_trial_keys=eeg_trial_keys,
            round_decimals=round_decimals,
        )
        y_label = f"fmri_signature.primary_targets.{target_column}"
        logger.info(
            "Loaded fMRI signature target for %s from primary target table: %s (matches=%d/%d)",
            subject_bids,
            y_label,
            int(np.isfinite(y.to_numpy(dtype=float)).sum()),
            len(y),
        )
        return y, y_label, extra_meta

    base = (
        deriv_root / subject_bids / "fmri" / ("beta_series" if method == "beta-series" else "lss")
    )
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
        {val for val in signature_values[finite_metric] if isinstance(val, str) and val.strip()}
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

    sig_trial = _first_finite_numeric(
        sig_df, ["events_trial_number", "trial_number", "trial_index"]
    )
    sig_df["__trial_key__"] = None
    if sig_trial is not None:
        sig_df["__trial_key__"] = [
            _mk_trial_key(r, t)
            for r, t in zip(
                sig_df["run_num"].to_numpy(dtype=float),
                sig_trial.to_numpy(dtype=float),
            )
        ]

    trial_values = _unique_values_by_key(sig_df, "__trial_key__", metric, "(run,trial)")
    onset_values = _unique_values_by_key(sig_df, "__key__", metric, "(run,onset,duration)")
    trial_y = _values_for_keys(eeg_trial_keys, trial_values)
    onset_y = _values_for_keys(eeg_keys, onset_values)
    trial_matches = int(np.isfinite(trial_y.to_numpy(dtype=float)).sum())
    onset_matches = int(np.isfinite(onset_y.to_numpy(dtype=float)).sum())
    if onset_matches > 0:
        onset_matched = np.isfinite(onset_y.to_numpy(dtype=float))
        trial_matched = np.isfinite(trial_y.to_numpy(dtype=float))
        if not np.array_equal(onset_matched, onset_matched & trial_matched):
            raise ValueError(
                "ambiguous fMRI signature alignment temporal audit: onset/duration keys match rows "
                "that are not matched by trial identifiers."
            )
        both_matched = onset_matched & trial_matched
        if np.any(both_matched) and not np.allclose(
            onset_y.to_numpy(dtype=float)[both_matched],
            trial_y.to_numpy(dtype=float)[both_matched],
            equal_nan=True,
        ):
            raise ValueError(
                "ambiguous fMRI signature alignment temporal audit: trial-number and onset/duration "
                "keys match different target values."
            )

    active_keys = eeg_trial_keys
    active_agg = trial_values
    active_sig_key_col = "__trial_key__"
    y = _values_for_keys(active_keys, active_agg)
    y_arr = y.to_numpy(dtype=float)
    if len(y_arr) != len(events_df) or trial_matches == 0:
        raise ValueError(
            "fMRI signature alignment must match finite values to at least one clean EEG "
            f"event by unique trial identifiers; matched {trial_matches}/{len(events_df)}."
        )

    norm = str(cfg["normalization"]).strip().lower()
    if norm != "none":
        if norm in {"zscore_within_run", "robust_zscore_within_run"}:
            out = np.full_like(y_arr, np.nan, dtype=float)
            for run in sorted({int(r) for r in run_int[np.isfinite(run_int)]}):
                idx = np.where(
                    (np.isfinite(y_arr)) & (np.isfinite(run_int)) & (run_int.astype(int) == run)
                )[0]
                if idx.size == 0:
                    continue
                vals = y_arr[idx]
                out[idx] = _robust_zscore(vals) if norm.startswith("robust") else _zscore(vals)
            y = pd.Series(out)
        elif norm in {"zscore_within_subject", "robust_zscore_within_subject"}:
            y = pd.Series(_robust_zscore(y_arr) if norm.startswith("robust") else _zscore(y_arr))

    extra_meta = pd.DataFrame(index=np.arange(len(events_df)))
    for col in ("n_voxels", "fd_mean", "dvars_mean", "n_motion_outliers", "confounds_n_cols"):
        if col not in sig_df.columns:
            continue
        mapped = _unique_values_by_key(
            sig_df[[active_sig_key_col, col]].rename(columns={active_sig_key_col: "key"}),
            "key",
            col,
            "(run,trial)",
        )
        extra_meta[f"fmri_{col}"] = [
            float(mapped.get(key)) if key is not None and key in mapped.index else np.nan
            for key in active_keys
        ]
    for col in ("scoring_mask_sha256",):
        if col not in sig_df.columns:
            continue
        mapped = _unique_values_by_key(
            sig_df[[active_sig_key_col, col]].rename(columns={active_sig_key_col: "key"}),
            "key",
            col,
            "(run,trial)",
        )
        extra_meta[f"fmri_{col}"] = _metadata_values_for_keys(active_keys, mapped)

    y_label = f"fmri_signature.{method}.{contrast}.{sig_name}.{metric}"
    logger.info(
        "Loaded fMRI signature target for %s: %s (norm=%s, mode=%s, matches=%d/%d, onset_matches=%d, trial_matches=%d)",
        subject_bids,
        y_label,
        norm,
        "trial",
        int(np.isfinite(y.to_numpy(dtype=float)).sum()),
        len(y),
        int(onset_matches),
        int(trial_matches),
    )
    return y, y_label, extra_meta


__all__ = [
    "find_run_column",
    "load_fmri_signature_target_for_subject",
    "parse_run_label_to_int",
]
