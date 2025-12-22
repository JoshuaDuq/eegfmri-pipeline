from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema, parse_legacy_power_feature_name
from eeg_pipeline.infra.tsv import read_tsv


###################################################################
# Column Selection Utilities
###################################################################

_DEFAULT_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def infer_power_band(column_name: str, *, bands: Optional[List[str]] = None) -> Optional[str]:
    name = str(column_name).lower()
    if name.startswith("qc_"):
        return None

    bands_use = [str(b).lower() for b in (bands or _DEFAULT_BANDS)]
    bands_set = set(bands_use)

    parsed = NamingSchema.parse(str(column_name))
    if parsed.get("valid") and parsed.get("group") == "power":
        band = str(parsed.get("band") or "").lower()
        return band if band in bands_set else None

    legacy = parse_legacy_power_feature_name(str(column_name))
    if legacy is None:
        return None
    legacy_band, _legacy_ch = legacy
    legacy_band = str(legacy_band).lower()
    return legacy_band if legacy_band in bands_set else None


def get_power_columns_by_band(df: pd.DataFrame, *, bands: Optional[List[str]] = None) -> Dict[str, List[str]]:
    bands = bands or _DEFAULT_BANDS
    power_cols: Dict[str, List[str]] = {}

    for band in bands:
        band_cols: List[str] = []
        for c in df.columns:
            parsed = NamingSchema.parse(str(c))
            if parsed.get("valid") and parsed.get("group") == "power":
                if str(parsed.get("band") or "").lower() == str(band).lower():
                    band_cols.append(str(c))
                continue

            legacy = parse_legacy_power_feature_name(str(c))
            if legacy is None:
                continue
            legacy_band, _legacy_ch = legacy
            if str(legacy_band).lower() == str(band).lower():
                band_cols.append(str(c))
        if band_cols:
            power_cols[band] = band_cols

    return power_cols


def get_connectivity_columns_by_band(df: pd.DataFrame, *, bands: Optional[List[str]] = None) -> Dict[str, List[str]]:
    bands = bands or _DEFAULT_BANDS
    conn_cols: Dict[str, List[str]] = {}

    bands_set = {str(b).lower() for b in bands}
    for band in bands:
        band_l = str(band).lower()
        if band_l not in bands_set:
            continue
        band_cols: List[str] = []
        for c in df.columns:
            parsed = NamingSchema.parse(str(c))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "connectivity":
                continue
            if str(parsed.get("band") or "").lower() != band_l:
                continue
            band_cols.append(str(c))
        if band_cols:
            conn_cols[band] = band_cols

    return conn_cols


def get_itpc_columns_by_band(df: pd.DataFrame, *, bands: Optional[List[str]] = None) -> Dict[str, List[str]]:
    bands = bands or _DEFAULT_BANDS
    itpc_cols: Dict[str, List[str]] = {}

    bands_set = {str(b).lower() for b in bands}
    for band in bands:
        band_l = str(band).lower()
        if band_l not in bands_set:
            continue
        band_cols: List[str] = []
        for c in df.columns:
            parsed = NamingSchema.parse(str(c))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "itpc":
                continue
            if str(parsed.get("band") or "").lower() != band_l:
                continue
            band_cols.append(str(c))
        if band_cols:
            itpc_cols[band] = band_cols

    return itpc_cols


def get_aperiodic_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    aper_cols: Dict[str, List[str]] = {}

    for metric in ["slope", "offset", "exponent"]:
        cols: List[str] = []
        for c in df.columns:
            parsed = NamingSchema.parse(str(c))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "aperiodic":
                continue
            stat = str(parsed.get("stat") or "")
            if stat == metric or stat.endswith(f"_{metric}"):
                cols.append(str(c))
        if cols:
            aper_cols[metric] = list(dict.fromkeys(cols))

    return aper_cols




###################################################################
# Block Registration and Validation
###################################################################

def validate_trial_alignment_manifest(
    aligned_events: pd.DataFrame,
    features_dir: Path,
    logger: logging.Logger,
) -> None:
    import json
    
    manifest_path = features_dir / "trial_alignment.json"
    
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Trial alignment manifest not found: {manifest_path}. "
            f"Re-run feature extraction to generate aligned trial manifests."
        )
    actual_path = manifest_path

    with open(actual_path, "r") as f:
        manifest = json.load(f)
    
    manifest_n_epochs = manifest.get("n_epochs", 0)
    if manifest_n_epochs != len(aligned_events):
        raise ValueError(
            f"Trial count mismatch: manifest has {manifest_n_epochs} trials, "
            f"aligned_events has {len(aligned_events)} trials"
        )
    logger.info(f"Trial alignment validated: {manifest_n_epochs} trials")


def register_feature_block(
    name: str,
    block: Optional[Union[pd.DataFrame, pd.Series]],
    registry: Dict[str, pd.DataFrame],
    lengths: Dict[str, int],
) -> None:
    if block is None:
        lengths[name] = 0
        return

    if isinstance(block, pd.Series):
        block_df = block.to_frame()
        block_length = len(block)
    else:
        block_df = block
        block_length = len(block_df) if not block_df.empty else 0

    lengths[name] = block_length
    if block_length > 0:
        registry[name] = block_df.reset_index(drop=True)


def validate_feature_block_lengths(
    lengths: Dict[str, int],
    logger: logging.Logger,
    critical_features: Optional[List[str]] = None,
) -> None:
    if critical_features is None:
        critical_features = ["power", "baseline", "target"]

    unique_lengths = set(lengths.values())
    nonzero_lengths = {length for length in unique_lengths if length > 0}

    if len(nonzero_lengths) > 1:
        mismatch = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(
            "Feature blocks have mismatched trial counts and cannot be safely aligned: "
            f"{mismatch}. Inspect the per-feature drop logs (e.g., features/dropped_trials.tsv) "
            "to ensure each extractor operates on the same trial manifest."
        )

    empty_blocks = [name for name, length in lengths.items() if length == 0]
    nonempty_blocks = [name for name, length in lengths.items() if length > 0]

    if not empty_blocks or not nonempty_blocks:
        return

    empty_critical = [name for name in empty_blocks if name in critical_features]

    if empty_critical:
        error_msg = (
            f"Critical feature blocks are empty while others are not: empty={empty_blocks}, "
            f"non-empty={nonempty_blocks}. Critical empty blocks: {empty_critical}. "
            "This indicates extraction failures and prevents valid analysis. "
            "Fix feature extraction before proceeding."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.warning(
        f"Some non-critical feature blocks are empty while others are not: empty={empty_blocks}, "
        f"non-empty={nonempty_blocks}. Analysis will proceed but may be incomplete."
    )


###################################################################
# Feature Alignment
###################################################################

def align_feature_dataframes(
    pow_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    conn_df: Optional[pd.DataFrame],
    aper_df: Optional[pd.DataFrame],
    y: pd.Series,
    aligned_events: pd.DataFrame,
    features_dir: Path,
    logger: logging.Logger,
    config,
    initial_trial_count: Optional[int] = None,
    critical_features: Optional[List[str]] = None,
    extra_blocks: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    pd.Series,
    Dict[str, Any],
]:
    """Align feature blocks and target vector while preserving valid trials."""

    def _block_length(block):
        if block is None or getattr(block, "empty", False):
            return None
        return len(block)

    n_trials = next(
        (
            l
            for l in [
                _block_length(pow_df),
                _block_length(baseline_df),
                _block_length(conn_df),
                _block_length(aper_df),
                len(y) if y is not None else None,
            ]
            if l is not None
        ),
        None,
    )

    drop_mask: Optional[np.ndarray] = None
    if n_trials is not None:
        combined_mask = np.ones(n_trials, dtype=bool)

        if pow_df is not None and not getattr(pow_df, "empty", False):
            min_valid_band_fraction = float(
                config.get("feature_engineering.features.min_valid_band_fraction", 0.0)
            )

            band_names = sorted({b for b in (infer_power_band(c) for c in pow_df.columns) if b})
            band_drop_counts = {}
            for band in band_names:
                band_cols = [c for c in pow_df.columns if infer_power_band(c) == band]
                if not band_cols:
                    continue
                values = pow_df[band_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                valid_fraction = np.isfinite(values).mean(axis=1)
                band_invalid = valid_fraction <= min_valid_band_fraction
                band_drop_counts[band] = int(band_invalid.sum())
                if np.any(band_invalid):
                    pow_df.loc[band_invalid, band_cols] = np.nan
                    qc_col = f"qc_power_{band}"
                    if qc_col not in pow_df.columns:
                        pow_df[qc_col] = True
                    pow_df.loc[band_invalid, qc_col] = False
            for band, n_bad in band_drop_counts.items():
                if n_bad > 0:
                    logger.warning(
                        "Nulling %d trials with insufficient valid power samples for band %s (threshold=%.2f); trials retained for other bands.",
                        n_bad,
                        band,
                        min_valid_band_fraction,
                    )

        def _finite_mask(block):
            if block is None or getattr(block, "empty", False):
                return np.ones(n_trials, dtype=bool)
            if isinstance(block, pd.DataFrame):
                try:
                    data = block.to_numpy(dtype=float, copy=False)
                except (TypeError, ValueError):
                    data = block.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            else:
                data = pd.to_numeric(block, errors="coerce").to_numpy(dtype=float)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return np.isfinite(data).any(axis=1)

        combined_mask &= _finite_mask(pow_df)
        combined_mask &= _finite_mask(baseline_df)
        combined_mask &= _finite_mask(aper_df)
        combined_mask &= _finite_mask(y)

        if not np.all(combined_mask):
            drop_mask = combined_mask

    if drop_mask is not None:
        n_mask = len(drop_mask)

        def _safe_mask_apply(df, name: str):
            if df is None or getattr(df, "empty", False):
                return df
            if len(df) != n_mask:
                logger.warning(
                    "Length mismatch for %s: DataFrame has %d rows but mask has %d. Skipping mask application to avoid misalignment.",
                    name,
                    len(df),
                    n_mask,
                )
                return df
            return df.loc[drop_mask].reset_index(drop=True)

        pow_df = _safe_mask_apply(pow_df, "power")
        baseline_df = _safe_mask_apply(baseline_df, "baseline")
        conn_df = _safe_mask_apply(conn_df, "connectivity")
        aper_df = _safe_mask_apply(aper_df, "aperiodic")
        if y is not None and len(y) == n_mask:
            y = y.loc[drop_mask].reset_index(drop=True)
        elif y is not None:
            logger.warning(
                "Length mismatch for target: Series has %d rows but mask has %d. Skipping mask application to avoid misalignment.",
                len(y),
                n_mask,
            )

    block_registry: Dict[str, pd.DataFrame] = {}
    before_lengths: Dict[str, int] = {}

    def _is_valid_df(df):
        return df if df is not None and (not hasattr(df, "empty") or not df.empty) else None

    register_feature_block("power", _is_valid_df(pow_df), block_registry, before_lengths)
    register_feature_block("baseline", _is_valid_df(baseline_df), block_registry, before_lengths)
    register_feature_block("connectivity", _is_valid_df(conn_df), block_registry, before_lengths)
    register_feature_block("aperiodic", _is_valid_df(aper_df), block_registry, before_lengths)
    register_feature_block("target", y, block_registry, before_lengths)

    if extra_blocks:
        for block_name, block_df in extra_blocks.items():
            register_feature_block(block_name, _is_valid_df(block_df), block_registry, before_lengths)

    if not block_registry:
        logger.warning("No features extracted; skipping save")
        return None, None, None, None, None, None, None

    validate_feature_block_lengths(before_lengths, logger, critical_features=critical_features)

    if aligned_events is not None and len(aligned_events) > 0:
        validate_trial_alignment_manifest(aligned_events, features_dir, logger)

    logger.info(
        "Validated feature block lengths (%s)",
        ", ".join(f"{k}={v}" for k, v in before_lengths.items()),
    )

    pow_df_aligned = block_registry.get("power")
    if pow_df_aligned is not None:
        pow_df_aligned = pow_df_aligned.reset_index(drop=True)

    baseline_df_aligned = block_registry.get("baseline")
    if baseline_df_aligned is not None:
        baseline_df_aligned = baseline_df_aligned.reset_index(drop=True)
    else:
        baseline_df_aligned = baseline_df if baseline_df is not None else pd.DataFrame()

    conn_df_aligned = block_registry.get("connectivity")
    if conn_df_aligned is not None:
        conn_df_aligned = conn_df_aligned.reset_index(drop=True)

    aper_df_aligned = block_registry.get("aperiodic")
    if aper_df_aligned is not None:
        aper_df_aligned = aper_df_aligned.reset_index(drop=True)

    target_block = block_registry.get("target")
    if target_block is not None:
        y_aligned = target_block.iloc[:, 0] if target_block.shape[1] == 1 else target_block
    else:
        y_aligned = y

    def _get_length(df):
        if df is None:
            return 0
        if hasattr(df, "empty") and df.empty:
            return 0
        return len(df) if hasattr(df, "__len__") else 0

    after_lengths = {
        "power": _get_length(pow_df_aligned),
        "baseline": _get_length(baseline_df_aligned),
        "connectivity": _get_length(conn_df_aligned),
        "aperiodic": _get_length(aper_df_aligned),
        "target": _get_length(y_aligned),
    }

    extra_aligned: Dict[str, Optional[pd.DataFrame]] = {}
    if extra_blocks:
        for block_name in extra_blocks.keys():
            block = block_registry.get(block_name)
            if block is not None:
                block = block.reset_index(drop=True)
            extra_aligned[block_name] = block
            after_lengths[block_name] = _get_length(block)

    retention_stats = {
        "initial_trial_count": initial_trial_count,
        "before_alignment": before_lengths,
        "after_alignment": after_lengths,
        "mask": drop_mask,
        "extra_aligned": extra_aligned,
    }

    return (
        pow_df_aligned,
        baseline_df_aligned,
        conn_df_aligned,
        aper_df_aligned,
        y_aligned,
        retention_stats,
    )


__all__ = [
    # Column selection
    "infer_power_band",
    "get_power_columns_by_band",
    "get_connectivity_columns_by_band",
    "get_itpc_columns_by_band",
    "get_aperiodic_columns",
    # Block registration
    "validate_trial_alignment_manifest",
    "register_feature_block",
    "validate_feature_block_lengths",
    # Alignment
    "align_feature_dataframes",
]
