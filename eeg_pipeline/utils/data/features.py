from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema


###################################################################
# Constants
###################################################################

_DEFAULT_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
_DEFAULT_CRITICAL_FEATURES = ["power", "baseline", "target"]
_APERIODIC_METRICS = ["slope", "offset", "exponent"]
_FEATURE_GROUP_POWER = "power"
_FEATURE_GROUP_CONN = "conn"
_FEATURE_GROUP_ITPC = "itpc"
_FEATURE_GROUP_APERIODIC = "aperiodic"


def _copy_attrs(target: Union[pd.DataFrame, pd.Series], source: Any) -> Union[pd.DataFrame, pd.Series]:
    attrs = dict(getattr(source, "attrs", {}) or {})
    if attrs:
        target.attrs.update(attrs)
    return target


def infer_power_band(column_name: str, *, bands: Optional[List[str]] = None) -> Optional[str]:
    """Infer power band from column name using NamingSchema."""
    column_name_lower = str(column_name).lower()
    if column_name_lower.startswith("qc_"):
        return None

    bands_normalized = [str(band).lower() for band in (bands or _DEFAULT_BANDS)]
    valid_bands = set(bands_normalized)

    parsed = NamingSchema.parse(str(column_name))
    is_valid_power = parsed.get("valid") and parsed.get("group") == _FEATURE_GROUP_POWER
    if not is_valid_power:
        return None

    band = str(parsed.get("band") or "").lower()
    return band if band in valid_bands else None


def _get_columns_by_band_and_group(
    df: pd.DataFrame,
    group: str,
    *,
    bands: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Extract columns grouped by band for a specific feature group."""
    bands_to_check = bands or _DEFAULT_BANDS
    columns_by_band: Dict[str, List[str]] = {}

    for band in bands_to_check:
        band_normalized = str(band).lower()
        matching_columns: List[str] = []
        for column in df.columns:
            parsed = NamingSchema.parse(str(column))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != group:
                continue
            parsed_band = str(parsed.get("band") or "").lower()
            if parsed_band != band_normalized:
                continue
            matching_columns.append(str(column))

        if matching_columns:
            columns_by_band[band] = matching_columns

    return columns_by_band


def get_power_columns_by_band(df: pd.DataFrame, *, bands: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Get power feature columns grouped by frequency band."""
    return _get_columns_by_band_and_group(df, _FEATURE_GROUP_POWER, bands=bands)


def get_connectivity_columns_by_band(df: pd.DataFrame, *, bands: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Get connectivity feature columns grouped by frequency band."""
    return _get_columns_by_band_and_group(df, _FEATURE_GROUP_CONN, bands=bands)


def get_itpc_columns_by_band(df: pd.DataFrame, *, bands: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Get ITPC feature columns grouped by frequency band."""
    return _get_columns_by_band_and_group(df, _FEATURE_GROUP_ITPC, bands=bands)


def get_aperiodic_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Get aperiodic feature columns grouped by metric type."""
    columns_by_metric: Dict[str, List[str]] = {}

    for metric in _APERIODIC_METRICS:
        matching_columns: List[str] = []
        for column in df.columns:
            parsed = NamingSchema.parse(str(column))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != _FEATURE_GROUP_APERIODIC:
                continue

            stat = str(parsed.get("stat") or "")
            matches_metric = stat == metric or stat.endswith(f"_{metric}")
            if matches_metric:
                matching_columns.append(str(column))

        if matching_columns:
            columns_by_metric[metric] = matching_columns

    return columns_by_metric


###################################################################
# Block Registration and Validation
###################################################################

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
        aligned = block_df.reset_index(drop=True)
        registry[name] = _copy_attrs(aligned, block)


def validate_feature_block_lengths(
    lengths: Dict[str, int],
    logger: logging.Logger,
    critical_features: Optional[List[str]] = None,
    requested_categories: Optional[List[str]] = None,
) -> None:
    """Validate that feature blocks have consistent lengths."""
    if critical_features is None:
        critical_features = _DEFAULT_CRITICAL_FEATURES

    unique_lengths = set(lengths.values())
    nonzero_lengths = unique_lengths - {0}

    if len(nonzero_lengths) > 1:
        mismatch_pairs = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(
            "Feature blocks have mismatched trial counts and cannot be safely aligned: "
            f"{mismatch_pairs}. Inspect the per-feature drop logs (e.g., features/dropped_trials.tsv) "
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

    # Only warn about empty blocks if they were actually requested
    if requested_categories is None:
        return

    requested_categories_set = set(requested_categories)
    requested_empty = [
        name
        for name in empty_blocks
        if name != "target"
        and ("power" if name == "baseline" else name.split("_")[0]) in requested_categories_set
    ]

    if not requested_empty:
        return

    logger.warning(
        f"Some non-critical feature blocks are empty while others are not: empty={empty_blocks}, "
        f"non-empty={nonempty_blocks}. Analysis will proceed but may be incomplete."
    )


###################################################################
# Feature Alignment
###################################################################

def _get_block_length(block: Optional[Union[pd.DataFrame, pd.Series]]) -> Optional[int]:
    """Get length of a feature block, returning None if empty or invalid."""
    if block is None or getattr(block, "empty", False):
        return None
    return len(block)


def _validate_power_bands(
    pow_df: pd.DataFrame,
    config: Any,
    logger: logging.Logger,
) -> None:
    """Validate and nullify invalid power band data per trial."""
    min_valid_band_fraction = float(
        config.get("feature_engineering.features.min_valid_band_fraction", 0.0)
    )

    band_names = sorted({band for band in (infer_power_band(col) for col in pow_df.columns) if band})
    band_drop_counts: Dict[str, int] = {}

    for band in band_names:
        band_columns = [col for col in pow_df.columns if infer_power_band(col) == band]
        if not band_columns:
            continue

        values = pow_df[band_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid_fraction_per_trial = np.isfinite(values).mean(axis=1)
        invalid_trials = valid_fraction_per_trial <= min_valid_band_fraction
        band_drop_counts[band] = int(invalid_trials.sum())

        if np.any(invalid_trials):
            pow_df.loc[invalid_trials, band_columns] = np.nan
            qc_column = f"qc_power_{band}"
            if qc_column not in pow_df.columns:
                pow_df[qc_column] = True
            pow_df.loc[invalid_trials, qc_column] = False

    for band, n_invalid in band_drop_counts.items():
        if n_invalid > 0:
            logger.warning(
                "Nulling %d trials with insufficient valid power samples for band %s (threshold=%.2f); "
                "trials retained for other bands.",
                n_invalid,
                band,
                min_valid_band_fraction,
            )


def _create_finite_mask(
    block: Optional[Union[pd.DataFrame, pd.Series]],
    n_trials: int,
) -> np.ndarray:
    """Create boolean mask indicating which trials have finite values."""
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


def _apply_drop_mask(
    block: Optional[Union[pd.DataFrame, pd.Series]],
    drop_mask: np.ndarray,
    block_name: str,
    logger: logging.Logger,
) -> Optional[Union[pd.DataFrame, pd.Series]]:
    """Apply drop mask to a feature block with length validation."""
    if block is None or getattr(block, "empty", False):
        return block

    block_length = len(block)
    mask_length = len(drop_mask)

    if block_length != mask_length:
        logger.warning(
            "Length mismatch for %s: block has %d rows but mask has %d. "
            "Skipping mask application to avoid misalignment.",
            block_name,
            block_length,
            mask_length,
        )
        return block

    masked_block = block.loc[drop_mask].reset_index(drop=True)
    return _copy_attrs(masked_block, block)


def _register_all_blocks(
    pow_df: Optional[pd.DataFrame],
    baseline_df: Optional[pd.DataFrame],
    conn_df: Optional[pd.DataFrame],
    aper_df: Optional[pd.DataFrame],
    y: Optional[pd.Series],
    extra_blocks: Optional[Dict[str, pd.DataFrame]],
    registry: Dict[str, pd.DataFrame],
    lengths: Dict[str, int],
) -> None:
    """Register all feature blocks in the registry."""
    register_feature_block("power", pow_df, registry, lengths)
    register_feature_block("baseline", baseline_df, registry, lengths)
    register_feature_block("connectivity", conn_df, registry, lengths)
    register_feature_block("aperiodic", aper_df, registry, lengths)
    register_feature_block("target", y, registry, lengths)

    if extra_blocks:
        for block_name, block_df in extra_blocks.items():
            register_feature_block(block_name, block_df, registry, lengths)


def _extract_aligned_blocks(
    registry: Dict[str, pd.DataFrame],
    pow_df: Optional[pd.DataFrame],
    baseline_df: Optional[pd.DataFrame],
    y: Optional[pd.Series],
) -> Tuple[
    Optional[pd.DataFrame],
    pd.DataFrame,
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    pd.Series,
]:
    """Extract and align blocks from registry."""
    pow_df_aligned = registry.get("power")
    if pow_df_aligned is not None:
        pow_df_aligned = _copy_attrs(
            pow_df_aligned.reset_index(drop=True),
            pow_df_aligned,
        )

    baseline_df_aligned = registry.get("baseline")
    if baseline_df_aligned is not None:
        baseline_df_aligned = _copy_attrs(
            baseline_df_aligned.reset_index(drop=True),
            baseline_df_aligned,
        )
    else:
        baseline_df_aligned = pd.DataFrame()

    conn_df_aligned = registry.get("connectivity")
    if conn_df_aligned is not None:
        conn_df_aligned = _copy_attrs(
            conn_df_aligned.reset_index(drop=True),
            conn_df_aligned,
        )

    aper_df_aligned = registry.get("aperiodic")
    if aper_df_aligned is not None:
        aper_df_aligned = _copy_attrs(
            aper_df_aligned.reset_index(drop=True),
            aper_df_aligned,
        )

    target_block = registry.get("target")
    if target_block is not None:
        y_aligned = target_block.iloc[:, 0] if target_block.shape[1] == 1 else target_block
    else:
        y_aligned = pd.Series(dtype=float)

    return pow_df_aligned, baseline_df_aligned, conn_df_aligned, aper_df_aligned, y_aligned


def _compute_after_lengths(
    pow_df: Optional[pd.DataFrame],
    baseline_df: pd.DataFrame,
    conn_df: Optional[pd.DataFrame],
    aper_df: Optional[pd.DataFrame],
    y: pd.Series,
    extra_blocks: Optional[Dict[str, pd.DataFrame]],
    registry: Dict[str, pd.DataFrame],
) -> Dict[str, int]:
    """Compute lengths of aligned blocks."""
    after_lengths = {
        "power": len(pow_df) if pow_df is not None and not pow_df.empty else 0,
        "baseline": len(baseline_df) if not baseline_df.empty else 0,
        "connectivity": len(conn_df) if conn_df is not None and not conn_df.empty else 0,
        "aperiodic": len(aper_df) if aper_df is not None and not aper_df.empty else 0,
        "target": len(y) if not y.empty else 0,
    }

    if extra_blocks:
        for block_name in extra_blocks.keys():
            block = registry.get(block_name)
            after_lengths[block_name] = len(block) if block is not None and not block.empty else 0

    return after_lengths


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
    requested_categories: Optional[List[str]] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    pd.Series,
    Dict[str, Any],
]:
    """Align feature blocks and target vector while preserving valid trials."""
    block_lengths = [
        _get_block_length(pow_df),
        _get_block_length(baseline_df),
        _get_block_length(conn_df),
        _get_block_length(aper_df),
        len(y) if y is not None else None,
    ]
    n_trials = next((length for length in block_lengths if length is not None), None)

    drop_mask: Optional[np.ndarray] = None
    if n_trials is not None:
        combined_mask = np.ones(n_trials, dtype=bool)

        if pow_df is not None and not getattr(pow_df, "empty", False):
            _validate_power_bands(pow_df, config, logger)

        combined_mask &= _create_finite_mask(pow_df, n_trials)
        combined_mask &= _create_finite_mask(baseline_df, n_trials)
        combined_mask &= _create_finite_mask(conn_df, n_trials)
        combined_mask &= _create_finite_mask(aper_df, n_trials)
        if extra_blocks:
            for block_df in extra_blocks.values():
                combined_mask &= _create_finite_mask(block_df, n_trials)
        combined_mask &= _create_finite_mask(y, n_trials)

        drop_mask = combined_mask if not np.all(combined_mask) else None

    if drop_mask is not None:
        pow_df = _apply_drop_mask(pow_df, drop_mask, "power", logger)
        baseline_df = _apply_drop_mask(baseline_df, drop_mask, "baseline", logger)
        conn_df = _apply_drop_mask(conn_df, drop_mask, "connectivity", logger)
        aper_df = _apply_drop_mask(aper_df, drop_mask, "aperiodic", logger)
        y = _apply_drop_mask(y, drop_mask, "target", logger)
        if extra_blocks:
            masked_extra_blocks: Dict[str, Optional[pd.DataFrame]] = {}
            for block_name, block_df in extra_blocks.items():
                masked_block = _apply_drop_mask(
                    block_df, drop_mask, f"extra:{block_name}", logger
                )
                if isinstance(masked_block, pd.Series):
                    masked_block = masked_block.to_frame()
                masked_extra_blocks[block_name] = masked_block
            extra_blocks = masked_extra_blocks

    block_registry: Dict[str, pd.DataFrame] = {}
    before_lengths: Dict[str, int] = {}

    _register_all_blocks(
        pow_df, baseline_df, conn_df, aper_df, y, extra_blocks, block_registry, before_lengths
    )

    if not block_registry:
        logger.warning("No features extracted; skipping save")
        return None, None, None, None, None, {}

    validate_feature_block_lengths(
        before_lengths, logger, critical_features=critical_features, requested_categories=requested_categories
    )
    
    if aligned_events is not None and len(aligned_events) > 0:
        n_events = len(aligned_events)
        n_trials = next((length for length in block_lengths if length is not None), None)
        if n_trials is not None and n_events != n_trials:
            raise ValueError(
                f"Events/features length mismatch: aligned_events has {n_events} rows "
                f"but features have {n_trials} rows"
            )

    logger.info(
        "Validated feature block lengths (%s)",
        ", ".join(f"{k}={v}" for k, v in before_lengths.items()),
    )

    pow_df_aligned, baseline_df_aligned, conn_df_aligned, aper_df_aligned, y_aligned = (
        _extract_aligned_blocks(block_registry, pow_df, baseline_df, y)
    )

    after_lengths = _compute_after_lengths(
        pow_df_aligned, baseline_df_aligned, conn_df_aligned, aper_df_aligned, y_aligned,
        extra_blocks, block_registry
    )

    extra_aligned: Dict[str, Optional[pd.DataFrame]] = {}
    if extra_blocks:
        for block_name in extra_blocks.keys():
            block = block_registry.get(block_name)
            if block is not None:
                block = _copy_attrs(block.reset_index(drop=True), block)
            extra_aligned[block_name] = block

    retention_stats = {
        "initial_trial_count": initial_trial_count,
        "before_filtering": before_lengths,
        "after_filtering": after_lengths,
        "mask": drop_mask,
        "extra_blocks": extra_aligned,
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
    "register_feature_block",
    "validate_feature_block_lengths",
    # Alignment
    "align_feature_dataframes",
]
