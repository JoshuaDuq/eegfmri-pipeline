"""
Event analysis utilities.

Functions for extracting specific event types and masks from event DataFrames.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd
from eeg_pipeline.utils.config.loader import get_config_value


def _resolve_comparison_column(config: Any) -> Optional[str]:
    """Resolve the comparison column name from config."""
    column = get_config_value(config, "plotting.comparisons.comparison_column", None)
    if column is None:
        return None
    column_str = str(column).strip()
    return column_str if column_str else None


def _resolve_comparison_values(config: Any) -> tuple[Any, Any]:
    """Resolve the two comparison values from config."""
    values_spec = get_config_value(config, "plotting.comparisons.comparison_values", [])
    if isinstance(values_spec, (list, tuple)) and len(values_spec) >= 2:
        return values_spec[0], values_spec[1]
    return 0, 1


def _resolve_comparison_labels(config: Any, *, v1: Any, v2: Any) -> tuple[str, str]:
    """Return display labels for a 2-level comparison (value1/value2)."""
    labels_spec = get_config_value(config, "plotting.comparisons.comparison_labels", None)
    if isinstance(labels_spec, (list, tuple)) and len(labels_spec) >= 2:
        label1 = str(labels_spec[0]).strip()
        label2 = str(labels_spec[1]).strip()
        if label1 and label2:
            return label1, label2
    return str(v1), str(v2)


def _is_comparison_enabled(config: Any) -> bool:
    """Check if comparison is enabled in config."""
    return bool(get_config_value(config, "plotting.comparisons.compare_columns", False))


def _validate_comparison_prerequisites(
    events_df: pd.DataFrame,
    config: Any,
    *,
    require_enabled: bool = True,
) -> Optional[str]:
    """Validate prerequisites for comparison and return column name if valid."""
    if require_enabled and not _is_comparison_enabled(config):
        return None

    column = _resolve_comparison_column(config)
    
    # Debug: Log config resolution
    import logging
    _debug_logger = logging.getLogger(__name__)
    raw_value = get_config_value(config, "plotting.comparisons.comparison_column", "<<NOT SET>>")
    _debug_logger.warning(f"[DEBUG] comparison_column raw value: {raw_value!r}, resolved: {column!r}")
    _debug_logger.warning(f"[DEBUG] config type: {type(config).__name__}")
    
    if column is None or column not in events_df.columns:
        if column is not None:
            _debug_logger.warning(f"[DEBUG] Column '{column}' not found in events_df columns: {list(events_df.columns)[:5]}...")
        return None

    return column


def _create_numeric_masks(
    events_df: pd.DataFrame,
    column: str,
    value1: Any,
    value2: Any,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Create masks using numeric matching. Returns None if numeric matching fails."""
    try:
        column_values = pd.to_numeric(events_df[column], errors='coerce')
        value1_numeric = pd.to_numeric(str(value1), errors='coerce')
        value2_numeric = pd.to_numeric(str(value2), errors='coerce')

        if np.isnan(value1_numeric) or np.isnan(value2_numeric):
            return None

        mask1 = (column_values == value1_numeric).values
        mask2 = (column_values == value2_numeric).values
        if np.any(mask1) or np.any(mask2):
            return mask1, mask2

    except (ValueError, TypeError):
        pass

    return None


def _create_string_masks(
    events_df: pd.DataFrame,
    column: str,
    value1: Any,
    value2: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Create masks using string matching."""
    column_values_str = events_df[column].astype(str)
    value1_str = str(value1)
    value2_str = str(value2)

    mask1 = (column_values_str == value1_str).values
    mask2 = (column_values_str == value2_str).values
    return mask1, mask2


def extract_comparison_mask(
    events_df: pd.DataFrame,
    config: Any,
    *,
    require_enabled: bool = True,
) -> Optional[tuple[np.ndarray, np.ndarray, str, str]]:
    """Extract dual masks for flexible comparison based on config.

    Args:
        events_df: DataFrame containing event data.
        config: Configuration object.
        require_enabled: If True, return None when comparison is disabled.

    Returns:
        Tuple of (mask1, mask2, label1, label2) or None if comparison cannot be extracted.
    """
    column = _validate_comparison_prerequisites(events_df, config, require_enabled=require_enabled)
    if column is None:
        return None

    value1, value2 = _resolve_comparison_values(config)
    label1, label2 = _resolve_comparison_labels(config, v1=value1, v2=value2)

    numeric_masks = _create_numeric_masks(events_df, column, value1, value2)
    if numeric_masks is not None:
        mask1, mask2 = numeric_masks
        return mask1, mask2, label1, label2

    mask1, mask2 = _create_string_masks(events_df, column, value1, value2)
    has_any_matches = np.any(mask1) or np.any(mask2)
    if not has_any_matches:
        return None

    return mask1, mask2, label1, label2


def resolve_comparison_spec(
    events_df: pd.DataFrame,
    config: Any,
    *,
    require_enabled: bool = True,
) -> Optional[tuple[str, Any, Any, str, str]]:
    """Resolve (column, value1, value2, label1, label2) for a configured comparison.

    Args:
        events_df: DataFrame containing event data.
        config: Configuration object.
        require_enabled: If True, return None when comparison is disabled.

    Returns:
        Tuple of (column, value1, value2, label1, label2) or None if comparison cannot be resolved.
    """
    column = _validate_comparison_prerequisites(events_df, config, require_enabled=require_enabled)
    if column is None:
        return None

    value1, value2 = _resolve_comparison_values(config)
    label1, label2 = _resolve_comparison_labels(config, v1=value1, v2=value2)
    return column, value1, value2, label1, label2


def extract_multi_group_masks(
    events_df: pd.DataFrame,
    config: Any,
    *,
    require_enabled: bool = True,
) -> Optional[tuple[dict[str, np.ndarray], list[str]]]:
    """Extract masks for multi-group comparison (3+ groups).

    Args:
        events_df: DataFrame containing event data.
        config: Configuration object.
        require_enabled: If True, return None when comparison is disabled.

    Returns:
        Tuple of (masks_dict, group_labels) where masks_dict maps group_label -> boolean mask,
        or None if multi-group comparison cannot be extracted.
    """
    column = _validate_comparison_prerequisites(events_df, config, require_enabled=require_enabled)
    if column is None:
        return None

    values_spec = get_config_value(config, "plotting.comparisons.comparison_values", [])
    labels_spec = get_config_value(config, "plotting.comparisons.comparison_labels", None)
    
    if not isinstance(values_spec, (list, tuple)) or len(values_spec) < 2:
        return None
    
    if isinstance(labels_spec, (list, tuple)) and len(labels_spec) >= len(values_spec):
        labels = [str(l).strip() for l in labels_spec[:len(values_spec)]]
    else:
        labels = [str(v) for v in values_spec]
    
    masks_dict = {}
    column_values = events_df[column]
    
    for val, label in zip(values_spec, labels):
        try:
            numeric_val = float(val)
            mask = (pd.to_numeric(column_values, errors="coerce") == numeric_val).values
        except (ValueError, TypeError):
            val_str = str(val).strip().lower()
            mask = (column_values.astype(str).str.strip().str.lower() == val_str).values
        
        if np.any(mask):
            masks_dict[label] = mask
    
    if len(masks_dict) < 2:
        return None
    
    return masks_dict, labels
