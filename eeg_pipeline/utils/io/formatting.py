"""
String formatting utilities for labels, bands, and display text.

This module provides functions for formatting various strings used
throughout the pipeline for display and file naming.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
import pandas as pd

try:
    from ..config.loader import load_settings
except ImportError:
    load_settings = None

from .paths import ensure_dir


def sanitize_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(label))


def format_baseline_window_string(baseline_used: Tuple[float, float]) -> str:
    b_start, b_end = baseline_used
    return f"bl{abs(b_start):.1f}to{abs(b_end):.2f}"


def format_baseline_string(baseline_window: Tuple[float, float]) -> str:
    return f"[{baseline_window[0]:.2f}, {baseline_window[1]:.2f}]"


def parse_analysis_type_from_filename(filename: str) -> str:
    if filename.startswith("corr_stats_pow_roi"):
        return "pow_roi"
    if filename.startswith("corr_stats_conn_roi_summary"):
        return "conn_roi_summary"
    if filename.startswith("corr_stats_edges"):
        return "conn_edges"
    return "other"


def parse_target_from_filename(filename: str) -> str:
    if "_vs_" not in filename:
        return ""
    return filename.split("_vs_", 1)[1].split(".", 1)[0]


def parse_measure_band_from_filename(analysis_type: str, filename: str) -> str:
    prefixes = {
        "conn_edges": "corr_stats_edges_",
        "conn_roi_summary": "corr_stats_conn_roi_summary_"
    }
    prefix = prefixes.get(analysis_type)
    if prefix and filename.startswith(prefix):
        return filename[len(prefix):].split("_vs_", 1)[0]
    return ""


def format_band_range(band: str, freq_bands: Dict[str, List[float]]) -> str:
    if not band or not freq_bands:
        return ""
    
    band_rng = freq_bands.get(band)
    if not band_rng or len(band_rng) < 2:
        return ""
    
    band_range_tuple = tuple(band_rng)
    return f"{band_range_tuple[0]:g}–{band_range_tuple[1]:g} Hz"


def build_partial_covars_string(covariates_df: Optional[pd.DataFrame]) -> str:
    if covariates_df is None or covariates_df.empty:
        return ""
    return ",".join(covariates_df.columns.tolist())


def format_band_label(band: str, config) -> str:
    if not band:
        return ""
    
    freq_bands = config.get("time_frequency_analysis.bands", {})
    band_range = freq_bands.get(band)
    if band_range:
        band_range = tuple(band_range)
        return f"{band} ({band_range[0]:g}\u2013{band_range[1]:g} Hz)"
    return band


def get_correlation_type_labels(correlation_type: str) -> Tuple[str, str]:
    labels = {
        "rating": ("Behavior", "behavior"),
        "temperature": ("Temperature", "temperature")
    }
    return labels.get(correlation_type, ("Temperature", "temperature"))


def format_temperature_label(val: Union[float, str]) -> str:
    try:
        return f"{float(val):.1f}".replace(".", "p")
    except (ValueError, TypeError):
        return sanitize_label(str(val))


def extract_subject_id_from_path(path: Path) -> Optional[str]:
    import re
    path_str = str(path)
    match = re.search(r'sub-(\d+)', path_str)
    return match.group(1) if match else None


def write_group_trial_counts(
    subjects: List[str],
    output_dir: Path,
    counts_file_name: str,
    pain_counts: Optional[List[Tuple[int, int]]] = None,
    logger: Optional[Any] = None,
) -> None:
    if pain_counts is None:
        pain_counts = [(0, 0)] * len(subjects)
    
    if len(subjects) != len(pain_counts):
        raise ValueError(f"Length mismatch: {len(subjects)} subjects but {len(pain_counts)} count tuples")
    
    rows = []
    for subject, (n_pain, n_nonpain) in zip(subjects, pain_counts):
        rows.append({
            "subject": subject,
            "n_pain": n_pain,
            "n_nonpain": n_nonpain,
            "n_total": n_pain + n_nonpain
        })
    
    if not rows:
        return
    
    counts_df = pd.DataFrame(rows)
    totals = counts_df[["n_pain", "n_nonpain", "n_total"]].sum()
    total_row = {
        "subject": "TOTAL",
        **{key: int(value) for key, value in totals.to_dict().items()}
    }
    counts_df = pd.concat([counts_df, pd.DataFrame([total_row])], ignore_index=True)
    
    ensure_dir(output_dir)
    output_path = output_dir / counts_file_name
    counts_df.to_csv(output_path, sep="\t", index=False)
    if logger:
        logger.info(f"Saved counts: {output_path}")


def format_channel_list_for_display(
    channels: List[str], 
    max_channels: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> str:
    if max_channels is None:
        if config is None and load_settings is not None:
            config = load_settings()
        max_channels = int(config.get("plotting.behavioral.max_channels_to_display", 10)) if config else 10
    
    displayed = channels[:max_channels]
    return "Channels: " + ", ".join(displayed)


def format_roi_description(roi_channels: Optional[List[str]]) -> str:
    if not roi_channels:
        return "Overall"
    if len(roi_channels) == 1:
        return f"Channel: {roi_channels[0]}"
    return f"ROI: {len(roi_channels)} channels"


def get_residual_labels(method_code: str, target_type: str) -> Tuple[str, str]:
    ranked_suffix = " (ranked)" if method_code == "spearman" else ""
    x_label = f"Partial residuals{ranked_suffix} of log10(power/baseline)"
    
    target_labels = {
        "rating": f"Partial residuals{ranked_suffix} of rating",
        "temperature": f"Partial residuals{ranked_suffix} of temperature (°C)"
    }
    y_label = target_labels.get(target_type, f"Partial residuals{ranked_suffix} of {target_type}")
    
    return x_label, y_label


def get_target_labels(target_type: str) -> Tuple[str, str]:
    target_labels = {
        "rating": "Rating",
        "temperature": "Temperature (°C)"
    }
    y_label = target_labels.get(target_type, target_type)
    return "log10(power/baseline [-5–0 s])", y_label


def get_temporal_xlabel(time_label: str) -> str:
    return f"log10(power/baseline) [{time_label} window]"


def format_time_suffix(time_label: Optional[str]) -> str:
    if time_label:
        return f" ({time_label})"
    return " (plateau)"


__all__ = [
    "sanitize_label",
    "format_baseline_window_string",
    "format_baseline_string",
    "parse_analysis_type_from_filename",
    "parse_target_from_filename",
    "parse_measure_band_from_filename",
    "format_band_range",
    "build_partial_covars_string",
    "format_band_label",
    "get_correlation_type_labels",
    "format_temperature_label",
    "extract_subject_id_from_path",
    "write_group_trial_counts",
    "format_channel_list_for_display",
    "format_roi_description",
    "get_residual_labels",
    "get_target_labels",
    "get_temporal_xlabel",
    "format_time_suffix",
]










