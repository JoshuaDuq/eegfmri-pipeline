"""String formatting utilities for labels, bands, and display text."""

from __future__ import annotations

from typing import Optional, List, Tuple

DEFAULT_MAX_CHANNELS_DISPLAY = 10


def sanitize_label(label: str) -> str:
    """Sanitize label string by replacing non-alphanumeric characters."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(label))


def format_baseline_window_string(baseline_used: Tuple[float, float]) -> str:
    """Format baseline window tuple as filename-safe string."""
    b_start, b_end = baseline_used
    return f"bl{abs(b_start):.1f}to{abs(b_end):.2f}"


def format_channel_list_for_display(
    channels: List[str], max_channels: Optional[int] = None
) -> str:
    """Format channel list for display, limiting to max_channels."""
    if max_channels is None:
        max_channels = DEFAULT_MAX_CHANNELS_DISPLAY
    displayed = channels[:max_channels]
    return "Channels: " + ", ".join(displayed)


def format_roi_description(roi_channels: Optional[List[str]]) -> str:
    """Format ROI description string."""
    if not roi_channels:
        return "Overall"
    if len(roi_channels) == 1:
        return f"Channel: {roi_channels[0]}"
    return f"ROI: {len(roi_channels)} channels"


def get_residual_labels(method_code: str, target_type: str) -> Tuple[str, str]:
    """Get axis labels for residual plots."""
    ranked_suffix = " (ranked)" if method_code == "spearman" else ""
    x_label = f"Partial residuals{ranked_suffix} of log10(power/baseline)"

    target_labels = {
        "rating": f"Partial residuals{ranked_suffix} of rating",
        "temperature": f"Partial residuals{ranked_suffix} of temperature (°C)",
    }
    y_label = target_labels.get(
        target_type, f"Partial residuals{ranked_suffix} of {target_type}"
    )

    return x_label, y_label


def get_target_labels(target_type: str) -> Tuple[str, str]:
    """Get axis labels for target plots."""
    target_labels = {
        "rating": "Rating",
        "temperature": "Temperature (°C)",
    }
    y_label = target_labels.get(target_type, target_type)
    return "log10(power/baseline [-5–0 s])", y_label


def format_time_suffix(time_label: Optional[str]) -> str:
    """Format time label as a title suffix."""
    if time_label is None:
        return ""
    return f" ({time_label})"


def get_temporal_xlabel(time_label: str) -> str:
    """Get x-axis label for temporal power plots."""
    return f"log10(power/baseline) — {time_label}"


__all__ = [
    "sanitize_label",
    "format_baseline_window_string",
    "format_channel_list_for_display",
    "format_roi_description",
    "get_residual_labels",
    "get_target_labels",
    "format_time_suffix",
    "get_temporal_xlabel",
]
