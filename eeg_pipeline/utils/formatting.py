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


__all__ = [
    "sanitize_label",
    "format_baseline_window_string",
    "format_channel_list_for_display",
    "format_roi_description",
]
