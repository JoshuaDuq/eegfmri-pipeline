"""Shared parsing helpers for reusable string argument formats."""

from __future__ import annotations

from typing import Optional, Sequence


def parse_group_arg(group: str) -> Optional[list[str]]:
    """Parse subject group strings into subject-id lists.

    Accepted "all" aliases return ``None`` to indicate no explicit filter.
    """
    normalized = (group or "").strip()
    if normalized.lower() in {"all", "*", "@all"}:
        return None
    normalized = normalized.replace(";", ",").replace(" ", ",")
    values = [subject.strip() for subject in normalized.split(",") if subject.strip()]
    return values or None


def parse_frequency_band_definitions(band_defs: Sequence[str]) -> dict[str, list[float]]:
    """Parse frequency bands from CLI format ``name:low:high``."""
    bands: dict[str, list[float]] = {}
    for band_def in band_defs:
        parts = str(band_def).split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid frequency band definition '{band_def}'; expected 'name:low:high'"
            )
        name = parts[0].strip().lower()
        try:
            low = float(parts[1].strip())
            high = float(parts[2].strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid frequency values in '{band_def}'; expected numeric low:high"
            ) from exc
        if low >= high:
            raise ValueError(
                f"Invalid frequency range in '{band_def}'; low must be < high"
            )
        bands[name] = [low, high]
    return bands


def parse_roi_definitions(roi_defs: Sequence[str]) -> dict[str, list[str]]:
    """Parse ROI definitions from CLI format ``name:ch1,ch2,...``."""
    rois: dict[str, list[str]] = {}
    for roi_def in roi_defs:
        text = str(roi_def)
        if ":" not in text:
            raise ValueError(
                f"Invalid ROI definition '{roi_def}'; expected 'name:ch1,ch2,...'"
            )
        name, channels_str = text.split(":", 1)
        name = name.strip()
        channels = [channel.strip() for channel in channels_str.split(",") if channel.strip()]
        if not channels:
            raise ValueError(f"Invalid ROI definition '{roi_def}'; no channels specified")
        rois[name] = [f"^({'|'.join(channels)})$"]
    return rois

