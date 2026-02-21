"""Text helpers for fMRI pipeline modules."""

from __future__ import annotations


def safe_slug(value: str, *, default: str = "item") -> str:
    """Return a filesystem-safe slug preserving alnum, '-' and '_' characters."""
    raw = "" if value is None else str(value)
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or default
