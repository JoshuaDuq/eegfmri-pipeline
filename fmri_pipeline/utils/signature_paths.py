"""Shared path helpers for fMRI signature resources."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def _get_signature_dir_config_value(config: Any) -> Any:
    if config is None:
        return None
    if hasattr(config, "get"):
        value = config.get("paths.signature_dir")
        if value is not None:
            return value
    if isinstance(config, dict):
        paths = config.get("paths")
        if isinstance(paths, dict):
            return paths.get("signature_dir")
    return None


def discover_signature_root(config: Any, deriv_root: Any) -> Optional[Path]:
    """Best-effort discovery for signature weight-map root directory."""
    cfg_path = _get_signature_dir_config_value(config)
    if cfg_path:
        candidate = Path(str(cfg_path)).expanduser()
        return candidate if candidate.exists() else None

    try:
        candidate = Path(deriv_root).expanduser().resolve().parent / "external"
        return candidate if candidate.exists() else None
    except Exception:
        return None
