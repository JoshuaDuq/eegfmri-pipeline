"""Shared path and spec helpers for fMRI multivariate signature resources."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _get_config_value(config: Any, key: str) -> Any:
    if config is None:
        return None
    if hasattr(config, "get"):
        try:
            value = config.get(key)
            if value is not None:
                return value
        except Exception:
            return None
    if isinstance(config, dict):
        parts = key.split(".")
        current = config
        for part in parts:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current
    return None


def discover_signature_root(config: Any, deriv_root: Any) -> Optional[Path]:
    """Resolve the root directory for signature weight maps from config."""
    cfg_path = _get_config_value(config, "paths.signature_dir")
    if cfg_path:
        candidate = Path(str(cfg_path)).expanduser()
        return candidate if candidate.exists() else None

    try:
        candidate = Path(deriv_root).expanduser().resolve().parent / "external"
        return candidate if candidate.exists() else None
    except Exception:
        return None


def get_signature_specs(config: Any) -> List[Dict[str, str]]:
    """
    Read the signature map specifications from config.

    Expected config key: ``paths.signature_maps`` — list of dicts with ``name`` and ``path`` keys.
    Returns an empty list if not configured.
    """
    specs = _get_config_value(config, "paths.signature_maps")
    if not isinstance(specs, list):
        return []
    validated: List[Dict[str, str]] = []
    for item in specs:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        path = str(item.get("path", "")).strip()
        if name and path:
            validated.append({"name": name, "path": path})
    return validated


def discover_signature_root_and_specs(
    config: Any,
    deriv_root: Any,
) -> Tuple[Optional[Path], List[Dict[str, str]]]:
    """Convenience wrapper returning (root, specs) from config."""
    root = discover_signature_root(config, deriv_root)
    specs = get_signature_specs(config)
    return root, specs
