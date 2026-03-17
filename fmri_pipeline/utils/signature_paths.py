"""Shared path and spec helpers for fMRI multivariate signature resources."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _get_config_value(config: Any, key: str) -> Any:
    if config is None:
        return None
    if hasattr(config, "get"):
        value = config.get(key)
        if value is not None:
            return value
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
        if not candidate.exists():
            raise FileNotFoundError(f"Configured paths.signature_dir does not exist: {candidate}")
        return candidate

    candidate = Path(deriv_root).expanduser().resolve().parent / "external"
    return candidate if candidate.exists() else None


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
    seen_names: set[str] = set()
    for item in specs:
        if not isinstance(item, dict):
            raise ValueError("Each entry in paths.signature_maps must be a mapping with 'name' and 'path'.")
        name = str(item.get("name", "")).strip()
        path = str(item.get("path", "")).strip()
        if not name or not path:
            raise ValueError("Each entry in paths.signature_maps must define non-empty 'name' and 'path'.")
        if name in seen_names:
            raise ValueError(f"Duplicate signature name in paths.signature_maps: {name!r}")
        seen_names.add(name)
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
