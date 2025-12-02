from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, List
import yaml
import os

###################################################################
# Config Path Resolution
###################################################################


def _looks_like_path_string(value: str) -> bool:
    if value in {".", ".."}:
        return True
    if value.startswith("~"):
        return True
    if len(value) >= 2 and value[1] == ":":
        return True
    if any(sep in value for sep in ("/", "\\")):
        return True
    return False


def _resolve_single_path(value: str, config_dir: Path, project_root: Path) -> str:
    if not value or value.strip() == "":
        return value
    
    try:
        float(value)
        return value
    except ValueError:
        pass

    if not _looks_like_path_string(value):
        return value
    
    path_obj = Path(value).expanduser()
    if path_obj.is_absolute():
        return str(path_obj.resolve())
    if value.startswith("eeg_pipeline/"):
        return str((project_root / value).resolve())
    return str((config_dir / value).resolve())


def resolve_config_paths(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    config_dir = config_path.parent
    project_root = Path(__file__).parent.parent
    _resolve_paths_recursive(config, config_dir, project_root)
    return config


def _resolve_paths_recursive(obj: Any, config_dir: Path, project_root: Path) -> None:
    non_path_keys = {"project_root", "task", "random_state", "picks"}
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and key not in non_path_keys:
                obj[key] = _resolve_single_path(value, config_dir, project_root)
            elif isinstance(value, (dict, list)):
                _resolve_paths_recursive(value, config_dir, project_root)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                _resolve_paths_recursive(item, config_dir, project_root)


###################################################################
# Configuration Loading
###################################################################


class ConfigError(Exception):
    pass


_CONFIG: Optional[Dict[str, Any]] = None
_CONFIG_PATH: Optional[Path] = None
_CONFIG_MTIME: Optional[float] = None


class ConfigDict(dict):
    def __init__(self, data: Dict[str, Any]):
        super().__init__(data)
    
    def get(self, key: str, default: Any = None) -> Any:
        return get_nested_value(self, key, default)
    
    def __getattr__(self, key: str) -> Any:
        """Attribute-style access for config values.

        Resolution order:
        1. Top-level key in the config dict
        2. Nested under ``paths.<key>`` (returned as ``Path`` when string)
        3. Nested under ``project.<key>`` (for metadata like subjects/task)
        """
        if key.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
        
        if key in self:
            value = self[key]
            return ConfigDict(value) if isinstance(value, dict) else value
        
        paths_value = get_nested_value(self, f"paths.{key}", None)
        if paths_value is not None:
            return Path(paths_value) if isinstance(paths_value, str) else paths_value
        
        project_value = get_nested_value(self, f"project.{key}", None)
        if project_value is not None:
            return project_value
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


def _should_reload_config(config_path: Path) -> bool:
    global _CONFIG, _CONFIG_PATH, _CONFIG_MTIME
    
    if _CONFIG is None:
        return True
    
    if _CONFIG_PATH != config_path:
        return True
    
    current_mtime = config_path.stat().st_mtime
    if _CONFIG_MTIME is not None and current_mtime != _CONFIG_MTIME:
        return True
    
    return False


def _load_config_from_file(config_path: Path) -> Dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML config: {e}") from e
    except OSError as e:
        raise ConfigError(f"Failed to load config file: {e}") from e
    
    return resolve_config_paths(config, config_path)


def _apply_thread_limits(config: Dict[str, Any]) -> None:
    limits = get_nested_value(config, "environment.thread_limits", {})
    for var, value in limits.items():
        os.environ.setdefault(var, str(value))


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    script_name: Optional[str] = None,
    apply_thread_limits: bool = True
) -> ConfigDict:
    global _CONFIG, _CONFIG_PATH, _CONFIG_MTIME
    
    resolved_path = _resolve_config_path(config_path)
    _validate_config_path(resolved_path)
    
    if _should_reload_config(resolved_path):
        config = _load_and_cache_config(resolved_path, apply_thread_limits)
    else:
        config = _CONFIG
    
    return ConfigDict(config)


def _resolve_config_path(config_path: Optional[Union[str, Path]]) -> Path:
    if config_path is None:
        return _get_default_config_path()
    return Path(config_path).expanduser().resolve()


def _validate_config_path(config_path: Path) -> None:
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")


def _load_and_cache_config(config_path: Path, apply_thread_limits: bool) -> Dict[str, Any]:
    global _CONFIG, _CONFIG_PATH, _CONFIG_MTIME
    
    config = _load_config_from_file(config_path)
    
    if apply_thread_limits:
        _apply_thread_limits(config)
    
    _CONFIG = config
    _CONFIG_PATH = config_path
    _CONFIG_MTIME = config_path.stat().st_mtime
    
    return config


def load_settings(config_path: Optional[Union[str, Path]] = None, script_name: Optional[str] = None) -> ConfigDict:
    """Alias for load_config for backward compatibility."""
    return load_config(config_path, script_name=script_name)


###################################################################
# Config Value Access Utilities
###################################################################


def get_nested_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    keys = key.split('.')
    value = config

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value


def get_config_value(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    
    if hasattr(config, "get"):
        return config.get(key, default)
    
    if isinstance(config, dict):
        return get_nested_value(config, key, default)
    
    return default


def ensure_config(config: Optional[Any] = None) -> Any:
    if config is not None:
        return config
    return load_config()


def _get_default_config_path() -> Path:
    config_dir = Path(__file__).parent
    return config_dir / "eeg_config.yaml"


def get_config_int(config: Any, key: str, default: int) -> int:
    value = get_config_value(config, key, default)
    return int(value)


def get_config_float(config: Any, key: str, default: float) -> float:
    value = get_config_value(config, key, default)
    return float(value)


def get_frequency_bands(config: Any) -> Dict[str, List[float]]:
    """Get frequency band definitions (ranges) from config."""
    if config is None:
        return {}
    
    if isinstance(config, ConfigDict):
        bands = config.get("time_frequency_analysis.bands")
        if bands:
            return bands
    elif isinstance(config, dict):
        bands = get_nested_value(config, "time_frequency_analysis.bands")
        if bands:
            return bands
    
    if hasattr(config, "frequency_bands"):
        return config.frequency_bands
    
    return {}


def get_frequency_band_names(config: Any) -> List[str]:
    """Get frequency band names (list of strings) from config.
    
    Derives band names from frequency_bands keys. Replaces the old
    features.frequency_bands config path.
    """
    freq_bands = get_frequency_bands(config)
    if freq_bands:
        return list(freq_bands.keys())
    
    # Fallback to default bands if config not available
    return ["delta", "theta", "alpha", "beta", "gamma"]


def parse_temporal_bin_config(bin_config: Any) -> Optional[Tuple[float, float, str]]:
    if isinstance(bin_config, dict):
        return (
            float(bin_config.get("start", 0.0)),
            float(bin_config.get("end", 0.0)),
            str(bin_config.get("label", "unknown"))
        )
    
    if isinstance(bin_config, (list, tuple)) and len(bin_config) >= 3:
        return (
            float(bin_config[0]),
            float(bin_config[1]),
            str(bin_config[2])
        )
    
    return None


###################################################################
# Default Configuration Values
###################################################################

def get_default_frequency_bands() -> Dict[str, List[float]]:
    return {
        "delta": [1.0, 3.9],
        "theta": [4.0, 7.9],
        "alpha": [8.0, 12.9],
        "beta": [13.0, 30.0],
        "gamma": [30.1, 80.0],
    }

def get_frequency_bands_for_aperiodic(config: Any) -> Dict[str, List[float]]:
    freq_bands = get_frequency_bands(config)
    if not freq_bands:
        freq_bands = get_default_frequency_bands()
    return freq_bands


###################################################################
# Unified Constants Loading
###################################################################


def get_constants(section: str, config: Optional[Any] = None) -> Dict[str, Any]:
    if config is None:
        config = load_config()
    
    constants = get_nested_value(config, f"{section}.constants", {})
    if not constants:
        raise ValueError(f"{section}.constants not found in config.")
    
    return dict(constants)
