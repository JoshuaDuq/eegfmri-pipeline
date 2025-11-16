from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, List
import yaml
import os

###################################################################
# Config Path Resolution
###################################################################


def _resolve_single_path(value: str, config_dir: Path, project_root: Path) -> str:
    path_obj = Path(value).expanduser()
    if path_obj.is_absolute():
        return str(path_obj.resolve())
    if value.startswith("eeg_pipeline/"):
        return str((project_root / value).resolve())
    return str((config_dir / value).resolve())


def resolve_config_paths(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    config_dir = config_path.parent
    project_root = Path(__file__).parent.parent

    _resolve_paths_section(config, config_dir, project_root)
    _resolve_features_paths(config, config_dir, project_root)
    _resolve_project_paths(config, config_dir, project_root)

    return config


def _resolve_paths_section(config: Dict[str, Any], config_dir: Path, project_root: Path) -> None:
    if "paths" not in config:
        return
    
    paths = config["paths"]
    for key, value in paths.items():
        if isinstance(value, str) and key != "project_root":
            paths[key] = _resolve_single_path(value, config_dir, project_root)


def _resolve_features_paths(config: Dict[str, Any], config_dir: Path, project_root: Path) -> None:
    if "features" not in config:
        return
    
    if "sourcecoords_file" not in config["features"]:
        return
    
    sourcecoords = config["features"]["sourcecoords_file"]
    config["features"]["sourcecoords_file"] = _resolve_single_path(
        sourcecoords, config_dir, project_root
    )


def _resolve_project_paths(config: Dict[str, Any], config_dir: Path, project_root: Path) -> None:
    if "project" not in config:
        return
    
    if "root" not in config["project"]:
        return
    
    proj_root = config["project"]["root"]
    if isinstance(proj_root, str):
        config["project"]["root"] = _resolve_single_path(proj_root, config_dir, project_root)


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
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert ConfigDict to legacy dict format for decoding scripts.
        
        Transforms decoding.* sections to top-level keys:
        - decoding.analysis -> analysis
        - decoding.models -> models
        - decoding.cv -> cv
        - decoding.flags -> flags
        - decoding.paths -> paths (merged with existing paths)
        """
        result = dict(self)
        
        if "decoding" in result:
            decoding = result["decoding"]
            
            if "analysis" in decoding:
                result["analysis"] = decoding["analysis"]
            
            if "models" in decoding:
                result["models"] = decoding["models"]
            
            if "cv" in decoding:
                result["cv"] = decoding["cv"]
            
            if "flags" in decoding:
                result["flags"] = decoding["flags"]
            
            if "paths" in decoding:
                if "paths" not in result:
                    result["paths"] = {}
                result["paths"].update(decoding["paths"])
        
        return result
    
    def __getattr__(self, key: str) -> Any:
        if key.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
        
        if key in self:
            value = self[key]
            return ConfigDict(value) if isinstance(value, dict) else value
        
        paths_value = self._get_paths_value(key)
        if paths_value is not None:
            return paths_value
        
        project_value = self._get_project_value(key)
        if project_value is not None:
            return project_value
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def _get_paths_value(self, key: str) -> Any:
        paths_value = get_nested_value(self, f"paths.{key}", None)
        if paths_value is None:
            return None
        return Path(paths_value) if isinstance(paths_value, str) else paths_value
    
    def _get_project_value(self, key: str) -> Any:
        return get_nested_value(self, f"project.{key}", None)


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
