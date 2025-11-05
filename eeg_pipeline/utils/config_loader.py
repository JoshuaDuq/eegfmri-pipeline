from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import os

###################################################################
# Config Path Resolution
###################################################################


def resolve_config_paths(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    pipeline_dir = Path(__file__).parent
    config_dir = config_path.parent
    project_root = pipeline_dir.parent

    if "paths" in config:
        paths = config["paths"]
        for key, value in paths.items():
            if isinstance(value, str) and key != "project_root":
                path_obj = Path(value).expanduser()
                if path_obj.is_absolute():
                    paths[key] = str(path_obj.resolve())
                elif value.startswith("eeg_pipeline/"):
                    paths[key] = str((project_root / value).resolve())
                else:
                    paths[key] = str((config_dir / value).resolve())

    if "features" in config and "sourcecoords_file" in config["features"]:
        sourcecoords = config["features"]["sourcecoords_file"]
        path_obj = Path(sourcecoords).expanduser()
        if path_obj.is_absolute():
            config["features"]["sourcecoords_file"] = str(path_obj.resolve())
        else:
            config["features"]["sourcecoords_file"] = str((config_dir / sourcecoords).resolve())

    if "project" in config and "root" in config["project"]:
        proj_root = config["project"]["root"]
        if isinstance(proj_root, str):
            path_obj = Path(proj_root).expanduser()
            if path_obj.is_absolute():
                config["project"]["root"] = str(path_obj.resolve())
            else:
                config["project"]["root"] = str((config_dir / proj_root).resolve())

    return config


###################################################################
# Configuration Loading
###################################################################


class ConfigError(Exception):
    pass


_CONFIG: Optional[Dict[str, Any]] = None
_CONFIG_PATH: Optional[Path] = None
_CONFIG_MTIME: Optional[float] = None


def get_nested_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    keys = key.split('.')
    value = config

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value


class ConfigDict(dict):
    def __init__(self, data: Dict[str, Any]):
        super().__init__(data)
    
    def get(self, key: str, default: Any = None) -> Any:
        return get_nested_value(self, key, default)
    
    def __getattr__(self, key: str) -> Any:
        if key.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
        if key not in self:
            nested_value = get_nested_value(self, f"paths.{key}", None)
            if nested_value is not None:
                return Path(nested_value) if isinstance(nested_value, str) else nested_value
            nested_value = get_nested_value(self, f"project.{key}", None)
            if nested_value is not None:
                return nested_value
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
        value = self[key]
        if isinstance(value, dict):
            return ConfigDict(value)
        return value


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    script_name: Optional[str] = None,
    apply_thread_limits: bool = True
) -> ConfigDict:
    global _CONFIG, _CONFIG_PATH, _CONFIG_MTIME
    
    if config_path is None:
        config_dir = Path(__file__).parent
        config_path = config_dir / "eeg_config.yaml"
    
    config_path = Path(config_path).expanduser().resolve()
    
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    
    current_mtime = config_path.stat().st_mtime
    
    should_reload = False
    if _CONFIG is None:
        should_reload = True
    elif _CONFIG_PATH != config_path:
        should_reload = True
    elif _CONFIG_MTIME is not None and current_mtime != _CONFIG_MTIME:
        should_reload = True
    
    if should_reload:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML config: {e}") from e
        except OSError as e:
            raise ConfigError(f"Failed to load config file: {e}") from e
        
        config = resolve_config_paths(config, config_path)
        
        if apply_thread_limits:
            limits = get_nested_value(config, "environment.thread_limits", {})
            for var, value in limits.items():
                os.environ.setdefault(var, str(value))
        
        _CONFIG = config
        _CONFIG_PATH = config_path
        _CONFIG_MTIME = current_mtime
    else:
        config = _CONFIG
    
    return ConfigDict(config)


def load_settings(config_path: Optional[Union[str, Path]] = None, script_name: Optional[str] = None) -> ConfigDict:
    return load_config(config_path, script_name=script_name)


load_settings.__doc__ = "Alias for load_config for backward compatibility"
