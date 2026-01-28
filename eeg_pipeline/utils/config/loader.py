from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, List
import json
import yaml
import os

###################################################################
# Config Path Resolution
#
# This module is the **single entry point** for configuration:
# - All code that needs settings should go through `load_config`
#   (or helpers defined below).
# - The underlying source of truth is `eeg_config.yaml` in this
#   directory; no other YAML or config files should be read elsewhere
#   in the package.
# - Helper accessors (e.g. `get_frequency_bands`, `get_constants`)
#   provide typed, domain-specific views on top of the raw config.
#
# CONFIG ACCESS PATTERNS:
# ------------------------
# Preferred access methods (in order of preference):
#
# 1. Using ConfigDict.get() with dot notation (recommended):
#    config = load_config()
#    value = config.get("section.subsection.key", default_value)
#
# 2. Using get_config_value() helper:
#    value = get_config_value(config, "section.subsection.key", default_value)
#
# 3. Using get_nested_value() for raw dicts:
#    value = get_nested_value(config_dict, "section.subsection.key", default_value)
#
# 4. Attribute-style access (for top-level sections only):
#    config = load_config()
#    paths = config.paths  # Returns ConfigDict for paths section
#    task = config.task  # Returns project.task value
#
# AVOID:
# - Direct dict access with [] unless accessing top-level keys
# - Hardcoding parameter values (always use config)
# - Accessing config without defaults (use get() with defaults)
###################################################################


def _looks_like_path_string(value: str) -> bool:
    if value in {".", ".."}:
        return True
    if value.startswith("~"):
        return True
    if len(value) >= 2 and value[1] == ":":
        return True
    if any(sep in value for sep in ("/", "\\")):
        # Exclude Docker image names (e.g., nipreps/fmriprep:25.2.4)
        # Docker images have a colon after the last slash (tag separator)
        if ":" in value:
            last_slash_idx = max(value.rfind("/"), value.rfind("\\"))
            if last_slash_idx < value.rfind(":"):
                return False
        return True
    return False


def _resolve_single_path(value: str, config_dir: Path, project_root: Path) -> str:
    if not value or value.strip() == "":
        return value
    
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
    project_root = Path(__file__).resolve().parents[3]
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
# TUI Overrides (JSON) Support
###################################################################


def _get_overrides_path(config_path: Path) -> Path:
    env_path = os.getenv("EEG_PIPELINE_TUI_OVERRIDES")
    if env_path:
        return Path(env_path).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "eeg_pipeline" / "data" / "derivatives" / ".tui_overrides.json"


def _merge_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_overrides(base[key], value)
        else:
            base[key] = value


def _apply_config_overrides(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    overrides_path = _get_overrides_path(config_path)
    if not overrides_path.exists():
        return config

    try:
        with open(overrides_path, "r", encoding="utf-8") as handle:
            overrides = json.load(handle) or {}
    except (OSError, json.JSONDecodeError):
        return config

    if not isinstance(overrides, dict):
        return config

    _merge_overrides(config, overrides)
    return resolve_config_paths(config, config_path)


###################################################################
# Configuration Loading
###################################################################


class ConfigError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class ConfigValidationError(ConfigError):
    """Exception raised when config validation fails."""
    pass


_CONFIG: Optional[Dict[str, Any]] = None
_CONFIG_PATH: Optional[Path] = None
_CONFIG_MTIME: Optional[float] = None


class ConfigDict(dict):
    def __init__(self, data: Dict[str, Any]):
        super().__init__(data)
    
    def get(self, key: str, default: Any = None) -> Any:
        return get_nested_value(self, key, default)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Support dot-notation for setting nested values."""
        if "." not in key:
            super().__setitem__(key, value)
            return

        keys = key.split('.')
        current = self
        for i, k in enumerate(keys[:-1]):
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value

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
        
        # Check for paths and project sections
        paths_value = get_nested_value(self, f"paths.{key}", None)
        if paths_value is not None:
            return Path(paths_value) if isinstance(paths_value, str) else paths_value
        
        project_value = get_nested_value(self, f"project.{key}", None)
        if project_value is not None:
            return project_value
        
        # Alias 'subjects' to 'project.subject_list'
        if key == "subjects":
            return get_nested_value(self, "project.subject_list", None)
        
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
    """Load and parse YAML config file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Parsed config dictionary with resolved paths
        
    Raises:
        ConfigError: If file cannot be read or parsed
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Failed to parse YAML config at {config_path}: {e}\n"
            "Please check the YAML syntax and ensure all anchors are properly defined."
        ) from e
    except OSError as e:
        raise ConfigError(
            f"Failed to load config file at {config_path}: {e}\n"
            "Please ensure the file exists and is readable."
        ) from e
    
    if not isinstance(config, dict):
        raise ConfigError(
            f"Config file {config_path} must contain a YAML dictionary/mapping, "
            f"got {type(config).__name__}"
        )
    
    config = resolve_config_paths(config, config_path)
    return _apply_config_overrides(config, config_path)


def _apply_thread_limits(config: Dict[str, Any]) -> None:
    limits = get_nested_value(config, "environment.thread_limits", {})
    for var, value in limits.items():
        os.environ.setdefault(var, str(value))


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    apply_thread_limits: bool = True
) -> ConfigDict:
    """Load configuration from YAML file.
    
    This is the main entry point for accessing configuration. The config is
    cached and automatically reloaded if the file changes.
    
    Args:
        config_path: Optional path to config file. If None, uses default
                    eeg_config.yaml in config directory.
        apply_thread_limits: Whether to apply thread limits from config
        
    Returns:
        ConfigDict instance providing dot-notation and dict access
        
    Raises:
        ConfigError: If config file cannot be loaded or parsed
        
    Example:
        >>> config = load_config()
        >>> task = config.get("project.task", "default_task")
        >>> alpha = config.get("statistics.sig_alpha", 0.05)
    """
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


###################################################################
# Config Value Access Utilities
###################################################################


def get_nested_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get nested config value using dot notation.
    
    Args:
        config: Configuration dictionary
        key: Dot-separated key path (e.g., "section.subsection.key")
        default: Default value to return if key not found
        
    Returns:
        Config value or default if not found
        
    Example:
        >>> config = {"section": {"subsection": {"key": "value"}}}
        >>> get_nested_value(config, "section.subsection.key", "default")
        'value'
        >>> get_nested_value(config, "section.missing", "default")
        'default'
    """
    if not isinstance(config, dict):
        return default
        
    keys = key.split('.')
    value = config

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value


def get_config_value(config: Any, key: str, default: Any) -> Any:
    """Get config value with fallback to default.
    
    Works with ConfigDict, regular dicts, or None. This is the preferred
    method for accessing config values when you need a default.
    
    Args:
        config: Configuration object (ConfigDict, dict, or None)
        key: Dot-separated key path (e.g., "section.subsection.key")
        default: Default value to return if key not found or config is None
        
    Returns:
        Config value or default
        
    Example:
        >>> config = load_config()
        >>> alpha = get_config_value(config, "statistics.sig_alpha", 0.05)
    """
    if config is None:
        return default
    
    if hasattr(config, "get"):
        return config.get(key, default)
    
    if isinstance(config, dict):
        return get_nested_value(config, key, default)
    
    return default


_MISSING = object()


def require_config_value(config: Any, key: str) -> Any:
    """Return a required config value or raise.

    This is the strict counterpart to ``get_config_value``. Use it at
    analysis/plotting entry points where missing configuration should surface
    immediately (no silent fallbacks).
    """
    if config is None:
        raise ConfigError(f"Missing required config '{key}': config is None")

    if hasattr(config, "get"):
        value = config.get(key, _MISSING)
    elif isinstance(config, dict):
        value = get_nested_value(config, key, _MISSING)
    else:
        value = _MISSING

    if value is _MISSING:
        raise ConfigError(f"Missing required config key: {key}")

    if value is None:
        raise ConfigError(f"Missing required config value (None): {key}")

    if isinstance(value, str) and value.strip() == "":
        raise ConfigError(f"Missing required config value (empty string): {key}")

    return value


def ensure_config(config: Optional[Any] = None) -> Any:
    if config is not None:
        return config
    return load_config()


def _get_default_config_path() -> Path:
    config_dir = Path(__file__).parent
    return config_dir / "eeg_config.yaml"


def get_project_root() -> Path:
    """Return the project/repo root (parent of the eeg_pipeline package)."""
    return Path(__file__).resolve().parents[3]


def get_config_int(config: Any, key: str, default: int) -> int:
    value = get_config_value(config, key, default)
    return int(value)


def get_config_float(config: Any, key: str, default: float) -> float:
    value = get_config_value(config, key, default)
    return float(value)


def get_config_bool(config: Any, key: str, default: bool) -> bool:
    value = get_config_value(config, key, default)
    return bool(value)


def get_config_str(config: Any, key: str, default: str) -> str:
    value = get_config_value(config, key, default)
    return str(value) if value is not None else default


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
    
    return get_default_frequency_bands()


def get_frequency_band_names(config: Any) -> List[str]:
    """Get frequency band names (list of strings) from config."""
    freq_bands = get_frequency_bands(config)
    if freq_bands:
        return list(freq_bands.keys())
    
    return ["delta", "theta", "alpha", "beta", "gamma"]


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
    """Get frequency bands for aperiodic analysis with fallback to defaults."""
    return get_frequency_bands(config) or get_default_frequency_bands()


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


###################################################################
# Behavior Analysis Constants
###################################################################


def get_min_samples(config: Any, sample_type: str = "default") -> int:
    """Get minimum samples threshold from config."""
    defaults = {"channel": 10, "roi": 20, "default": 5, "edge": 30}
    if config is None:
        return defaults.get(sample_type, 5)
    return int(get_config_value(config, f"behavior_analysis.min_samples.{sample_type}",
                                 defaults.get(sample_type, 5)))


###################################################################
# Statistical Constants
###################################################################


def get_fisher_z_clip_values(config: Any) -> Tuple[float, float]:
    """Get Fisher z-transform clipping bounds from config.
    
    Args:
        config: Configuration object (ConfigDict, dict, or None)
        
    Returns:
        Tuple of (clip_min, clip_max) for Fisher z-transform clipping
    """
    clip_min = get_config_value(config, "statistics.constants.fisher_z_clip_min", -0.999999)
    clip_max = get_config_value(config, "statistics.constants.fisher_z_clip_max", 0.999999)
    return float(clip_min), float(clip_max)


###################################################################
# Feature Extraction Constants
###################################################################


def get_feature_constant(config: Any, constant_name: str, default: Any = None) -> Any:
    """Get a feature extraction constant from config.
    
    Automatically converts string representations of numbers to the appropriate
    numeric type based on the default value's type.
    """
    if config is None:
        return default

    constant_map = {
        "EPSILON_STD": "feature_engineering.constants.epsilon_std",
        "EPSILON_PSD": "feature_engineering.constants.epsilon_psd",
        "EPSILON_AMP": "feature_engineering.constants.epsilon_amp",
        "MIN_EPOCHS_FOR_FEATURES": "feature_engineering.constants.min_epochs_for_features",
        "MIN_CHANNELS_FOR_CONNECTIVITY": "feature_engineering.constants.min_channels_for_connectivity",
        "MIN_SAMPLES_FOR_PSD": "feature_engineering.constants.min_samples_for_psd",
        "MIN_VALID_FRACTION": "feature_engineering.constants.min_valid_fraction",
        "MIN_EPOCHS_FOR_PLV": "feature_engineering.constants.min_epochs_for_plv",
        "MIN_EDGE_SAMPLES": "feature_engineering.constants.min_edge_samples",
        "DEFAULT_PE_ORDER": "feature_engineering.complexity.pe_order",
        "DEFAULT_PE_DELAY": "feature_engineering.complexity.pe_delay",
    }

    config_path = constant_map.get(constant_name)
    if config_path is None:
        return default
    
    value = get_config_value(config, config_path, default)
    
    # Auto-convert string representations of numbers to numeric types
    if isinstance(value, str) and default is not None:
        try:
            if isinstance(default, float):
                return float(value)
            elif isinstance(default, int):
                return int(float(value))  # Handle "1e-12" -> 0 for ints
        except (ValueError, TypeError):
            return default
    
    return value
