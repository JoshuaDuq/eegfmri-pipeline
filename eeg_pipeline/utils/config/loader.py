from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

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


_NON_PATH_KEYS = {
    "project_root",
    "task",
    "random_state",
    "picks",
    "volume_marker_description",
    "pulse_marker_description",
}
_PROJECT_ROOT_PREFIXES = ("data/", "eeg_pipeline/")


def _is_blank(value: str) -> bool:
    return not value or value.strip() == ""


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


def _is_windows_drive_absolute_path(value: str) -> bool:
    return len(value) >= 3 and value[0].isalpha() and value[1] == ":" and value[2] in ("/", "\\")


def _resolve_single_path(value: str, config_dir: Path, project_root: Path) -> str:
    if _is_blank(value):
        return value

    if not _looks_like_path_string(value):
        return value

    if _is_windows_drive_absolute_path(value):
        if os.name == "nt":
            return Path(value).expanduser().resolve().as_posix()
        return value

    path_obj = Path(value).expanduser()
    if path_obj.is_absolute():
        return path_obj.resolve().as_posix()
    if value.startswith(_PROJECT_ROOT_PREFIXES):
        return (project_root / value).resolve().as_posix()
    return (config_dir / value).resolve().as_posix()


def resolve_config_paths(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    config_dir = config_path.parent
    project_root = get_project_root()
    _resolve_paths_recursive(config, config_dir, project_root)
    return config


def _resolve_paths_recursive(obj: Any, config_dir: Path, project_root: Path) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and key not in _NON_PATH_KEYS:
                obj[key] = _resolve_single_path(value, config_dir, project_root)
            elif isinstance(value, (dict, list)):
                _resolve_paths_recursive(value, config_dir, project_root)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            if isinstance(item, str):
                obj[index] = _resolve_single_path(item, config_dir, project_root)
            elif isinstance(item, (dict, list)):
                _resolve_paths_recursive(item, config_dir, project_root)


###################################################################
# TUI Overrides (JSON) Support
###################################################################


#: Process-wide override for which config file ``load_config()`` reads when no path is
#: passed. Set by the CLI's ``--config``. It exists because almost all library code
#: reaches configuration through ``ensure_config()``/``load_config()`` with no argument:
#: threading a path down every one of those call sites would be a far larger change, and
#: one missed site would silently read the packaged default while the rest read the
#: user's file — a study running half on someone else's settings.
_DEFAULT_CONFIG_PATH_OVERRIDE: Optional[Path] = None


def set_default_config_path(config_path: Optional[Union[str, Path]]) -> None:
    """Point every subsequent argument-less ``load_config()`` at this file.

    ``None`` restores the packaged default. Raises if the file does not exist, so a typo
    in ``--config`` is reported against the flag rather than silently ignored in favour
    of the packaged config.
    """
    global _DEFAULT_CONFIG_PATH_OVERRIDE

    if config_path is None:
        _DEFAULT_CONFIG_PATH_OVERRIDE = None
        return

    resolved = Path(config_path).expanduser().resolve()
    if not resolved.exists():
        raise ConfigError(f"Configuration file not found: {resolved}")
    _DEFAULT_CONFIG_PATH_OVERRIDE = resolved


def _get_overrides_path(config_path: Path) -> Path:
    env_path = os.getenv("EEG_PIPELINE_TUI_OVERRIDES")
    if env_path:
        return Path(env_path).expanduser().resolve()
    project_root = get_project_root()
    preferred = project_root / "data" / "derivatives" / ".tui_overrides.json"
    return preferred


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
    except OSError as exc:
        raise ConfigError(f"Failed to read TUI overrides at {overrides_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Failed to parse TUI overrides at {overrides_path}: {exc}") from exc

    if not isinstance(overrides, dict):
        raise ConfigError(f"TUI overrides at {overrides_path} must contain a JSON object.")

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
_CONFIG_OVERRIDES_PATH: Optional[Path] = None
_CONFIG_OVERRIDES_MTIME: Optional[float] = None
_CONFIG_LOCK = threading.Lock()


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

        keys = key.split(".")
        current = self
        for k in keys[:-1]:
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
        if key.startswith("_"):
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
    global _CONFIG_OVERRIDES_PATH, _CONFIG_OVERRIDES_MTIME

    if _CONFIG is None:
        return True

    if _CONFIG_PATH != config_path:
        return True

    overrides_path, overrides_mtime = _get_overrides_cache_state(config_path)
    if _CONFIG_OVERRIDES_PATH != overrides_path:
        return True
    if _CONFIG_OVERRIDES_MTIME != overrides_mtime:
        return True

    current_mtime = config_path.stat().st_mtime
    if _CONFIG_MTIME is not None and current_mtime != _CONFIG_MTIME:
        return True

    return False


def _get_overrides_cache_state(config_path: Path) -> tuple[Path, Optional[float]]:
    """Return the resolved overrides path and its current mtime, if it exists."""
    overrides_path = _get_overrides_path(config_path)
    overrides_mtime = overrides_path.stat().st_mtime if overrides_path.exists() else None
    return overrides_path, overrides_mtime


def _parse_config_yaml(config_path: Path) -> Dict[str, Any]:
    """Read one YAML file into a dict, with the parse errors this project reports."""
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
    return config


def get_presets_dir() -> Path:
    """Directory holding the packaged starting-point configs."""
    return Path(__file__).parent / "presets"


def _resolve_extends_target(extends: Any, config_path: Path) -> Path:
    """Resolve one ``extends:`` value to a config file.

    A bare name (no separator, no suffix) names a packaged preset; anything else is a
    path, taken relative to the extending file so a study directory can be moved whole.
    """
    if not isinstance(extends, str) or not extends.strip():
        raise ConfigError(
            f"'extends' in {config_path} must be a preset name or a path to a YAML file, "
            f"got {extends!r}."
        )

    value = extends.strip()
    if "/" not in value and "\\" not in value and not value.endswith((".yaml", ".yml")):
        candidate = get_presets_dir() / f"{value}.yaml"
        if not candidate.exists():
            available = sorted(p.stem for p in get_presets_dir().glob("*.yaml"))
            raise ConfigError(
                f"{config_path} extends unknown preset {value!r}. "
                f"Available presets: {', '.join(available) or 'none'}."
            )
        return candidate

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise ConfigError(f"{config_path} extends {candidate}, which does not exist.")
    return candidate


def _load_config_layers(config_path: Path, seen: Tuple[Path, ...] = ()) -> Dict[str, Any]:
    """Load one config and everything it extends, nearest layer winning.

    ``extends:`` is what makes a per-study config viable. Without it a study that differs
    from the default in nine keys has to copy all 1200-odd lines, and every later
    correction to a scientific default has to be re-applied by hand to each copy.

    Each layer's relative paths resolve against that layer's own directory, so a preset
    keeps meaning what it said no matter where the file extending it lives.
    """
    resolved_path = config_path.resolve()
    if resolved_path in seen:
        chain = " -> ".join(str(p) for p in (*seen, resolved_path))
        raise ConfigError(f"Circular 'extends' in configuration: {chain}")

    config = _parse_config_yaml(resolved_path)
    extends = config.pop("extends", None)
    config = resolve_config_paths(config, resolved_path)

    if extends is None:
        return config

    base_path = _resolve_extends_target(extends, resolved_path)
    base = _load_config_layers(base_path, (*seen, resolved_path))
    _merge_overrides(base, config)
    return base


def _load_config_from_file(config_path: Path) -> Dict[str, Any]:
    """Load a config, its ``extends`` chain, and the TUI overrides on top.

    Args:
        config_path: Path to config YAML file

    Returns:
        Parsed config dictionary with resolved paths

    Raises:
        ConfigError: If file cannot be read or parsed
    """
    config = _load_config_layers(config_path)
    config = _apply_config_overrides(config, config_path)
    _apply_paradigm(config, declared=_keys_declared_by_the_study(config_path))
    return config


#: The four ``task_is_rest`` flags ``project.paradigm`` stands in for. They are read by
#: different subsystems, and two separate validators already raise when a pair of them
#: disagrees, so in practice they were never four independent choices — only four places
#: to make the same one and one chance in two of making it inconsistently by hand.
_PARADIGM_REST_FLAGS = (
    "preprocessing.task_is_rest",
    "feature_engineering.task_is_rest",
    "fmri_preprocessing.task_is_rest",
    "fmri_resting_state.task_is_rest",
)

#: Feature families defined by their relationship to an event. A fixed-length
#: resting-state segment has no event, so these are not merely unset but unavailable.
#: :func:`eeg_pipeline.analysis.features.rest.validate_rest_feature_categories` raises on
#: them; it imports this name rather than restating it.
REST_INCOMPATIBLE_FEATURE_CATEGORIES = frozenset({"erp", "erds", "itpc", "phase"})

#: Keys a base config written for task epochs sets to values a resting-state run cannot
#: satisfy, with what ``paradigm: rest`` reduces each to. Every one of these was already
#: an error — from :mod:`eeg_pipeline.utils.config.coherence` for the band report, from
#: feature extraction for the families — which made adapting a task config to rest a
#: matter of clearing errors the user did not choose and could not have known to expect.
_REST_NEUTRALIZED_KEYS = (
    "feature_engineering.feature_categories",
    "ica.band_specific_report.tfr.enabled",
    "ica.band_specific_report.comparisons",
)


def _keys_declared_by_the_study(config_path: Path) -> frozenset:
    """The neutralizable keys the study's own file sets, as opposed to inheriting.

    Provenance is what separates "this base was written for someone else's acquisition"
    from "this study asked for this". The merged config cannot tell them apart, so the
    top layer is re-read on its own here — the same two sources
    :func:`_load_config_from_file` layers, minus everything ``extends`` brought in.
    """
    sources = [_parse_config_yaml(config_path.resolve())]
    overrides_path = _get_overrides_path(config_path)
    if overrides_path.exists():
        try:
            with open(overrides_path, "r", encoding="utf-8") as handle:
                overrides = json.load(handle) or {}
        except (OSError, json.JSONDecodeError):
            # _apply_config_overrides reads the same file and reports these properly.
            overrides = {}
        if isinstance(overrides, dict):
            sources.append(overrides)

    return frozenset(
        dotted_key
        for dotted_key in _REST_NEUTRALIZED_KEYS
        if any(get_nested_value(source, dotted_key, _MISSING) is not _MISSING for source in sources)
    )


def _set_nested_value(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Assign a dotted key, creating the sections along the way."""
    *sections, leaf = dotted_key.split(".")
    target = config
    for section_name in sections:
        section = target.get(section_name)
        if not isinstance(section, dict):
            section = {}
            target[section_name] = section
        target = section
    target[leaf] = value


def _apply_paradigm(config: Dict[str, Any], *, declared: frozenset = frozenset()) -> None:
    """Derive the ``task_is_rest`` flags from ``project.paradigm`` when it is set.

    The paradigm wins over any individual flag it covers, including one inherited
    through ``extends``. That is what lets a preset say ``paradigm: rest`` in one line
    instead of restating four booleans that the base config set to ``false``.

    Absent (or null) it changes nothing, so a config written before this key existed
    keeps setting the flags directly.
    """
    paradigm = get_nested_value(config, "project.paradigm", None)
    if paradigm is None:
        return

    normalized = str(paradigm).strip().lower()
    if normalized not in {"task", "rest"}:
        raise ConfigError(
            f"project.paradigm must be 'task' or 'rest', got {paradigm!r}. "
            "Use 'rest' for resting-state or baseline-only acquisitions."
        )

    task_is_rest = normalized == "rest"
    for dotted_key in _PARADIGM_REST_FLAGS:
        _set_nested_value(config, dotted_key, task_is_rest)

    if task_is_rest:
        _neutralize_task_only_settings(config, declared)


def _neutralize_task_only_settings(config: Dict[str, Any], declared: frozenset) -> None:
    """Reduce inherited task-epoch settings to what a fixed-length segment supports.

    Only what arrived through ``extends`` is touched. A key the study's own config names
    is an instruction, and the answer to an impossible instruction is to say so — the
    coherence report and the feature-category validator both name it against the
    paradigm — not to quietly do something else. That distinction is the whole reason
    this takes a provenance set rather than rewriting whatever it finds.
    """
    for dotted_key in _REST_NEUTRALIZED_KEYS:
        if dotted_key in declared:
            continue

        current = get_nested_value(config, dotted_key, _MISSING)
        if current is _MISSING:
            continue

        if dotted_key == "feature_engineering.feature_categories":
            if not current:
                continue
            kept = [
                category
                for category in current
                if str(category) not in REST_INCOMPATIBLE_FEATURE_CATEGORIES
            ]
            if not kept:
                dropped = ", ".join(sorted(str(category) for category in current))
                raise ConfigError(
                    f"project.paradigm is 'rest', which leaves "
                    f"feature_engineering.feature_categories empty: every family it "
                    f"inherits ({dropped}) is defined relative to an event, and a "
                    f"fixed-length resting-state segment has none. Name the families "
                    f"this study wants, or set project.paradigm to 'task'."
                )
            _set_nested_value(config, dotted_key, kept)
        elif dotted_key == "ica.band_specific_report.tfr.enabled":
            _set_nested_value(config, dotted_key, False)
        elif dotted_key == "ica.band_specific_report.comparisons":
            _set_nested_value(config, dotted_key, [])


def _apply_thread_limits(config: Dict[str, Any]) -> None:
    limits = get_nested_value(config, "environment.thread_limits", {})
    for var, value in limits.items():
        os.environ.setdefault(var, str(value))


def load_config(
    config_path: Optional[Union[str, Path]] = None, apply_thread_limits: bool = True
) -> ConfigDict:
    """Load configuration from YAML file.

    This is the main entry point for accessing configuration. The config is
    cached and automatically reloaded if the file changes.
    Thread-safe: concurrent calls are serialized via _CONFIG_LOCK.

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

    with _CONFIG_LOCK:
        if _should_reload_config(resolved_path):
            config = _load_and_cache_config(resolved_path, apply_thread_limits)
        else:
            config = _CONFIG

    # Return an isolated per-call copy so runtime overrides/mutations do not
    # leak back into the shared loader cache.
    return ConfigDict(copy.deepcopy(config))


def _resolve_config_path(config_path: Optional[Union[str, Path]]) -> Path:
    if config_path is None:
        return _get_default_config_path()
    return Path(config_path).expanduser().resolve()


def _validate_config_path(config_path: Path) -> None:
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")


def _load_and_cache_config(config_path: Path, apply_thread_limits: bool) -> Dict[str, Any]:
    global _CONFIG, _CONFIG_PATH, _CONFIG_MTIME
    global _CONFIG_OVERRIDES_PATH, _CONFIG_OVERRIDES_MTIME

    config = _load_config_from_file(config_path)

    if apply_thread_limits:
        _apply_thread_limits(config)

    _CONFIG = config
    _CONFIG_PATH = config_path
    _CONFIG_MTIME = config_path.stat().st_mtime
    _CONFIG_OVERRIDES_PATH, _CONFIG_OVERRIDES_MTIME = _get_overrides_cache_state(config_path)

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

    keys = key.split(".")
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

    if isinstance(config, dict):
        return get_nested_value(config, key, default)

    if hasattr(config, "get"):
        return config.get(key, default)

    return default


def get_condition_column_candidates(config: Any) -> List[str]:
    """Return configured candidate event columns used for condition labels.

    Resolution source:
    1. `event_columns.condition` from config (list/tuple/string)
    """
    missing = object()
    raw = get_config_value(config, "event_columns.condition", missing)
    candidates: List[str] = []

    if raw is missing:
        return ["condition", "trial_type", "binary_outcome"]

    if isinstance(raw, (list, tuple)):
        candidates.extend(str(v).strip() for v in raw if str(v).strip())
    elif isinstance(raw, str):
        if "," in raw:
            candidates.extend(part.strip() for part in raw.split(",") if part.strip())
        else:
            text = raw.strip()
            if text:
                candidates.append(text)

    deduped: List[str] = []
    seen: set[str] = set()
    for col in candidates:
        key = col.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(col)

    return deduped


_MISSING = object()


def require_config_value(config: Any, key: str) -> Any:
    """Return a required config value or raise.

    This is the strict counterpart to ``get_config_value``. Use it at
    analysis/plotting entry points where missing configuration should surface
    immediately (no silent fallbacks).
    """
    if config is None:
        raise ConfigError(f"Missing required config '{key}': config is None")

    if isinstance(config, dict):
        value = get_nested_value(config, key, _MISSING)
    elif hasattr(config, "get"):
        value = config.get(key, _MISSING)
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
    """Resolve the config file to read when no path is given.

    Three sources, most explicit first: the CLI's ``--config`` (via
    :func:`set_default_config_path`), the ``EEG_PIPELINE_CONFIG`` environment variable,
    and the packaged ``eeg_config.yaml``.

    The packaged file is the last resort rather than the only option because it lives
    inside the installed package: editing it in place is the only way to configure a
    study otherwise, which means two studies cannot coexist and any edit is lost on
    reinstall.
    """
    if _DEFAULT_CONFIG_PATH_OVERRIDE is not None:
        return _DEFAULT_CONFIG_PATH_OVERRIDE

    env_path = os.getenv("EEG_PIPELINE_CONFIG")
    if env_path and env_path.strip():
        resolved = Path(env_path).expanduser().resolve()
        if not resolved.exists():
            raise ConfigError(
                f"EEG_PIPELINE_CONFIG points at {resolved}, which does not exist. "
                "Unset it to use the packaged default."
            )
        return resolved

    config_dir = Path(__file__).parent
    return config_dir / "eeg_config.yaml"


def get_project_root() -> Path:
    """Return the project/repo root (parent containing pyproject.toml)."""
    env_root = os.getenv("EEG_PIPELINE_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "eeg_pipeline").exists():
            return parent

    # Fallback to historical behavior for safety.
    return here.parents[3]


def get_config_int(config: Any, key: str, default: int) -> int:
    value = get_config_value(config, key, default)
    return int(value)


def get_config_float(config: Any, key: str, default: float) -> float:
    value = get_config_value(config, key, default)
    return float(value)


def get_config_bool(config: Any, key: str, default: bool) -> bool:
    value = get_config_value(config, key, default)
    return bool(value)


def get_frequency_bands(config: Any) -> Dict[str, List[float]]:
    """Get frequency band definitions (ranges) from config."""
    bands = get_config_value(config, "time_frequency_analysis.bands", None)
    return bands if bands else get_default_frequency_bands()


def get_frequency_band_names(config: Any) -> List[str]:
    """Get frequency band names (list of strings) from config."""
    bands = get_frequency_bands(config)
    return list(bands.keys()) if bands else ["delta", "theta", "alpha", "beta", "gamma"]


###################################################################
# Default Configuration Values
###################################################################


def get_default_frequency_bands() -> Dict[str, List[float]]:
    return {
        "delta": [1.0, 3.9],
        "theta": [4.0, 7.9],
        "alpha": [8.0, 12.9],
        "beta": [13.0, 30.0],
        "beta_low_clean": [13.0, 17.9],
        "beta_high_clean": [23.1, 30.0],
        "gamma": [30.1, 80.0],
        "gamma_low_clean": [30.1, 38.0],
        "gamma_mid_clean": [43.0, 56.0],
        "gamma_high_clean": [67.0, 77.0],
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
    return int(
        get_config_value(
            config, f"behavior_analysis.min_samples.{sample_type}", defaults.get(sample_type, 5)
        )
    )


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
