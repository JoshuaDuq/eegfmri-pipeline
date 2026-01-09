"""
Windowing Utilities (Canonical)
===============================

Single source of truth for time/frequency masking and window computation.
All windowing logic should import from this module.

Provides:
- time_mask, freq_mask: Basic masking functions
- sliding_window_centers: Connectivity sliding windows
- compute_time_windows: Full window computation returning TimeWindows
- TimeWindowSpec: Builder class for complex windowing scenarios
- WindowMetadata: Metadata for individual windows
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict, Optional

from eeg_pipeline.types import TimeWindows


# Numerical tolerances for floating-point comparisons
TIME_TOLERANCE = 1e-9
WINDOW_EDGE_TOLERANCE = 1e-6
CLAMP_TOLERANCE = 1e-12


###################################################################
# Data Classes
###################################################################


@dataclass
class WindowMetadata:
    """Metadata for a single time window."""
    start: float
    end: float
    clamped: bool
    n_samples: int
    valid: bool
    coverage: float  # Fraction of requested window covered by available data


###################################################################
# Basic Masking Functions
###################################################################


def time_mask(times: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    """Create boolean mask for time window [tmin, tmax)."""
    return (times >= tmin) & (times < tmax)


def time_mask_strict(times: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    """Create boolean mask for time window [tmin, tmax), raising error if empty."""
    mask = (times >= float(tmin)) & (times < float(tmax))

    if not np.any(mask):
        time_min = times.min()
        time_max = times.max()
        msg = f"Time window [{tmin}, {tmax}] outside data range [{time_min:.2f}, {time_max:.2f}]"
        raise ValueError(msg)

    return mask


def time_mask_loose(
    times: np.ndarray,
    tmin: float,
    tmax: float,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Create boolean mask for time window [tmin, tmax), using full range if empty."""
    mask = (times >= float(tmin)) & (times < float(tmax))

    if not np.any(mask):
        time_min = times.min()
        time_max = times.max()
        msg = f"Time window [{tmin}, {tmax}] outside data range [{time_min:.2f}, {time_max:.2f}]"
        log = logger or logging.getLogger(__name__)
        log.warning(f"{msg}; using entire time span")
        mask = np.ones_like(times, dtype=bool)

    return mask


def freq_mask(freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Create boolean mask for frequency window [fmin, fmax]."""
    return (freqs >= fmin) & (freqs <= fmax)


def sliding_window_centers(config: Any, n_windows: int) -> np.ndarray:
    """
    Compute centers of sliding windows for connectivity features based on config.
    Uses active start/end and window length/step to cap window count.
    """
    feat_cfg = config.get("feature_engineering.features", {})
    active_default = config.get("time_frequency_analysis.active_window", [0.0, 0.0])
    
    active_window = feat_cfg.get("active_window", active_default)
    if isinstance(active_window, (list, tuple)) and len(active_window) >= 2:
        active_start = float(active_window[0])
        active_end = float(active_window[1])
    else:
        active_start = float(active_default[0])
        active_end = float(active_default[1])

    conn_cfg = config.get("feature_engineering.connectivity", {})
    window_length = float(conn_cfg.get("sliding_window_len", 1.0))
    window_step = float(conn_cfg.get("sliding_window_step", 0.5))

    if active_end <= active_start:
        return np.array([])

    active_duration = active_end - active_start
    available_duration = active_duration - window_length
    max_windows = int(np.floor(available_duration / window_step) + 1)
    max_windows = max(0, max_windows)
    n_use = min(n_windows, max_windows)

    window_center_offset = window_length / 2.0
    centers = active_start + np.arange(n_use) * window_step + window_center_offset
    return centers


def build_time_windows(
    window_len: float,
    step: float,
    tmin: float,
    tmax: float,
) -> List[Tuple[float, float]]:
    """Build list of time windows from start to end with given length and step."""
    if window_len <= 0 or step <= 0:
        return []

    windows: List[Tuple[float, float]] = []
    current_start = float(tmin)
    window_length = float(window_len)
    step_size = float(step)
    time_max = float(tmax)

    while current_start + window_length <= time_max + WINDOW_EDGE_TOLERANCE:
        window_end = current_start + window_length
        windows.append((current_start, window_end))
        current_start += step_size

    return windows


def build_time_windows_fixed_size_clamped(
    tmin: float,
    tmax: float,
    window_len: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build fixed-size windows covering [tmin, tmax], adding final partial window if needed."""
    time_min = float(tmin)
    time_max = float(tmax)
    window_length = float(window_len)

    if window_length <= 0:
        return np.array([]), np.array([])

    windows = build_time_windows(
        window_len=window_length,
        step=window_length,
        tmin=time_min,
        tmax=time_max,
    )
    
    if windows:
        last_window_end = float(windows[-1][1])
    else:
        last_window_end = time_min

    needs_final_window = last_window_end < time_max - CLAMP_TOLERANCE
    if needs_final_window:
        windows.append((last_window_end, time_max))

    if not windows:
        return np.array([]), np.array([])

    starts = np.array([w[0] for w in windows], dtype=float)
    ends = np.array([w[1] for w in windows], dtype=float)
    valid = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)
    return starts[valid], ends[valid]


def build_time_windows_fixed_count(
    tmin: float,
    tmax: float,
    n_windows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build n_windows evenly spaced windows covering [tmin, tmax]."""
    time_min = float(tmin)
    time_max = float(tmax)
    num_windows = int(n_windows)

    if num_windows <= 0:
        return np.array([]), np.array([])
    
    is_valid_range = (
        np.isfinite(time_min)
        and np.isfinite(time_max)
        and time_max > time_min
    )
    if not is_valid_range:
        return np.array([]), np.array([])

    edges = np.linspace(time_min, time_max, num_windows + 1)
    starts = edges[:-1]
    ends = edges[1:]
    valid = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)
    return starts[valid], ends[valid]


def compute_time_windows(
    times: np.ndarray,
    config: Any,
    *,
    logger: Optional[logging.Logger] = None,
    strict: bool = True,
) -> TimeWindows:
    """
    Compute all time window masks once.
    
    Uses configuration to determine baseline and active windows.
    """
    if config is not None:
        sampling_rate = float(getattr(config, "sfreq", 1.0))
    else:
        sampling_rate = 1.0
    
    spec = TimeWindowSpec(
        times=times,
        config=config,
        sampling_rate=sampling_rate,
        logger=logger,
    )
    return time_windows_from_spec(
        spec,
        logger=logger,
        strict=strict,
    )


###################################################################
# TimeWindowSpec Builder Class
###################################################################


class TimeWindowSpec:
    """
    Builder for all temporal masks used in feature extraction.
    
    Provides a more flexible interface than compute_time_windows for
    complex windowing scenarios with custom bins and sliding windows.
    """
    
    def __init__(
        self,
        times: np.ndarray,
        config: Any,
        sampling_rate: float,
        logger: Optional[logging.Logger] = None,
        name: Optional[str] = None,
        explicit_windows: Optional[List[Dict[str, Any]]] = None,
    ):
        if times is None or len(times) == 0:
            raise ValueError("times must be a non-empty array")
        if not isinstance(times, np.ndarray):
            raise TypeError("times must be a numpy array")
        
        self.times = times
        self.sfreq = sampling_rate
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.name = name
        self.explicit_windows = explicit_windows
        
        self.masks: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, WindowMetadata] = {}
        self.errors: List[str] = []
        
        self._build_all_windows()
        
    def _build_all_windows(self):
        """Construct windows based on explicit user input or configuration."""
        explicit_by_name = self._parse_explicit_windows()
        
        if self.name:
            self._build_named_window_set(explicit_by_name)
        else:
            self._build_default_window_set(explicit_by_name)
    
    def _parse_explicit_windows(self) -> Dict[str, Tuple[float, float]]:
        """Parse explicit windows from user input into name -> (start, end) mapping."""
        explicit_by_name: Dict[str, Tuple[float, float]] = {}
        explicit = self.explicit_windows or []
        
        for win in explicit:
            try:
                win_name = str(win.get("name") or "").strip()
                win_tmin = win.get("tmin")
                win_tmax = win.get("tmax")
                
                if not win_name or win_tmin is None or win_tmax is None:
                    continue
                
                explicit_by_name[win_name] = (float(win_tmin), float(win_tmax))
            except (ValueError, TypeError, AttributeError):
                continue
        
        return explicit_by_name
    
    def _build_named_window_set(self, explicit_by_name: Dict[str, Tuple[float, float]]):
        """Build windows for a named/targeted iteration."""
        self._add_baseline_window(explicit_by_name)
        self._add_targeted_window(explicit_by_name)
        self._add_remaining_explicit_windows(explicit_by_name)
    
    def _build_default_window_set(self, explicit_by_name: Dict[str, Tuple[float, float]]):
        """Build default window set (baseline, active, custom)."""
        self._add_baseline_window(explicit_by_name)
        self._add_active_window(explicit_by_name)
        self._add_custom_windows()
        self._add_remaining_explicit_windows(explicit_by_name)
    
    def _add_baseline_window(self, explicit_by_name: Dict[str, Tuple[float, float]]):
        """Add baseline window from explicit input or config."""
        if "baseline" in explicit_by_name:
            start, end = explicit_by_name["baseline"]
            self._add_window("baseline", float(start), float(end))
            return
        
        baseline_def = self._get_config_window("baseline_window")
        if baseline_def is not None:
            self._add_window("baseline", float(baseline_def[0]), float(baseline_def[1]))
    
    def _add_active_window(self, explicit_by_name: Dict[str, Tuple[float, float]]):
        """Add active window from explicit input or config."""
        if "active" in explicit_by_name:
            start, end = explicit_by_name["active"]
            self._add_window("active", float(start), float(end))
            return
        
        active_def = self._get_config_window("active_window")
        if active_def is not None:
            self._add_window("active", float(active_def[0]), float(active_def[1]))
    
    def _add_targeted_window(self, explicit_by_name: Dict[str, Tuple[float, float]]):
        """Add the targeted/named window for this iteration."""
        if self.name in explicit_by_name:
            start, end = explicit_by_name[self.name]
            self._add_window(self.name, float(start), float(end))
            return
        
        window_def = self._find_named_window_in_config(self.name)
        if window_def is not None:
            self._add_window(self.name, float(window_def[0]), float(window_def[1]))
            return
        
        name_lower = str(self.name).strip().lower()
        if name_lower in {"full", "all"}:
            self._add_window(self.name, self.times[0], self.times[-1])
        else:
            self._add_empty_window(self.name, reason="missing_named_window")
    
    def _add_custom_windows(self):
        """Add custom windows from config."""
        feat_cfg = self.config.get("feature_engineering.windows", {}) if self.config else {}
        custom = feat_cfg.get("custom_windows", [])
        
        if not isinstance(custom, list):
            return
        
        for win in custom:
            if not isinstance(win, dict):
                continue
            if "name" not in win or "start" not in win or "end" not in win:
                continue
            
            try:
                self._add_window(
                    win["name"],
                    float(win["start"]),
                    float(win["end"]),
                )
            except (ValueError, TypeError):
                continue
    
    def _add_remaining_explicit_windows(self, explicit_by_name: Dict[str, Tuple[float, float]]):
        """Add any explicit windows not yet processed."""
        for win_name, (start, end) in explicit_by_name.items():
            if win_name == self.name:
                continue
            if win_name in self.masks:
                continue
            self._add_window(win_name, float(start), float(end))
    
    def _get_config_window(self, key: str) -> Optional[Tuple[float, float]]:
        """Get window definition from config for given key."""
        if self.config is None:
            return None
        
        feat_cfg = self.config.get("feature_engineering.windows", {})
        tf_cfg = self.config.get("time_frequency_analysis", {})
        
        window_def = feat_cfg.get(key) or tf_cfg.get(key)
        if isinstance(window_def, (list, tuple)) and len(window_def) >= 2:
            return (float(window_def[0]), float(window_def[1]))
        
        return None
    
    def _find_named_window_in_config(self, name: str) -> Optional[Tuple[float, float]]:
        """Find named window in config sections."""
        if self.config is None:
            return None
        
        config_sections = [
            self.config.get("feature_engineering.windows", {}),
            self.config.get("feature_engineering.features", {}),
            self.config.get("time_frequency_analysis", {}),
        ]
        
        for cfg in config_sections:
            for key in [name, f"{name}_window"]:
                val = cfg.get(key)
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    try:
                        return (float(val[0]), float(val[1]))
                    except (ValueError, TypeError):
                        continue
        
        return None

    def _add_window(self, name: str, start: float, end: float, prefix: str = ""):
        """Add a window with clamping and validation."""
        full_name = f"{prefix}_{name}" if prefix else name
        
        time_min_available = self.times[0]
        time_max_available = self.times[-1]
        
        final_start, final_end, was_clamped = self._clamp_window_bounds(
            start, end, time_min_available, time_max_available
        )
        
        mask = (self.times >= final_start) & (self.times < final_end)
        n_samples = int(np.sum(mask))
        is_valid = n_samples > 0
        
        requested_duration = end - start
        observed_duration = final_end - final_start
        coverage = observed_duration / requested_duration if requested_duration > 0 else 0.0
        
        if was_clamped and is_valid:
            self.logger.info(
                f"Window '{full_name}' clamped: "
                f"req=[{start:.2f}, {end:.2f}], "
                f"obs=[{final_start:.2f}, {final_end:.2f}]"
            )
        
        if not is_valid:
            self.logger.warning(
                f"Window '{full_name}' is empty! req=[{start:.2f}, {end:.2f}]"
            )
            self.errors.append(full_name)
        
        self.masks[full_name] = mask
        self.metadata[full_name] = WindowMetadata(
            start=final_start,
            end=final_end,
            clamped=was_clamped,
            n_samples=n_samples,
            valid=is_valid,
            coverage=coverage,
        )
    
    def _clamp_window_bounds(
        self,
        start: float,
        end: float,
        time_min: float,
        time_max: float,
    ) -> Tuple[float, float, bool]:
        """Clamp window bounds to available time range."""
        final_start = start
        final_end = end
        was_clamped = False
        
        if final_start < time_min - TIME_TOLERANCE:
            final_start = time_min
            was_clamped = True
        
        if final_end > time_max + TIME_TOLERANCE:
            final_end = time_max
            was_clamped = True
        
        return final_start, final_end, was_clamped

    def _add_empty_window(self, name: str, reason: str = "empty_window") -> None:
        """Register an empty window with metadata for validation."""
        full_name = str(name)
        if not full_name:
            full_name = "unnamed"
        if self.logger:
            self.logger.error("Window '%s' is undefined; %s.", full_name, reason)
        self.errors.append(f"{full_name}:{reason}")
        self.masks[full_name] = np.zeros_like(self.times, dtype=bool)
        self.metadata[full_name] = WindowMetadata(
            start=np.nan,
            end=np.nan,
            clamped=False,
            n_samples=0,
            valid=False,
            coverage=0.0,
        )
        
    def get_mask(self, name: str) -> np.ndarray:
        """Get mask by name, returns empty mask if not found."""
        return self.masks.get(name, np.zeros_like(self.times, dtype=bool))
        
    def get_sliding_windows(self, length: float, step: float) -> List[Tuple[str, np.ndarray]]:
        """Generate sliding windows within the active window."""
        active_meta = self.metadata.get("active")
        if active_meta is None or not active_meta.valid:
            return []
        
        active_start = active_meta.start
        active_end = active_meta.end
        
        windows = []
        current_start = active_start
        window_index = 0
        
        while current_start + length <= active_end + TIME_TOLERANCE:
            window_end = current_start + length
            mask = (self.times >= current_start) & (self.times < window_end)
            
            if np.sum(mask) > 0:
                window_name = f"slide{window_index}"
                windows.append((window_name, mask))
            
            current_start += step
            window_index += 1
        
        return windows


def time_windows_from_spec(
    spec: TimeWindowSpec,
    *,
    logger: Optional[logging.Logger] = None,
    strict: bool = True,
) -> TimeWindows:
    """Create a TimeWindows object from a specification."""
    errors = _validate_spec_windows(spec, logger, strict)
    
    masks = dict(spec.masks)
    ranges = _build_ranges_dict(spec.metadata)
    active_key = _determine_active_key(spec.name, masks)
    
    has_any_clamped = any(meta.clamped for meta in spec.metadata.values())
    empty_mask = np.zeros_like(spec.times, dtype=bool)
    
    return TimeWindows(
        baseline_mask=spec.get_mask("baseline"),
        active_mask=spec.get_mask(active_key) if active_key else empty_mask,
        baseline_range=ranges.get("baseline", (np.nan, np.nan)),
        active_range=ranges.get(active_key, (np.nan, np.nan)) if active_key else (np.nan, np.nan),
        masks=masks,
        ranges=ranges,
        clamped=has_any_clamped,
        valid=len(errors) == 0,
        errors=errors,
        times=spec.times,
        name=spec.name,
    )


def _validate_spec_windows(
    spec: TimeWindowSpec,
    logger: Optional[logging.Logger],
    strict: bool,
) -> List[str]:
    """Validate that spec has at least one valid window."""
    has_valid_window = any(meta.valid for meta in spec.metadata.values())
    errors: List[str] = []
    
    if not has_valid_window:
        errors.append("No valid time windows defined or found in data")
    
    if errors and logger:
        error_msg = "; ".join(errors)
        if strict:
            logger.error("Time window validation failed: %s", error_msg)
        else:
            logger.warning("Time window validation failed: %s", error_msg)
    
    if errors and strict:
        raise ValueError("; ".join(errors))
    
    return errors


def _build_ranges_dict(metadata: Dict[str, WindowMetadata]) -> Dict[str, Tuple[float, float]]:
    """Build ranges dictionary from metadata."""
    return {
        name: (float(meta.start), float(meta.end))
        for name, meta in metadata.items()
    }


def _determine_active_key(name: Optional[str], masks: Dict[str, np.ndarray]) -> Optional[str]:
    """Determine the active window key, honoring named iteration if present."""
    if name and name in masks:
        return name
    
    if "active" in masks:
        return "active"
    
    for key in masks:
        if key != "baseline":
            return key
    
    return None


###################################################################
# Pain Window Utilities (migrated from general.py)
###################################################################


def get_active_window(
    constants: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    """Get the active window from config or constants."""
    if config is not None:
        active_window = config.get("time_frequency_analysis.active_window")
        if active_window is None:
            raise ValueError("active_window not found in config")
        return tuple(active_window)
    
    if constants is None:
        raise ValueError("Either constants or config must be provided")
    
    if "ACTIVE_WINDOW" not in constants:
        raise KeyError(
            "ACTIVE_WINDOW not found in constants. "
            "Use ACTIVE_WINDOW (tuple) not ACTIVE_END (float)"
        )
    
    return constants["ACTIVE_WINDOW"]


###################################################################
# Segment Masks for Feature Extraction
###################################################################


def get_segment_masks(
    times: np.ndarray,
    windows: Optional[TimeWindows],
    config: Optional[Any] = None,
) -> Dict[str, Optional[np.ndarray]]:
    """Return all named masks from the TimeWindows object."""
    if windows is None:
        return {}

    masks = dict(windows.masks)
    
    if windows.baseline_mask is not None and "baseline" not in masks:
        if np.any(windows.baseline_mask):
            masks["baseline"] = windows.baseline_mask

    if windows.active_mask is not None:
        active_name = getattr(windows, "name", "active") or "active"
        if active_name not in masks and np.any(windows.active_mask):
            masks[active_name] = windows.active_mask

    if windows.name:
        targeted = {}
        if "baseline" in masks:
            targeted["baseline"] = masks["baseline"]
        if windows.name in masks:
            targeted[windows.name] = masks[windows.name]
        return targeted

    return masks


def make_mask_for_times(spec: Any, window_name: str, times: np.ndarray) -> np.ndarray:
    """Get a window mask aligned to an arbitrary time vector.

    Use this when downstream computations have a different time axis than the
    one used to build `spec` (e.g., decimated TFR time points).
    """
    if spec is None:
        raise ValueError("spec is required")
    if times is None:
        raise ValueError("times is required")

    spec_times = getattr(spec, "times", None)
    if spec_times is not None:
        can_reuse_mask = (
            len(times) == len(spec_times)
            and hasattr(spec, "get_mask")
        )
        if can_reuse_mask:
            mask = spec.get_mask(window_name)
            if mask is not None and mask.shape == times.shape:
                return mask

    window_range = _get_window_range_from_spec(spec, window_name)
    if window_range is not None:
        start, end = window_range
        if np.isfinite(start) and np.isfinite(end) and end > start:
            return (times >= float(start)) & (times < float(end))

    return np.zeros_like(times, dtype=bool)


def _get_window_range_from_spec(spec: Any, window_name: str) -> Optional[Tuple[float, float]]:
    """Extract window range from spec, trying TimeWindows then TimeWindowSpec."""
    key = str(window_name).lower()
    
    ranges = getattr(spec, "ranges", None)
    if isinstance(ranges, dict):
        window_range = ranges.get(key) or ranges.get(window_name)
        if window_range is not None:
            return window_range
        
        baseline_aliases = {"baseline", "pre", "prestim"}
        active_aliases = {"active", "stim", "task"}
        
        if key in baseline_aliases:
            window_range = getattr(spec, "baseline_range", None)
            if window_range is not None:
                return window_range
        
        if key in active_aliases:
            window_range = getattr(spec, "active_range", None)
            if window_range is not None:
                return window_range

    metadata = getattr(spec, "metadata", {})
    meta = metadata.get(window_name)
    if meta is not None:
        start = float(getattr(meta, "start", np.nan))
        end = float(getattr(meta, "end", np.nan))
        if np.isfinite(start) and np.isfinite(end):
            return (start, end)
    
    return None
