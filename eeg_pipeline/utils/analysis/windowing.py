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
    tmask = (times >= float(tmin)) & (times < float(tmax))

    if not np.any(tmask):
        msg = f"Time window [{tmin}, {tmax}] outside data range [{times.min():.2f}, {times.max():.2f}]"
        raise ValueError(msg)

    return tmask


def time_mask_loose(
    times: np.ndarray,
    tmin: float,
    tmax: float,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    tmask = (times >= float(tmin)) & (times < float(tmax))

    if not np.any(tmask):
        msg = f"Time window [{tmin}, {tmax}] outside data range [{times.min():.2f}, {times.max():.2f}]"
        if logger:
            logger.warning(f"{msg}; using entire time span")
        else:
            logging.getLogger(__name__).warning(f"{msg}; using entire time span")
        tmask = np.ones_like(times, dtype=bool)

    return tmask


def freq_mask(freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Create boolean mask for frequency window [fmin, fmax]."""
    return (freqs >= fmin) & (freqs <= fmax)


def sliding_window_centers(config: Any, n_windows: int) -> np.ndarray:
    """
    Compute centers of sliding windows for connectivity features based on config.
    Uses plateau start/end and window length/step to cap window count.
    """
    feat_cfg = config.get("feature_engineering.features", {})
    plateau_default = config.get("time_frequency_analysis.plateau_window", [0.0, 0.0])
    
    plateau_window = feat_cfg.get("plateau_window", plateau_default)
    if isinstance(plateau_window, (list, tuple)) and len(plateau_window) >= 2:
        plateau_start = float(plateau_window[0])
        plateau_end = float(plateau_window[1])
    else:
        plateau_start = float(plateau_default[0])
        plateau_end = float(plateau_default[1])

    conn_cfg = config.get("feature_engineering.connectivity", {})
    win_len = float(conn_cfg.get("sliding_window_len", 1.0))
    win_step = float(conn_cfg.get("sliding_window_step", 0.5))

    if plateau_end <= plateau_start:
        return np.array([])

    max_windows = int(np.floor((plateau_end - plateau_start - win_len) / win_step) + 1)
    max_windows = max(0, max_windows)
    n_use = min(n_windows, max_windows)

    centers = plateau_start + np.arange(n_use) * win_step + (win_len / 2.0)
    return centers


def build_time_windows(
    window_len: float,
    step: float,
    tmin: float,
    tmax: float,
) -> List[Tuple[float, float]]:
    windows: List[Tuple[float, float]] = []
    t = float(tmin)
    window_len = float(window_len)
    step = float(step)
    tmax = float(tmax)

    if window_len <= 0 or step <= 0:
        return windows

    while t + window_len <= tmax + 1e-6:
        windows.append((t, t + window_len))
        t += step

    return windows


def build_time_windows_fixed_size_clamped(
    tmin: float,
    tmax: float,
    window_len: float,
) -> Tuple[np.ndarray, np.ndarray]:
    tmin = float(tmin)
    tmax = float(tmax)
    window_len = float(window_len)

    if window_len <= 0:
        return np.array([]), np.array([])

    windows = build_time_windows(window_len=window_len, step=window_len, tmin=tmin, tmax=tmax)
    if windows:
        last_end = float(windows[-1][1])
    else:
        last_end = tmin

    if last_end < tmax - 1e-12:
        windows.append((last_end, tmax))

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
    tmin = float(tmin)
    tmax = float(tmax)
    n_windows = int(n_windows)

    if n_windows <= 0:
        return np.array([]), np.array([])
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
        return np.array([]), np.array([])

    edges = np.linspace(tmin, tmax, n_windows + 1)
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
    spec = TimeWindowSpec(
        times=times,
        config=config,
        sampling_rate=float(getattr(config, "sfreq", 1.0)) if config is not None else 1.0,
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
        self.times = times
        self.sfreq = sampling_rate
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.name = name
        self.explicit_windows = explicit_windows  # User-specified time ranges from CLI/TUI
        
        self.masks: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, WindowMetadata] = {}
        self.errors: List[str] = []
        
        self._build_all_windows()
        
    def _build_all_windows(self):
        """Construct windows based on explicit user input or configuration."""
        # If explicit windows are provided (from CLI/TUI), use ONLY those
        if self.explicit_windows:
            for win in self.explicit_windows:
                win_name = win.get("name")
                win_tmin = win.get("tmin")
                win_tmax = win.get("tmax")
                if win_name and win_tmin is not None and win_tmax is not None:
                    self._add_window(win_name, float(win_tmin), float(win_tmax))
            return
        
        # Fall back to config-based windows only if no explicit windows
        feat_cfg = self.config.get("feature_engineering.windows", {})
        tf_cfg = self.config.get("time_frequency_analysis", {})
        
        # 1. Build baseline ONLY if explicitly defined (usually for normalization)
        baseline_def = feat_cfg.get("baseline_window", tf_cfg.get("baseline_window"))
        if baseline_def and isinstance(baseline_def, (list, tuple)) and len(baseline_def) >= 2:
            self._add_window("baseline", float(baseline_def[0]), float(baseline_def[1]))

        # 2. Targeted window (context/user-defined iteration)
        if self.name:
            # Look for this specific name in all possible config sections
            all_cfgs = [
                self.config.get("feature_engineering.windows", {}),
                self.config.get("feature_engineering.features", {}),
                self.config.get("time_frequency_analysis", {}),
            ]
            found = False
            for cfg in all_cfgs:
                # Check for exact name or common suffix "window"
                for key in [self.name, f"{self.name}_window"]:
                    val = cfg.get(key)
                    if isinstance(val, (list, tuple)) and len(val) >= 2:
                        try:
                            self._add_window(self.name, float(val[0]), float(val[1]))
                            found = True
                            break
                        except (ValueError, TypeError):
                            continue
                if found:
                    break
            
            if not found:
                if str(self.name).strip().lower() in {"full", "all"}:
                    self._add_window(self.name, self.times[0], self.times[-1])
                else:
                    self._add_empty_window(self.name, reason="missing_named_window")
            return

        # 1b. Build active/plateau window if explicitly defined
        active_def = (
            feat_cfg.get("active_window")
            or tf_cfg.get("active_window")
            or tf_cfg.get("plateau_window")
        )
        if active_def and isinstance(active_def, (list, tuple)) and len(active_def) >= 2:
            self._add_window("active", float(active_def[0]), float(active_def[1]))

        # 3. Batch windows: process 'custom_windows' if defined
        custom = feat_cfg.get("custom_windows", [])
        if isinstance(custom, list):
            for win in custom:
                if isinstance(win, dict) and "name" in win and "start" in win and "end" in win:
                    self._add_window(win["name"], float(win["start"]), float(win["end"]))

    def _add_window(self, name: str, start: float, end: float, prefix: str = ""):
        """Add a window with clamping and validation."""
        full_name = f"{prefix}_{name}" if prefix else name
        
        tmin_avail = self.times[0]
        tmax_avail = self.times[-1]
        
        clamped = False
        final_start, final_end = start, end
        
        if final_start < tmin_avail - 1e-9:
            final_start = tmin_avail
            clamped = True
        if final_end > tmax_avail + 1e-9:
            final_end = tmax_avail
            clamped = True
            
        mask = (self.times >= final_start) & (self.times < final_end)
        
        n_samples = np.sum(mask)
        valid = n_samples > 0
        
        req_dur = end - start
        obs_dur = final_end - final_start
        coverage = obs_dur / req_dur if req_dur > 0 else 0.0
        
        if clamped and valid:
            self.logger.info(f"Window '{full_name}' clamped: req=[{start:.2f}, {end:.2f}], obs=[{final_start:.2f}, {final_end:.2f}]")
        
        if not valid:
            self.logger.warning(f"Window '{full_name}' is empty! req=[{start:.2f}, {end:.2f}]")
            self.errors.append(full_name)
            
        self.masks[full_name] = mask
        self.metadata[full_name] = WindowMetadata(
            start=final_start,
            end=final_end,
            clamped=clamped,
            n_samples=int(n_samples),
            valid=valid,
            coverage=coverage
        )

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
        """Generate sliding windows within the plateau."""
        if "plateau" not in self.metadata or not self.metadata["plateau"].valid:
            return []
            
        p_start = self.metadata["plateau"].start
        p_end = self.metadata["plateau"].end
        
        windows = []
        curr = p_start
        idx = 0
        while curr + length <= p_end + 1e-9:
            win_start = curr
            win_end = curr + length
            mask = (self.times >= win_start) & (self.times < win_end)
            if np.sum(mask) > 0:
                windows.append((f"slide{idx}", mask))
            curr += step
            idx += 1
            
        return windows


def time_windows_from_spec(
    spec: TimeWindowSpec,
    *,
    logger: Optional[logging.Logger] = None,
    strict: bool = True,
) -> TimeWindows:
    """Create a TimeWindows object from a specification."""
    baseline_meta = spec.metadata.get("baseline")
    # Prefer 'active' (paradigm-neutral) over 'plateau' (experiment-specific)
    active_meta = spec.metadata.get("active") or spec.metadata.get("plateau")
    
    # Check if we at least have ONE valid window
    valid_any = any(m.valid for m in spec.metadata.values())
    errors: List[str] = []
    if not valid_any:
        errors.append("No valid time windows defined or found in data")

    if errors:
        if logger:
            logger_method = logger.error if strict else logger.warning
            logger_method("Time window validation failed: %s", "; ".join(errors))
        if strict:
            raise ValueError("; ".join(errors))

    # Build the generic masks and ranges dictionaries
    masks = {k: v for k, v in spec.masks.items()}
    ranges = {}
    for k, meta in spec.metadata.items():
        ranges[k] = (float(meta.start), float(meta.end))

    # Determine active mask/range - honor explicit named iteration first
    active_key = None
    if spec.name and spec.name in masks:
        active_key = spec.name
    else:
        for key in ["active", "plateau"]:
            if key in masks:
                active_key = key
                break
        if active_key is None:
            # Use first non-baseline window
            for key in masks:
                if key != "baseline":
                    active_key = key
                    break

    return TimeWindows(
        baseline_mask=spec.get_mask("baseline"),
        active_mask=spec.get_mask(active_key) if active_key else np.zeros_like(spec.times, dtype=bool),
        baseline_range=ranges.get("baseline", (np.nan, np.nan)),
        active_range=ranges.get(active_key, (np.nan, np.nan)) if active_key else (np.nan, np.nan),
        masks=masks,
        ranges=ranges,
        clamped=any(m.clamped for m in spec.metadata.values()),
        valid=not errors,
        errors=errors,
        times=spec.times,
        name=spec.name,
    )


###################################################################
# Pain Window Utilities (migrated from general.py)
###################################################################


def get_pain_window(constants=None, config: Optional[Any] = None) -> Tuple[float, float]:
    """Get the plateau/pain window from config or constants."""
    if config is not None:
        plateau_window = config.get("time_frequency_analysis.plateau_window")
        return tuple(plateau_window)
    
    if constants is None:
        raise ValueError("Either constants or config must be provided to get_pain_window")
    
    if "PLATEAU_WINDOW" not in constants:
        raise KeyError(
            "PLATEAU_WINDOW not found in constants. "
            "Use PLATEAU_WINDOW (tuple) not PLATEAU_END (float)"
        )
    
    return constants["PLATEAU_WINDOW"]


###################################################################
# Segment Masks for Feature Extraction
###################################################################


def get_segment_masks(
    times: np.ndarray,
    windows: Optional[TimeWindows],
    config: Optional[Any] = None,
) -> Dict[str, Optional[np.ndarray]]:
    """Return all named masks from the TimeWindows object."""
    if not windows:
        return {}

    # 1. Start with all generic masks
    out = {k: v for k, v in windows.masks.items()}

    # 2. Add baseline explicitly if not already in masks
    if windows.baseline_mask is not None and "baseline" not in out:
        if np.any(windows.baseline_mask):
            out["baseline"] = windows.baseline_mask

    # 3. Add active/plateau explicitly if not already in masks
    if windows.active_mask is not None:
        name = getattr(windows, "name", "active") or "active"
        if name not in out and np.any(windows.active_mask):
            out[name] = windows.active_mask

    # 4. Filter for only the targeted name if in a named iteration
    if windows.name:
        # We always keep baseline for potential normalization
        targeted = {"baseline": out.get("baseline")} if "baseline" in out else {}
        if windows.name in out:
            targeted[windows.name] = out[windows.name]
        return targeted

    return out


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
    if spec_times is not None and len(times) == len(spec_times) and hasattr(spec, "get_mask"):
        mask = spec.get_mask(window_name)
        if mask is not None and mask.shape == times.shape:
            return mask

    key = str(window_name).lower()

    # TimeWindows path: use explicit ranges when time vectors differ
    ranges = getattr(spec, "ranges", None)
    if isinstance(ranges, dict):
        window_range = ranges.get(key) or ranges.get(window_name)
        if window_range is None:
            if key in {"baseline", "pre", "prestim"}:
                window_range = getattr(spec, "baseline_range", None)
            elif key in {"active", "plateau", "stim", "task"}:
                window_range = getattr(spec, "active_range", None)
        if window_range is not None:
            start, end = window_range
            if np.isfinite(start) and np.isfinite(end) and end > start:
                return (times >= float(start)) & (times < float(end))

    # TimeWindowSpec path: use metadata if available
    meta = getattr(spec, "metadata", {}).get(window_name)
    if meta is None:
        return np.zeros_like(times, dtype=bool)

    start = float(getattr(meta, "start", np.nan))
    end = float(getattr(meta, "end", np.nan))
    if not np.isfinite(start) or not np.isfinite(end):
        return np.zeros_like(times, dtype=bool)

    return (times >= start) & (times < end)
