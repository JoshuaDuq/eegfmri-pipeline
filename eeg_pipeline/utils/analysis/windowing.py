"""
Windowing Utilities (Canonical)
===============================

Single source of truth for time/frequency masking and window computation.
All windowing logic should import from this module.

Provides:
- time_mask, freq_mask: Basic masking functions
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
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
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
        self.tmin = tmin
        self.tmax = tmax
        
        self.masks: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, WindowMetadata] = {}
        self.errors: List[str] = []
        
        self._build_all_windows()
        
    def _build_all_windows(self):
        """Construct windows from explicit user input or tmin/tmax fallback.
        
        Priority:
        1. Explicit windows from TUI (Step 5)
        2. Auto-generated window from tmin/tmax CLI arguments
        3. Full epoch window if nothing else is specified
        """
        explicit_by_name = {k.lower(): v for k, v in self._parse_explicit_windows().items()}
        
        if explicit_by_name:
            self._build_from_explicit_windows(explicit_by_name)
        elif self.tmin is not None or self.tmax is not None:
            self._build_from_tmin_tmax()
        else:
            self._build_full_epoch_window()

    def _build_full_epoch_window(self) -> None:
        """Build a default full-epoch analysis window when no other window is defined."""
        window_name = self.name if self.name else "analysis"
        full_mask = np.ones_like(self.times, dtype=bool)
        start = float(self.times[0])
        end = float(self._time_axis_upper_bound())

        self.masks[window_name] = full_mask
        self.metadata[window_name] = WindowMetadata(
            start=start,
            end=end,
            clamped=False,
            n_samples=int(full_mask.sum()),
            valid=True,
            coverage=1.0,
        )

        if self.logger:
            self.logger.info(
                "Auto-generating default full-epoch window '%s': [%.3f, %.3f]",
                window_name,
                start,
                end,
            )
    
    def _build_from_tmin_tmax(self):
        """Auto-generate a window from CLI tmin/tmax when no explicit windows defined.
        
        This prevents silent zero-masking when users specify tmin/tmax without
        explicit window definitions.
        """
        t_start = self.tmin if self.tmin is not None else float(self.times[0])
        t_end = self.tmax if self.tmax is not None else float(self.times[-1])
        
        window_name = self.name if self.name else "analysis"
        
        self.logger.info(
            f"Auto-generating window '{window_name}' from tmin/tmax: [{t_start:.3f}, {t_end:.3f}]"
        )
        self._add_window(window_name, t_start, t_end)
        
        if t_start > 0:
            baseline_end = min(0.0, t_start)
            baseline_start = self.times[0]
            if baseline_start < baseline_end:
                self._add_window("baseline", float(baseline_start), baseline_end)
    
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
    
    def _build_from_explicit_windows(self, explicit_by_name: Dict[str, Tuple[float, float]]):
        """Build windows from user-defined explicit windows only.
        
        When self.name is set, ONLY build that specific window to ensure
        output files contain only data for their designated time window.
        However, always store baseline metadata for ERP baseline correction.
        """
        if self.name:
            # Only build the targeted window when name is specified
            name_key = str(self.name).strip().lower()
            if name_key in explicit_by_name:
                start, end = explicit_by_name[name_key]
                self._add_window(self.name, float(start), float(end))
            else:
                self._add_empty_window(self.name, reason="window_not_in_user_input")
            
            # Always store baseline range for ERP baseline correction
            if "baseline" in explicit_by_name and name_key != "baseline":
                bl_start, bl_end = explicit_by_name["baseline"]
                self._add_window("baseline", float(bl_start), float(bl_end))
            
            return  # Don't build other windows when targeting a specific one
        
        # Only build all windows when no specific target is set
        for win_name, (start, end) in explicit_by_name.items():
            if win_name in self.masks:
                continue
            self._add_window(win_name, float(start), float(end))
    
    def _add_window(self, name: str, start: float, end: float, prefix: str = ""):
        """Add a window with clamping and validation."""
        full_name = f"{prefix}_{name}" if prefix else name
        
        time_min_available = self.times[0]
        time_max_available = self.times[-1]
        time_upper_bound = self._time_axis_upper_bound()
        
        final_start, final_end, was_clamped = self._clamp_window_bounds(
            start, end, time_min_available, time_upper_bound
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
            is_targeted_window = self.name and full_name.lower() == self.name.lower()
            if is_targeted_window:
                self.logger.warning(
                    f"Window '{full_name}' is empty! req=[{start:.2f}, {end:.2f}]"
                )
                self.errors.append(full_name)
            else:
                self.logger.debug(
                    f"Window '{full_name}' outside current range (expected): "
                f"req=[{start:.2f}, {end:.2f}], available=[{time_min_available:.2f}, {time_max_available:.2f}]"
                )
        
        self.masks[full_name] = mask
        self.metadata[full_name] = WindowMetadata(
            start=final_start,
            end=final_end,
            clamped=was_clamped,
            n_samples=n_samples,
            valid=is_valid,
            coverage=coverage,
        )

    def _time_axis_upper_bound(self) -> float:
        """Return the exclusive upper bound for half-open masks on this time axis."""
        if self.times.size <= 1:
            sample_period = 1.0 / self.sfreq if np.isfinite(self.sfreq) and self.sfreq > 0 else 1e-9
            return float(self.times[-1]) + float(sample_period)

        diffs = np.diff(self.times.astype(float))
        positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if positive_diffs.size == 0:
            sample_period = 1.0 / self.sfreq if np.isfinite(self.sfreq) and self.sfreq > 0 else 1e-9
        else:
            sample_period = float(np.median(positive_diffs))
        return float(self.times[-1]) + float(sample_period)
    
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
        mask = self.masks.get(name)
        if mask is not None:
            return mask
        key = str(name).strip().lower()
        mask = self.masks.get(key)
        if mask is not None:
            return mask
        return np.zeros_like(self.times, dtype=bool)
        
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

    all_masks = dict(windows.masks)
    target_name = getattr(windows, "name", None)
    if target_name and target_name not in all_masks:
        for name, mask in all_masks.items():
            if str(name).lower() == str(target_name).lower():
                all_masks[str(target_name)] = mask
                break

    return all_masks


def make_mask_for_times(spec: Any, window_name: str, times: np.ndarray) -> np.ndarray:
    """Get a window mask aligned to an arbitrary time vector.

    Use this when downstream computations have a different time axis than the
    one used to build `spec` (e.g., decimated TFR time points).
    """
    if spec is None or times is None:
        return np.zeros_like(times, dtype=bool)

    # Try exact match first
    window_range = _get_window_range_from_spec(spec, window_name)
    if window_range is not None:
        start, end = window_range
        if np.isfinite(start) and np.isfinite(end) and end > start:
            return (times >= float(start)) & (times < float(end))

    return np.zeros_like(times, dtype=bool)


def _get_window_range_from_spec(spec: Any, window_name: str) -> Optional[Tuple[float, float]]:
    """Extract window range from spec, strictly by name."""
    key = str(window_name).strip().lower()
    
    ranges = getattr(spec, "ranges", None)
    if isinstance(ranges, dict):
        if window_name in ranges:
            return ranges[window_name]
        if key in ranges:
            return ranges[key]
        
        # Internal field fallbacks (only if name matches exactly)
        if key == "baseline":
            return getattr(spec, "baseline_range", None)
        if key == "active":
            return getattr(spec, "active_range", None)

    metadata = getattr(spec, "metadata", {})
    meta = metadata.get(window_name) or metadata.get(key)
    if meta is not None:
        start = float(getattr(meta, "start", np.nan))
        end = float(getattr(meta, "end", np.nan))
        if np.isfinite(start) and np.isfinite(end):
            return (start, end)
    
    return None
