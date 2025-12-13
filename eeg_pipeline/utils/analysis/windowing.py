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


def compute_time_windows(
    times: np.ndarray,
    config: Any,
    n_plateau_windows: int = 5,
    *,
    logger: Optional[logging.Logger] = None,
    strict: bool = True,
) -> TimeWindows:
    """
    Compute all time window masks once.
    
    Creates both coarse (early/mid/late) and fine (t1-t7) temporal bins
    based on config settings.
    """
    errors: List[str] = []
    tf_cfg = config.get("time_frequency_analysis", {})
    baseline_window = tf_cfg.get("baseline_window", [-5.0, -0.01])
    fe_cfg = config.get("feature_engineering.features", {})
    plateau_window = tf_cfg.get("plateau_window", [3.0, 10.5])
    available_range = (
        (float(times[0]), float(times[-1])) if times.size > 0 else (np.nan, np.nan)
    )

    def _build_mask(start: float, end: float, label: str) -> Tuple[np.ndarray, Tuple[float, float], bool]:
        mask = time_mask(times, start, end)
        if np.any(mask):
            return mask, (start, end), False

        # Clamp to available time range to avoid empty masks
        if times.size == 0:
            if logger:
                logger.warning("Time vector is empty; mask '%s' will be empty.", label)
            return np.zeros_like(times, dtype=bool), (start, end), True

        clamped_start = max(start, float(times[0]))
        clamped_end = min(end, float(times[-1]))
        clamped = time_mask(times, clamped_start, clamped_end)
        if np.any(clamped):
            if logger:
                logger.warning(
                    "Clamped %s window from [%.3f, %.3f] to available range [%.3f, %.3f].",
                    label,
                    start,
                    end,
                    clamped_start,
                    clamped_end,
                )
            return clamped, (clamped_start, clamped_end), True

        if logger:
            logger.warning(
                "No samples found for %s window [%.3f, %.3f]; features using this window will be NaN.",
                label,
                start,
                end,
            )
        return clamped, (clamped_start, clamped_end), True

    baseline_mask, (baseline_start, baseline_end), baseline_clamped = _build_mask(
        baseline_window[0], baseline_window[1], "baseline"
    )
    active_mask, (plateau_start, plateau_end), active_clamped = _build_mask(
        plateau_window[0], plateau_window[1], "plateau"
    )
    clamped_any = baseline_clamped or active_clamped
    if not np.any(baseline_mask):
        errors.append(
            f"Baseline window [{baseline_start:.3f}, {baseline_end:.3f}] is empty; "
            f"available time range: [{available_range[0]:.3f}, {available_range[1]:.3f}]"
        )
    if not np.any(active_mask):
        errors.append(
            f"Active/plateau window [{plateau_start:.3f}, {plateau_end:.3f}] is empty; "
            f"available time range: [{available_range[0]:.3f}, {available_range[1]:.3f}]"
        )
    
    # Coarse temporal bins from config
    coarse_bins = fe_cfg.get("temporal_bins", [
        {"start": 3.0, "end": 5.0, "label": "early"},
        {"start": 5.0, "end": 7.5, "label": "mid"},
        {"start": 7.5, "end": 10.5, "label": "late"},
    ])
    coarse_masks = []
    coarse_labels = []
    for bin_def in coarse_bins:
        mask, _, _ = _build_mask(bin_def["start"], bin_def["end"], f"coarse-{bin_def['label']}")
        coarse_masks.append(mask)
        coarse_labels.append(bin_def["label"])
    
    # Fine temporal bins from config (for HRF modeling)
    fine_masks = []
    fine_labels = []
    use_fine = fe_cfg.get("use_fine_temporal_bins", True)
    if use_fine:
        fine_bins = fe_cfg.get("temporal_bins_fine", [])
        if not fine_bins:
            # Generate default fine bins (7 bins of ~1s each)
            n_fine = 7
            duration = (plateau_end - plateau_start) / n_fine
            fine_bins = [
                {"start": plateau_start + i * duration, 
                 "end": plateau_start + (i + 1) * duration, 
                 "label": f"t{i+1}"}
                for i in range(n_fine)
            ]
        for bin_def in fine_bins:
            label = bin_def["label"]
            mask, _, _ = _build_mask(bin_def["start"], bin_def["end"], f"fine-{label}")
            fine_masks.append(mask)
            fine_labels.append(label)
    
    # Legacy plateau windows (for backward compatibility)
    plateau_masks = []
    window_labels = []
    if n_plateau_windows > 0:
        window_duration = (plateau_end - plateau_start) / n_plateau_windows
        for i in range(n_plateau_windows):
            win_start = plateau_start + i * window_duration
            win_end = win_start + window_duration
            mask, _, _ = _build_mask(win_start, win_end, f"plateau-w{i}")
            plateau_masks.append(mask)
            window_labels.append(f"w{i}")
    
    # Explicitly warn when masks are empty after clamping so downstream code can bail early
    if errors:
        if logger:
            logger_method = logger.error if strict else logger.warning
            logger_method("Time window validation failed: %s", "; ".join(errors))
        if strict:
            raise ValueError("; ".join(errors))

    return TimeWindows(
        baseline_mask=baseline_mask,
        active_mask=active_mask,
        baseline_range=(baseline_start, baseline_end),
        active_range=(plateau_start, plateau_end),
        clamped=clamped_any,
        valid=not errors,
        errors=errors,
        coarse_masks=coarse_masks,
        coarse_labels=coarse_labels,
        fine_masks=fine_masks,
        fine_labels=fine_labels,
        plateau_masks=plateau_masks,
        window_labels=window_labels,
        times=times,
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
        logger: Optional[logging.Logger] = None
    ):
        self.times = times
        self.sfreq = sampling_rate
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.masks: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, WindowMetadata] = {}
        self.errors: List[str] = []
        
        self._build_all_windows()
        
    def _build_all_windows(self):
        """Construct all standard windows defined in config."""
        tf_cfg = self.config.get("time_frequency_analysis", {})
        feat_cfg = self.config.get("feature_engineering.windows", {})
        feat_features_cfg = self.config.get("feature_engineering.features", {})
        
        baseline_def = feat_cfg.get("baseline_window", tf_cfg.get("baseline_window", [-2.0, 0.0]))
        plateau_def = feat_cfg.get(
            "plateau_window",
            feat_features_cfg.get("plateau_window", tf_cfg.get("plateau_window", [3.0, 10.0])),
        )
        
        self._add_window("baseline", baseline_def[0], baseline_def[1])
        
        plateau_start = plateau_def[0]
        plateau_end = plateau_def[1]
        
        if "ramp_window" in feat_cfg:
            ramp_def = feat_cfg["ramp_window"]
            self._add_window("ramp", ramp_def[0], ramp_def[1])
        else:
            ramp_end = feat_features_cfg.get("ramp_end")
            if ramp_end is not None:
                try:
                    ramp_end = float(ramp_end)
                except Exception:
                    ramp_end = None
            self._add_window("ramp", 0.0, ramp_end if ramp_end is not None else plateau_start)
             
        self._add_window("plateau", plateau_start, plateau_end)
        
        # Coarse bins (early/mid/late)
        coarse_bins = feat_cfg.get("coarse_bins")
        if not coarse_bins:
            coarse_bins = feat_features_cfg.get("temporal_bins")
        if not coarse_bins:
            duration = plateau_end - plateau_start
            bin_size = duration / 3.0
            coarse_bins = [
                {"label": "early", "start": plateau_start, "end": plateau_start + bin_size},
                {"label": "mid", "start": plateau_start + bin_size, "end": plateau_start + 2*bin_size},
                {"label": "late", "start": plateau_start + 2*bin_size, "end": plateau_end},
            ]
            
        for b in coarse_bins:
            self._add_window(b["label"], b["start"], b["end"], prefix="coarse")
            
        # Fine bins (t1..tN)
        fine_bins = feat_features_cfg.get("temporal_bins_fine")
        if fine_bins:
            for entry in fine_bins:
                if not isinstance(entry, dict):
                    continue
                if "label" not in entry or "start" not in entry or "end" not in entry:
                    continue
                try:
                    label = str(entry["label"])
                    start = float(entry["start"])
                    end = float(entry["end"])
                except Exception:
                    continue
                self._add_window(label, start, end, prefix="fine")
        else:
            n_fine = int(feat_cfg.get("n_fine_bins", 7))
            fine_dur = (plateau_end - plateau_start) / max(n_fine, 1)
            for i in range(n_fine):
                start = plateau_start + i * fine_dur
                end = start + fine_dur
                self._add_window(f"t{i+1}", start, end, prefix="fine")

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
    n_plateau_windows: int = 5,
    *,
    logger: Optional[logging.Logger] = None,
    strict: bool = True,
) -> TimeWindows:
    baseline_meta = spec.metadata.get("baseline")
    plateau_meta = spec.metadata.get("plateau")

    errors: List[str] = []
    if baseline_meta is None or not bool(getattr(baseline_meta, "valid", False)):
        errors.append("Baseline window is empty")
    if plateau_meta is None or not bool(getattr(plateau_meta, "valid", False)):
        errors.append("Active/plateau window is empty")

    if errors:
        if logger:
            logger_method = logger.error if strict else logger.warning
            logger_method("Time window validation failed: %s", "; ".join(errors))
        if strict:
            raise ValueError("; ".join(errors))

    coarse_masks: List[np.ndarray] = []
    coarse_labels: List[str] = []
    fine_masks: List[np.ndarray] = []
    fine_labels: List[str] = []
    for key, mask in spec.masks.items():
        if key.startswith("coarse_"):
            coarse_labels.append(key.split("coarse_", 1)[1])
            coarse_masks.append(mask)
        elif key.startswith("fine_"):
            fine_labels.append(key.split("fine_", 1)[1])
            fine_masks.append(mask)

    plateau_masks: List[np.ndarray] = []
    window_labels: List[str] = []
    if n_plateau_windows > 0 and plateau_meta is not None and bool(getattr(plateau_meta, "valid", False)):
        plateau_start = float(getattr(plateau_meta, "start", np.nan))
        plateau_end = float(getattr(plateau_meta, "end", np.nan))
        window_duration = (plateau_end - plateau_start) / n_plateau_windows
        for i in range(n_plateau_windows):
            win_start = plateau_start + i * window_duration
            win_end = win_start + window_duration
            plateau_masks.append((spec.times >= win_start) & (spec.times < win_end))
            window_labels.append(f"w{i}")

    return TimeWindows(
        baseline_mask=spec.get_mask("baseline"),
        active_mask=spec.get_mask("plateau"),
        baseline_range=(
            (float(baseline_meta.start), float(baseline_meta.end))
            if baseline_meta is not None
            else (np.nan, np.nan)
        ),
        active_range=(
            (float(plateau_meta.start), float(plateau_meta.end))
            if plateau_meta is not None
            else (np.nan, np.nan)
        ),
        clamped=bool(
            (baseline_meta is not None and bool(getattr(baseline_meta, "clamped", False)))
            or (plateau_meta is not None and bool(getattr(plateau_meta, "clamped", False)))
        ),
        valid=not errors,
        errors=errors,
        coarse_masks=coarse_masks,
        coarse_labels=coarse_labels,
        fine_masks=fine_masks,
        fine_labels=fine_labels,
        plateau_masks=plateau_masks,
        window_labels=window_labels,
        times=spec.times,
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


WINDOW_PAIN = get_pain_window


###################################################################
# Segment Masks for Feature Extraction
###################################################################


def get_segment_masks(
    times: np.ndarray,
    windows: Optional[TimeWindows],
    config: Optional[Any] = None,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Derive ramp/plateau/offset/baseline masks based on times and config.
    
    This is the canonical implementation - import from here instead of
    duplicating in dynamics.py or connectivity.py.
    
    Args:
        times: Time array from epochs/precomputed data
        windows: TimeWindows object with active_mask, baseline_mask, etc.
        config: Configuration dict for ramp_end, offset_start values
        
    Returns:
        Dict with keys 'ramp', 'plateau', 'baseline', 'offset' mapping to boolean masks
    """
    from eeg_pipeline.utils.config.loader import get_config_value
    
    cfg = config or {}
    ramp_end = float(get_config_value(cfg, "feature_engineering.features.ramp_end", 3.0))
    offset_start = get_config_value(cfg, "feature_engineering.features.offset_start", None)

    ramp_mask = (times >= 0) & (times <= ramp_end)
    plateau_mask = getattr(windows, "active_mask", None) if windows else None
    baseline_mask = getattr(windows, "baseline_mask", None) if windows else None
    
    offset_mask = None
    if offset_start is not None:
        try:
            offset_start_f = float(offset_start)
            if offset_start_f < times[-1]:
                offset_mask = times >= offset_start_f
        except Exception:
            offset_mask = None

    return {
        "ramp": ramp_mask,
        "plateau": plateau_mask,
        "baseline": baseline_mask,
        "offset": offset_mask,
    }
