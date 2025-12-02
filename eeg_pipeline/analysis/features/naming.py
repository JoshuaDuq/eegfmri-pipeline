"""
Feature Naming Utilities
========================

Standardized naming convention for all EEG features:
    {domain}_{measure}_{band}_{location}_{time}_{statistic}

Examples:
    power_alpha_Cz_early_mean
    erds_beta_Sensorimotor_Contra_mid_percent
    conn_wpli_theta_global_late_mean
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import re


# Domain prefixes
DOMAINS = {
    "power": "power",
    "erds": "erds",
    "conn": "conn",
    "phase": "phase",
    "pac": "pac",
    "aper": "aper",
    "spec": "spec",
    "ms": "ms",
    "comp": "comp",
    "temp": "temp",
    "asym": "asym",
    "gfp": "gfp",
    "roi": "roi",
}

# Standard time window labels
TIME_LABELS = {
    "baseline": "baseline",
    "early": "early",
    "mid": "mid",
    "late": "late",
    "full": "full",
    "t1": "t1",
    "t2": "t2",
    "t3": "t3",
    "t4": "t4",
    "t5": "t5",
    "t6": "t6",
    "t7": "t7",
}

# Standard statistics
STATISTICS = {
    "mean": "mean",
    "std": "std",
    "max": "max",
    "min": "min",
    "cv": "cv",
    "zscore": "zscore",
    "percent": "percent",
    "logratio": "logratio",
    "slope": "slope",
    "diff": "diff",
    "onset": "onset",
    "peak": "peak",
    "duration": "duration",
    "auc": "auc",
    "count": "count",
}


@dataclass
class FeatureName:
    """Parsed feature name components."""
    domain: str
    measure: Optional[str] = None
    band: Optional[str] = None
    location: Optional[str] = None
    time: Optional[str] = None
    statistic: Optional[str] = None
    
    def to_string(self) -> str:
        """Convert to standardized feature name string."""
        parts = [self.domain]
        if self.measure:
            parts.append(self.measure)
        if self.band:
            parts.append(self.band)
        if self.location:
            parts.append(self.location)
        if self.time:
            parts.append(self.time)
        if self.statistic:
            parts.append(self.statistic)
        return "_".join(parts)


def make_feature_name(
    domain: str,
    band: Optional[str] = None,
    location: Optional[str] = None,
    time: Optional[str] = None,
    stat: str = "mean",
    measure: Optional[str] = None,
) -> str:
    """
    Create a standardized feature name.
    
    Parameters
    ----------
    domain : str
        Feature domain (power, erds, conn, phase, etc.)
    band : str, optional
        Frequency band (alpha, beta, etc.)
    location : str, optional
        Channel name, ROI name, or 'global'
    time : str, optional
        Time window label (early, mid, late, t1-t7, etc.)
    stat : str
        Statistic type (mean, std, percent, etc.)
    measure : str, optional
        Measure type (wpli, plv, itpc, etc.) - used for connectivity/phase
    
    Returns
    -------
    str
        Standardized feature name
    """
    parts = [domain]
    if measure:
        parts.append(measure)
    if band:
        parts.append(band)
    if location:
        parts.append(location)
    if time:
        parts.append(time)
    if stat:
        parts.append(stat)
    return "_".join(parts)


def make_power_name(band: str, channel: str, time: str, stat: str = "mean") -> str:
    """Create power feature name."""
    return f"power_{band}_{channel}_{time}_{stat}"


def make_erds_name(band: str, channel: str, time: str, stat: str = "percent") -> str:
    """Create ERD/ERS feature name."""
    return f"erds_{band}_{channel}_{time}_{stat}"


def make_conn_name(measure: str, band: str, time: str, stat: str = "mean", location: str = "global") -> str:
    """Create connectivity feature name."""
    return f"conn_{measure}_{band}_{location}_{time}_{stat}"


def make_phase_name(measure: str, band: str, channel: str, time: str, stat: str = "mean") -> str:
    """Create phase feature name."""
    return f"phase_{measure}_{band}_{channel}_{time}_{stat}"


def make_aper_name(measure: str, channel: str, time: str = "baseline") -> str:
    """Create aperiodic feature name."""
    return f"aper_{measure}_{channel}_{time}"


def make_spec_name(measure: str, band: str, channel: str) -> str:
    """Create spectral shape feature name."""
    return f"spec_{measure}_{band}_{channel}"


def make_ms_name(measure: str, state: str, time: str = "full") -> str:
    """Create microstate feature name."""
    return f"ms_{measure}_{state}_{time}"


def make_comp_name(measure: str, band: str, channel: str, time: str) -> str:
    """Create complexity feature name."""
    return f"comp_{measure}_{band}_{channel}_{time}"


def make_temp_name(measure: str, band: str, channel: str, time: str) -> str:
    """Create temporal feature name."""
    return f"temp_{measure}_{band}_{channel}_{time}"


def make_asym_name(band: str, pair: str, time: str, stat: str = "index") -> str:
    """Create asymmetry feature name."""
    return f"asym_{stat}_{band}_{pair}_{time}"


def make_gfp_name(band: str, time: str, stat: str = "mean") -> str:
    """Create GFP feature name."""
    if band:
        return f"gfp_{band}_{time}_{stat}"
    return f"gfp_{time}_{stat}"


def make_roi_name(domain: str, band: str, roi: str, time: str, stat: str = "mean") -> str:
    """Create ROI-aggregated feature name."""
    return f"roi_{domain}_{band}_{roi}_{time}_{stat}"


def get_fine_time_bins(
    plateau_start: float = 3.0,
    plateau_end: float = 10.5,
    n_bins: int = 7,
) -> List[Dict[str, Any]]:
    """
    Generate fine temporal bins for HRF modeling.
    
    Returns list of dicts with 'start', 'end', 'label' keys.
    """
    duration = (plateau_end - plateau_start) / n_bins
    bins = []
    for i in range(n_bins):
        start = plateau_start + i * duration
        end = start + duration
        bins.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "label": f"t{i+1}",
        })
    return bins


def get_coarse_time_bins() -> List[Dict[str, Any]]:
    """Get standard coarse temporal bins (early, mid, late)."""
    return [
        {"start": 3.0, "end": 5.0, "label": "early"},
        {"start": 5.0, "end": 7.5, "label": "mid"},
        {"start": 7.5, "end": 10.5, "label": "late"},
    ]


def get_all_time_bins(include_fine: bool = True) -> List[Dict[str, Any]]:
    """Get all temporal bins (coarse + optionally fine)."""
    bins = get_coarse_time_bins()
    if include_fine:
        bins.extend(get_fine_time_bins())
    return bins


