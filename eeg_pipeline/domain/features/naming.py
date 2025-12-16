"""
Feature Metadata and Naming (Canonical)
========================================

Single source of truth for feature naming conventions, metadata inference,
and manifest generation. All feature naming and metadata logic should import
from this module.

Provides:
- NamingSchema: Builder class for structured names (group.segment.band.scope.stat)
- FeatureName: Dataclass for parsed feature names
- FeatureMetadata: Dataclass for structured metadata
- DOMAINS, TIME_LABELS, STATISTICS: Standard constants
- make_power_name, make_erds_name, etc.: Domain-specific helpers
- infer_feature_metadata, generate_feature_sidecar: Legacy metadata inference
- generate_manifest, save_manifest: Manifest generation
- save_features_organized: I/O helpers
- get_fine_time_bins, get_coarse_time_bins: Temporal bin generators
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd


###################################################################
# Constants
###################################################################


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


###################################################################
# Data Classes
###################################################################


@dataclass
class FeatureName:
    """Parsed feature name components (legacy format)."""

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


@dataclass
class FeatureMetadata:
    """Metadata for a single feature (NamingSchema v2 format)."""

    name: str
    group: str
    segment: str
    band: str
    scope: str
    identifier: Optional[str] = None
    statistic: Optional[str] = None
    description: Optional[str] = None


###################################################################
# NamingSchema Class
###################################################################


class NamingSchema:
    """Helper to build standardized feature column names."""

    @staticmethod
    def build(
        group: str,
        segment: str,
        band: str,
        scope: str,
        stat: str,
        channel: Optional[str] = None,
        channel_pair: Optional[str] = None,
    ) -> str:
        parts = [group, segment, band, scope]

        if scope == "ch":
            if not channel:
                raise ValueError("Channel must be provided for scope='ch'")
            parts.append(channel)
        elif scope == "chpair":
            if not channel_pair:
                raise ValueError("Channel pair must be provided for scope='chpair'")
            parts.append(channel_pair)
        elif scope == "global":
            pass
        else:
            raise ValueError(f"Unknown scope: {scope}")

        parts.append(stat)
        return "_".join(parts)

    @staticmethod
    def parse(name: str) -> dict:
        parts = name.split("_")
        if len(parts) < 5:
            return {"valid": False}

        try:
            res = {
                "group": parts[0],
                "segment": parts[1],
                "band": parts[2],
                "scope": parts[3],
                "valid": True,
            }

            if res["scope"] == "global":
                res["stat"] = "_".join(parts[4:])
            elif res["scope"] in ["ch", "chpair"]:
                res["identifier"] = parts[4]
                res["stat"] = "_".join(parts[5:])

            return res
        except IndexError:
            return {"valid": False}


###################################################################
# Domain-Specific Helper Functions
###################################################################


def make_feature_name(
    domain: str,
    band: Optional[str] = None,
    location: Optional[str] = None,
    time: Optional[str] = None,
    stat: str = "mean",
    measure: Optional[str] = None,
) -> str:
    """Create a standardized feature name."""
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


def make_conn_name(
    measure: str, band: str, time: str, stat: str = "mean", location: str = "global"
) -> str:
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


###################################################################
# Time Bin Generators
###################################################################


def get_fine_time_bins(
    plateau_start: float = 3.0,
    plateau_end: float = 10.5,
    n_bins: int = 7,
) -> List[Dict[str, Any]]:
    """Generate fine temporal bins for HRF modeling."""
    duration = (plateau_end - plateau_start) / n_bins
    bins = []
    for i in range(n_bins):
        start = plateau_start + i * duration
        end = start + duration
        bins.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "label": f"t{i+1}",
            }
        )
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


###################################################################
# Simple Metadata Inference (Legacy API)
###################################################################


def infer_feature_metadata(column_name: str) -> Dict[str, str]:
    """Parse a feature column name to infer its metadata."""
    parts = column_name.split("_")
    meta = {"name": column_name}

    if not parts:
        return meta

    prefix = parts[0]

    if prefix == "pow":
        meta["type"] = "power"
        if len(parts) >= 3:
            meta["band"] = parts[1]
            meta["channel"] = "_".join(parts[2:])

    elif prefix == "conn":
        meta["type"] = "connectivity"
        if len(parts) >= 2:
            meta["metric"] = parts[1]
        if len(parts) >= 4:
            meta["band"] = parts[2]
            meta["channel_pair"] = "_".join(parts[3:])

    elif prefix == "ms":
        meta["type"] = "microstate"
        if len(parts) >= 2:
            meta["metric"] = parts[1]
        if len(parts) >= 3:
            if parts[1] == "transition":
                meta["transition_from"] = parts[2]
                if len(parts) >= 4:
                    meta["transition_to"] = parts[3]
            else:
                meta["state"] = parts[2]

    elif prefix == "aper":
        meta["type"] = "aperiodic"
        if len(parts) >= 2:
            meta["metric"] = parts[1]
        if len(parts) >= 3:
            meta["channel"] = "_".join(parts[2:])

    elif prefix == "pac":
        meta["type"] = "phase_amplitude_coupling"

    elif prefix == "erds":
        meta["type"] = "erds"

    else:
        meta["type"] = "other"

    return meta


def generate_feature_sidecar(
    feature_df_columns: List[str],
    description: str = "Extracted EEG Features",
    subject: Optional[str] = None,
    task: Optional[str] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a JSON sidecar structure for feature files."""
    features_meta = {}
    for col in feature_df_columns:
        features_meta[col] = infer_feature_metadata(col)

    sidecar = {
        "Description": description,
        "SourcePipeline": "eeg_pipeline",
        "Subject": subject,
        "Task": task,
        "FeatureCount": len(feature_df_columns),
        "Features": features_meta,
    }

    if additional_metadata:
        sidecar.update(additional_metadata)

    return sidecar


###################################################################
# Structured Metadata Parsing (NamingSchema v2)
###################################################################


def parse_feature_name(name: str) -> FeatureMetadata:
    """Parse a feature name (NamingSchema v2)."""
    parts = name.split("_")

    if len(parts) < 5:
        return FeatureMetadata(
            name=name,
            group="unknown",
            segment="unknown",
            band="unknown",
            scope="unknown",
        )

    group = parts[0]
    segment = parts[1]
    band = parts[2]
    scope = parts[3]

    if scope == "global":
        identifier = None
        stat = "_".join(parts[4:])
    elif scope in ("ch", "chpair"):
        if len(parts) >= 6:
            identifier = parts[4]
            stat = "_".join(parts[5:])
        else:
            identifier = parts[4] if len(parts) > 4 else "unknown"
            stat = "unknown"
    else:
        group = parts[0]
        stat = parts[-1]
        return FeatureMetadata(
            name,
            group=group,
            segment="unknown",
            band="unknown",
            scope="unknown",
            statistic=stat,
        )

    return FeatureMetadata(
        name=name,
        group=group,
        segment=segment,
        band=band,
        scope=scope,
        identifier=identifier,
        statistic=stat,
        description=f"{group} feature on {segment} in {band} band ({scope if not identifier else identifier})",
    )


def parse_legacy_power_feature_name(name: str) -> Optional[Tuple[str, str]]:
    if not isinstance(name, str):
        return None

    if name.startswith("pow_"):
        parts = name[4:].split("_")
        if len(parts) < 2:
            return None
        band = parts[0]
        channel = parts[1]
        return band, channel

    if name.startswith("power_") and "_ch_" in name:
        parts = name.split("_")
        if len(parts) < 6:
            return None
        band = parts[2]
        if parts[3] != "ch":
            return None
        remainder = name.split("_ch_", 1)[-1]
        if "_" not in remainder:
            return None
        channel = remainder.rsplit("_", 1)[0]
        return band, channel

    return None


###################################################################
# Manifest Generation (Canonical)
###################################################################


def generate_manifest(
    feature_columns: List[str],
    config: Any = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
    qc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    features = []
    for col in feature_columns:
        name = str(col)
        parsed = NamingSchema.parse(name)
        if parsed.get("valid"):
            features.append(
                {
                    "name": name,
                    "group": parsed.get("group"),
                    "segment": parsed.get("segment"),
                    "band": parsed.get("band"),
                    "scope": parsed.get("scope"),
                    "identifier": parsed.get("identifier"),
                    "statistic": parsed.get("stat"),
                }
            )
        else:
            legacy = infer_feature_metadata(name)
            features.append(
                {
                    "name": name,
                    "group": legacy.get("type", "unknown"),
                    "segment": legacy.get("time", "unknown"),
                    "band": legacy.get("band", "unknown"),
                    "scope": "unknown",
                    "identifier": legacy.get("channel") or legacy.get("channel_pair"),
                    "statistic": legacy.get("statistic"),
                }
            )

    return {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "subject": subject,
        "task": task,
        "n_features": len(features),
        "features": features,
        "qc": qc or None,
        "config": None if config is None else {},
    }


def save_manifest(manifest: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def save_features_organized(
    df: pd.DataFrame,
    output_dir: Path,
    subject: str,
    task: str,
    *,
    config: Any = None,
    qc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    sub_dir = output_dir / f"sub-{subject}" / "eeg" / "features"
    sub_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"sub-{subject}_task-{task}"
    features_path = sub_dir / f"{base_name}_features.tsv"
    df.to_csv(features_path, sep="\t", index=False)

    feature_cols = [c for c in df.columns if c not in ["condition", "trial", "epoch", "subject"]]
    manifest = generate_manifest(feature_cols, config=config, subject=subject, task=task, qc=qc)
    manifest_path = sub_dir / f"{base_name}_features_manifest.json"
    save_manifest(manifest, manifest_path)

    return {
        "features": features_path,
        "manifest": manifest_path,
    }


###################################################################
# Exports
###################################################################


__all__ = [
    "DOMAINS",
    "TIME_LABELS",
    "STATISTICS",
    "FeatureName",
    "FeatureMetadata",
    "NamingSchema",
    "make_feature_name",
    "make_power_name",
    "make_erds_name",
    "make_conn_name",
    "make_phase_name",
    "make_aper_name",
    "make_spec_name",
    "make_ms_name",
    "make_comp_name",
    "make_temp_name",
    "make_asym_name",
    "make_gfp_name",
    "make_roi_name",
    "get_fine_time_bins",
    "get_coarse_time_bins",
    "get_all_time_bins",
    "infer_feature_metadata",
    "generate_feature_sidecar",
    "parse_feature_name",
    "parse_legacy_power_feature_name",
    "generate_manifest",
    "save_manifest",
    "save_features_organized",
]
