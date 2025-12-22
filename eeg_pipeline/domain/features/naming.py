"""
Feature Metadata and Naming (Canonical)
========================================

Single source of truth for feature naming conventions, metadata inference,
and manifest generation. All feature naming and metadata logic should import
from this module.

Provides:
- NamingSchema: Builder class for structured names (group.segment.band.scope.stat)
- FeatureMetadata: Dataclass for structured metadata
- DOMAINS, TIME_LABELS, STATISTICS: Standard constants
- generate_manifest, save_manifest: Manifest generation
- save_features_organized: I/O helpers
- get_fine_time_bins, get_coarse_time_bins: Temporal bin generators
- parse_feature_name: Parse NamingSchema v2 format
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd


###################################################################
# Constants
###################################################################


DOMAINS = {
    "power": "power",         # Band power
    "spectral": "spectral",   # Peak frequency, IAF
    "aper": "aper",           # Aperiodic 1/f
    "erp": "erp",             # ERP/LEP features
    "erds": "erds",           # Event-related desync/sync
    "ratio": "ratio",         # Band power ratios
    "asym": "asym",           # Hemispheric asymmetry
    "conn": "conn",           # Connectivity
    "itpc": "itpc",           # Inter-trial phase coherence
    "pac": "pac",             # Phase-amplitude coupling
    "comp": "comp",           # Complexity
    "bursts": "bursts",       # Burst dynamics
    "qual": "qual",           # Quality metrics
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
    "latency": "latency",
    "latency_diff": "latency_diff",
    "duration": "duration",
    "auc": "auc",
    "count": "count",
    "ptp": "ptp",
}


###################################################################
# Data Classes
###################################################################


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
        elif scope == "roi":
            if not channel:
                raise ValueError("ROI name must be provided for scope='roi'")
            parts.append(channel)  # ROI name passed via channel parameter
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
            elif res["scope"] in ["ch", "chpair", "roi"]:
                res["identifier"] = parts[4]
                res["stat"] = "_".join(parts[5:])

            return res
        except IndexError:
            return {"valid": False}


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
# Structured Metadata Parsing
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
    elif scope in ("ch", "chpair", "roi"):
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


###################################################################
# Manifest Generation (Canonical)
###################################################################


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy arrays and other non-JSON types to serializable forms."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    return obj


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
            features.append(
                {
                    "name": name,
                    "group": "unknown",
                    "segment": "unknown",
                    "band": "unknown",
                    "scope": "unknown",
                    "identifier": None,
                    "statistic": None,
                }
            )

    return {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "subject": subject,
        "task": task,
        "n_features": len(features),
        "features": features,
        "qc": _make_json_serializable(qc) if qc else None,
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
    "FeatureMetadata",
    "NamingSchema",
    "get_fine_time_bins",
    "get_coarse_time_bins",
    "get_all_time_bins",
    "parse_feature_name",
    "generate_manifest",
    "save_manifest",
    "save_features_organized",
]
