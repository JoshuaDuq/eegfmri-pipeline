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
    "power": "power",
    "spectral": "spectral",
    "aper": "aper",
    "erp": "erp",
    "erds": "erds",
    "ratio": "ratio",
    "asym": "asym",
    "conn": "conn",
    "itpc": "itpc",
    "pac": "pac",
    "comp": "comp",
    "bursts": "bursts",
    "qual": "qual",
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
        if scope not in ("ch", "chpair", "roi", "global"):
            raise ValueError(f"Unknown scope: {scope}")

        parts = [group, segment, band, scope]

        if scope == "ch":
            if channel is None:
                raise ValueError("Channel must be provided for scope='ch'")
            parts.append(channel)
        elif scope == "roi":
            if channel is None:
                raise ValueError("ROI name must be provided for scope='roi'")
            parts.append(channel)
        elif scope == "chpair":
            if channel_pair is None:
                raise ValueError("Channel pair must be provided for scope='chpair'")
            parts.append(channel_pair)

        parts.append(stat)
        return "_".join(parts)

    @staticmethod
    def parse(name: str) -> dict:
        parts = name.split("_")
        if len(parts) < 5:
            return {"valid": False}

        scope_tokens = {"global", "ch", "chpair", "roi"}
        scope_idx = None
        for idx in range(2, len(parts)):
            if parts[idx] in scope_tokens:
                scope_idx = idx
                break

        if scope_idx is None:
            return {"valid": False}

        group = parts[0]
        segment = parts[1]
        band = "_".join(parts[2:scope_idx])
        scope = parts[scope_idx]

        result = {
            "group": group,
            "segment": segment,
            "band": band,
            "scope": scope,
            "valid": True,
        }

        if scope == "global":
            result["stat"] = "_".join(parts[scope_idx + 1:])
        elif scope in ("ch", "chpair", "roi"):
            if scope_idx + 1 >= len(parts):
                return {"valid": False}
            result["identifier"] = parts[scope_idx + 1]
            result["stat"] = "_".join(parts[scope_idx + 2:])
        else:
            return {"valid": False}

        return result


###################################################################
# Time Bin Generators
###################################################################


def get_fine_time_bins(
    active_start: float = 3.0,
    active_end: float = 10.5,
    n_bins: int = 7,
) -> List[Dict[str, Any]]:
    """Generate fine temporal bins for HRF modeling."""
    if active_end <= active_start:
        raise ValueError("active_end must be greater than active_start")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    bin_duration = (active_end - active_start) / n_bins
    bins = []
    for bin_index in range(n_bins):
        bin_start = active_start + bin_index * bin_duration
        bin_end = bin_start + bin_duration
        bins.append(
            {
                "start": round(bin_start, 2),
                "end": round(bin_end, 2),
                "label": f"t{bin_index + 1}",
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
        statistic = "_".join(parts[4:])
    elif scope in ("ch", "chpair", "roi"):
        if len(parts) >= 6:
            identifier = parts[4]
            statistic = "_".join(parts[5:])
        else:
            identifier = parts[4] if len(parts) > 4 else "unknown"
            statistic = "unknown"
    else:
        statistic = parts[-1]
        return FeatureMetadata(
            name=name,
            group=group,
            segment="unknown",
            band="unknown",
            scope="unknown",
            statistic=statistic,
        )

    scope_description = identifier if identifier else scope
    description = f"{group} feature on {segment} in {band} band ({scope_description})"

    return FeatureMetadata(
        name=name,
        group=group,
        segment=segment,
        band=band,
        scope=scope,
        identifier=identifier,
        statistic=statistic,
        description=description,
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


def _create_feature_entry(feature_name: str, parsed: dict) -> Dict[str, Any]:
    """Create a feature entry from parsed naming schema."""
    if parsed.get("valid"):
        return {
            "name": feature_name,
            "group": parsed.get("group"),
            "segment": parsed.get("segment"),
            "band": parsed.get("band"),
            "scope": parsed.get("scope"),
            "identifier": parsed.get("identifier"),
            "statistic": parsed.get("stat"),
        }
    return {
        "name": feature_name,
        "group": "unknown",
        "segment": "unknown",
        "band": "unknown",
        "scope": "unknown",
        "identifier": None,
        "statistic": None,
    }


def generate_manifest(
    feature_columns: List[str],
    config: Any = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
    qc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    features = []
    for column_name in feature_columns:
        feature_name = str(column_name)
        parsed = NamingSchema.parse(feature_name)
        feature_entry = _create_feature_entry(feature_name, parsed)
        features.append(feature_entry)

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
    subject_dir = output_dir / f"sub-{subject}" / "eeg" / "features"
    subject_dir.mkdir(parents=True, exist_ok=True)

    base_filename = f"sub-{subject}_task-{task}"
    features_path = subject_dir / f"{base_filename}_features.tsv"
    df.to_csv(features_path, sep="\t", index=False)

    metadata_columns = {"condition", "trial", "epoch", "subject"}
    feature_columns = [
        column for column in df.columns if column not in metadata_columns
    ]
    manifest = generate_manifest(
        feature_columns, config=config, subject=subject, task=task, qc=qc
    )
    manifest_path = subject_dir / f"{base_filename}_features_manifest.json"
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
