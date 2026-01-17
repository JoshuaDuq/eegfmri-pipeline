"""
Feature Metadata and Naming (Canonical)
========================================

Single source of truth for feature naming conventions, metadata inference,
and manifest generation. All feature naming and metadata logic should import
from this module.

Provides:
- NamingSchema: Builder class for structured names (group.segment.band.scope.stat)
- generate_manifest, save_manifest: Manifest generation
- save_features_organized: I/O helpers
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd


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
        else:
            if scope_idx + 1 >= len(parts):
                return {"valid": False}
            result["identifier"] = parts[scope_idx + 1]
            result["stat"] = "_".join(parts[scope_idx + 2:])

        return result


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
    df_attrs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    df_attrs = dict(df_attrs or {})
    features = []
    columns_by_group: Dict[str, List[str]] = {}
    columns_by_band: Dict[str, List[str]] = {}
    columns_by_segment: Dict[str, List[str]] = {}

    for column_name in feature_columns:
        parsed = NamingSchema.parse(column_name)
        feature_entry = _create_feature_entry(column_name, parsed)
        features.append(feature_entry)

        group = feature_entry.get("group", "unknown")
        band = feature_entry.get("band", "unknown")
        segment = feature_entry.get("segment", "unknown")

        columns_by_group.setdefault(group, []).append(column_name)
        if band and band != "unknown":
            columns_by_band.setdefault(band, []).append(column_name)
        if segment and segment != "unknown":
            columns_by_segment.setdefault(segment, []).append(column_name)

    provenance = infer_feature_provenance(
        feature_columns=feature_columns,
        config=config,
        df_attrs=df_attrs,
    )

    return {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "subject": subject,
        "task": task,
        "n_features": len(features),
        "features": features,
        "columns_by_group": columns_by_group,
        "columns_by_band": columns_by_band,
        "columns_by_segment": columns_by_segment,
        "provenance": provenance,
        "qc": _make_json_serializable(qc) if qc else None,
        "config": None if config is None else {},
    }


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    """Extract value from config object (dict or object with get method)."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            try:
                return getter(key)
            except (TypeError, AttributeError):
                return default
    return default


def infer_feature_provenance(
    *,
    feature_columns: List[str],
    config: Any = None,
    df_attrs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Infer per-column statistical validity properties for downstream analyses."""
    df_attrs = dict(df_attrs or {})

    analysis_mode = str(_config_get(config, "feature_engineering.analysis_mode", "group_stats")).strip().lower()
    itpc_method = str(_config_get(config, "feature_engineering.itpc.method", "fold_global")).strip().lower()
    conn_granularity_cfg = str(_config_get(config, "feature_engineering.connectivity.granularity", "trial")).strip().lower()
    conn_phase_estimator_cfg = str(_config_get(config, "feature_engineering.connectivity.phase_estimator", "within_epoch")).strip().lower()

    conn_granularity = str(df_attrs.get("feature_granularity") or conn_granularity_cfg).strip().lower()
    broadcast_warning = df_attrs.get("broadcast_warning")

    def group_props(group: str) -> Dict[str, Any]:
        group = str(group or "unknown").strip().lower()
        if group == "itpc":
            if itpc_method == "condition":
                return {
                    "analysis_unit": "condition",
                    "broadcasted": True,
                    "cross_trial_dependence": True,
                    "trialwise_valid": False,
                    "reason": "ITPC is computed across trials within condition and broadcast to each trial.",
                }
            if itpc_method in {"fold_global", "global"}:
                return {
                    "analysis_unit": "subject",
                    "broadcasted": True,
                    "cross_trial_dependence": True,
                    "trialwise_valid": False,
                    "reason": f"ITPC(method='{itpc_method}') is computed across trials and broadcast to each trial.",
                }
            if itpc_method == "loo":
                return {
                    "analysis_unit": "trial",
                    "broadcasted": False,
                    "cross_trial_dependence": True,
                    "trialwise_valid": False,
                    "reason": "ITPC(method='loo') uses other trials to compute each trial's value (non-i.i.d.).",
                }
            return {
                "analysis_unit": "unknown",
                "broadcasted": False,
                "cross_trial_dependence": True,
                "trialwise_valid": False,
                "reason": f"Unknown ITPC method '{itpc_method}'.",
            }

        if group in {"conn", "dconn"}:
            broadcasted = conn_granularity in {"subject", "condition"} or conn_phase_estimator_cfg == "across_epochs"
            analysis_unit = conn_granularity if conn_granularity in {"trial", "condition", "subject"} else "trial"
            trialwise_valid = not broadcasted
            reason = "Connectivity is computed within-epoch per trial." if trialwise_valid else "Connectivity is aggregated across epochs and broadcast."
            return {
                "analysis_unit": analysis_unit,
                "broadcasted": broadcasted,
                "cross_trial_dependence": broadcasted,
                "trialwise_valid": trialwise_valid,
                "reason": reason,
            }

        return {
            "analysis_unit": "trial",
            "broadcasted": False,
            "cross_trial_dependence": False,
            "trialwise_valid": True,
            "reason": "Computed per trial (assumed i.i.d.).",
        }

    columns: Dict[str, Any] = {}
    for name in feature_columns:
        parsed = NamingSchema.parse(name)
        group = parsed.get("group") if parsed.get("valid") else "unknown"
        columns[name] = group_props(group)

    out: Dict[str, Any] = {
        "analysis_mode": analysis_mode,
        "methods": {
            "itpc_method": itpc_method,
            "connectivity_granularity": conn_granularity,
            "connectivity_phase_estimator": conn_phase_estimator_cfg,
        },
        "file_attrs": _make_json_serializable(df_attrs) if df_attrs else {},
        "columns": columns,
    }
    if broadcast_warning:
        out["warnings"] = [str(broadcast_warning)]
    return out


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
        feature_columns,
        config=config,
        subject=subject,
        task=task,
        qc=qc,
        df_attrs=dict(getattr(df, "attrs", None) or {}),
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
    "NamingSchema",
    "generate_manifest",
    "save_manifest",
    "save_features_organized",
]
