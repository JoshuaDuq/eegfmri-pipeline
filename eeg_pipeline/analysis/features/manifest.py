"""
Feature Manifest Generator
==========================

Generates a manifest describing all extracted features with metadata
for downstream ML pipelines and documentation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import pandas as pd


@dataclass
class FeatureMetadata:
    """Metadata for a single feature."""
    name: str
    domain: str
    band: Optional[str] = None
    location: Optional[str] = None
    time_window: Optional[str] = None
    statistic: Optional[str] = None
    unit: Optional[str] = None
    description: Optional[str] = None


def parse_feature_name(name: str) -> FeatureMetadata:
    """
    Parse a feature name into its components.
    
    Naming convention: {domain}_{measure}_{band}_{location}_{time}_{stat}
    """
    parts = name.split("_")
    if not parts:
        return FeatureMetadata(name=name, domain="unknown")
    
    domain = parts[0]
    
    # Domain-specific parsing
    if domain in ("power", "erds"):
        # power_band_channel_time_stat or erds_band_channel_time_stat
        return FeatureMetadata(
            name=name,
            domain=domain,
            band=parts[1] if len(parts) > 1 else None,
            location=parts[2] if len(parts) > 2 else None,
            time_window=parts[3] if len(parts) > 3 else None,
            statistic=parts[4] if len(parts) > 4 else None,
            unit="log(µV²/µV²)" if domain == "power" else "%",
        )
    elif domain == "conn":
        # conn_measure_band_time_stat
        return FeatureMetadata(
            name=name,
            domain="connectivity",
            statistic=parts[1] if len(parts) > 1 else None,
            band=parts[2] if len(parts) > 2 else None,
            time_window=parts[3] if len(parts) > 3 else None,
            unit="a.u.",
        )
    elif domain == "phase":
        # phase_measure_band_location_time_stat
        return FeatureMetadata(
            name=name,
            domain="phase",
            statistic=parts[1] if len(parts) > 1 else None,
            band=parts[2] if len(parts) > 2 else None,
            location=parts[3] if len(parts) > 3 else None,
            time_window=parts[4] if len(parts) > 4 else None,
            unit="a.u.",
        )
    elif domain == "gfp":
        # gfp_[band]_time_stat
        if len(parts) == 3:
            return FeatureMetadata(
                name=name,
                domain="gfp",
                time_window=parts[1],
                statistic=parts[2],
                unit="µV",
            )
        return FeatureMetadata(
            name=name,
            domain="gfp",
            band=parts[1] if len(parts) > 1 else None,
            time_window=parts[2] if len(parts) > 2 else None,
            statistic=parts[3] if len(parts) > 3 else None,
            unit="µV",
        )
    elif domain == "roi":
        # roi_subdomain_band_roi_time_stat
        return FeatureMetadata(
            name=name,
            domain="roi",
            statistic=parts[1] if len(parts) > 1 else None,
            band=parts[2] if len(parts) > 2 else None,
            location=parts[3] if len(parts) > 3 else None,
            time_window=parts[4] if len(parts) > 4 else None,
            unit="log(µV²/µV²)" if "power" in name else "%",
        )
    elif domain == "asym":
        # asym_band_pair_time_stat
        return FeatureMetadata(
            name=name,
            domain="asymmetry",
            band=parts[1] if len(parts) > 1 else None,
            location=parts[2] if len(parts) > 2 else None,
            time_window=parts[3] if len(parts) > 3 else None,
            statistic=parts[4] if len(parts) > 4 else None,
            unit="index",
        )
    elif domain in ("temp", "comp"):
        # temp_measure_band_channel_time or comp_measure_band_channel_time
        return FeatureMetadata(
            name=name,
            domain="temporal" if domain == "temp" else "complexity",
            statistic=parts[1] if len(parts) > 1 else None,
            band=parts[2] if len(parts) > 2 else None,
            location=parts[3] if len(parts) > 3 else None,
            time_window=parts[4] if len(parts) > 4 else None,
            unit="a.u.",
        )
    else:
        return FeatureMetadata(name=name, domain=domain)


def generate_manifest(
    feature_columns: List[str],
    config: Any = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a manifest describing all features.
    
    Parameters
    ----------
    feature_columns : List[str]
        List of feature column names
    config : Any, optional
        Configuration object for additional metadata
    subject : str, optional
        Subject identifier
    task : str, optional
        Task name
    
    Returns
    -------
    Dict[str, Any]
        Manifest dictionary with feature metadata
    """
    features = []
    domains = set()
    bands = set()
    time_windows = set()
    
    for col in feature_columns:
        meta = parse_feature_name(col)
        features.append(asdict(meta))
        domains.add(meta.domain)
        if meta.band:
            bands.add(meta.band)
        if meta.time_window:
            time_windows.add(meta.time_window)
    
    manifest = {
        "version": "2.0",
        "generated_at": datetime.now().isoformat(),
        "subject": subject,
        "task": task,
        "summary": {
            "n_features": len(features),
            "domains": sorted(domains),
            "bands": sorted(bands),
            "time_windows": sorted(time_windows),
        },
        "features": features,
    }
    
    # Add config info if available
    if config is not None:
        try:
            tf_cfg = config.get("time_frequency_analysis", {})
            fe_cfg = config.get("feature_engineering", {}).get("features", {})
            manifest["config"] = {
                "baseline_window": tf_cfg.get("baseline_window", [-5.0, -0.01]),
                "plateau_window": tf_cfg.get("plateau_window", [3.0, 10.5]),
                "temporal_bins_coarse": fe_cfg.get("temporal_bins", []),
                "temporal_bins_fine": fe_cfg.get("temporal_bins_fine", []),
            }
        except (AttributeError, TypeError):
            pass
    
    return manifest


def save_manifest(
    manifest: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save manifest to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)


def save_features_organized(
    df: pd.DataFrame,
    output_dir: Path,
    subject: str,
    task: str,
    config: Any = None,
) -> Dict[str, Path]:
    """
    Save features in organized structure with manifest.
    
    Creates:
    - features_all.tsv: Complete feature matrix
    - features_manifest.json: Feature metadata
    - by_domain/: Features split by domain
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (epochs x features)
    output_dir : Path
        Output directory
    subject : str
        Subject identifier
    task : str
        Task name
    config : Any, optional
        Configuration object
    
    Returns
    -------
    Dict[str, Path]
        Mapping of output type to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    
    # Save complete feature matrix
    all_path = output_dir / f"{subject}_{task}_features_all.tsv"
    df.to_csv(all_path, sep="\t", index=False)
    outputs["all"] = all_path
    
    # Generate and save manifest
    feature_cols = [c for c in df.columns if c != "condition"]
    manifest = generate_manifest(feature_cols, config, subject, task)
    manifest_path = output_dir / f"{subject}_{task}_features_manifest.json"
    save_manifest(manifest, manifest_path)
    outputs["manifest"] = manifest_path
    
    # Save by domain
    domain_dir = output_dir / "by_domain"
    domain_dir.mkdir(exist_ok=True)
    
    # Group columns by domain
    domain_cols: Dict[str, List[str]] = {}
    for col in feature_cols:
        meta = parse_feature_name(col)
        domain = meta.domain
        if domain not in domain_cols:
            domain_cols[domain] = []
        domain_cols[domain].append(col)
    
    for domain, cols in domain_cols.items():
        domain_df = df[["condition"] + cols] if "condition" in df.columns else df[cols]
        domain_path = domain_dir / f"{subject}_{task}_{domain}.tsv"
        domain_df.to_csv(domain_path, sep="\t", index=False)
        outputs[f"domain_{domain}"] = domain_path
    
    return outputs
