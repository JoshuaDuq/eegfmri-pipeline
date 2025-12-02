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
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

import numpy as np
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
    
    Supports feature families:
    - spectral: pow, logpow, relpow, peakfreq, peakpow, ratio, logratio, sef*, slope, iaf
    - connectivity: wpli, plv, aec, imcoh, pli, graph, conn
    - pac: pac_mi, pac_coupling, pac_* (phase-amplitude coupling)
    - effectsize: effectsize_*, cohens_d_*, hedges_g_*
    - phase_itpc: phase_itpc_*, itpc_* (inter-trial phase coherence)
    - ratio globals: ratio_* for band ratio features
    """
    parts = name.split("_")
    if not parts:
        return FeatureMetadata(name=name, domain="unknown")
    
    domain = parts[0]

    # Common prefix-based domains that aren't covered by the simple domain token
    spectral_prefixes = {
        "pow", "logpow", "relpow", "peakfreq", "peakpow", "peakprom",
        "sef50", "sef75", "sef90", "sef95", "slope", "logratio",
        "iaf",
    }
    connectivity_prefixes = {"wpli", "plv", "aec", "imcoh", "pli", "graph", "conn"}
    
    # === PAC (Phase-Amplitude Coupling) ===
    # pac_mi_phase_amp_location[_time][_stat], pac_coupling_*, etc.
    if domain == "pac":
        measure = parts[1] if len(parts) > 1 else None
        # PAC typically has phase_band and amp_band
        if len(parts) >= 4:
            phase_band = parts[2] if len(parts) > 2 else None
            amp_band = parts[3] if len(parts) > 3 else None
            band_label = f"{phase_band}:{amp_band}" if phase_band and amp_band else None
        else:
            band_label = parts[2] if len(parts) > 2 else None
        location = parts[4] if len(parts) > 4 else None
        time_window = parts[5] if len(parts) > 5 else None
        statistic = parts[6] if len(parts) > 6 else (parts[-1] if len(parts) > 4 else None)
        return FeatureMetadata(
            name=name,
            domain="pac",
            statistic=measure,
            band=band_label,
            location=location,
            time_window=time_window,
            unit="MI" if measure == "mi" else "a.u.",
            description="Phase-amplitude coupling measure",
        )
    
    # === Effect Size Features ===
    # effectsize_measure_band_location[_time], cohens_d_*, hedges_g_*
    if domain in {"effectsize", "cohens", "hedges"}:
        if domain in {"cohens", "hedges"}:
            # cohens_d_band_location or hedges_g_band_location
            measure = f"{domain}_{parts[1]}" if len(parts) > 1 else domain
            band_label = parts[2] if len(parts) > 2 else None
            location = parts[3] if len(parts) > 3 else None
            time_window = parts[4] if len(parts) > 4 else None
        else:
            measure = parts[1] if len(parts) > 1 else None
            band_label = parts[2] if len(parts) > 2 else None
            location = parts[3] if len(parts) > 3 else None
            time_window = parts[4] if len(parts) > 4 else None
        return FeatureMetadata(
            name=name,
            domain="effectsize",
            statistic=measure,
            band=band_label,
            location=location,
            time_window=time_window,
            unit="d" if "cohens" in name or "hedges" in name else "a.u.",
            description="Effect size measure",
        )
    
    # === ITPC (Inter-Trial Phase Coherence) ===
    # phase_itpc_band_location[_time][_stat] or itpc_band_location[_time][_stat]
    if domain == "itpc" or (domain == "phase" and len(parts) > 1 and parts[1] == "itpc"):
        if domain == "phase":
            band_label = parts[2] if len(parts) > 2 else None
            location = parts[3] if len(parts) > 3 else None
            time_window = parts[4] if len(parts) > 4 else None
            statistic = parts[5] if len(parts) > 5 else None
        else:
            band_label = parts[1] if len(parts) > 1 else None
            location = parts[2] if len(parts) > 2 else None
            time_window = parts[3] if len(parts) > 3 else None
            statistic = parts[4] if len(parts) > 4 else None
        return FeatureMetadata(
            name=name,
            domain="phase_itpc",
            statistic=statistic or "itpc",
            band=band_label,
            location=location,
            time_window=time_window,
            unit="a.u.",
            description="Inter-trial phase coherence",
        )
    
    # === Ratio Features (global band ratios) ===
    # ratio_num_denom[_location][_time][_stat]
    if domain == "ratio" and len(parts) >= 3:
        band_label = f"{parts[1]}:{parts[2]}"
        location = parts[3] if len(parts) > 3 else None
        time_window = parts[4] if len(parts) > 4 else None
        statistic = parts[5] if len(parts) > 5 else None
        return FeatureMetadata(
            name=name,
            domain="spectral",
            band=band_label,
            location=location,
            time_window=time_window,
            statistic=statistic or "ratio",
            unit="a.u.",
            description="Band power ratio",
        )

    if domain in spectral_prefixes:
        # pow_band_channel_* or logratio_band_channel[_time[_stat]]
        band_label = parts[1] if len(parts) > 1 else None
        location = parts[2] if len(parts) > 2 else None
        statistic = parts[-1] if len(parts) > 3 else None
        time_window = parts[3] if len(parts) > 3 and statistic != parts[3] else None
        return FeatureMetadata(
            name=name,
            domain="spectral",
            band=band_label,
            location=location,
            time_window=time_window,
            statistic=statistic,
            unit="a.u.",
        )

    def _parse_connectivity(parts_list: List[str]) -> FeatureMetadata:
        """Single connectivity parser for all connectivity-like prefixes."""
        band_label = parts_list[1] if len(parts_list) > 1 else None
        time_window = (
            parts_list[3]
            if len(parts_list) > 3
            else (parts_list[2] if len(parts_list) > 2 else None)
        )
        statistic = "_".join(parts_list[2:]) if len(parts_list) > 2 else None
        return FeatureMetadata(
            name=name,
            domain="connectivity",
            band=band_label,
            time_window=time_window,
            statistic=statistic,
            unit="a.u.",
        )

    if domain in connectivity_prefixes or domain == "connectivity":
        # wpli_band_stat, graph_band_metric, conn_plv_band_time_mean, etc.
        return _parse_connectivity(parts)
    
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


def _json_safe(value: Any) -> Any:
    """Convert common numpy/pandas objects to JSON-serializable structures."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def generate_manifest(
    feature_columns: List[str],
    config: Any = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
    qc: Optional[Dict[str, Any]] = None,
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

    if qc is not None:
        manifest["qc"] = _json_safe(qc)
    
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
    qc: Optional[Dict[str, Any]] = None,
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
    manifest = generate_manifest(feature_cols, config, subject, task, qc=qc)
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


# =============================================================================
# BIDS-Compliant Output
# =============================================================================


def _create_bids_sidecar(
    feature_columns: List[str],
    config: Any = None,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    qc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create BIDS-compliant sidecar JSON for features file."""
    manifest = generate_manifest(feature_columns, config, subject, task, qc=qc)
    
    # Add BIDS-specific fields
    sidecar = {
        "Description": "EEG-derived features for fMRI prediction",
        "Sources": [f"sub-{subject}/eeg/"],
        "RawSources": [f"sub-{subject}/eeg/"],
        "SamplingFrequency": None,  # Features are per-epoch
        "StartTime": 0,
        "Columns": {},
    }
    
    # Add column descriptions
    for col in feature_columns:
        meta = parse_feature_name(col)
        sidecar["Columns"][col] = {
            "Description": f"{meta.domain} feature",
            "Units": meta.unit or "a.u.",
        }
        if meta.band:
            sidecar["Columns"][col]["FrequencyBand"] = meta.band
        if meta.time_window:
            sidecar["Columns"][col]["TimeWindow"] = meta.time_window
    
    # Merge manifest info
    sidecar["FeatureManifest"] = manifest
    
    if session:
        sidecar["Session"] = session
    if run:
        sidecar["Run"] = run
    
    return sidecar


def save_features_bids(
    df: pd.DataFrame,
    bids_root: Path,
    subject: str,
    task: str,
    *,
    session: Optional[str] = None,
    run: Optional[str] = None,
    config: Any = None,
    qc: Optional[Dict[str, Any]] = None,
    create_derivatives_structure: bool = True,
) -> Dict[str, Path]:
    """
    Save features in BIDS-compliant structure.
    
    Creates:
    ```
    derivatives/features/
    ├── dataset_description.json
    └── sub-{subject}/
        └── [ses-{session}/]
            ├── sub-{subject}[_ses-{session}]_task-{task}[_run-{run}]_features.tsv
            ├── sub-{subject}[_ses-{session}]_task-{task}[_run-{run}]_features.json
            └── sub-{subject}[_ses-{session}]_task-{task}[_run-{run}]_qc.json
    ```
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    bids_root : Path
        Root of BIDS dataset (derivatives/features will be created here)
    subject : str
        Subject ID (without "sub-" prefix)
    task : str
        Task name
    session : Optional[str]
        Session ID (without "ses-" prefix)
    run : Optional[str]
        Run ID (without "run-" prefix)
    config : Any
        Configuration object
    qc : Optional[Dict[str, Any]]
        QC metrics
    create_derivatives_structure : bool
        Create derivatives folder structure if not exists
    
    Returns
    -------
    Dict[str, Path]
        Mapping of output type to file path
    """
    bids_root = Path(bids_root)
    deriv_root = bids_root / "derivatives" / "features"
    
    # Build path components
    sub_dir = deriv_root / f"sub-{subject}"
    if session:
        sub_dir = sub_dir / f"ses-{session}"
    
    # Build filename base
    parts = [f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")
    parts.append(f"task-{task}")
    if run:
        parts.append(f"run-{run}")
    
    base_name = "_".join(parts)
    
    # Create directories
    if create_derivatives_structure:
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset_description.json if not exists
        desc_path = deriv_root / "dataset_description.json"
        if not desc_path.exists():
            desc = {
                "Name": "EEG Feature Derivatives",
                "BIDSVersion": "1.8.0",
                "DatasetType": "derivative",
                "GeneratedBy": [{
                    "Name": "eeg_pipeline",
                    "Description": "EEG feature extraction pipeline for fMRI prediction",
                }],
            }
            with open(desc_path, "w") as f:
                json.dump(desc, f, indent=2)
    
    outputs = {}
    
    # Save features TSV
    features_path = sub_dir / f"{base_name}_features.tsv"
    df.to_csv(features_path, sep="\t", index=False)
    outputs["features"] = features_path
    
    # Save sidecar JSON
    feature_cols = [c for c in df.columns if c != "condition"]
    sidecar = _create_bids_sidecar(
        feature_cols, config, subject, session, task, run, qc
    )
    sidecar_path = sub_dir / f"{base_name}_features.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    outputs["sidecar"] = sidecar_path
    
    # Save QC if provided
    if qc:
        qc_path = sub_dir / f"{base_name}_qc.json"
        with open(qc_path, "w") as f:
            json.dump(_json_safe(qc), f, indent=2)
        outputs["qc"] = qc_path
    
    return outputs


def load_bids_features(
    bids_root: Path,
    subject: str,
    task: str,
    *,
    session: Optional[str] = None,
    run: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Load features from BIDS-compliant structure.
    
    Parameters
    ----------
    bids_root : Path
        Root of BIDS dataset
    subject : str
        Subject ID
    task : str
        Task name
    session : Optional[str]
        Session ID
    run : Optional[str]
        Run ID
    
    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]
        (features DataFrame, sidecar metadata) or (None, None) if not found
    """
    bids_root = Path(bids_root)
    deriv_root = bids_root / "derivatives" / "features"
    
    # Build path
    sub_dir = deriv_root / f"sub-{subject}"
    if session:
        sub_dir = sub_dir / f"ses-{session}"
    
    # Build filename
    parts = [f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")
    parts.append(f"task-{task}")
    if run:
        parts.append(f"run-{run}")
    
    base_name = "_".join(parts)
    features_path = sub_dir / f"{base_name}_features.tsv"
    sidecar_path = sub_dir / f"{base_name}_features.json"
    
    if not features_path.exists():
        return None, None
    
    df = pd.read_csv(features_path, sep="\t")
    
    sidecar = None
    if sidecar_path.exists():
        with open(sidecar_path) as f:
            sidecar = json.load(f)
    
    return df, sidecar


def collect_group_features_bids(
    bids_root: Path,
    task: str,
    *,
    subjects: Optional[List[str]] = None,
    session: Optional[str] = None,
) -> pd.DataFrame:
    """
    Collect features from all subjects into a single DataFrame.
    
    Parameters
    ----------
    bids_root : Path
        Root of BIDS dataset
    task : str
        Task name
    subjects : Optional[List[str]]
        List of subject IDs (if None, finds all available)
    session : Optional[str]
        Session ID
    
    Returns
    -------
    pd.DataFrame
        Combined features with subject column added
    """
    bids_root = Path(bids_root)
    deriv_root = bids_root / "derivatives" / "features"
    
    # Find subjects if not specified
    if subjects is None:
        subjects = []
        for sub_dir in deriv_root.glob("sub-*"):
            if sub_dir.is_dir():
                sub_id = sub_dir.name.replace("sub-", "")
                subjects.append(sub_id)
        subjects.sort()
    
    dfs = []
    for subject in subjects:
        df, _ = load_bids_features(bids_root, subject, task, session=session)
        if df is not None:
            df = df.copy()
            df.insert(0, "subject", subject)
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, axis=0, ignore_index=True)
