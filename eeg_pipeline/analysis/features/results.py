"""
Feature Extraction Results (Canonical)
======================================

Containers for feature extraction results. This is the single source of truth
for all feature result containers used across the pipeline.

Classes:
- FeatureSet: Single feature group container
- ExtractionResult: Dictionary-based container for precomputed pipeline
- FeatureExtractionResult: Flat container for TFR-based pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.domain.features.naming import generate_manifest, save_features_organized


@dataclass
class FeatureSet:
    """
    Container for a single feature group's extraction results.
    
    Attributes
    ----------
    df : pd.DataFrame
        Feature values, shape (n_epochs, n_features)
    columns : List[str]
        Column names for the features
    name : str
        Name of this feature group (e.g., "erds", "spectral")
    """
    df: pd.DataFrame
    columns: List[str]
    name: str


@dataclass
class ExtractionResult:
    """
    Container for all extracted feature groups, epoch-aligned with condition labels.
    
    Each feature group produces a DataFrame with one row per epoch.
    If events_df is provided, a 'condition' column is added indicating
    'pain' or 'nonpain' for each trial.
    
    Attributes
    ----------
    features : Dict[str, FeatureSet]
        Mapping from feature group name to FeatureSet
    precomputed : PrecomputedData
        The precomputed intermediates (can be reused)
    condition : Optional[np.ndarray]
        Array of condition labels ('pain' or 'nonpain') per epoch
    """
    features: Dict[str, FeatureSet] = field(default_factory=dict)
    precomputed: Optional[PrecomputedData] = None
    condition: Optional[np.ndarray] = None
    qc: Dict[str, Any] = field(default_factory=dict)
    
    def get_combined_df(self, include_condition: bool = True) -> pd.DataFrame:
        """
        Combine all feature sets into a single DataFrame.
        
        Parameters
        ----------
        include_condition : bool
            If True and condition labels exist, adds 'condition' column.
        
        Returns
        -------
        pd.DataFrame
            Combined features with one row per epoch.
        """
        if not self.features:
            return pd.DataFrame()
        
        dfs = [fs.df for fs in self.features.values() if not fs.df.empty]
        
        if not dfs:
            return pd.DataFrame()
        
        combined = pd.concat(dfs, axis=1)
        
        # Add condition column if available
        if include_condition and self.condition is not None:
            combined.insert(0, "condition", self.condition)
        
        # Stable column order for reproducibility (condition first, then sorted)
        fixed_cols = ["condition"] if "condition" in combined.columns else []
        other_cols = sorted([c for c in combined.columns if c not in fixed_cols])
        return combined[fixed_cols + other_cols] if fixed_cols else combined[other_cols]
    
    def get_feature_group_df(self, group: str, include_condition: bool = True) -> pd.DataFrame:
        """Get DataFrame for a specific feature group."""
        if group not in self.features:
            return pd.DataFrame()
        
        df = self.features[group].df.copy()
        
        if include_condition and self.condition is not None:
            df.insert(0, "condition", self.condition)
        
        return df
    
    def get_all_columns(self) -> List[str]:
        """Get all column names across all feature groups."""
        cols = []
        for fs in self.features.values():
            cols.extend(fs.columns)
        return cols
    
    @property
    def n_epochs(self) -> int:
        """Number of epochs."""
        if self.features:
            first_fs = next(iter(self.features.values()))
            return len(first_fs.df)
        return 0
    
    @property
    def n_pain(self) -> int:
        """Number of pain trials."""
        if self.condition is not None:
            return int(np.sum(self.condition == "pain"))
        return 0
    
    @property
    def n_nonpain(self) -> int:
        """Number of non-pain trials."""
        if self.condition is not None:
            return int(np.sum(self.condition == "nonpain"))
        return 0
    
    def __repr__(self) -> str:
        n_features = sum(len(fs.columns) for fs in self.features.values())
        groups = list(self.features.keys())
        if self.condition is not None:
            condition_str = f" (pain={self.n_pain}, nonpain={self.n_nonpain})"
        else:
            condition_str = ""
        return f"ExtractionResult({self.n_epochs} epochs, {n_features} features from {groups}{condition_str})"

    def get_qc_summary(self) -> Dict[str, Any]:
        """
        Return aggregated QC metrics across all feature groups.
        
        Returns
        -------
        Dict[str, Any]
            Summary including:
            - n_feature_groups: number of successfully extracted groups
            - total_features: total feature count
            - groups_with_issues: list of groups that had QC issues or were skipped
            - per_group_status: dict mapping group name to success/skip status
        """
        summary: Dict[str, Any] = {
            "n_feature_groups": len(self.features),
            "total_features": sum(len(fs.columns) for fs in self.features.values()),
            "n_epochs": self.n_epochs,
            "groups_extracted": list(self.features.keys()),
            "groups_with_issues": [],
            "per_group_status": {},
        }
        
        for name, qc_data in self.qc.items():
            if name == "precomputed":
                continue
            if isinstance(qc_data, dict):
                if qc_data.get("skipped_reason"):
                    summary["groups_with_issues"].append(name)
                    summary["per_group_status"][name] = f"skipped: {qc_data['skipped_reason']}"
                elif qc_data.get("error"):
                    summary["groups_with_issues"].append(name)
                    summary["per_group_status"][name] = f"error: {qc_data['error']}"
                else:
                    summary["per_group_status"][name] = "ok"
        
        # Add condition summary if available
        if self.condition is not None:
            summary["n_pain"] = self.n_pain
            summary["n_nonpain"] = self.n_nonpain
        
        return summary

    def build_manifest(self, config: Any = None, subject: Optional[str] = None, task: Optional[str] = None) -> Dict[str, Any]:
        """Generate manifest for current feature columns."""
        feature_cols = [c for c in self.get_combined_df(include_condition=False).columns]
        return generate_manifest(feature_cols, config=config, subject=subject, task=task, qc=self.qc or None)

    def save_with_manifest(
        self,
        output_dir: Union[str, Path],
        subject: str,
        task: str,
        config: Any = None,
        include_condition: bool = True,
    ) -> Dict[str, Path]:
        """
        Save combined features and manifest in a reproducible, organized structure.
        """
        df = self.get_combined_df(include_condition=include_condition)
        return save_features_organized(df, Path(output_dir), subject, task, config=config, qc=self.qc or None)


###################################################################
# TFR-Based Pipeline Result Container
###################################################################


@dataclass
class FeatureExtractionResult:
    """Typed container for TFR-based feature extraction outputs.
    
    This flat container is used by the TFR-on-the-fly pipeline in pipelines/features.py.
    Each field corresponds to a specific feature group's DataFrame and column names.
    """
    tfr: Any = None
    baseline_df: Optional[pd.DataFrame] = None
    baseline_cols: List[str] = field(default_factory=list)
    pow_df: Optional[pd.DataFrame] = None
    pow_cols: List[str] = field(default_factory=list)
    conn_df: Optional[pd.DataFrame] = None
    conn_cols: List[str] = field(default_factory=list)
    aper_df: Optional[pd.DataFrame] = None
    aper_cols: List[str] = field(default_factory=list)
    aper_qc: Optional[Any] = None
    erp_df: Optional[pd.DataFrame] = None
    erp_cols: List[str] = field(default_factory=list)
    phase_df: Optional[pd.DataFrame] = None
    phase_cols: List[str] = field(default_factory=list)
    itpc_trial_df: Optional[pd.DataFrame] = None
    itpc_trial_cols: List[str] = field(default_factory=list)
    pac_df: Optional[pd.DataFrame] = None
    pac_phase_freqs: Optional[Any] = None
    pac_amp_freqs: Optional[Any] = None
    pac_trials_df: Optional[pd.DataFrame] = None
    pac_time_df: Optional[pd.DataFrame] = None
    # ERDS features
    erds_df: Optional[pd.DataFrame] = None
    erds_cols: List[str] = field(default_factory=list)
    # Spectral features (IAF, peak frequency, spectral edge)
    spectral_df: Optional[pd.DataFrame] = None
    spectral_cols: List[str] = field(default_factory=list)
    # Band power ratios
    ratios_df: Optional[pd.DataFrame] = None
    ratios_cols: List[str] = field(default_factory=list)
    # Hemispheric asymmetry
    asymmetry_df: Optional[pd.DataFrame] = None
    asymmetry_cols: List[str] = field(default_factory=list)
    # Complexity features
    comp_df: Optional[pd.DataFrame] = None
    comp_cols: List[str] = field(default_factory=list)
    # Burst features
    bursts_df: Optional[pd.DataFrame] = None
    bursts_cols: List[str] = field(default_factory=list)
    # Quality metrics
    quality_df: Optional[pd.DataFrame] = None
    quality_cols: List[str] = field(default_factory=list)
    # Temporal binned features
    temp_df: Optional[pd.DataFrame] = None
    temp_cols: List[str] = field(default_factory=list)



def combine_feature_groups(result: ExtractionResult, groups: List[str]) -> tuple:
    """Combine specific feature groups from an ExtractionResult.
    
    Parameters
    ----------
    result : ExtractionResult
        Container with feature groups
    groups : List[str]
        Names of groups to combine
        
    Returns
    -------
    tuple
        (combined_df, column_list) where column_list matches combined_df.columns exactly
    """
    dfs: List[pd.DataFrame] = []
    cols: List[str] = []
    for group in groups:
        fs = result.features.get(group)
        if fs is None or fs.df.empty:
            continue
        dfs.append(fs.df)
        cols.extend(fs.columns)
    if not dfs:
        return pd.DataFrame(), []
    combined = pd.concat(dfs, axis=1)
    if result.condition is not None:
        combined.insert(0, "condition", result.condition)
        cols = ["condition"] + cols
    return combined, cols
