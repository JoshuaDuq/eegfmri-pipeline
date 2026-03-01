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


CONDITION_COLUMN_NAME = "condition"


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
    If events_df is provided, a 'condition' column is added with per-trial labels
    derived from the configured binary outcome column.

    Attributes
    ----------
    features : Dict[str, FeatureSet]
        Mapping from feature group name to FeatureSet
    precomputed : PrecomputedData
        The precomputed intermediates (can be reused)
    condition : Optional[np.ndarray]
        Array of condition labels per epoch (values depend on the paradigm)
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
            If True and condition labels exist, adds condition column.
        
        Returns
        -------
        pd.DataFrame
            Combined features with one row per epoch.
        """
        non_empty_dfs = [
            feature_set.df
            for feature_set in self.features.values()
            if not feature_set.df.empty
        ]
        
        if not non_empty_dfs:
            return pd.DataFrame()
        
        combined = pd.concat(non_empty_dfs, axis=1)
        
        if include_condition and self.condition is not None:
            combined.insert(0, CONDITION_COLUMN_NAME, self.condition)
        
        return self._reorder_columns(combined)
    
    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns: condition first (if present), then sorted feature columns."""
        if CONDITION_COLUMN_NAME not in df.columns:
            return df[sorted(df.columns)]
        
        feature_columns = sorted([col for col in df.columns if col != CONDITION_COLUMN_NAME])
        return df[[CONDITION_COLUMN_NAME] + feature_columns]
    
    def get_feature_group_df(self, group: str, include_condition: bool = True) -> pd.DataFrame:
        """
        Get DataFrame for a specific feature group.
        
        Parameters
        ----------
        group : str
            Name of the feature group to retrieve.
        include_condition : bool
            If True and condition labels exist, adds condition column.
        
        Returns
        -------
        pd.DataFrame
            Feature group DataFrame, optionally with condition column.
        """
        if group not in self.features:
            return pd.DataFrame()
        
        feature_set = self.features[group]
        df = feature_set.df.copy()
        
        if include_condition and self.condition is not None:
            df.insert(0, CONDITION_COLUMN_NAME, self.condition)
        
        return df
    
    def get_all_columns(self) -> List[str]:
        """
        Get all column names across all feature groups.
        
        Returns
        -------
        List[str]
            All feature column names from all groups.
        """
        all_columns = []
        for feature_set in self.features.values():
            all_columns.extend(feature_set.columns)
        return all_columns
    
    @property
    def n_epochs(self) -> int:
        """Number of epochs across all feature groups."""
        if not self.features:
            return 0
        
        first_feature_set = next(iter(self.features.values()))
        return len(first_feature_set.df)
    
    def condition_counts(self) -> Dict[str, int]:
        """Return a mapping from each condition label to its trial count."""
        if self.condition is None:
            return {}
        unique, counts = np.unique(self.condition, return_counts=True)
        return {str(label): int(count) for label, count in zip(unique, counts)}

    def __repr__(self) -> str:
        total_features = sum(
            len(feature_set.columns)
            for feature_set in self.features.values()
        )
        group_names = list(self.features.keys())

        condition_info = ""
        if self.condition is not None:
            counts = self.condition_counts()
            condition_info = " (" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + ")"

        return (
            f"ExtractionResult({self.n_epochs} epochs, "
            f"{total_features} features from {group_names}{condition_info})"
        )

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
        total_features = sum(
            len(feature_set.columns) 
            for feature_set in self.features.values()
        )
        
        summary: Dict[str, Any] = {
            "n_feature_groups": len(self.features),
            "total_features": total_features,
            "n_epochs": self.n_epochs,
            "groups_extracted": list(self.features.keys()),
            "groups_with_issues": [],
            "per_group_status": {},
        }
        
        self._add_qc_status_to_summary(summary)
        self._add_condition_summary_to_summary(summary)
        
        return summary
    
    def _add_qc_status_to_summary(self, summary: Dict[str, Any]) -> None:
        """Add QC status information for each feature group."""
        for group_name, qc_data in self.qc.items():
            if group_name == "precomputed" or not isinstance(qc_data, dict):
                continue
            
            if "skipped_reason" in qc_data:
                status = f"skipped: {qc_data['skipped_reason']}"
            elif "error" in qc_data:
                status = f"error: {qc_data['error']}"
            else:
                status = "ok"
            
            summary["per_group_status"][group_name] = status
            
            if status.startswith("skipped:") or status.startswith("error:"):
                summary["groups_with_issues"].append(group_name)
    
    def _add_condition_summary_to_summary(self, summary: Dict[str, Any]) -> None:
        """Add condition label summary if available."""
        if self.condition is not None:
            summary["condition_counts"] = self.condition_counts()

    def build_manifest(
        self, 
        config: Any = None, 
        subject: Optional[str] = None, 
        task: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate manifest for current feature columns.
        
        Parameters
        ----------
        config : Any, optional
            Configuration object for manifest generation.
        subject : str, optional
            Subject identifier.
        task : str, optional
            Task identifier.
        
        Returns
        -------
        Dict[str, Any]
            Feature manifest dictionary.
        """
        feature_dataframe = self.get_combined_df(include_condition=False)
        feature_columns = list(feature_dataframe.columns)
        
        return generate_manifest(
            feature_columns,
            config=config,
            subject=subject,
            task=task,
            qc=self.qc,
            df_attrs=dict(getattr(feature_dataframe, "attrs", {}) or {}),
        )

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
        
        Parameters
        ----------
        output_dir : Union[str, Path]
            Directory path for saving features and manifest.
        subject : str
            Subject identifier.
        task : str
            Task identifier.
        config : Any, optional
            Configuration object for manifest generation.
        include_condition : bool
            If True, includes condition column in saved DataFrame.
        
        Returns
        -------
        Dict[str, Path]
            Mapping of saved file types to their paths.
        """
        feature_dataframe = self.get_combined_df(include_condition=include_condition)
        output_path = Path(output_dir)
        
        return save_features_organized(
            feature_dataframe,
            output_path,
            subject,
            task,
            config=config,
            qc=self.qc,
        )


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
    # Microstate dynamics features (A-D coverage/duration/occurrence/transitions)
    microstates_df: Optional[pd.DataFrame] = None
    microstates_cols: List[str] = field(default_factory=list)
    # Directed connectivity features (PSI, DTF, PDC)
    dconn_df: Optional[pd.DataFrame] = None
    dconn_cols: List[str] = field(default_factory=list)
    # Source localization features (LCMV, eLORETA)
    source_df: Optional[pd.DataFrame] = None
    source_cols: List[str] = field(default_factory=list)
    # Source condition-contrast features (subject-level deltas/effect sizes)
    source_contrast_df: Optional[pd.DataFrame] = None
    source_contrast_cols: List[str] = field(default_factory=list)
