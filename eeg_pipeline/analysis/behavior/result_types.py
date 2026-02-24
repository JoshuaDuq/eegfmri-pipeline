from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class CorrelationDesign:
    """Output contract for correlate_design stage."""

    targets: List[str]
    feature_cols: List[str]
    partial_covars: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CorrelationResults:
    """Output contract for correlation stages."""

    records: List[Dict[str, Any]]
    df: Optional[pd.DataFrame] = None
    n_tests: int = 0
    n_significant: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConditionResults:
    """Output contract for condition comparison stages."""

    df: pd.DataFrame
    comparison_type: str = "column"
    n_condition_a: int = 0
    n_condition_b: int = 0
    n_significant: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RegressionResults:
    """Output contract for regression stage."""

    df: pd.DataFrame
    primary_unit: str = "trial"
    n_features: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TemporalResults:
    """Output contract for temporal stages."""

    power: Optional[Dict[str, Any]] = None
    itpc: Optional[Dict[str, Any]] = None
    erds: Optional[Dict[str, Any]] = None
    correction_method: str = "fdr"
    n_tests: int = 0
    n_significant: int = 0


@dataclass
class ClusterResults:
    """Output contract for cluster permutation stage."""

    n_clusters: int = 0
    n_significant: int = 0
    clusters: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FDRResults:
    """Output contract for FDR correction stages."""

    n_tests: int = 0
    n_significant: int = 0
    alpha: float = 0.05
    method: str = "fdr_bh"
    family_structure: Optional[Dict[str, Any]] = None
    family_df: Optional[pd.DataFrame] = None


@dataclass
class MixedEffectsResult:
    """Output contract for mixed-effects models (group-level)."""

    df: pd.DataFrame
    n_subjects: int = 0
    n_features: int = 0
    n_significant: int = 0
    random_effects: str = "intercept"
    family_structure: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GroupLevelResult:
    """Output contract for group-level analysis."""

    mixed_effects: Optional[MixedEffectsResult] = None
    multilevel_correlations: Optional[pd.DataFrame] = None
    n_subjects: int = 0
    subjects: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FeatureQCResult:
    """Result from feature QC screening."""

    passed_features: List[str]
    failed_features: Dict[str, List[str]]
    qc_df: pd.DataFrame
    metadata: Dict[str, Any]


@dataclass
class TrialTableResult:
    """Result from compute_trial_table."""

    df: pd.DataFrame
    metadata: Dict[str, Any]


@dataclass
class PredictorModelComparisonResult:
    """Result from compute_predictor_model_comparison."""

    df: Optional[pd.DataFrame]
    metadata: Dict[str, Any]


@dataclass
class PredictorBreakpointResult:
    """Result from compute_predictor_breakpoints."""

    df: Optional[pd.DataFrame]
    metadata: Dict[str, Any]
