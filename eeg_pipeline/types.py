"""
Type definitions for the EEG pipeline.

This module provides type hints, protocols, and dataclasses used throughout
the pipeline for type safety and IDE support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas as pd

# Type aliases
PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, List[float], pd.Series]
FrequencyBand = Tuple[float, float]
CorrelationMethod = Literal["spearman", "pearson"]
AlignMode = Literal["strict", "warn", "none"]


###################################################################
# Configuration Protocol
###################################################################


@runtime_checkable
class EEGConfig(Protocol):
    """Protocol for configuration objects."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get nested config value using dot notation."""
        ...

    @property
    def deriv_root(self) -> Path:
        """Path to derivatives directory."""
        ...

    @property
    def bids_root(self) -> Path:
        """Path to BIDS root directory."""
        ...


###################################################################
# Feature Extraction Types
###################################################################


class FrequencyBands(TypedDict, total=False):
    """Frequency band definitions."""

    delta: Tuple[float, float]
    theta: Tuple[float, float]
    alpha: Tuple[float, float]
    beta: Tuple[float, float]
    gamma: Tuple[float, float]
    low_gamma: Tuple[float, float]
    high_gamma: Tuple[float, float]


@dataclass
class FeatureResult:
    """Container for feature extraction results."""

    df: pd.DataFrame
    columns: List[str]
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def empty(self) -> bool:
        return self.df.empty


# Re-export ValidationResult from utils.validation to avoid duplication
from eeg_pipeline.utils.validation import ValidationResult


###################################################################
# Correlation Types
###################################################################


@dataclass
class CorrelationResult:
    """Result of a correlation analysis."""

    r: float
    p: float
    n: int
    method: CorrelationMethod = "spearman"
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    r_partial: Optional[float] = None
    p_partial: Optional[float] = None
    p_perm: Optional[float] = None
    q: Optional[float] = None  # FDR-corrected p-value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r": self.r,
            "p": self.p,
            "n": self.n,
            "method": self.method,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "r_partial": self.r_partial,
            "p_partial": self.p_partial,
            "p_perm": self.p_perm,
            "q": self.q,
        }


###################################################################
# Decoding Types
###################################################################


@dataclass
class DecodingResult:
    """Result of a decoding analysis."""

    y_true: np.ndarray
    y_pred: np.ndarray
    r: float
    r2: float
    mse: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    p_perm: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class CVFoldResult:
    """Result from a single cross-validation fold."""

    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    best_params: Optional[Dict[str, Any]] = None
    subject: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold": self.fold,
            "y_true": self.y_true.tolist(),
            "y_pred": self.y_pred.tolist(),
            "test_idx": self.test_idx.tolist(),
            "best_params": self.best_params,
            "subject": self.subject,
        }


###################################################################
# Pipeline Types
###################################################################


@dataclass
class SubjectResult:
    """Result of processing a single subject."""

    subject: str
    success: bool
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    outputs: Dict[str, Path] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of batch processing multiple subjects."""

    subjects: List[str]
    results: List[SubjectResult]
    total_duration: float

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def failed_subjects(self) -> List[str]:
        return [r.subject for r in self.results if not r.success]


###################################################################
# Callback Types
###################################################################

ProgressCallback = Callable[[int, int, str], None]
LoggerType = Any  # logging.Logger, but avoid import


###################################################################
# Constants
###################################################################

DEFAULT_BANDS: FrequencyBands = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}

EPSILON = 1e-12
