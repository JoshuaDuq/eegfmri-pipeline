from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class GroupLevelResult:
    """Output contract for group-level analysis."""

    multilevel_correlations: Optional[pd.DataFrame] = None
    n_subjects: int = 0
    subjects: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrialTableResult:
    """Result from compute_trial_table."""

    df: pd.DataFrame
    metadata: Dict[str, Any]
