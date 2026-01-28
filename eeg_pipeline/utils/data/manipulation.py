"""
Data Manipulation Utilities.

Common functions for manipulating DataFrames and other data structures.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column from a list of candidates.
    
    Args:
        df: DataFrame to search
        candidates: List of column names to try in order
    
    Returns:
        First matching column name, or None if no match
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None

