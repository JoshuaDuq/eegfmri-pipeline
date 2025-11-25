from pathlib import Path
from typing import Optional, List, Tuple
import logging

import numpy as np

from eeg_pipeline.utils.config.loader import load_settings


###################################################################
# Shared Utility Functions
###################################################################


def parse_pow_column(col: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse power column names consistently: pow_{band}_{channel}_{bin} or pow_{band}_{channel}
    Returns (band, channel, bin_label)
    """
    parts = str(col).split("_")
    if len(parts) < 3 or not parts[0].startswith("pow"):
        return None
    band = parts[1]
    if len(parts) == 3:
        channel = parts[2]
        bin_label = "plateau"
    else:
        channel = "_".join(parts[2:-1])
        bin_label = parts[-1]
    return band, channel, bin_label


def extract_bin_token(col_name: str) -> Optional[str]:
    """Extract temporal bin token from pow column name."""
    parsed = parse_pow_column(col_name)
    if parsed is None:
        return None
    _, _, bin_label = parsed
    return bin_label


###################################################################
# Validation Functions
###################################################################


def validate_subjects_for_group_analysis(subjects: List[str], logger: logging.Logger, config=None) -> bool:
    """
    Validate that there are enough subjects for group-level analysis.
    
    Parameters
    ----------
    subjects : List[str]
        List of subject identifiers
    logger : logging.Logger
        Logger instance
    config : optional
        Configuration object
        
    Returns
    -------
    bool
        True if validation passes, False otherwise
    """
    if config is None:
        config = load_settings()
    min_subjects_for_group = config.get("analysis.min_subjects_for_group", 10)
    if subjects is None or len(subjects) < min_subjects_for_group:
        n_found = len(subjects) if subjects else 0
        logger.warning(
            f"Scientific Validity Warning: Group-level aggregation requires at least {min_subjects_for_group} subjects "
            f"for reliable statistical inference (t-tests, correlations). "
            f"Found only {n_found} subject(s); skipping group-level aggregation to prevent misleading results."
        )
        return False
    return True

