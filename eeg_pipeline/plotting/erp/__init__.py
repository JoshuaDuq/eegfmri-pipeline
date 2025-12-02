"""
ERP plotting module.

Provides functions for ERP visualization including pain contrasts and temperature analysis.
"""

from .contrasts import erp_contrast_pain
from .temperature import erp_by_temperature
from .viz import (
    visualize_subject_erp,
    visualize_erp_for_subjects,
)

__all__ = [
    "erp_contrast_pain",
    "erp_by_temperature",
    "visualize_subject_erp",
    "visualize_erp_for_subjects",
]

