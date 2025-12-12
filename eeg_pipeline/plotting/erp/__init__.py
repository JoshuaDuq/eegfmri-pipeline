"""
ERP plotting module.

Provides functions for ERP visualization including pain contrasts and temperature analysis.
High-level visualization entrypoints are defined in the pipeline layer (`pipelines.viz.erp`)
to keep orchestration separate from plotting primitives.
"""

from .contrasts import erp_contrast_pain
from .temperature import erp_by_temperature

# Orchestration lives in the pipeline layer; re-export for convenience/backward compatibility.
from eeg_pipeline.pipelines.viz.erp import (
    visualize_subject_erp,
    visualize_erp_for_subjects,
)

__all__ = [
    "erp_contrast_pain",
    "erp_by_temperature",
    "visualize_subject_erp",
    "visualize_erp_for_subjects",
]
