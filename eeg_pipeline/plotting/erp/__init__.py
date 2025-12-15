"""
ERP plotting module.

Provides functions for ERP visualization including pain contrasts and temperature analysis.
High-level visualization entrypoints are defined in the pipeline layer (`pipelines.viz.erp`)
to keep orchestration separate from plotting primitives.
"""

from .contrasts import erp_contrast_pain
from .temperature import erp_by_temperature

# Visualization orchestration wrappers (pipeline layer)
# These lightweight wrappers avoid an import-time dependency on
# `eeg_pipeline.pipelines.viz.erp`, preventing circular imports while
# preserving the public API.

def visualize_subject_erp(*args, **kwargs):
    from eeg_pipeline.pipelines.viz.erp import visualize_subject_erp as _impl

    return _impl(*args, **kwargs)


def visualize_erp_for_subjects(*args, **kwargs):
    from eeg_pipeline.pipelines.viz.erp import visualize_erp_for_subjects as _impl

    return _impl(*args, **kwargs)

__all__ = [
    "erp_contrast_pain",
    "erp_by_temperature",
    "visualize_subject_erp",
    "visualize_erp_for_subjects",
]
