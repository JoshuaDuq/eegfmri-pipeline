"""Shared rendering conventions for the subject preprocessing HTML report."""

from eeg_pipeline.preprocessing.report.style import (
    DIVERGING_POWER_COLORMAP,
    OKABE_ITO,
    REPORT_IMAGE_FORMAT,
    REPORT_RASTER_IMAGE_FORMAT,
    apply_report_style,
    report_image_format,
    robust_symmetric_limit,
)

__all__ = [
    "DIVERGING_POWER_COLORMAP",
    "OKABE_ITO",
    "REPORT_IMAGE_FORMAT",
    "REPORT_RASTER_IMAGE_FORMAT",
    "apply_report_style",
    "report_image_format",
    "robust_symmetric_limit",
]
