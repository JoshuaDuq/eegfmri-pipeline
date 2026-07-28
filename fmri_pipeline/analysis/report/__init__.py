"""Rendering layer for fMRI post-preprocessing reports.

Importing this package fixes the Matplotlib backend. That is global on purpose:
a backend cannot be scoped to a context manager, and pipeline processes have no
display. Everything else in this package is scoped -- see
:func:`fmri_pipeline.analysis.report.style.plot_context`.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=False)
