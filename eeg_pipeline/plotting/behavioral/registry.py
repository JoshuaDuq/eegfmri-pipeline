"""
Behavioral Visualization Registry
==================================

Registry-based orchestration for behavioral correlation plots.
Consolidates plotting functions from scatter.py, temporal.py, dose_response.py, etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from eeg_pipeline.plotting.core.registry import (
    CategorizedPlotManager,
    CategorizedPlotRegistry,
    PlotterFunc,
)


class BehaviorPlotRegistry(CategorizedPlotRegistry["BehaviorPlotContext"]):
    """Registry for behavioral plotting functions."""


@dataclass
class BehaviorPlotContext:
    """Context for behavioral visualization."""
    subject: str
    task: str
    config: Any
    logger: logging.Logger
    deriv_root: Path
    plots_dir: Path
    stats_dir: Path
    use_spearman: bool = True
    
    rating_stats: Optional[Any] = None
    temp_stats: Optional[Any] = None
    all_results: List[Any] = None
    
    def __post_init__(self):
        if self.all_results is None:
            self.all_results = []


class BehaviorPlotManager(CategorizedPlotManager["BehaviorPlotContext"]):
    """Orchestrates the execution of registered behavioral plotters."""

    def __init__(self, ctx: "BehaviorPlotContext"):
        super().__init__(ctx, logger=ctx.logger)

    def run_category(self, category: str) -> None:
        plotters = BehaviorPlotRegistry.get_plotters(category)
        from eeg_pipeline.plotting.style import use_style

        with use_style():
            super().run_category(category, plotters=plotters)

    def run_all(self) -> Dict[str, Path]:
        categories = BehaviorPlotRegistry.get_categories()
        preferred_order = [
            "psychometrics",
            "scatter",
            "temporal",
            "dose_response",
            "mediation",
            "moderation",
            "diagnostics",
            "summary",
        ]

        ordered = [c for c in preferred_order if c in categories]
        ordered += [c for c in categories if c not in preferred_order]

        for cat in ordered:
            self.run_category(cat)

        return self.saved_plots

    def run_selected(self, plot_names: List[str]) -> Dict[str, Path]:
        all_plotters = BehaviorPlotRegistry.get_all_plotters()
        from eeg_pipeline.plotting.style import use_style

        with use_style():
            return super().run_selected(plot_names, all_plotters=all_plotters)


__all__ = [
    "BehaviorPlotRegistry",
    "BehaviorPlotContext",
    "BehaviorPlotManager",
    "PlotterFunc",
]
