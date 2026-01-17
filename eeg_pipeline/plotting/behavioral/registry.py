"""
Behavioral Visualization Registry
==================================

Registry-based orchestration for behavioral correlation plots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from eeg_pipeline.plotting.core.registry import (
    CategorizedPlotManager,
    CategorizedPlotRegistry,
)
from eeg_pipeline.plotting.style import use_style

CATEGORY_EXECUTION_ORDER = [
    "psychometrics",
    "scatter",
    "temporal",
    "dose_response",
    "summary",
]


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
    all_results: Optional[List[Any]] = None

    def __post_init__(self) -> None:
        if self.all_results is None:
            self.all_results = []


class BehaviorPlotManager(CategorizedPlotManager["BehaviorPlotContext"]):
    """Orchestrates the execution of registered behavioral plotters."""

    def __init__(self, ctx: "BehaviorPlotContext"):
        super().__init__(ctx, logger=ctx.logger)

    def run_category(self, category: str) -> None:
        plotters = BehaviorPlotRegistry.get_plotters(category)
        with use_style():
            super().run_category(category, plotters=plotters)

    def run_all(self) -> Dict[str, Path]:
        categories = BehaviorPlotRegistry.get_categories()
        category_set = set(categories)

        ordered = [cat for cat in CATEGORY_EXECUTION_ORDER if cat in category_set]
        remaining = [cat for cat in categories if cat not in CATEGORY_EXECUTION_ORDER]
        ordered.extend(remaining)

        for category in ordered:
            self.run_category(category)

        return self.saved_plots

    def run_selected(self, plot_names: List[str]) -> Dict[str, Path]:
        all_plotters = BehaviorPlotRegistry.get_all_plotters()
        with use_style():
            return super().run_selected(plot_names, all_plotters=all_plotters)


__all__ = [
    "BehaviorPlotRegistry",
    "BehaviorPlotContext",
    "BehaviorPlotManager",
]
