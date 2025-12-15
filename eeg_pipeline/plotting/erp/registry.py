from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from eeg_pipeline.plotting.core.registry import (
    FlatPlotManager,
    FlatPlotRegistry,
    PlotterFunc,
)


class ERPPlotRegistry(FlatPlotRegistry["ERPPlotContext"]):
    """Registry for ERP plotting functions."""


@dataclass
class ERPPlotContext:
    subject: str
    task: str
    config: Any
    plots_dir: Path
    epochs: Any
    erp_cfg: Dict[str, Any]
    logger: logging.Logger


class ERPPlotManager(FlatPlotManager["ERPPlotContext"]):
    """Run ERP plotters registered to the ERPPlotRegistry."""

    def __init__(self, ctx: ERPPlotContext):
        super().__init__(ctx, logger=ctx.logger)

    def run_all(self) -> Dict[str, Path]:
        return super().run_all(plotters=ERPPlotRegistry.get_plotters())

    def run_selected(self, plot_names: List[str]) -> Dict[str, Path]:
        return super().run_selected(plot_names, plotters=ERPPlotRegistry.get_plotters())

    def _run_single(self, name: str, func: PlotterFunc) -> None:
        try:
            self.logger.info(f"Running ERP plotter: {name}")
            func(self.ctx, self.saved_plots)
        except Exception as exc:
            self.logger.error(f"ERP plotter '{name}' failed: {exc}", exc_info=True)
