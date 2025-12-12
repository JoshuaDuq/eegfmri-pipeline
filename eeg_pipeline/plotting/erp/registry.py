from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

PlotterFunc = Callable[[Any, Dict[str, Path]], None]


class ERPPlotRegistry:
    """Registry for ERP plotting functions."""

    _registry: List[Tuple[str, PlotterFunc]] = []

    @classmethod
    def register(cls, name: str):
        def decorator(func: PlotterFunc):
            cls._registry.append((name, func))
            return func

        return decorator

    @classmethod
    def get_plotters(cls) -> List[Tuple[str, PlotterFunc]]:
        return list(cls._registry)


@dataclass
class ERPPlotContext:
    subject: str
    task: str
    config: Any
    plots_dir: Path
    epochs: Any
    erp_cfg: Dict[str, Any]
    logger: logging.Logger


class ERPPlotManager:
    """Run ERP plotters registered to the ERPPlotRegistry."""

    def __init__(self, ctx: ERPPlotContext):
        self.ctx = ctx
        self.logger = ctx.logger
        self.saved_plots: Dict[str, Path] = {}

    def run_all(self) -> Dict[str, Path]:
        for name, func in ERPPlotRegistry.get_plotters():
            self._run_single(name, func)
        return self.saved_plots

    def run_selected(self, plot_names: List[str]) -> Dict[str, Path]:
        names = set(plot_names)
        for name, func in ERPPlotRegistry.get_plotters():
            if name in names:
                self._run_single(name, func)
        return self.saved_plots

    def _run_single(self, name: str, func: PlotterFunc) -> None:
        try:
            self.logger.info(f"Running ERP plotter: {name}")
            func(self.ctx, self.saved_plots)
        except Exception as exc:
            self.logger.error(f"ERP plotter '{name}' failed: {exc}", exc_info=True)

