from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar

CtxT = TypeVar("CtxT")
PlotterFunc = Callable[[CtxT, Dict[str, Path]], None]


def _resolve_logger(ctx: CtxT, logger: Optional[logging.Logger]) -> logging.Logger:
    if logger is not None:
        return logger
    if hasattr(ctx, "logger"):
        return ctx.logger
    return logging.getLogger(__name__)


class CategorizedPlotRegistry(Generic[CtxT]):
    """Base class for categorized plot registries.

    Each subclass gets its own isolated _registry dict via __init_subclass__.
    This prevents plotters registered in FeaturePlotRegistry from appearing
    in BehaviorPlotRegistry and vice versa.
    """
    _registry: Dict[str, List[Tuple[str, PlotterFunc[CtxT]]]]
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Each subclass gets its own fresh registry
        cls._registry = defaultdict(list)

    @classmethod
    def register(cls, category: str, name: Optional[str] = None):
        def decorator(func: PlotterFunc[CtxT]):
            func_name = name or func.__name__
            cls._registry[category].append((func_name, func))
            return func

        return decorator

    @classmethod
    def get_categories(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def get_plotters(cls, category: str) -> List[Tuple[str, PlotterFunc[CtxT]]]:
        return cls._registry.get(category, [])

    @classmethod
    def get_all_plotters(cls) -> List[Tuple[str, str, PlotterFunc[CtxT]]]:
        result: List[Tuple[str, str, PlotterFunc[CtxT]]] = []
        for category, plotters in cls._registry.items():
            for name, func in plotters:
                result.append((category, name, func))
        return result


class CategorizedPlotManager(Generic[CtxT]):
    def __init__(self, ctx: CtxT, *, logger: Optional[logging.Logger] = None):
        self.ctx = ctx
        self.logger = _resolve_logger(ctx, logger)
        self.saved_plots: Dict[str, Path] = {}

    def _execute_plotter(self, name: str, func: PlotterFunc[CtxT]) -> None:
        try:
            func(self.ctx, self.saved_plots)
        except Exception as exc:
            self.logger.error(f"Plotter '{name}' failed: {exc}", exc_info=True)
            raise RuntimeError(f"Plotter '{name}' failed") from exc

    def run_category(self, category: str, *, plotters: List[Tuple[str, PlotterFunc[CtxT]]]) -> None:
        if not plotters:
            self.logger.debug(f"No plotters found for category '{category}'")
            return

        n_plotters = len(plotters)
        self.logger.info(f"Generating {n_plotters} plots for '{category}'...")

        for idx, (name, func) in enumerate(plotters, 1):
            self.logger.info(f"  [{idx}/{n_plotters}] {name}...")
            self._execute_plotter(name, func)

    def run_selected(
        self,
        plot_names: List[str],
        *,
        all_plotters: List[Tuple[str, str, PlotterFunc[CtxT]]],
    ) -> Dict[str, Path]:
        selected_names = set(plot_names)
        for _category, name, func in all_plotters:
            if name in selected_names:
                self.logger.info(f"Running {name}...")
                self._execute_plotter(name, func)

        return self.saved_plots
