from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

CtxT = TypeVar("CtxT")
PlotterFunc = Callable[[CtxT, Dict[str, Path]], None]


class CategorizedPlotRegistry(Generic[CtxT]):
    _registry: Dict[str, List[Tuple[str, PlotterFunc[CtxT]]]] = defaultdict(list)

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
        self.logger = logger or getattr(ctx, "logger", logging.getLogger(__name__))
        self.saved_plots: Dict[str, Path] = {}

    def run_category(self, category: str, *, plotters: List[Tuple[str, PlotterFunc[CtxT]]]) -> None:
        if not plotters:
            self.logger.debug(f"No plotters found for category '{category}'")
            return

        n_plotters = len(plotters)
        self.logger.info(f"Generating {n_plotters} plots for '{category}'...")

        for idx, (name, func) in enumerate(plotters, 1):
            try:
                self.logger.info(f"  [{idx}/{n_plotters}] {name}...")
                func(self.ctx, self.saved_plots)
            except Exception as exc:
                self.logger.error(f"Plotter '{name}' failed: {exc}", exc_info=True)

    def run_selected(
        self,
        plot_names: List[str],
        *,
        all_plotters: List[Tuple[str, str, PlotterFunc[CtxT]]],
    ) -> Dict[str, Path]:
        names = set(plot_names)
        for _category, name, func in all_plotters:
            if name not in names:
                continue
            try:
                self.logger.info(f"Running {name}...")
                func(self.ctx, self.saved_plots)
            except Exception as exc:
                self.logger.error(f"Plotter '{name}' failed: {exc}", exc_info=True)

        return self.saved_plots


class FlatPlotRegistry(Generic[CtxT]):
    _registry: List[Tuple[str, PlotterFunc[CtxT]]] = []

    @classmethod
    def register(cls, name: str):
        def decorator(func: PlotterFunc[CtxT]):
            cls._registry.append((name, func))
            return func

        return decorator

    @classmethod
    def get_plotters(cls) -> List[Tuple[str, PlotterFunc[CtxT]]]:
        return list(cls._registry)


@dataclass
class FlatPlotManager(Generic[CtxT]):
    ctx: CtxT
    logger: logging.Logger
    saved_plots: Dict[str, Path]

    def __init__(self, ctx: CtxT, *, logger: Optional[logging.Logger] = None):
        self.ctx = ctx
        self.logger = logger or getattr(ctx, "logger", logging.getLogger(__name__))
        self.saved_plots = {}

    def run_all(self, *, plotters: List[Tuple[str, PlotterFunc[CtxT]]]) -> Dict[str, Path]:
        for name, func in plotters:
            self._run_single(name, func)
        return self.saved_plots

    def run_selected(
        self,
        plot_names: List[str],
        *,
        plotters: List[Tuple[str, PlotterFunc[CtxT]]],
    ) -> Dict[str, Path]:
        names = set(plot_names)
        for name, func in plotters:
            if name in names:
                self._run_single(name, func)
        return self.saved_plots

    def _run_single(self, name: str, func: PlotterFunc[CtxT]) -> None:
        try:
            self.logger.info(f"Running plotter: {name}")
            func(self.ctx, self.saved_plots)
        except Exception as exc:
            self.logger.error(f"Plotter '{name}' failed: {exc}", exc_info=True)
