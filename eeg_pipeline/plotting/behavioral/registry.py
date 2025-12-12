"""
Behavioral Visualization Registry
==================================

Registry-based orchestration for behavioral correlation plots.
Consolidates plotting functions from scatter.py, temporal.py, dose_response.py, etc.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PlotterFunc = Callable[[Any, Dict[str, Path]], None]


class BehaviorPlotRegistry:
    """Registry for behavioral plotting functions."""
    _registry: Dict[str, List[Tuple[str, PlotterFunc]]] = defaultdict(list)
    
    @classmethod
    def register(cls, category: str, name: str = None):
        """Decorator to register a plotting function."""
        def decorator(func: PlotterFunc):
            func_name = name or func.__name__
            cls._registry[category].append((func_name, func))
            return func
        return decorator
        
    @classmethod
    def get_categories(cls) -> List[str]:
        return list(cls._registry.keys())
        
    @classmethod
    def get_plotters(cls, category: str) -> List[Tuple[str, PlotterFunc]]:
        return cls._registry.get(category, [])
    
    @classmethod
    def get_all_plotters(cls) -> List[Tuple[str, str, PlotterFunc]]:
        """Get all plotters as (category, name, func) tuples."""
        result = []
        for category, plotters in cls._registry.items():
            for name, func in plotters:
                result.append((category, name, func))
        return result


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


class BehaviorPlotManager:
    """Orchestrates the execution of registered behavioral plotters."""
    
    def __init__(self, ctx: BehaviorPlotContext):
        self.ctx = ctx
        self.logger = ctx.logger
        self.saved_plots: Dict[str, Path] = {}
        
    def run_category(self, category: str) -> None:
        """Run all plotters for a specific category."""
        plotters = BehaviorPlotRegistry.get_plotters(category)
        if not plotters:
            self.logger.debug(f"No plotters found for category '{category}'")
            return
            
        n_plotters = len(plotters)
        self.logger.info(f"Generating {n_plotters} plots for '{category}'...")
        
        from eeg_pipeline.plotting.style import use_style
        
        with use_style():
            for idx, (name, func) in enumerate(plotters, 1):
                try:
                    self.logger.info(f"  [{idx}/{n_plotters}] {name}...")
                    func(self.ctx, self.saved_plots)
                except Exception as e:
                    self.logger.error(f"Plotter '{name}' failed: {e}", exc_info=True)
                    
    def run_all(self) -> Dict[str, Path]:
        """Run all registered plotters."""
        categories = BehaviorPlotRegistry.get_categories()
        preferred_order = [
            "psychometrics", "scatter", "temporal", "dose_response", 
            "mediation", "moderation", "diagnostics", "summary"
        ]
        
        ordered = [c for c in preferred_order if c in categories]
        ordered += [c for c in categories if c not in preferred_order]
        
        for cat in ordered:
            self.run_category(cat)
            
        return self.saved_plots
    
    def run_selected(self, plot_names: List[str]) -> Dict[str, Path]:
        """Run only selected plotters by name."""
        all_plotters = BehaviorPlotRegistry.get_all_plotters()
        
        from eeg_pipeline.plotting.style import use_style
        
        with use_style():
            for category, name, func in all_plotters:
                if name in plot_names:
                    try:
                        self.logger.info(f"Running {name}...")
                        func(self.ctx, self.saved_plots)
                    except Exception as e:
                        self.logger.error(f"Plotter '{name}' failed: {e}", exc_info=True)
        
        return self.saved_plots


__all__ = [
    "BehaviorPlotRegistry",
    "BehaviorPlotContext",
    "BehaviorPlotManager",
    "PlotterFunc",
]
