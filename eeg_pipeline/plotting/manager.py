"""
Visualization Manager
=====================

Registry-based orchestration for generating feature plots.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Any, Tuple

# We use Any for context to avoid circular imports during definition
# Ideally, define Protocol or abstract base class
PlotterFunc = Callable[[Any, Dict[str, Path]], None]

class VisualizationRegistry:
    """Registry for feature plotting functions."""
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


class VisualizationManager:
    """Orchestrates the execution of registered plotters."""
    
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = getattr(ctx, "logger", logging.getLogger(__name__))
        self.saved_plots: Dict[str, Path] = {}
        
    def run_category(self, category: str) -> None:
        """Run all plotters for a specific category."""
        plotters = VisualizationRegistry.get_plotters(category)
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
        categories = VisualizationRegistry.get_categories()
        preferred_order = [
            "power", "connectivity", "microstates", 
            "pac", "complexity", "burst", "erds", "aperiodic", "itpc"
        ]
        
        # Sort categories: preferred first, then others alpha
        ordered = [c for c in preferred_order if c in categories]
        ordered += [c for c in categories if c not in preferred_order]
        
        for cat in ordered:
            self.run_category(cat)
            
        return self.saved_plots
