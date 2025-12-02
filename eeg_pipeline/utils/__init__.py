"""
Utility modules for the EEG pipeline.

This package provides low-level utilities used throughout the pipeline.
These modules should not depend on higher-level packages like plotting or analysis.

Submodules:
- analysis: Analysis utilities (statistics, TFR computation, windowing, signal metrics)
- config: Configuration loading and YAML file management
- data: Data loading, feature manipulation, and file I/O for data structures
- io: File I/O utilities (reading/writing TSV files, path management, figure saving)
- types: Progress callback types and utilities
- validation: Data validation utilities
- progress: Progress tracking and reporting
- decorators: Function decorators for common patterns
"""

__all__ = []
