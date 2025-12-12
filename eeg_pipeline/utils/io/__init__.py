"""
I/O utilities for the EEG pipeline.

This module provides file I/O operations, path management, and logging utilities.

Submodules:
- paths: Core path utilities for derivatives and directories
- tsv: TSV reading and writing helpers
- logging: Subject/group logger helpers
- plotting: Matplotlib setup utilities
- decoding: Decoding-specific I/O utilities (import separately to avoid circular imports)
"""

from .paths import (
    ensure_dir,
    deriv_features_path,
    deriv_stats_path,
    deriv_plots_path,
)
from .tsv import (
    read_tsv,
    write_tsv,
)
from .logging import (
    get_subject_logger,
    get_group_logger,
)
from .plotting import setup_matplotlib

# Decoding functions - import lazily to avoid circular imports
# Import these directly: from eeg_pipeline.utils.io.decoding import ...

__all__ = [
    "ensure_dir",
    "read_tsv",
    "write_tsv",
    "deriv_features_path",
    "deriv_stats_path",
    "deriv_plots_path",
    "get_subject_logger",
    "get_group_logger",
    "setup_matplotlib",
]
