"""
I/O utilities for the EEG pipeline.

This package provides file I/O operations, path management, and logging utilities.

Submodules:
- paths: Core path utilities for derivatives and directories
- tsv: TSV/parquet reading and writing helpers
- logging: Subject/group logger helpers
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
    read_table,
    write_table,
)
from .logging import (
    get_subject_logger,
    get_group_logger,
)

__all__ = [
    "ensure_dir",
    "read_tsv",
    "write_tsv",
    "read_table",
    "write_table",
    "deriv_features_path",
    "deriv_stats_path",
    "deriv_plots_path",
    "get_subject_logger",
    "get_group_logger",
]
