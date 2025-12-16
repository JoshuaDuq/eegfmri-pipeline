"""Infrastructure utilities for the EEG pipeline.

This package contains low-level modules that are safe to depend on from anywhere:
- filesystem/path resolution
- logging setup
- table I/O
"""

from __future__ import annotations

from . import logging as _logging
from . import paths as _paths
from . import tsv as _tsv

from .logging import *
from .paths import *
from .tsv import *

__all__ = list(_paths.__all__) + list(_logging.__all__) + list(_tsv.__all__)
