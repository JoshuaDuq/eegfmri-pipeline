"""
CLI Subcommand Modules
======================

Each module contains the parser setup and run function for a specific pipeline.
The main launcher imports these and registers them with argparse.

Modules:
- behavior: Behavior correlation analysis
- features: Feature extraction
- erp: Event-related potential analysis
- tfr: Time-frequency visualization
- decoding: ML-based prediction
- preprocessing: Raw-to-BIDS and behavior merge
"""

from eeg_pipeline.cli.behavior import setup_behavior_parser, run_behavior
from eeg_pipeline.cli.features import setup_features_parser, run_features
from eeg_pipeline.cli.erp import setup_erp_parser, run_erp
from eeg_pipeline.cli.tfr import setup_tfr_parser, run_tfr
from eeg_pipeline.cli.decoding import setup_decoding_parser, run_decoding
from eeg_pipeline.cli.preprocessing import setup_preprocessing_parser, run_preprocessing

__all__ = [
    "setup_behavior_parser",
    "run_behavior",
    "setup_features_parser",
    "run_features",
    "setup_erp_parser",
    "run_erp",
    "setup_tfr_parser",
    "run_tfr",
    "setup_decoding_parser",
    "run_decoding",
    "setup_preprocessing_parser",
    "run_preprocessing",
]
