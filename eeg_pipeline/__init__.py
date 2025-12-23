"""
EEG Analysis Pipeline

A comprehensive pipeline for EEG data processing, feature extraction,
behavioral correlation analysis, and machine learning decoding.

Package Structure
-----------------
- analysis/: Core analysis modules (features, behavior, decoding, group)
- pipelines/: High-level pipeline orchestration
- plotting/: Visualization utilities
- preprocessing/: BIDS conversion and event processing
- types: Type definitions and protocols

Quick Start
-----------
>>> from eeg_pipeline.pipelines import FeaturePipeline
>>> pipeline = FeaturePipeline()
>>> pipeline.run_batch(["0001", "0002"])

>>> from eeg_pipeline.pipelines import BehaviorPipeline
>>> pipeline = BehaviorPipeline()
>>> pipeline.run_batch(["0001", "0002"])
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "EEGConfig",
    "ConfigLike",
    "CorrelationResult",
    "DecodingResult",
    "ValidationResult",
    "BandData",
    "PSDData",
    "TimeWindows",
    "PrecomputedQC",
    "PrecomputedData",
]


def __getattr__(name: str):
    if name in {
        "EEGConfig",
        "ConfigLike",
        "CorrelationResult",
        "DecodingResult",
        "ValidationResult",
        "BandData",
        "PSDData",
        "TimeWindows",
        "PrecomputedQC",
        "PrecomputedData",
    }:
        try:
            import importlib

            types_mod = importlib.import_module("eeg_pipeline.types")
            return getattr(types_mod, name)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Cannot import '{name}' because optional dependencies are missing. "
                "Install the project's requirements to use eeg_pipeline types."
            ) from exc

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
