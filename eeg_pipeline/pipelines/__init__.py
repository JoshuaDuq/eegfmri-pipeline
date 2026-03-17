"""Pipeline orchestration modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "PipelineBase",
    "FeaturePipeline",
    "extract_all_features",
    "extract_precomputed_features",
    "BehaviorPipeline",
    "BehaviorPipelineConfig",
    "BehaviorPipelineResults",
    "MLPipeline",
    "PreprocessingPipeline",
]

_EXPORTS = {
    "PipelineBase": ("eeg_pipeline.pipelines.base", "PipelineBase"),
    "FeaturePipeline": ("eeg_pipeline.pipelines.features", "FeaturePipeline"),
    "extract_all_features": ("eeg_pipeline.pipelines.features", "extract_all_features"),
    "extract_precomputed_features": (
        "eeg_pipeline.pipelines.features",
        "extract_precomputed_features",
    ),
    "BehaviorPipeline": ("eeg_pipeline.pipelines.behavior", "BehaviorPipeline"),
    "BehaviorPipelineConfig": (
        "eeg_pipeline.pipelines.behavior",
        "BehaviorPipelineConfig",
    ),
    "BehaviorPipelineResults": (
        "eeg_pipeline.pipelines.behavior",
        "BehaviorPipelineResults",
    ),
    "MLPipeline": ("eeg_pipeline.pipelines.machine_learning", "MLPipeline"),
    "PreprocessingPipeline": (
        "eeg_pipeline.pipelines.preprocessing",
        "PreprocessingPipeline",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_path, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_path)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
