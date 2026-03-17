"""EEG analysis modules."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


__all__ = [
    "behavior",
    "machine_learning",
    "features",
    "utilities",
]

_SUBMODULES = {
    "behavior": "eeg_pipeline.analysis.behavior",
    "machine_learning": "eeg_pipeline.analysis.machine_learning",
    "features": "eeg_pipeline.analysis.features",
    "utilities": "eeg_pipeline.analysis.utilities",
}


def __getattr__(name: str) -> ModuleType:
    try:
        module_path = _SUBMODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return import_module(module_path)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
