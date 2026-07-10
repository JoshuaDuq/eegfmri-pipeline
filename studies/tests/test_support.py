from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


class DotConfig(dict):
    """Config helper supporting dotted key access and attribute access."""

    def get(self, key, default=None):  # type: ignore[override]
        if isinstance(key, str) and "." in key:
            cur = self
            for part in key.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return default
            return cur
        return super().get(key, default)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)


def validity_figure_test_config(temperatures: Sequence[float]) -> dict:
    from studies.pain_study.study1.config.loader import load_study1_config

    figures = copy.deepcopy(load_study1_config()["study1"]["figures"])
    validity = figures["validity"]
    validity["temperatures"] = [float(value) for value in temperatures]
    validity["bootstrap"].update(iterations=20, max_invalid_fraction=0.20)
    return figures
