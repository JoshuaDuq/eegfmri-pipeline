from __future__ import annotations

from pathlib import Path


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
