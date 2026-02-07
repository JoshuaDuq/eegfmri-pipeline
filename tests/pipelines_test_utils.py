from __future__ import annotations


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


class DummyProgress:
    def start(self, *_args, **_kwargs):
        return None

    def step(self, *_args, **_kwargs):
        return None

    def subject_start(self, *_args, **_kwargs):
        return None

    def subject_done(self, *_args, **_kwargs):
        return None

    def complete(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


class NoopProgress:
    def start(self, *_a, **_k):
        return None

    def complete(self, *_a, **_k):
        return None

    def step(self, *_a, **_k):
        return None

    def subject_start(self, *_a, **_k):
        return None

    def subject_done(self, *_a, **_k):
        return None


class NoopBatchProgress:
    def __init__(self, subjects, logger, desc):
        self.subjects = subjects

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def start_subject(self, _subject):
        return 0.0

    def finish_subject(self, _subject, _start_time):
        return None

