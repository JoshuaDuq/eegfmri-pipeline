from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence


def normalize_trial_type_list(value: Any) -> Optional[List[str]]:
    """
    Normalize a user-provided list of trial_type values.

    Accepts:
    - None
    - list/tuple of values (strings or other scalars)
    - comma-separated string
    """
    if value is None:
        return None

    items: Sequence[Any]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        items = [part.strip() for part in s.split(",")]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]

    out: List[str] = []
    seen = set()
    for it in items:
        if it is None:
            continue
        s = str(it).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)

    return out or None


def filter_trial_types(trial_types: Iterable[Any], allowed: Optional[Sequence[str]]) -> List[str]:
    """
    Filter a sequence of trial_type values using an allow-list (case-sensitive).

    Returns normalized strings for kept trial types.
    """
    allowed_norm = normalize_trial_type_list(allowed)
    if not allowed_norm:
        return [str(t) for t in trial_types]

    allow_set = set(allowed_norm)
    return [str(t) for t in trial_types if str(t) in allow_set]

