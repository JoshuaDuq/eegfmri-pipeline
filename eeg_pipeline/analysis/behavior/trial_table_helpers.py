from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from eeg_pipeline.context.behavior import BehaviorContext


def sanitize_path_component(value: str) -> str:
    """Sanitize a string for use as a single path component."""
    value = str(value).strip()
    if not value:
        return "all"
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    out = []
    for ch in value:
        out.append(ch if ch in allowed else "_")
    cleaned = "".join(out).strip("._-")
    return cleaned if cleaned else "all"


def feature_folder_from_list(feature_files: List[str]) -> str:
    items = [str(x).strip() for x in feature_files if str(x).strip()]
    items = [x for x in items if x.lower() != "all"]
    if not items:
        return "all"
    return sanitize_path_component("_".join(sorted(items)))


def feature_folder_from_context(ctx: BehaviorContext) -> str:
    raw = ctx.selected_feature_files or ctx.feature_categories or []
    return feature_folder_from_list(list(raw))


def normalize_trial_table_feature_selection(feature_files: Optional[List[str]]) -> List[str]:
    from eeg_pipeline.utils.data.trial_table import normalize_trial_table_feature_selection as _normalize

    return _normalize(feature_files)


def trial_table_suffix_from_features(feature_files: Optional[List[str]]) -> str:
    from eeg_pipeline.utils.data.trial_table import trial_table_suffix_from_features as _suffix

    return _suffix(feature_files)


def trial_table_feature_folder_from_features(feature_files: Optional[List[str]]) -> str:
    from eeg_pipeline.utils.data.trial_table import trial_table_feature_folder_from_features as _folder

    return _folder(feature_files)


def trial_table_suffix_from_context(ctx: BehaviorContext) -> str:
    selected = ctx.selected_feature_files or ctx.feature_categories or []
    return trial_table_suffix_from_features(selected)


def trial_table_metadata_path(table_path: Path) -> Path:
    return table_path.parent / f"{table_path.stem}.metadata.json"


def validate_trial_table_contract_metadata(
    ctx: BehaviorContext,
    table_path: Path,
    df: pd.DataFrame,
) -> None:
    """Validate trial-table metadata contract (if present)."""
    from eeg_pipeline.utils.data.trial_table import validate_trial_table_contract

    meta_path = trial_table_metadata_path(table_path)
    if not meta_path.exists():
        ctx.logger.warning("Trial table metadata missing: %s", meta_path)
        return
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid trial table metadata JSON: {meta_path}") from exc

    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid trial table metadata payload (not a dict): {meta_path}")

    errors = validate_trial_table_contract(df, metadata)
    if errors:
        raise ValueError(
            f"Trial table contract validation failed for {table_path.name}: " + "; ".join(errors)
        )


def compute_feature_signature(ctx: BehaviorContext) -> str:
    """Compute hash signature of feature tables for caching."""
    parts = []
    for name, df in ctx.iter_feature_tables():
        if df is None or df.empty:
            continue
        column_names = ",".join(str(c) for c in df.columns)
        parts.append(f"{name}:{df.shape[0]}:{df.shape[1]}:{column_names}")
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_trial_table_input_hash(ctx: BehaviorContext) -> str:
    """Compute a deterministic input hash for trial-table stage caching."""
    parts: List[str] = [
        f"subject={ctx.subject}",
        f"task={ctx.task}",
        f"feature_signature={compute_feature_signature(ctx)}",
    ]

    events = getattr(ctx, "aligned_events", None)
    if isinstance(events, pd.DataFrame) and not events.empty:
        try:
            events_hash_raw = pd.util.hash_pandas_object(
                events.reset_index(drop=True),
                index=True,
            ).to_numpy(dtype=np.uint64)
            events_hash = hashlib.sha256(events_hash_raw.tobytes()).hexdigest()
        except Exception:
            cols = ",".join(str(c) for c in events.columns)
            events_hash = f"fallback:{len(events)}:{events.shape[1]}:{cols}"
        parts.append(f"events_hash={events_hash}")
    else:
        parts.append("events_hash=none")

    feature_paths = getattr(ctx, "feature_paths", {}) or {}
    for name, path in sorted(feature_paths.items()):
        try:
            st = Path(path).stat()
        except Exception:
            continue
        parts.append(f"feature_path:{name}:{Path(path).resolve()}:{st.st_size}:{st.st_mtime_ns}")

    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def find_trial_table_path(stats_dir: Path, feature_files: Optional[List[str]] = None) -> Optional[Path]:
    """Find trial table path using shared trial-table resolution helpers."""
    from eeg_pipeline.utils.data.trial_table import find_trial_table_path as _find

    return _find(stats_dir, feature_files=feature_files)
