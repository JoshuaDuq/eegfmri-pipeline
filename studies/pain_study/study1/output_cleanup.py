"""Study 1 output cleanup helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

WINDOWED_RANGE_SUFFIXES = ("active", "baseline")


def remove_appledouble_sidecars(root: Path) -> int:
    """Remove macOS AppleDouble sidecars from a study-owned output tree."""
    if not root.exists():
        return 0

    removed = 0
    for path in sorted(root.rglob("._*")):
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


def prune_windowed_feature_artifacts(
    *,
    feature_root: Path,
    subjects: Iterable[str],
    feature_families: Iterable[str],
    range_suffixes: Iterable[str] = WINDOWED_RANGE_SUFFIXES,
) -> int:
    """Remove redundant per-range Study 1 artifacts once merged canonical files exist."""
    suffixes = tuple(str(suffix).strip() for suffix in range_suffixes)
    if any(not suffix for suffix in suffixes):
        raise ValueError("Window suffixes must be non-empty.")

    removed = 0
    for subject_id in subjects:
        subject_label = str(subject_id).strip()
        if not subject_label:
            raise ValueError("Subject identifiers must be non-empty.")
        for family in feature_families:
            family_name = str(family).strip()
            if not family_name:
                raise ValueError("Feature family names must be non-empty.")

            family_dir = feature_root / subject_label / "eeg" / "features" / family_name
            metadata_dir = family_dir / "metadata"
            for suffix in suffixes:
                candidates = (
                    family_dir / f"features_{family_name}_{suffix}.parquet",
                    metadata_dir / f"features_{family_name}_{suffix}.json",
                    metadata_dir / f"extraction_config_{suffix}.json",
                )
                for candidate in candidates:
                    try:
                        if candidate.exists():
                            candidate.unlink()
                            removed += 1
                    except FileNotFoundError:
                        continue
    return removed


__all__ = [
    "prune_windowed_feature_artifacts",
    "remove_appledouble_sidecars",
]
