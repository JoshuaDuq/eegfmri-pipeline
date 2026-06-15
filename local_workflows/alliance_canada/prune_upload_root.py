#!/usr/bin/env python3
"""Prune uploaded files that are absent from one or more upload manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, action="append", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    try:
        prune_root(
            root=args.root,
            manifests=tuple(args.manifest),
            apply=args.apply,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def prune_root(*, root: Path, manifests: tuple[Path, ...], apply: bool) -> None:
    resolved_root = root.expanduser()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"Prune root does not exist: {resolved_root}")

    allowed = _load_allowed_paths(manifests)
    existing = _existing_files(resolved_root)
    delete_candidates = [path for path in existing if path.as_posix() not in allowed]
    action = "apply" if apply else "dry-run"
    print(f"[{action}] {resolved_root}: {len(delete_candidates)} file(s) outside manifest.")
    for rel in delete_candidates:
        print(rel.as_posix())

    if not apply:
        return

    for rel in delete_candidates:
        (resolved_root / rel).unlink()
    _remove_empty_directories(resolved_root)


def _load_allowed_paths(manifests: tuple[Path, ...]) -> set[str]:
    if not manifests:
        raise ValueError("At least one manifest is required.")

    allowed: set[str] = set()
    for manifest in manifests:
        if not manifest.exists():
            raise FileNotFoundError(f"Upload manifest does not exist: {manifest}")
        for line in manifest.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            _validate_relative_manifest_path(text, manifest)
            allowed.add(Path(text).as_posix())

    if not allowed:
        raise ValueError(f"Upload manifests contain no file paths: {manifests}")
    return allowed


def _validate_relative_manifest_path(path: str, manifest: Path) -> None:
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Manifest path must be relative and stay inside root: {manifest}: {path}")


def _existing_files(root: Path) -> list[Path]:
    return sorted(
        (path.relative_to(root) for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.as_posix(),
    )


def _remove_empty_directories(root: Path) -> None:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            continue


if __name__ == "__main__":
    raise SystemExit(main())
