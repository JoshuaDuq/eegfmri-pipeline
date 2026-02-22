from __future__ import annotations

import ast
from pathlib import Path

from tests import REPO_ROOT


CLI_FACADE_FILES = (
    "eeg_pipeline/cli/commands/info.py",
    "eeg_pipeline/cli/commands/stats.py",
    "eeg_pipeline/cli/commands/features.py",
    "eeg_pipeline/cli/commands/plotting.py",
    "eeg_pipeline/cli/commands/behavior.py",
    "eeg_pipeline/cli/commands/machine_learning.py",
    "eeg_pipeline/cli/commands/preprocessing.py",
    "eeg_pipeline/cli/commands/utilities.py",
    "eeg_pipeline/cli/commands/validate.py",
)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_cli_facades_do_not_use_wildcard_imports() -> None:
    violations: list[str] = []
    for rel in CLI_FACADE_FILES:
        path = REPO_ROOT / rel
        tree = _parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if any(alias.name == "*" for alias in node.names):
                violations.append(rel)
                break
    assert not violations, "Wildcard imports are forbidden in CLI facades:\n" + "\n".join(sorted(violations))


def test_cli_facades_use_static_all_exports() -> None:
    violations: list[str] = []
    for rel in CLI_FACADE_FILES:
        path = REPO_ROOT / rel
        tree = _parse(path)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                continue
            if not isinstance(node.value, (ast.List, ast.Tuple)):
                violations.append(rel)
            break
    assert not violations, "CLI facades must define static __all__ sequences:\n" + "\n".join(sorted(violations))


def test_cli_facades_do_not_export_private_names() -> None:
    violations: list[str] = []
    for rel in CLI_FACADE_FILES:
        path = REPO_ROOT / rel
        tree = _parse(path)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                continue
            if not isinstance(node.value, (ast.List, ast.Tuple)):
                continue
            for elt in node.value.elts:
                if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
                    continue
                if elt.value.startswith("_"):
                    violations.append(f"{rel}: {elt.value}")
            break
    assert not violations, "CLI facades must not export private names:\n" + "\n".join(sorted(violations))
