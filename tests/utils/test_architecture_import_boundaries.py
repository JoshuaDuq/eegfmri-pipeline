from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from tests import REPO_ROOT


@dataclass(frozen=True)
class Rule:
    source_prefix: str
    forbidden_prefix: str
    description: str


RULES: tuple[Rule, ...] = (
    Rule(
        source_prefix="eeg_pipeline.analysis",
        forbidden_prefix="eeg_pipeline.cli",
        description="analysis must not depend on cli",
    ),
    Rule(
        source_prefix="eeg_pipeline.utils",
        forbidden_prefix="eeg_pipeline.cli",
        description="utils must not depend on cli",
    ),
    Rule(
        source_prefix="eeg_pipeline.analysis.features",
        forbidden_prefix="eeg_pipeline.analysis.machine_learning",
        description="feature extraction must not depend on machine learning orchestration",
    ),
    Rule(
        source_prefix="eeg_pipeline.pipelines",
        forbidden_prefix="eeg_pipeline.cli",
        description="pipelines must not depend on cli",
    ),
)


def _module_name_from_path(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    return ".".join(rel.parts)


def _iter_python_sources() -> list[Path]:
    roots = [
        REPO_ROOT / "eeg_pipeline",
        REPO_ROOT / "fmri_pipeline",
        REPO_ROOT / "scripts",
    ]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(root.rglob("*.py"))
    return sorted(paths)


def _imported_modules(tree: ast.AST) -> list[str]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    imported.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.append(node.module)
    return imported


def test_architecture_import_boundaries() -> None:
    violations: list[str] = []

    for path in _iter_python_sources():
        module_name = _module_name_from_path(path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue

        imports = _imported_modules(tree)
        for rule in RULES:
            if not module_name.startswith(rule.source_prefix):
                continue
            for imported in imports:
                if imported.startswith(rule.forbidden_prefix):
                    rel = path.relative_to(REPO_ROOT)
                    violations.append(
                        f"{rel}: {rule.description} "
                        f"(imports {imported})"
                    )

    assert not violations, "Architecture boundary violations:\n" + "\n".join(violations)
