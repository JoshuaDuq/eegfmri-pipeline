from __future__ import annotations

import ast

from tests import REPO_ROOT


PACKAGE_INIT_FILES = (
    "eeg_pipeline/analysis/__init__.py",
    "eeg_pipeline/analysis/machine_learning/__init__.py",
    "eeg_pipeline/utils/data/__init__.py",
    "eeg_pipeline/pipelines/__init__.py",
)


def _parse(rel_path: str) -> ast.Module:
    path = REPO_ROOT / rel_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_package_init_files_do_not_eagerly_import_internal_modules() -> None:
    violations: list[str] = []

    for rel_path in PACKAGE_INIT_FILES:
        module_name = rel_path.replace("/", ".").removesuffix(".__init__.py")
        tree = _parse(rel_path)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    violations.append(rel_path)
                    break
                if node.module and node.module.startswith(module_name):
                    violations.append(rel_path)
                    break
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(module_name):
                        violations.append(rel_path)
                        break

    assert not violations, (
        "Package __init__ files must not eagerly import internal modules; use lazy exports instead.\n"
        + "\n".join(sorted(violations))
    )


def test_package_init_files_define_lazy_getattr_exports() -> None:
    missing: list[str] = []

    for rel_path in PACKAGE_INIT_FILES:
        tree = _parse(rel_path)
        has_getattr = any(
            isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
            for node in tree.body
        )
        if not has_getattr:
            missing.append(rel_path)

    assert not missing, (
        "Package __init__ files must define module-level __getattr__ for lazy exports.\n"
        + "\n".join(sorted(missing))
    )
