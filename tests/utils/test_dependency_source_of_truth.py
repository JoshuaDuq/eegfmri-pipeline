from __future__ import annotations

from tests import REPO_ROOT


def test_requirements_txt_is_thin_pyproject_shim() -> None:
    repo_root = REPO_ROOT
    requirements_path = repo_root / "requirements.txt"
    lines = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert lines == ['-e ".[dev,ml]"']
