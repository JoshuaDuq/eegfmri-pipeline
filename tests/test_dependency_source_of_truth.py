from __future__ import annotations

from pathlib import Path


def test_requirements_txt_is_thin_pyproject_shim() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    requirements_path = repo_root / "requirements.txt"
    lines = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert lines == ['-e ".[ml]"']
