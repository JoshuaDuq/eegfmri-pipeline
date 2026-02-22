from __future__ import annotations

import re
from tests import REPO_ROOT


def test_no_broad_silent_exception_handlers() -> None:
    repo_root = REPO_ROOT
    targets = [repo_root / "eeg_pipeline", repo_root / "fmri_pipeline", repo_root / "scripts"]

    pattern = re.compile(
        r"except\s+(?:Exception|\([^\)]*Exception[^\)]*\))\s*:\s*\n\s*pass\b",
        re.MULTILINE,
    )

    violations: list[str] = []
    for root in targets:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if pattern.search(text):
                violations.append(str(path.relative_to(repo_root)))

    assert not violations, (
        "Found broad silent exception handlers (except ... Exception ...: pass) in:\n"
        + "\n".join(sorted(violations))
    )
