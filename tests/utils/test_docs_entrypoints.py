from __future__ import annotations

from tests import REPO_ROOT


def test_docs_entrypoints_exist() -> None:
    expected_paths = [
        "docs/index.md",
        "docs/eeg/index.md",
        "docs/fmri/index.md",
        "docs/eeg/source-localization.md",
        "docs/fmri/raw-to-bids.md",
    ]
    missing = [path for path in expected_paths if not (REPO_ROOT / path).exists()]
    assert not missing, f"Missing docs entrypoints: {missing}"
