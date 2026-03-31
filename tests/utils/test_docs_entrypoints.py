from __future__ import annotations

import runpy

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


def test_sphinx_excludes_markdown_entrypoints_from_build() -> None:
    config = runpy.run_path(str(REPO_ROOT / "docs/conf.py"))
    exclude_patterns = set(config["exclude_patterns"])

    expected_exclusions = {
        "index.md",
        "eeg/**",
        "fmri/**",
    }
    missing = sorted(expected_exclusions - exclude_patterns)
    assert not missing, f"Missing Sphinx exclude patterns: {missing}"
