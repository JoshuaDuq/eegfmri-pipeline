from __future__ import annotations

from tests import TESTS_ROOT


def test_no_root_level_test_modules() -> None:
    root_level_tests = sorted(TESTS_ROOT.glob("test_*.py"))
    assert not root_level_tests, (
        "Root-level test modules are not allowed. "
        "Move them into a domain folder under tests/.\n"
        f"Found: {[p.name for p in root_level_tests]}"
    )
