# Test Layout

Tests are organized by domain under `tests/`:

- `tests/cli/`
- `tests/config/`
- `tests/features/`
- `tests/behavior/`
- `tests/machine_learning/`
- `tests/fmri/`
- `tests/pipelines/`
- `tests/scripts/`
- `tests/utils/`

Conventions:

1. Place new files in the closest domain folder.
2. Keep shared helpers in `tests/` (for example `tests/pipelines_test_utils.py`).
3. Use `tests.REPO_ROOT` for repository-relative file resolution.
4. `tests/utils/test_test_layout_enforcement.py` enforces no root-level `tests/test_*.py`.
5. `tests/utils/test_repo_hygiene_guards.py` enforces basic tracked-file hygiene.

This layout is intentionally behavior-neutral and improves ownership/discoverability.

Quick check:

`make verify-structure`
