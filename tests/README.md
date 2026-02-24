# Tests

Domain-organized test suite under `tests/`:

| Directory | Scope |
|-----------|-------|
| `cli/` | CLI argument parsing and help output |
| `config/` | Configuration loading and validation |
| `features/` | Feature extraction modules |
| `behavior/` | Behavioral analysis stages |
| `machine_learning/` | ML pipelines, CV, metrics |
| `fmri/` | fMRI analysis modules |
| `pipelines/` | Pipeline orchestration |
| `scripts/` | Standalone utility scripts |
| `utils/` | Shared utilities, architecture guards, repo hygiene |

## Conventions

1. Place new test files in the closest domain folder.
2. Keep shared helpers at the `tests/` level (e.g., `tests/pipelines_test_utils.py`).
3. Use `tests.REPO_ROOT` for repository-relative file resolution.
4. `tests/utils/test_test_layout_enforcement.py` enforces no root-level `tests/test_*.py`.
5. `tests/utils/test_repo_hygiene_guards.py` enforces basic tracked-file hygiene.

## Quick Check

```bash
make verify-structure
```
