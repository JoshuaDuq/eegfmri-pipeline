# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `eeg_pipeline/` and `fmri_pipeline/`. Shared study-specific
workflow code is under `studies/pain_study/`. Tests mirror the domain layout in `tests/`
with subdirectories such as `tests/features/`, `tests/pipelines/`, `tests/fmri/`, and
`tests/utils/`. Documentation is split between the root `README.md` and domain READMEs in
`docs/`, `eeg_pipeline/`, `fmri_pipeline/`, `studies/pain_study/scripts/`, and `tests/`.
Configuration defaults live in `eeg_pipeline/utils/config/` and `fmri_pipeline/utils/config/`.

## Build, Test, and Development Commands

Install for development with `pip install -e ".[dev,ml]"`.
Run the full test suite with `make test` or `python -m pytest`.
Check repo structure and hygiene with `make verify-structure`.
Check import-boundary rules with `make verify-architecture`.
Run the stricter maintenance gate with `make verify-maintainability`.
Lint the code with `ruff check eeg_pipeline fmri_pipeline tests scripts`.
Format Python with `black .`.
Build the TUI locally with `cd eeg_pipeline/cli/tui && go build -o eeg-tui .`.

## Coding Style & Naming Conventions

Use Python 3.11+, 100-character lines, and standard Black/Ruff formatting.
Prefer clear module names, small functions, and explicit variables over compact but opaque
logic. Keep imports organized by dependency flow. Test files should follow
`tests/<domain>/test_*.py`; shared helpers belong at the `tests/` root. Use YAML for
pipeline configuration instead of hard-coded constants.

## Testing Guidelines

Pytest is the test runner. Place tests beside the closest domain and keep
repository-relative paths through `tests.REPO_ROOT`. Add coverage for CLI behavior,
configuration validation, pipeline contracts, and repo-hygiene checks when changing
shared infrastructure. Avoid root-level `tests/test_*.py`; the layout guard enforces this.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, often with prefixes such as `fix:`,
`refactor:`, or `docs:`. Keep commit titles specific and behavior-focused. Pull requests
should summarize the change, list the commands run, link the relevant issue or spec when
available, and include screenshots for TUI or documentation updates.

## Agent-Specific Instructions

Do not add fallback behavior or backward-compatibility shims unless explicitly requested.
Fail fast on invalid inputs, preserve existing outputs and side effects, and prefer the
smallest change that makes the code clearer and more maintainable.
