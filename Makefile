PYTHON ?= python3
ifeq ($(wildcard .venv/bin/python),.venv/bin/python)
PYTHON := .venv/bin/python
endif

.PHONY: verify-structure verify-architecture verify-maintainability

verify-structure:
	$(PYTHON) -m compileall -q tests
	$(PYTHON) -m pytest -q --no-cov tests/utils/test_test_layout_enforcement.py tests/utils/test_repo_hygiene_guards.py

verify-architecture:
	$(PYTHON) -m pytest -q --no-cov tests/utils/test_architecture_import_boundaries.py

verify-maintainability:
	$(PYTHON) -m pytest -q --no-cov tests/utils/test_test_layout_enforcement.py tests/utils/test_repo_hygiene_guards.py tests/utils/test_architecture_import_boundaries.py tests/utils/test_docs_entrypoints.py tests/utils/test_phase2_organization_shims.py tests/utils/test_cli_facade_hygiene.py
