PYTHON ?= python3
ifeq ($(wildcard .venv/bin/python),.venv/bin/python)
PYTHON := .venv/bin/python
endif

SPHINXBUILD ?= sphinx-build
ifeq ($(wildcard .venv/bin/sphinx-build),.venv/bin/sphinx-build)
SPHINXBUILD := .venv/bin/sphinx-build
endif
DOCS_SRC    := docs
DOCS_BUILD  := docs/_build/html

.PHONY: test verify-structure verify-architecture verify-maintainability docs docs-clean

test:
	$(PYTHON) -m pytest

verify-structure:
	$(PYTHON) -m compileall -q tests
	$(PYTHON) -m pytest -q --no-cov tests/utils/test_test_layout_enforcement.py tests/utils/test_repo_hygiene_guards.py

verify-architecture:
	$(PYTHON) -m pytest -q --no-cov tests/utils/test_architecture_import_boundaries.py

verify-maintainability:
	$(PYTHON) -m pytest -q --no-cov tests/utils/test_test_layout_enforcement.py tests/utils/test_repo_hygiene_guards.py tests/utils/test_architecture_import_boundaries.py tests/utils/test_docs_entrypoints.py tests/utils/test_cli_facade_hygiene.py

docs:
	$(SPHINXBUILD) -W --keep-going -b html $(DOCS_SRC) $(DOCS_BUILD)
	@echo "Build finished. Open $(DOCS_BUILD)/index.html"

docs-clean:
	rm -rf $(DOCS_BUILD)
