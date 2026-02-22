.PHONY: verify-structure

verify-structure:
	python3 -m compileall -q tests
	python3 -m pytest -q tests/utils/test_test_layout_enforcement.py tests/utils/test_repo_hygiene_guards.py
