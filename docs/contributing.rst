Contributing
============

All contributions are welcome — bug fixes, new feature families, documentation
improvements, and test coverage. Follow the workflow below to keep changes
reviewable and CI-passing.

.. grid:: 4
   :gutter: 2

   .. grid-item-card:: :octicon:`git-branch` Branch

      ``fix/``, ``feat/``,
      ``refactor/``, ``docs/``

   .. grid-item-card:: :octicon:`git-commit` Commit

      Short imperative subject.
      Prefix: ``fix:``, ``feat:``,
      ``docs:``, ``test:``

   .. grid-item-card:: :octicon:`check-circle` Test

      ``make test`` must pass.
      Add tests beside the
      affected domain.

   .. grid-item-card:: :octicon:`git-pull-request` PR

      Summary, commands run,
      issue link, screenshots
      for TUI / docs changes.

Setup
-----

.. code-block:: bash

   git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git
   cd eegfmri-pipeline
   python3.11 -m venv .venv311
   source .venv311/bin/activate
   pip install -e ".[dev,ml]"
   pre-commit install

Branches and Commits
--------------------

- Branch: ``fix/<topic>``, ``feat/<topic>``, ``refactor/<topic>``
- Commit subject: short imperative, ≤ 72 characters
- Prefix: ``fix:``, ``feat:``, ``refactor:``, ``docs:``, ``test:``
- PR description: summary, commands run, issue link, screenshots (TUI / docs changes)

Code Style
----------

.. code-block:: bash

   ruff check eeg_pipeline fmri_pipeline tests scripts   # lint
   black .                                               # format

Rules are configured in ``pyproject.toml``.

Testing
-------

.. code-block:: bash

   make test                   # full test suite
   make verify-structure       # repo layout guard
   make verify-architecture    # import boundary enforcement

Place new tests in ``tests/<domain>/test_*.py`` beside the closest domain.

Documentation
-------------

.. code-block:: bash

   make docs                          # build (warnings treated as errors)
   open docs/_build/html/index.html   # preview

All pages are ``.rst`` under ``docs/``. CI deploys to GitHub Pages on push to ``main``.
