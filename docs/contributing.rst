Contributing
============

.. raw:: html

   <p class="hero-lede">
     Bug fixes, new feature families, documentation improvements, and test
     coverage are all welcome. Follow the <strong>four-step workflow</strong>
     below to keep changes reviewable and CI-passing.
   </p>

.. grid:: 4
   :gutter: 2

   .. grid-item-card:: 1 · Branch

      ``fix/<topic>`` ·
      ``feat/<topic>`` ·
      ``refactor/<topic>`` ·
      ``docs/<topic>``

   .. grid-item-card:: 2 · Commit

      Short imperative subject
      (≤ 72 chars). Prefix:
      ``fix:`` · ``feat:`` ·
      ``docs:`` · ``test:``

   .. grid-item-card:: 3 · Test

      ``make test`` must pass.
      Add tests in
      ``tests/<domain>/test_*.py``
      beside the affected domain.

   .. grid-item-card:: 4 · PR

      Summary · commands run ·
      issue link · screenshots
      for TUI / docs changes.

Setup
-----

macOS / Linux
~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git
   cd eegfmri-pipeline
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,ml]"
   pre-commit install

Windows PowerShell
~~~~~~~~~~~~~~~~~~

.. code-block:: powershell

   git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git
   cd eegfmri-pipeline
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -e ".[dev,ml]"
   pre-commit install

Use any Python interpreter that is ``3.11+``. If multiple versions are
installed, select one explicitly, for example ``python3.12 -m venv .venv``
or ``py -3.12 -m venv .venv``.

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

The Sphinx build uses ``.rst`` as the primary source format. CI deploys to GitHub Pages
on push to ``main``.

Architecture Decisions
----------------------

Module boundary rules are recorded in ``docs/architecture/adr-0001-module-boundaries.md``
and enforced by ``make verify-architecture``. Update the ADR and the enforcement tests in
the same commit whenever boundaries change.
