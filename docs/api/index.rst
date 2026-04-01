API Reference
=============

.. raw:: html

   <p class="hero-intro">
     Public symbols for both packages, listed by module. Docstrings render
     when the package is installed (<code>pip install -e ".[dev]"</code>).
   </p>

.. note::

   Full ``autodoc`` generation requires the complete project environment.
   In CI, heavy dependencies are mocked (see ``autodoc_mock_imports`` in
   ``docs/conf.py``). To regenerate with live docstrings locally, run
   ``make docs`` inside the activated virtual environment.

.. toctree::
   :maxdepth: 1
   :hidden:

   eeg_pipeline
   fmri_pipeline

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: EEG Pipeline
      :link: eeg_pipeline
      :link-type: doc

      Pipelines, analysis modules, feature families, plotting utilities,
      preprocessing helpers, CLI entry points, and shared infrastructure.

   .. grid-item-card:: fMRI Pipeline
      :link: fmri_pipeline
      :link-type: doc

      fMRI pipelines (fMRIPrep, first-level GLM, second-level, beta-series,
      resting-state), confound selection, BEM generation, signature readouts,
      and QC reporting.
