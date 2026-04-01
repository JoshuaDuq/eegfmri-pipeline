fMRI Methods
============

.. container:: page-intro

   Scientific reference for the fMRI analysis pipeline. These pages define the
   expected inputs, method choices, configuration keys, and derivative schemas
   for raw-to-BIDS preparation, preprocessing, GLM analysis, and resting-state
   workflows.

.. toctree::
   :maxdepth: 1
   :hidden:

   raw_to_bids
   pipeline

.. rst-class:: section-kicker

At A Glance

.. rst-class:: summary-grid

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Inputs

      BIDS fMRI datasets and, for analysis stages, fMRIPrep derivatives or explicitly selected raw BOLD inputs.

   .. grid-item-card:: Outputs

      Preprocessed fMRI derivatives, first-level contrasts, beta-series volumes, and group results.

   .. grid-item-card:: Stability note

      Parts of the fMRI command surface are still evolving, so confirm derivative paths after upgrades.

.. rst-class:: dashboard-grid

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Raw-to-BIDS Contract
      :link: raw_to_bids
      :link-type: doc

      BIDS input contract: directory layout, required NIfTI sidecars,
      ``events.tsv`` requirements, DICOM conversion, and validation.

   .. grid-item-card:: fMRI Analysis Pipeline
      :link: pipeline
      :link-type: doc

      fMRIPrep preprocessing, first-level GLM, group-level inference,
      trial-wise beta estimation, resting-state connectivity, and
      multivariate signature readouts.
