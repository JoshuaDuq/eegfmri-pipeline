fMRI Methods
============

Scientific reference for the fMRI analysis pipeline. Each page covers the
expected inputs, methods, configuration keys, and output schemas.

.. toctree::
   :maxdepth: 1
   :hidden:

   raw_to_bids
   pipeline

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
