fMRI Analysis
=============

.. note::

   The fMRI pipeline is **still under active development**. GLM modes, contrast
   discovery, and file naming conventions may evolve between releases.

Subject-level and group-level GLM analysis plus trial-wise beta estimation
via nilearn.

.. note::

   ``fmri_contrast.enabled`` and ``fmri_group_level.enabled`` default to
   ``false``, but CLI ``first-level`` / ``second-level`` modes run when invoked.
   You do not need to flip those toggles to use the CLI modes directly.
   Use ``eeg-pipeline info fmri-conditions`` to list available condition
   values before specifying ``--cond-a-value`` / ``--cond-b-value``.

.. code-block:: bash

   eeg-pipeline fmri-analysis [mode] [options]

Modes
-----

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - ``first-level``
     - First-level GLM with user-defined contrasts → contrast maps
   * - ``second-level``
     - Explicit group GLM from existing first-level MNI cope/effect-size maps
   * - ``beta-series``
     - Trial-wise beta-series estimation (LSA method)
   * - ``lss``
     - Least-squares-separate (LSS) trial betas
   * - ``rest``
     - Resting-state ROI connectivity analysis (atlas-based, Fisher-z averaged across runs)

For full methods, see :doc:`../../methods/fmri/pipeline`.

Examples
--------

.. tab-set::

   .. tab-item:: First-level

      .. code-block:: bash

         eeg-pipeline fmri-analysis first-level --subject 0001 \
           --contrast-name contrast \
           --cond-a-value stimulation --cond-b-value fixation_rest

         # With fMRIPrep BOLD in MNI space
         eeg-pipeline fmri-analysis first-level --subject 0001 \
           --input-source fmriprep --fmriprep-space MNI152NLin2009cAsym \
           --cond-a-value stimulation --cond-b-value fixation_rest

         # With plots and a self-contained HTML report
         eeg-pipeline fmri-analysis first-level --subject 0001 \
           --cond-a-value stimulation --cond-b-value fixation_rest \
           --plots --plot-html-report

   .. tab-item:: Second-level

      .. code-block:: bash

         # Group mean from existing first-level MNI cope/effect-size maps
         eeg-pipeline fmri-analysis second-level --subject 0001 --subject 0002 \
           --group-model one-sample \
           --group-contrast-names stimulation_vs_rest

   .. tab-item:: Beta-series

      .. code-block:: bash

         eeg-pipeline fmri-analysis beta-series --subject 0001 \
           --cond-a-value stimulation --cond-b-value fixation_rest

   .. tab-item:: LSS

      .. code-block:: bash

         eeg-pipeline fmri-analysis lss --subject 0001 \
           --cond-a-value stimulation --cond-b-value fixation_rest

   .. tab-item:: Rest

      Requires a parcellation atlas label image (NIfTI, integer ROI indices).
      A matching TSV with ``index`` and ``name`` columns is optional but recommended.

      .. code-block:: bash

         # Resting-state ROI connectivity (atlas required)
         eeg-pipeline fmri-analysis rest --subject 0001 \
           --atlas-labels-img /path/to/atlas_parc.nii.gz \
           --atlas-labels-tsv /path/to/atlas_labels.tsv

         # Custom bandpass and smoothing
         eeg-pipeline fmri-analysis rest --subject 0001 \
           --atlas-labels-img /path/to/atlas_parc.nii.gz \
           --high-pass-hz 0.01 --low-pass-hz 0.08 --smoothing-fwhm 6.0

Key Options
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Option
     - Description
     - Default
   * - ``--input-source``
     - ``fmriprep`` only; raw BIDS inputs are rejected for inferential analyses
     - ``fmriprep``
   * - ``--hrf-model``
     - ``spm``, ``flobs``, ``fir``
     - ``spm``
   * - ``--confounds-strategy``
     - ``auto``, ``none``, ``motion6``…``motion24+wmcsf+fd``
     - ``auto``
   * - ``--smoothing-fwhm``
     - Spatial smoothing kernel (mm)
     - ``null``
   * - ``--output-type``
     - ``z-score``, ``t-stat``, ``cope``; ``beta`` currently aliases ``cope``/effect-size rather than exporting a distinct raw-beta map. Plot/report generation now requires ``z-score`` because the plotting thresholds and labels are calibrated only for z-statistics.
     - ``z-score``
   * - ``--group-model``
     - ``one-sample``, ``two-sample``, ``paired``, ``repeated-measures``
     - ``one-sample``
   * - ``--group-contrast-names``
     - First-level contrast names consumed by second-level mode
     - required for ``second-level``
   * - ``--group-covariates-file``
     - Subject-level TSV/CSV for groups and covariates
     - none
   * - ``--group-permutation-inference``
     - Add max-T permutation inference to second-level mode; avoid this for repeated-measures designs until restricted permutations are implemented
     - disabled
   * - ``--plots``
     - Generate per-subject figures
     - disabled
   * - ``--plot-html-report``
     - Write self-contained HTML report
     - disabled
   * - ``--write-design-matrix``
     - Save design matrices (TSV + PNG)
     - first-level: disabled; second-level: enabled

.. seealso::

   :doc:`../../methods/fmri/pipeline`
      GLM specification, confound strategy, beta estimation, and
      multivariate signature readout methods.

   :doc:`fmri_preprocessing`
      Containerized fMRIPrep preprocessing (run before analysis).

   :doc:`../configuration`
      ``fmri_contrast``, ``fmri_group_level``, and ``fmri_resting_state`` config sections.

   :doc:`../data_layout`
      fMRI BIDS layout and required ``*_bold.json`` sidecar fields.
