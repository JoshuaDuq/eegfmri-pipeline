fMRI Analysis
=============

Subject-level and group-level GLM analysis plus trial-wise beta estimation
via nilearn.

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
     - Explicit group GLM from existing first-level MNI effect-size maps
   * - ``beta-series``
     - Trial-wise beta-series estimation (LSA method)
   * - ``lss``
     - Least-squares-separate (LSS) trial betas
   * - ``rest``
     - Resting-state ROI connectivity analysis (atlas-based, Fisher-z averaged across runs)

For full methods, see :doc:`../../methods/fmri/pipeline`.

Examples
--------

.. code-block:: bash

   # First-level GLM
   eeg-pipeline fmri-analysis first-level --subject 0001 \
     --contrast-name contrast \
     --cond-a-value stimulation --cond-b-value fixation_rest

   # With fMRIPrep preprocessed BOLD in MNI space
   eeg-pipeline fmri-analysis first-level --subject 0001 \
     --input-source fmriprep --fmriprep-space MNI152NLin2009cAsym \
     --cond-a-value stimulation --cond-b-value fixation_rest

   # Group mean from existing first-level MNI cope/effect-size maps
   eeg-pipeline fmri-analysis second-level --subject 0001 --subject 0002 \
     --group-model one-sample \
     --group-contrast-names stimulation_vs_rest

   # Beta-series for EEG–fMRI fusion
   eeg-pipeline fmri-analysis beta-series --subject 0001 \
     --cond-a-value stimulation --cond-b-value fixation_rest

   # LSS betas
   eeg-pipeline fmri-analysis lss --subject 0001 \
     --cond-a-value stimulation --cond-b-value fixation_rest

   # Resting-state ROI connectivity (atlas required)
   eeg-pipeline fmri-analysis rest --subject 0001 \
     --atlas-labels-img /path/to/atlas_parc.nii.gz \
     --atlas-labels-tsv /path/to/atlas_labels.tsv

   # Resting-state with custom bandpass and smoothing
   eeg-pipeline fmri-analysis rest --subject 0001 \
     --atlas-labels-img /path/to/atlas_parc.nii.gz \
     --high-pass-hz 0.01 --low-pass-hz 0.08 --smoothing-fwhm 6.0

   # With HTML report
   eeg-pipeline fmri-analysis first-level --subject 0001 \
     --cond-a-value stimulation --cond-b-value fixation_rest \
     --plots --plot-html-report

Key Options
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Option
     - Description
     - Default
   * - ``--input-source``
     - ``fmriprep`` or ``bids_raw``
     - ``fmriprep``
   * - ``--hrf-model``
     - ``spm``, ``flobs``, ``fir``
     - ``spm``
   * - ``--confounds-strategy``
     - ``auto``, ``none``, ``motion6``…``motion24+wmcsf+fd``
     - ``auto``
   * - ``--smoothing-fwhm``
     - Spatial smoothing kernel (mm)
     - ``5.0``
   * - ``--output-type``
     - ``z-score``, ``t-stat``, ``cope``, ``beta``
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
     - Add max-T permutation inference to second-level mode
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
