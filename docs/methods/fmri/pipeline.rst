fMRI Analysis Pipeline
======================

.. note::

   This fMRI pipeline is **still under active development**. APIs, defaults,
   and on-disk layouts may change between releases; re-check outputs after upgrades.

**Module:** ``fmri_pipeline``

.. seealso::

   :doc:`raw_to_bids`
      BIDS input contract and validation requirements for fMRI data.

   :doc:`../eeg/source_localization`
      EEG source localization uses fMRI contrast maps as spatial priors.

   :doc:`../../user_guide/configuration`
      Full ``fmri_preprocessing``, ``fmri_contrast``, ``fmri_group_level``, and ``fmri_resting_state`` key reference.

   :doc:`../../user_guide/cli/fmri_analysis`
      CLI flags for all fMRI analysis modes.

Methods reference for the fMRI analysis pipeline. The pipeline starts from
:term:`BIDS`-formatted inputs and covers :term:`fMRIPrep` preprocessing,
first-level :term:`GLM` contrast analysis, explicit second-level group inference,
trial-wise beta estimation (:term:`LSA` / :term:`LSS`), and multivariate signature
readout using user-configured weight maps.

The resulting fMRI statistical maps are used to constrain EEG inverse solutions
in the source localization stage.

.. list-table::
   :header-rows: 1
   :widths: 20 80
   :stub-columns: 1

   * - Inputs
     - BIDS fMRI data + fMRIPrep derivatives, ``*_events.tsv``
   * - Outputs
     - Contrast maps, trial-wise beta volumes, group inference maps, connectivity matrices
   * - CLI
     - ``eeg-pipeline fmri preprocess`` / ``eeg-pipeline fmri-analysis [first-level | second-level | beta-series | lss | rest]``
   * - Config
     - ``fmri_preprocessing``, ``fmri_contrast``, ``fmri_group_level``, ``fmri_resting_state`` sections

Notation
--------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Symbol
     - Meaning
   * - :math:`\mathbf{x} \in \mathbb{R}^V`
     - Vectorized voxel image (after masking)
   * - :math:`\mathbf{w} \in \mathbb{R}^V`
     - Signature weight map (configured reference pattern)
   * - :math:`V`
     - Number of finite, unmasked voxels
   * - :math:`\beta_i`
     - Beta estimate for trial :math:`i`
   * - :math:`\sigma^2_i`
     - Variance of beta estimate for trial :math:`i`
   * - :math:`w_i = 1/\sigma^2_i`
     - Inverse-variance weight for fixed-effects averaging
   * - TR
     - Repetition time (s)
   * - FD
     - Framewise displacement (mm); threshold set by ``fmri_preprocessing.fmriprep.fd_spike_threshold``
   * - :term:`HRF`
     - Hemodynamic response function

Pipeline Overview
-----------------

.. code-block:: text

   BIDS ──► fMRIPrep ──► First-Level GLM ──► Contrast Maps ──► EEG Source Localization
              │                │                    │
              │                │                    └──► Second-Level GLM ──► Group Inference
              │                │
              │                └──► Trial-Wise Betas ──► Signature Expression
              │
              └──► Resting-State ──► ROI Timeseries ──► Connectivity Matrix

.. list-table::
   :header-rows: 1
   :widths: 10 35 55

   * - Stage
     - Module
     - Purpose
   * - 1
     - BIDS input dataset + ``events.tsv``
     - Required input contract for downstream modeling
   * - 2
     - ``pipelines/fmri_preprocessing.py``
     - fMRIPrep containerized preprocessing
   * - 3
     - ``pipelines/fmri_analysis.py`` + ``analysis/contrast_builder.py``
     - Multi-run first-level GLM and contrast computation
   * - 3b
     - ``pipelines/fmri_second_level.py`` + ``analysis/second_level.py``
     - Explicit group-level inference from first-level MNI cope/effect-size maps
   * - 4
     - ``pipelines/fmri_trial_signatures.py`` + ``analysis/trial_signatures.py``
     - Trial-wise beta estimation and signature readout
   * - 5
     - ``analysis/reporting.py``
     - HTML report generation with QC diagnostics
   * - 6
     - ``pipelines/fmri_resting_state.py`` + ``analysis/resting_state.py``
     - Resting-state ROI connectivity analysis

Stage 1 — BIDS Inputs
----------------------

See :doc:`raw_to_bids` for the input contract and ``events.tsv`` requirements.

Events undergo three filtering stages before GLM fitting:

1. **``events_to_model``** — Restricts which ``trial_type`` rows enter the GLM.
2. **``stim_phases_to_model``** — Restricts stimulation events to specified sub-phases.
3. **Condition remapping** — Rows matching condition A → ``cond_a_<name>``; rows
   matching condition B → ``cond_b_<name>``.

Stage 2 — fMRIPrep Preprocessing
----------------------------------

Runs `fMRIPrep <https://fmriprep.org/>`_ in a Docker or Apptainer container per subject.

**Default output spaces:** ``MNI152NLin2009cAsym`` and ``T1w``.

Key outputs consumed downstream:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - File pattern
     - Description
   * - ``*_space-T1w_desc-preproc_bold.nii.gz``
     - Preprocessed BOLD in subject space
   * - ``*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz``
     - Preprocessed BOLD in MNI space
   * - ``*_desc-brain_mask.nii.gz``
     - Brain mask per run
   * - ``*_desc-confounds_timeseries.tsv``
     - Nuisance regressors

See :doc:`../../user_guide/cli/fmri_preprocessing` for all CLI options.

Stage 3 — First-Level GLM
--------------------------

Subject-level statistical contrasts between experimental conditions via nilearn's
``FirstLevelModel``.

Confound Regression
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Strategy
     - Regressors
   * - ``none``
     - No confounds
   * - ``motion6``
     - 6 rigid-body parameters (3 translation + 3 rotation)
   * - ``motion12``
     - motion6 + temporal derivatives
   * - ``motion24``
     - motion12 + quadratic terms + derivative quadratics
   * - ``motion24+wmcsf``
     - motion24 + white matter and CSF mean signals
   * - ``motion24+wmcsf+fd``
     - motion24+wmcsf + framewise displacement
   * - ``auto`` *(default)*
     - Adaptive: motion24 + WM, CSF, FD, and up to ``auto_compcor_n`` (default 5) aCompCor components

Additional regressors always included when present: ``motion_outlier*``,
``non_steady_state_outlier*``, ``outlier*`` (spike regressors).
NaN values in confounds are replaced with 0.

GLM Specification
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``hrf_model``
     - ``spm``
     - HRF. Options: ``spm``, ``flobs``, ``fir``
   * - ``drift_model``
     - ``cosine``
     - Slow-drift removal. Options: ``cosine``, ``polynomial``, ``none``
   * - ``high_pass``
     - 0.008 Hz
     - High-pass filter cutoff (128 s period)
   * - ``noise_model``
     - ``ar1``
     - Temporal autocorrelation model
   * - ``standardize``
     - ``true``
     - Standardize BOLD signal
   * - ``signal_scaling``
     - ``0``
     - Scale signal to percent signal change
   * - ``smoothing_fwhm``
     - ``null``
     - Optional spatial smoothing (mm FWHM)
   * - ``mask_img``
     - auto
     - fMRIPrep brain mask; intersection across runs for multi-run models

Multi-Run GLM
~~~~~~~~~~~~~~

Runs are combined using nilearn's native multi-run ``FirstLevelModel``
(fixed-effects across runs). Averaging per-run z/t maps is explicitly avoided.

- Runs where condition A or B are absent are excluded with a logged warning.
- The GLM raises an error only if no runs contain both conditions.
- Skipped runs and their reasons are recorded in the provenance sidecar.

Contrast Computation
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Contrast type
     - Definition
   * - Two-condition
     - ``cond_a_<name> - cond_b_<name>``
   * - Single-condition
     - ``cond_a_<name>``
   * - Custom formula
     - User-provided string (e.g., ``"stimulation - fixation_rest"``)

Output types: ``z-score`` (default), ``t-stat`` (t-statistic), ``cope`` (contrast of parameter estimates), ``beta`` (raw parameter estimates).

Caching: contrast maps are named with an MD5 hash of key configuration parameters.
A JSON sidecar records full provenance (subject, task, contrast definition, run inputs,
confound columns, skipped runs, event counts).

Stage 3b — Second-Level Group Inference
-----------------------------------------

Explicit mode (``eeg-pipeline fmri-analysis second-level``) consuming previously
generated first-level maps. Inputs must be first-level ``cope`` or ``beta`` maps in
``MNI152NLin2009cAsym`` space.

Supported designs:

- **``one-sample``** — Group mean/random-effects inference for one first-level contrast.
- **``two-sample``** — Between-group comparison using a subject-level TSV/CSV.
- **``paired``** — Within-subject comparison via subject-wise difference maps.
- **``repeated-measures``** — Within-subject multi-condition model across two or more contrasts.

Optional permutation inference (``--group-permutation-inference``) adds max-T
permutation inference for second-level t-contrasts.

Stage 4 — Trial-Wise Beta Estimation
--------------------------------------

Per-trial beta maps and multivariate signature expression.

Beta-Series (LSA)
~~~~~~~~~~~~~~~~~~

One GLM per run; one unique regressor per selected trial. Non-selected events
modeled as nuisance regressors grouped by ``trial_type``. Computationally efficient;
susceptible to correlated regressors when trials are temporally close.

Least Squares Separate (LSS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One GLM per trial within each run:

- Target trial → ``"target"`` regressor.
- Other selected trials → ``"other_cond_a"`` / ``"other_cond_b"`` (``per_condition`` mode, default).
- Non-selected events → nuisance regressors by ``trial_type``.

LSS requires :math:`N` model fits per run but produces less biased beta estimates
when trials are temporally close.

Condition-Level Averaging
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Formula
   * - ``variance`` *(default)*
     - Inverse-variance weighted mean: :math:`\hat\beta = \sum_i w_i \beta_i / \sum_i w_i`, where :math:`w_i = 1/\sigma^2_i`
   * - ``mean``
     - Simple arithmetic mean

Trial-Wise Outputs
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - File
     - Content
   * - ``trials.tsv``
     - Per-trial metadata: run, trial index, condition, onset, duration, event columns
   * - ``signatures/trial_signature_expression.tsv``
     - Per-trial × per-signature: dot product, cosine similarity, Pearson :math:`r`
   * - ``signatures/condition_signature_expression.tsv``
     - Per-condition-average × per-signature
   * - ``condition_betas/*.nii.gz``
     - Condition-averaged NIfTI beta maps
   * - ``trial_betas/<run>/*.nii.gz``
     - Per-trial NIfTI beta maps (optional)
   * - ``provenance.json``
     - Full configuration, signature root, run count

Stage 5 — Reporting and QC
----------------------------

Self-contained HTML report (``--plot-html-report``).

Statistical visualizations per space (native and/or MNI):

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Plot
     - Description
   * - ``slices``
     - Mosaic stat-map overlay (thresholded and unthresholded)
   * - ``glass``
     - Glass-brain projection
   * - ``hist``
     - Z-statistic voxel-distribution histogram with threshold lines
   * - ``clusters``
     - Cluster/peak table via ``nilearn.reporting.get_clusters_table``

QC diagnostics: motion QC (FD and DVARS time series), carpet plot (voxel × time
matrix, z-scored, clipped to ±3), tSNR map, design matrix images and TSVs,
design QC summary (regressor count, max absolute correlation, condition number, max VIF).

Thresholding modes: ``z`` (default, \|z\| > 2.3), ``fdr`` (BH :math:`q = 0.05`), ``none``.

Stage 6 — Resting-State Connectivity
--------------------------------------

Atlas-based ROI connectivity analysis from fMRIPrep resting-state BOLD data.

**Per-subject workflow:**

1. Discover BOLD runs and load fMRIPrep confound regressors.
2. Build a ``NiftiLabelsMasker`` for the configured atlas; extract per-ROI time series
   with simultaneous denoising (band-pass filtering, standardization, detrending).
3. Scrub motion-outlier frames via ``sample_mask``.
4. Compute per-run Pearson correlation connectivity matrices.
5. Aggregate multi-run matrices via **Fisher-z weighted averaging**
   (weights = number of retained frames per run):

.. math::

   \bar{Z}_{ij} = \frac{\sum_r n_r \cdot \mathrm{arctanh}(r_{ij}^{(r)})}{\sum_r n_r},
   \qquad
   \hat{r}_{ij} = \tanh(\bar{Z}_{ij}).

Key configuration (``RestingStateAnalysisConfig``):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Default
     - Description
   * - ``input_source``
     - ``fmriprep``
     - BOLD source (``fmriprep`` or ``bids_raw``)
   * - ``confounds_strategy``
     - ``auto``
     - fMRIPrep confound strategy
   * - ``high_pass_hz``
     - ``0.008``
     - High-pass filter (Hz)
   * - ``low_pass_hz``
     - ``0.1``
     - Low-pass filter (Hz)
   * - ``smoothing_fwhm``
     - ``null``
     - Spatial smoothing kernel FWHM (mm)
   * - ``atlas_labels_img``
     - required
     - NIfTI atlas parcellation image
   * - ``atlas_labels_tsv``
     - ``null``
     - Optional TSV with ROI names

Resting-state outputs written to:
``<deriv_root>/sub-<ID>/fmri/rest/task-<task>/atlas-<atlas_name>/``

BEM and Coregistration
-----------------------

**Module:** ``analysis/bem_generation.py``

Docker-based BEM model, BEM solution, and EEG↔MRI coregistration via FreeSurfer
and MNE-Python.

**BEM model:**

1. Watershed BEM (``mne watershed_bem``) → inner skull, outer skull, outer skin surfaces.
2. ``mne.make_bem_model()`` with ICO downsampling (default ico=4 → 5,120 triangles/surface)
   and conductivity [0.3, 0.006, 0.3] S/m.
3. ``mne.make_bem_solution()`` computes the forward model matrix.

**Identity transform guard:** auto-generation of an identity transform is refused by
default (``allow_identity_trans = false``).

Docker image: ``freesurfer-mne:7.4.1``.
Dockerfile: ``eeg_pipeline/docker_setup/Dockerfile.freesurfer-mne``.

Multivariate Signature Readouts
---------------------------------

**Modules:** ``analysis/trial_signatures.py`` and
``analysis/multivariate_signatures.py``

No signatures are hard-coded. Supply signature maps through configuration:

.. code-block:: yaml

   paths:
     signature_dir: /path/to/signature_maps
     signature_maps:
       - name: "SIG_A"
         path: "maps/sig_a_weights.nii.gz"
       - name: "SIG_B"
         path: "maps/sig_b_weights.nii.gz"

For each image :math:`\mathbf{x}` and signature weight map :math:`\mathbf{w}`:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Metric
     - Formula
   * - Dot product
     - :math:`s_\text{dot} = \mathbf{w}^\top \mathbf{x}`
   * - Cosine similarity
     - :math:`s_\text{cos} = \mathbf{w}^\top \mathbf{x} / (\|\mathbf{w}\| \cdot \|\mathbf{x}\|)`
   * - Pearson :math:`r`
     - :math:`s_r = \mathrm{corr}(\mathbf{w}, \mathbf{x})`

Spatial constraint: trial-wise signature extraction requires MNI-space images.

Output Layout
-------------

.. code-block:: text

   derivatives/
   ├── sub-XXXX/
   │   └── fmri/
   │       ├── first_level/<task>/<contrast_name>/
   │       │   ├── sub-XXXX_task-<task>_contrast-<name>_stat-z_score_<hash>.nii.gz
   │       │   ├── sub-XXXX_task-<task>_contrast-<name>_stat-effect_size_<hash>.nii.gz
   │       │   ├── sub-XXXX_<contrast>_provenance.json
   │       │   ├── design_matrix_run-01.tsv
   │       │   └── report.html
   │       ├── beta_series/<task>/<contrast_name>/
   │       │   ├── trials.tsv
   │       │   ├── signatures/trial_signature_expression.tsv
   │       │   └── trial_betas/<run>/*.nii.gz
   │       └── rest/<task>/atlas-<name>/
   │           ├── *_correlation_connectivity.tsv
   │           ├── *_correlation_connectivity_fisher_z.tsv
   │           └── provenance.json
   └── group/
       └── fmri/second_level/<task>/<contrast>/
           ├── group_<contrast>_z_score.nii.gz
           └── provenance.json

.. seealso::

   :doc:`raw_to_bids`
      BIDS input contract, events.tsv requirements, and DICOM conversion.

   :doc:`../../methods/eeg/source_localization`
      fMRI contrast maps from this pipeline constrain the EEG source prior.

   :doc:`../../user_guide/configuration`
      Full ``fmri_contrast``, ``fmri_group_level``, ``fmri_resting_state``,
      and ``fmri_preprocessing`` key reference.

   :doc:`../../user_guide/cli/fmri_analysis`
      CLI flags for first-level, second-level, beta-series, and resting-state.
