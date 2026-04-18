fMRI Analysis Pipeline
======================

.. warning::

   This pipeline is **still under active development**. APIs, defaults,
   and on-disk layouts may change between releases; re-check outputs after upgrades.

.. raw:: html

   <p class="hero-lede">
     Containerized fMRIPrep preprocessing, first-level GLM contrast analysis,
     second-level group inference, trial-wise beta estimation (LSA / LSS),
     resting-state ROI connectivity, and multivariate EEG–fMRI signature
     readout. fMRI statistical maps feed back into EEG source localization.
   </p>

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Inputs

      BIDS fMRI data · fMRIPrep derivatives · ``*_events.tsv``

   .. grid-item-card:: Outputs

      Contrast maps · trial-wise beta volumes · group inference maps ·
      ROI connectivity matrices

   .. grid-item-card:: CLI

      ``eeg-pipeline fmri preprocess`` ·
      ``eeg-pipeline fmri-analysis [first-level | second-level |
      beta-series | lss | rest]``

   .. grid-item-card:: Config

      ``fmri_preprocessing`` · ``fmri_contrast`` ·
      ``fmri_group_level`` · ``fmri_resting_state``

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

.. warning::

   First-level GLM and trial-wise beta estimation now require ``input_source=fmriprep``.
   Raw BIDS inputs are rejected because they bypass fMRIPrep preprocessing, spatial
   normalization, and the standard confounds outputs that the downstream models assume.

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
     - ``false``
     - Do not z-score the BOLD signal inside nilearn
   * - ``signal_scaling``
     - ``0``
     - Disable nilearn signal scaling; keep the input image scaling unchanged
   * - ``smoothing_fwhm``
     - ``null``
     - Optional spatial smoothing (mm FWHM)
   * - ``mask_img``
     - auto
     - Required fMRIPrep brain mask; intersection across runs for multi-run models

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

Output types: ``z-score`` (default), ``t-stat`` (t-statistic), and ``cope`` (contrast of parameter estimates).

Current implementation note:
``beta`` is not a distinct raw-beta export. It is currently an alias of nilearn's
``effect_size`` output, the same quantity used for ``cope``.

Current plotting/reporting note:
the HTML report and plotting utilities now require z-statistic maps. If you
request plotting/report generation for a first-level contrast, use
``output_type=z-score``. ``t-stat`` maps can still be written as analysis
outputs, but they are rejected by the report/plotting path because its
threshold calibration and labels are defined only for z-statistics.

Caching: contrast maps are named with an MD5 hash of key configuration parameters.
A JSON sidecar records full provenance (subject, task, contrast definition, run inputs,
confound columns, skipped runs, event counts).

Stage 3b — Second-Level Group Inference
-----------------------------------------

Explicit mode (``eeg-pipeline fmri-analysis second-level``) consuming previously
generated first-level effect-size maps in ``MNI152NLin2009cAsym`` space.

Supported designs:

- **``one-sample``** — Group mean/random-effects inference for one first-level contrast.
- **``two-sample``** — Between-group comparison using a subject-level TSV/CSV.
- **``paired``** — Within-subject comparison via subject-wise difference maps.
- **``repeated-measures``** — Within-subject multi-condition model across two or more contrasts.

Optional permutation inference (``--group-permutation-inference``) adds max-T
permutation inference for second-level t-contrasts.

.. warning::

   The current permutation path does not encode exchangeability blocks. It is therefore
   appropriate for one-sample models, two-sample models, and paired analyses after
   collapsing each subject to a difference map, but it is not exchangeability-safe for
   repeated-measures designs that keep multiple rows per subject in the design matrix.
   Repeated-measures permutation inference should be treated as unsupported until
   restricted permutations are implemented.

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
     - Implemented as inverse-variance weighting, but this should be treated cautiously for LSS condition summaries because trial-wise beta estimates within a subject/run are correlated rather than independent fixed-effect observations
   * - ``mean``
     - Simple arithmetic mean

Current implementation note:
for ``beta-series``, condition summary maps are built from run-level averaged contrasts
and then combined across runs. For ``lss``, condition summary maps are built by combining
all trial-level beta images directly, so the default inverse-variance weighting is a
descriptive heuristic rather than a valid fixed-effects estimator.

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

.. warning::

   Resting-state connectivity now requires ``input_source=fmriprep``. Raw BIDS inputs are
   rejected because, without fMRIPrep preprocessing and confounds, motion and spatial
   misalignment can dominate the ROI correlation structure.

**Per-subject workflow:**

1. Discover BOLD runs and load fMRIPrep confound regressors.
2. Build a ``NiftiLabelsMasker`` for the configured atlas; extract per-ROI time series
   with simultaneous denoising (band-pass filtering, standardization, detrending).
3. Scrub motion-outlier frames via ``sample_mask``.
4. Compute per-run Pearson correlation connectivity matrices.
5. Aggregate multi-run matrices via Fisher-z averaging.

Current implementation detail:
the run weights are the number of retained frames per run, i.e.

.. math::

   \bar{Z}_{ij} = \frac{\sum_r n_r \cdot \mathrm{arctanh}(r_{ij}^{(r)})}{\sum_r n_r},
   \qquad
   \hat{r}_{ij} = \tanh(\bar{Z}_{ij}).

This should be treated as a validity limitation rather than a target method:
for Fisher-z averaging the variance-stabilizing weight is proportional to
:math:`n_r - 3`, not :math:`n_r`, so short runs are currently misweighted.

Current implementation limitation:
the masker is built from the atlas alone and does not yet intersect each run with
the corresponding fMRIPrep brain mask. ROIs near susceptibility dropout or partial
coverage can therefore contribute non-brain voxels without necessarily becoming
degenerate enough to trigger the existing guards.

Key configuration (``RestingStateAnalysisConfig``):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Default
     - Description
   * - ``input_source``
     - ``fmriprep``
     - Required BOLD source for scientifically valid resting-state connectivity analysis
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

Current implementation note:
signature expression rejects continuous resampling whenever the moving image
contains non-finite voxels. This avoids mixing unsupported NaN-coded voxels into
neighboring weights or effect estimates during interpolation. In practice,
signature maps and target images should already share a compatible finite-valued
grid whenever possible.

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
