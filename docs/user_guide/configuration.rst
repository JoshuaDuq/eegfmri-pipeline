Configuration
=============

.. raw:: html

   <p class="hero-lede">
     Canonical reference for the public YAML configuration entry points and
     their keys. Override any key at runtime with
     <code>--set KEY=VALUE</code> without editing files.
   </p>

.. tip::

   The quickest way to set paths and task name is the **TUI Global Setup**
   screen (main menu → *Utilities → Global Setup*, or press ``C``). It edits
   the same values without touching any YAML file. See :doc:`tui`.

**Override precedence** (highest wins): ``--set`` overrides → CLI flags → YAML defaults.

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: ``eeg_config.yaml``

      EEG preprocessing · feature extraction · machine learning ·
      fMRI integration defaults.

   .. grid-item-card:: ``behavior_config.yaml``

      Behavioral statistics pipeline: predictor type, analysis stages,
      permutation settings, and FDR parameters.

   .. grid-item-card:: ``fmri_config.yaml``

      fMRI pipeline: fMRIPrep options · GLM specification ·
      confound strategy · group-level inference.

.. note::

   Relative paths are resolved from the directory of the config file that
   defines them (for example ``eeg_config.yaml``, ``behavior_config.yaml``,
   or ``fmri_config.yaml``). Use absolute paths when you need location-independent
   configuration.

.. _configuration-quick-nav:

Quick Navigation
----------------

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: :ref:`Project & Paths <configuration-project-paths>`

      Task naming and all filesystem roots.

   .. grid-item-card:: :ref:`EEG & Preprocessing <configuration-eeg-preprocessing>`

      Montage, reference, filtering, and preprocessing stage settings.

   .. grid-item-card:: :ref:`Bad Channels <configuration-pyprep>`

      PyPREP and RANSAC bad-channel detection controls.

   .. grid-item-card:: :ref:`ICA <configuration-ica>`

      ICA algorithm, labeling, and ICLabel thresholds.

   .. grid-item-card:: :ref:`Epochs <configuration-epochs>`

      Epoch windows, baselines, and rejection settings.

   .. grid-item-card:: :ref:`Bands & Time Windows <configuration-bands-windows>`

      Default band edges and named time windows.

   .. grid-item-card:: :ref:`Feature Engineering <configuration-feature-engineering>`

      Feature-family selection, transforms, and per-family controls.

   .. grid-item-card:: :ref:`fMRI Preprocessing <configuration-fmri-preprocessing>`

      Container engine and fMRIPrep defaults.

   .. grid-item-card:: :ref:`First-Level GLM <configuration-first-level-glm>`

      GLM specification and condition selection.

   .. grid-item-card:: :ref:`Group Inference <configuration-second-level>`

      Second-level model and permutation inference.

   .. grid-item-card:: :ref:`Behavioral Statistics <configuration-behavior>`

      Predictor type, correlation/regression, permutation, FDR controls.

   .. grid-item-card:: :ref:`Runtime Overrides <configuration-runtime-overrides>`

      Final-precedence ``--set`` overrides for long-tail keys.

----

.. _configuration-project-paths:

Project & Paths
---------------

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Key
     - Default
     - Description
   * - ``project.task``
     - ``"thermalactive"``
     - Task name used in BIDS paths and file naming
   * - ``project.random_state``
     - ``42``
     - Primary random seed for reproducible analyses
   * - ``project.subject_list``
     - ``null``
     - Optional list of subjects to process; ``null`` = all found
   * - ``paths.bids_root``
     - ``"../../../data/bids_output/eeg"``
     - BIDS-formatted EEG data directory
   * - ``paths.bids_rest_root``
     - ``null``
     - Optional resting-state EEG BIDS directory (``task_is_rest`` mode)
   * - ``paths.bids_fmri_root``
     - ``"../../../data/bids_output/fmri"``
     - BIDS-formatted fMRI data directory
   * - ``paths.deriv_root``
     - ``"../../../data/derivatives"``
     - Processed derivatives output directory
   * - ``paths.deriv_rest_root``
     - ``null``
     - Resting-state EEG derivatives directory
   * - ``paths.source_data``
     - ``"../../../data/source_data"``
     - Raw source data directory
   * - ``paths.freesurfer_dir``
     - ``"../../../data/derivatives/freesurfer"``
     - FreeSurfer ``SUBJECTS_DIR``
   * - ``paths.freesurfer_license``
     - ``null``
     - Path to FreeSurfer ``license.txt``; falls back to ``EEG_PIPELINE_FREESURFER_LICENSE`` env var, then ``~/license.txt``
   * - ``paths.signature_dir``
     - ``null``
     - Root directory for multivariate signature weight maps
   * - ``paths.signature_maps``
     - ``[]``
     - List of ``{name, path}`` entries relative to ``signature_dir``

.. _configuration-eeg-preprocessing:

EEG & Preprocessing
-------------------

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Key
     - Default
     - Description
   * - ``eeg.montage``
     - ``"easycap-M1"``
     - EEG cap montage name (MNE-Python format)
   * - ``eeg.reference``
     - ``"average"``
     - Re-reference target: ``"average"``, ``"REST"``, or channel name
   * - ``eeg.eog_channels``
     - ``null``
     - EOG channel names; ``null`` = auto-detect
   * - ``eeg.ecg_channels``
     - ``["ECG"]``
     - ECG channel names for ICA cardiac labeling
   * - ``preprocessing.resample_freq``
     - ``500``
     - Target sampling rate (Hz)
   * - ``preprocessing.l_freq``
     - ``0.1``
     - High-pass filter cutoff (Hz)
   * - ``preprocessing.h_freq``
     - ``100``
     - Low-pass filter cutoff (Hz)
   * - ``preprocessing.notch_freq``
     - ``60``
     - Notch filter frequency (Hz)
   * - ``preprocessing.task_is_rest``
     - ``false``
     - ``true`` = resting-state mode: fixed-length segments, no event conditions.
       Also set ``paths.bids_rest_root`` and ``paths.deriv_rest_root``.
   * - ``preprocessing.rest_epochs_duration``
     - ``10.0``
     - Resting-state segment duration (s); only used when ``task_is_rest: true``
   * - ``preprocessing.rest_epochs_overlap``
     - ``0.0``
     - Overlap between resting-state segments (s); only used when ``task_is_rest: true``
   * - ``preprocessing.find_breaks``
     - ``true``
     - Detect and annotate recording breaks
   * - ``preprocessing.write_clean_events``
     - ``true``
     - Write post-rejection ``*_proc-clean_events.tsv`` to derivatives

.. _configuration-pyprep:

Bad Channels (PyPREP)
---------------------

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Key
     - Default
     - Description
   * - ``pyprep.ransac``
     - ``true``
     - Use RANSAC for bad channel detection
   * - ``pyprep.repeats``
     - ``3``
     - Number of PREP iterations
   * - ``pyprep.bad_channel_sync_policy``
     - ``per_run``
     - Bad-channel policy across runs: ``per_run`` keeps run-specific bads; ``subject_union`` applies the subject-level union
   * - ``pyprep.average_reref``
     - ``false``
     - Apply average re-reference inside PyPREP
   * - ``pyprep.consider_previous_bads``
     - ``true``
     - Carry forward bads from previous runs
   * - ``pyprep.overwrite_chans_tsv``
     - ``true``
     - Overwrite BIDS ``*_channels.tsv`` with updated bad-channel status
   * - ``pyprep.delete_breaks``
     - ``false``
     - Mark detected breaks as ``BAD_break`` annotations before bad-channel detection; sample timing is not cropped

.. _configuration-ica:

ICA
---

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Key
     - Default
     - Description
   * - ``ica.algorithm``
     - ``"extended_infomax"``
     - ICA decomposition algorithm (``"extended_infomax"``, ``"fastica"``, ``"picard"``)
   * - ``ica.n_components``
     - ``0.99``
     - Number of components; float < 1 = explained-variance fraction
   * - ``ica.l_freq``
     - ``1.0``
     - High-pass before ICA fitting (Hz); suppresses slow drift
   * - ``ica.probability_threshold``
     - ``0.8``
     - ICLabel probability threshold for artifact exclusion
   * - ``ica.labels_to_keep``
     - ``["brain", "other"]``
     - ICLabel classes to retain as clean components

.. _configuration-epochs:

Epochs
------

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Key
     - Default
     - Description
   * - ``epochs.tmin``
     - ``-7.0``
     - Epoch start relative to event onset (s); leave generous padding for TFR baselines
   * - ``epochs.tmax``
     - ``15.0``
     - Epoch end relative to event onset (s)
   * - ``epochs.baseline``
     - ``[-0.2, 0.0]``
     - ERP baseline window (s); ``null`` = no baseline
   * - ``epochs.reject``
     - ``"autoreject_local"``
     - Artifact rejection: ``"autoreject_local"``, ``"autoreject_global"``, ``null``
   * - ``epochs.autoreject_n_interpolate``
     - ``[4, 8, 16]``
     - Bad channel interpolation counts tried by Autoreject

.. _configuration-bands-windows:

Frequency Bands & Time Windows
------------------------------

Default frequency bands:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Band
     - Range (Hz)
     - Key
   * - Delta
     - 1.0 – 3.9
     - ``frequency_bands.delta``
   * - Theta
     - 4.0 – 7.9
     - ``frequency_bands.theta``
   * - Alpha
     - 8.0 – 12.9
     - ``frequency_bands.alpha``
   * - Beta
     - 13.0 – 30.0
     - ``frequency_bands.beta``
   * - Gamma
     - 30.1 – 80.0
     - ``frequency_bands.gamma``

Default time windows:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Key
     - Default (s)
     - Description
   * - ``time_windows.active``
     - ``[3.0, 10.5]``
     - Task/active period for power and connectivity
   * - ``time_windows.baseline_tfr``
     - ``[-5.0, -0.01]``
     - TFR baseline window for logratio normalization
   * - ``time_windows.baseline_erp``
     - ``[-0.2, 0.0]``
     - ERP baseline window

.. _configuration-feature-engineering:

Feature Engineering
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Key
     - Default
     - Description
   * - ``feature_engineering.feature_categories``
     - *(all 14 families)*
     - Active feature families; see :doc:`../methods/eeg/features`
   * - ``feature_engineering.analysis_mode``
     - ``"group_stats"``
     - ``"group_stats"`` or ``"trial_ml_safe"`` — controls cross-trial leakage guards
   * - ``feature_engineering.spatial_transform``
     - ``"none"``
     - Global override; prefer per-family settings below
   * - ``feature_engineering.spatial_transform_per_family.connectivity``
     - ``"csd"``
     - CSD applied to phase-based connectivity (recommended)
   * - ``feature_engineering.spatial_transform_per_family.power``
     - ``"none"``
     - No CSD for amplitude features (changes units)
   * - ``feature_engineering.parallel.n_jobs_bands``
     - ``-1``
     - Workers for band-parallel computation (``-1`` = all cores)
   * - ``feature_engineering.power.subtract_evoked``
     - ``true``
     - Subtract ERP before power → induced oscillations
   * - ``feature_engineering.power.emit_db``
     - ``true``
     - Emit dB-scaled (``10·log10``) power alongside log-ratio
   * - ``feature_engineering.connectivity.measures``
     - ``["wpli", "aec"]``
     - Active connectivity measures
   * - ``feature_engineering.connectivity.aec_output``
     - ``["r"]``
     - AEC output format: ``"r"`` (raw), ``"z"`` (Fisher-z), or both
   * - ``feature_engineering.aperiodic.model``
     - ``"fixed"``
     - Spectral parameterization model: ``"fixed"`` or ``"knee"``
   * - ``feature_engineering.aperiodic.min_r2``
     - ``0.6``
     - Minimum :math:`R^2` to accept an aperiodic fit
   * - ``feature_engineering.pac.method``
     - ``"mvl"``
     - PAC method: ``"mvl"`` (mean vector length)
   * - ``feature_engineering.pac.pairs``
     - ``[["theta","gamma"],["alpha","gamma"]]``
     - Phase–amplitude coupling frequency pairs
   * - ``feature_engineering.bands.use_iaf``
     - ``false``
     - Use individualized alpha frequency (IAF) to shift alpha band
   * - ``feature_engineering.output.also_save_csv``
     - ``false``
     - Also export feature tables as CSV alongside Parquet

.. _configuration-fmri-preprocessing:

fMRI Preprocessing (fMRIPrep)
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 22 40

   * - Key
     - Default
     - Description
   * - ``fmri_preprocessing.engine``
     - ``"docker"``
     - Container engine: ``"docker"`` or ``"apptainer"``
   * - ``fmri_preprocessing.fmriprep.image``
     - ``"nipreps/fmriprep:25.2.4"``
     - Docker image tag or Apptainer URI
   * - ``fmri_preprocessing.fmriprep.output_spaces``
     - ``["MNI152NLin2009cAsym","T1w"]``
     - Output template spaces
   * - ``fmri_preprocessing.fmriprep.level``
     - ``"full"``
     - fMRIPrep processing level: ``"full"``, ``"resampling"``, ``"minimal"``
   * - ``fmri_preprocessing.fmriprep.fd_spike_threshold``
     - ``0.5``
     - FD spike threshold (mm) for motion scrubbing
   * - ``fmri_preprocessing.fmriprep.dvars_spike_threshold``
     - ``1.5``
     - DVARS spike threshold
   * - ``fmri_preprocessing.fmriprep.bold2t1w_dof``
     - ``6``
     - Degrees of freedom for BOLD→T1w registration
   * - ``fmri_preprocessing.fmriprep.skull_strip_template``
     - ``"OASIS30ANTs"``
     - Template for skull stripping
   * - ``fmri_preprocessing.fmriprep.nthreads``
     - ``0``
     - CPU threads (0 = auto)
   * - ``fmri_preprocessing.fmriprep.extra_args``
     - ``null``
     - Additional CLI arguments appended verbatim to fMRIPrep

.. _configuration-first-level-glm:

First-Level GLM
---------------

.. list-table::
   :header-rows: 1
   :widths: 38 22 40

   * - Key
     - Default
     - Description
  * - ``fmri_contrast.enabled``
    - ``false``
    - Configuration gate for config-driven first-level analysis paths; explicit CLI ``fmri-analysis first-level`` runs regardless of this toggle
  * - ``fmri_contrast.input_source``
    - ``"fmriprep"``
    - BOLD source for inferential fMRI analysis (currently ``"fmriprep"`` only)
   * - ``fmri_contrast.fmriprep_space``
     - ``"T1w"``
     - fMRIPrep output space to use
   * - ``fmri_contrast.hrf_model``
     - ``"spm"``
     - HRF model: ``"spm"``, ``"flobs"``, ``"fir"``
   * - ``fmri_contrast.drift_model``
     - ``"cosine"``
     - Drift removal: ``"cosine"``, ``"polynomial"``, ``"none"``
   * - ``fmri_contrast.high_pass_hz``
     - ``0.008``
     - High-pass filter (128 s period)
   * - ``fmri_contrast.confounds_strategy``
     - ``"auto"``
     - Nuisance regressor strategy; see :doc:`../methods/fmri/pipeline`
   * - ``fmri_contrast.output_type``
     - ``"z-score"``
     - Output statistic: ``"z-score"``, ``"t-stat"``, ``"cope"``, ``"beta"``
   * - ``fmri_contrast.resample_to_freesurfer``
     - ``true``
     - Resample contrast maps into FreeSurfer subject space
   * - ``fmri_contrast.condition_a.column``
     - ``"trial_type"``
     - ``events.tsv`` column for condition A selection
   * - ``fmri_contrast.condition_a.value``
     - ``null``
     - Value in that column identifying condition A trials

.. _configuration-second-level:

Second-Level (Group) Inference
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 22 40

   * - Key
     - Default
     - Description
   * - ``fmri_group_level.enabled``
     - ``false``
     - Configuration default; explicit CLI ``fmri-analysis second-level`` runs regardless of this toggle
   * - ``fmri_group_level.model``
     - ``"one-sample"``
     - Design: ``"one-sample"``, ``"two-sample"``, ``"paired"``, ``"repeated-measures"``
   * - ``fmri_group_level.permutation.enabled``
     - ``false``
     - Enable max-T permutation inference
   * - ``fmri_group_level.permutation.n_permutations``
     - ``5000``
     - Number of permutations

.. _configuration-behavior:

Behavioral Statistics
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Key
     - Description
   * - ``behavior_analysis.predictor_type``
     - Predictor variable type: ``"continuous"`` (default), ``"binary"``, or ``"categorical"``
   * - ``behavior_analysis.statistics.correlation_method``
     - Correlation method: ``"spearman"`` (default), ``"pearson"``, ``"kendall"``
   * - ``behavior_analysis.correlations.loso_stability``
     - ``true`` — compute LOSO stability of feature–behavior correlations
   * - ``behavior_analysis.statistics.fdr_alpha``
     - FDR :math:`q`-value for multiple comparison correction (default ``0.05``)
   * - ``behavior_analysis.statistics.n_permutations``
     - Permutation count for non-parametric tests (default ``1000``)
   * - ``behavior_analysis.predictor_residual.method``
     - Residualization method: ``"spline"``, ``"poly"``

.. _configuration-runtime-overrides:

Runtime Overrides (``--set``)
-----------------------------

Use ``--set`` for long-tail parameters that do not justify dedicated CLI flags.
This keeps the CLI and TUI maintainable while preserving configurability.

Where to use it:

- CLI: repeat ``--set KEY=VALUE``
- TUI: Advanced settings, ``Config Overrides`` (``key=value;key2=value2``)

.. code-block:: bash

   # Override behavior statistics at runtime
   eeg-pipeline behavior compute --subject 0001 \
     --set behavior_analysis.statistics.fdr_alpha=0.01 \
     --set behavior_analysis.cluster.n_permutations=5000

   # Override plotting style defaults
   eeg-pipeline plotting visualize --subject 0001 --all-plots \
     --set plotting.defaults.dpi=400 \
     --set plotting.styling.colors.significant=\"#D62728\"

   # Override ML data/feature filters
   eeg-pipeline ml regression --all-subjects \
     --set machine_learning.data.feature_harmonization=union_impute \
     --set machine_learning.data.feature_bands='[\"alpha\",\"beta\"]'

Notes:

- Values are type-coerced (``true/false``, ``null``, ints, floats, JSON arrays/objects).
- ``--set`` is applied after command flags and has final precedence.
