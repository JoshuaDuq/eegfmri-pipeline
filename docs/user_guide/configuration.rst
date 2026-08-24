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

**Override precedence** (highest wins): ``--set`` overrides → CLI flags →
persisted TUI overrides → YAML defaults.

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

   Relative paths are generally resolved from the directory of the config file
   that defines them (for example ``eeg_config.yaml``,
   ``behavior_config.yaml``, or ``fmri_config.yaml``).
   Paths beginning with ``data/`` or ``eeg_pipeline/`` are resolved from the
   project root. Use absolute paths when you need location-independent
   configuration.

.. _configuration-quick-nav:

Quick Navigation
----------------

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: :ref:`Your Own Study <configuration-your-own-config>`

      Per-study configs, presets, EEG-only and resting-state setups.

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

.. _configuration-your-own-config:

Configuring Your Own Study
--------------------------

The packaged ``eeg_config.yaml`` describes **one study**: a thermal-pain EEG-fMRI
acquisition. Keys marked ``STUDY-SPECIFIC`` in that file encode that paradigm.

Do not edit it. It lives inside the installed package, so edits are lost on
reinstall and two studies cannot coexist. Write your own file that ``extends``
a preset instead:

.. code-block:: yaml

   # my_study.yaml
   extends: "eeg_only"

   project:
     task: "oddball"

   paths:
     bids_root: "/data/my_study/bids"
     deriv_root: "/data/my_study/derivatives"

.. code-block:: bash

   eeg-pipeline --config my_study.yaml preprocessing full --all-subjects

Only the keys you name are overridden; everything else is inherited, so later
corrections to the scientific defaults reach your study without being copied.
``--config`` is accepted before or after the subcommand, and
``EEG_PIPELINE_CONFIG`` sets the same thing for a whole shell session.

``extends`` accepts a packaged preset name or a path to another YAML file
(relative paths resolve against the extending file). Chains are followed, and
a cycle is reported rather than recursed.

.. list-table:: Packaged presets
   :header-rows: 1
   :widths: 20 80

   * - Preset
     - Purpose
   * - ``eeg_only``
     - EEG recorded outside an MR scanner, in **either** paradigm. Turns off
       every scanner-only stage: Analyzer pulse QC, cardiac attenuation QC,
       scanner-harmonic QC, the ECG coupling metric, the ICA cardiac review,
       and volume-bound trimming.
   * - ``rest``
     - Resting-state acquired **inside** a scanner. Keeps the gradient and
       pulse-artifact handling; only the paradigm differs from the packaged
       config.

Most studies want ``eeg_only`` and the paradigm switch. These are the two
independent choices — scanner or not, events or not — and setting both is the
whole configuration:

.. code-block:: yaml

   # Resting-state EEG recorded outside a scanner
   extends: "eeg_only"
   project:
     paradigm: "rest"
     task: "rest"        # the BIDS task- entity on your files, not a condition

``project.task`` is required in both paradigms. BIDS puts a ``task-`` entity on
every EEG file, resting-state ones included, and it is how both the recordings
and the cleaned epochs are located; ``task-rest`` is the usual label.

Setting ``paradigm: rest`` also removes the inherited settings that only mean
something when an event happened — the event-locked feature families (``erp``,
``erds``, ``itpc``) and the baseline-relative component TFR with its condition
contrasts. A family your own config **names** is left in place and reported
against the paradigm instead, so an explicit request is answered rather than
quietly dropped.

.. _configuration-paradigm:

Paradigm and acquisition
~~~~~~~~~~~~~~~~~~~~~~~~

Two keys describe what was recorded. Set these first; most other differences
follow from them.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Default
     - Description
   * - ``project.paradigm``
     - ``task``
     - ``task`` (event-related) or ``rest`` (fixed-length segments). Sets
       ``preprocessing.task_is_rest``, ``feature_engineering.task_is_rest`` and
       both fMRI equivalents together, overriding whatever they say
       individually. Leave ``null`` to keep setting those four by hand.
   * - ``preprocessing.eeg_fmri``
     - ``true``
     - Whether the EEG was recorded inside an MR scanner. Gates every stage
       whose inputs only exist for EEG-fMRI. Set ``false`` and the ordinary
       path runs — filtering, PyPREP, ICA with ICLabel, the ocular review,
       epoching — without asking for an ECG channel, volume markers, or
       Analyzer output that will not be found.

.. tip::

   .. code-block:: bash

      eeg-pipeline validate --config-only --config my_study.yaml

   Reports every contradiction at once, reads no derivatives, and takes under a
   second. **Errors** are things the run cannot produce. **Warnings** are
   stages switched on that this dataset gives the pipeline no way to compute,
   which will be skipped — expected in a config adapted from another study.

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
     - Resting-state EEG BIDS directory, for a study that acquires **both**
       task and rest and keeps them in separate trees. A rest-only study
       leaves this ``null``; ``paths.bids_root`` is used instead.
   * - ``paths.bids_fmri_root``
     - ``"../../../data/bids_output/fmri"``
     - BIDS-formatted fMRI data directory
   * - ``paths.deriv_root``
     - ``"../../../data/derivatives"``
     - Processed derivatives output directory
   * - ``paths.deriv_rest_root``
     - ``null``
     - Resting-state EEG derivatives directory. As above, a rest-only study
       leaves this ``null`` and uses ``paths.deriv_root``.
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
     - Root directory for multivariate signature weight maps; required when ``paths.signature_maps`` is non-empty
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
     - Write post-rejection ``*_proc-clean_events.tsv`` for event-related preprocessing (rest mode skips this export)

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
   * - ``ica.cardiac_review.enabled``
     - ``false``
     - Add direct ECG, R-locked EEG, and MNE component evidence for manual review
   * - ``ica.cardiac_review.ecg_channel``
     - ``"ECG"``
     - ECG channel used for signal-based R-peak detection
   * - ``ica.cardiac_review.epoch_window`` / ``baseline``
     - ``[-0.4, 0.6]`` / ``[-0.4, -0.1]``
     - R-locked review epoch and source-standardization baseline (seconds)
   * - ``ica.cardiac_review.measurement_window``
     - ``[0.0, 0.4]``
     - Window used to select the displayed R-locked EEG topography
   * - ``ica.cardiac_review.ctps_threshold``
     - ``"auto"``
     - Sampling-aware MNE CTPS threshold; may instead be an explicit number in ``(0, 1]``
   * - ``ica.band_specific_report.enabled``
     - ``false``
     - Add standard-ICA component dossiers by band and exploratory independent band-ICA appendices
   * - ``ica.band_specific_report.fit_decim``
     - ``2``
     - Temporal decimation used only while fitting the additional diagnostic ICAs
   * - ``ica.band_specific_report.tfr.frequency_step_hz``
     - ``1.0``
     - FieldTrip-style frequency spacing
   * - ``ica.band_specific_report.tfr.time_min_s`` / ``time_max_s``
     - ``-5.0`` / ``14.4``
     - Displayed TFR interval
   * - ``ica.band_specific_report.tfr.time_step_s``
     - ``0.1``
     - TFR output time spacing
   * - ``ica.band_specific_report.tfr.baseline_tmin_s`` / ``baseline_tmax_s``
     - ``-5.0`` / ``-0.01``
     - Baseline used for decibel normalization
   * - ``ica.band_specific_report.comparisons``
     - ``[]``
     - Optional clean-events metadata columns and value groups for condition contrasts

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
     - *(14 default families)*
     - Active default feature families; ``directedconnectivity`` and
       ``sourcelocalization`` are available but not enabled by default. See
       :doc:`../methods/eeg/features`
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
     - ``"nipreps/fmriprep:25.2.5"``
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
   * - ``fmri_preprocessing.fmriprep.bold2anat_dof``
     - ``6``
     - Degrees of freedom for BOLD→anatomical registration
   * - ``fmri_preprocessing.fmriprep.skull_strip_template``
     - ``"OASIS30ANTs"``
     - Template for skull stripping
   * - ``fmri_preprocessing.fmriprep.nthreads``
     - ``0``
     - CPU threads (0 = auto)
   * - ``fmri_preprocessing.fmriprep.extra_args``
     - ``""`` *(effective runtime default)*
     - Additional CLI arguments appended verbatim to fMRIPrep.
       ``null`` and empty string are both treated as no extra arguments.

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
     - CLI/config selector is ``"fmriprep"``; when paired with ``fmri_contrast.require_fmriprep=false``, missing runs may fall back to raw BIDS BOLD
   * - ``fmri_contrast.require_fmriprep``
     - ``true``
     - Strict mode for first-level inputs; when ``true``, missing fMRIPrep BOLD/masks raise a hard error
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
     - ``"motion24+wmcsf+fd+compcor"``
     - Fixed nuisance regressor strategy; ``"auto"`` is available only as an explicit derivative-adaptive choice
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

fMRI Statistical Artifacts
--------------------------

``fmri_stats`` contains only choices that cause statistical derivatives to be
computed. It is deliberately separate from ``fmri_report``, which only renders
existing derivatives.

.. list-table::
   :header-rows: 1
   :widths: 38 22 40

   * - Key
     - Default
     - Description
   * - ``fmri_stats.space``
     - ``"native"``
     - ``"mni"`` or ``"both"`` explicitly requests an additional standard-space fit
   * - ``fmri_stats.include_effect_size``
     - ``true``
     - Save the contrast effect map used by effect/evidence diagnostics
   * - ``fmri_stats.include_standard_error``
     - ``true``
     - Save contrast variance for uncertainty diagnostics
   * - ``fmri_stats.include_signatures``
     - ``true``
     - Evaluate configured standard-space signature maps; requires MNI output
   * - ``fmri_stats.threshold_mode``
     - ``"z"``
     - Inferential display rule: ``"z"``, ``"fdr"``, or ``"none"``
   * - ``fmri_stats.z_threshold`` / ``fdr_q``
     - ``2.3`` / ``0.05``
     - Explicit height or FDR threshold recorded in the report manifest
   * - ``fmri_stats.cluster_min_voxels``
     - ``0``
     - Optional display-only extent filter; zero disables it

.. _configuration-fmri-report:

fMRI Subject Report
-------------------

The report command reads existing first-level manifests and never refits a model.
Dense maps and carpets are written as 200-DPI PNG; line, design, and matrix figures
are SVG and therefore resolution-independent.
Whole-model :math:`R^2` and whole-mask run-effect correlations are generated from
persisted fitted-model artifacts whenever available. They have no tuning keys: neither
panel thresholds data, selects peaks, excludes runs, or changes the analysis.

.. list-table::
   :header-rows: 1
   :widths: 38 22 40

   * - Key
     - Default
     - Description
   * - ``fmri_report.enabled`` / ``html_report``
     - ``true`` / ``true``
     - Required for ``fmri-analysis report``
   * - ``fmri_report.include_motion_qc``
     - ``true``
     - Run-level FD summary and motion–DVARS coupling
   * - ``fmri_report.include_carpet_qc``
     - ``true``
     - Analysis-mask carpet plot aligned with motion and censoring
   * - ``fmri_report.include_tsnr_qc``
     - ``true``
     - Analysis-mask tSNR map and run summaries
   * - ``fmri_report.include_design_qc``
     - ``true``
     - Design matrix, VIF, correlation, efficiency, and event diagnostics
   * - ``fmri_report.include_unthresholded``
     - ``true``
     - Separate diagnostic map; it is not presented as inferential evidence
   * - ``fmri_report.embed_images``
     - ``true``
     - Produce a self-contained HTML document

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
   * - ``fmri_group_level.permutation.two_sided``
     - ``true``
     - Test both tails for max-T t inference; use ``false`` only for a predeclared directional hypothesis
   * - ``fmri_group_level.permutation.random_state``
     - ``42``
     - Fixed seed passed to Nilearn for reproducible sampled permutations
   * - ``fmri_group_level.permutation.tfce``
     - ``false``
     - Add Nilearn's threshold-free cluster enhancement to the same permutation run; considerably slower
   * - ``fmri_group_level.permutation.cluster_forming_p``
     - ``null``
     - Cluster-forming threshold for cluster-extent and cluster-mass FWE, **in p-scale** (Nilearn converts it to a t height); ``null`` skips cluster-level inference
   * - ``fmri_group_level.threshold.height_control``
     - ``"fdr"``
     - Parametric cohort-report procedure: ``"fdr"``, ``"fpr"``, ``"bonferroni"``, or ``"none"``
   * - ``fmri_group_level.threshold.alpha``
     - ``0.05``
     - Predeclared p/q level used by the selected height-control procedure
   * - ``fmri_group_level.threshold.uncorrected_z_threshold``
     - ``3.09``
     - Fixed z height used only when ``height_control: "none"``
   * - ``fmri_group_level.threshold.cluster_min_voxels``
     - ``0``
     - Optional display extent filter; it is not cluster-level corrected inference
   * - ``fmri_group_level.threshold.two_sided``
     - ``true``
     - Test both t-statistic tails; ignored for omnibus F tests, which are one-sided
   * - ``fmri_group_level.threshold.min_distance_mm``
     - ``8.0``
     - Nilearn cluster-table subpeak separation, for display only
   * - ``fmri_group_level.report.enabled``
     - ``true``
     - Write the cohort report after successful second-level inference
   * - ``fmri_group_level.report.html_report``
     - ``true``
     - Render the HTML document; kept separate from ``enabled`` for explicit output control
   * - ``fmri_group_level.report.formats``
     - ``["png"]``
     - Dense maps are 200-DPI PNG; design and matrix panels are also written as SVG
   * - ``fmri_group_level.report.embed_images``
     - ``true``
     - Embed figures into a self-contained HTML document
   * - ``fmri_group_level.report.include_design_correlation``
     - ``true``
     - Include Nilearn's regressor-correlation plot when three to twenty non-constant columns make it informative and legible
   * - ``fmri_group_level.report.atlas_labels_img``
     - ``null``
     - MNI label volume naming the structure each cluster peak falls in; adds a ``Region`` column to the cluster and max-T peak tables
   * - ``fmri_group_level.report.atlas_labels_tsv``
     - ``null``
     - Optional index-to-name table for ``atlas_labels_img``; without it the raw label index is reported
   * - ``fmri_group_level.report.include_leave_one_out_influence``
     - ``true``
     - Refit the model without each participant and rethreshold, reporting how many voxels survive without them; costs one extra fit per participant
   * - ``fmri_group_level.report.include_residual_diagnostics``
     - ``true``
     - Fit the reported design with Nilearn's ``OLSModel`` and show its residuals against the Gaussian errors the report assumes
   * - ``fmri_group_level.report.include_interactive_viewer``
     - ``true``
     - Inline Nilearn's ``view_img`` volume viewer; self-contained and offline, and adds roughly 0.6 MB to the HTML
   * - ``fmri_group_level.report.surface_mesh``
     - ``null``
     - Cortical mesh for the surface projection panel (e.g. ``fsaverage``); left unset the panel is skipped, because Nilearn would download a mesh it cannot find
   * - ``fmri_group_level.report.include_true_discovery_proportion``
     - ``true``
     - Include Nilearn's ``cluster_level_inference`` panel, which bounds how much of each cluster is a true discovery
   * - ``fmri_group_level.report.include_unthresholded``
     - ``true``
     - Include the complete z field in a collapsed diagnostic section, not as inferential evidence

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
     - Correlation method: ``"spearman"`` (default) or ``"pearson"``
   * - ``behavior_analysis.correlations.loso_stability``
     - ``true`` — compute LOSO stability of feature–behavior correlations
   * - ``behavior_analysis.statistics.fdr_alpha``
     - FDR :math:`q`-value for multiple comparison correction (default ``0.05``)
   * - ``behavior_analysis.statistics.n_permutations``
     - Global/default permutation count (default ``1000``). Some stages use dedicated keys (for example ``behavior_analysis.regression.n_permutations`` and ``behavior_analysis.cluster.n_permutations``).
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

   # Override ML data/feature filters
   eeg-pipeline ml regression --all-subjects \
     --set machine_learning.data.feature_harmonization=union_impute \
     --set machine_learning.data.feature_bands='[\"alpha\",\"beta\"]'

Notes:

- Values are type-coerced (``true/false``, ``null``, ints, floats, JSON arrays/objects).
- ``--set`` is applied after command flags and has final precedence.
