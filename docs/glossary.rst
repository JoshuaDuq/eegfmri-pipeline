Glossary
========

Definitions for domain-specific terms and acronyms used throughout this
documentation. Terms are linked from methods pages via ``:term:`` references.

.. glossary::
   :sorted:

   BIDS
      Brain Imaging Data Structure. A standard for organizing and describing
      neuroimaging and electrophysiology data and metadata.
      See `bids-specification.readthedocs.io <https://bids-specification.readthedocs.io/>`_.

   ICA
      Independent Component Analysis. A blind source separation method used
      to decompose EEG recordings into statistically independent components.
      Artifact components (eye movements, cardiac, muscle) are identified
      with ICLabel and removed before epoching.

   ICLabel
      A deep learning classifier that assigns probability scores to ICA
      components across seven categories: brain, muscle, eye, heart, line
      noise, channel noise, and other. Components above
      ``ica.probability_threshold`` in artifact classes are removed.

   TFR
      Time-Frequency Representation. A 2-D power (or complex-valued) signal
      decomposed across both time and frequency using Morlet wavelets. Used
      as the basis for power, ERDS, ITPC, and PAC feature extraction.

   wPLI
      Weighted Phase Lag Index. A connectivity measure that weights
      cross-spectrum phase differences by the imaginary component, suppressing
      volume-conduction artefacts. Ranges from −1 to 1; unsigned values
      typically used.

   AEC
      Amplitude Envelope Correlation. A connectivity measure computed as the
      Pearson correlation between band-passed signal envelopes, optionally
      after orthogonalization (``aec_mode: "orth"``) to reduce spurious
      zero-lag correlations.

   PAC
      Phase-Amplitude Coupling. A cross-frequency coupling measure quantifying
      how much the amplitude of a high-frequency band (e.g., gamma) is
      modulated by the phase of a low-frequency band (e.g., theta). Computed
      here using the Mean Vector Length (MVL) metric with surrogate testing.

   ITPC
      Inter-Trial Phase Clustering (also called inter-trial coherence, ITC).
      Measures phase consistency of the EEG signal across trials at each
      time-frequency point. Values range from 0 (random phase) to 1
      (perfectly phase-locked).

   ERDS
      Event-Related Desynchronization / Synchronization. TFR-derived measure
      of relative power change from a baseline period: negative = power
      decrease (desynchronization), positive = power increase (synchronization).

   IAF
      Individual Alpha Frequency. The dominant oscillatory frequency in the
      alpha band estimated from each subject's baseline PSD, used to
      adaptively shift the alpha (and surrounding) band definitions.

   LOSO
      Leave-One-Subject-Out cross-validation. A subject-level CV scheme where
      one subject is held out as the test set and the model is trained on all
      remaining subjects. Repeated for every subject.

   LSA
      Least Squares All. Trial-wise beta estimation where a single GLM
      includes all trial regressors simultaneously. Fast but can suffer from
      collinearity between adjacent trials.

   LSS
      Least Squares Separate. Trial-wise beta estimation where a separate GLM
      is fit for each trial, modelling the target trial as one regressor and
      all other trials as a single nuisance regressor. More robust to
      trial collinearity than LSA.

   HRF
      Hemodynamic Response Function. The canonical shape used to convolve
      stimulus timing in a GLM. This pipeline uses the SPM double-gamma HRF
      by default (``hrf_model: "spm"``).

   GLM
      General Linear Model. A statistical model expressing the BOLD signal as
      a linear combination of predictors (stimulus regressors, confounds, drift
      terms). Used for both first-level (per-subject) and second-level
      (group) fMRI analysis.

   confound strategy
      A named preset for selecting nuisance regressors from fMRIPrep's
      ``*_confounds_timeseries.tsv``. ``"auto"`` selects motion parameters,
      WM/CSF signals, and scrubbing spikes automatically.

   fMRIPrep
      A robust and reproducible fMRI preprocessing pipeline based on
      Nipype and ANTs. Runs inside Docker or Apptainer.
      See `fmriprep.readthedocs.io <https://fmriprep.readthedocs.io/>`_.

   BEM
      Boundary Element Model. A three-layer head model (scalp, skull, brain)
      derived from a subject's T1w MRI using FreeSurfer. Used to compute the
      EEG forward model for source localization.

   LCMV
      Linearly Constrained Minimum Variance beamformer. A spatial filter for
      EEG source localization that minimizes total output power subject to a
      unity-gain constraint at the target location.

   eLORETA
      Exact Low-Resolution Brain Electromagnetic Tomography. A standardized
      minimum-norm inverse solution with zero localization error for test
      sources. Used as an alternative to LCMV.

   CSD
      Current Source Density (also called surface Laplacian). A spatial
      filter that estimates the local current flow by computing the second
      spatial derivative of the scalp potential. Applied by default to
      phase-based connectivity and ITPC features to reduce volume conduction.

   FDR
      False Discovery Rate. Multiple comparison correction procedure
      (Benjamini-Hochberg) controlling the expected proportion of false
      positives among rejected hypotheses.

   Fisher-z
      Fisher's :math:`z`-transformation of a Pearson correlation coefficient:
      :math:`z = \text{arctanh}(r)`. Used to symmetrize and stabilize
      variance before averaging connectivity values across runs or subjects.

   trial_id
      A canonical integer column written by the preprocessing stage into
      ``*_proc-clean_events.tsv``. Each value identifies a single kept epoch
      after artifact rejection. All downstream tables (feature Parquet files,
      fMRI beta volumes, behavioral targets) must be joined on ``trial_id``;
      row-order alignment across files is not valid.

   task_is_rest
      A boolean config key (``preprocessing.task_is_rest`` and
      ``feature_engineering.task_is_rest``) that switches the pipeline into
      resting-state mode. When ``true``, preprocessing creates fixed-length
      overlapping segments instead of event-locked epochs, no ``events.tsv``
      conditions are required, and event-locked feature families (``erp``,
      ``erds``, ``itpc``, ``pac``) are rejected as invalid.

   Parquet
      A columnar binary file format (Apache Parquet) used for storing feature
      tables. Preserves dtype precision, supports metadata sidecar JSON,
      and is significantly faster to read than CSV for wide tables.

   SUBJECTS_DIR
      FreeSurfer's environment variable pointing to the directory containing
      reconstructed subject folders. Set in the pipeline via
      ``paths.freesurfer_dir``.

   MNE-Python
      An open-source Python package for EEG/MEG data analysis.
      See `mne.tools <https://mne.tools/>`_.

   Nilearn
      A Python library for fast and easy statistical learning on neuroimaging
      data. Used here for GLM fitting, contrast computation, and group-level
      inference.
      See `nilearn.github.io <https://nilearn.github.io/>`_.
