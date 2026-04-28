Frequently Asked Questions
==========================

.. raw:: html

   <p class="hero-lede">
     Common operational issues organized by pipeline stage. Expand a question
     to see the diagnosis and resolution.
   </p>

.. _faq-installation:

Installation
------------

.. dropdown:: ModuleNotFoundError: No module named 'mne' after installing.
   :animate: fade-in

   The wrong Python interpreter is being used. The most reliable fix on
   Windows is to skip activation and call the virtual environment executables
   directly:

   .. code-block:: powershell

      .\.venv\Scripts\python.exe -m pip install -e ".[dev,ml]"
      .\.venv\Scripts\eeg-pipeline.exe --help

   Activation also works if your shell allows it:

   .. code-block:: bash

      source .venv/bin/activate

   .. code-block:: powershell

      .venv\Scripts\Activate.ps1

   Then reinstall if needed: ``pip install -e ".[dev,ml]"``.

   Windows and macOS/Linux use different activation paths:

   - macOS/Linux: ``source .venv/bin/activate``
   - Windows: ``.venv\Scripts\Activate.ps1``

   The project requires Python ``3.11+``, but it does not require exactly
   ``3.11``. Any supported interpreter version is fine.

.. dropdown:: PowerShell says running scripts is disabled when I activate the venv.
   :animate: fade-in

   You can avoid activation entirely:

   .. code-block:: powershell

      .\.venv\Scripts\python.exe -m pip install -e ".[dev,ml]"
      .\.venv\Scripts\eeg-pipeline.exe info subjects

   Or enable scripts for the current PowerShell window only:

   .. code-block:: powershell

      Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
      .\.venv\Scripts\Activate.ps1

.. dropdown:: PyTorch install fails.
   :animate: fade-in

   PyTorch is only needed for the CNN classifier
   (``ml classify --classification-model cnn``). Omit ``[ml]`` for all other
   workflows, or install the CPU-only build separately:

   .. code-block:: bash

      pip install torch --index-url https://download.pytorch.org/whl/cpu

.. dropdown:: Go TUI build fails: 'go' not found.
   :animate: fade-in

   Install Go 1.21+ from `go.dev/dl <https://go.dev/dl/>`_, then:

   .. code-block:: bash

      cd eeg_pipeline/cli/tui
      go mod download
      go build -o eeg-tui .

   .. code-block:: powershell

      cd eeg_pipeline/cli/tui
      go mod download
      go build -o eeg-tui.exe .

   Launch the TUI with the platform-specific binary:

   - macOS/Linux: ``./eeg-tui``
   - Windows PowerShell: ``.\eeg-tui.exe``

----

.. _faq-data:

Data
----

.. dropdown:: The pipeline reports zero subjects.
   :animate: fade-in

   1. ``paths.bids_root`` must point to the folder *containing* ``sub-XXXX/``
      folders (not inside a subject folder).
   2. Subject folders must be named exactly ``sub-XXXX`` (BIDS convention).
   3. EEG files must match ``pyprep.file_extension`` (default ``.vhdr``).
   4. ``project.task`` must match the ``task-<name>`` segment in your BIDS
      filenames if you want the TUI and status views to discover epochs and
      features consistently.

   Run ``eeg-pipeline info subjects`` to see what is discovered and why
   subjects might be missing.

.. dropdown:: The TUI shows subjects, but says epochs are missing.
   :animate: fade-in

   This usually means the configured task label does not match the epoch
   filenames already on disk. For example, files named
   ``sub-0001_task-thermalactive_proc-clean_epo.fif`` will not be found if
   ``project.task`` is still ``task``.

   Check both subject discovery and derivative status:

   .. code-block:: bash

      eeg-pipeline info subjects
      eeg-pipeline info subjects --status

   Then update ``project.task`` in ``eeg_config.yaml`` or via the TUI Global
   Setup screen so it matches the ``task-<name>`` segment used by your data.

.. dropdown:: Do I need one events.tsv per run?
   :animate: fade-in

   For event-related workflows, yes — name each file
   ``sub-XXXX_task-<task>_run-0N_events.tsv``. The pipeline concatenates them
   automatically with run-offset alignment. Required columns: ``onset``,
   ``duration``, ``trial_type``.
   Any additional predictor or outcome columns are read alongside these.

.. dropdown:: What is trial_id and why does it matter?
   :animate: fade-in

   For event-related preprocessing (``task_is_rest: false``), ``proc-clean_events.tsv``
   is written to derivatives containing only kept epochs, each assigned a
   canonical ``trial_id`` integer. Resting-state preprocessing does not export
   this file.
   Every downstream stage (feature tables, fMRI betas, behavioral targets) must
   join on ``trial_id`` — it is the only valid alignment key across modalities.
   Row-order alignment is not accepted.

.. dropdown:: How do I process only a subset of subjects?
   :animate: fade-in

   Either pass ``--subject XXXX`` flags on the command line:

   .. code-block:: bash

      eeg-pipeline features compute --subject 0001 --subject 0002

   Or set a fixed list in the config to apply globally:

   .. code-block:: yaml

      project:
        subject_list: ["0001", "0002", "0003"]

----

.. _faq-preprocessing:

Preprocessing
-------------

.. dropdown:: ICA fitting fails: no ICA components found.
   :animate: fade-in

   ``ica.n_components`` exceeds the data rank after bad-channel removal.
   Lower it:

   .. code-block:: yaml

      ica:
        n_components: 0.95

.. dropdown:: Most epochs are being rejected.
   :animate: fade-in

   Long epochs accumulate slow drift that trips fixed PTP thresholds.
   Use local autoreject instead:

   .. code-block:: yaml

      epochs:
        reject: "autoreject_local"
        autoreject_n_interpolate: [4, 8, 16]

   Also verify that ``epochs.tmin`` ≤ ``time_windows.baseline_tfr[0]``
   so the baseline window falls inside the epoch.

.. dropdown:: Bad channels detected in one run are not propagated to other runs.
   :animate: fade-in

   The ``full`` and ``bad-channels`` modes synchronize bads across runs by
   default (union of bads per subject and task). If you re-run only ``epochs``
   after modifying ``channels.tsv`` manually, run ``bad-channels`` first to
   trigger re-synchronization.

----

.. _faq-feature-extraction:

Feature Extraction
------------------

.. dropdown:: How do I run the pipeline on resting-state EEG data?
   :animate: fade-in

   Set ``task_is_rest: true`` in ``eeg_config.yaml`` (or pass
   ``--task-is-rest`` to preprocessing) and point ``paths.bids_rest_root``
   at your resting-state BIDS directory:

   .. code-block:: yaml

      preprocessing:
        task_is_rest: true
        rest_epochs_duration: 10.0
        rest_epochs_overlap: 0.0
      paths:
        bids_rest_root: "../../../data/bids_output/eeg_rest"
        deriv_rest_root: "../../../data/derivatives/rest"

   No ``events.tsv`` is required. Event-locked families (``erp``, ``erds``,
   ``itpc``, ``pac``) are invalid in rest mode and raise an error if requested.

   .. code-block:: bash

      eeg-pipeline preprocessing full --subject 0001 --task-is-rest
      eeg-pipeline features compute --subject 0001 \
        --categories power connectivity aperiodic spectral complexity

   See :doc:`user_guide/output_formats` and :doc:`user_guide/configuration`
   for the full resting-state config reference.

.. dropdown:: Feature extraction is slow.
   :animate: fade-in

   Enable parallelism in the config:

   .. code-block:: yaml

      feature_engineering:
        parallel:
          n_jobs_bands: -1
          n_jobs_connectivity: -1
          n_jobs_aperiodic: -1

   Or pass it at runtime:

   .. code-block:: bash

      eeg-pipeline features compute --all-subjects \
        --n-jobs-bands -1 \
        --n-jobs-connectivity -1 \
        --n-jobs-aperiodic -1

   Limit to the families you need:

   .. code-block:: bash

      eeg-pipeline features compute --all-subjects \
        --categories power aperiodic connectivity

.. dropdown:: Aperiodic fits are failing or have very low R².
   :animate: fade-in

   Relax the minimum R² or switch to the knee model:

   .. code-block:: yaml

      feature_engineering:
        aperiodic:
          min_r2: 0.5
          model: "knee"

   Also ensure the PSD frequency range covers the aperiodic background
   (at least 1–40 Hz).

.. dropdown:: The connectivity family is very slow.
   :animate: fade-in

   Restrict to the most informative measure and band:

   .. code-block:: yaml

      feature_engineering:
        connectivity:
          methods: ["wpli"]
          granularity: "trial"

   Or compute only connectivity with ``--categories connectivity`` and a
   low ``n_jobs_connectivity`` to avoid memory pressure.

----

.. _faq-machine-learning:

Machine Learning
----------------

.. dropdown:: LOSO regression returns NaN metrics for every subject.
   :animate: fade-in

   - Check for all-NaN feature columns: ``eeg-pipeline info features 0001``
   - At least two subjects are required for LOSO; check ``info subjects``
   - Try the union-impute harmonization strategy:

   .. code-block:: bash

      eeg-pipeline ml regression --all-subjects \
        --set machine_learning.data.feature_harmonization=union_impute

.. dropdown:: ML uses features I haven't extracted yet.
   :animate: fade-in

   Run feature extraction with ``--analysis-mode trial_ml_safe`` before ML:

   .. code-block:: bash

      eeg-pipeline features compute --all-subjects --analysis-mode trial_ml_safe
      eeg-pipeline ml regression --all-subjects

.. dropdown:: Classification returns uniform predictions (all same class).
   :animate: fade-in

   ``class_weight="balanced"`` is always applied, but extreme class imbalance
   can still collapse predictions. Try SMOTE resampling:

   .. code-block:: bash

      eeg-pipeline ml classify --subject 0001 --subject 0002 \
        --set machine_learning.classification.resampler=smote

----

.. _faq-fmri-and-source-localization:

fMRI and Source Localization
-----------------------------

.. dropdown:: fMRIPrep crashes: Permission denied on the work directory.
   :animate: fade-in

   Set an absolute writable path:

   .. code-block:: yaml

      fmri_preprocessing:
        fmriprep:
          work_dir: "/tmp/fmriprep_work"

.. dropdown:: First-level GLM produces empty contrast maps.
   :animate: fade-in

   1. ``fmri_contrast.condition_a.value`` must exactly match a ``trial_type``
      value in your ``events.tsv``. Use ``eeg-pipeline info fmri-conditions``
      to list available values.
   2. Confirm fMRIPrep outputs exist under the expected derivatives path.

   .. code-block:: bash

      eeg-pipeline info fmri-conditions
      eeg-pipeline validate bids
      bids-validator /path/to/bids_root

.. dropdown:: BEM generation fails: no T1w image found.
   :animate: fade-in

   Run FreeSurfer ``recon-all`` first, then point the config at the output:

   .. code-block:: bash

      recon-all -s sub-0001 -i /path/to/T1w.nii.gz -all

   .. code-block:: yaml

      paths:
        freesurfer_dir: "../../../data/derivatives/freesurfer"

----

.. _faq-tui:

TUI
---

.. dropdown:: Do I need to edit YAML files before running a pipeline?
   :animate: fade-in

   Not if you use the TUI. Open **Global Setup** (press ``C`` from the main
   menu or navigate to *Utilities → Global Setup*) to set the task name and
   all data paths interactively. Settings persist across sessions. YAML editing
   is only necessary for parameters not exposed in Global Setup (ICA
   thresholds, aperiodic model, etc.).

.. dropdown:: The TUI cannot find my Python environment.
   :animate: fade-in

   The TUI searches for a virtual environment in this order:
   ``eeg_pipeline/.venv311`` → ``.venv311`` → ``.venv`` → ``venv`` →
   system interpreter. On macOS/Linux it falls back to ``python3``; on
   Windows it tries ``python`` and then ``py -3``. Ensure one of those
   environments has the package installed.

.. dropdown:: Why does native Windows reject fMRI preprocessing or Docker-based BEM helpers?
   :animate: fade-in

   That support boundary is intentional. Native Windows is supported for the
   repo-owned interface layer: install, CLI bootstrap, TUI, validation, and
   smoke checks. Container-backed fMRI preprocessing and Docker-based
   BEM/source-localization helpers should be run from WSL2 or a Linux/macOS
   host.

.. dropdown:: The TUI exits immediately with "panic: terminal not attached".
   :animate: fade-in

   The TUI requires an interactive TTY. Do not run it inside a non-interactive
   shell, a subshell without a TTY, or a pipe. Use a regular terminal session.
