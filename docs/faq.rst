Frequently Asked Questions
==========================

Jump to a section:

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Installation
      :link: #installation
      :link-type: url

      Virtual environment, PyTorch, Go TUI, missing modules.

   .. grid-item-card:: Data
      :link: #data
      :link-type: url

      Subject discovery, events files, BIDS layout.

   .. grid-item-card:: Preprocessing
      :link: #preprocessing
      :link-type: url

      ICA failures, epoch rejection, epoch windows.

   .. grid-item-card:: Feature Extraction
      :link: #feature-extraction
      :link-type: url

      Performance, aperiodic fits, parallel jobs.

   .. grid-item-card:: Machine Learning
      :link: #machine-learning
      :link-type: url

      NaN metrics, feature harmonization, LOSO debugging.

   .. grid-item-card:: fMRI & Source
      :link: #fmri-and-source-localization
      :link-type: url

      fMRIPrep paths, contrast maps, BEM generation.

----

.. _faq-installation:

Installation
------------

**ModuleNotFoundError: No module named 'mne' after installing.**

The virtual environment is not activated. Run:

.. code-block:: bash

   source .venv311/bin/activate     # macOS / Linux
   .venv311\Scripts\activate        # Windows

Then reinstall if needed: ``pip install -e ".[dev,ml]"``.

----

**PyTorch install fails.**

PyTorch is only needed for the CNN classifier
(``ml classify --classification-model cnn``). Omit ``[ml]`` for all other
workflows, or install the CPU-only build separately:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cpu

----

**Go TUI build fails: 'go' not found.**

Install Go 1.21+ from `go.dev/dl <https://go.dev/dl/>`_, then:

.. code-block:: bash

   cd eeg_pipeline/cli/tui
   go mod download
   go build -o eeg-tui .

----

.. _faq-data:

Data
----

**The pipeline reports zero subjects.**

1. ``paths.bids_root`` must point to the folder *containing* ``sub-XXXX/``
   folders (not inside a subject folder).
2. Subject folders must be named exactly ``sub-XXXX`` (BIDS convention).
3. EEG files must match ``pyprep.file_extension`` (default ``.vhdr``).

Run ``eeg-pipeline info subjects`` to see what is discovered and why
subjects might be missing.

----

**Do I need one** ``events.tsv`` **per run?**

Yes — name each file ``sub-XXXX_task-<task>_run-0N_events.tsv``.
The pipeline concatenates them automatically with run-offset alignment.
The required columns are ``onset``, ``duration``, and ``trial_type``.
Any additional predictor or outcome columns are read alongside these.

----

**What is** ``trial_id`` **and why does it matter?**

After preprocessing, the pipeline writes ``proc-clean_events.tsv`` to
derivatives. This file contains only rows for kept epochs, each assigned a
canonical ``trial_id`` integer. Every downstream stage (feature tables, fMRI
betas, behavioral targets) must join on ``trial_id`` — it is the only valid
alignment key across modalities. Row-order alignment is not accepted.

----

**How do I process only a subset of subjects?**

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

**ICA fitting fails: no ICA components found.**

``ica.n_components`` exceeds the data rank after bad-channel removal.
Lower it:

.. code-block:: yaml

   ica:
     n_components: 0.95

----

**Most epochs are being rejected.**

Long epochs accumulate slow drift that trips fixed PTP thresholds.
Use local autoreject instead:

.. code-block:: yaml

   epochs:
     reject: "autoreject_local"
     autoreject_n_interpolate: [4, 8, 16]

Also verify that ``epochs.tmin`` ≤ ``time_windows.baseline_tfr[0]``
so the baseline window falls inside the epoch.

----

**Bad channels detected in one run are not propagated to other runs.**

The ``full`` and ``bad-channels`` modes synchronize bads across runs
by default (union of bads per subject and task). If you re-run only
``epochs`` after modifying ``channels.tsv`` manually, run ``bad-channels``
first to trigger re-synchronization.

----

.. _faq-feature-extraction:

Feature Extraction
------------------

**How do I run the pipeline on resting-state EEG data?**

Set ``task_is_rest: true`` in ``eeg_config.yaml`` (or pass ``--task-is-rest``
to preprocessing) and point ``paths.bids_rest_root`` at your resting-state
BIDS directory:

.. code-block:: yaml

   preprocessing:
     task_is_rest: true
     rest_epochs_duration: 10.0   # seconds per segment
     rest_epochs_overlap: 0.0
   paths:
     bids_rest_root: "../../../data/bids_output/eeg_rest"
     deriv_rest_root: "../../../data/derivatives/rest"

No ``events.tsv`` is required. Event-locked families (``erp``, ``erds``,
``itpc``, ``pac``) are not valid in rest mode and raise an error if requested.
Use only rest-compatible families (for example: ``power``, ``connectivity``,
``aperiodic``, ``spectral``, ``complexity``).

.. code-block:: bash

   eeg-pipeline preprocessing full --subject 0001 --task-is-rest
   eeg-pipeline features compute --subject 0001 \
     --categories power connectivity aperiodic spectral complexity

See :doc:`user_guide/output_formats` for the full resting-state workflow, and
:doc:`user_guide/configuration` for all related config keys.

----

**Feature extraction is slow.**

Enable parallelism in the config:

.. code-block:: yaml

   feature_engineering:
     parallel:
       n_jobs_bands: -1
       n_jobs_connectivity: -1
       n_jobs_aperiodic: -1

Or pass it at runtime for a specific run:

.. code-block:: bash

   eeg-pipeline features compute --all-subjects \
     --n-jobs-bands -1 \
     --n-jobs-connectivity -1 \
     --n-jobs-aperiodic -1

Limit to the families you need for the current analysis:

.. code-block:: bash

   eeg-pipeline features compute --all-subjects \
     --categories power aperiodic connectivity

----

**Aperiodic fits are failing or have very low R².**

Relax the minimum R² or switch to the knee model:

.. code-block:: yaml

   feature_engineering:
     aperiodic:
       min_r2: 0.5
       model: "knee"   # better for data with strong low-frequency peaks

Also ensure the PSD is computed over a frequency range that includes the
aperiodic background (at least 1–40 Hz).

----

**The** ``connectivity`` **family is very slow.**

Restrict to the most informative measure and band:

.. code-block:: yaml

   feature_engineering:
     connectivity:
       methods: ["wpli"]
       granularity: "trial"

Alternatively, compute only the connectivity family with ``--categories connectivity``
and a low ``n_jobs_connectivity`` to avoid memory pressure.

----

.. _faq-machine-learning:

Machine Learning
----------------

**LOSO regression returns NaN metrics for every subject.**

- Check for all-NaN feature columns: ``eeg-pipeline info features 0001``
- At least two subjects are required for LOSO; check ``info subjects``
- Try the union-impute harmonization strategy:

.. code-block:: bash

   eeg-pipeline ml regression --all-subjects \
     --set machine_learning.data.feature_harmonization=union_impute

----

**ML uses features I haven't extracted yet.**

Run feature extraction with ``--analysis-mode trial_ml_safe`` before ML:

.. code-block:: bash

   eeg-pipeline features compute --all-subjects --analysis-mode trial_ml_safe
   eeg-pipeline ml regression --all-subjects

----

**Classification returns uniform predictions (all same class).**

``class_weight="balanced"`` is always applied, but extreme class imbalance
can still collapse predictions. Lower the classification threshold or try
SMOTE resampling:

.. code-block:: bash

   eeg-pipeline ml classify --subject 0001 --subject 0002 \
     --set machine_learning.classification.resampler=smote

----

.. _faq-fmri-and-source-localization:

fMRI and Source Localization
-----------------------------

**fMRIPrep crashes: Permission denied on the work directory.**

Set an absolute writable path:

.. code-block:: yaml

   fmri_preprocessing:
     fmriprep:
       work_dir: "/tmp/fmriprep_work"

----

**First-level GLM produces empty contrast maps.**

1. ``fmri_contrast.condition_a.value`` must exactly match a ``trial_type``
   value in your ``events.tsv``. Use ``eeg-pipeline info fmri-conditions``
   to list available values.
2. Confirm fMRIPrep outputs exist under the expected derivatives path.

.. code-block:: bash

   eeg-pipeline info fmri-conditions
   eeg-pipeline validate bids
   bids-validator /path/to/bids_root

----

**BEM generation fails: no T1w image found.**

Run FreeSurfer ``recon-all`` first, then point the config at the output:

.. code-block:: bash

   recon-all -s sub-0001 -i /path/to/T1w.nii.gz -all

.. code-block:: yaml

   paths:
     freesurfer_dir: "../../../data/derivatives/freesurfer"

----

**Do I need to edit YAML files before running a pipeline?**

Not if you use the TUI. Open **Global Setup** (press ``C`` from the main menu
or navigate to *Utilities → Global Setup*) to set the task name and all data
paths interactively. Settings persist across sessions. YAML editing is only
necessary for parameters not exposed in Global Setup (ICA thresholds,
aperiodic model, etc.).

----

**The TUI cannot find my Python environment.**

The TUI searches for a virtual environment in this order:
``eeg_pipeline/.venv311`` → ``.venv311`` → ``.venv`` → ``venv`` →
system ``python3``. Ensure one exists at a recognized path and has the
package installed.

----

**The TUI exits immediately with** ``panic: terminal not attached``.

The TUI requires an interactive TTY. Do not run it inside a non-interactive
shell, a subshell without a TTY, or a pipe. Use a regular terminal session.
