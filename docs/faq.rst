Frequently Asked Questions
==========================

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: :octicon:`download` Installation

      Virtual environment, PyTorch, Go TUI.

   .. grid-item-card:: :octicon:`database` Data

      Subject discovery, events files,
      BIDS layout.

   .. grid-item-card:: :octicon:`pulse` Preprocessing

      ICA components, epoch rejection,
      epoch windows.

   .. grid-item-card:: :octicon:`graph` Feature Extraction

      Performance, aperiodic fits,
      parallel jobs.

   .. grid-item-card:: :octicon:`dependabot` Machine Learning

      NaN metrics, feature harmonization,
      LOSO debugging.

   .. grid-item-card:: :octicon:`workflow` fMRI & Source

      fMRIPrep paths, GLM contrast maps,
      BEM generation, TUI.

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

Install the CPU-only build:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cpu

PyTorch is only needed for ``ml classify --classification-model cnn``.
Omit ``[ml]`` for all other workflows.

----

Data
----

**The pipeline reports zero subjects.**

1. ``paths.bids_root`` must point to the folder *containing* ``sub-XXXX/`` folders.
2. Subject folders must be named exactly ``sub-XXXX`` (BIDS convention).
3. EEG files must match ``pyprep.file_extension`` (default ``.vhdr``).

Run ``eeg-pipeline info subjects`` to see what is discovered.

----

**Do I need one** ``events.tsv`` **per run?**

Yes — name each file ``sub-XXXX_task-<task>_run-0N_events.tsv``.
The pipeline concatenates them automatically with run-offset alignment.

----

Preprocessing
-------------

**ICA fitting fails: No ICA components found.**

``ica.n_components`` exceeds the data rank after bad-channel removal. Lower it:

.. code-block:: yaml

   ica:
     n_components: 0.95

----

**Most epochs are being rejected.**

Long epochs accumulate slow drift that triggers fixed PTP thresholds.
Switch to local autoreject:

.. code-block:: yaml

   epochs:
     reject: "autoreject_local"
     autoreject_n_interpolate: [4, 8, 16]

Also check that ``epochs.tmin`` ≤ ``time_windows.baseline_tfr[0]``.

----

Feature Extraction
------------------

**Feature extraction is slow.**

Enable parallelism and limit categories:

.. code-block:: yaml

   feature_engineering:
     parallel:
       n_jobs_bands: -1
       n_jobs_connectivity: -1
       n_jobs_aperiodic: -1

.. code-block:: bash

   eeg-pipeline features compute --all-subjects \
       --categories power aperiodic connectivity

----

**Aperiodic fits failing (min_r2 threshold not met).**

.. code-block:: yaml

   feature_engineering:
     aperiodic:
       min_r2: 0.5
       model: "knee"   # better for data with strong low-frequency peaks

----

Machine Learning
----------------

**LOSO regression returns NaN for every subject.**

- Check for all-NaN feature columns: ``eeg-pipeline info features --subject XXXX``
- Switch harmonization strategy:

.. code-block:: bash

   eeg-pipeline ml regression --all-subjects \
       --set machine_learning.data.feature_harmonization=union_impute

----

fMRI
----

**fMRIPrep crashes: Permission denied on work directory.**

Set an absolute writable path:

.. code-block:: yaml

   fmri_preprocessing:
     fmriprep:
       work_dir: "/tmp/fmriprep_work"

----

**First-level GLM produces empty contrast maps.**

1. ``fmri_contrast.enabled: true`` must be set (disabled by default).
2. ``fmri_contrast.condition_a.value`` must match a ``trial_type`` value in your ``events.tsv``.
3. Confirm fMRIPrep outputs exist: ``eeg-pipeline validate derivatives --subject XXXX``

----

Source Localization
-------------------

**BEM generation fails: No T1w image found.**

Run FreeSurfer recon-all first, then point the config at the output:

.. code-block:: bash

   recon-all -s sub-0001 -i /path/to/T1w.nii.gz -all

.. code-block:: yaml

   feature_engineering:
     sourcelocalization:
       subjects_dir: "../../../data/derivatives/freesurfer"
       subject: "sub-0001"

----

TUI
---

**The TUI cannot find my Python environment.**

The TUI searches these paths in order:
``eeg_pipeline/.venv311``, ``.venv311``, ``.venv``, ``venv``, system ``python3``.
Ensure one exists and has the pipeline installed.

----

**The TUI exits immediately with** ``panic: terminal not attached``.

The TUI requires an interactive terminal (TTY). Do not run it inside a
non-interactive shell or pipe.
