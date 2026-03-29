Quick Start
===========

A complete EEG/fMRI analysis run from ``git clone`` to plotted results.

.. note::

   Prefer a guided interface? :doc:`tui` wraps every step below in an
   interactive wizard — no command memorisation needed.

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Steps 1–3 · Setup

      Install · Place data · Validate

   .. grid-item-card:: Steps 4–7 · EEG Analysis

      Preprocess · Extract features ·
      Behaviour · Machine learning

   .. grid-item-card:: Steps 8–9 · Optional

      fMRI pipeline · Plot results

Every CLI step follows the same pattern:
``eeg-pipeline <command> <mode> [--subject XXXX | --all-subjects]``

1 — Install
-----------

.. code-block:: bash

   git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git
   cd eegfmri-pipeline
   python3.11 -m venv .venv311
   source .venv311/bin/activate
   pip install -e ".[dev,ml]"
   eeg-pipeline --help

See :doc:`../install` for TUI and Docker setup.

2 — Place Your Data
--------------------

Put BIDS-formatted EEG under ``data/bids_output/eeg/``
(see :doc:`data_layout` for the full layout):

.. code-block:: text

   data/bids_output/eeg/
   ├── dataset_description.json
   └── sub-0001/eeg/
       ├── sub-0001_task-task_run-01_eeg.vhdr
       ├── sub-0001_task-task_run-01_events.tsv
       └── sub-0001_task-task_run-01_channels.tsv

3 — Validate
-------------

.. code-block:: bash

   eeg-pipeline validate bids
   eeg-pipeline info subjects

4 — Preprocess
--------------

.. code-block:: bash

   # Single subject: bad channels → ICA → epochs
   eeg-pipeline preprocessing full --subject 0001

   # All subjects
   eeg-pipeline preprocessing full --all-subjects

Outputs: ``data/derivatives/sub-0001/eeg/*_proc-clean_epo.fif``

5 — Extract Features
---------------------

.. code-block:: bash

   # All default feature categories
   eeg-pipeline features compute --all-subjects

   # Specific categories
   eeg-pipeline features compute --all-subjects \
       --categories power connectivity aperiodic

Outputs: ``data/derivatives/sub-0001/eeg/features/<category>/*.parquet``

6 — Behavioral Analysis
------------------------

.. code-block:: bash

   eeg-pipeline behavior compute --all-subjects

7 — Machine Learning
--------------------

.. code-block:: bash

   # LOSO regression
   eeg-pipeline ml regression --all-subjects \
       --feature-categories power aperiodic

   # Classification
   eeg-pipeline ml classify --all-subjects --classification-model svm

8 — fMRI Analysis
------------------

.. code-block:: bash

   # fMRIPrep preprocessing
   eeg-pipeline fmri preprocess --subject 0001

   # First-level GLM
   eeg-pipeline fmri-analysis first-level --subject 0001 \
       --condition-a stimulation --condition-b rest

   # Group inference
   eeg-pipeline fmri-analysis second-level

See :doc:`../methods/fmri/pipeline` for the full fMRI methods reference.

9 — Plot
--------

.. code-block:: bash

   eeg-pipeline plotting visualize --subject 0001 --all-plots
   eeg-pipeline plotting visualize --all-subjects --mode group
