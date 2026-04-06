Installation
============

.. raw:: html

   <p class="hero-intro">
     Clone, create a virtual environment, and install. The TUI and Docker
     image are optional — needed only for interactive use and EEG source
     localization respectively.
   </p>

Prerequisites
-------------

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Python ≥ 3.11
      :link: https://www.python.org/downloads/
      :link-type: url

      Required for all workflows.

   .. grid-item-card:: Git
      :link: https://git-scm.com/
      :link-type: url

      Required to clone the repository.

   .. grid-item-card:: Go 1.21+ *(optional)*
      :link: https://go.dev/dl/
      :link-type: url

      Required only for the interactive TUI.

Setup
-----

macOS / Linux
~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git
   cd eegfmri-pipeline
   python3.11 -m venv .venv311
   source .venv311/bin/activate
   pip install -e ".[dev,ml]"
   eeg-pipeline --help

Windows PowerShell
~~~~~~~~~~~~~~~~~~

.. code-block:: powershell

   git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git
   cd eegfmri-pipeline
   py -3.11 -m venv .venv311
   .venv311\Scripts\Activate.ps1
   pip install -e ".[dev,ml]"
   eeg-pipeline --help

.. note::

   ``[ml]`` adds PyTorch, required only for the CNN classifier
   (``ml classify --classification-model cnn``). For all other
   workflows: ``pip install -e ".[dev]"``.

TUI *(optional)*
----------------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: What it is

      Guided wizard UI for running any pipeline stage interactively.
      Compiles to a single static binary — no runtime dependencies.
      Requires **Go 1.21+**.

   .. grid-item-card:: Build & launch

      .. code-block:: bash

         cd eeg_pipeline/cli/tui
         go build -o eeg-tui .
         ./eeg-tui

      .. code-block:: powershell

         cd eeg_pipeline/cli/tui
         go build -o eeg-tui.exe .
         .\eeg-tui.exe

On first launch, open **Global Setup** (press ``C`` from the main menu) to
set your task name and data paths. See :doc:`user_guide/tui` for the full reference.

Windows support contract
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Area
     - Support level
   * - Install, CLI, TUI, validation, smoke checks
     - Supported natively on macOS and Windows.
   * - fMRI preprocessing (container-backed)
     - Use WSL2 or a Linux/macOS host. Native Windows is not supported in this pass.
   * - Docker-based BEM/source-localization helpers
     - Use WSL2 or a Linux/macOS host. Native Windows is not supported in this pass.

Docker Image (FreeSurfer + MNE) *(optional)*
--------------------------------------------

.. note::

   Required **only** for EEG source localization (BEM generation and
   coregistration). Skip if you are not running ``features sourcelocalization``.

.. code-block:: bash

   docker build --platform linux/amd64 \
     -t freesurfer-mne:7.4.1 \
     -f eeg_pipeline/docker_setup/Dockerfile.freesurfer-mne .

Environment Variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Purpose
   * - ``EEG_PIPELINE_FREESURFER_LICENSE``
     - Path to FreeSurfer ``license.txt``
   * - ``SUBJECTS_DIR``
     - FreeSurfer ``SUBJECTS_DIR``; overridden by ``paths.freesurfer_dir``
   * - ``FMRIPREP_DOCKER_IMAGE``
     - Override the fMRIPrep Docker image tag at runtime
   * - ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``
     - Limit BLAS/MKL threads; also configurable via ``environment.thread_limits``

Dependencies
------------

``pyproject.toml`` is the single source of truth for all dependencies and
version bounds.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Package
     - Version
     - Role
   * - MNE-Python
     - ≥ 1.9.0
     - EEG processing, source localization
   * - MNE-BIDS
     - ≥ 0.16.0
     - BIDS I/O
   * - MNE-Connectivity
     - ≥ 0.7.0
     - Functional connectivity
   * - MNE-ICALabel
     - ≥ 0.7.0
     - Automatic ICA classification
   * - MNE-BIDS-Pipeline
     - ≥ 1.9.0
     - ICA detection fallback
   * - PyPREP
     - ≥ 0.4.3
     - Bad channel detection
   * - specparam
     - ≥ 2.0.0rc3
     - Aperiodic (1/f) fitting
   * - Nilearn
     - ≥ 0.11.1
     - fMRI GLM and neuroimaging
   * - NiBabel
     - ≥ 3.2.0, < 6.0
     - NIfTI/CIFTI I/O
   * - scikit-learn
     - ≥ 1.0.0, < 2.0
     - Machine learning models
   * - SHAP
     - ≥ 0.40.0
     - Feature importance
   * - PyTorch
     - ≥ 2.7.1 *(optional)*
     - Deep learning (EEGNet CNN)
   * - NumPy
     - ≥ 1.24, < 2.0
     - Array computation
   * - SciPy
     - ≥ 1.15.3
     - Scientific computing
   * - pandas
     - ≥ 2.3.0
     - Data manipulation
   * - pyarrow
     - ≥ 17.0.0
     - Parquet I/O for feature tables
   * - matplotlib
     - ≥ 3.10.3
     - Plotting backend
   * - PyYAML
     - ≥ 6.0, < 7.0
     - Configuration file parsing
