Installation
============

.. raw:: html

   <p class="hero-intro">
     Environment setup, optional TUI build, FreeSurfer + MNE Docker image,
     and required environment variables.
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

.. code-block:: bash

   git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git
   cd eegfmri-pipeline
   python3.11 -m venv .venv311
   source .venv311/bin/activate        # Windows: .venv311\Scripts\activate
   pip install -e ".[dev,ml]"
   eeg-pipeline --help

.. note::

   The ``[ml]`` extra adds PyTorch, required only for the CNN classifier
   (``ml classify --classification-model cnn``). Omit it for all other
   workflows: ``pip install -e ".[dev]"``.

TUI
---

Recommended for interactive use. Requires **Go 1.21+** and compiles to a
single static binary with no runtime dependencies.

.. code-block:: bash

   cd eeg_pipeline/cli/tui && go build -o eeg-tui . && cd -
   ./eeg_pipeline/cli/tui/eeg-tui

On first launch, open **Global Setup** (press ``C`` from the main menu) to
set your task name and data paths before running any pipeline.

See :doc:`user_guide/tui` for the full reference.

Docker Image (FreeSurfer + MNE)
--------------------------------

Required **only** for EEG source localization (BEM generation and coregistration).
Skip this if you are not running ``features source-localization``.

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
