fMRI Preprocessing
==================

.. container:: page-intro

   Interface for containerized fMRIPrep execution via Docker or Apptainer.
   This command is the pipeline's BIDS-to-derivatives entry point for fMRI
   data and writes the preprocessed outputs consumed by downstream GLM and
   resting-state workflows.

.. note::

   The fMRI pipeline is **still under active development**. Container wiring,
   defaults, and derivative paths may change between releases. Confirm outputs
   after upgrades before using them in downstream analyses.

.. rst-class:: section-kicker

At A Glance

.. rst-class:: at-a-glance

.. grid:: 4
   :gutter: 2

   .. grid-item-card:: Purpose

      Run fMRIPrep in a containerized, pipeline-managed workflow.

   .. grid-item-card:: Inputs

      BIDS fMRI data, a container engine, and a valid FreeSurfer license.

   .. grid-item-card:: Outputs

      Preprocessed fMRI derivatives under the configured derivatives tree.

   .. grid-item-card:: Read this page when

      You need engine selection, output-space control, or resource and container options.

**Prerequisites:**

- BIDS-formatted fMRI data under ``paths.bids_fmri_root``
  (or ``paths.bids_rest_root`` when ``--task-is-rest`` is enabled)
- Docker or Apptainer installed and accessible on ``PATH``
- A FreeSurfer ``license.txt`` at ``paths.freesurfer_license`` or
  ``$EEG_PIPELINE_FREESURFER_LICENSE`` (required by current fMRIPrep execution path)

.. code-block:: bash

   eeg-pipeline fmri preprocess [options]

Mode
----

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - Purpose
   * - ``preprocess``
     - Run fMRIPrep and write outputs under the derivatives directory.

Examples
--------

.. tab-set::

   .. tab-item:: Docker

      .. code-block:: bash

         eeg-pipeline fmri preprocess --subject 0001 --engine docker

   .. tab-item:: Apptainer (HPC)

      .. code-block:: bash

         eeg-pipeline fmri preprocess --subject 0001 --engine apptainer

   .. tab-item:: Output Spaces

      .. code-block:: bash

         eeg-pipeline fmri preprocess --subject 0001 \
           --output-spaces T1w MNI152NLin2009cAsym

   .. tab-item:: Resources / Advanced

      .. code-block:: bash

         # Threads and memory constraints
         eeg-pipeline fmri preprocess --subject 0001 \
           --nthreads 8 --omp-nthreads 4 --mem-mb 24000

         # Pass through raw fMRIPrep args (parsed with shlex)
         eeg-pipeline fmri preprocess --subject 0001 \
           --fmriprep-extra-args '--verbose'

Key Options
-----------

.. rst-class:: doc-table

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Option
     - Description
     - Default
   * - ``--engine``
     - ``docker`` or ``apptainer``
     - ``docker``
   * - ``--task-is-rest`` / ``--no-task-is-rest``
     - Toggle resting-state fMRI mode (switches dataset roots to rest-specific paths)
     - from config (default: disabled)
   * - ``--fmriprep-image``
     - Container image tag or URI
     - ``nipreps/fmriprep:25.2.4``
   * - ``--fmriprep-output-dir``
     - Parent output directory (fMRIPrep writes to ``<output_dir>/fmriprep``)
     - ``<deriv_root>/preprocessed/fmri``
   * - ``--fmriprep-work-dir``
     - Scratch/work directory for intermediate files
     - ``<deriv_root>/work/fmriprep``
   * - ``--output-spaces``
     - Output coordinate spaces
     - ``MNI152NLin2009cAsym``, ``T1w``
   * - ``--fs-license-file``
     - FreeSurfer license path
     - ``paths.freesurfer_license``, else ``$EEG_PIPELINE_FREESURFER_LICENSE``, else ``~/license.txt``
   * - ``--fs-subjects-dir``
     - FreeSurfer ``SUBJECTS_DIR``
     - unset (optional)
   * - ``--ignore``
     - Skip steps (e.g. ``fieldmaps slicetiming``)
     - none
   * - ``--bids-filter-file``
     - Optional BIDS filter JSON passed to fMRIPrep
     - none
   * - ``--level``
     - Processing level: ``full``, ``resampling``, or ``minimal``
     - ``full``
   * - ``--low-mem``
     - Reduce memory usage (may increase runtime)
     - disabled
   * - ``--cifti-output``
     - Write CIFTI dense timeseries (``91k`` or ``170k``)
     - disabled
   * - ``--task-id``
     - Restrict preprocessing to a single task label
     - all tasks
   * - ``--nthreads``
     - Max threads across all processes (``0`` = auto)
     - ``0``
   * - ``--omp-nthreads``
     - Max threads per process (``0`` = auto)
     - ``0``
   * - ``--dummy-scans``
     - Non-steady-state volumes to discard
     - ``0``
   * - ``--random-seed``
     - Reproducibility seed for fMRIPrep stochastic steps (``0`` = no explicit seed)
     - ``0``
   * - ``--skull-strip-template``
     - Anatomical skull-strip template
     - ``OASIS30ANTs``
   * - ``--skull-strip-fixed-seed``
     - Use a fixed seed for skull-stripping
     - disabled
   * - ``--bold2t1w-init``
     - BOLD-to-T1w initialization strategy (``register`` or ``header``)
     - ``register``
   * - ``--bold2t1w-dof``
     - Degrees of freedom for BOLD-to-T1w registration
     - ``6``
   * - ``--slice-time-ref``
     - Slice timing reference fraction (0=start, 0.5=middle, 1=end)
     - ``0.5``
   * - ``--fd-spike-threshold``
     - Framewise displacement spike threshold (mm)
     - ``0.5``
   * - ``--dvars-spike-threshold``
     - Standardized DVARS spike threshold
     - ``1.5``
   * - ``--me-output-echos``
     - Emit separate outputs for each echo in multi-echo data
     - disabled
   * - ``--medial-surface-nan``
     - Fill medial cortical surface vertices with NaN
     - disabled
   * - ``--no-msm``
     - Disable MSM-Sulc alignment to fsLR surface space
     - disabled
   * - ``--mem-mb``
     - Memory limit in MB
     - fMRIPrep default
   * - ``--fmriprep-extra-args``
     - Raw extra fMRIPrep CLI arguments (parsed with ``shlex``)
     - none
   * - ``--skip-bids-validation`` / ``--no-skip-bids-validation``
     - Skip or enforce the bids-validator step
     - disabled
   * - ``--clean-workdir`` / ``--no-clean-workdir``
     - Remove or keep fMRIPrep work dir after successful run
     - enabled
   * - ``--stop-on-first-crash`` / ``--no-stop-on-first-crash``
     - Stop immediately after first crash report
     - disabled
   * - ``--use-aroma`` / ``--no-use-aroma``
     - Enable or disable ICA-AROMA denoising
     - disabled
   * - ``--fs-no-reconall`` / ``--fs-reconall``
     - Disable or enable FreeSurfer ``recon-all``
     - enabled
   * - ``--longitudinal``
     - Create unbiased structural template (longitudinal mode)
     - disabled

.. note::

   For container runs, the pipeline mounts a sanitized temporary BIDS view that
   excludes macOS metadata files such as ``._*`` and ``.DS_Store``.

.. note::

   ``fmri preprocess`` accepts shared CLI flags such as ``--task`` for
   interface consistency, but actual fMRIPrep task filtering is controlled by
   ``--task-id``.

.. seealso::

   :doc:`../../methods/fmri/pipeline`
      GLM specification, confound strategy, and beta estimation methods.

   :doc:`fmri_analysis`
      First-level GLM, group inference, beta-series, LSS, and resting-state.

   :doc:`../data_layout`
      Required BIDS fMRI directory layout and ``*_bold.json`` sidecar fields.

   :doc:`../configuration`
      ``fmri_preprocessing`` config section (engine, image, FD threshold, etc.).
