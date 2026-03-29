fMRI Preprocessing
==================

Containerized fMRIPrep-style preprocessing via Docker or Apptainer.

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

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Option
     - Description
     - Default
   * - ``--engine``
     - ``docker`` or ``apptainer``
     - ``docker``
   * - ``--fmriprep-image``
     - Container image tag or URI
     - ``nipreps/fmriprep:25.2.4``
   * - ``--output-spaces``
     - Output coordinate spaces
     - ``MNI152NLin2009cAsym``, ``T1w``
   * - ``--fs-license-file``
     - FreeSurfer license path
     - ``paths.freesurfer_license``, else ``~/license.txt``
   * - ``--fs-subjects-dir``
     - FreeSurfer ``SUBJECTS_DIR``
     - auto
   * - ``--ignore``
     - Skip steps (e.g. ``fieldmaps slicetiming``)
     - none
   * - ``--level``
     - Processing level: ``full``, ``resampling``, or ``minimal``
     - ``full``
   * - ``--nthreads``
     - Max threads across all processes (``0`` = auto)
     - ``0``
   * - ``--omp-nthreads``
     - Max threads per process (``0`` = auto)
     - ``0``
   * - ``--dummy-scans``
     - Non-steady-state volumes to discard
     - ``0``
   * - ``--fd-spike-threshold``
     - Framewise displacement spike threshold (mm)
     - ``0.5``
   * - ``--dvars-spike-threshold``
     - Standardized DVARS spike threshold
     - ``1.5``
   * - ``--mem-mb``
     - Memory limit in MB
     - fMRIPrep default
   * - ``--fmriprep-extra-args``
     - Raw extra fMRIPrep CLI arguments (parsed with ``shlex``)
     - none
   * - ``--skip-bids-validation``
     - Skip bids-validator step
     - disabled
   * - ``--fs-no-reconall``
     - Disable FreeSurfer ``recon-all``
     - enabled
   * - ``--longitudinal``
     - Create unbiased structural template (longitudinal mode)
     - disabled

.. note::

   For container runs, the pipeline automatically ignores macOS metadata files
   (``._*``, ``.DS_Store``) by mounting a sanitized temporary BIDS view.

For full methods, see :doc:`../../methods/fmri/pipeline`.
