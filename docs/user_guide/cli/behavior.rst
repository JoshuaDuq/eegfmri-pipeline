Behavioral Analysis
===================

Statistical analyses linking EEG features to behavior (e.g., ratings,
temperature, and condition effects). Joins are always performed via the
canonical ``trial_id``.

.. code-block:: bash

   eeg-pipeline behavior [mode] [options]

Modes
-----

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - ``compute``
     - Run behavioral analysis stages and write numerical outputs
   * - ``visualize``
     - Plot from previously computed results

For stage definitions, statistical safeguards, and configuration, see
:doc:`../../methods/eeg/behavior`.

Examples
--------

.. code-block:: bash

   # Default analysis suite
   eeg-pipeline behavior compute --subject 0001

   # Selected stages only
   eeg-pipeline behavior compute --subject 0001 \
     --computations correlations condition temporal

   # Temperature-controlled with permutation testing
   eeg-pipeline behavior compute --subject 0001 \
     --control-temperature --n-perm 1000

   # Robust correlations with Bayes factors
   eeg-pipeline behavior compute --subject 0001 \
     --robust-correlation percentage_bend --compute-bayes-factors

   # Visualize from existing results
   eeg-pipeline behavior visualize --subject 0001

   # List available stages
   eeg-pipeline behavior compute --list-stages

See also:
:doc:`../subject_selection` (shared subject/task flags) and
:doc:`../../methods/eeg/behavior` (methods + configuration).
