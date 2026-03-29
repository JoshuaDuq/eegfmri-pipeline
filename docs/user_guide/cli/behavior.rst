Behavioral Analysis
===================

Statistical analyses linking EEG features to behavior (e.g. pain ratings,
temperature, conditions). All stages operate on a trial table with explicit
column semantics. The trialwise join contract is canonical ``trial_id``, not
inferred paradigm columns.

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
     - Run behavioral analysis stages and save numerical outputs
   * - ``visualize``
     - Generate standardized plots from previously computed results

For all 17 stage definitions, the pipeline DAG, statistical safeguards, and
configuration details, see :doc:`../../methods/eeg/behavior`.

Examples
--------

.. code-block:: bash

   # All behavioral analyses
   eeg-pipeline behavior compute --subject 0001

   # Specific stages
   eeg-pipeline behavior compute --subject 0001 \
     --computations correlations condition temporal

   # Temperature-controlled with permutation testing
   eeg-pipeline behavior compute --subject 0001 \
     --control-temperature --n-perm 1000

   # Robust correlations with Bayes factors
   eeg-pipeline behavior compute --subject 0001 \
     --robust-correlation percentage_bend --compute-bayes-factors

   # Visualize
   eeg-pipeline behavior visualize --subject 0001

   # List available stages
   eeg-pipeline behavior compute --list-stages
