# PyPREP RANSAC Warning Boundary Design

## Objective

Prevent verified spurious floating-point `matmul` warnings from flooding PyPREP logs while
preserving a strict failure whenever RANSAC actually produces non-finite scientific results.

## Evidence

The corrected-v3 EEG has finite, non-duplicated electrode coordinates with plausible head
radii and spacing. Fifty deterministic RANSAC interpolation matrices were finite, as were a
representative raw EEG window and its predicted signals. The current NumPy/SciPy stack still
emits `divide by zero`, `overflow`, and `invalid value` RuntimeWarnings during the matrix
operations even though their returned arrays are finite.

## Design

Add one focused helper around `NoisyChannels.find_bad_by_ransac()`. Within that call only,
suppress RuntimeWarnings whose exact messages are one of the three verified `matmul` messages
and whose modules are SciPy linear algebra, MNE interpolation, or PyPREP RANSAC. Do not alter
global warning filters and do not suppress other warning categories or messages.

After RANSAC returns, read its correlation matrix from PyPREP's result metadata and require it
to be a non-empty finite two-dimensional array. Missing, malformed, empty, or non-finite
correlations raise an explicit error. This postcondition turns a genuinely invalid numerical
result into a pipeline failure instead of hidden warning noise.

## Pipeline Restart

Restart full task preprocessing with all ten logical cores and
`pyprep.bad_channel_sync_policy=subject_union`. The previous run failed because per-run bad
channel sets cannot be concatenated for shared ICA and epoch processing; this is independent
of the warning suppression.

## Validation

Tests verify that the exact known warnings are absent, unrelated RuntimeWarnings remain
visible, finite RANSAC correlations pass, and non-finite correlations fail explicitly.
