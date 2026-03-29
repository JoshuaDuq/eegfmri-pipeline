# Plotting Workflow

Use this workflow to generate figures from preprocessing, feature, behavior, machine-learning,
or fMRI derivatives after the underlying computations are already complete.

## Main Command Surface

```bash
eeg-pipeline plotting --help
```

## Typical Run Order

1. Confirm the upstream derivatives exist.
2. Select the plot family and required inputs.
3. Run the plotting command with explicit subjects, groups, or feature filters.
4. Review output files in the derivatives tree.

## Outputs

- static figures and summaries ready for reports or QC review

For command options and config entrypoints, start with the [CLI reference](../reference/cli/index.md).
