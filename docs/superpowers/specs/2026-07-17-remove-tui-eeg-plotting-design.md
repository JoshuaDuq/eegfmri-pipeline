# Remove TUI-Listed EEG Plotting

## Objective

Remove the dedicated EEG plotting utility exposed by the TUI, together with the Python
CLI and implementation code that exists exclusively to support its listed plots. Preserve
all plotting outside that surface, especially preprocessing quality-control figures and
component time-frequency representation (TFR) plotting.

## Scope Boundary

The authoritative removal boundary is the dedicated **Plotting** utility selected from the
TUI main menu and backed by `eeg_pipeline/plotting/plot_catalog.json`. In addition, all
behavior plotting owned by `eeg_pipeline` and the `behavior visualize` command path are
removed. Plot-related controls embedded in other pipelines remain out of scope.

The following are explicitly preserved:

- preprocessing plots, including native EEG-fMRI correction and scanner-harmonic QC;
- component-TFR computation and plotting;
- machine-learning plots and fMRI-analysis plots exposed by their own pipelines;
- feature visualization reached through `features visualize`;
- every study-specific figure and plotting module under `studies/`;
- shared plotting helpers still imported by any preserved plotting path.

No fallback command, compatibility alias, deprecated entry point, or empty plotting screen
will remain. Invoking the removed `plotting` subcommand must fail through the CLI parser as
an unknown command.

## Removal Design

### TUI surface

Remove the Plotting utility from pipeline identifiers, main-menu navigation, wizard step
registration, model state, rendering, input handlers, command construction, plotter
discovery, messages, and execution-path inference. Remove tests and fixtures whose sole
purpose is the removed utility. Keep generic TUI mechanisms when another pipeline uses
them.

### Python CLI surface

Remove registration and exports for `eeg-pipeline plotting`. Remove its parser,
orchestrator, runner helpers, selection/configuration helpers, catalog loader, and plot
catalog. Remove `info plotters` only if dependency tracing confirms it has no consumer after
the TUI removal; otherwise remove the obsolete consumer first and then the unused mode.
Remove the `behavior visualize` parser surface and orchestration while retaining behavior
computation and analysis.

### Plot implementations

Build an inventory from every catalog entry to its registered plotter and concrete module.
Delete a plotting function, module, registry, orchestration layer, configuration section,
or dependency only when repository-wide reference analysis shows that it serves no
preserved path. If a module mixes removed and preserved behavior, retain the module and
remove only the exclusively catalog-driven functions. Imports and public exports must be
cleaned at the same time so missing references surface immediately.

`eeg_pipeline/plotting/` is not deleted wholesale because it contains preprocessing,
component-TFR, scanner-harmonic, and potentially other non-TUI plotting paths.
The `eeg_pipeline/plotting/behavioral/` implementation tree is deleted because the user
explicitly removed EEG-pipeline behavior plots. No file under `studies/` is deleted or
rewritten as part of that removal.

### Configuration and documentation

Remove configuration keys and documentation that describe only the removed plotting
utility or its catalog entries. Preserve global-looking plotting configuration when a
retained caller still reads it. Update CLI/TUI help and repository documentation so they do
not advertise the removed utility.

## Dependency-Trace Rule

For each candidate deletion:

1. Trace from the TUI Plotting utility or catalog entry to the Python entry point.
2. Trace the entry point to plot registries, orchestration, concrete functions, helpers,
   configuration, and tests.
3. Search the full repository for non-test and test consumers.
4. Delete only nodes with no path from a preserved feature.
5. Re-run repository searches after deletion to detect stale imports and identifiers.

This rule is the safeguard for the user's protected preprocessing and component-TFR work.

## Testing Strategy

Use test-driven removal. First add or change structural tests so they fail while the
Plotting utility and CLI command still exist. The tests must establish that:

- the TUI main menu and pipeline registry no longer contain Plotting;
- TUI command construction and discovery no longer support the plotting pipeline;
- the Python top-level parser no longer registers `plotting`;
- the behavior parser no longer accepts `visualize` and its plotting modules are absent;
- the removed catalog and exclusive entry-point modules are absent;
- representative preprocessing plotting, scanner-harmonic plotting, and component-TFR
  plotting modules remain importable or structurally present as appropriate to the branch.

Then remove production code until those tests pass. Delete obsolete behavior tests for
implementation that no longer exists; do not rewrite them to assert meaningless internals.

Verification includes:

- all Go tests under `eeg_pipeline/cli/tui`;
- focused Python CLI, structure, preprocessing-plot, scanner-harmonic, and component-TFR
  tests available on the branch;
- Ruff on changed Python files;
- `make verify-structure` and `make verify-architecture`;
- repository-wide searches for removed pipeline identifiers, catalog IDs, command imports,
  and stale documentation;
- a final diff audit confirming protected plot paths were not deleted.
- a `studies/` diff audit confirming no study-owned figure was changed or deleted.

## Success Criteria

- No dedicated Plotting entry or workflow is visible or executable in the TUI.
- `eeg-pipeline plotting` is not a registered CLI command.
- All implementations and support code used only by the TUI-listed plots are removed.
- EEG-pipeline behavior plots and the `behavior visualize` mode are removed.
- Preprocessing, component-TFR, and non-TUI plot paths remain intact.
- Study-owned figures remain intact.
- No compatibility or fallback behavior masks stale calls.
- Relevant tests and repository validation gates pass.
