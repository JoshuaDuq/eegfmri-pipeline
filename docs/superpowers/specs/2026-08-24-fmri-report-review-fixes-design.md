# fMRI report review fixes

Date: 2026-08-24

## Scope

Implement the four concrete findings from the individual/cohort report review:

1. never associate an MNI companion map from a different fitted configuration;
2. reject malformed signature-expression derivatives;
3. describe each contrast's actual methods in a multi-contrast subject report; and
4. bound the subject cluster table in HTML while retaining the complete TSV.

The larger cohort-to-participant QC index, preprocessing provenance panel, dependency
build record, signature coverage fields, and extended leave-one-out metrics remain
separate features. They change report data contracts or layout more broadly and should
not be mixed into these correctness fixes.

## Design

### Exact MNI companion resolution

The native and standard-space filenames already carry the same configuration hash.
Companion discovery will require an exact trailing hash match for every selected MNI
artifact. If no MNI z-map exists, the optional companion remains absent. If MNI z-map
candidates exist but none matches the native map's hash, discovery raises a descriptive
error listing the expected hash and candidates. Optional effect and variance maps remain
optional, but an unmatched candidate is never substituted.

This is preferred over adding a second manifest schema in this change. A second manifest
would also be discovered as an independent contrast by the current report loader and
would require coordinated changes to fitting, discovery, grouping, and historical
derivatives. Strict hash resolution removes the unsafe fallback at the existing explicit
naming boundary.

### Strict signature TSV boundary

An absent signature TSV continues to mean signatures were not configured. Once a file is
present, its header, field count, signature name, required numeric values, and optional
numeric values are validated. Blank optional similarity metrics remain `None`; malformed
numbers, missing required columns, truncated rows, empty names, non-integral or negative
voxel counts, and non-finite numbers raise `ValueError` with the file, row, and column.

The report does not repair or reinterpret a corrupt scientific derivative.

### Per-contrast methods

The Methods section will build one applied-settings block per manifest. A one-contrast
report preserves the existing `Applied settings` title. A multi-contrast report names
each block with the contrast and includes its own height threshold, multiple-comparison
statement, sidedness, confound strategy, orientation, and extent filter.

This is preferred over rejecting heterogeneous contrasts because the configuration and
report model deliberately support contrast-specific analyses.

### Bounded subject cluster table

`FmriReportConfig` gains `cluster_table_max_rows`, defaulting to 10 and validated as a
YAML integer in `[1, 50]`, matching the cohort report contract. The full enriched cluster
frame is always written to `clusters.tsv`. The HTML table selects leading primary
clusters by absolute peak statistic and includes their subpeaks until the configured
row budget is reached. Its caption states how many rows are shown and that the complete
table is in the TSV.

The report passes the configured limit into cluster-table construction; direct callers
retain the same default. Peak coordinates used by other panels continue to come from the
complete frame, so limiting the HTML table does not change maps or numbered peak markers.

## Error handling

- Optional absence remains optional only where the data contract defines it as such.
- Ambiguous or inconsistent fitted artifacts raise rather than selecting a fallback.
- Present malformed TSV data raises at the reader boundary.
- Invalid configuration raises during configuration validation.

## Testing

Each behavior is developed test-first:

- companion discovery rejects mismatched hashes and selects exact matches;
- signature parsing rejects missing columns, truncated rows, invalid/non-finite metrics,
  invalid voxel counts, and empty signature names while retaining blank optional metrics;
- multi-contrast Methods output identifies and reports distinct settings per contrast;
- subject report config validates the new limit, HTML is bounded, the omission caption is
  explicit, the TSV remains complete, and peak coordinates remain complete.

Run focused tests after each red/green cycle, then run the complete
`tests/fmri/report` suite, Ruff on modified Python files, and the repository architecture
and maintainability gates required for shared report infrastructure.

## Non-goals

- No changes to statistical fitting, thresholds, maps, or side effects.
- No fallback or backward-compatibility shim for malformed derivatives.
- No new scientific-analysis package.
- No redesign of cohort report layout in this change.
