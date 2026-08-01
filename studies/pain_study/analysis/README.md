# Analysis engines for the artifact workflows

## Why this folder exists

Each folder here is the *measurement and algorithm* half of a workflow whose *runnable*
half lives in [`../scripts/`](../scripts/). The split is deliberate:

- code here takes arrays and returns numbers, and can be tested without a filesystem, a
  drive, or a BIDS tree;
- code in `scripts/` decides which files to read, in what order, and where results go.

That is why the test suites mirror the same shape — the engine suites run in seconds on
synthetic signals with known answers, which is what makes it possible to state that a
removal takes out the comb and leaves the probes standing.

| Engine | Driven by | Test suite |
|---|---|---|
| [`line_comb/`](line_comb/) | [`../scripts/line_comb/`](../scripts/line_comb/) | [`tests/analysis/line_comb/`](../../../tests/analysis/line_comb/) |
| [`bcg/`](bcg/) | [`../scripts/cardiac_gaps/`](../scripts/cardiac_gaps/) | [`tests/analysis/bcg/`](../../../tests/analysis/bcg/) |

## Why these are paradigm-specific

Both engines were built for this dataset and carry its findings. `line_comb` encodes the
frequencies of one scanner room and means nothing at another site. `bcg` exists to repair
one vendor's failure mode on one set of exports. Neither belongs in `eeg_pipeline/`, which
is meant to hold what is true of any EEG–fMRI study.
