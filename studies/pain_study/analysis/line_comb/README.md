# Line comb engine

**Driven by:** [`../../scripts/line_comb/`](../../scripts/line_comb/) ·
**Tests:** [`tests/analysis/line_comb/`](../../../../tests/analysis/line_comb/) ·
**Findings:** [`docs/scanner_harmonic_diagnosis.md`](../../../../docs/scanner_harmonic_diagnosis.md)

## Why this folder exists

The spectral machinery for identifying and removing a narrowband comb, kept separate from
the scripts that decide which recordings to point it at. Everything here works on arrays,
so the whole engine is exercised on synthetic signals whose answer is known in advance —
which is the only reason a claim like "the removal takes out the comb and leaves the probes
standing" can be made at all.

## The files

| File | What it contributes |
|---|---|
| `diagnosis.py` | The measurement primitives: periodograms, prominence against local background, and `refine_peak_frequency`, which is what resolves a line to the 5–6 mHz that separates a comb member from a coincidence. |
| `cohort.py` | Lifts a single recording's measurement to the cohort — pooling across participants and sessions, which is what established the fundamental as 1.199998 Hz ± 61 µHz across 5 months. |
| `removal.py` | The removal: fitting the comb fundamental from well-determined harmonics, then subtracting each line. Holds the nominal constants (`NOMINAL_FUNDAMENTAL_HZ`, `COMB_HARMONIC_RANGE`, `REMOVAL_HARMONIC_RANGE`, `ISOLATED_NOMINAL_HZ`) that the workflow config overrides. |

## Two things worth knowing before changing anything

**The fit range and the removal range are different on purpose.** The fundamental is fitted
from harmonics 24–79, which are well determined. Removal reaches down to harmonic 22,
because the diagnosis found comb membership at harmonics 22 and 23 within 5 mHz of their
predicted positions. Harmonic 11 is deliberately left alone: 30 mHz off, present in only 2
of 15 participants, and sitting where alpha meets beta.

**Notch width, not multitaper bandwidth, decides how much spectrum is lost.** `spectrum_fit`
subtracts a sinusoid at every bin inside the notch window, so that window is the size of
the hole. It scales as `freq / ratio` because a mains-locked comb wanders in proportion to
harmonic number. MNE's default of `freq/200` would empty a quarter of 28–95 Hz; `freq/450`
is the narrowest setting measured to push every line below its local background, at 12% of
the band. Going narrower leaves the high harmonics standing.
