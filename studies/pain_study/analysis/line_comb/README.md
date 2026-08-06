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
| `removal.py` | The removal: fitting the comb fundamental from well-determined harmonics, automatically detecting replicated isolated lines, then subtracting each validated target. Holds the nominal comb constants (`NOMINAL_FUNDAMENTAL_HZ`, `COMB_HARMONIC_RANGE`, `REMOVAL_HARMONIC_RANGE`) that the workflow config overrides. |

## Two things worth knowing before changing anything

**The fit range and the removal range are different on purpose.** The fundamental is fitted
from harmonics 24–79, which are well determined. Removal reaches down to harmonic 22,
because the diagnosis found comb membership at harmonics 22 and 23 within 5 mHz of their
predicted positions. Harmonic 11 is deliberately left alone: 30 mHz off, present in only 2
of 15 participants, and sitting where alpha meets beta.

**Automatic line evidence is scoped to where it was observed.** Sources recurring across a
session are shared only after independent run or block support. A strong isolated source
confined to one recording must recur in three non-overlapping 54-second or -5/+15 second
thermal-study intervals. A distinct narrow, channel-median source beside a validated comb
target may be supported in either interval type. It receives its own narrow target in every
continuous overlap-add window contributing to the evidenced samples, but an exact thermal
interval receives an isolated or adjacent target only from its own spectrum. Two summits in the same
spectrum are kept as distinct source tracks even inside the nominal line-claim distance, and
a narrow target is deduplicated only when an exact fitted target width already covers it.
Static frequency seeds are not accepted because they cannot follow participant- and
window-specific drift.
Evidence-qualified sources are never truncated by a numeric count cap. Their count is
reported for provenance; preservation and removed-band gates bound the transform itself.

**Exact analysis samples are anchored, not inherited from neighboring data.** After the
continuous overlap-add pass, each -5/+15 second thermal interval is re-estimated and cleaned
directly. Matched aggregate and channel-local power searches plus a Bonferroni-corrected
Thomson multitaper F-test add residual targets only inside authorized artifact regions.
Each family is searched once; repeating it to force zero detections would defeat its error
control. Exact samples are fully exact, while a correction taper lies outside the interval
to prevent a seam. Background floors remain frozen from the uncleaned data.

**Continuous residuals are measured after synthesis.** Overlap-add can expose a line that
is absent from every isolated segment output, so refinement reads the reconstructed
continuous transform. Residual support is routed across every overlapping synthesis window
and removed with a sliding, sub-bin sinusoid regression at half the main fit duration. The
exact thermal transforms remain isolated from this outside-window evidence.

**BIDS channel types define EEG.** The workflow reads through MNE-BIDS with mismatches set
to raise. ECG/EOG are preserved as auxiliary channels but excluded from EEG detection,
filtering, and validation.

**Notch width, not multitaper bandwidth, decides how much spectrum is lost.** `spectrum_fit`
subtracts a sinusoid at every bin inside the notch window, so that window is the size of
the hole. It scales as `freq / ratio` because a mains-locked comb wanders in proportion to
harmonic number. MNE's default of `freq/200` would empty a quarter of 28–95 Hz; `freq/450`
is the narrowest calibrated base setting that suppresses the lines without excessive band
cost. Per-window uncertainty, observed same-window support, and statistically detected
residuals are added explicitly. The only cost gate is the worst total 28–95 Hz fraction actually removed by a
continuous or exact channel-level transform; the base-width fraction remains descriptive.
