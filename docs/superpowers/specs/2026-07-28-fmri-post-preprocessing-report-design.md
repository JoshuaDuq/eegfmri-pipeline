# fMRI post-preprocessing report and plotting redesign

Date: 2026-07-28

## Problem

Every figure the fMRI analysis pipeline produces after fMRIPrep is generated inside
`fmri_pipeline/analysis/reporting.py`, a 1605-line module that mixes figure drawing,
HTML string assembly, signature tables, and orchestration. Three consequences:

1. **Figures carry scientific errors that change what a reader concludes.** Glass brains
   render `|z|` rather than `z`; thresholded overlays are drawn with a colour limit taken
   from the unthresholded map and routinely saturate; slice mosaics have their coordinates
   and left/right markers suppressed; the tSNR montage slices the raw voxel array and
   labels the panels by anatomy without consulting the affine.
2. **No figure can be tested without rendering a report.** Panel code is inlined in the
   orchestration path, so there is nowhere to assert that a panel got the right arguments.
3. **There is no style layer.** `fmri_pipeline` never calls anything equivalent to
   `eeg_pipeline.infra.matplotlib.setup_matplotlib`, so figures inherit whatever global
   rcParams are live in the process, and colours are hardcoded matplotlib defaults while a
   documented Okabe-Ito convention exists in `eeg_pipeline/preprocessing/report/style.py`.

Separately, the pipeline has no per-subject report of the kind fMRIPrep and the EEG
pipeline both provide. Today's `report.html` is scoped to a single contrast, so the QC
panels are recomputed and duplicated for every contrast of the same subject.

## Scope

This spec covers the **per-subject** post-preprocessing report and the plotting layer
underneath it. A cohort/group report is a follow-on that reuses the same style and figure
modules; it is named here only where it constrains an interface, and is not built in this
pass.

## Decisions taken

**Report scope is per subject.** One `sub-XXXX_task-YYYY_report.html` covers the shared QC
once and then every first-level contrast as a section. This replaces the per-contrast
`report.html`.

**QC is analysis-relevant only.** The report does not restate fMRIPrep's preprocessing QC.
Every QC panel is computed on what actually entered the GLM — the included runs, after
confound regression and smoothing — and is labelled *as-modelled* rather than
*as-preprocessed*. Registration, susceptibility, and surface QC stay fMRIPrep's job.

**The report container stays in-house.** `build_fmri_report_html` is restructured rather
than replaced by `mne.Report` or nilearn's `make_glm_report`. Both alternatives surrender
control over figure design, which is the substance of this work; `mne.Report` would also
couple `fmri_pipeline` to MNE purely as a templating engine and inherit the SVG-in-figure-
list defect documented in `eeg_pipeline/preprocessing/report/style.py`. Visual kinship with
the EEG report comes from sharing the style conventions, not the container.

**No verdicts.** The report states measured values and the thresholds actually applied. It
does not convert a measurement into a pass/fail badge against a cutoff the pipeline
invented.

## Architecture

```
fmri_pipeline/analysis/report/
  style.py       # rcParams context, palette, colormap policy, robust limits, format policy
  figures/
    stat_maps.py     # thresholded mosaic, glass brain, effect, standard error
    distributions.py # z histogram, tSNR histogram
    carpet.py        # tissue-ordered carpet with aligned FD/DVARS
    volumes.py       # tSNR volume rendering
    design.py        # design matrix, contrast strip, VIF, regressor correlation
    signatures.py    # signature expression dot plot
  html.py        # Document / Section / Figure / Table primitives, TOC, sticky nav
  subject.py     # assembles the per-subject report
```

The load-bearing rule: **every function in `figures/` accepts arrays or nibabel images and
returns a `matplotlib.figure.Figure`. It knows nothing about HTML, output paths, or
`FmriPlottingConfig`.** Saving, formats, and configuration belong to `subject.py`. This is
what makes panels testable in isolation.

`reporting.py` retains `run_fmri_plotting_and_report` as the public entry point and
delegates, so callers under `fmri_pipeline/pipelines/` do not move. Its figure-drawing code
is deleted, not ported.

A later `report/group.py` reuses `style.py` and `figures/` unchanged.

### Style layer

`style.py` supplies three things.

**A scoped rcParams context.** Figures render inside `with plt.rc_context(FMRI_RC):`
rather than mutating global state at import. `eeg_pipeline.infra.matplotlib.setup_matplotlib`
calls `sns.set_theme`, which is a global mutation; running both pipelines in one process
currently lets one silently restyle the other. The Agg backend is still forced once, since
that must be global to be effective.

**A stated colormap policy.**

| Data | Colormap | Rule |
|---|---|---|
| Signed maps (z, effect size) | `RdBu_r` | symmetric limits always, light neutral at zero |
| Unsigned magnitude (tSNR, standard error) | `cividis` | single hue, perceptually uniform, CVD-safe; limits from the data, never anchored at zero when the data does not live near zero |
| Categorical series | Okabe-Ito | imported from `eeg_pipeline/preprocessing/report/style.py`, not redefined |

This resolves an existing inconsistency: z maps take nilearn's default `RdBu_r` while
effect maps are drawn with `cold_hot` (`reporting.py:1152`) — the same kind of data in two
colormaps, and `cold_hot`'s dark midpoint is wrong against the `black_bg=False` background
these panels use.

**Format as a property of the figure.** Dense image layers (brain mosaics, carpets) are
raster; line and bar figures (histograms, motion traces, VIF) are SVG. `cfg.formats` keeps
its meaning for files written to disk, but the report embeds exactly one rendition per
figure. This is the fix for figures currently appearing twice in the report.

`style.py` also exports two limit helpers. `robust_symmetric_limit` mirrors the EEG helper:
a high percentile of the pooled absolute values, so a few extreme voxels cannot flatten the
rest. `suprathreshold_limit` serves thresholded panels — it takes the same percentile over
*only* the voxels surviving the threshold, and returns at least `threshold * 1.5` so that a
panel always has usable dynamic range even when the map is pure noise. Both are stated on
the figure, so clipping is declared rather than hidden.

## Figure inventory

### Removed

| Figure | Reason |
|---|---|
| `glass_unthresholded` | An unthresholded glass brain is a saturated blob at any threshold setting. |
| `tsnr_map.png` (3-panel montage) | Slices the raw voxel array and labels panels by anatomy without consulting the affine. Rebuilt, not repaired. |
| `motion_qc.png` as a standalone figure | FD/DVARS on separate axes cannot be related to the carpet. Absorbed into the carpet figure on a shared time axis. |
| `scripts/analyze_fmri_motion.py` figures | Cohort-level; moves to the group report. The viridis-anchored-at-zero heatmap compresses all real variation into indistinguishable mid-band greens. |

### Demoted to a per-contrast collapsed "diagnostics" disclosure

Unthresholded mosaic, effect-size mosaic, standard-error mosaic, z histogram. The
unthresholded map is the honest counterpart to the thresholded one and must remain
available; the standard-error map is how a reader distinguishes a true null from a
dropout-driven false negative. Neither belongs at top level competing with the result.

### Rebuilt

- **Thresholded mosaic** — `annotate=True` restored (slice coordinates and L/R markers);
  colour limits taken from suprathreshold voxels; colorbar labelled `z`; threshold and
  cluster criteria in the caption rather than the title.
- **Thresholded glass brain** — `plot_abs=False`.
- **z histogram** — log y-axis, N(0,1) null overlaid, threshold lines, masked voxels only.
  The deviation from the null is the diagnostic and a linear y-axis hides it.
- **Carpet** — GM/WM/CSF row blocks, FD and DVARS above on a shared axis in seconds, run
  boundaries labelled with real run identifiers. Tissue labels are read from the fMRIPrep
  anatomical derivatives in the BOLD's own space, preferring
  `*_label-{GM,WM,CSF}_probseg.nii.gz` (assigning each voxel to its highest-probability
  class) and falling back to `*_desc-aseg_dseg.nii.gz`. Where neither is discoverable the
  carpet falls back to mask order and states that on the figure rather than implying
  structure it does not have. Discovery is best-effort by design: the fMRIPrep root is
  configured per study and may not be mounted.
- **tSNR** — rendered through the affine via nilearn, plus a histogram with the median
  annotated, as one figure.
- **Design matrix** — legible regressor labels, task/confound/drift column groups visually
  separated, diverging and zero-centred.

### Added

- **Regressor correlation matrix and VIF bars.** `_vif_from_design` already runs at
  `reporting.py:1492` and its output reaches only a table cell. This is the figure that
  shows whether a contrast is estimable.
- **Contrast vector strip** beneath the design matrix, so a contrast is legible rather than
  merely named.
- **Numbered cluster peaks** on the glass brain, keyed to the existing cluster table.
- **Signature expression dot plot.** Signature expression is currently a five-column table;
  the comparison across signatures is the point and a table does not show it.

## Report structure

```
Header      subject, task, spaces, runs included/excluded with reasons,
            TR, volumes, smoothing kernel, confound model
TOC         sticky
1 Model     design matrix + contrast strip; VIF and regressor correlation
2 QC        carpet with aligned motion; tSNR          (all labelled as-modelled)
3 Results   one subsection per contrast x space:
              thresholded mosaic, glass brain with numbered peaks, cluster table
              collapsed diagnostics: unthresholded, effect, standard error, z histogram
4 Signatures  expression dot plot and table
5 Methods   provenance payload (existing, retained)
```

## Correctness fixes carried by this work

These are defects in current behaviour, listed so the implementation plan can pin a
regression test to each.

| Location | Defect |
|---|---|
| `reporting.py:1025`, `:1036`, `:1157` | `plot_glass_brain` without `plot_abs=False`; nilearn 0.14.0 defaults it to `True`, so activation and deactivation render identically and the glass panel contradicts the signed mosaic beside it. |
| `reporting.py:981`, `:1010` | Thresholded panels reuse `p99(\|z\|)` of the unthresholded map as `vmax`. For z ~ N(0,1) that is ≈2.58 against a default threshold of 2.30, leaving a 0.28-wide colour band. Nothing guards `vmax > threshold`. |
| `reporting.py:998`, `:1014`, `:1153`, `:1194` | `annotate=False` strips slice coordinates and L/R markers from every mosaic. |
| `reporting.py:383`–`:387` | tSNR montage slices the voxel array directly and labels panels Sagittal/Coronal/Axial; correct only for RAS-canonical data, silently wrong (including L/R flips) otherwise. |
| `reporting.py:1395` | `fd.fillna(0)` fabricates a zero for the first frame of each run, where FD is undefined by construction. Draws a dip to "no motion" at every run boundary. |
| `reporting.py:1391`, `:1399` | Y-axis hardcoded `"DVARS"` while the code falls back to `std_dvars`, which is on a different scale. |
| `reporting.py:962`–`:966` | `_add_image` appends one `ReportImage` per format, so `formats=("png","svg")` renders every figure twice in the report. Same pattern in the histogram and QC loops. |
| `reporting.py:1019`, `:1047`, `:1092`, `:1125`, `:1169`, `:1198` | `logger.warning(...)` followed by `raise`, so one failed panel aborts the contrast — while the QC blocks below catch and continue. Two policies in one module. |
| `reporting.py:1083` | `ax.legend()` called unconditionally though a label is added only when a threshold exists; emits a warning and an empty legend under `threshold_mode=none`. |
| carpet/tSNR/histogram paths | `plt.close(fig)` sits after `savefig` inside the `try`, so any failure leaks the figure. Unbounded growth across a cohort. |
| `reporting.py:219` | Carpet voxels are ordered by raw mask index, so the banding that makes a carpet diagnostic is not present. |
| `second_level.py:819`, `contrast_builder.py:975` | Design-matrix dpi differs (200 vs 150) and neither matches the 300 used elsewhere; `dpi=300` is also passed on SVG saves, where it has no meaning. |

## Error handling

One policy throughout: a panel that fails is logged and rendered as a placeholder card
naming the exception, and the report always builds. This matches the established principle
that a measurement stage resolving nothing is a measurement, not a fault, and must not kill
a run. The six `raise` statements above are removed. Figures are closed in `finally`.

## Testing

Panels are tested directly, against synthetic nibabel images, without rendering a report —
which the current structure makes impossible. Regression tests are pinned to the table
above:

- `plot_abs=False` reaches nilearn for every glass-brain call.
- A thresholded panel's `vmax` exceeds its threshold, including when the source map is pure
  noise.
- FD `NaN` survives into the plotted array rather than becoming zero.
- The DVARS axis label matches the column actually plotted.
- `formats=("png","svg")` yields one report entry per figure and two files on disk.
- The report builds when every panel raises.
- The carpet falls back to mask order, and says so, when no segmentation is discoverable.
- The tSNR figure's orientation is correct for a non-RAS affine.

Per the project convention, verification runs targeted subsets rather than the full suite.

## Out of scope

- The cohort/group report (follow-on pass, reusing `style.py` and `figures/`).
- fMRIPrep-domain QC: registration, susceptibility distortion, surface reconstruction.
- Any change to GLM estimation, contrast construction, or confound selection.
