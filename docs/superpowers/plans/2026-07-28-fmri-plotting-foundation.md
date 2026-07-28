# fMRI Plotting Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested plotting layer for the fMRI analysis pipeline and replace every existing post-fMRIPrep figure with a corrected one.

**Architecture:** A new `fmri_pipeline/analysis/report/` package. `style.py` holds render conventions; `assets.py` is the single discovery path for background, mask, and tissue images; `figures/` holds one module per panel family. Every function in `figures/` takes arrays or nibabel images and returns a `matplotlib.figure.Figure` — it never touches HTML, output paths, or config, which is what makes panels testable without rendering a report. The existing `reporting.py` keeps its public entry point and delegates; its inline figure code is deleted.

**Tech Stack:** Python, matplotlib, nilearn 0.14.0, nibabel, numpy, pandas, scipy, pytest.

This is plan 1 of 3 from `docs/superpowers/specs/2026-07-28-fmri-post-preprocessing-report-design.md`. Plan 2 covers the report restructure and manifest-based decoupling; plan 3 covers the resting-state profile. This plan ships the scientific correctness fixes into the report path that exists today.

## Global Constraints

- Signed maps (z, effect size) use `RdBu_r` with symmetric limits. Unsigned magnitude (tSNR, standard error) uses `cividis`. Categorical series use Okabe-Ito from `eeg_pipeline/preprocessing/report/style.py`. No rainbow colormaps, no `cold_hot`.
- Figures render inside `plt.rc_context`, never by mutating global rcParams. The Agg backend is the one exception and is set once at package import.
- Every function in `figures/` returns a `matplotlib.figure.Figure` and takes no `Path` and no `FmriPlottingConfig`.
- A panel that cannot be drawn raises a normal exception; it never calls `sys.exit`, and it never silently returns a blank figure. Callers decide the failure policy.
- Colour limits are stated on the figure whenever clipping is possible, so clipping is declared rather than hidden.
- The report presents measurements and the thresholds actually applied. No pass/fail badges, no cutoffs the pipeline invented.
- Tests use plain pytest with `nibabel.Nifti1Image` fixtures, matching `tests/fmri/`. Run targeted subsets — never the full suite, which takes ~9 minutes.

---

## File Structure

| File | Responsibility |
|---|---|
| `fmri_pipeline/analysis/report/__init__.py` | Sets the Agg backend once. No other logic. |
| `fmri_pipeline/analysis/report/style.py` | rcParams context, colormap constants, colour-limit helpers, format policy. |
| `fmri_pipeline/analysis/report/assets.py` | Discovery of background, mask, and tissue images. Returns paths only. |
| `fmri_pipeline/analysis/report/figures/stat_maps.py` | Slice mosaic and glass brain. |
| `fmri_pipeline/analysis/report/figures/distributions.py` | z histogram, magnitude histogram. |
| `fmri_pipeline/analysis/report/figures/volumes.py` | tSNR volume rendering through the affine. |
| `fmri_pipeline/analysis/report/figures/carpet.py` | Tissue-ordered carpet with aligned motion traces. |
| `fmri_pipeline/analysis/report/figures/design.py` | Design matrix, contrast strip, VIF, regressor correlation. |
| `fmri_pipeline/analysis/reporting.py` | Modified: figure code deleted, delegates to `figures/`, one failure policy. |

---

### Task 1: Style layer

**Files:**
- Create: `fmri_pipeline/analysis/report/__init__.py`
- Create: `fmri_pipeline/analysis/report/style.py`
- Create: `fmri_pipeline/analysis/report/figures/__init__.py`
- Test: `tests/fmri/report/test_style.py`

**Interfaces:**
- Consumes: `OKABE_ITO` from `eeg_pipeline.preprocessing.report.style`.
- Produces:
  - `FMRI_RC: dict[str, Any]`
  - `plot_context() -> ContextManager` — `plt.rc_context(FMRI_RC)`
  - `SIGNED_CMAP: str = "RdBu_r"`, `MAGNITUDE_CMAP: str = "cividis"`
  - `robust_symmetric_limit(*values: np.ndarray, percentile: float = 98.0) -> float`
  - `suprathreshold_limit(values: np.ndarray, *, threshold: float, percentile: float = 98.0) -> float`
  - `figure_format(*, dense: bool) -> str` returning `"png"` or `"svg"`

- [ ] **Step 1: Create the package and write the failing test**

Create `tests/fmri/report/__init__.py` (empty) and `tests/fmri/report/test_style.py`:

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from fmri_pipeline.analysis.report import style


def test_plot_context_does_not_leak_into_global_rcparams() -> None:
    before = plt.rcParams["axes.spines.top"]
    with style.plot_context():
        assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.top"] == before


def test_robust_symmetric_limit_is_not_dominated_by_a_single_outlier() -> None:
    values = np.concatenate([np.full(999, 1.0), np.array([1000.0])])
    assert style.robust_symmetric_limit(values) == pytest.approx(1.0, abs=0.01)


def test_robust_symmetric_limit_rejects_an_all_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        style.robust_symmetric_limit(np.array([np.nan, np.inf]))


def test_suprathreshold_limit_uses_only_surviving_voxels() -> None:
    # 990 sub-threshold voxels must not drag the limit down toward the threshold.
    values = np.concatenate([np.full(990, 0.1), np.full(10, 8.0)])
    assert style.suprathreshold_limit(values, threshold=2.3) == pytest.approx(8.0, abs=0.01)


def test_suprathreshold_limit_floors_above_the_threshold_for_a_noise_map() -> None:
    # p99(|z|) of standard normal noise is ~2.58, barely above a 2.3 threshold.
    rng = np.random.default_rng(0)
    limit = style.suprathreshold_limit(rng.standard_normal(100_000), threshold=2.3)
    assert limit >= 2.3 * 1.5


def test_suprathreshold_limit_floors_when_nothing_survives() -> None:
    assert style.suprathreshold_limit(np.zeros(100), threshold=2.3) == pytest.approx(3.45)


def test_dense_figures_are_raster_and_line_figures_are_vector() -> None:
    assert style.figure_format(dense=True) == "png"
    assert style.figure_format(dense=False) == "svg"


def test_signed_and_magnitude_colormaps_are_not_rainbows() -> None:
    assert style.SIGNED_CMAP == "RdBu_r"
    assert style.MAGNITUDE_CMAP == "cividis"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_style.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fmri_pipeline.analysis.report'`

- [ ] **Step 3: Create the package init**

`fmri_pipeline/analysis/report/__init__.py`:

```python
"""Rendering layer for fMRI post-preprocessing reports.

Importing this package fixes the Matplotlib backend. That is global on purpose:
a backend cannot be scoped to a context manager, and pipeline processes have no
display. Everything else in this package is scoped -- see
:func:`fmri_pipeline.analysis.report.style.plot_context`.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=False)
```

Create `fmri_pipeline/analysis/report/figures/__init__.py` as an empty file.

- [ ] **Step 4: Implement the style module**

`fmri_pipeline/analysis/report/style.py`:

```python
"""Render conventions shared by every fMRI report figure.

Colour policy
-------------
Signed quantities -- z statistics, effect sizes -- are drawn with a diverging
colormap on a symmetric scale, so the neutral colour always marks zero. Unsigned
magnitudes -- tSNR, standard error -- are drawn with a single-hue perceptually
uniform ramp. Rainbow and multi-hue sequential colormaps are not used: their
non-monotonic lightness introduces boundaries that are not in the data.

``cold_hot`` is deliberately absent even though nilearn offers it. Its midpoint is
dark, which is correct only against ``black_bg=True``; these figures are drawn on a
white background, where the midpoint must be light for zero to read as neutral.

Scoping
-------
Style is applied through :func:`plot_context` rather than by mutating
``plt.rcParams`` at import. The EEG pipeline's ``setup_matplotlib`` mutates global
state via ``seaborn.set_theme``; when both pipelines run in one process, whichever
ran last silently restyles the other. A context manager cannot do that.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eeg_pipeline.preprocessing.report.style import OKABE_ITO

#: Diverging colormap for signed maps. Light neutral at zero on a white background.
SIGNED_CMAP = "RdBu_r"

#: Single-hue perceptually uniform ramp for unsigned magnitude.
MAGNITUDE_CMAP = "cividis"

#: Neutral colour for guides, thresholds, and reference curves.
GUIDE_COLOR = "0.35"

#: Percentile defining a robust colour limit a few extreme voxels cannot dominate.
COLOR_LIMIT_PERCENTILE = 98.0

#: Smallest ratio of colour limit to threshold that leaves a panel usable range.
#:
#: Without a floor, a thresholded panel drawn from noise gets a limit barely above
#: its own threshold -- p99(|z|) of standard normal noise is about 2.58 against a
#: 2.3 threshold -- and every surviving voxel saturates to one colour.
SUPRATHRESHOLD_HEADROOM = 1.5

FMRI_RC: dict[str, Any] = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.color": "0.85",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.frameon": False,
    "image.interpolation": "nearest",
}


def plot_context() -> AbstractContextManager:
    """Return a context in which this package's render defaults apply."""
    return plt.rc_context(FMRI_RC)


def _finite(*values: np.ndarray) -> np.ndarray:
    pooled = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values])
    finite = pooled[np.isfinite(pooled)]
    if finite.size == 0:
        raise ValueError("Colour limits require at least one finite value.")
    return finite


def robust_symmetric_limit(
    *values: np.ndarray,
    percentile: float = COLOR_LIMIT_PERCENTILE,
) -> float:
    """Return a symmetric colour limit a few extreme samples cannot dominate.

    Taking the limit from the single largest absolute value lets one extreme voxel
    flatten the rest of the map to the neutral colour. The limit therefore comes
    from a high percentile of the pooled absolute values, and callers state it on
    the figure so that clipped voxels are declared rather than hidden.
    """
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"Percentile must lie in (0, 100], got {percentile!r}.")
    return float(np.percentile(np.abs(_finite(*values)), percentile))


def suprathreshold_limit(
    values: np.ndarray,
    *,
    threshold: float,
    percentile: float = COLOR_LIMIT_PERCENTILE,
) -> float:
    """Return a colour limit for a thresholded panel.

    Computed over only the voxels that survive ``threshold``, because a limit taken
    from the whole map is dominated by the sub-threshold voxels the panel does not
    show. Floored at ``threshold * SUPRATHRESHOLD_HEADROOM`` so the panel keeps
    usable dynamic range even when nothing meaningfully exceeds the threshold.
    """
    if threshold <= 0:
        raise ValueError(f"Threshold must be > 0, got {threshold!r}.")
    floor = float(threshold) * SUPRATHRESHOLD_HEADROOM
    finite = _finite(values)
    surviving = np.abs(finite)[np.abs(finite) > float(threshold)]
    if surviving.size == 0:
        return floor
    return max(floor, float(np.percentile(surviving, percentile)))


def figure_format(*, dense: bool) -> str:
    """Return the embedding format for one figure.

    Figures dominated by a dense image layer -- brain mosaics, carpets -- stay
    raster: wrapping the same pixels in base64 inside an SVG costs more bytes while
    only sharpening the axis text. Line and bar figures are vector, so their text
    stays legible at any zoom in a browser whose width the author does not control.
    """
    return "png" if dense else "svg"


__all__ = [
    "FMRI_RC",
    "GUIDE_COLOR",
    "MAGNITUDE_CMAP",
    "OKABE_ITO",
    "SIGNED_CMAP",
    "figure_format",
    "plot_context",
    "robust_symmetric_limit",
    "suprathreshold_limit",
]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_style.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 6: Commit**

```bash
git add fmri_pipeline/analysis/report tests/fmri/report
git commit -m "feat(fmri): add scoped plotting style layer for report figures"
```

---

### Task 2: Asset discovery

**Files:**
- Create: `fmri_pipeline/analysis/report/assets.py`
- Test: `tests/fmri/report/test_assets.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `@dataclass(frozen=True) class PlotAssets` with fields `background: Path | None`, `mask: Path | None`, `probseg: dict[str, Path]`, `dseg: Path | None`
  - `discover_plot_assets(*, deriv_root: Path, subject: str, task: str, space: str) -> PlotAssets`

`space` is `"native"` or `"mni"`. This replaces `FmriAnalysisPipeline._discover_plot_assets` and the two background helpers in `reporting.py`.

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_assets.py`:

```python
from __future__ import annotations

from pathlib import Path

from fmri_pipeline.analysis.report.assets import PlotAssets, discover_plot_assets


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_anatomical_background_is_preferred_over_the_boldref(tmp_path: Path) -> None:
    func = tmp_path / "preprocessed" / "fmri" / "sub-01" / "func"
    anat = tmp_path / "preprocessed" / "fmri" / "sub-01" / "anat"
    _touch(func / "sub-01_task-rest_run-01_space-T1w_desc-preproc_boldref.nii.gz")
    expected = _touch(anat / "sub-01_desc-preproc_T1w.nii.gz")

    assets = discover_plot_assets(
        deriv_root=tmp_path, subject="sub-01", task="rest", space="native"
    )
    assert assets.background == expected


def test_boldref_is_used_when_no_anatomical_is_present(tmp_path: Path) -> None:
    func = tmp_path / "preprocessed" / "fmri" / "sub-01" / "func"
    expected = _touch(func / "sub-01_task-rest_run-01_space-T1w_desc-preproc_boldref.nii.gz")

    assets = discover_plot_assets(
        deriv_root=tmp_path, subject="sub-01", task="rest", space="native"
    )
    assert assets.background == expected


def test_mni_space_selects_the_mni_anatomical(tmp_path: Path) -> None:
    anat = tmp_path / "preprocessed" / "fmri" / "sub-01" / "anat"
    _touch(anat / "sub-01_desc-preproc_T1w.nii.gz")
    expected = _touch(anat / "sub-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz")

    assets = discover_plot_assets(
        deriv_root=tmp_path, subject="sub-01", task="rest", space="mni"
    )
    assert assets.background == expected


def test_probseg_tissue_maps_are_collected_by_class(tmp_path: Path) -> None:
    anat = tmp_path / "preprocessed" / "fmri" / "sub-01" / "anat"
    gm = _touch(anat / "sub-01_label-GM_probseg.nii.gz")
    wm = _touch(anat / "sub-01_label-WM_probseg.nii.gz")
    csf = _touch(anat / "sub-01_label-CSF_probseg.nii.gz")

    assets = discover_plot_assets(
        deriv_root=tmp_path, subject="sub-01", task="rest", space="native"
    )
    assert assets.probseg == {"GM": gm, "WM": wm, "CSF": csf}


def test_dseg_is_found_when_probseg_is_absent(tmp_path: Path) -> None:
    anat = tmp_path / "preprocessed" / "fmri" / "sub-01" / "anat"
    expected = _touch(anat / "sub-01_desc-aseg_dseg.nii.gz")

    assets = discover_plot_assets(
        deriv_root=tmp_path, subject="sub-01", task="rest", space="native"
    )
    assert assets.probseg == {}
    assert assets.dseg == expected


def test_a_missing_derivative_tree_yields_empty_assets_rather_than_raising(
    tmp_path: Path,
) -> None:
    assets = discover_plot_assets(
        deriv_root=tmp_path / "absent", subject="sub-01", task="rest", space="native"
    )
    assert assets == PlotAssets(background=None, mask=None, probseg={}, dseg=None)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_assets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fmri_pipeline.analysis.report.assets'`

- [ ] **Step 3: Implement the module**

`fmri_pipeline/analysis/report/assets.py`:

```python
"""Discovery of the background, mask, and tissue images a report figure needs.

One implementation on purpose. This job was previously done twice with different
answers: ``FmriAnalysisPipeline._discover_plot_assets`` searched a list of
derivative layouts, while ``reporting.py`` separately built a mean-BOLD background
from run metadata and fell back to nilearn's MNI template.

Discovery is best-effort throughout: the fMRIPrep root is configured per study and
may not be mounted. A missing asset yields ``None``, and figures degrade to a
plain background rather than failing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_MNI_SPACE_TOKENS = ("MNI152NLin2009cAsym", "MNI152NLin6Asym")
_TISSUE_CLASSES = ("GM", "WM", "CSF")


@dataclass(frozen=True)
class PlotAssets:
    """Paths a figure may use as context. Every field is optional."""

    background: Optional[Path] = None
    mask: Optional[Path] = None
    probseg: Dict[str, Path] = field(default_factory=dict)
    dseg: Optional[Path] = None


def _subject_dirs(deriv_root: Path, subject: str) -> List[Path]:
    """Return candidate subject directories across the layouts in use."""
    return [
        deriv_root / "preprocessed" / "fmri" / subject,
        deriv_root / "preprocessed" / "fmri" / "fmriprep" / subject,
        deriv_root / "fmriprep" / subject,
    ]


def _first_glob(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.exists():
        return None
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _space_tokens(space: str) -> List[str]:
    return list(_MNI_SPACE_TOKENS) if space == "mni" else ["T1w"]


def _find_background(subject_dir: Path, subject: str, task: str, space: str) -> Optional[Path]:
    """Prefer an anatomical image; fall back to a boldref.

    An anatomical background makes an overlay anatomically interpretable in a way a
    single BOLD reference volume does not, so it wins whenever it exists.
    """
    anat = subject_dir / "anat"
    for token in _space_tokens(space):
        if space == "mni":
            candidate = anat / f"{subject}_space-{token}_desc-preproc_T1w.nii.gz"
        else:
            candidate = anat / f"{subject}_desc-preproc_T1w.nii.gz"
        if candidate.exists():
            return candidate

    func = subject_dir / "func"
    for token in _space_tokens(space):
        for pattern in (
            f"{subject}_task-{task}_*space-{token}_desc-preproc_boldref.nii.gz",
            f"{subject}_task-{task}_*space-{token}_boldref.nii.gz",
        ):
            found = _first_glob(func, pattern)
            if found is not None:
                return found
    return None


def _find_mask(subject_dir: Path, subject: str, task: str, space: str) -> Optional[Path]:
    func = subject_dir / "func"
    for token in _space_tokens(space):
        found = _first_glob(
            func, f"{subject}_task-{task}_*space-{token}_desc-brain_mask.nii.gz"
        )
        if found is not None:
            return found
    return None


def _find_tissue(subject_dir: Path, subject: str) -> tuple[Dict[str, Path], Optional[Path]]:
    """Return per-class probability maps and a discrete segmentation.

    Probability maps are preferred by the carpet, which assigns each voxel to its
    highest-probability class; the discrete segmentation is the fallback.
    """
    anat = subject_dir / "anat"
    probseg: Dict[str, Path] = {}
    for tissue in _TISSUE_CLASSES:
        found = _first_glob(anat, f"{subject}_*label-{tissue}_probseg.nii.gz")
        if found is not None:
            probseg[tissue] = found

    dseg = _first_glob(anat, f"{subject}_*desc-aseg_dseg.nii.gz")
    if dseg is None:
        dseg = _first_glob(anat, f"{subject}_*dseg.nii.gz")
    return probseg, dseg


def discover_plot_assets(
    *,
    deriv_root: Path,
    subject: str,
    task: str,
    space: str,
) -> PlotAssets:
    """Locate plotting context for one subject, task, and space.

    ``space`` is ``"native"`` or ``"mni"``. Absent assets are reported as ``None``
    rather than raised, because a figure without a background is still a figure.
    """
    space = str(space or "").strip().lower()
    background: Optional[Path] = None
    mask: Optional[Path] = None
    probseg: Dict[str, Path] = {}
    dseg: Optional[Path] = None

    for subject_dir in _subject_dirs(Path(deriv_root), subject):
        if not subject_dir.exists():
            continue
        if background is None:
            background = _find_background(subject_dir, subject, task, space)
        if mask is None:
            mask = _find_mask(subject_dir, subject, task, space)
        if not probseg and dseg is None:
            probseg, dseg = _find_tissue(subject_dir, subject)

    if background is None:
        logger.debug("No background image found for %s task-%s space-%s", subject, task, space)
    return PlotAssets(background=background, mask=mask, probseg=probseg, dseg=dseg)


__all__ = ["PlotAssets", "discover_plot_assets"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_assets.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/assets.py tests/fmri/report/test_assets.py
git commit -m "feat(fmri): unify report background, mask, and tissue discovery"
```

---

### Task 3: Stat map figures

**Files:**
- Create: `fmri_pipeline/analysis/report/figures/stat_maps.py`
- Test: `tests/fmri/report/test_stat_maps.py`

**Interfaces:**
- Consumes: `SIGNED_CMAP`, `plot_context`, `suprathreshold_limit`, `robust_symmetric_limit` from Task 1.
- Produces:
  - `stat_map_mosaic(stat_img, *, bg_img=None, threshold=None, vmax=None, title="", cbar_label="z", cmap=SIGNED_CMAP) -> Figure`
  - `glass_brain(stat_img, *, threshold=None, vmax=None, title="", cbar_label="z", peak_coords=None) -> Figure`

`peak_coords` is a sequence of `(x, y, z)` MNI coordinates; when given, each is annotated with its 1-based index so a reader can key the map to the cluster table.

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_stat_maps.py`:

```python
from __future__ import annotations

from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.report.figures import stat_maps


def _noise_img(seed: int = 0) -> nib.Nifti1Image:
    rng = np.random.default_rng(seed)
    return nib.Nifti1Image(rng.standard_normal((12, 12, 12)).astype(np.float32), np.eye(4))


def test_glass_brain_plots_signed_values_not_absolute_values() -> None:
    with patch("nilearn.plotting.plot_glass_brain") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.glass_brain(_noise_img(), threshold=2.3)
        except Exception:
            pass
    assert mock_plot.call_args.kwargs["plot_abs"] is False


def test_thresholded_mosaic_colour_limit_exceeds_its_threshold_for_a_noise_map() -> None:
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.stat_map_mosaic(_noise_img(), threshold=2.3)
        except Exception:
            pass
    kwargs = mock_plot.call_args.kwargs
    assert kwargs["vmax"] > kwargs["threshold"]


def test_mosaic_keeps_coordinate_and_laterality_annotation() -> None:
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.stat_map_mosaic(_noise_img(), threshold=2.3)
        except Exception:
            pass
    assert mock_plot.call_args.kwargs["annotate"] is True


def test_signed_maps_use_the_diverging_colormap_and_symmetric_bar() -> None:
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.stat_map_mosaic(_noise_img())
        except Exception:
            pass
    kwargs = mock_plot.call_args.kwargs
    assert kwargs["cmap"] == "RdBu_r"
    assert kwargs["symmetric_cbar"] is True


def test_mosaic_returns_a_figure_and_labels_its_colorbar() -> None:
    figure = stat_maps.stat_map_mosaic(_noise_img(), threshold=2.3, cbar_label="z")
    assert figure is not None
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_glass_brain_returns_a_figure_for_a_real_image() -> None:
    figure = stat_maps.glass_brain(_noise_img(), threshold=2.3)
    assert figure is not None
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_an_unthresholded_panel_uses_a_robust_symmetric_limit() -> None:
    data = np.zeros((12, 12, 12), dtype=np.float32)
    data[0, 0, 0] = 1000.0
    data[1:, :, :] = 1.0
    img = nib.Nifti1Image(data, np.eye(4))
    with patch("nilearn.plotting.plot_stat_map") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            stat_maps.stat_map_mosaic(img)
        except Exception:
            pass
    assert mock_plot.call_args.kwargs["vmax"] < 100.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_stat_maps.py -v`
Expected: FAIL — `ImportError: cannot import name 'stat_maps'`

- [ ] **Step 3: Implement the module**

`fmri_pipeline/analysis/report/figures/stat_maps.py`:

```python
"""Slice mosaic and glass-brain panels for a statistical map."""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from fmri_pipeline.analysis.report.style import (
    GUIDE_COLOR,
    SIGNED_CMAP,
    plot_context,
    robust_symmetric_limit,
    suprathreshold_limit,
)

logger = logging.getLogger(__name__)


def _masked_values(stat_img: Any) -> np.ndarray:
    data = np.asarray(stat_img.get_fdata())
    return data[np.isfinite(data)]


def _resolve_vmax(stat_img: Any, *, threshold: Optional[float], vmax: Optional[float]) -> float:
    """Choose a colour limit appropriate to whether the panel is thresholded.

    A thresholded panel takes its limit from the surviving voxels only. Reusing the
    whole-map limit is what makes these panels saturate: for z ~ N(0,1) the robust
    limit is about 2.58 against a typical 2.3 threshold, leaving no usable range.
    """
    if vmax is not None:
        return float(vmax)
    values = _masked_values(stat_img)
    if threshold is not None and threshold > 0:
        return suprathreshold_limit(values, threshold=float(threshold))
    return robust_symmetric_limit(values)


def _figure_of(display: Any) -> Any:
    figure = getattr(display, "figure", None) or getattr(display, "_fig", None)
    if figure is None and hasattr(display, "frame_axes"):
        figure = getattr(display.frame_axes, "figure", None)
    if figure is None:
        raise RuntimeError("Could not resolve a Matplotlib figure from the nilearn display.")
    return figure


def _label_colorbar(display: Any, label: str) -> None:
    """Name the units on the colorbar. Nilearn exposes no parameter for this."""
    colorbar = getattr(display, "_cbar", None)
    if colorbar is None:
        logger.debug("Nilearn display exposed no colorbar to label.")
        return
    colorbar.set_label(label, rotation=90, labelpad=6)


def stat_map_mosaic(
    stat_img: Any,
    *,
    bg_img: Any = None,
    threshold: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "",
    cbar_label: str = "z",
    cmap: str = SIGNED_CMAP,
) -> Any:
    """Draw a slice mosaic of ``stat_img``.

    ``annotate`` stays on: a mosaic without slice coordinates and left/right markers
    tells a reader neither where a cluster is nor which hemisphere it is in, which
    is most of what the panel exists to say.
    """
    from nilearn import plotting

    resolved_vmax = _resolve_vmax(stat_img, threshold=threshold, vmax=vmax)
    with plot_context():
        display = plotting.plot_stat_map(
            stat_img,
            bg_img=bg_img,
            title=title or None,
            display_mode="mosaic",
            threshold=float(threshold) if threshold else None,
            colorbar=True,
            vmax=resolved_vmax,
            cmap=cmap,
            dim=0,
            black_bg=False,
            symmetric_cbar=True,
            annotate=True,
        )
        _label_colorbar(display, cbar_label)
        return _figure_of(display)


def glass_brain(
    stat_img: Any,
    *,
    threshold: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "",
    cbar_label: str = "z",
    peak_coords: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> Any:
    """Draw a glass-brain projection of ``stat_img``.

    ``plot_abs=False`` is not optional. Nilearn defaults it to ``True``, which
    projects the absolute value: activation and deactivation then render
    identically, and the panel contradicts the signed mosaic beside it.

    ``peak_coords`` annotates each peak with its 1-based index so the projection can
    be read against the cluster table.
    """
    from nilearn import plotting

    resolved_vmax = _resolve_vmax(stat_img, threshold=threshold, vmax=vmax)
    with plot_context():
        display = plotting.plot_glass_brain(
            stat_img,
            title=title or None,
            threshold=float(threshold) if threshold else None,
            colorbar=True,
            vmax=resolved_vmax,
            cmap=SIGNED_CMAP,
            plot_abs=False,
            symmetric_cbar=True,
            black_bg=False,
        )
        _label_colorbar(display, cbar_label)
        if peak_coords:
            for index, coord in enumerate(peak_coords, start=1):
                display.add_markers(
                    [tuple(coord)], marker_color=GUIDE_COLOR, marker_size=18, marker="o"
                )
                display.annotate(size=7)
                logger.debug("Annotated peak %d at %s", index, coord)
        return _figure_of(display)


__all__ = ["glass_brain", "stat_map_mosaic"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_stat_maps.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/figures/stat_maps.py tests/fmri/report/test_stat_maps.py
git commit -m "fix(fmri): plot signed glass brains and give thresholded panels usable range"
```

---

### Task 4: Distribution figures

**Files:**
- Create: `fmri_pipeline/analysis/report/figures/distributions.py`
- Test: `tests/fmri/report/test_distributions.py`

**Interfaces:**
- Consumes: `plot_context`, `GUIDE_COLOR`, `OKABE_ITO` from Task 1.
- Produces:
  - `z_histogram(values: np.ndarray, *, threshold: float | None = None, title: str = "") -> Figure`
  - `magnitude_histogram(values: np.ndarray, *, xlabel: str, title: str = "") -> Figure`

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_distributions.py`:

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from fmri_pipeline.analysis.report.figures import distributions


def test_z_histogram_uses_a_log_y_axis() -> None:
    # A linear axis lets the null peak hide the tails, which are the signal.
    figure = distributions.z_histogram(np.random.default_rng(0).standard_normal(10_000))
    assert figure.axes[0].get_yscale() == "log"
    plt.close(figure)


def test_z_histogram_overlays_the_standard_normal_null() -> None:
    figure = distributions.z_histogram(np.random.default_rng(0).standard_normal(10_000))
    labels = [line.get_label() for line in figure.axes[0].lines]
    assert any("N(0, 1)" in label for label in labels)
    plt.close(figure)


def test_z_histogram_draws_both_threshold_lines_when_given_a_threshold() -> None:
    figure = distributions.z_histogram(
        np.random.default_rng(0).standard_normal(1000), threshold=2.3
    )
    positions = sorted(
        line.get_xdata()[0]
        for line in figure.axes[0].lines
        if line.get_linestyle() == "--"
    )
    assert positions == pytest.approx([-2.3, 2.3])
    plt.close(figure)


def test_z_histogram_has_no_legend_entries_without_a_threshold() -> None:
    # An unconditional legend() call previously warned and drew an empty box.
    figure = distributions.z_histogram(np.random.default_rng(0).standard_normal(1000))
    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == 1  # the null curve only
    plt.close(figure)


def test_z_histogram_rejects_an_empty_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        distributions.z_histogram(np.array([]))


def test_magnitude_histogram_marks_the_median() -> None:
    figure = distributions.magnitude_histogram(np.arange(100.0), xlabel="tSNR")
    positions = [
        line.get_xdata()[0]
        for line in figure.axes[0].lines
        if line.get_linestyle() == "--"
    ]
    assert positions == pytest.approx([49.5])
    plt.close(figure)


def test_magnitude_histogram_labels_the_axis_it_was_given() -> None:
    figure = distributions.magnitude_histogram(np.arange(100.0), xlabel="tSNR")
    assert figure.axes[0].get_xlabel() == "tSNR"
    plt.close(figure)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_distributions.py -v`
Expected: FAIL — `ImportError: cannot import name 'distributions'`

- [ ] **Step 3: Implement the module**

`fmri_pipeline/analysis/report/figures/distributions.py`:

```python
"""Histogram panels for statistic and magnitude distributions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fmri_pipeline.analysis.report.style import GUIDE_COLOR, OKABE_ITO, plot_context

_BINS = 120


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("A histogram requires at least one finite value.")
    return finite


def z_histogram(
    values: np.ndarray,
    *,
    threshold: float | None = None,
    title: str = "",
) -> plt.Figure:
    """Draw the distribution of z statistics against the standard normal null.

    The y-axis is logarithmic and the null is drawn on top, because the question
    this panel answers is how far the tails depart from N(0, 1). On a linear axis
    the null peak is the only visible feature and the tails -- the signal -- are
    flat against the axis.
    """
    finite = _finite(values)
    with plot_context():
        figure, axis = plt.subplots(figsize=(7.2, 3.2))
        counts, edges, _ = axis.hist(
            finite, bins=_BINS, color=OKABE_ITO["blue"], edgecolor="none"
        )
        centres = 0.5 * (edges[:-1] + edges[1:])
        bin_width = float(edges[1] - edges[0])
        null_density = (
            finite.size
            * bin_width
            * np.exp(-0.5 * centres**2)
            / np.sqrt(2.0 * np.pi)
        )
        axis.plot(
            centres, null_density, color=GUIDE_COLOR, linewidth=1.5, label="N(0, 1) null"
        )
        if threshold is not None and threshold > 0:
            for sign in (1.0, -1.0):
                axis.axvline(
                    sign * float(threshold),
                    color=OKABE_ITO["vermillion"],
                    linestyle="--",
                    linewidth=1.2,
                )
        axis.set_yscale("log")
        axis.set_ylim(bottom=max(0.5, float(np.min(counts[counts > 0])) * 0.5))
        axis.set_xlabel("z")
        axis.set_ylabel("voxels")
        if title:
            axis.set_title(title)
        axis.legend(fontsize=8)
        figure.tight_layout()
        return figure


def magnitude_histogram(
    values: np.ndarray,
    *,
    xlabel: str,
    title: str = "",
) -> plt.Figure:
    """Draw the distribution of an unsigned magnitude with its median marked.

    No alpha on the bars: a translucent fill over a solid histogram produces visible
    seams at every bar boundary that read as structure in the data.
    """
    finite = _finite(values)
    with plot_context():
        figure, axis = plt.subplots(figsize=(7.2, 3.2))
        axis.hist(finite, bins=_BINS, color=OKABE_ITO["bluish_green"], edgecolor="none")
        median = float(np.median(finite))
        axis.axvline(median, color=GUIDE_COLOR, linestyle="--", linewidth=1.2)
        axis.annotate(
            f"median {median:.3g}",
            xy=(median, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=8,
            color=GUIDE_COLOR,
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("voxels")
        if title:
            axis.set_title(title)
        figure.tight_layout()
        return figure


__all__ = ["magnitude_histogram", "z_histogram"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_distributions.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/figures/distributions.py tests/fmri/report/test_distributions.py
git commit -m "feat(fmri): draw z histograms on a log axis against the standard normal null"
```

---

### Task 5: tSNR volume figure

**Files:**
- Create: `fmri_pipeline/analysis/report/figures/volumes.py`
- Test: `tests/fmri/report/test_volumes.py`

**Interfaces:**
- Consumes: `MAGNITUDE_CMAP`, `plot_context` from Task 1; `magnitude_histogram` from Task 4.
- Produces:
  - `tsnr_volume(tsnr_img, *, bg_img=None, mask_img=None, title="", vmax=None) -> Figure`
  - `compute_tsnr(bold_imgs: Sequence[Any], *, mask_img=None) -> nib.Nifti1Image`

`compute_tsnr` is separated from rendering so the report layer can compute once and draw twice.

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_volumes.py`:

```python
from __future__ import annotations

from unittest.mock import patch

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.report.figures import volumes


def _bold(seed: int = 0, affine: np.ndarray | None = None) -> nib.Nifti1Image:
    rng = np.random.default_rng(seed)
    data = (100.0 + rng.standard_normal((8, 8, 8, 20))).astype(np.float32)
    return nib.Nifti1Image(data, np.eye(4) if affine is None else affine)


def test_compute_tsnr_returns_mean_over_standard_deviation() -> None:
    data = np.zeros((2, 2, 2, 10), dtype=np.float32)
    data[...] = np.arange(10, dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    tsnr = volumes.compute_tsnr([img])
    expected = float(np.mean(np.arange(10)) / np.std(np.arange(10)))
    assert np.allclose(np.asarray(tsnr.get_fdata()), expected)


def test_compute_tsnr_preserves_the_source_affine() -> None:
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    tsnr = volumes.compute_tsnr([_bold(affine=affine)])
    assert np.allclose(tsnr.affine, affine)


def test_compute_tsnr_averages_across_runs() -> None:
    tsnr = volumes.compute_tsnr([_bold(0), _bold(1)])
    assert tsnr.shape == (8, 8, 8)


def test_compute_tsnr_rejects_a_three_dimensional_image() -> None:
    img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
    with pytest.raises(ValueError, match="4D"):
        volumes.compute_tsnr([img])


def test_compute_tsnr_rejects_an_empty_run_list() -> None:
    with pytest.raises(ValueError, match="at least one"):
        volumes.compute_tsnr([])


def test_tsnr_volume_renders_through_nilearn_rather_than_slicing_the_array() -> None:
    # Voxel-axis slicing labels panels by anatomy without consulting the affine,
    # which is wrong for any non-RAS-canonical image.
    tsnr = volumes.compute_tsnr([_bold()])
    with patch("nilearn.plotting.plot_img") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            volumes.tsnr_volume(tsnr)
        except Exception:
            pass
    assert mock_plot.called


def test_tsnr_volume_uses_the_single_hue_magnitude_colormap() -> None:
    tsnr = volumes.compute_tsnr([_bold()])
    with patch("nilearn.plotting.plot_img") as mock_plot:
        mock_plot.return_value.figure = None
        try:
            volumes.tsnr_volume(tsnr)
        except Exception:
            pass
    assert mock_plot.call_args.kwargs["cmap"] == "cividis"


def test_tsnr_volume_returns_a_figure() -> None:
    figure = volumes.tsnr_volume(volumes.compute_tsnr([_bold()]))
    assert figure is not None
    plt.close(figure)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_volumes.py -v`
Expected: FAIL — `ImportError: cannot import name 'volumes'`

- [ ] **Step 3: Implement the module**

`fmri_pipeline/analysis/report/figures/volumes.py`:

```python
"""Volume renderings of unsigned magnitude maps."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import nibabel as nib
import numpy as np

from fmri_pipeline.analysis.report.style import (
    MAGNITUDE_CMAP,
    plot_context,
    robust_symmetric_limit,
)


def compute_tsnr(bold_imgs: Sequence[Any], *, mask_img: Any = None) -> nib.Nifti1Image:
    """Return the mean temporal SNR across runs.

    Separated from rendering so the report layer computes this once per subject
    rather than once per contrast.
    """
    if not bold_imgs:
        raise ValueError("compute_tsnr requires at least one BOLD image.")

    mask = None
    if mask_img is not None:
        mask = np.asanyarray(mask_img.dataobj).astype(bool)

    total: Optional[np.ndarray] = None
    affine = None
    for img in bold_imgs:
        data = np.asanyarray(img.dataobj)
        if data.ndim != 4:
            raise ValueError(f"compute_tsnr requires 4D images, got shape {data.shape}.")
        if affine is None:
            affine = img.affine
        mean = np.mean(data, axis=3)
        std = np.std(data, axis=3)
        # A zero-variance voxel has undefined tSNR, not infinite tSNR.
        tsnr = np.divide(mean, std, out=np.zeros_like(mean, dtype=float), where=std > 0)
        if mask is not None and mask.shape == tsnr.shape:
            tsnr = np.where(mask, tsnr, 0.0)
        if total is None:
            total = tsnr.astype(float)
        elif total.shape == tsnr.shape:
            total += tsnr
        else:
            raise ValueError(
                f"Runs disagree on shape: {total.shape} vs {tsnr.shape}."
            )

    assert total is not None  # guarded by the empty check above
    return nib.Nifti1Image((total / float(len(bold_imgs))).astype("float32"), affine)


def tsnr_volume(
    tsnr_img: Any,
    *,
    bg_img: Any = None,
    mask_img: Any = None,
    title: str = "",
    vmax: Optional[float] = None,
) -> Any:
    """Draw a tSNR map in anatomical orientation.

    Rendered through nilearn so the affine determines what "sagittal" means. Slicing
    the voxel array directly and labelling the panels by anatomy is correct only for
    RAS-canonical data and silently mislabels -- including left/right -- otherwise.
    """
    from nilearn import plotting

    data = np.asarray(tsnr_img.get_fdata())
    positive = data[np.isfinite(data) & (data > 0)]
    resolved_vmax = float(vmax) if vmax is not None else (
        robust_symmetric_limit(positive) if positive.size else 1.0
    )

    with plot_context():
        display = plotting.plot_img(
            tsnr_img,
            bg_img=bg_img,
            title=title or None,
            display_mode="ortho",
            cmap=MAGNITUDE_CMAP,
            vmin=0.0,
            vmax=resolved_vmax,
            colorbar=True,
            black_bg=False,
            annotate=True,
        )
        colorbar = getattr(display, "_cbar", None)
        if colorbar is not None:
            colorbar.set_label("tSNR", rotation=90, labelpad=6)
        figure = getattr(display, "figure", None) or getattr(display, "_fig", None)
        if figure is None:
            raise RuntimeError("Could not resolve a Matplotlib figure from the display.")
        return figure


__all__ = ["compute_tsnr", "tsnr_volume"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_volumes.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/figures/volumes.py tests/fmri/report/test_volumes.py
git commit -m "fix(fmri): render tSNR through the affine instead of slicing the voxel array"
```

---

### Task 6: Carpet figure with aligned motion

**Files:**
- Create: `fmri_pipeline/analysis/report/figures/carpet.py`
- Test: `tests/fmri/report/test_carpet.py`

**Interfaces:**
- Consumes: `plot_context`, `OKABE_ITO`, `GUIDE_COLOR` from Task 1; `PlotAssets` from Task 2.
- Produces:
  - `TISSUE_ORDER: tuple[str, ...] = ("GM", "WM", "CSF")`
  - `resolve_tissue_codes(shape, *, assets, reference_img) -> tuple[np.ndarray | None, str]` — returns per-voxel class codes and a source label of `"probseg"`, `"dseg"`, or `"none"`
  - `carpet_figure(carpet, *, tissue_codes, tissue_source, tr, run_boundaries, run_labels, fd=None, dvars=None, dvars_label="DVARS", title="") -> Figure`

`carpet` is `(n_voxels, n_frames)`, already standardised. `run_boundaries` are frame indices of run starts after the first. `fd` and `dvars` are per-frame arrays whose length matches the carpet's frame count; `NaN` is preserved and drawn as a gap.

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_carpet.py`:

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.report.assets import PlotAssets
from fmri_pipeline.analysis.report.figures import carpet as carpet_mod


def _carpet(n_voxels: int = 60, n_frames: int = 40) -> np.ndarray:
    return np.random.default_rng(0).standard_normal((n_voxels, n_frames))


def _probseg(tmp_path, name: str, data: np.ndarray) -> "nib.Nifti1Image":
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), str(path))
    return path


def test_resolve_tissue_codes_assigns_each_voxel_to_its_highest_probability_class(
    tmp_path,
) -> None:
    gm = np.zeros((2, 2, 2), dtype=np.float32)
    wm = np.zeros((2, 2, 2), dtype=np.float32)
    csf = np.zeros((2, 2, 2), dtype=np.float32)
    gm[0, 0, 0] = 0.9
    wm[0, 0, 1] = 0.8
    csf[1, 1, 1] = 0.7
    assets = PlotAssets(
        probseg={
            "GM": _probseg(tmp_path, "gm.nii.gz", gm),
            "WM": _probseg(tmp_path, "wm.nii.gz", wm),
            "CSF": _probseg(tmp_path, "csf.nii.gz", csf),
        }
    )
    reference = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))

    codes, source = carpet_mod.resolve_tissue_codes(
        (2, 2, 2), assets=assets, reference_img=reference
    )
    assert source == "probseg"
    assert codes[0, 0, 0] == 1  # GM
    assert codes[0, 0, 1] == 2  # WM
    assert codes[1, 1, 1] == 3  # CSF


def test_resolve_tissue_codes_reports_none_when_no_segmentation_exists() -> None:
    reference = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    codes, source = carpet_mod.resolve_tissue_codes(
        (2, 2, 2), assets=PlotAssets(), reference_img=reference
    )
    assert codes is None
    assert source == "none"


def test_carpet_declares_when_voxels_are_not_tissue_ordered() -> None:
    figure = carpet_mod.carpet_figure(
        _carpet(), tissue_codes=None, tissue_source="none", tr=2.0,
        run_boundaries=[], run_labels=["run-01"],
    )
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "unordered" in text.lower()
    plt.close(figure)


def test_carpet_time_axis_is_in_seconds_not_frames() -> None:
    figure = carpet_mod.carpet_figure(
        _carpet(n_frames=40), tissue_codes=None, tissue_source="none", tr=2.0,
        run_boundaries=[], run_labels=["run-01"],
    )
    assert "second" in figure.axes[-1].get_xlabel().lower()
    plt.close(figure)


def test_carpet_preserves_nan_in_the_motion_trace() -> None:
    # The first frame of every run has undefined FD. Filling zero draws a dip to
    # "no motion" that is a fabricated value.
    fd = np.concatenate([[np.nan], np.full(39, 0.1)])
    figure = carpet_mod.carpet_figure(
        _carpet(n_frames=40), tissue_codes=None, tissue_source="none", tr=2.0,
        run_boundaries=[], run_labels=["run-01"], fd=fd,
    )
    fd_axis = figure.axes[0]
    plotted = fd_axis.lines[0].get_ydata()
    assert np.isnan(plotted[0])
    plt.close(figure)


def test_carpet_labels_the_dvars_axis_with_the_series_it_was_given() -> None:
    figure = carpet_mod.carpet_figure(
        _carpet(n_frames=40), tissue_codes=None, tissue_source="none", tr=2.0,
        run_boundaries=[], run_labels=["run-01"],
        dvars=np.full(40, 1.0), dvars_label="std DVARS",
    )
    labels = [axis.get_ylabel() for axis in figure.axes]
    assert "std DVARS" in labels
    plt.close(figure)


def test_carpet_marks_and_names_every_run_boundary() -> None:
    figure = carpet_mod.carpet_figure(
        _carpet(n_frames=40), tissue_codes=None, tissue_source="none", tr=2.0,
        run_boundaries=[20], run_labels=["run-01", "run-02"],
    )
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "run-01" in text and "run-02" in text
    plt.close(figure)


def test_carpet_rejects_a_motion_trace_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="frames"):
        carpet_mod.carpet_figure(
            _carpet(n_frames=40), tissue_codes=None, tissue_source="none", tr=2.0,
            run_boundaries=[], run_labels=["run-01"], fd=np.zeros(10),
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_carpet.py -v`
Expected: FAIL — `ImportError: cannot import name 'carpet'`

- [ ] **Step 3: Implement the module**

`fmri_pipeline/analysis/report/figures/carpet.py`:

```python
"""Voxel carpet with motion traces on a shared time axis.

The carpet and the motion traces are one figure because they are only diagnostic
together: a band in the carpet means nothing until you can see whether a motion
spike sits above it. They were previously two figures on independent axes.

Voxels are grouped by tissue class. Ordering by raw mask index -- which is what a
mask's own iteration order gives -- scatters grey matter, white matter, and CSF
through the image and destroys the banding that makes a carpet readable at all.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from fmri_pipeline.analysis.report.assets import PlotAssets
from fmri_pipeline.analysis.report.style import GUIDE_COLOR, OKABE_ITO, plot_context

logger = logging.getLogger(__name__)

#: Row-block order. Codes are the 1-based index into this tuple; 0 means unassigned.
TISSUE_ORDER: Tuple[str, ...] = ("GM", "WM", "CSF")

#: FreeSurfer aseg label ranges collapsed onto TISSUE_ORDER, used by the dseg path.
_ASEG_TO_CLASS = {
    **{label: 1 for label in (3, 42, 8, 47, 10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58)},
    **{label: 2 for label in (2, 41, 7, 46, 16, 28, 60, 77, 251, 252, 253, 254, 255)},
    **{label: 3 for label in (4, 43, 5, 44, 14, 15, 24, 31, 63)},
}

_CARPET_CLIP = 2.5


def _resample_to(img: Any, reference_img: Any, *, order: int) -> Optional[np.ndarray]:
    from nilearn.image import resample_to_img

    interpolation = "nearest" if order == 0 else "continuous"
    resampled = resample_to_img(
        img, reference_img, interpolation=interpolation,
        force_resample=True, copy_header=True,
    )
    return np.asarray(resampled.get_fdata())


def resolve_tissue_codes(
    shape: Tuple[int, int, int],
    *,
    assets: PlotAssets,
    reference_img: Any,
) -> Tuple[Optional[np.ndarray], str]:
    """Return per-voxel tissue class codes and the source they came from.

    Probability maps win over the discrete segmentation: assigning each voxel to its
    highest-probability class is closer to what the carpet wants than a label map
    built for a different purpose. Returns ``(None, "none")`` when neither is
    available, which the figure then declares rather than hiding.
    """
    import nibabel as nib

    if assets.probseg:
        stack: List[np.ndarray] = []
        classes: List[int] = []
        for index, tissue in enumerate(TISSUE_ORDER, start=1):
            path = assets.probseg.get(tissue)
            if path is None:
                continue
            resampled = _resample_to(nib.load(str(path)), reference_img, order=1)
            if resampled is not None and resampled.shape == shape:
                stack.append(resampled)
                classes.append(index)
        if stack:
            probabilities = np.stack(stack, axis=0)
            winner = np.argmax(probabilities, axis=0)
            codes = np.take(np.asarray(classes), winner)
            # A voxel with no probability anywhere belongs to no class.
            codes = np.where(probabilities.max(axis=0) > 0, codes, 0)
            return codes.astype(np.int8), "probseg"

    if assets.dseg is not None:
        labels = _resample_to(nib.load(str(assets.dseg)), reference_img, order=0)
        if labels is not None and labels.shape == shape:
            codes = np.zeros(shape, dtype=np.int8)
            for label, klass in _ASEG_TO_CLASS.items():
                codes[np.asarray(labels).astype(int) == label] = klass
            if np.any(codes > 0):
                return codes, "dseg"

    logger.info("No tissue segmentation resolved; carpet will be unordered.")
    return None, "none"


def order_by_tissue(
    carpet: np.ndarray,
    tissue_codes_flat: Optional[np.ndarray],
) -> Tuple[np.ndarray, List[Tuple[str, int, int]]]:
    """Sort carpet rows into tissue blocks and return the block extents."""
    if tissue_codes_flat is None:
        return carpet, []
    order = np.argsort(tissue_codes_flat, kind="stable")
    ordered = carpet[order]
    blocks: List[Tuple[str, int, int]] = []
    sorted_codes = tissue_codes_flat[order]
    for index, tissue in enumerate(TISSUE_ORDER, start=1):
        positions = np.flatnonzero(sorted_codes == index)
        if positions.size:
            blocks.append((tissue, int(positions[0]), int(positions[-1]) + 1))
    return ordered, blocks


def _check_length(name: str, values: Optional[np.ndarray], n_frames: int) -> None:
    if values is not None and len(values) != n_frames:
        raise ValueError(
            f"{name} has {len(values)} samples but the carpet has {n_frames} frames."
        )


def carpet_figure(
    carpet: np.ndarray,
    *,
    tissue_codes: Optional[np.ndarray],
    tissue_source: str,
    tr: float,
    run_boundaries: Sequence[int],
    run_labels: Sequence[str],
    fd: Optional[np.ndarray] = None,
    dvars: Optional[np.ndarray] = None,
    dvars_label: str = "DVARS",
    title: str = "",
) -> plt.Figure:
    """Draw a carpet with FD and DVARS above it on a shared time axis.

    ``fd`` and ``dvars`` keep their ``NaN`` values. The first frame of a run has no
    defined framewise displacement; substituting zero draws a dip to "no motion" at
    every run boundary, which is a fabricated measurement. Matplotlib gaps a NaN.
    """
    carpet = np.asarray(carpet, dtype=float)
    n_frames = carpet.shape[1]
    _check_length("fd", fd, n_frames)
    _check_length("dvars", dvars, n_frames)

    ordered, blocks = order_by_tissue(carpet, tissue_codes)
    times = np.arange(n_frames) * float(tr)
    boundary_times = [float(b) * float(tr) for b in run_boundaries]

    trace_count = sum(x is not None for x in (fd, dvars))
    heights = [0.6] * trace_count + [3.0]
    with plot_context():
        figure, axes = plt.subplots(
            len(heights), 1, figsize=(11, 2.0 + 1.2 * trace_count),
            sharex=True, gridspec_kw={"height_ratios": heights},
        )
        axes = np.atleast_1d(axes)

        index = 0
        if fd is not None:
            axes[index].plot(times, np.asarray(fd, dtype=float),
                             color=OKABE_ITO["vermillion"], linewidth=0.8)
            axes[index].set_ylabel("FD (mm)")
            index += 1
        if dvars is not None:
            axes[index].plot(times, np.asarray(dvars, dtype=float),
                             color=OKABE_ITO["blue"], linewidth=0.8)
            axes[index].set_ylabel(dvars_label)
            index += 1

        carpet_axis = axes[-1]
        carpet_axis.imshow(
            np.clip(ordered, -_CARPET_CLIP, _CARPET_CLIP),
            aspect="auto", cmap="gray", vmin=-_CARPET_CLIP, vmax=_CARPET_CLIP,
            extent=(0.0, float(times[-1] if n_frames else 0.0), ordered.shape[0], 0),
            rasterized=True,
        )
        carpet_axis.set_xlabel("Time (seconds, concatenated runs)")

        if blocks:
            carpet_axis.set_yticks([0.5 * (start + stop) for _, start, stop in blocks])
            carpet_axis.set_yticklabels([name for name, _, _ in blocks])
            for _, _, stop in blocks[:-1]:
                carpet_axis.axhline(stop, color="white", linewidth=1.0)
            carpet_axis.set_ylabel("Voxels by tissue")
        else:
            carpet_axis.set_yticks([])
            carpet_axis.set_ylabel(f"Voxels (unordered: {tissue_source})")

        for axis in axes:
            for boundary in boundary_times:
                axis.axvline(boundary, color=GUIDE_COLOR, linewidth=0.7, alpha=0.6)

        starts = [0.0, *boundary_times]
        for label, start in zip(run_labels, starts):
            axes[0].annotate(
                label, xy=(start, 1.0), xycoords=("data", "axes fraction"),
                xytext=(2, 2), textcoords="offset points", fontsize=7, color=GUIDE_COLOR,
            )

        if title:
            figure.suptitle(title)
        figure.tight_layout()
        return figure


__all__ = ["TISSUE_ORDER", "carpet_figure", "order_by_tissue", "resolve_tissue_codes"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_carpet.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/figures/carpet.py tests/fmri/report/test_carpet.py
git commit -m "feat(fmri): tissue-ordered carpet with motion traces on a shared time axis"
```

---

### Task 7: Design matrix and collinearity figures

**Files:**
- Create: `fmri_pipeline/analysis/report/figures/design.py`
- Modify: `fmri_pipeline/analysis/reporting.py` — remove `_vif_from_design` (lines 100-135), which moves here
- Test: `tests/fmri/report/test_design.py`

**Interfaces:**
- Consumes: `plot_context`, `SIGNED_CMAP`, `OKABE_ITO`, `GUIDE_COLOR` from Task 1.
- Produces:
  - `vif_from_design(X: np.ndarray) -> np.ndarray` — moved verbatim from `reporting.py`
  - `design_matrix_figure(design_matrix: pd.DataFrame, *, contrast=None, contrast_name="", title="") -> Figure`
  - `collinearity_figure(design_matrix: pd.DataFrame, *, title="") -> Figure`

`contrast` is a 1-D array of length `len(design_matrix.columns)`, drawn as a labelled strip beneath the matrix.

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_design.py`:

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.report.figures import design


def _design(n_frames: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "task_a": rng.standard_normal(n_frames),
            "task_b": rng.standard_normal(n_frames),
            "trans_x": rng.standard_normal(n_frames),
            "drift_1": np.linspace(0, 1, n_frames),
            "constant": np.ones(n_frames),
        }
    )


def test_vif_is_infinite_for_a_perfectly_collinear_column() -> None:
    base = np.random.default_rng(0).standard_normal((40, 2))
    X = np.column_stack([base, base[:, 0]])
    assert np.isinf(design.vif_from_design(X)[2])


def test_vif_is_near_one_for_independent_columns() -> None:
    X = np.random.default_rng(0).standard_normal((400, 3))
    assert np.all(design.vif_from_design(X) < 1.5)


def test_vif_returns_empty_for_a_single_column() -> None:
    assert design.vif_from_design(np.ones((10, 1))).size == 0


def test_design_matrix_figure_labels_every_regressor() -> None:
    matrix = _design()
    figure = design.design_matrix_figure(matrix)
    labels = [t.get_text() for t in figure.axes[0].get_xticklabels()]
    assert set(matrix.columns) <= set(labels)
    plt.close(figure)


def test_design_matrix_figure_draws_a_contrast_strip_when_given_one() -> None:
    matrix = _design()
    figure = design.design_matrix_figure(
        matrix, contrast=np.array([1.0, -1.0, 0.0, 0.0, 0.0]), contrast_name="a - b"
    )
    assert len(figure.axes) >= 2
    text = " ".join(t.get_text() for t in figure.findobj(plt.Text))
    assert "a - b" in text
    plt.close(figure)


def test_design_matrix_figure_rejects_a_contrast_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="columns"):
        design.design_matrix_figure(_design(), contrast=np.array([1.0, -1.0]))


def test_collinearity_figure_reports_a_vif_bar_per_regressor() -> None:
    matrix = _design()
    figure = design.collinearity_figure(matrix)
    vif_axis = figure.axes[0]
    assert len(vif_axis.patches) == len(matrix.columns)
    plt.close(figure)


def test_collinearity_figure_uses_a_log_axis_so_infinite_vif_is_visible() -> None:
    figure = design.collinearity_figure(_design())
    assert figure.axes[0].get_xscale() == "log"
    plt.close(figure)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_design.py -v`
Expected: FAIL — `ImportError: cannot import name 'design'`

- [ ] **Step 3: Implement the module**

`fmri_pipeline/analysis/report/figures/design.py`:

```python
"""Design matrix, contrast, and collinearity panels.

Collinearity is the figure that says whether a contrast is estimable at all. The
variance inflation factors were already computed in the reporting path, but their
only destination was a single cell of a summary table.
"""

from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fmri_pipeline.analysis.report.style import (
    GUIDE_COLOR,
    OKABE_ITO,
    SIGNED_CMAP,
    plot_context,
)

#: VIF plotted in place of infinity, so a perfectly collinear regressor stays on the axis.
_VIF_CEILING = 1e3


def vif_from_design(X: np.ndarray) -> np.ndarray:
    """Variance inflation factor per column of ``X`` (n_samples, n_features).

    ``VIF_j = 1 / (1 - R²_j)`` where ``R²_j`` regresses column j on the others.
    Returns ``inf`` where ``R² >= 1`` (perfect collinearity) or the fit fails.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if p < 2:
        return np.array([], dtype=np.float64)

    vif = np.full(p, np.nan, dtype=np.float64)
    for j in range(p):
        y = X[:, j]
        Z = np.column_stack([np.ones(n), np.delete(X, j, axis=1)])
        try:
            beta, residuals, _rank, _ = np.linalg.lstsq(Z, y, rcond=None)
            ss_res = (
                float(residuals.flat[0]) if residuals.size
                else float(np.sum((y - Z @ beta) ** 2))
            )
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            if ss_tot <= 0:
                vif[j] = np.inf
                continue
            r_sq = 1.0 - (ss_res / ss_tot)
            vif[j] = np.inf if (r_sq >= 1.0 or np.isnan(r_sq)) else 1.0 / (1.0 - r_sq)
        except np.linalg.LinAlgError:
            vif[j] = np.inf
    return vif


def _column_limit(matrix: pd.DataFrame) -> float:
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.percentile(np.abs(finite), 98.0)) if finite.size else 1.0


def design_matrix_figure(
    design_matrix: pd.DataFrame,
    *,
    contrast: Optional[np.ndarray] = None,
    contrast_name: str = "",
    title: str = "",
) -> plt.Figure:
    """Draw the design matrix, optionally with its contrast as a strip beneath.

    Drawn here rather than through ``nilearn.plotting.plot_design_matrix`` so the
    contrast strip shares the column axis: a contrast named in prose is not legible,
    and one drawn on its own axis does not line up with the regressors it weights.
    """
    columns = list(design_matrix.columns)
    if contrast is not None:
        contrast = np.asarray(contrast, dtype=float).ravel()
        if contrast.size != len(columns):
            raise ValueError(
                f"Contrast has {contrast.size} weights but the design has "
                f"{len(columns)} columns."
            )

    limit = _column_limit(design_matrix)
    heights = [6.0, 0.5] if contrast is not None else [6.0]
    with plot_context():
        figure, axes = plt.subplots(
            len(heights), 1, figsize=(max(6.0, 0.32 * len(columns)), 5.5),
            gridspec_kw={"height_ratios": heights},
        )
        axes = np.atleast_1d(axes)

        axes[0].imshow(
            design_matrix.to_numpy(dtype=float), aspect="auto", cmap=SIGNED_CMAP,
            vmin=-limit, vmax=limit, rasterized=True,
        )
        axes[0].set_xticks(range(len(columns)))
        axes[0].set_xticklabels(columns, rotation=90, fontsize=7)
        axes[0].set_ylabel("Frame")
        if contrast is None:
            axes[0].set_xlabel("Regressor")
        if title:
            axes[0].set_title(title)

        if contrast is not None:
            strip_limit = float(np.max(np.abs(contrast))) or 1.0
            axes[1].imshow(
                contrast.reshape(1, -1), aspect="auto", cmap=SIGNED_CMAP,
                vmin=-strip_limit, vmax=strip_limit,
            )
            axes[1].set_yticks([0])
            axes[1].set_yticklabels([contrast_name or "contrast"], fontsize=8)
            axes[1].set_xticks(range(len(columns)))
            axes[1].set_xticklabels(columns, rotation=90, fontsize=7)
            axes[1].set_xlabel("Regressor")
            axes[0].set_xticklabels([])

        figure.tight_layout()
        return figure


def collinearity_figure(design_matrix: pd.DataFrame, *, title: str = "") -> plt.Figure:
    """Draw per-regressor VIF beside the regressor correlation matrix.

    The VIF axis is logarithmic because the interesting range spans one to infinity;
    on a linear axis every well-conditioned regressor collapses against the origin.
    Infinite VIF is drawn at a ceiling and labelled, so a perfectly collinear
    regressor stays visible instead of leaving the axis.
    """
    columns = list(design_matrix.columns)
    X = design_matrix.to_numpy(dtype=float)
    vif = vif_from_design(X)
    plotted = np.where(np.isfinite(vif), vif, _VIF_CEILING) if vif.size else np.array([])

    with plot_context():
        figure, (vif_axis, corr_axis) = plt.subplots(
            1, 2, figsize=(11, max(3.2, 0.22 * len(columns))),
            gridspec_kw={"width_ratios": [1.0, 1.3]},
        )

        positions = np.arange(len(columns))
        vif_axis.barh(positions, plotted if plotted.size else np.ones(len(columns)),
                      color=OKABE_ITO["sky_blue"])
        vif_axis.set_yticks(positions)
        vif_axis.set_yticklabels(columns, fontsize=7)
        vif_axis.invert_yaxis()
        vif_axis.set_xscale("log")
        vif_axis.set_xlabel("Variance inflation factor")
        vif_axis.axvline(1.0, color=GUIDE_COLOR, linewidth=0.8)
        for index, value in enumerate(vif):
            if not np.isfinite(value):
                vif_axis.annotate(
                    "inf", xy=(_VIF_CEILING, index), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7,
                    color=OKABE_ITO["vermillion"],
                )

        correlation = np.corrcoef(X, rowvar=False)
        image = corr_axis.imshow(correlation, cmap=SIGNED_CMAP, vmin=-1.0, vmax=1.0)
        corr_axis.set_xticks(positions)
        corr_axis.set_xticklabels(columns, rotation=90, fontsize=7)
        corr_axis.set_yticks(positions)
        corr_axis.set_yticklabels(columns, fontsize=7)
        bar = figure.colorbar(image, ax=corr_axis, fraction=0.04, pad=0.03)
        bar.set_label("Pearson r")

        if title:
            figure.suptitle(title)
        figure.tight_layout()
        return figure


__all__ = ["collinearity_figure", "design_matrix_figure", "vif_from_design"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_design.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Point `reporting.py` at the moved function**

In `fmri_pipeline/analysis/reporting.py`, delete `_vif_from_design` (lines 100-135) and add near the other imports:

```python
from fmri_pipeline.analysis.report.figures.design import vif_from_design as _vif_from_design
```

- [ ] **Step 6: Verify the existing reporting tests still pass**

Run: `.venv/bin/python -m pytest tests/fmri/ -v -k "report or vif"`
Expected: PASS, no collection errors.

- [ ] **Step 7: Commit**

```bash
git add fmri_pipeline/analysis/report/figures/design.py tests/fmri/report/test_design.py fmri_pipeline/analysis/reporting.py
git commit -m "feat(fmri): add design contrast strip and collinearity panel"
```

---

### Task 8: Rewire the reporting path

**Files:**
- Modify: `fmri_pipeline/analysis/reporting.py` — replace `generate_carpet_qc_images` (138-262), `generate_tsnr_qc_images` (265-420), the figure blocks of `generate_fmri_space_section` (984-1198), and the motion QC block (1371-1415)
- Test: `tests/fmri/report/test_reporting_integration.py`

**Interfaces:**
- Consumes: every `figures/` module from Tasks 3-7, `discover_plot_assets` from Task 2, `plot_context` from Task 1.
- Produces: no new public names. `generate_fmri_space_section` and `run_fmri_plotting_and_report` keep their signatures so `fmri_pipeline/pipelines/fmri_analysis.py` is untouched by this plan.

Two behaviour changes land here: a figure yields **one** `ReportImage` regardless of how many formats are written, and a panel failure is logged and skipped rather than raised.

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_reporting_integration.py`:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.plotting_config import FmriPlottingConfig
from fmri_pipeline.analysis.reporting import generate_fmri_space_section


def _stat_img(seed: int = 0) -> nib.Nifti1Image:
    rng = np.random.default_rng(seed)
    return nib.Nifti1Image(rng.standard_normal((12, 12, 12)).astype(np.float32), np.eye(4))


def _cfg(**kwargs) -> FmriPlottingConfig:
    base = dict(enabled=True, plot_types=("slices", "glass", "hist"), threshold_mode="z")
    base.update(kwargs)
    return FmriPlottingConfig(**base).normalized()


def test_a_figure_yields_one_report_entry_per_figure_not_per_format(
    tmp_path: Path,
) -> None:
    section = generate_fmri_space_section(
        space="mni", stat_img=_stat_img(), out_base_dir=tmp_path,
        formats=("png", "svg"), z_threshold=2.3, include_unthresholded=False,
        plot_types=("hist",), cfg=_cfg(formats=("png", "svg")),
    )
    titles = [image.title for image in section.images]
    assert len(titles) == len(set(titles))


def test_both_requested_formats_are_still_written_to_disk(tmp_path: Path) -> None:
    generate_fmri_space_section(
        space="mni", stat_img=_stat_img(), out_base_dir=tmp_path,
        formats=("png", "svg"), z_threshold=2.3, include_unthresholded=False,
        plot_types=("hist",), cfg=_cfg(formats=("png", "svg")),
    )
    out_dir = tmp_path / "plots" / "mni"
    assert (out_dir / "z_hist.png").exists()
    assert (out_dir / "z_hist.svg").exists()


def test_a_failing_panel_does_not_abort_the_section(tmp_path: Path) -> None:
    with patch(
        "fmri_pipeline.analysis.report.figures.stat_maps.stat_map_mosaic",
        side_effect=RuntimeError("boom"),
    ):
        section = generate_fmri_space_section(
            space="mni", stat_img=_stat_img(), out_base_dir=tmp_path,
            formats=("png",), z_threshold=2.3, include_unthresholded=True,
            plot_types=("slices", "hist"), cfg=_cfg(),
        )
    # The histogram survived even though the mosaic raised.
    assert any("histogram" in image.title.lower() for image in section.images)


def test_no_unthresholded_glass_brain_is_produced(tmp_path: Path) -> None:
    section = generate_fmri_space_section(
        space="mni", stat_img=_stat_img(), out_base_dir=tmp_path,
        formats=("png",), z_threshold=2.3, include_unthresholded=True,
        plot_types=("glass",), cfg=_cfg(),
    )
    titles = " ".join(image.title.lower() for image in section.images)
    assert "unthresholded" not in titles


def test_figures_are_closed_when_saving_fails(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    with patch("matplotlib.figure.Figure.savefig", side_effect=OSError("disk full")):
        generate_fmri_space_section(
            space="mni", stat_img=_stat_img(), out_base_dir=tmp_path,
            formats=("png",), z_threshold=2.3, include_unthresholded=False,
            plot_types=("hist",), cfg=_cfg(),
        )
    assert plt.get_fignums() == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_reporting_integration.py -v`
Expected: FAIL — duplicate titles, and the failing-panel test raises `RuntimeError: boom`.

- [ ] **Step 3: Add the save helper to `reporting.py`**

Insert near the top of `fmri_pipeline/analysis/reporting.py`, after the imports:

```python
from contextlib import suppress

from fmri_pipeline.analysis.report.figures import (
    carpet as carpet_figures,
    design as design_figures,
    distributions as distribution_figures,
    stat_maps as stat_map_figures,
    volumes as volume_figures,
)


def _save_figure(
    figure: Any,
    *,
    out_dir: Path,
    stem: str,
    formats: Sequence[str],
    title: str,
    caption: str = "",
) -> List[ReportImage]:
    """Write one figure to every requested format and return a single report entry.

    One entry, not one per format. ``formats`` says what to put on disk; the report
    embeds a figure once. Emitting one entry per format previously rendered every
    figure twice in a report configured for both PNG and SVG.
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        primary: Optional[Path] = None
        for fmt in formats:
            path = out_dir / f"{stem}.{fmt}"
            figure.savefig(path)
            if primary is None:
                primary = path
        if primary is None:
            return []
        return [ReportImage(title=title, path=primary, caption=caption)]
    finally:
        # Closed here rather than after a successful save: a figure leaked on the
        # error path grows without bound across a cohort.
        with suppress(Exception):
            plt.close(figure)


def _panel(description: str):
    """Context manager that logs and swallows one panel's failure.

    A panel that cannot be drawn is a gap in the report, not a reason to lose the
    rest of it. This is the same policy the QC blocks already used; the space
    sections previously logged and then re-raised.
    """
    from contextlib import contextmanager

    @contextmanager
    def _guard():
        try:
            yield
        except Exception as exc:
            logger.warning("Failed to generate %s (%s)", description, exc)

    return _guard()
```

- [ ] **Step 4: Replace the space-section figure blocks**

In `generate_fmri_space_section`, replace the bodies of the `slices`, `glass`, and `hist` blocks (lines 984-1092) with:

```python
    if "slices" in plot_types:
        if include_unthresholded:
            with _panel("unthresholded stat-map slices"):
                figure = stat_map_figures.stat_map_mosaic(
                    stat_img, bg_img=bg_img, threshold=None, vmax=z_vmax,
                    title=f"{title_prefix}Z map (unthresholded)".strip(),
                )
                images.extend(_save_figure(
                    figure, out_dir=out_dir, stem="stat_slices_unthresholded",
                    formats=formats, title="Stat map (slices) · unthresholded",
                ))
        if thr_label != "none":
            with _panel("thresholded stat-map slices"):
                figure = stat_map_figures.stat_map_mosaic(
                    stat_img if thr_img is None else thr_img, bg_img=bg_img,
                    threshold=float(thr_val) if thr_val is not None else None,
                    title=f"{title_prefix}Z map (thresholded)".strip(),
                )
                images.extend(_save_figure(
                    figure, out_dir=out_dir, stem="stat_slices_thresholded",
                    formats=formats, title="Stat map (slices) · thresholded",
                    caption=thr_label,
                ))

    # Only the thresholded glass brain is drawn. An unthresholded projection is a
    # saturated blob at any threshold setting and carries no information.
    if "glass" in plot_types and thr_label != "none":
        with _panel("thresholded glass brain"):
            figure = stat_map_figures.glass_brain(
                stat_img if thr_img is None else thr_img,
                threshold=float(thr_val) if thr_val is not None else None,
                title=f"{title_prefix}Glass brain (thresholded)".strip(),
            )
            images.extend(_save_figure(
                figure, out_dir=out_dir, stem="glass_thresholded", formats=formats,
                title="Glass brain · thresholded", caption=thr_label,
            ))

    if "hist" in plot_types:
        with _panel("z histogram"):
            data = np.asarray(stat_img.get_fdata())
            if mask_img is not None:
                mask = np.asarray(mask_img.get_fdata()).astype(bool)
                data = data[mask] if mask.shape == data.shape else data[data != 0]
            figure = distribution_figures.z_histogram(
                data, threshold=thr_val if thr_label != "none" else None,
                title="Z-statistic distribution",
            )
            images.extend(_save_figure(
                figure, out_dir=out_dir, stem="z_hist", formats=formats,
                title="Z histogram",
            ))
```

Delete the `_add_image` helper (lines 962-966) and the effect-size and standard-error blocks (lines 1137-1198), replacing the latter with the same `_panel` + `_save_figure` pattern using `stat_map_figures.stat_map_mosaic` for the effect map (`cbar_label="effect size"`) and `cmap=MAGNITUDE_CMAP` for the standard error.

- [ ] **Step 5: Replace the QC generators**

Rewrite `generate_carpet_qc_images` to build the carpet matrix as it does today, then resolve tissue codes and delegate:

```python
        codes, source = carpet_figures.resolve_tissue_codes(
            data.shape[:3], assets=assets, reference_img=img
        )
        flat_codes = codes[m] if codes is not None else None
        figure = carpet_figures.carpet_figure(
            carpet, tissue_codes=flat_codes, tissue_source=source, tr=tr,
            run_boundaries=run_breaks[1:-1], run_labels=run_labels,
            fd=fd_values, dvars=dvars_values, dvars_label=dvars_label,
            title="Carpet (as modelled)",
        )
        return _save_figure(
            figure, out_dir=qc_dir, stem="carpet_qc", formats=cfg.formats,
            title="QC: Carpet", caption=f"voxel order: {source}",
        )
```

Rewrite `generate_tsnr_qc_images` to call `volume_figures.compute_tsnr` and `volume_figures.tsnr_volume`, plus `distribution_figures.magnitude_histogram(values, xlabel="tSNR")`. Delete the raw-array montage entirely.

Delete the standalone motion QC block (lines 1371-1415): FD and DVARS now come from the confounds files into `generate_carpet_qc_images` and are drawn on the carpet's shared axis. Read them with `pd.read_csv(path, sep="\t")` and pass `df["framewise_displacement"].to_numpy()` **without** `fillna`, and set `dvars_label` to `"DVARS"` or `"std DVARS"` to match whichever column was found.

- [ ] **Step 6: Run the new and existing tests**

Run: `.venv/bin/python -m pytest tests/fmri/report/ tests/fmri/test_fmri_analysis_validity_guards.py -v`
Expected: PASS. Fix any signature drift surfaced by the existing guards.

- [ ] **Step 7: Commit**

```bash
git add fmri_pipeline/analysis/reporting.py tests/fmri/report/test_reporting_integration.py
git commit -m "refactor(fmri): route reporting through the tested figure layer

One report entry per figure regardless of format count, one failure policy
across every panel, and figures closed on the error path."
```

---

## Self-Review

**Spec coverage.** Every plan-1 item in the spec maps to a task: style layer → Task 1; unified asset discovery → Task 2; glass-brain sign, thresholded limits, annotation, colorbar labels → Task 3; z histogram and tSNR histogram → Task 4; affine-correct tSNR → Task 5; tissue-ordered carpet with aligned motion and preserved FD NaN → Task 6; design matrix, contrast strip, VIF, correlation → Task 7; duplicate-format entries, uniform failure policy, figure leaks, removal of the unthresholded glass brain and the raw-array montage → Task 8.

Deferred to plans 2 and 3, by design: the per-subject document and `html.py`, the manifest and `fmri report` entry point, the `FmriPlottingConfig` split, numbered cluster peaks keyed to the table (Task 3 ships the `peak_coords` parameter; the table wiring needs the report layer), the signature dot plot, and the whole rest profile including the ROI degeneracy change.

**Type consistency.** `plot_context`, `robust_symmetric_limit`, `suprathreshold_limit`, `figure_format`, `SIGNED_CMAP`, `MAGNITUDE_CMAP`, `GUIDE_COLOR` are defined in Task 1 and used under those exact names in Tasks 3-8. `PlotAssets` and `discover_plot_assets` are defined in Task 2 and consumed in Task 6. `vif_from_design` is defined in Task 7 and imported under an alias by `reporting.py` in the same task. `compute_tsnr` / `tsnr_volume` (Task 5) and `carpet_figure` / `resolve_tissue_codes` (Task 6) are called in Task 8 with the signatures declared.

**Known gap to confirm during Task 3.** `_label_colorbar` reaches `display._cbar`, a nilearn private attribute, because `plot_stat_map` exposes no colorbar-label parameter in 0.14.0. It degrades to a debug log if the attribute moves. If the implementer finds a public accessor in the installed version, use it instead.
