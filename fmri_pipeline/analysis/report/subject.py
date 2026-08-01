"""Assembles one subject-task report from contrast manifests.

Everything here reads derivatives. Nothing fits a model: that separation is what
lets QC be computed once per subject rather than once per contrast, and what lets a
figure be reworked without re-running a GLM. The invariant is enforced by a test
that imports this module and asserts the fitting modules stay absent, so be careful
what you import at module scope.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from fmri_pipeline.analysis.plotting_config import FmriReportConfig
from fmri_pipeline.analysis.report import atlas, html, inference
from fmri_pipeline.analysis.report.figures import carpet as carpet_figures
from fmri_pipeline.analysis.report.figures import coverage as coverage_figures
from fmri_pipeline.analysis.report.figures import distributions as distribution_figures
from fmri_pipeline.analysis.report.figures import motion as motion_figures
from fmri_pipeline.analysis.report.figures import stat_maps as stat_map_figures
from fmri_pipeline.analysis.report.figures import volumes as volume_figures
from fmri_pipeline.analysis.report.manifest import (
    ContrastManifest,
    validate_manifest_artifacts,
    validate_manifest_collection,
)
from fmri_pipeline.analysis.report.style import (
    figure_format,
    plot_context,
    savefig_kwargs,
)

logger = logging.getLogger(__name__)


class SummaryFacts:
    """The numbers the top of the report states, collected as panels compute them.

    A subject report runs to eight sections, and deciding whether a subject's result
    is usable currently means scrolling all of them: the censoring is in the motion
    table, the tSNR in a QC panel, the survivor count in a calibration figure, the
    signature score in its own section. Across ninety subjects that is the difference
    between triage and reading ninety documents.

    Collected rather than recomputed. Every number here is already produced by a panel
    -- a second pass over the 4D data to restate the tSNR would cost more than the rest
    of the document put together, and a recomputation that drifted from its panel would
    be worse than no summary at all.
    """

    def __init__(self) -> None:
        self._items: List[Tuple[str, str]] = []

    def add(self, label: str, value: Any) -> None:
        """Record one fact. Later facts with the same label replace earlier ones."""
        text = str(value)
        for index, (existing, _value) in enumerate(self._items):
            if existing == label:
                self._items[index] = (label, text)
                return
        self._items.append((label, text))

    @property
    def items(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)


def _note(facts: Optional[SummaryFacts], label: str, value: Any) -> None:
    """Record a summary fact when a summary is being collected."""
    if facts is not None:
        facts.add(label, value)


@contextmanager
def _panel(description: str) -> Iterator[None]:
    """Log and swallow one panel's failure so the document still builds."""
    try:
        yield
    except Exception as exc:
        logger.warning("Failed to generate %s (%s)", description, exc)


def _save(
    figure: Any,
    *,
    out_dir: Path,
    stem: str,
    formats: Sequence[str],
    dense: bool = True,
) -> Optional[Path]:
    """Write a figure and return the path the report should embed.

    ``dense`` decides which format is embedded, through
    :func:`~fmri_pipeline.analysis.report.style.figure_format`: a brain mosaic or a
    carpet is a dense image layer and stays raster, while a line or bar figure is
    written as vector so its text stays legible at any zoom -- in a browser whose
    width the author does not control, and in a manuscript where the figure is
    scaled again.

    That helper existed, was documented, and was tested, and nothing called it: every
    panel in this report was rasterised at 150 dpi regardless of the ``dense`` flag
    each caller was carefully setting. The preferred format is written first and
    returned; anything else the config asks for is still written beside it.

    Saving happens inside the style context because ``svg.hashsalt`` and
    ``savefig.dpi`` are read at save time, not draw time.
    """
    import matplotlib.pyplot as plt

    preferred = figure_format(dense=dense)
    wanted = [preferred] + [fmt for fmt in formats if fmt != preferred]

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        primary: Optional[Path] = None
        with plot_context():
            for fmt in wanted:
                path = out_dir / f"{stem}.{fmt}"
                figure.savefig(path, **savefig_kwargs(path))
                if primary is None:
                    primary = path
        return primary
    finally:
        try:
            plt.close(figure)
        except Exception:
            logger.debug("Could not close figure %s", stem)


def _slug(manifest: ContrastManifest) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch == "-" else "-" for ch in manifest.contrast_name
    )
    return "contrast-" + cleaned.strip("-").lower()


#: What each recorded signal-scaling mode makes an effect a percentage *of*.
#:
#: Only ``voxel-mean`` divides a voxel by its own temporal mean, which is what percent
#: signal change means. The others are percentages of a different denominator, and a
#: colourbar reading "% signal change" over either of them would name a quantity the
#: map does not carry.
_SCALED_EFFECT_UNITS = {
    "voxel-mean": "% signal change",
    "grand-mean": "% of the grand mean signal",
    "timepoint-mean": "% of the per-volume mean signal",
}


def _unit_name(manifest: ContrastManifest) -> str:
    """Name the units the effect and its standard error are both measured in.

    A contrast effect is in arbitrary BOLD units unless the model applied signal
    scaling. Printing "% signal change" on a map that is not in those units invites
    a quantitative reading the number cannot support -- and declining to print it on a
    map that *is* in those units throws the reading away, which is the error this
    pipeline was actually making: the model scales unconditionally, and the manifest
    recorded otherwise.

    A scaled map whose mode is unrecognised gets the neutral label rather than a
    guessed denominator.
    """
    if not manifest.signal_scaling:
        return "arbitrary BOLD units"
    return _SCALED_EFFECT_UNITS.get(
        str(manifest.signal_scaling_mode or ""), "scaled BOLD units"
    )


def _effect_units(manifest: ContrastManifest) -> str:
    """Label for an effect map's colourbar."""
    units = _unit_name(manifest)
    return units if units.startswith("%") else f"effect ({units})"


def _error_units(manifest: ContrastManifest) -> str:
    """Label for a standard-error colourbar.

    The same units as the effect, but the quantity is not the effect. Reusing
    :func:`_effect_units` verbatim labelled the standard-error colourbar "effect",
    which names the wrong map.
    """
    return f"standard error ({_unit_name(manifest)})"


def build_header_section(
    manifests: Sequence[ContrastManifest], *, background_source: str = ""
) -> html.Section:
    """Summarise the acquisition and what entered the model.

    Excluded runs carry their reasons: a report that says a run was dropped without
    saying why gives a reader nothing to act on. The anatomical underlay is named for
    the same reason: panels drawn over nothing look like panels drawn over something
    until a reader tries to locate a cluster on them.
    """
    first = manifests[0]
    items = [
        ("Subject", first.subject),
        ("Task", first.task),
        ("Space", first.space),
        ("Contrasts", str(len(manifests))),
        ("Runs included", ", ".join(first.included_runs) or "none"),
        ("TR", f"{first.t_r:.3g} s" if first.t_r else "unknown"),
        (
            "Smoothing",
            f"{first.smoothing_fwhm:.3g} mm FWHM" if first.smoothing_fwhm else "none",
        ),
        ("Confound strategy", first.confound_strategy or "unspecified"),
        ("Effect units", _effect_units(first)),
        (
            "Analysis mask",
            "the mask the GLM was fitted inside"
            if first.mask_is_analysis_mask
            else "discovered from preprocessing; not verified as the fitted mask",
        ),
    ]
    if background_source:
        items.append(
            (
                "Volume underlay",
                Path(background_source).name
                if Path(background_source).suffix
                else background_source,
            )
        )
    blocks: List[html.Block] = [
        html.KeyValues(title="Acquisition and model", items=tuple(items))
    ]
    if first.excluded_runs:
        blocks.append(
            html.KeyValues(
                title="Runs excluded",
                items=tuple((run, reason) for run, reason in first.excluded_runs),
            )
        )
    return html.Section(slug="overview", title="Overview", blocks=tuple(blocks))


def load_background(
    *, deriv_root: Path, manifest: ContrastManifest
) -> Tuple[Any, str]:
    """Load the anatomical image the volume panels are drawn over.

    Every volume panel in this report previously passed ``bg_img=None``, so a cluster
    floated in empty space and could not be judged against grey matter, a ventricle,
    or the edge of the brain -- which is most of what localising a result means. The
    background was already discovered for the carpet's tissue ordering and thrown away.

    Returns ``(None, reason)`` when no anatomy is available, so the caller can say the
    panels are unbacked rather than leaving a reader to assume they are not.
    """
    from fmri_pipeline.analysis.report.assets import discover_plot_assets

    try:
        assets = discover_plot_assets(
            deriv_root=Path(deriv_root),
            subject=manifest.subject,
            task=manifest.task,
            space=manifest.space,
        )
    except Exception as exc:
        logger.warning("Could not discover plotting assets (%s)", exc)
        return None, "asset discovery failed"

    if assets.background is None:
        return None, "no anatomical image found in the derivatives"

    try:
        import nibabel as nib

        return nib.load(str(assets.background)), str(assets.background)
    except Exception as exc:
        logger.warning("Could not load background %s (%s)", assets.background, exc)
        return None, "the discovered anatomical image could not be read"


def _retained_frame_masks(
    bold_imgs: Sequence[Any],
    retained_frame_indices: Sequence[Sequence[int]],
) -> List[np.ndarray]:
    """Return exact boolean masks from the frame indices supplied to the GLM."""
    if len(bold_imgs) != len(retained_frame_indices):
        raise ValueError("Retained frame indices must align with loaded BOLD runs.")

    masks: List[np.ndarray] = []
    for image, retained_indices in zip(bold_imgs, retained_frame_indices):
        mask = np.zeros(int(image.shape[3]), dtype=bool)
        mask[np.asarray(retained_indices, dtype=int)] = True
        masks.append(mask)
    return masks


def build_qc_sections(
    *,
    manifests: Sequence[ContrastManifest],
    deriv_root: Path,
    out_dir: Path,
    cfg: FmriReportConfig,
    background: Any = None,
    facts: Optional[SummaryFacts] = None,
) -> List[html.Section]:
    """Build the as-modelled QC section, once for the whole subject-task.

    Driven by the first manifest's run list because every contrast of a subject-task
    shares it. Recomputing per contrast produced byte-identical figures and read the
    entire 4D dataset once per contrast.

    "As modelled" rather than "as preprocessed": these panels describe the runs that
    entered the GLM, after confound selection and smoothing. fMRIPrep's own report
    covers the preprocessing, and this document does not restate it.
    """
    import nibabel as nib

    first = manifests[0]
    qc_dir = out_dir / "plots" / "qc"
    blocks: List[html.Block] = []

    bold_imgs = [
        nib.load(str(path)) for path in first.bold_paths if Path(path).exists()
    ]

    sample_masks = _retained_frame_masks(
        bold_imgs,
        first.retained_frame_indices,
    )

    # tSNR is computed before the run-level table so its per-run medians can join the
    # motion columns rather than getting a near-flat dot plot of their own.
    tsnr_result = None
    if bold_imgs and cfg.include_tsnr_qc:
        with _panel("tSNR"):
            tsnr_result = volume_figures.compute_tsnr(
                bold_imgs,
                # The mask keeps partial-volume rim voxels out of the median and the
                # colour limit. Without it, `tsnr > 0` admits edge voxels sitting at
                # very low tSNR and drags the reported value down.
                mask_img=_load_mask(first),
                sample_masks=sample_masks,
            )

    if first.confounds_paths and cfg.include_motion_qc:
        with _panel("motion summary"):
            blocks.extend(
                _motion_blocks(
                    manifest=first,
                    sample_masks=sample_masks,
                    qc_dir=qc_dir,
                    cfg=cfg,
                    facts=facts,
                    tsnr=tsnr_result,
                )
            )

    if bold_imgs and cfg.include_carpet_qc:
        with _panel("carpet"):
            blocks.extend(
                _carpet_blocks(
                    bold_imgs=bold_imgs,
                    manifest=first,
                    sample_masks=sample_masks,
                    qc_dir=qc_dir,
                    cfg=cfg,
                    deriv_root=deriv_root,
                )
            )

    if tsnr_result is not None:
        # The map only. Per-run tSNR is a column of the run-level table above: it has
        # no published reference level to be located against, so comparing six runs
        # is comparing six numbers, and the dot plot this used to draw put them
        # within 2 tSNR of each other and showed nothing.
        with _panel("tSNR map"):
            path = _save(
                volume_figures.tsnr_volume(
                    tsnr_result,
                    bg_img=background,
                    # The same mask the median was computed inside. Omitted, the
                    # colour limit came from a different population of voxels than
                    # the number printed beside it.
                    mask_img=_load_mask(first),
                    radiological=first.radiological,
                    title="tSNR (as modelled)",
                ),
                out_dir=qc_dir,
                stem="tsnr_map",
                formats=cfg.formats,
            )
            if path:
                blocks.append(
                    html.Figure(
                        title="tSNR",
                        path=path,
                        caption=(
                            "Where the measurement is precise enough to detect an "
                            "effect. Per-run values are in the run-level table above."
                        ),
                    )
                )
            tsnr_values = np.asarray(tsnr_result.mean_img.get_fdata())
            measurable = tsnr_values[np.isfinite(tsnr_values) & (tsnr_values > 0)]
            if measurable.size:
                _note(facts, "Median tSNR", f"{float(np.median(measurable)):.1f}")

    if first.mask and Path(first.mask).exists():
        with _panel("coverage"):
            # The extent claim comes from the manifest's own record of how the mask was
            # derived. This panel used to assert "intersection across N runs" for
            # whatever mask it was handed, which was false whenever a single run's
            # fMRIPrep brain mask had been recorded instead of the GLM's own.
            extent_note = (
                f"intersection across {len(first.included_runs)} run(s), as fitted"
                if first.mask_is_analysis_mask
                else "as recorded in the manifest; not verified against the fitted model"
            )
            path = _save(
                coverage_figures.coverage_figure(
                    nib.load(str(first.mask)),
                    bg_img=background,
                    extent_note=extent_note,
                    radiological=first.radiological,
                    title="Analysis mask",
                ),
                out_dir=qc_dir,
                stem="coverage",
                formats=cfg.formats,
            )
            if path:
                blocks.append(
                    html.Figure(
                        title="Coverage",
                        path=path,
                        caption="Voxels outside this mask were not tested.",
                    )
                )

    return [
        html.Section(
            slug="qc",
            title="Quality control (as modelled)",
            blocks=tuple(blocks)
            or (html.Note(text="No QC panels could be generated."),),
        )
    ]


def _motion_blocks(
    *,
    manifest: ContrastManifest,
    sample_masks: Optional[Sequence[np.ndarray]],
    qc_dir: Path,
    cfg: FmriReportConfig,
    facts: Optional[SummaryFacts] = None,
    tsnr: Any = None,
) -> List[html.Block]:
    """Build the run-level QC table, and the one motion panel worth drawing.

    The framewise-displacement figure survives because displacement has published
    reference levels -- Power et al. (2012, 2014) -- and locating a run against a
    level is a comparison a reader makes by eye. It also separates the typical frame
    from the single worst one, which are different measurements on a quantity
    dominated by isolated spikes.

    Everything else is a column. ``tsnr`` joins the table rather than getting a panel:
    it carries no reference level, so "is one run unlike the others" is a comparison
    between six numbers.
    """
    summaries = motion_figures.summarise_run_motion(
        manifest.confounds_paths,
        run_labels=manifest.included_runs,
        sample_masks=sample_masks,
    )
    if not summaries:
        return []

    blocks: List[html.Block] = []
    with _panel("motion figure"):
        path = _save(
            motion_figures.run_motion_figure(
                summaries, title="Head motion by run (as modelled)"
            ),
            out_dir=qc_dir,
            stem="motion_by_run",
            dense=False,
            formats=cfg.formats,
        )
        if path:
            blocks.append(
                html.Figure(
                    title="Head motion by run",
                    path=path,
                    dense=False,
                    caption=(
                        "Dot: median framewise displacement. Bar: interquartile "
                        "range. Open marker: the single worst frame, which motion's "
                        "spikiness makes a different measurement from the typical "
                        "one. Reference levels are published conventions; no run is "
                        "scored against them here."
                    ),
                )
            )

    with _panel("motion-signal coupling"):
        path = _save(
            motion_figures.motion_coupling_figure(
                manifest.confounds_paths,
                run_labels=manifest.included_runs,
                title="Motion against signal change",
            ),
            out_dir=qc_dir,
            stem="motion_coupling",
            formats=cfg.formats,
        )
        if path:
            blocks.append(
                html.Figure(
                    title="Motion against signal change",
                    path=path,
                    dense=True,
                    caption=(
                        "DVARS that tracks framewise displacement is signal change "
                        "driven by head motion. DVARS that moves independently of it "
                        "is another source, which motion regressors will not remove "
                        "and censoring on displacement will not catch. The "
                        "correlation is a measurement; no run is scored against it."
                    ),
                )
            )

    tsnr_median = list(getattr(tsnr, "per_run_median", ()) or ()) or None
    tsnr_iqr = list(getattr(tsnr, "per_run_iqr", ()) or ()) or None
    table_html, _rows = motion_figures.motion_table(
        summaries, tsnr_median=tsnr_median, tsnr_iqr=tsnr_iqr
    )
    tsv_path = motion_figures.write_motion_tsv(
        summaries,
        path=qc_dir / "run_qc.tsv",
        tsnr_median=tsnr_median,
        tsnr_iqr=tsnr_iqr,
    )
    censored = sum(run.n_censored for run in summaries)

    acquired = sum(run.n_frames for run in summaries)
    _note(facts, "Runs modelled", f"{len(summaries)}")
    _note(
        facts,
        "Frames censored",
        f"{censored:,} of {acquired:,}"
        + (f" ({censored / acquired:.1%})" if acquired else ""),
    )
    medians = [run.median_fd for run in summaries if run.median_fd is not None]
    if medians:
        _note(
            facts,
            "Median framewise displacement",
            f"{float(np.median(medians)):.3f} mm across runs "
            f"(worst run {float(np.max(medians)):.3f} mm)",
        )
    blocks.append(
        html.Table(
            title="Run-level quality control",
            html=table_html,
            tsv_path=tsv_path,
            caption=(
                f"{sum(run.n_frames for run in summaries):,} frames acquired, "
                f"{censored:,} censored, "
                f"{sum(run.n_retained for run in summaries):,} entered the model. "
                "Censoring is the model's own, read from the confound columns the GLM "
                "used, so these counts describe the analysis that ran. tSNR is "
                "measured inside the analysis mask, after the same censoring."
            ),
        )
    )
    return blocks


def _carpet_blocks(
    *,
    bold_imgs: Sequence[Any],
    manifest: ContrastManifest,
    sample_masks: Optional[Sequence[np.ndarray]],
    qc_dir: Path,
    cfg: FmriReportConfig,
    deriv_root: Path,
) -> List[html.Block]:
    """Build the carpet panel from already-loaded runs."""
    import pandas as pd

    from fmri_pipeline.analysis.report.assets import discover_plot_assets

    standardised: List[np.ndarray] = []
    run_breaks = [0]
    voxel_mask = None
    mask_source = "nonzero mean signal"

    # The analysis mask decides which voxels the carpet shows, when there is one.
    # Falling back on "mean signal is not exactly zero" admits the whole field of
    # view on any acquisition whose background carries noise rather than true zeros
    # -- measured here, 136,416 voxels of a 50,626-voxel brain -- so the panel
    # sampled air, the row count it reported was a fraction of the wrong total, and a
    # carpet captioned "as modelled" showed voxels the model never saw.
    analysis_mask = _load_mask(manifest) if manifest.mask_is_analysis_mask else None
    if analysis_mask is not None:
        candidate = np.asanyarray(analysis_mask.dataobj).astype(bool)
        reference = np.asanyarray(bold_imgs[0].dataobj).shape[:3]
        if candidate.shape == reference:
            voxel_mask = candidate
            mask_source = "analysis mask"

    for index, img in enumerate(bold_imgs):
        data = np.asanyarray(img.dataobj)
        if data.ndim != 4:
            continue
        if voxel_mask is None:
            mean_volume = np.mean(data, axis=3)
            voxel_mask = np.isfinite(mean_volume) & (mean_volume != 0)
        voxels = data[voxel_mask]
        mask = sample_masks[index] if sample_masks and index < len(sample_masks) else None
        if mask is not None and mask.size != voxels.shape[1]:
            mask = None
        standardised.append(
            carpet_figures.standardise_carpet(voxels, sample_mask=mask)
        )
        run_breaks.append(run_breaks[-1] + int(voxels.shape[1]))

    if not standardised:
        return []

    carpet = np.concatenate(standardised, axis=1)

    fd_parts: List[np.ndarray] = []
    dvars_parts: List[np.ndarray] = []
    dvars_label = "DVARS"
    for path in manifest.confounds_paths:
        try:
            frame = pd.read_csv(str(path), sep="\t")
        except OSError:
            continue
        # The first frame of a run has no defined framewise displacement.
        # Substituting zero draws a dip to "no motion" at every run boundary,
        # which is a fabricated measurement; matplotlib gaps a NaN.
        fd_parts.append(
            frame["framewise_displacement"].to_numpy(dtype=float)
            if "framewise_displacement" in frame.columns
            else np.full(len(frame), np.nan)
        )
        if "dvars" in frame.columns:
            dvars_parts.append(frame["dvars"].to_numpy(dtype=float))
        elif "std_dvars" in frame.columns:
            dvars_parts.append(frame["std_dvars"].to_numpy(dtype=float))
            dvars_label = "std DVARS"

    fd = np.concatenate(fd_parts) if fd_parts else None
    dvars = np.concatenate(dvars_parts) if dvars_parts else None
    if fd is not None and fd.size != carpet.shape[1]:
        fd = None
    if dvars is not None and dvars.size != carpet.shape[1]:
        dvars = None

    codes, source = (None, "none")
    with _panel("tissue segmentation"):
        assets = discover_plot_assets(
            deriv_root=Path(deriv_root),
            subject=manifest.subject,
            task=manifest.task,
            space=manifest.space,
        )
        volume_codes, source = carpet_figures.resolve_tissue_codes(
            np.asanyarray(bold_imgs[0].dataobj).shape[:3],
            assets=assets,
            reference_img=bold_imgs[0],
        )
        if volume_codes is not None and voxel_mask is not None:
            codes = volume_codes[voxel_mask]

    # The censoring the GLM applied, concatenated onto the carpet's own time axis so
    # the frames that left the model can be read against the motion that removed them.
    censored = None
    if sample_masks is not None and len(sample_masks) == len(bold_imgs):
        joined = np.concatenate([~np.asarray(m, dtype=bool) for m in sample_masks])
        if joined.size == carpet.shape[1]:
            censored = joined

    figure = carpet_figures.carpet_figure(
        carpet,
        tissue_codes=codes,
        tissue_source=source,
        tr=float(manifest.t_r or 1.0),
        run_boundaries=run_breaks[1:-1],
        run_labels=manifest.included_runs,
        fd=fd,
        dvars=dvars,
        dvars_label=dvars_label,
        censored=censored,
        voxel_source=mask_source,
        title="Carpet (as modelled)",
    )
    path = _save(figure, out_dir=qc_dir, stem="carpet", formats=cfg.formats)
    if path is None:
        return []
    return [
        html.Figure(
            title="Carpet with motion",
            path=path,
            caption=(
                f"Voxel order: {source}. Motion shares the carpet's time axis, "
                "which is the only arrangement in which either is diagnostic."
            ),
        )
    ]


def masked_stat_values(stat_img: Any, mask_img: Any) -> Tuple[np.ndarray, str]:
    """Return the statistic values inside the analysis mask, and their source.

    Everything inferential in this module reads this rather than the raw volume. On a
    typical map over 60% of the volume is exact background zero, and including it
    inflates the test count behind every corrected threshold, drags a fitted null
    toward zero, and puts a spike at the origin that dominates any distribution panel.
    """
    data = np.asarray(stat_img.get_fdata())
    finite = np.isfinite(data)
    if mask_img is not None:
        mask = np.asanyarray(mask_img.dataobj).astype(bool)
        if mask.shape == data.shape:
            return data[finite & mask], "analysis mask"
        logger.warning(
            "Analysis mask shape %s does not match the map's %s.", mask.shape, data.shape
        )
    nonzero = finite & (data != 0)
    if nonzero.any() and int(nonzero.sum()) < int(finite.sum()):
        return data[nonzero], "nonzero voxels (no usable mask)"
    return data[finite], "all voxels (no usable mask)"


def resolve_threshold(
    manifest: ContrastManifest, *, values: np.ndarray
) -> Tuple[Optional[float], str]:
    """Return the height threshold this contrast's panels are drawn at, and its label.

    Every ``threshold_mode`` the config accepts resolves here. Previously only ``z``
    did, and the other two -- both validated, both configurable -- produced a contrast
    section containing no panels at all: no dual-coded map, no thresholded map, no
    glass brain, no cluster table. A supported setting has to produce a report.

    ``fdr`` resolves against the map's own p-values inside the analysis mask, so the
    threshold is a property of this contrast rather than a number carried over from a
    config file. It returns ``None`` when Benjamini-Hochberg rejects nothing, which is
    a finding and is stated as one.
    """
    mode = str(manifest.threshold_mode or "").strip().lower()
    if mode == "z":
        threshold = float(manifest.z_threshold)
        return (threshold, f"|z| > {threshold:.2f} (uncorrected)") if threshold > 0 else (None, "none")
    if mode == "fdr":
        from fmri_pipeline.analysis.report import inference

        threshold = inference.fdr_threshold(
            values, q=float(manifest.fdr_q), two_sided=manifest.two_sided
        )
        if threshold is None:
            return None, f"FDR q = {manifest.fdr_q:g}: no voxel survives correction"
        return threshold, f"FDR q = {manifest.fdr_q:g} (|z| > {threshold:.2f})"
    return None, "none"


def _load_mask(manifest: ContrastManifest) -> Any:
    """Load the analysis mask, or None when the manifest records none.

    Best-effort: a missing or unreadable mask leaves the colour-limit helpers to
    fall back on excluding exact zeros, which they state on the figure.
    """
    if not manifest.mask:
        return None
    path = Path(manifest.mask)
    if not path.exists():
        return None
    try:
        import nibabel as nib

        return nib.load(str(path))
    except Exception as exc:  # pragma: no cover - depends on a corrupt file
        logger.warning("Could not load analysis mask %s (%s)", path, exc)
        return None


def supports_glass_brain(space: str) -> bool:
    """Whether a glass-brain projection is defined for ``space``.

    The projection is drawn against a fixed MNI schematic. A map in native or T1w
    space projected onto it lands on anatomy it does not correspond to, which is an
    error rather than an approximation -- and one that is invisible in the result,
    since the output looks like a perfectly ordinary glass brain either way.
    """
    return str(space or "").strip().lower() == "mni"


def coordinate_space_label(space: str) -> str:
    """Name the space a cluster table's coordinates are actually in.

    An unlabelled X/Y/Z column in an fMRI cluster table reads as MNI, because that
    is the overwhelming convention. For a native-space contrast that is a silent
    misreport, and nothing in the table lets a reader detect it -- so the label says
    outright that the coordinates are not MNI and cannot be looked up in an atlas.

    The manifest records the space as ``native`` for everything that is not MNI, which
    made the interpolated form read "native scanner-native". Named spaces keep their
    name; the generic one does not repeat itself.
    """
    text = str(space or "").strip()
    if text.lower() == "mni":
        return "coordinates: MNI152 (mm)"
    if text.lower() in {"", "native"}:
        return "coordinates: scanner-native (mm), not MNI; not atlas-referable"
    return f"coordinates: {text} (mm), not MNI; not atlas-referable"


def _cluster_peaks(frame: Any) -> Tuple[Tuple[str, Tuple[float, float, float]], ...]:
    """Return one labelled peak per cluster, skipping nilearn's sub-peak rows.

    ``get_clusters_table`` writes secondary local maxima as extra rows whose
    ``Cluster ID`` is the parent's number with a letter appended -- 1a, 1b -- and
    whose size column is an empty string. Treating every row as a peak numbered the
    markers 1..N while the table read 1, 1a, 2, so marker 2 pointed at a sub-peak of
    cluster 1 while the reader looked up cluster 2. The caption says the two key to
    each other.

    Labels are the table's own cluster IDs rather than a fresh count, so the two
    agree even if this filter ever changes.

    Ordered by ``|Peak Stat|`` descending rather than by the table's own order. nilearn
    sorts clusters by the *signed* statistic, so every negative cluster sorts below
    every positive one no matter how strong: on this study's data the largest effect in
    the map peaks at z = -8.81 over 138,213 mm3 and lands last, which kept it out of
    every panel that caps at three or six peaks. The table itself keeps nilearn's order
    and lists every cluster, so nothing is hidden there; this ordering decides only
    which peaks the capped panels spend their space on.

    Alignment is unaffected: ``enrich_cluster_frame`` matches rows by ``Cluster ID``
    rather than by position.
    """
    if not {"X", "Y", "Z", "Cluster ID"} <= set(frame.columns):
        return ()

    # Read columns directly rather than iterating rows. The Cluster ID column holds
    # a mix of integers and strings, and `iterrows` upcasts each row to a common
    # dtype -- turning cluster 1 into "1.0", which no longer looks like an integer.
    identifiers = list(frame["Cluster ID"])
    xs, ys, zs = list(frame["X"]), list(frame["Y"]), list(frame["Z"])
    stats = (
        list(frame["Peak Stat"]) if "Peak Stat" in frame.columns else [None] * len(xs)
    )

    peaks: List[Tuple[str, Tuple[float, float, float]]] = []
    strengths: List[float] = []
    for identifier, x, y, z, stat in zip(identifiers, xs, ys, zs, stats):
        label = _cluster_identifier(identifier)
        if label is None:
            continue
        peaks.append((label, (float(x), float(y), float(z))))
        try:
            strength = abs(float(stat))
        except (TypeError, ValueError):
            strength = float("nan")
        strengths.append(strength)

    if not any(np.isfinite(strength) for strength in strengths):
        # No usable statistic: the caller's order is better than an arbitrary one.
        return tuple(peaks)

    order = sorted(
        range(len(peaks)),
        key=lambda i: (
            -strengths[i] if np.isfinite(strengths[i]) else float("inf"),
            i,
        ),
    )
    return tuple(peaks[i] for i in order)


def _cluster_identifier(value: Any) -> Optional[str]:
    """Return a cluster's own label, or None when the row is a sub-peak.

    A cluster's ID is a bare number; a sub-peak's carries a letter suffix (1a, 1b).
    Numbers arrive as ints, floats, or strings depending on how pandas typed the
    column, so all three are normalised to the same plain integer text.
    """
    if isinstance(value, str):
        text = value.strip()
        return text if text.isdigit() else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number != int(number):
        return None
    return str(int(number))


def noise_mask(
    stat_img: Any, *, mask_img: Any, threshold: Optional[float]
) -> Tuple[Optional[np.ndarray], str]:
    """The in-mask voxels a smoothness estimator may treat as noise, and their name.

    Smoothness is a property of the residual field, and this pipeline saves no
    residuals -- so it is estimated from the statistic map, where real activation
    inflates it: signal is spatially structured, and the estimator cannot tell that
    structure from smoothing. Measured on this study's own contrast the whole-map
    estimate is 8.8 mm against 6.5 mm from the sub-threshold voxels, where 6 mm of
    smoothing was applied to a 3 mm grid and the theoretical answer is 6.7 mm. The
    36% error propagates into every quantity expressed in resels, which scale as the
    inverse cube of the estimate.

    Excluding the voxels above the display threshold is what removes it. Under
    ``threshold_mode: none`` there is no height to exclude by, and the whole mask is
    returned with a name that says the result is an upper bound rather than an
    estimate.
    """
    in_mask: Optional[np.ndarray] = None
    data = np.asarray(stat_img.get_fdata())
    if mask_img is not None:
        candidate = np.asanyarray(mask_img.dataobj).astype(bool)
        if candidate.shape == data.shape:
            in_mask = candidate

    if threshold is None or threshold <= 0:
        return in_mask, "statistic map, including suprathreshold voxels (upper bound)"

    below = np.isfinite(data) & (np.abs(data) <= float(threshold))
    combined = below if in_mask is None else (below & in_mask)
    if not combined.any():
        # Every voxel is suprathreshold. Excluding them all leaves nothing to
        # estimate from, so the honest fallback is the whole mask, named as such.
        return in_mask, "statistic map, including suprathreshold voxels (upper bound)"
    return combined, "sub-threshold voxels of the statistic map"


def smoothness_facts(
    stat_img: Any,
    *,
    mask_img: Any,
    cluster_min_voxels: int,
    threshold: Optional[float] = None,
) -> List[str]:
    """Describe the map's spatial smoothness and what it makes a voxel count mean.

    ``cluster_min_voxels`` is configured as a bare count, and a bare count is not
    comparable to anything: the same twenty voxels is a strong constraint on
    unsmoothed 3 mm data and almost none at 8 mm FWHM. Expressing it in resolution
    elements says how many independent bumps of noise a surviving cluster spans.

    The search volume gets the same treatment, and for a sharper reason: every
    corrected threshold in this report divides alpha across the *voxel* count, while
    smoothing has already made neighbouring voxels the same measurement. Stating both
    numbers is what lets a reader see how far a Bonferroni height over 50,626 voxels
    overshoots a family of roughly 5,000 independent ones.

    Best-effort. A smoothness that cannot be estimated costs these lines and nothing
    else -- an unresolved measurement is not a reason to lose the cluster table.
    """
    from fmri_pipeline.analysis.report.figures import coverage as coverage_figures

    try:
        mask, source = noise_mask(stat_img, mask_img=mask_img, threshold=threshold)
        fwhm = coverage_figures.estimate_fwhm(stat_img, mask=mask)
    except Exception as exc:
        logger.info("Could not estimate smoothness (%s)", exc)
        return []

    facts = [coverage_figures.smoothness_note(fwhm, source=source)]
    if mask_img is not None:
        with _panel("search volume in resels"):
            resels = coverage_figures.search_volume_resels(mask_img, fwhm=fwhm)
            facts.append(
                f"the search volume is {resels:,.0f} resels; the corrected heights "
                f"above divide alpha across voxels, not resels"
            )
    if cluster_min_voxels > 0:
        resels = coverage_figures.extent_in_resels(
            cluster_min_voxels, reference_img=stat_img, fwhm=fwhm
        )
        facts.append(f"the {cluster_min_voxels}-voxel extent filter is {resels:.2f} resels")
    return facts


def _peak_values(img: Any, coords: Sequence[Tuple[float, float, float]]) -> List[float]:
    """Sample a volume at a list of world coordinates.

    Nearest voxel, not interpolation: a peak is a voxel, and interpolating between it
    and its neighbours would report a number no voxel in the map carries.
    """
    data = np.asarray(img.get_fdata())
    inverse = np.linalg.inv(np.asarray(img.affine))
    shape = np.asarray(data.shape[:3])

    out: List[float] = []
    for coord in coords:
        voxel = np.rint(
            (np.append(np.asarray(coord, dtype=float), 1.0) @ inverse.T)[:3]
        ).astype(int)
        if np.any(voxel < 0) or np.any(voxel >= shape):
            out.append(float("nan"))
            continue
        out.append(float(data[tuple(voxel)]))
    return out


#: Digits kept per cluster-table column when the table is rendered for reading.
#:
#: A peak coordinate is a voxel centre on a 3 mm grid, so nilearn's
#: ``54.884781`` claims a precision of about a thousandth of a millimetre that the
#: acquisition does not have. Effects and their errors get three significant figures,
#: which is what a methods section quotes. Full precision stays in the TSV beside the
#: table: that file is read by machines, and rounding it would lose real information.
_DISPLAY_DECIMALS = {"X": 0, "Y": 0, "Z": 0, "Peak Stat": 2}
_DISPLAY_SIGNIFICANT = 3


def _rounded(value: Any, *, decimals: Optional[int]) -> Any:
    """Round one cell, leaving anything that is not a finite number untouched.

    Element-wise rather than column-wise because the enriched columns are object
    dtype: a sub-peak row carries an empty string where a cluster row carries a
    float, so a whole-column ``round`` skips them and the table showed a peak effect
    of 0.419831 beside a coordinate rounded to the millimetre.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return value
    if isinstance(value, (int, np.integer)):
        # An integer in this table is an identifier or a count, and neither has a
        # precision to reduce. Rounding them to three significant figures turned
        # cluster 2 into "2.0" and a 5,211 mm3 cluster into 5,210.
        return value
    if not np.isfinite(value):
        return value
    if decimals is None:
        # Significant figures rather than decimal places: a percent signal change of
        # 0.42 and a standard error of 0.0689 need different decimal counts to carry
        # the same information.
        return float(f"%.{_DISPLAY_SIGNIFICANT}g" % float(value))
    rounded = round(float(value), decimals)
    return int(rounded) if decimals <= 0 else rounded


def for_display(frame: Any) -> Any:
    """Round a cluster table's numbers to the precision the measurement supports.

    Applied to a copy. The TSV is written from the unrounded frame, so a reader gets
    a legible table and a script gets the exact values.
    """
    shown = frame.copy()
    for column in shown.columns:
        decimals = _DISPLAY_DECIMALS.get(column)
        if column in _DISPLAY_DECIMALS:
            shown[column] = [_rounded(v, decimals=decimals) for v in shown[column]]
        else:
            shown[column] = [_rounded(v, decimals=None) for v in shown[column]]
    return shown


def enrich_cluster_frame(
    frame: Any,
    *,
    manifest: ContrastManifest,
    peaks: Sequence[Tuple[str, Tuple[float, float, float]]],
    labeller: Any = None,
    source: Optional["ClusterTableSource"] = None,
) -> Tuple[Any, List[str]]:
    """Add the effect, its standard error, and an anatomical label at each peak.

    nilearn's cluster table carries a peak z and a size, which says how strong the
    evidence is and how far it spreads but not how large the effect *is*. Two peaks at
    z = 4 can differ tenfold in percent signal change, and only one of them is worth
    reporting as a result. The standard error beside it is what distinguishes a large
    effect from an imprecise one.

    Sub-peak rows -- nilearn writes secondary local maxima as ``1a``, ``1b`` -- are
    left untouched: ``peaks`` holds one entry per cluster, and the columns are aligned
    on the table's own ``Cluster ID``.

    Returns the frame and the notes the caption should carry. Best-effort: a map that
    cannot be read costs its column and nothing else.
    """
    import nibabel as nib

    notes: List[str] = []
    if not peaks or "Cluster ID" not in getattr(frame, "columns", ()):
        return frame, notes

    labels = [label for label, _coord in peaks]
    coords = [coord for _label, coord in peaks]
    by_cluster = {label: index for index, label in enumerate(labels)}
    row_index = [by_cluster.get(_cluster_identifier(v)) for v in frame["Cluster ID"]]

    def _column(values: Sequence[float]) -> List[Any]:
        return [
            "" if position is None or not np.isfinite(values[position])
            else values[position]
            for position in row_index
        ]

    # Read the effect at each peak from the same maps the table's coordinates came
    # from. Sampling the native effect map at an MNI coordinate would return the
    # value at those millimetres in a different brain.
    if source is None:
        source = cluster_table_source(manifest)

    if source.effect_map and Path(source.effect_map).exists():
        with _panel("peak effect sizes"):
            effects = _peak_values(nib.load(str(source.effect_map)), coords)
            frame[f"Peak effect ({_unit_name(manifest)})"] = _column(effects)
            notes.append("peak effect and error are the value at the peak voxel")

    if source.variance_map and Path(source.variance_map).exists():
        with _panel("peak standard errors"):
            variances = _peak_values(nib.load(str(source.variance_map)), coords)
            errors = [float(np.sqrt(v)) if v >= 0 else float("nan") for v in variances]
            frame["Peak SE"] = _column(errors)

    if labeller is not None:
        with _panel("anatomical labels"):
            named = labeller.label_all(coords)
            frame["Region"] = [
                "" if position is None else (named[position] or "unlabelled")
                for position in row_index
            ]
            notes.append(f"regions from {labeller.source}")
    elif not atlas.atlas_applies_to(source.space):
        notes.append(atlas.space_refusal(source.space))

    return frame, notes


#: BIDS space entity of the standard-space companion the pipeline writes.
_MNI_ENTITY = "space-MNI152NLin2009cAsym"


@dataclass(frozen=True)
class ClusterTableSource:
    """The maps a cluster table is built from, and the space its coordinates are in.

    Usually the fitted maps themselves. When a standard-space companion exists it is
    preferred, because a table of scanner-native millimetres is not referable to any
    atlas and not comparable to any published coordinate -- which is most of what a
    peak table is read for.
    """

    stat_map: Path
    effect_map: Optional[Path]
    variance_map: Optional[Path]
    space: str
    #: Whether these maps come from a different fit than the report's own.
    separately_fitted: bool = False


def _mni_companion(stat_map: Path) -> Optional[ClusterTableSource]:
    """Find the standard-space maps the pipeline writes beside a fitted contrast.

    Discovered by naming convention rather than read from the manifest, for two
    reasons. The maps are produced after the manifest is written, in the plotting
    path, so no manifest recorded them. And discovery works on a derivatives tree
    built by an earlier run, which is the case this report is meant to serve.

    The companion is a *separate fit* -- the pipeline refits the contrast against
    fMRIPrep's standard-space BOLD rather than resampling the native statistic. That
    is the stronger choice statistically, and it is why the caption has to say the
    table and the maps above it describe two fits of the same contrast rather than
    one map shown twice.
    """
    stat_map = Path(stat_map)
    name = stat_map.name
    if _MNI_ENTITY in name or "_stat-" not in name:
        return None

    head = name.partition("_stat-")[0]
    # The config hash, so a directory holding two configurations does not cross them.
    # Taken from the native name's own trailing token; when there is none, the glob
    # still matches and the first candidate is used.
    stem = name
    for extension in (".nii.gz", ".nii"):
        if stem.endswith(extension):
            stem = stem[: -len(extension)]
            break
    wanted_hash = stem.rsplit("_", 1)[-1]

    def _beside(quantity: str) -> Optional[Path]:
        # Globbed rather than reconstructed: the quantity token itself contains
        # underscores -- "z_score", "effect_size" -- so splitting the name on "_" to
        # recover the trailing hash silently truncates it.
        candidates = sorted(
            stat_map.parent.glob(f"{head}_{_MNI_ENTITY}_stat-{quantity}_*.nii*")
        )
        if not candidates:
            return None
        for candidate in candidates:
            if wanted_hash in candidate.name:
                return candidate
        return candidates[0]

    z_map = _beside("z_score")
    if z_map is None:
        return None
    return ClusterTableSource(
        stat_map=z_map,
        effect_map=_beside("effect_size"),
        variance_map=_beside("effect_variance"),
        space="mni",
        separately_fitted=True,
    )


def cluster_table_source(manifest: ContrastManifest) -> ClusterTableSource:
    """The maps the cluster table is built from: the fitted ones.

    A standard-space companion is *not* preferred here, though one is often present
    and :func:`_mni_companion` will find it. The peaks this table returns are not only
    labels: the run-consistency panel samples the per-run maps at them and the
    peak-response panel samples the BOLD series at them, and both of those are in the
    fitted space. The companion is a separate fit with its own clusters, so its peaks
    have no one-to-one correspondence with the fitted map's -- substituting them would
    make the forest panel's "peak 1" a different location from the table's "cluster 1"
    while both kept the same number, and would sample native maps at standard-space
    millimetres.

    Reporting standard-space coordinates therefore needs the companion presented as
    its own contrast rather than spliced into this one's table. Until then this
    returns the fitted maps and the caption says the coordinates are not referable.
    """
    return ClusterTableSource(
        stat_map=Path(manifest.stat_map),
        effect_map=Path(manifest.effect_map) if manifest.effect_map else None,
        variance_map=Path(manifest.variance_map) if manifest.variance_map else None,
        space=manifest.space,
    )


def companion_manifest(manifest: ContrastManifest) -> Optional[ContrastManifest]:
    """Describe the standard-space fit as a contrast in its own right.

    The pipeline refits each contrast against fMRIPrep's standard-space BOLD rather
    than resampling the fitted statistic, so the companion is a second analysis of the
    same data, not a second view of the same numbers. Reported as its own section for
    that reason: every panel in a section then describes one fit, and the coordinates
    in its table are referable to an atlas and to published work, which scanner-native
    millimetres never are.

    Everything the companion does not itself possess is cleared rather than inherited,
    because inheriting it would be wrong in a way nothing in the output would show:

    - per-run maps, the sign-flip null and the run-influence table are properties of
      the fitted model, and sampling them at this section's coordinates would read
      native maps at standard-space millimetres;
    - the BOLD and residual series likewise, which is what disables the peak-response
      and model-fit panels here;
    - the analysis mask, which was written for the fitted grid and does not share this
      one. The distribution helpers fall back on excluding exact zeros and say so, and
      on these maps that recovers the fitted extent exactly.

    ``None`` when no companion was written -- the common case, since producing one
    depends on ``fmri_stats.space`` including a standard space.
    """
    import dataclasses

    companion = _mni_companion(Path(manifest.stat_map))
    if companion is None:
        return None

    return dataclasses.replace(
        manifest,
        contrast_name=f"{manifest.contrast_name} · MNI152NLin2009cAsym",
        space="mni",
        stat_map=companion.stat_map,
        effect_map=companion.effect_map,
        variance_map=companion.variance_map,
        mask=None,
        mask_is_analysis_mask=False,
        run_effect_map=None,
        run_variance_map=None,
        sign_flip_null_tsv=None,
        run_influence_tsv=None,
        sign_flip_fwe_height=None,
        sign_flip_fwe_survivors=None,
        sign_flip_global_p=None,
        sign_flip_p_floor=None,
        sign_flip_n_patterns=None,
        sign_flip_n_runs=None,
        sign_flip_observed_max=None,
        bold_paths=(),
        residual_paths=(),
        predicted_paths=(),
    )


def build_cluster_table(
    *,
    manifest: ContrastManifest,
    out_dir: Path,
    threshold: Optional[float] = None,
    threshold_label: str = "",
    extra_facts: Sequence[str] = (),
    labeller: Any = None,
) -> Tuple[Optional[html.Table], Tuple[Tuple[str, Tuple[float, float, float]], ...]]:
    """Return the cluster table and its peak coordinates.

    The peaks are handed to the glass brain so the numbered markers on the
    projection key to the numbered rows in the table. Coordinates in one place and
    a picture in another leaves the matching to the reader.

    The caption states the height threshold and any extent filter as two separate
    facts. This pipeline performs no cluster-level familywise correction, and
    Eklund, Nichols & Knutsson (2016) measured false-positive rates up to 70% for
    parametric cluster inference, so nothing here may read as inferential about
    extent.

    ``threshold`` is resolved from the manifest when not supplied. Callers that have
    already resolved it pass it in, because doing so under ``threshold_mode: fdr``
    means a second pass over every voxel's p value.
    """
    import nibabel as nib

    try:
        from nilearn import reporting
    except ImportError:
        return None, ()

    if threshold is None:
        values, _source = masked_stat_values(
            nib.load(str(manifest.stat_map)), _load_mask(manifest)
        )
        threshold, threshold_label = resolve_threshold(manifest, values=values)
    if threshold is None:
        return None, ()

    plots_dir = out_dir / "plots" / _slug(manifest)
    source = cluster_table_source(manifest)
    frame = reporting.get_clusters_table(
        nib.load(str(source.stat_map)),
        stat_threshold=float(threshold),
        cluster_threshold=manifest.cluster_min_voxels or 0,
        two_sided=manifest.two_sided,
    )

    peaks = _cluster_peaks(frame)
    frame, enrichment_notes = enrich_cluster_frame(
        frame, manifest=manifest, peaks=peaks, labeller=labeller, source=source
    )

    plots_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = plots_dir / "clusters.tsv"
    frame.to_csv(tsv_path, sep="\t", index=False)

    caption_parts = [
        "two-sided" if manifest.two_sided else "one-sided",
        f"height threshold: {threshold_label or f'|z| > {threshold:.2f}'}",
        coordinate_space_label(source.space),
    ]
    caption_parts.extend(enrichment_notes)
    caption_parts.extend(str(fact) for fact in extra_facts if fact)
    if manifest.cluster_min_voxels > 0:
        caption_parts.append(
            f"clusters smaller than {manifest.cluster_min_voxels} voxels removed for "
            "display; this is an extent filter, not familywise-error-corrected "
            "cluster-level inference"
        )

    return (
        html.Table(
            title="Clusters and peaks",
            html=for_display(frame).to_html(index=False, border=0, classes=""),
            tsv_path=tsv_path,
            caption="; ".join(caption_parts),
        ),
        peaks,
    )


def resolve_labeller(manifest: ContrastManifest, cfg: FmriReportConfig) -> Any:
    """Load the configured atlas, if it may be read at this contrast's coordinates.

    Gated on the space of the *table's coordinates* rather than on the space the model
    was fitted in. Those differ whenever a standard-space companion exists: the fit is
    native, the table is not, and it is the table the atlas is sampled at.

    Still gated, though. An MNI atlas sampled at a native-space coordinate returns the
    name of whatever structure sits at those millimetres in a different brain, and the
    result is indistinguishable from a correct label -- so a contrast with no
    standard-space companion gets no column and a caption saying why.
    """
    if not atlas.atlas_applies_to(cluster_table_source(manifest).space):
        return None
    return atlas.load_atlas(
        labels_img=getattr(cfg, "atlas_labels_img", None),
        labels_tsv=getattr(cfg, "atlas_labels_tsv", None),
    )


def build_contrast_section(
    *,
    manifest: ContrastManifest,
    out_dir: Path,
    cfg: FmriReportConfig,
    background: Any = None,
    facts: Optional[SummaryFacts] = None,
    deriv_root: Optional[Path] = None,
) -> html.Section:
    """Build the results section for one contrast.

    Leads with the dual-coded panel, which shows the whole map, then the
    hard-thresholded panel the cluster table refers to. Both are needed: the first
    so a reader can see near-threshold structure, the second so the figure and the
    table describe the same voxels.

    Closes on the calibration panel, which is what says whether the threshold the
    other panels were drawn at means what it claims. It sits here rather than in the
    collapsed diagnostics because a reader who does not see it will read every panel
    above as more decisive than it is.
    """
    import nibabel as nib

    from fmri_pipeline.analysis.report import inference

    plots_dir = out_dir / "plots" / _slug(manifest)
    stat_img = nib.load(str(manifest.stat_map))
    # Colour limits, the fitted null, and every corrected threshold are computed
    # inside this mask. Over the whole volume a percentile is taken from a
    # distribution that is mostly background zeros and lands far too low.
    mask_img = _load_mask(manifest)
    values, mask_source = masked_stat_values(stat_img, mask_img)
    threshold, threshold_label = resolve_threshold(manifest, values=values)
    blocks: List[html.Block] = []

    prefix = f"{manifest.contrast_name}: "
    _note(facts, f"{prefix}height threshold", threshold_label or "none")
    # The survivor count is recorded further down, from the calibration panel's own
    # context, so that it can state the count expected under the fitted null beside
    # it. Fitting that null here as well would compute it twice per contrast and
    # leave two copies free to drift apart.

    if manifest.effect_map and Path(manifest.effect_map).exists() and threshold:
        with _panel(f"dual-coded panel for {manifest.contrast_name}"):
            path = _save(
                stat_map_figures.dual_coded_mosaic(
                    nib.load(str(manifest.effect_map)),
                    stat_img=stat_img,
                    bg_img=background,
                    mask_img=mask_img,
                    threshold=float(threshold),
                    two_sided=manifest.two_sided,
                    radiological=manifest.radiological,
                    cbar_label=_effect_units(manifest),
                    title=f"{manifest.contrast_name}: effect, opacity-coded by evidence",
                ),
                out_dir=plots_dir,
                stem="dual_coded",
                formats=cfg.formats,
            )
            if path:
                blocks.append(
                    html.Figure(
                        title="Effect map · dual-coded",
                        path=path,
                        caption=(
                            "Colour is effect magnitude; opacity is statistical "
                            "evidence. No voxels are hidden."
                        ),
                    )
                )

    table, peaks = (None, ())
    if threshold:
        # Named for what it is. Called `facts`, this shadowed the SummaryFacts
        # parameter of the same name with a list of strings for the rest of the
        # function, so every later panel that recorded a summary fact raised
        # AttributeError -- swallowed by _panel, which made the panel vanish from the
        # document with only a log line to say why.
        smoothness = smoothness_facts(
            stat_img,
            mask_img=mask_img,
            cluster_min_voxels=manifest.cluster_min_voxels,
            threshold=threshold,
        )
        with _panel(f"cluster table for {manifest.contrast_name}"):
            table, peaks = build_cluster_table(
                manifest=manifest,
                out_dir=out_dir,
                threshold=threshold,
                threshold_label=threshold_label,
                extra_facts=smoothness,
                labeller=resolve_labeller(manifest, cfg),
            )

        with _panel(f"thresholded panel for {manifest.contrast_name}"):
            path = _save(
                stat_map_figures.stat_map_mosaic(
                    stat_img,
                    bg_img=background,
                    mask_img=mask_img,
                    threshold=float(threshold),
                    two_sided=manifest.two_sided,
                    radiological=manifest.radiological,
                    title=f"{manifest.contrast_name}: z map, {threshold_label}",
                ),
                out_dir=plots_dir,
                stem="stat_thresholded",
                formats=cfg.formats,
            )
            if path:
                blocks.append(html.Figure(title="Stat map · thresholded", path=path))

        if supports_glass_brain(manifest.space):
            with _panel(f"glass brain for {manifest.contrast_name}"):
                marked = _marker_peaks(peaks)
                path = _save(
                    stat_map_figures.glass_brain(
                        stat_img,
                        mask_img=mask_img,
                        threshold=float(threshold),
                        two_sided=manifest.two_sided,
                        radiological=manifest.radiological,
                        peak_coords=[coord for _label, coord in marked] or None,
                        peak_labels=[label for label, _coord in marked] or None,
                        title=f"{manifest.contrast_name}: glass brain",
                    ),
                    out_dir=plots_dir,
                    stem="glass",
                    formats=cfg.formats,
                )
                if path:
                    blocks.append(
                        html.Figure(
                            title="Glass brain · thresholded",
                            path=path,
                            caption=_marker_caption(len(marked), len(peaks)),
                        )
                    )
        else:
            # Stated rather than simply absent: a panel that vanishes without
            # explanation is indistinguishable from one that failed to render.
            blocks.append(
                html.Note(
                    text=(
                        f"No glass brain for this contrast: the projection is defined "
                        f"only against the MNI schematic, and these results are in "
                        f"{manifest.space} space."
                    )
                )
            )
    else:
        # A mode that resolved no height is a fact about this contrast, not a gap.
        blocks.append(
            html.Note(
                text=(
                    f"No thresholded panels for this contrast: {threshold_label}. The "
                    "calibration panel below still reports where a corrected threshold "
                    "would fall."
                )
            )
        )

    if table is not None:
        blocks.append(table)

    # Both key to the cluster table's rows, so both follow it.
    with _panel(f"peak response for {manifest.contrast_name}"):
        block = build_peak_response_block(
            manifest=manifest, peaks=peaks, out_dir=out_dir, cfg=cfg
        )
        if block is not None:
            blocks.append(block)

    with _panel(f"run consistency for {manifest.contrast_name}"):
        block = build_run_consistency_block(
            manifest=manifest, peaks=peaks, out_dir=out_dir, cfg=cfg
        )
        if block is not None:
            blocks.append(block)

    # Directly beneath the forest panel: that one shows the runs at the chosen peaks,
    # this one shows what they do to the whole map, and the two are read together.
    with _panel(f"run contributions for {manifest.contrast_name}"):
        block = build_run_contribution_block(manifest=manifest, out_dir=out_dir)
        if block is not None:
            blocks.append(block)

    if manifest.effect_map and Path(manifest.effect_map).exists():
        with _panel(f"effect versus evidence for {manifest.contrast_name}"):
            block = _effect_versus_evidence_block(
                manifest=manifest,
                mask_img=mask_img,
                stat_img=stat_img,
                threshold=threshold,
                out_dir=out_dir,
                cfg=cfg,
            )
            if block is not None:
                blocks.append(block)

    if threshold and deriv_root is not None:
        with _panel(f"tissue distribution for {manifest.contrast_name}"):
            block = build_tissue_block(
                manifest=manifest,
                stat_img=stat_img,
                mask_img=mask_img,
                threshold=threshold,
                deriv_root=Path(deriv_root),
                out_dir=out_dir,
                cfg=cfg,
                facts=facts,
            )
            if block is not None:
                blocks.append(block)

    with _panel(f"threshold calibration for {manifest.contrast_name}"):
        context = inference.threshold_context(
            values,
            applied_threshold=threshold,
            fdr_q=float(manifest.fdr_q),
            alpha=0.05,
            two_sided=manifest.two_sided,
            sign_flip=_sign_flip_summary(manifest),
        )

        if context.applied_survivors is not None:
            _note(
                facts,
                f"{prefix}voxels above the threshold",
                survivor_summary(
                    surviving=context.applied_survivors,
                    n_voxels=context.n_voxels,
                    expected_under_fitted_null=(
                        None
                        if context.calibration is None
                        else context.calibration.expected_survivors
                    ),
                ),
            )
        if context.sign_flip is not None:
            _note(
                facts,
                f"{prefix}familywise (run sign-flip)",
                familywise_summary(context.sign_flip),
            )

        path = _save(
            distribution_figures.null_calibration_figure(
                values,
                context=context,
                mask_source=mask_source,
                title=f"{manifest.contrast_name}: threshold calibration",
            ),
            out_dir=plots_dir,
            stem="threshold_calibration",
            dense=False,
            formats=cfg.formats,
        )
        # The counts as a table. They rode in the figure's legend as four sentences of
        # 7-point type occupying a third of the canvas -- a results table drawn in the
        # wrong medium, beside the very lines it described.
        table_html, rows = distribution_figures.threshold_table(context)
        plots_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = plots_dir / "thresholds.tsv"
        tsv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        blocks.append(
            html.Table(
                title="Thresholds and survivors",
                html=table_html,
                tsv_path=tsv_path,
                caption=_threshold_table_caption(context),
            )
        )

        # Beneath the calibration panel: that one shows where the thresholds fall on
        # the map's voxel distribution, this one where the observed maximum falls
        # among the maxima run relabelling produces. Different distributions, and the
        # second is the only one whose null is the data's own.
        if context.sign_flip is not None and manifest.sign_flip_null_tsv:
            with _panel(f"sign-flip null for {manifest.contrast_name}"):
                block = _sign_flip_block(
                    manifest=manifest,
                    summary=context.sign_flip,
                    out_dir=out_dir,
                    cfg=cfg,
                )
                if block is not None:
                    blocks.append(block)

        if path:
            blocks.append(
                html.Figure(
                    title="Threshold calibration",
                    path=path,
                    dense=False,
                    caption=CALIBRATION_CAPTION,
                )
            )

    if not blocks:
        blocks.append(html.Note(text="No panels could be generated for this contrast."))
    return html.Section(
        slug=_slug(manifest),
        title=f"Contrast: {manifest.contrast_name}",
        blocks=tuple(blocks),
    )


#: What the calibration panel says about why the map is over-dispersed.
#:
#: The previous wording blamed "unmodelled autocorrelation", which this study's own
#: data contradicts: median residual ACF(1) runs 0.05-0.07 across runs, far too small
#: to widen a null to sigma 1.51. Naming a cause the report elsewhere measures and
#: refutes is worse than naming none, so this points at the three panels that carry
#: the evidence instead of asserting a mechanism.
CALIBRATION_CAPTION = (
    "Where each threshold in the table above falls on the map's own distribution, "
    "with the fitted null beside the theoretical N(0, 1) the threshold assumes. "
    "Over-dispersion relative to N(0, 1) is a measurement, not an assumption: this "
    "panel states the fitted null's centre and width, the residual autocorrelation "
    "panel states what the residuals do, and the run-level panels state what each run "
    "contributes. Nothing in a thresholded mosaic reveals any of the three."
)


#: How many numbered peak markers a glass brain carries.
#:
#: The markers key the table's strongest rows to the projection. A whole-brain
#: contrast at an uncorrected height yields hundreds of clusters -- 557 on this
#: study's standard-space fit -- and drawing a numbered marker for each covers the
#: map with the labels of clusters nobody reads. Peaks arrive ordered by |z|, so the
#: cap keeps the ones a reader is looking for.
GLASS_BRAIN_MAX_MARKERS = 10


def _marker_peaks(
    peaks: Sequence[Tuple[str, Tuple[float, float, float]]],
) -> Tuple[Tuple[str, Tuple[float, float, float]], ...]:
    """The peaks a glass brain draws markers for."""
    return tuple(peaks[:GLASS_BRAIN_MAX_MARKERS])


def _marker_caption(shown: int, total: int) -> str:
    """Say what the markers key to, and what they leave out."""
    if not total:
        return ""
    if shown >= total:
        return "Markers number the peaks in the cluster table below."
    return (
        f"Markers number the {shown} strongest peaks by |z|, of {total:,} in the "
        f"cluster table below; the map itself is drawn in full."
    )


def survivor_summary(
    *,
    surviving: int,
    n_voxels: int,
    expected_under_fitted_null: Optional[float],
) -> str:
    """State how many voxels cleared the height, against how many were going to.

    The expected count is the point of the line rather than an addition to it. A bare
    "8,463 of 50,626 (16.72%)" reads as a result; stated beside the 8,001 the map's own
    fitted null predicts, the same numbers read as a 5.8% excess. The reader cannot
    recover the second figure from the first, and the summary is where most readers
    stop.

    N(0, 1) is deliberately not the comparison here. It is in the threshold table,
    where there is room to say which null each column assumes; a summary line carrying
    the theoretical expectation alone would overstate the enrichment by roughly
    eightfold on an over-dispersed map.
    """
    line = f"{surviving:,} of {n_voxels:,}"
    if n_voxels:
        line += f" ({surviving / n_voxels:.2%})"
    if expected_under_fitted_null is not None:
        line += (
            f" — {expected_under_fitted_null:,.0f} expected under this map's own "
            f"fitted null"
        )
    return line


def familywise_summary(summary: inference.SignFlipSummary) -> str:
    """State the familywise height, its survivors, and what its p is worth."""
    line = f"|z| > {summary.height:.2f} — {summary.survivors:,} voxels"
    if summary.floor_limited:
        line += (
            f"; global p = {summary.global_p:.3f}, at its floor for "
            f"{summary.n_runs} runs"
        )
    else:
        line += f"; global p = {summary.global_p:.3f}"
    return line


def _sign_flip_summary(
    manifest: ContrastManifest,
) -> Optional[inference.SignFlipSummary]:
    """Build the report-side view of the sign-flip null from the manifest scalars.

    Read from the manifest rather than from the analysis package: importing
    ``run_level`` here would pull the fitting stack into the report's import path,
    which a test forbids and which is what lets a report render from a derivatives
    tree with no model present.
    """
    height = manifest.sign_flip_fwe_height
    if height is None or manifest.sign_flip_n_runs is None:
        return None
    return inference.SignFlipSummary(
        height=float(height),
        survivors=int(manifest.sign_flip_fwe_survivors or 0),
        global_p=float(manifest.sign_flip_global_p or 0.0),
        p_floor=float(
            manifest.sign_flip_p_floor
            if manifest.sign_flip_p_floor is not None
            else inference.sign_flip_p_floor(int(manifest.sign_flip_n_runs))
        ),
        n_runs=int(manifest.sign_flip_n_runs),
        n_patterns=int(manifest.sign_flip_n_patterns or 0),
        observed_max=float(manifest.sign_flip_observed_max or 0.0),
    )


def _threshold_table_caption(context: inference.ThresholdContext) -> str:
    """Describe the table, including what the sign-flip row is and is not worth."""
    caption = (
        "Every count is stated against both nulls where both apply: the count "
        "expected under N(0, 1) is what an over-dispersed map makes look like "
        "enrichment, and the count expected under the map's own fitted null is what "
        "the observed survivors have to exceed to be a finding."
    )

    sign_flip = context.sign_flip
    if sign_flip is None:
        return caption + (
            " No familywise correction over runs is available for this contrast, and "
            "no cluster-extent correction is applied, so none of these heights is "
            "corrected for extent."
        )

    caption += (
        f" The sign-flip row is the one whose null is this data's own: "
        f"{sign_flip.n_patterns} exact sign patterns over {sign_flip.n_runs} runs, "
        f"exchangeable by run, assuming nothing about the distribution the other rows "
        f"assume. Global p = {sign_flip.global_p:.3f}"
    )
    if sign_flip.floor_limited:
        caption += (
            f", which is the smallest value this test can return: the unflipped "
            f"pattern is always a member of the null and always ties the observed "
            f"maximum, so with {sign_flip.n_runs} runs no map-level p below "
            f"{sign_flip.p_floor:.3f} is reachable. The height is unaffected by that "
            f"floor."
        )
    else:
        caption += f" against a floor of {sign_flip.p_floor:.3f}."
    return caption + (
        " No cluster-extent correction is applied; the sign-flip height is "
        "familywise-corrected over voxels, not over extent."
    )


def _sign_flip_block(
    *,
    manifest: ContrastManifest,
    summary: inference.SignFlipSummary,
    out_dir: Path,
    cfg: FmriReportConfig,
) -> Optional[html.Figure]:
    """Draw the enumerated null the analysis wrote, with the observed value on it."""
    import pandas as pd

    from fmri_pipeline.analysis.report.figures import sign_flip as sign_flip_figures

    path = Path(manifest.sign_flip_null_tsv or "")
    if not path.exists():
        return None
    try:
        maxima = pd.read_csv(path, sep="\t")["max_abs_z"].tolist()
    except Exception as exc:
        logger.warning("Could not read the sign-flip null %s (%s)", path.name, exc)
        return None
    if not maxima:
        return None

    saved = _save(
        sign_flip_figures.sign_flip_figure(
            maxima,
            summary=summary,
            title=f"{manifest.contrast_name}: run sign-flip null",
        ),
        out_dir=out_dir / "plots" / _slug(manifest),
        stem="sign_flip_null",
        dense=False,
        formats=cfg.formats,
    )
    if not saved:
        return None
    return html.Figure(
        title="Run sign-flip null",
        path=saved,
        dense=False,
        caption=(
            "Every threshold in the table above assumes a distribution for the map's "
            "voxels; this one assumes only that the runs are exchangeable in sign. "
            "Each mark is one relabelling of which runs count positively, recombined "
            "exactly as the reported map combines all of them, and the value is that "
            "recombination's largest |z| anywhere in the mask. The observed maximum "
            "sitting inside the spread means a map like this one is reachable by "
            "relabelling alone; sitting clear of it means it is not."
        ),
    )


def build_run_contribution_block(
    *,
    manifest: ContrastManifest,
    out_dir: Path,
) -> Optional[html.Table]:
    """What each run contributes to the contrast, as one table.

    A table rather than a figure. Six runs and four numbers each is a table already;
    drawn as a chart it would carry the same values at lower precision, and the
    reading here -- which run is the odd one, by how much -- is a comparison of
    numbers rather than of shapes.

    One table rather than two, because both measurements are keyed by run and a reader
    comparing them across two panels has to hold six rows in mind to do it.

    ``None`` when neither measurement is available, which a single-run contrast and a
    derivatives tree written before these existed both are.
    """
    import pandas as pd

    from fmri_pipeline.analysis.report import contributions

    labels = list(manifest.included_runs)
    if not labels:
        return None

    offsets = None
    if manifest.run_effect_map and Path(manifest.run_effect_map).exists():
        import nibabel as nib

        offsets = contributions.run_offsets(
            nib.load(str(manifest.run_effect_map)), _load_mask(manifest)
        )

    influence = contributions.read_run_influence(manifest.run_influence_tsv)
    if offsets is None and not influence:
        return None

    rows = contributions.contribution_rows(
        run_labels=labels, offsets=offsets, influence=influence
    )
    frame = pd.DataFrame(rows)
    if frame.shape[1] <= 1:
        return None

    plots_dir = out_dir / "plots" / _slug(manifest)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = plots_dir / "run_contributions.tsv"
    frame.to_csv(tsv_path, sep="\t", index=False)

    notes = [
        "Each run's own estimate of this contrast, and what the combined map loses "
        "when that run is left out."
    ]
    if offsets is not None:
        notes.append(
            "The mean effect over the mask is a whole-brain offset: a contrast that "
            "differences two conditions has no reason to carry one, so a non-zero "
            "value is signal shared across the mask rather than anatomy. It is also "
            "the fitted null's centre, and a map centred away from zero produces "
            "large clusters of the offset's sign and shifts survival toward whichever "
            "tissue class the offset reaches most."
        )
    if influence:
        notes.append(
            "Runs are dropped by giving that run a null contrast, so the remaining "
            "runs are combined exactly as the reported map combines all of them. The "
            "forest panel answers a related question at the chosen peaks; a run can "
            "carry the largest peak estimates while a different run moves the map "
            "more."
        )
    notes.append("Runs differing is a measurement, not a fault.")

    return html.Table(
        title="What each run contributes",
        html=for_display(frame).to_html(index=False, border=0, classes=""),
        tsv_path=tsv_path,
        caption=" ".join(notes),
    )


def build_run_consistency_block(
    *,
    manifest: ContrastManifest,
    peaks: Sequence[Tuple[str, Tuple[float, float, float]]],
    out_dir: Path,
    cfg: FmriReportConfig,
) -> Optional[html.Figure]:
    """Each cluster peak's estimate in each run, against the combined estimate.

    Returns ``None`` when the manifest records no run-level maps -- a single-run
    contrast has nothing to compare, and a manifest written before those maps existed
    carries none. Neither is a fault, so neither produces a note.
    """
    if not (manifest.run_effect_map and manifest.run_variance_map):
        return None
    if not peaks:
        return None
    if not (
        Path(manifest.run_effect_map).exists() and Path(manifest.run_variance_map).exists()
    ):
        return None

    import nibabel as nib

    from fmri_pipeline.analysis.report.figures import run_consistency

    estimates = run_consistency.collect_peak_estimates(
        peaks,
        run_effect_img=nib.load(str(manifest.run_effect_map)),
        run_variance_img=nib.load(str(manifest.run_variance_map)),
        combined_effect_img=(
            nib.load(str(manifest.effect_map))
            if manifest.effect_map and Path(manifest.effect_map).exists()
            else None
        ),
        combined_variance_img=(
            nib.load(str(manifest.variance_map))
            if manifest.variance_map and Path(manifest.variance_map).exists()
            else None
        ),
    )
    if not estimates:
        return None

    path = _save(
        run_consistency.peak_forest_figure(
            estimates,
            run_labels=manifest.included_runs,
            effect_units=_unit_name(manifest),
            title=f"{manifest.contrast_name}: per-run estimates at each peak",
        ),
        out_dir=out_dir / "plots" / _slug(manifest),
        stem="run_consistency",
        dense=False,
        formats=cfg.formats,
    )
    if path is None:
        return None
    return html.Figure(
        title="Run consistency at each peak",
        path=path,
        dense=False,
        caption=(
            "A first-level contrast over several runs is a fixed-effects combination, "
            "weighted equally per run: an effect resting on one run and an effect "
            "present in all of them produce the same map and the same cluster table. "
            "Each run's own estimate is shown against the combined one. Runs differing "
            "is a measurement, not a fault — a task with habituation should show "
            "exactly that."
        ),
    )


def build_peak_response_block(
    *,
    manifest: ContrastManifest,
    peaks: Sequence[Tuple[str, Tuple[float, float, float]]],
    out_dir: Path,
    cfg: FmriReportConfig,
) -> Optional[html.Figure]:
    """The response shape at each peak, averaged over the events that drove it.

    Returns ``None`` when the pieces are not on disk -- no design matrices, no TR, no
    recorded contrast weights. Reading every run's 4D data again is the panel's real
    cost, so it is skipped outright rather than half-built.
    """
    if not peaks or not manifest.design_matrices or not manifest.contrast_vector:
        return None

    from fmri_pipeline.analysis.report.figures import timeseries

    responses = timeseries.collect_peak_responses(
        peaks,
        bold_paths=manifest.bold_paths,
        design_paths=manifest.design_matrices,
        contrast_columns=manifest.contrast_columns,
        contrast_vector=manifest.contrast_vector,
        t_r=manifest.t_r,
    )
    if not responses:
        return None

    path = _save(
        timeseries.peak_response_figure(
            responses,
            title=f"{manifest.contrast_name}: response at each peak",
        ),
        out_dir=out_dir / "plots" / _slug(manifest),
        stem="peak_response",
        dense=False,
        formats=cfg.formats,
    )
    if path is None:
        return None
    return html.Figure(
        title="Response shape at each peak",
        path=path,
        dense=False,
        caption=(
            "The signal at each peak, averaged over the onsets of every condition the "
            "contrast weights. A peak driven by the task carries a rise, a plateau, "
            "and an undershoot, and the conditions separate in the direction the "
            "contrast weights them; a peak driven by a few coincident frames carries "
            "neither, and reaches the same z either way. Descriptive only — the map's "
            "z is the test."
        ),
    )


def build_tissue_block(
    *,
    manifest: ContrastManifest,
    stat_img: Any,
    mask_img: Any,
    threshold: float,
    deriv_root: Path,
    out_dir: Path,
    cfg: FmriReportConfig,
    facts: Optional[SummaryFacts] = None,
) -> Optional[html.Figure]:
    """Where in the brain the surviving voxels sit.

    Returns ``None`` when no segmentation is available, which is a property of the
    derivatives rather than a fault -- the carpet declines the same way when it cannot
    order its rows by tissue.
    """
    from fmri_pipeline.analysis.report.assets import discover_plot_assets
    from fmri_pipeline.analysis.report.figures import carpet as carpet_figures
    from fmri_pipeline.analysis.report.figures import tissue as tissue_figures

    assets = discover_plot_assets(
        deriv_root=Path(deriv_root),
        subject=manifest.subject,
        task=manifest.task,
        space=manifest.space,
    )
    codes, source = carpet_figures.resolve_tissue_codes(
        np.asarray(stat_img.get_fdata()).shape[:3],
        assets=assets,
        reference_img=stat_img,
    )
    if codes is None:
        return None

    slices = tissue_figures.split_by_tissue(
        stat_img, tissue_codes=codes, mask_img=mask_img
    )
    if not slices:
        return None

    rates = tissue_figures.enrichment(
        slices, threshold=float(threshold), two_sided=manifest.two_sided
    )
    _note(
        facts,
        f"{manifest.contrast_name}: survival by tissue",
        ", ".join(
            f"{name} {100.0 * share:.1f}%"
            for name, share, *_rest in sorted(rates, key=lambda entry: -entry[1])
        ),
    )

    path = _save(
        tissue_figures.tissue_distribution_figure(
            slices,
            threshold=float(threshold),
            two_sided=manifest.two_sided,
            tissue_source=source,
            title=f"{manifest.contrast_name}: where the result sits",
        ),
        out_dir=out_dir / "plots" / _slug(manifest),
        stem="tissue_distribution",
        dense=False,
        formats=cfg.formats,
    )
    if path is None:
        return None
    return html.Figure(
        title="Where the result sits",
        path=path,
        dense=False,
        caption=(
            "A BOLD effect is a grey-matter phenomenon. Voxels surviving "
            "disproportionately in white matter, in the ventricles, or around the "
            "brain edge indicate residual motion, a coregistration shift, or "
            "pulsatility — all of which reach the cluster table looking like a "
            "result. How much grey-matter enrichment to expect depends on the "
            "contrast and on the segmentation's accuracy at this resolution, so the "
            "rates are reported without a criterion attached."
        ),
    )


def _in_mask(img: Any, mask_img: Any) -> np.ndarray:
    """Flatten a volume over the analysis mask, or over its finite voxels."""
    data = np.asarray(img.get_fdata())
    if mask_img is not None:
        mask = np.asanyarray(mask_img.dataobj).astype(bool)
        if mask.shape == data.shape:
            return data[mask]
    return data[np.isfinite(data)]


def _effect_versus_evidence_block(
    *,
    manifest: ContrastManifest,
    mask_img: Any,
    stat_img: Any,
    threshold: Optional[float],
    out_dir: Path,
    cfg: FmriReportConfig,
) -> Optional[html.Figure]:
    """The panel that separates a large effect from a precisely measured one.

    Both maps are already on disk, so this costs a read and no model fit.
    """
    import nibabel as nib

    effect = _in_mask(nib.load(str(manifest.effect_map)), mask_img)
    statistic = _in_mask(stat_img, mask_img)
    if effect.size != statistic.size:
        logger.info(
            "Effect and statistic maps disagree on voxel count (%d vs %d); skipping "
            "the effect-versus-evidence panel.",
            effect.size,
            statistic.size,
        )
        return None

    error = None
    if manifest.variance_map and Path(manifest.variance_map).exists():
        variance = _in_mask(nib.load(str(manifest.variance_map)), mask_img)
        if variance.size == effect.size:
            error = np.sqrt(np.clip(variance, 0, None))

    path = _save(
        distribution_figures.effect_versus_evidence_figure(
            effect,
            statistic,
            standard_error=error,
            threshold=threshold,
            effect_units=_unit_name(manifest),
            title=f"{manifest.contrast_name}: effect against evidence",
        ),
        out_dir=out_dir / "plots" / _slug(manifest),
        stem="effect_versus_evidence",
        formats=cfg.formats,
    )
    if path is None:
        return None
    return html.Figure(
        title="Effect against evidence",
        path=path,
        dense=True,
        caption=(
            "Every voxel in the analysis mask: its effect against the evidence for "
            "it. A small effect measured precisely clears the threshold; a large one "
            "measured in a dropout region does not. Neither is visible in a z map or "
            "an effect map alone. Colour is the standard error, which is what "
            "separates the two cases."
        ),
    )


def build_diagnostics_section(
    *,
    manifest: ContrastManifest,
    deriv_root: Path,
    out_dir: Path,
    cfg: FmriReportConfig,
    background: Any = None,
) -> html.Section:
    """Build the collapsed diagnostics for one contrast.

    Demoted, not deleted. The unthresholded map is the honest counterpart to the
    thresholded one, and the standard error is how a reader tells a true null from a
    dropout-driven absence of effect -- but neither should compete with the result.

    The threshold calibration panel used to live here as a z histogram. It is now in
    the contrast section: a reader who never opens this block would otherwise read
    every result above at face value.
    """
    import nibabel as nib

    plots_dir = out_dir / "plots" / _slug(manifest)
    stat_img = nib.load(str(manifest.stat_map))
    mask_img = _load_mask(manifest)
    blocks: List[html.Block] = [
        build_model_fit_measurement_block(
            manifest=manifest,
            out_dir=out_dir,
        ),
        build_residual_carpet_block(
            manifest=manifest,
            deriv_root=deriv_root,
            out_dir=out_dir,
            cfg=cfg,
        ),
        build_residual_standard_deviation_block(
            manifest=manifest,
            out_dir=out_dir,
            cfg=cfg,
            background=background,
            mask_img=mask_img,
        ),
        build_residual_autocorrelation_block(
            manifest=manifest,
            out_dir=out_dir,
            cfg=cfg,
        ),
    ]

    if cfg.include_unthresholded:
        with _panel(f"unthresholded panel for {manifest.contrast_name}"):
            path = _save(
                stat_map_figures.stat_map_mosaic(
                    stat_img,
                    bg_img=background,
                    mask_img=mask_img,
                    threshold=None,
                    two_sided=manifest.two_sided,
                    radiological=manifest.radiological,
                    title=f"{manifest.contrast_name}: z map (unthresholded)",
                ),
                out_dir=plots_dir,
                stem="stat_unthresholded",
                formats=cfg.formats,
            )
            if path:
                blocks.append(html.Figure(title="Stat map · unthresholded", path=path))

    if manifest.variance_map and Path(manifest.variance_map).exists():
        with _panel(f"standard error for {manifest.contrast_name}"):
            variance_img = nib.load(str(manifest.variance_map))
            variance = np.asarray(variance_img.get_fdata())
            se_img = nib.Nifti1Image(
                np.sqrt(np.clip(variance, 0, None)),
                variance_img.affine,
                variance_img.header,
            )
            path = _save(
                # A standard error has no negative half, so it goes through the
                # magnitude path. On the signed path it got a symmetric scale, spent
                # half the ramp on values that cannot occur, and rendered flat.
                stat_map_figures.magnitude_mosaic(
                    se_img,
                    bg_img=background,
                    mask_img=mask_img,
                    radiological=manifest.radiological,
                    cbar_label=_error_units(manifest),
                    title=f"{manifest.contrast_name}: standard error",
                ),
                out_dir=plots_dir,
                stem="se",
                formats=cfg.formats,
            )
            if path:
                blocks.append(
                    html.Figure(
                        title="Standard error",
                        path=path,
                        caption=(
                            "Where the model is least certain; distinguishes a true "
                            "null from a dropout-driven absence of effect."
                        ),
                    )
                )

    return html.Section(
        slug=f"{_slug(manifest)}-diagnostics",
        title=f"Diagnostics: {manifest.contrast_name}",
        blocks=tuple(blocks),
        collapsed=True,
    )


def build_model_fit_measurement_block(
    *,
    manifest: ContrastManifest,
    out_dir: Path,
) -> html.Table:
    """Tabulate direct measurements from the exact fitted-model series."""
    if manifest.mask is None:
        raise ValueError("Model-fit measurements require the fitted analysis mask.")

    from fmri_pipeline.analysis.report.figures import model_fit

    measurements = model_fit.summarize_model_fit(
        run_labels=manifest.included_runs,
        residual_paths=manifest.residual_paths,
        predicted_paths=manifest.predicted_paths,
        mask_path=manifest.mask,
    )
    response_units = _unit_name(manifest)
    table_html, _rows = model_fit.model_fit_table(
        measurements,
        response_units=response_units,
    )
    tsv_path = model_fit.write_model_fit_tsv(
        measurements,
        path=(out_dir / "plots" / _slug(manifest) / "model_fit_measurements.tsv"),
        response_units=response_units,
    )
    return html.Table(
        title="Model-fit measurements by run",
        html=table_html,
        tsv_path=tsv_path,
        caption=(
            "Computed voxelwise inside the fitted analysis mask, then reported as "
            "the median and 25th–75th percentiles across voxels. "
            "R² = 1 − Σ(Y − Ŷ)² / Σ(Y − Ȳ)², with Y = Ŷ + e from the recorded "
            "prediction and residual series. Residual SD uses population scaling "
            "(ddof = 0). ACF(1) = Σ(eₜ − ē)(eₜ₊₁ − ē) / Σ(eₜ − ē)². "
            f"Series space: {manifest.model_fit_series_space.replace('-', ' ', 1)}; "
            "only the manifest's retained frames are present. No criterion is applied."
        ),
    )


def build_residual_carpet_block(
    *,
    manifest: ContrastManifest,
    deriv_root: Path,
    out_dir: Path,
    cfg: FmriReportConfig,
) -> html.Figure:
    """Draw exact fitted residuals on their acquired-frame axis."""
    if manifest.mask is None:
        raise ValueError("The residual carpet requires the fitted analysis mask.")

    import nibabel as nib

    from fmri_pipeline.analysis.report.assets import discover_plot_assets
    from fmri_pipeline.analysis.report.figures import model_fit

    mask_image = nib.load(str(manifest.mask))
    mask = np.asanyarray(mask_image.dataobj).astype(bool)
    residual_reference = nib.load(str(manifest.residual_paths[0]))
    assets = discover_plot_assets(
        deriv_root=Path(deriv_root),
        subject=manifest.subject,
        task=manifest.task,
        space=manifest.space,
    )
    tissue_volume, tissue_source = carpet_figures.resolve_tissue_codes(
        tuple(mask_image.shape),
        assets=assets,
        reference_img=residual_reference,
    )
    tissue_codes = None if tissue_volume is None else tissue_volume[mask]
    acquired_frame_counts = tuple(int(nib.load(str(path)).shape[3]) for path in manifest.bold_paths)
    residual_carpet = model_fit.collect_residual_carpet(
        residual_paths=manifest.residual_paths,
        retained_frame_indices=manifest.retained_frame_indices,
        acquired_frame_counts=acquired_frame_counts,
        mask_path=manifest.mask,
        tissue_codes=tissue_codes,
    )
    path = _save(
        carpet_figures.carpet_figure(
            residual_carpet.values,
            tissue_codes=residual_carpet.tissue_codes,
            tissue_source=tissue_source,
            tr=float(manifest.t_r),
            run_boundaries=residual_carpet.run_boundaries,
            run_labels=manifest.included_runs,
            not_retained=residual_carpet.not_retained,
            voxel_source="fitted analysis mask",
            voxel_count_total=residual_carpet.total_voxels,
            title=f"{manifest.contrast_name}: model-response residuals",
        ),
        out_dir=out_dir / "plots" / _slug(manifest),
        stem="residual_carpet",
        formats=cfg.formats,
    )
    if path is None:
        raise RuntimeError("The model-response residual carpet was not written.")
    return html.Figure(
        title="Model-response residual carpet",
        path=path,
        caption=(
            "Residual = Y − Xβ, reconstructed in the unwhitened model-response "
            "space. Each voxel is standardized within each run using only its "
            "retained residual frames. Grey columns locate acquired frames without "
            "a fitted-series value. No criterion is applied."
        ),
    )


def build_residual_standard_deviation_block(
    *,
    manifest: ContrastManifest,
    out_dir: Path,
    cfg: FmriReportConfig,
    background: Any,
    mask_img: Any,
) -> html.Figure:
    """Persist and draw pooled temporal SD of the exact fitted residuals."""
    if manifest.mask is None:
        raise ValueError("The residual SD map requires the fitted analysis mask.")

    import nibabel as nib

    from fmri_pipeline.analysis.report.figures import model_fit

    result = model_fit.pooled_residual_standard_deviation(
        residual_paths=manifest.residual_paths,
        mask_path=manifest.mask,
    )
    artifact_dir = out_dir / "plots" / _slug(manifest)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    map_path = artifact_dir / "residual_standard_deviation.nii.gz"
    nib.save(result.image, str(map_path))
    if not map_path.is_file():
        raise RuntimeError("The pooled residual SD map was not written.")

    series_space = manifest.model_fit_series_space.replace("-", " ", 1)
    response_units = _unit_name(manifest)
    path = _save(
        stat_map_figures.magnitude_mosaic(
            result.image,
            bg_img=background,
            mask_img=mask_img,
            radiological=manifest.radiological,
            cbar_label=f"Residual SD ({response_units})",
            title=f"{manifest.contrast_name}: pooled residual SD",
            extra_provenance=(
                f"{result.retained_frames:,} retained residual samples from "
                f"{result.run_count} run(s)",
                "population SD (ddof = 0)",
                f"series space: {series_space}",
            ),
        ),
        out_dir=artifact_dir,
        stem="residual_standard_deviation",
        formats=cfg.formats,
    )
    if path is None:
        raise RuntimeError("The pooled residual SD figure was not written.")
    return html.Figure(
        title="Pooled residual standard deviation",
        path=path,
        caption=(
            "At each fitted-mask voxel, SD(e) = √[Σ(e − ē)² / N] across all "
            f"{result.retained_frames:,} retained samples from {result.run_count} "
            f"run(s) (ddof = 0). Series space: {series_space}. Raw map: "
            f"{map_path.name}. No criterion is applied."
        ),
    )


def build_residual_autocorrelation_block(
    *,
    manifest: ContrastManifest,
    out_dir: Path,
    cfg: FmriReportConfig,
) -> html.Figure:
    """Measure and draw residual ACF at exact acquired-frame lags."""
    if manifest.mask is None:
        raise ValueError("Residual autocorrelation requires the fitted analysis mask.")

    import nibabel as nib

    from fmri_pipeline.analysis.report.figures import residual_autocorrelation

    acquired_frame_counts = tuple(int(nib.load(str(path)).shape[3]) for path in manifest.bold_paths)
    max_lag_frames = min(
        residual_autocorrelation.DEFAULT_MAX_LAG_FRAMES,
        min(acquired_frame_counts) - 1,
    )
    runs = residual_autocorrelation.collect_residual_autocorrelation(
        run_labels=manifest.included_runs,
        residual_paths=manifest.residual_paths,
        retained_frame_indices=manifest.retained_frame_indices,
        acquired_frame_counts=acquired_frame_counts,
        mask_path=manifest.mask,
        max_lag_frames=max_lag_frames,
    )
    artifact_dir = out_dir / "plots" / _slug(manifest)
    tsv_path = residual_autocorrelation.write_residual_autocorrelation_tsv(
        runs,
        tr=float(manifest.t_r),
        path=artifact_dir / "residual_autocorrelation.tsv",
    )
    path = _save(
        residual_autocorrelation.residual_autocorrelation_figure(
            runs,
            tr=float(manifest.t_r),
            title=f"{manifest.contrast_name}: residual autocorrelation",
        ),
        out_dir=artifact_dir,
        stem="residual_autocorrelation",
        formats=cfg.formats,
        dense=False,
    )
    if path is None:
        raise RuntimeError("The residual-autocorrelation figure was not written.")

    series_space = manifest.model_fit_series_space.replace("-", " ", 1)
    return html.Figure(
        title="Residual autocorrelation by run",
        path=path,
        dense=False,
        caption=(
            "At each fitted-mask voxel and acquired-frame lag k, ACF(k) = "
            "Σ(eₜ − ē)(eₜ₊ₖ − ē) / Σ(eₜ − ē)². A pair is included only when "
            "both retained samples' original acquired-frame indices differ by k. "
            "Lines are voxel medians; bands are the 25th–75th percentiles. "
            f"Series space: {series_space}. Exact plotted values: {tsv_path.name}. "
            "No criterion is applied."
        ),
    )


def _contrast_for_run(
    manifest: ContrastManifest, columns: Sequence[str]
) -> Tuple[Optional[Dict[str, float]], List[str]]:
    """Map the recorded contrast onto one run's design columns.

    Matching by name rather than position, because runs need not share a column
    order -- or even a column set, once a run lacks a condition. Returns the weights
    that landed, and the names of any *non-zero* weight that did not: those are
    reported on the figure, since a contrast quietly missing one of its regressors
    is not the contrast the section claims to show.
    """
    if manifest.contrast_vector is None or not manifest.contrast_columns:
        return None, []

    available = set(columns)
    weights: Dict[str, float] = {}
    dropped: List[str] = []
    for name, weight in zip(manifest.contrast_columns, manifest.contrast_vector):
        if name in available:
            weights[str(name)] = float(weight)
        elif float(weight) != 0.0:
            dropped.append(str(name))
    return (weights or None), dropped


def build_design_section(
    *,
    manifest: ContrastManifest,
    out_dir: Path,
    cfg: FmriReportConfig,
) -> Optional[html.Section]:
    """Build the design matrix and collinearity panels, if a design was recorded.

    The per-run scalars are one table rather than one key-value block per run. Six
    runs produced six stacked blocks of the same seven labels, and comparing a
    condition number across runs meant scrolling between them -- while comparison
    across runs is the entire reason those numbers are reported per run.
    """
    import pandas as pd

    from fmri_pipeline.analysis.report.figures import design as design_figures

    existing = [Path(p) for p in manifest.design_matrices if Path(p).exists()]
    if not existing:
        return None

    plots_dir = out_dir / "plots" / _slug(manifest)
    blocks: List[html.Block] = []
    summaries: List[Any] = []
    summary_labels: List[str] = []
    event_counts: List[Dict[str, int]] = []
    onsets_per_run: List[Dict[str, Any]] = []
    frames: List[Any] = []
    # The conditions this contrast weights, in the order it weights them.
    weighted = [
        str(name)
        for name, weight in zip(
            manifest.contrast_columns, manifest.contrast_vector or ()
        )
        if float(weight) != 0.0
    ]

    for index, path in enumerate(existing):
        run_label = manifest.included_runs[index] if index < len(manifest.included_runs) else f"run-{index + 1:02d}"
        with _panel(f"design matrix for {run_label}"):
            frame = pd.read_csv(path, sep="\t")
            frame = frame.drop(columns=[c for c in ("frame",) if c in frame.columns])
            contrast, dropped = _contrast_for_run(manifest, list(frame.columns))

            saved = _save(
                design_figures.design_matrix_figure(
                    frame,
                    contrast=contrast,
                    tr_seconds=manifest.t_r,
                    run_label=f"{manifest.contrast_name} · {run_label}",
                ),
                out_dir=plots_dir,
                stem=f"design_{run_label}",
                formats=cfg.formats,
            )
            if saved:
                caption = (
                    "Columns grouped by role and scaled individually; the contrast "
                    "actually tested is drawn beneath, on the same axis."
                )
                if dropped:
                    # Silently dropping a weighted regressor would misrepresent the
                    # contrast this run contributed to.
                    caption += (
                        f" Weighted regressors absent from this run's design: "
                        f"{', '.join(dropped)}."
                    )
                blocks.append(
                    html.Figure(
                        title=f"Design matrix · {run_label}",
                        path=saved,
                        caption=caption,
                    )
                )

            # Collinearity is compared *between* runs, so both panels are drawn once
            # over all of them rather than once per run. Six near-identical bar charts
            # made a reader hold six pictures in mind to answer one question, and the
            # spread across runs -- which distinguishes a property of the design from a
            # property of one run -- was never shown at all.
            frames.append(frame)
            summaries.append(design_figures.summarize_design(frame, contrast=contrast))
            summary_labels.append(run_label)
            event_counts.append(design_figures.count_events(frame, weighted))
            onsets_per_run.append(
                {
                    name: design_figures.onset_rows(frame[name].to_numpy(dtype=float))
                    for name in weighted
                    if name in frame.columns
                }
            )

    if len(frames) > 1:
        with _panel(f"variance inflation for {manifest.contrast_name}"):
            saved = _save(
                design_figures.variance_inflation_across_runs_figure(
                    frames,
                    contrast=_contrast_for_run(manifest, list(frames[0].columns))[0],
                    run_labels=summary_labels,
                ),
                out_dir=plots_dir,
                stem="design_vif",
                dense=False,
                formats=cfg.formats,
            )
            if saved:
                blocks.append(
                    html.Figure(
                        title="Variance inflation",
                        path=saved,
                        dense=False,
                        caption=(
                            "Variance inflation per regressor, every run on one axis. "
                            "The regressors this contrast weights are drawn separately "
                            "above, on the same scale: inflation on those is what "
                            "costs the comparison its precision, and inflation "
                            "elsewhere costs it nothing. A regressor inflated in every "
                            "run is a property of the design; one inflated in a single "
                            "run is a property of that run — a lost condition, a "
                            "censored block — and the two call for different "
                            "responses. Reported as a measurement; no cutoff is "
                            "applied."
                        ),
                    )
                )

        with _panel(f"regressor correlation for {manifest.contrast_name}"):
            saved = _save(
                design_figures.regressor_correlation_across_runs_figure(
                    frames, run_labels=summary_labels
                ),
                out_dir=plots_dir,
                stem="design_correlation",
                formats=cfg.formats,
            )
            if saved:
                blocks.append(
                    html.Figure(
                        title="Regressor correlation",
                        path=saved,
                        dense=True,
                        caption=(
                            "The strongest correlation each pair reaches in any run. "
                            "A pair collinear in a single run costs the contrast its "
                            "precision in that run, and averaging across runs would "
                            "dilute exactly that away."
                        ),
                    )
                )

    if summaries:
        # Columns on the table that already reports this design per run, rather than a
        # panel of their own: they are two numbers per run, and the reading is against
        # the event counts and the efficiency already in the same row.
        confounding = [
            design_figures.contrast_confounding(
                frame, _contrast_for_run(manifest, list(frame.columns))[0]
            )
            for frame in frames
        ]
        table_html, rows = design_figures.design_summary_table(
            summaries,
            run_labels=summary_labels,
            event_counts=event_counts,
            condition_names=weighted,
            confounding=confounding,
        )
        tsv_path = plots_dir / "design_summary.tsv"
        plots_dir.mkdir(parents=True, exist_ok=True)
        tsv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        # The raster after the table: the table counts the events, the raster shows
        # where they fell -- and timing is what decides whether two conditions are
        # separable at all, which no count and no condition number reveals.
        if any(onsets_per_run):
            with _panel(f"event raster for {manifest.contrast_name}"):
                raster_path = _save(
                    design_figures.event_raster_figure(
                        onsets_per_run,
                        run_labels=summary_labels,
                        condition_names=weighted,
                        tr_seconds=manifest.t_r,
                        title=f"{manifest.contrast_name}: event timing",
                    ),
                    out_dir=plots_dir,
                    stem="event_raster",
                    dense=False,
                    formats=cfg.formats,
                )
                if raster_path:
                    blocks.insert(
                        0,
                        html.Figure(
                            title="Event timing",
                            path=raster_path,
                            dense=False,
                            caption=(
                                "Conditions that alternate are separable; conditions "
                                "that block against one another share their variance "
                                "with drift, and a run whose conditions are ordered "
                                "rather than interleaved confounds the contrast with "
                                "time-on-task. None of that shows in a count or a "
                                "condition number."
                            ),
                        ),
                    )

        # First, before the per-run figures: it is the comparison across runs, and the
        # figures are what a reader opens after the table sends them to one run.
        blocks.insert(
            0,
            html.Table(
                title="Design summary by run",
                html=table_html,
                tsv_path=tsv_path,
                caption=(
                    "Efficiency is comparable between designs for the same contrast "
                    "and meaningless as an absolute number, so no cutoff is applied "
                    "to it or to anything else here. Event counts are the onsets in "
                    "each run's own convolved regressor, so a trial the model's "
                    "scoping dropped is already absent — and a run whose two counts "
                    "are lopsided estimates the comparison from whichever is smaller, "
                    "while still contributing its own precision to the combination. "
                    "The last two columns quantify what the raster shows: r with "
                    "elapsed time is the correlation of this run's contrast regressor "
                    "with a ramp across the run, so a design whose conditions are "
                    "ordered rather than interleaved confounds the comparison with "
                    "time-on-task; r with drift is the strongest correlation that "
                    "regressor reaches against any single drift column, which is what "
                    "the high-pass basis can absorb. A blocked design is expected to "
                    "correlate with time, so both are reported without a cutoff."
                ),
            ),
        )

    if not blocks:
        return None
    return html.Section(
        slug=f"{_slug(manifest)}-design",
        title=f"Design: {manifest.contrast_name}",
        blocks=tuple(blocks),
        collapsed=True,
    )


def build_signature_section(
    *,
    manifest: ContrastManifest,
    out_dir: Path,
    cfg: FmriReportConfig,
) -> Optional[html.Section]:
    """Signature expression as a dot plot beside the numbers.

    Reads the expression table the analysis run wrote rather than recomputing it:
    expression needs the weight maps and the study's signature configuration, and
    the render path reaches for neither.

    Returns ``None`` when no signatures were configured, which is the stock
    configuration rather than a misconfiguration.
    """
    from fmri_pipeline.analysis.report.figures import signatures as signature_figures

    tsv_path = Path(manifest.stat_map).parent / "signature_expression.tsv"
    points = signature_figures.read_expression_tsv(tsv_path)
    if not points:
        return None

    plots_dir = out_dir / "plots" / _slug(manifest)
    blocks: List[html.Block] = []

    # The table always. It carries every quantity, and it is what a reader quotes.
    #
    # The download links the analysis run's own TSV rather than a copy written here.
    # A copy would carry this table's *formatted* values, and writing it under the
    # source's own name is one path collision away from overwriting the derivative.
    table_html, _rows = signature_figures.signature_table(points)
    blocks.append(
        html.Table(
            title="Signature expression",
            html=table_html,
            tsv_path=tsv_path,
            caption=(
                "Similarity between the unthresholded effect map and each signature's "
                "weight map. Sign carries the interpretation; no threshold is applied. "
                "Cosine is bounded and unitless, so it compares across subjects; the "
                "dot product is in the effect map's own units and does not. Voxels is "
                "the overlap the score was computed over."
            ),
        )
    )

    # The figure only once the ordering is the point. Two signatures on an axis are
    # two numbers the table already gives, printed twice on one screen.
    if len(points) >= signature_figures.MIN_SIGNATURES_FOR_A_PLOT:
        with _panel(f"signature expression for {manifest.contrast_name}"):
            path = _save(
                signature_figures.signature_dot_plot(
                    points, title=f"{manifest.contrast_name}: signature expression"
                ),
                out_dir=plots_dir,
                stem="signature_expression",
                dense=False,
                formats=cfg.formats,
            )
            if path:
                blocks.append(
                    html.Figure(
                        title="Signature expression · ordered",
                        path=path,
                        dense=False,
                        caption=(
                            "The same cosine similarities, ordered. Drawn because "
                            f"{len(points)} signatures are more readily compared as a "
                            "shape than as a column of numbers."
                        ),
                    )
                )

    if not blocks:
        return None
    return html.Section(
        slug=f"{_slug(manifest)}-signatures",
        title=f"Signatures: {manifest.contrast_name}",
        blocks=tuple(blocks),
    )


def build_summary_section(facts: SummaryFacts) -> Optional[html.Section]:
    """The numbers that decide whether this subject's result is usable.

    Every value here is restated from the panel that produced it, and every one of
    them is a measurement. No threshold is applied and nothing is scored: which
    censoring fraction or which tSNR makes a subject usable is a study's decision, and
    a summary that answered it would be inventing the study's criteria.

    Returns ``None`` when no panel reported anything, which happens when every QC
    panel is switched off -- an empty summary block would claim the document had been
    summarised.
    """
    if not facts:
        return None
    return html.Section(
        slug="summary",
        title="At a glance",
        blocks=(
            html.KeyValues(title="Measurements", items=facts.items),
            html.Note(
                text=(
                    "Restated from the panels below, which is where each number's "
                    "provenance is. Measurements only: nothing here is scored against "
                    "a criterion."
                )
            ),
        ),
    )


def build_methods_section(manifests: Sequence[ContrastManifest]) -> html.Section:
    """Record the thresholds and settings actually applied.

    Values and settings only. No verdicts: the report states what was done and what
    was measured, and leaves the judgement to the reader.
    """
    first = manifests[0]
    threshold = (
        f"|z| > {first.z_threshold:.2f}, uncorrected"
        if first.threshold_mode == "z"
        else f"Benjamini-Hochberg FDR, q = {first.fdr_q:.3f}"
        if first.threshold_mode == "fdr"
        else "none"
    )
    items = [
        ("Height threshold", threshold),
        (
            "Multiple comparisons",
            "no familywise correction is applied to the map; each contrast's "
            "calibration panel states where FDR and Bonferroni thresholds fall for "
            "that map and how many voxels survive each"
            if first.threshold_mode != "fdr"
            else "voxelwise FDR across the analysis mask; no cluster-level correction",
        ),
        ("Sidedness", "two-sided" if first.two_sided else "one-sided"),
        ("Confound strategy", first.confound_strategy or "unspecified"),
        (
            "Orientation",
            "radiological (R on left)"
            if first.radiological
            else "neurological (L on left)",
        ),
    ]
    if first.cluster_min_voxels > 0:
        items.append(
            (
                "Cluster extent filter",
                f"clusters smaller than {first.cluster_min_voxels} voxels removed for "
                "display; this is not familywise-error-corrected cluster-level "
                "inference",
            )
        )
    return html.Section(
        slug="methods",
        title="Methods",
        blocks=(html.KeyValues(title="Applied settings", items=tuple(items)),),
    )


def build_configuration_section(
    manifests: Sequence[ContrastManifest],
) -> html.Section:
    """Record the full configuration every contrast in this document was fit under.

    A result shown without the settings that produced it cannot be reproduced from and
    cannot be compared against another study: an HRF basis, a high-pass cutoff, or a
    different resolved confound set changes the numbers, and none of those differences
    is visible in any map. The Methods section states the few settings a reader needs
    to interpret the figures; this states everything needed to run it again.

    Per contrast rather than once, because contrasts of one subject need not share a
    model -- a different formula, a different confound strategy, a different space are
    all configurable per contrast, and collapsing them onto the first would be a
    quiet misreport for the rest.
    """
    blocks: List[html.Block] = []
    for manifest in manifests:
        items: List[Tuple[str, str]] = list(manifest.model_settings)
        items.append(("Smoothing", f"{manifest.smoothing_fwhm:.3g} mm FWHM" if manifest.smoothing_fwhm else "none"))
        items.append(
            (
                "Signal scaling",
                # The mode, not a yes. Voxel-mean and grand-mean scaling both answer
                # "yes" and produce different numbers from the same data, so a bare
                # yes does not let a reader reproduce or compare the effect sizes.
                f"{manifest.signal_scaling_mode} ({_unit_name(manifest)})"
                if manifest.signal_scaling and manifest.signal_scaling_mode
                else "yes, mode not recorded"
                if manifest.signal_scaling
                else "none",
            )
        )
        items.append(("TR", f"{manifest.t_r:.4g} s" if manifest.t_r else "unknown"))
        items.append(
            (
                "Confound columns",
                # Named in full. A count would not let a reader tell one "auto"
                # resolution from another, which is the whole reason for recording it.
                ", ".join(manifest.confound_columns)
                if manifest.confound_columns
                else "none recorded",
            )
        )
        if not items:
            continue
        blocks.append(
            html.KeyValues(title=f"Model · {manifest.contrast_name}", items=tuple(items))
        )

    if not blocks:
        blocks.append(
            html.Note(
                text=(
                    "No model configuration was recorded for these contrasts. Manifests "
                    "written before the configuration was captured carry none; re-run "
                    "the first-level analysis to record it."
                )
            )
        )
    return html.Section(
        slug="configuration",
        title="Configuration",
        blocks=tuple(blocks),
        collapsed=True,
    )


def write_configuration_json(
    manifests: Sequence[ContrastManifest], *, out_path: Path
) -> Path:
    """Write the same configuration as machine-readable JSON beside the report.

    The HTML section is for reading; this is for a script that has to check a cohort
    was fit under one configuration, which is not a question anyone should answer by
    opening ninety reports.
    """
    import json

    payload = {
        "subject": manifests[0].subject,
        "task": manifests[0].task,
        "contrasts": [
            {
                "contrast_name": manifest.contrast_name,
                "space": manifest.space,
                "model_settings": {
                    label: value for label, value in manifest.model_settings
                },
                "confound_columns": list(manifest.confound_columns),
                "smoothing_fwhm_mm": manifest.smoothing_fwhm,
                "signal_scaling": manifest.signal_scaling,
                "signal_scaling_mode": manifest.signal_scaling_mode,
                "t_r_seconds": manifest.t_r,
                "threshold_mode": manifest.threshold_mode,
                "z_threshold": manifest.z_threshold,
                "fdr_q": manifest.fdr_q,
                "cluster_min_voxels": manifest.cluster_min_voxels,
                "two_sided": manifest.two_sided,
                "radiological": manifest.radiological,
                "included_runs": list(manifest.included_runs),
                "excluded_runs": [list(pair) for pair in manifest.excluded_runs],
                "mask_is_analysis_mask": manifest.mask_is_analysis_mask,
            }
            for manifest in manifests
        ],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def build_subject_report(
    *,
    manifests: Sequence[ContrastManifest],
    deriv_root: Path,
    out_path: Path,
    cfg: FmriReportConfig,
) -> Path:
    """Render one document covering every contrast of a subject and task."""
    if not manifests:
        raise ValueError("Cannot build a subject report with no contrasts.")
    validate_manifest_collection(manifests)
    for manifest in manifests:
        validate_manifest_artifacts(manifest)

    first = manifests[0]
    out_path = Path(out_path)
    out_dir = out_path.parent

    # Loaded once for the whole document. Every volume panel is drawn over it, and
    # re-reading a T1w per panel is the most expensive way to get the same image.
    background, background_source = load_background(
        deriv_root=Path(deriv_root), manifest=first
    )

    # Put on the analysis mask's own grid: axis-aligned, and bounded by what was
    # modelled. Nilearn chooses slice positions across the underlay's extent, so an
    # untrimmed whole-head T1w put the vertex and the neck in every mosaic; and this
    # study's T1w is 10.7 degrees oblique against an axis-aligned mask, which rendered
    # the head visibly tilted with black wedges in every tile. The analysis mask is a
    # property of the subject-task rather than of one contrast -- it is the
    # intersection across the runs they all share -- so it is applied once here.
    from fmri_pipeline.analysis.report.figures._display import report_underlay

    background = report_underlay(background, _load_mask(first))

    # Collected as the panels compute their numbers, then placed at the top. A summary
    # that recomputed them would need a second pass over the 4D data for the tSNR
    # alone, and could drift from the panel it claims to restate.
    facts = SummaryFacts()

    sections = [build_header_section(manifests, background_source=background_source)]
    sections.extend(
        build_qc_sections(
            manifests=manifests,
            deriv_root=Path(deriv_root),
            out_dir=out_dir,
            cfg=cfg,
            background=background,
            facts=facts,
        )
    )
    for manifest in manifests:
        sections.append(
            build_contrast_section(
                manifest=manifest,
                out_dir=out_dir,
                cfg=cfg,
                background=background,
                facts=facts,
                deriv_root=Path(deriv_root),
            )
        )
        # The standard-space refit of the same contrast, if one was written, as its
        # own section directly beneath. Adjacent because the comparison between the
        # two fits is the reading; separate because they are two fits, and a section
        # that mixed one fit's coordinates with the other's per-run maps would sample
        # each at the other's millimetres.
        companion = companion_manifest(manifest)
        if companion is not None:
            with _panel(f"standard-space section for {manifest.contrast_name}"):
                # Its own anatomy, discovered against its own space. The subject's
                # native T1w over a standard-space map would put every cluster
                # somewhere it is not, and nothing in the panel would show it.
                companion_background, _source = load_background(
                    deriv_root=Path(deriv_root), manifest=companion
                )
                sections.append(
                    build_contrast_section(
                        manifest=companion,
                        out_dir=out_dir,
                        cfg=cfg,
                        background=companion_background,
                        facts=facts,
                        deriv_root=Path(deriv_root),
                    )
                )
        if cfg.include_design_qc:
            design_section = build_design_section(
                manifest=manifest, out_dir=out_dir, cfg=cfg
            )
            if design_section is not None:
                sections.append(design_section)
        signature_section = build_signature_section(
            manifest=manifest, out_dir=out_dir, cfg=cfg
        )
        if signature_section is not None:
            sections.append(signature_section)
        sections.append(
            build_diagnostics_section(
                manifest=manifest,
                deriv_root=Path(deriv_root),
                out_dir=out_dir,
                cfg=cfg,
                background=background,
            )
        )
    sections.append(build_methods_section(manifests))
    sections.append(build_configuration_section(manifests))

    # Second in the document, after the overview that names the subject. Built last
    # because it restates what the panels above measured.
    summary_section = build_summary_section(facts)
    if summary_section is not None:
        sections.insert(1, summary_section)

    with _panel("configuration sidecar"):
        write_configuration_json(manifests, out_path=out_dir / "config.json")

    document = html.Document(
        title=f"{first.subject} · task-{first.task}",
        subtitle="First-level GLM report (post-fMRIPrep)",
        sections=tuple(sections),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        html.render(document, base_dir=out_dir, embed=cfg.embed_images),
        encoding="utf-8",
    )
    return out_path


__all__ = [
    "build_cluster_table",
    "build_configuration_section",
    "build_contrast_section",
    "build_design_section",
    "build_diagnostics_section",
    "build_header_section",
    "build_methods_section",
    "build_qc_sections",
    "build_summary_section",
    "build_signature_section",
    "build_subject_report",
    "coordinate_space_label",
    "load_background",
    "masked_stat_values",
    "noise_mask",
    "resolve_threshold",
    "smoothness_facts",
    "supports_glass_brain",
    "write_configuration_json",
]
