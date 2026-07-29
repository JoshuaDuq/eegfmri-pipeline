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
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from fmri_pipeline.analysis.plotting_config import FmriReportConfig
from fmri_pipeline.analysis.report import html
from fmri_pipeline.analysis.report.figures import carpet as carpet_figures
from fmri_pipeline.analysis.report.figures import coverage as coverage_figures
from fmri_pipeline.analysis.report.figures import distributions as distribution_figures
from fmri_pipeline.analysis.report.figures import stat_maps as stat_map_figures
from fmri_pipeline.analysis.report.figures import volumes as volume_figures
from fmri_pipeline.analysis.report.manifest import (
    ContrastManifest,
    sample_masks_from_confounds,
)
from fmri_pipeline.analysis.report.style import plot_context, savefig_kwargs

logger = logging.getLogger(__name__)


@contextmanager
def _panel(description: str) -> Iterator[None]:
    """Log and swallow one panel's failure so the document still builds."""
    try:
        yield
    except Exception as exc:
        logger.warning("Failed to generate %s (%s)", description, exc)


def _save(
    figure: Any, *, out_dir: Path, stem: str, formats: Sequence[str]
) -> Optional[Path]:
    """Write a figure and return the path the report should embed.

    Saving happens inside the style context because ``svg.hashsalt`` and
    ``savefig.dpi`` are read at save time, not draw time.
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        primary: Optional[Path] = None
        with plot_context():
            for fmt in formats:
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


def _effect_units(manifest: ContrastManifest) -> str:
    """Name the units of an effect size, or decline to.

    A contrast effect is in arbitrary BOLD units unless the model applied signal
    scaling. Printing "% signal change" on a map that is not in those units invites
    a quantitative reading the number cannot support.
    """
    return "% signal change" if manifest.signal_scaling else "effect (arbitrary BOLD units)"


def _error_units(manifest: ContrastManifest) -> str:
    """Name the units of a standard error.

    The same units as the effect, but the quantity is not the effect. Reusing
    :func:`_effect_units` verbatim labelled the standard-error colourbar "effect",
    which names the wrong map.
    """
    return (
        "standard error (% signal change)"
        if manifest.signal_scaling
        else "standard error (arbitrary BOLD units)"
    )


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


def build_qc_sections(
    *,
    manifests: Sequence[ContrastManifest],
    deriv_root: Path,
    out_dir: Path,
    cfg: FmriReportConfig,
    background: Any = None,
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

    sample_masks = None
    if first.confounds_paths and bold_imgs:
        with _panel("censoring masks"):
            candidate = sample_masks_from_confounds(first.confounds_paths)
            if len(candidate) == len(bold_imgs):
                sample_masks = candidate

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

    if bold_imgs and cfg.include_tsnr_qc:
        with _panel("tSNR"):
            # The mask keeps partial-volume rim voxels out of the median and the
            # colour limit. Without it, `tsnr > 0` admits edge voxels sitting at
            # very low tSNR and drags the reported value down.
            result = volume_figures.compute_tsnr(
                bold_imgs,
                mask_img=_load_mask(first),
                sample_masks=sample_masks,
            )
            path = _save(
                volume_figures.tsnr_volume(
                    result, bg_img=background, title="tSNR (as modelled)"
                ),
                out_dir=qc_dir,
                stem="tsnr_map",
                formats=cfg.formats,
            )
            if path:
                blocks.append(html.Figure(title="tSNR", path=path))
            if len(result.per_run_median) > 1:
                path = _save(
                    volume_figures.per_run_tsnr_figure(
                        result, run_labels=first.included_runs, title="tSNR by run"
                    ),
                    out_dir=qc_dir,
                    stem="tsnr_by_run",
                    formats=cfg.formats,
                )
                if path:
                    blocks.append(
                        html.Figure(
                            title="tSNR by run",
                            path=path,
                            dense=False,
                            caption=(
                                "Shown per run because averaging maps across runs "
                                "hides a single bad run."
                            ),
                        )
                    )

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
    misreport, and nothing in the table lets a reader detect it.
    """
    if str(space or "").strip().lower() == "mni":
        return "coordinates: MNI152 (mm)"
    return f"coordinates: {space} scanner-native (mm), not MNI"


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
    """
    if not {"X", "Y", "Z", "Cluster ID"} <= set(frame.columns):
        return ()

    # Read columns directly rather than iterating rows. The Cluster ID column holds
    # a mix of integers and strings, and `iterrows` upcasts each row to a common
    # dtype -- turning cluster 1 into "1.0", which no longer looks like an integer.
    identifiers = list(frame["Cluster ID"])
    xs, ys, zs = list(frame["X"]), list(frame["Y"]), list(frame["Z"])

    peaks: List[Tuple[str, Tuple[float, float, float]]] = []
    for identifier, x, y, z in zip(identifiers, xs, ys, zs):
        label = _cluster_identifier(identifier)
        if label is None:
            continue
        peaks.append((label, (float(x), float(y), float(z))))
    return tuple(peaks)


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


def smoothness_facts(
    stat_img: Any, *, mask_img: Any, cluster_min_voxels: int
) -> List[str]:
    """Describe the map's spatial smoothness and what it makes an extent filter mean.

    ``cluster_min_voxels`` is configured as a bare count, and a bare count is not
    comparable to anything: the same twenty voxels is a strong constraint on
    unsmoothed 3 mm data and almost none at 8 mm FWHM. Expressing it in resolution
    elements says how many independent bumps of noise a surviving cluster spans.

    Best-effort. A smoothness that cannot be estimated costs these lines and nothing
    else -- an unresolved measurement is not a reason to lose the cluster table.
    """
    from fmri_pipeline.analysis.report.figures import coverage as coverage_figures

    try:
        mask = None
        if mask_img is not None:
            candidate = np.asanyarray(mask_img.dataobj).astype(bool)
            data_shape = np.asarray(stat_img.get_fdata()).shape
            mask = candidate if candidate.shape == data_shape else None
        fwhm = coverage_figures.estimate_fwhm(stat_img, mask=mask)
    except Exception as exc:
        logger.info("Could not estimate smoothness (%s)", exc)
        return []

    facts = [coverage_figures.smoothness_note(fwhm, source="statistic map")]
    if cluster_min_voxels > 0:
        resels = coverage_figures.extent_in_resels(
            cluster_min_voxels, mask_img=stat_img, fwhm=fwhm
        )
        facts.append(f"the {cluster_min_voxels}-voxel extent filter is {resels:.2f} resels")
    return facts


def build_cluster_table(
    *,
    manifest: ContrastManifest,
    out_dir: Path,
    threshold: Optional[float] = None,
    threshold_label: str = "",
    extra_facts: Sequence[str] = (),
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
    frame = reporting.get_clusters_table(
        nib.load(str(manifest.stat_map)),
        stat_threshold=float(threshold),
        cluster_threshold=manifest.cluster_min_voxels or 0,
        two_sided=manifest.two_sided,
    )

    peaks = _cluster_peaks(frame)

    plots_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = plots_dir / "clusters.tsv"
    frame.to_csv(tsv_path, sep="\t", index=False)

    caption_parts = [
        "two-sided" if manifest.two_sided else "one-sided",
        f"height threshold: {threshold_label or f'|z| > {threshold:.2f}'}",
        coordinate_space_label(manifest.space),
    ]
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
            html=frame.to_html(index=False, border=0, classes=""),
            tsv_path=tsv_path,
            caption="; ".join(caption_parts),
        ),
        peaks,
    )


def build_contrast_section(
    *,
    manifest: ContrastManifest,
    out_dir: Path,
    cfg: FmriReportConfig,
    background: Any = None,
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

    if manifest.effect_map and Path(manifest.effect_map).exists() and threshold:
        with _panel(f"dual-coded panel for {manifest.contrast_name}"):
            path = _save(
                stat_map_figures.dual_coded_mosaic(
                    nib.load(str(manifest.effect_map)),
                    stat_img=stat_img,
                    bg_img=background,
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
        facts = smoothness_facts(
            stat_img, mask_img=mask_img, cluster_min_voxels=manifest.cluster_min_voxels
        )
        with _panel(f"cluster table for {manifest.contrast_name}"):
            table, peaks = build_cluster_table(
                manifest=manifest,
                out_dir=out_dir,
                threshold=threshold,
                threshold_label=threshold_label,
                extra_facts=facts,
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
                path = _save(
                    stat_map_figures.glass_brain(
                        stat_img,
                        mask_img=mask_img,
                        threshold=float(threshold),
                        two_sided=manifest.two_sided,
                        radiological=manifest.radiological,
                        peak_coords=[coord for _label, coord in peaks] or None,
                        peak_labels=[label for label, _coord in peaks] or None,
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
                            caption=(
                                "Markers number the peaks in the cluster table below."
                                if peaks
                                else ""
                            ),
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

    with _panel(f"threshold calibration for {manifest.contrast_name}"):
        context = inference.threshold_context(
            values,
            applied_threshold=threshold,
            fdr_q=float(manifest.fdr_q),
            alpha=0.05,
            two_sided=manifest.two_sided,
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
            formats=cfg.formats,
        )
        if path:
            blocks.append(
                html.Figure(
                    title="Threshold calibration",
                    path=path,
                    dense=False,
                    caption=(
                        "The applied height beside the corrected ones, and the map's "
                        "own fitted null beside the theoretical N(0, 1) the threshold "
                        "assumes. A null wider than 1 means the applied threshold is "
                        "weaker than its nominal p value."
                    ),
                )
            )

    if not blocks:
        blocks.append(html.Note(text="No panels could be generated for this contrast."))
    return html.Section(
        slug=_slug(manifest),
        title=f"Contrast: {manifest.contrast_name}",
        blocks=tuple(blocks),
    )


def build_diagnostics_section(
    *,
    manifest: ContrastManifest,
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
    blocks: List[html.Block] = []

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


def _design_summary_items(summary: Any) -> Tuple[Tuple[str, str], ...]:
    """Render a DesignSummary as label/value pairs.

    Efficiency is stated without a threshold: it is comparable between designs for
    the same contrast and meaningless as an absolute number, so a cutoff here would
    be a verdict the pipeline invented.
    """

    def _number(value: Optional[float]) -> str:
        if value is None:
            return "not estimable"
        if not np.isfinite(value):
            return "∞ (exactly collinear)"
        return f"{value:.3g}"

    items = [
        ("Scans", f"{summary.n_scans:,}"),
        ("Regressors", str(summary.n_regressors)),
        ("Condition number", _number(summary.condition_number)),
        ("Largest VIF", _number(summary.max_vif)),
    ]
    if summary.max_vif_regressor:
        items.append(("Most inflated regressor", summary.max_vif_regressor))
    items.append(("Contrast efficiency", _number(summary.efficiency)))
    return tuple(items)


def build_design_section(
    *,
    manifest: ContrastManifest,
    out_dir: Path,
    cfg: FmriReportConfig,
) -> Optional[html.Section]:
    """Build the design matrix and collinearity panels, if a design was recorded."""
    import pandas as pd

    from fmri_pipeline.analysis.report.figures import design as design_figures

    existing = [Path(p) for p in manifest.design_matrices if Path(p).exists()]
    if not existing:
        return None

    plots_dir = out_dir / "plots" / _slug(manifest)
    blocks: List[html.Block] = []

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

            saved = _save(
                design_figures.regressor_correlation_figure(frame, run_label=run_label),
                out_dir=plots_dir,
                stem=f"design_correlation_{run_label}",
                formats=cfg.formats,
            )
            if saved:
                blocks.append(
                    html.Figure(
                        title=f"Regressor correlation · {run_label}",
                        path=saved,
                        dense=False,
                        caption=(
                            "Correlation between design columns. Strong off-diagonal "
                            "structure means the contrast's regressors share variance."
                        ),
                    )
                )

            saved = _save(
                design_figures.variance_inflation_figure(
                    frame, contrast=contrast, run_label=run_label
                ),
                out_dir=plots_dir,
                stem=f"design_vif_{run_label}",
                formats=cfg.formats,
            )
            if saved:
                blocks.append(
                    html.Figure(
                        title=f"Variance inflation · {run_label}",
                        path=saved,
                        dense=False,
                        caption=(
                            "Variance inflation factor per regressor, reported as a "
                            "measurement. No cutoff is applied. The regressors this "
                            "contrast weights are marked, since inflation on those is "
                            "what costs the comparison above its precision."
                        ),
                    )
                )

            summary = design_figures.summarize_design(frame, contrast=contrast)
            blocks.append(
                html.KeyValues(
                    title=f"Design summary · {run_label}",
                    items=_design_summary_items(summary),
                )
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
    with _panel(f"signature expression for {manifest.contrast_name}"):
        path = _save(
            signature_figures.signature_dot_plot(
                points, title=f"{manifest.contrast_name}: signature expression"
            ),
            out_dir=plots_dir,
            stem="signature_expression",
            formats=cfg.formats,
        )
        if path:
            blocks.append(
                html.Figure(
                    title="Signature expression",
                    path=path,
                    dense=False,
                    caption=(
                        "Cosine similarity between the unthresholded effect map and "
                        "each signature's weight map. Sign carries the "
                        "interpretation; no threshold is applied."
                    ),
                )
            )

    blocks.append(
        html.KeyValues(
            title="Expression values",
            items=tuple(
                (
                    point.name,
                    "cosine "
                    + ("n/a" if point.cosine is None else f"{point.cosine:+.3f}")
                    + f" · dot {point.dot:+.3g} · {point.n_voxels:,} voxels",
                )
                for point in points
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

    first = manifests[0]
    out_path = Path(out_path)
    out_dir = out_path.parent

    # Loaded once for the whole document. Every volume panel is drawn over it, and
    # re-reading a T1w per panel is the most expensive way to get the same image.
    background, background_source = load_background(
        deriv_root=Path(deriv_root), manifest=first
    )

    sections = [build_header_section(manifests, background_source=background_source)]
    sections.extend(
        build_qc_sections(
            manifests=manifests,
            deriv_root=Path(deriv_root),
            out_dir=out_dir,
            cfg=cfg,
            background=background,
        )
    )
    for manifest in manifests:
        sections.append(
            build_contrast_section(
                manifest=manifest, out_dir=out_dir, cfg=cfg, background=background
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
                manifest=manifest, out_dir=out_dir, cfg=cfg, background=background
            )
        )
    sections.append(build_methods_section(manifests))

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
    "build_contrast_section",
    "build_design_section",
    "build_diagnostics_section",
    "build_header_section",
    "build_methods_section",
    "build_qc_sections",
    "build_signature_section",
    "build_subject_report",
    "coordinate_space_label",
    "load_background",
    "masked_stat_values",
    "resolve_threshold",
    "smoothness_facts",
    "supports_glass_brain",
]
