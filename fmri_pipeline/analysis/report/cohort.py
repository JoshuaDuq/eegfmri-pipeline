"""Cohort-level HTML report for a fitted Nilearn second-level GLM.

The report consumes only the design, manifest, and persisted statistical maps. It
does not refit the model or choose subjects. Thresholding follows Nilearn's documented
``threshold_stats_img`` procedure with choices fixed in YAML before the maps are read.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fmri_pipeline.analysis.cohort_config import (
    CohortReportConfig,
    CohortThresholdConfig,
    cohort_report_config_from_mapping,
    cohort_threshold_config_from_mapping,
)
from fmri_pipeline.analysis.report import atlas, html
from fmri_pipeline.analysis.report.figures import coverage as coverage_figures
from fmri_pipeline.analysis.report.figures import influence as influence_figures
from fmri_pipeline.analysis.report.figures import residuals as residual_figures
from fmri_pipeline.analysis.report.figures import stat_maps as stat_map_figures
from fmri_pipeline.analysis.report.figures._validation import validated_binary_mask
from fmri_pipeline.analysis.report.style import SIGNED_CMAP, plot_context, save_report_figure
from fmri_pipeline.utils.text import safe_slug

logger = logging.getLogger(__name__)

_MAX_GLASS_BRAIN_PEAK_MARKERS = 6

#: Every familywise correction Nilearn's one permutation run can produce, as
#: ``(saved-map suffix, panel label, peak-table stem)``. Ordered least to most sensitive,
#: which is also the order in which a reader should meet them: voxel max-T asks whether
#: any single voxel survives, and the cluster and TFCE corrections ask a weaker question
#: of a broader object. They share a null, a seed and a permutation count, so a
#: disagreement between them is about the shape of the signal, not about sampling.
_PERMUTATION_CORRECTIONS: tuple[tuple[str, str, str], ...] = (
    ("logp_max_t", "Max-T", "max_t"),
    ("logp_max_size", "Cluster extent", "cluster_size"),
    ("logp_max_mass", "Cluster mass", "cluster_mass"),
    ("logp_max_tfce", "TFCE", "tfce"),
)


@dataclass(frozen=True)
class ResolvedGroupThreshold:
    """Threshold returned by Nilearn and the exact text shown beside it."""

    image: Any
    threshold: float
    label: str
    two_sided: bool


@dataclass(frozen=True)
class ClusterTableResult:
    """Rendered table and the leading primary peaks used to key its overview."""

    block: html.Table | html.Note
    peak_coordinates: tuple[tuple[float, float, float], ...] = ()
    peak_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class CohortReportInputs:
    """Persisted second-level products needed to render one cohort report."""

    task: str
    model: str
    output_name: str
    stat_type: str
    output_dir: Path
    design_matrix: pd.DataFrame
    contrast_spec: Any
    input_manifest: pd.DataFrame
    metadata: Mapping[str, Any]
    saved_maps: Mapping[str, str | Path]
    design_outputs: Mapping[str, str]
    n_permutations: int | None = None
    permutation_two_sided: bool | None = None
    permutation_random_state: int | None = None
    #: Cluster-forming threshold, in p-scale, behind the cluster-extent and cluster-mass
    #: corrections. Those results are answers to this number; reporting them without it
    #: leaves them unreproducible, so a cluster-shaped map without one is refused below.
    permutation_cluster_forming_p: float | None = None
    #: Smoothing applied at second level. Recorded rather than assumed: every
    #: cluster-shaped correction is a function of spatial smoothness, and a report that
    #: says nothing about it is relying on a library default staying what it is today.
    smoothing_fwhm: float | None = None


def _configured_threshold_level(config: CohortThresholdConfig) -> str:
    if config.height_control == "fdr":
        return f"FDR q = {config.alpha:g}"
    if config.height_control == "none":
        return f"uncorrected z = {config.uncorrected_z_threshold:g}"
    return f"{config.height_control.upper()} α = {config.alpha:g}"


def effective_two_sided(*, stat_type: str, configured: bool) -> bool:
    """Return sidedness appropriate for a t or omnibus F statistic."""
    normalized = str(stat_type).strip()
    if normalized not in {"t", "F"}:
        raise ValueError(f"Unsupported second-level stat_type {stat_type!r}.")
    return bool(configured) if normalized == "t" else False


def resolve_group_threshold(
    *,
    stat_img: Any,
    mask_img: Any,
    config: CohortThresholdConfig,
    stat_type: str,
) -> ResolvedGroupThreshold:
    """Threshold a group z map through Nilearn's documented GLM helper."""
    from nilearn.glm import threshold_stats_img

    config.validate()
    two_sided = effective_two_sided(stat_type=stat_type, configured=config.two_sided)
    height_control = None if config.height_control == "none" else config.height_control
    threshold_kwargs = {
        "mask_img": mask_img,
        "alpha": float(config.alpha),
        "height_control": height_control,
        "cluster_threshold": config.cluster_min_voxels,
        "two_sided": two_sided,
    }
    if height_control is None:
        threshold_kwargs["threshold"] = float(config.uncorrected_z_threshold)
    thresholded, threshold = threshold_stats_img(stat_img, **threshold_kwargs)
    if config.height_control == "none":
        label = f"uncorrected |z| > {float(threshold):.3g}"
    else:
        label = f"{_configured_threshold_level(config)}; |z| > {threshold:.3g}"
    if not two_sided:
        label = label.replace("|z|", "z")
    return ResolvedGroupThreshold(
        image=thresholded,
        threshold=float(threshold),
        label=label,
        two_sided=two_sided,
    )


def _required_map(inputs: CohortReportInputs, key: str) -> Path:
    value = inputs.saved_maps.get(key)
    if value is None:
        raise ValueError(f"Cohort report requires the persisted {key!r} map.")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Missing cohort report map: {path}")
    return path


def _optional_map(inputs: CohortReportInputs, key: str) -> Path | None:
    value = inputs.saved_maps.get(key)
    if value is None:
        return None
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Missing cohort report map: {path}")
    return path


def validate_cohort_map(*, image: Any, mask_image: Any, label: str) -> np.ndarray:
    """Require a finite 3D map on the exact fitted-mask voxel grid."""
    mask = validated_binary_mask(mask_image)
    if len(image.shape) != 3 or image.shape != mask_image.shape:
        raise ValueError(f"Cohort {label} and analysis mask must share one 3D shape.")
    if not np.allclose(image.affine, mask_image.affine, rtol=0.0, atol=1e-6):
        raise ValueError(f"Cohort {label} and analysis mask must share the same affine.")
    values = np.asanyarray(image.dataobj)
    if not np.isfinite(values[mask]).all():
        raise ValueError(f"Cohort {label} must be finite inside the analysis mask.")
    return mask


def _design_matrix_figure(design_matrix: pd.DataFrame, *, contrast_spec: Any = None) -> Any:
    """Draw the second-level design with its contrast strip beneath it.

    Drawn through the report's figure layer rather than ``nilearn.plot_design_matrix``
    and ``nilearn.plot_contrast_matrix``, which is what ``_write_design_matrix_files``
    already does for the ``qc/`` copy; this was the last caller of the bare plotters.

    They failed hardest on the design this report is most often run on. A one-sample
    model is a single column of ones, and ``plot_design_matrix(rescale=False)`` hands a
    constant array to ``imshow``, where autoscale collapses to ``vmin == vmax`` and
    every cell normalises to 0.0 -- so the value 1.0 was drawn in the colormap's floor
    colour, the one a reader takes for "empty". The contrast then stood alone in a
    second panel, a lone saturated cell on a symmetric scale. Two flat blocks, neither
    labelled with the value it stood for, and nothing on either saying which regressor
    the weight landed on.

    The figure layer states the scale on a colorbar, writes each weight into the cell
    it belongs to, and shares one x axis between the design and the contrast.
    """
    from fmri_pipeline.analysis.report.figures import design as design_figures
    from fmri_pipeline.analysis.second_level import contrast_weights_for_design

    return design_figures.design_matrix_figure(
        design_matrix,
        contrast=contrast_weights_for_design(
            contrast_spec=contrast_spec, design_columns=list(design_matrix.columns)
        ),
        run_label="second level",
    )


def _design_correlation_figure(design_matrix: pd.DataFrame) -> Any | None:
    """Draw Nilearn's regressor-correlation diagnostic when it stays legible."""
    from nilearn.plotting import plot_design_matrix_correlation

    eligible_columns = [
        column
        for column in design_matrix.columns
        if column not in {"intercept", "constant"} and not column.startswith("drift_")
    ]
    if not 3 <= len(eligible_columns) <= 20:
        return None
    with plot_context():
        image = plot_design_matrix_correlation(
            design_matrix,
            tri="diag",
            cmap="RdBu_r",
            colorbar=True,
            figure=(max(4.6, 0.48 * len(eligible_columns)), max(4.2, 0.44 * len(eligible_columns))),
        )
        image.axes.set_title("Regressor correlation")
        return image.figure


def _standard_error_image(*, variance_img: Any, mask_img: Any, output_path: Path) -> Any:
    import nibabel as nib
    from nilearn.image import math_img

    mask = validate_cohort_map(
        image=variance_img,
        mask_image=mask_img,
        label="effect variance",
    )
    # validate_cohort_map has already proved finiteness inside the mask; only the sign
    # is still open, and a negative variance would pass silently into sqrt as a NaN.
    if np.any(np.asarray(variance_img.dataobj, dtype=float)[mask] < 0):
        raise ValueError("Cohort effect variance must be finite and non-negative in the mask.")
    image = math_img(
        "np.sqrt(np.where(mask > 0, variance, 0))",
        variance=variance_img,
        mask=mask_img,
        copy_header_from="variance",
    )
    # math_img returns the float64 its expression produced. The map is a derived QC
    # artifact, not an input to anything, so it is stored at the precision it is read at.
    image.set_data_dtype(np.float32)
    nib.save(image, output_path)
    return image


def _surface_projection_block(
    *,
    stat_img: Any,
    mask_img: Any,
    threshold: ResolvedGroupThreshold,
    mesh: str,
    plots_dir: Path,
    formats: Sequence[str],
) -> html.Block | None:
    """Project the thresholded map onto cortex, or return None and say why in the log.

    Best-effort, like the atlas: the mesh is a fetched dataset, so a machine reading a
    derivatives tree offline may simply not have it, and Nilearn's response to a mesh it
    cannot find is to go to the network. Losing the panel is the right cost -- a report
    should not acquire a download, or a traceback, as a side effect of drawing a figure.
    """
    try:
        figure = stat_map_figures.surface_projection(
            threshold.image,
            mask_img=mask_img,
            threshold=threshold.threshold,
            two_sided=threshold.two_sided,
            mesh=mesh,
            title="Cortical surface projection",
            cbar_label="z",
        )
    except Exception as exc:
        logger.warning("Could not draw the %s surface panel (%s); it is omitted.", mesh, exc)
        return None
    return html.Figure(
        title="Cortical surface",
        path=save_report_figure(
            figure,
            out_dir=plots_dir,
            stem="z_surface_projection",
            formats=formats,
            dense=True,
        ),
        caption=(
            f"{threshold.label}, sampled between the white and pial surfaces of {mesh}. "
            "A slice cuts across the sheet the signal sits on, so extent along a gyrus is "
            "the one thing the mosaic and glass brain cannot show. Cortex only: "
            "cerebellum, brainstem and subcortex are absent by construction, not because "
            "nothing survived there."
        ),
    )


def _permutation_evidence_blocks(
    *,
    evidence_path: Path,
    key: str,
    label: str,
    tsv_stem: str,
    inputs: CohortReportInputs,
    z_img: Any,
    effect_img: Any,
    mask_img: Any,
    background: Any,
    threshold_config: CohortThresholdConfig,
    report_config: CohortReportConfig,
    labeller: Any,
    plots_dir: Path,
) -> list[html.Block]:
    """Report one corrected -log10(p) map: where it survives, and which peaks.

    Every correction Nilearn returns is a -log10(p FWE) map on the same grid, read at
    the same alpha, so they differ only in the object whose null they took the maximum
    over. One routine therefore reports all of them, and the label is what says which.
    """
    import nibabel as nib
    from nilearn.image import threshold_img

    evidence_img = nib.load(str(evidence_path))
    validate_cohort_map(image=evidence_img, mask_image=mask_img, label=f"{label} evidence")
    logp_data = np.asarray(evidence_img.get_fdata(), dtype=float)
    if not np.isfinite(logp_data).all() or np.any(logp_data < 0):
        raise ValueError(f"{label} permutation evidence must be finite and non-negative.")

    cutoff = -math.log10(float(threshold_config.alpha))
    tail = "two-sided" if inputs.permutation_two_sided else "one-sided"
    minimum_p = 1.0 / (inputs.n_permutations + 1)
    # Nilearn keeps values at the threshold, matching the `p <= alpha` this cutoff was
    # derived from. The evidence map is already one-tailed -log10(p).
    thresholded = threshold_img(evidence_img, threshold=cutoff, two_sided=False, copy_header=True)
    thresholded.set_data_dtype(np.float32)
    nib.save(thresholded, plots_dir / f"permutation_{key}_thresholded.nii.gz")

    if not np.any(np.asarray(thresholded.dataobj, dtype=float) > 0):
        return [
            html.Note(
                text=(
                    f"{label} permutation inference was run, but no voxel survived "
                    f"familywise α={threshold_config.alpha:g}."
                )
            )
        ]
    if effect_img is None:
        raise ValueError(f"{label} peak reporting requires the fitted effect-size map.")

    result = _max_t_peak_table(
        evidence_img=evidence_img,
        z_img=z_img,
        effect_img=effect_img,
        cutoff=cutoff,
        min_distance_mm=threshold_config.min_distance_mm,
        max_rows=report_config.cluster_table_max_rows,
        output_path=plots_dir / f"{tsv_stem}_peaks.tsv",
        labeller=labeller,
        label=label,
    )
    figure_path = save_report_figure(
        stat_map_figures.evidence_ortho(
            thresholded,
            threshold=cutoff,
            cut_coords=result.peak_coordinates[0],
            bg_img=background,
            mask_img=mask_img,
            title=f"{label} evidence at the leading corrected peak",
            cbar_label="−log10(p FWE)",
            extra_provenance=(
                f"α = {threshold_config.alpha:g} (−log10 p ≥ {cutoff:.3g})",
                f"{inputs.n_permutations} permutations ({tail})",
                f"random seed {inputs.permutation_random_state}",
                f"smallest attainable p = {minimum_p:.3g}",
            ),
        ),
        out_dir=plots_dir,
        stem=f"permutation_{key}",
        formats=report_config.formats,
        dense=True,
    )
    return [
        html.Figure(
            title=f"{label} permutation evidence",
            path=figure_path,
            caption=(
                f"{tail.capitalize()} familywise-error-corrected −log10(p) from "
                f"{inputs.n_permutations} permutations; only p ≤ "
                f"{threshold_config.alpha:g} is shown; smallest attainable p = "
                f"{minimum_p:.3g}; random seed {inputs.permutation_random_state}. The "
                "crosshair is centred on the leading corrected peak; the table reports "
                "all leading clusters and supplies effect direction."
            ),
        ),
        result.block,
    ]


def _true_discovery_image(*, z_img: Any, mask_img: Any, threshold: float, alpha: float) -> Any:
    """Nilearn's all-resolutions bound on how much of each cluster is signal.

    Formed at the height this report already displays and tabulates, rather than at a
    separately configured one. The bound is valid simultaneously at every threshold, so
    a forming height is free to choose -- but choosing a different one partitions the
    brain into different clusters, and a bound computed for clusters the reader cannot
    see cannot be read against the rows beside it. Sharing the height is what lets the
    number land on a table row.
    """
    from nilearn.glm import cluster_level_inference

    return cluster_level_inference(
        z_img,
        mask_img=mask_img,
        threshold=float(threshold),
        alpha=float(alpha),
    )


def _true_discovery_declaration(*, threshold: float, alpha: float) -> str:
    # Nilearn forms clusters on `stat_map > threshold`, so this covers the positive tail
    # only -- said out loud because the display threshold is two-sided by default, and a
    # reader would otherwise carry that sidedness across.
    return (
        f"Nilearn cluster_level_inference (all-resolutions inference); "
        f"cluster-forming z > {threshold:.3g}; α = {alpha:g}; positive tail only"
    )


def _with_true_discovery_bound(frame: pd.DataFrame, proportion_img: Any) -> pd.DataFrame:
    """Attach each cluster's bound to its own row.

    Nilearn's proportion map is constant within a cluster, so sampling it at the peak
    returns the bound for the cluster that peak belongs to.

    A negative-tail peak lies outside every bounded cluster and reads 0.0 there. Written
    into the table that zero would say "none of this cluster is real", when what it means
    is that the cluster was never covered -- so those rows are left empty instead.
    """
    bounds = _sample_world_coordinates(proportion_img, _peak_coordinates(frame))
    labelled = frame.copy()
    labelled["true-discovery bound"] = [
        f"{bound:.3f}" if peak > 0 else ""
        for peak, bound in zip(frame["Peak z"].to_numpy(dtype=float), bounds)
    ]
    return labelled


def _true_discovery_proportion_block(
    *,
    proportion_img: Any,
    mask_img: Any,
    background: Any,
    declared: str,
    plots_dir: Path,
    formats: Sequence[str],
) -> html.Block:
    """Show where the bounded clusters are, beside the table that quantifies them.

    The other two inference panels leave a gap between them. Voxelwise FWE names the
    single voxels that survive, and FDR bounds the false share of the whole thresholded
    map; neither says how much of one large cluster is signal. A cohort can therefore
    show a very wide FDR map and a nearly empty FWE map at once, with nothing in the
    report reconciling the two.
    """
    proportion = np.asarray(proportion_img.get_fdata(), dtype=float)
    if not np.any(proportion > 0):
        return html.Note(
            text=f"{declared}. No cluster carried a non-zero bound, so no map is shown."
        )
    return html.Figure(
        title="True-discovery proportion",
        path=save_report_figure(
            stat_map_figures.magnitude_mosaic(
                proportion_img,
                bg_img=background,
                mask_img=mask_img,
                vmax=1.0,
                title="Proportion of true discoveries",
                cbar_label="proportion of cluster that is signal",
                extra_provenance=(declared,),
            ),
            out_dir=plots_dir,
            stem="true_discovery_proportion",
            formats=formats,
            dense=True,
        ),
        caption=(
            f"{declared}. Every voxel of a cluster carries that cluster's lower bound on "
            "the share of its voxels that are true discoveries; the largest bound anywhere "
            f"in this map is {proportion.max():.3f}, and the cluster table above gives the "
            "bound for each cluster it lists. The bound holds simultaneously over every "
            "cluster, so it is not spent by inspecting more of them."
        ),
    )


def _primary_cluster_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one peak row per Nilearn cluster, excluding lettered subpeaks."""
    size_columns = [column for column in frame if str(column).startswith("Cluster Size")]
    if len(size_columns) != 1:
        raise ValueError("Nilearn cluster output must contain one cluster-size column.")
    sizes = frame[size_columns[0]].replace("", np.nan)
    return frame.loc[sizes.notna()].copy()


def _ranked_by_evidence(frame: pd.DataFrame, column: str, *, two_sided: bool) -> pd.DataFrame:
    """Order clusters by the evidence they carry, across both tails.

    Nilearn builds a two-sided table as ``signs = [1, -1]``: the whole positive tail,
    peak-sorted, then the whole negative tail. Taking the head of that concatenation
    shows positive clusters only, whenever the positive tail alone fills the table --
    so the strongest finding in a contrast can be one the displayed table structurally
    cannot reach, while the panels beside it draw that cluster faithfully.

    The TSV keeps Nilearn's own order; this reorders the rows that get shown.
    """
    if not two_sided:
        return frame
    order = frame[column].abs().sort_values(ascending=False, kind="stable").index
    return frame.loc[order]


def _label_peaks(frame: pd.DataFrame, labeller: Any) -> pd.DataFrame:
    """Name the structure each displayed peak falls in, following the subject report."""
    named = labeller.label_all(_peak_coordinates(frame))
    labelled = frame.copy()
    labelled["Region"] = [name or "unlabelled" for name in named]
    return labelled


def _tail_counts(frame: pd.DataFrame, column: str) -> str:
    positive = int((frame[column] > 0).sum())
    return f"{positive} positive, {len(frame) - positive} negative"


def _cluster_identifiers(frame: pd.DataFrame) -> tuple[str, ...]:
    identifiers = []
    for value in frame["Cluster ID"]:
        numeric = float(value) if isinstance(value, (int, float, np.number)) else None
        identifiers.append(
            str(int(numeric)) if numeric is not None and numeric.is_integer() else str(value)
        )
    return tuple(identifiers)


def _peak_coordinates(frame: pd.DataFrame) -> tuple[tuple[float, float, float], ...]:
    required = ("X", "Y", "Z")
    if any(column not in frame for column in required):
        raise ValueError("Nilearn cluster output must contain X, Y, and Z coordinates.")
    coordinates = np.asarray(frame.loc[:, required], dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Nilearn cluster coordinates must be finite.")
    return tuple(tuple(float(value) for value in row) for row in coordinates)


def _cluster_table(
    *,
    stat_img: Any,
    threshold: ResolvedGroupThreshold,
    config: CohortThresholdConfig,
    output_path: Path,
    max_rows: int,
    labeller: Any = None,
    true_discovery_img: Any = None,
) -> ClusterTableResult:
    from nilearn.reporting import get_clusters_table

    frame = get_clusters_table(
        stat_img,
        stat_threshold=threshold.threshold,
        cluster_threshold=config.cluster_min_voxels,
        two_sided=threshold.two_sided,
        min_distance=float(config.min_distance_mm),
    )
    frame = frame.rename(columns={"Peak Stat": "Peak z"})
    frame.to_csv(output_path, sep="\t", index=False)
    if frame.empty:
        return ClusterTableResult(block=html.Note(text=f"No clusters survived {threshold.label}."))

    primary = _primary_cluster_rows(frame)
    display = _ranked_by_evidence(primary, "Peak z", two_sided=threshold.two_sided).head(max_rows)
    caption = f"{threshold.label}; one primary peak per cluster; coordinates are MNI152 millimetres"
    if threshold.two_sided:
        caption += f"; ranked by |z| across both tails ({_tail_counts(primary, 'Peak z')})"
    if true_discovery_img is not None:
        display = _with_true_discovery_bound(display, true_discovery_img)
        caption += (
            "; the true-discovery bound is Nilearn's all-resolutions lower bound on the "
            "share of that cluster which is signal, and covers the positive tail only"
        )
    if labeller is not None:
        display = _label_peaks(display, labeller)
        caption += f"; regions from {labeller.source}"
    if config.cluster_min_voxels:
        caption += (
            f"; clusters smaller than {config.cluster_min_voxels} voxels are hidden "
            "for display, not used as cluster-level corrected inference"
        )
    if len(primary) > len(display):
        caption += f"; showing {len(display)} of {len(primary)} clusters"
    if len(frame) > len(primary):
        caption += f"; the TSV retains all {len(frame)} peak and subpeak rows"
    marker_rows = display.head(_MAX_GLASS_BRAIN_PEAK_MARKERS)
    return ClusterTableResult(
        block=html.Table(
            title="Leading parametric clusters",
            html=display.to_html(index=False, border=0, classes=""),
            tsv_path=output_path,
            caption=caption,
        ),
        peak_coordinates=_peak_coordinates(marker_rows),
        peak_labels=_cluster_identifiers(marker_rows),
    )


def _manifest_map_paths(manifest: pd.DataFrame) -> tuple[Path, ...]:
    candidates = [column for column in ("map_path", "difference_map_path") if column in manifest]
    if len(candidates) != 1:
        raise ValueError(
            "Cohort input concordance requires exactly one fitted-map path column: "
            "map_path or difference_map_path."
        )
    paths = tuple(Path(value) for value in manifest[candidates[0]].astype(str))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing fitted cohort input map: {missing[0]}")
    return paths


def _manifest_map_labels(manifest: pd.DataFrame) -> tuple[str, ...]:
    subjects = manifest["subject"].astype(str)
    if not subjects.duplicated(keep=False).any():
        return tuple(subjects)
    qualifier_columns = [
        column for column in ("condition_label", "contrast_name") if column in manifest
    ]
    if not qualifier_columns:
        raise ValueError(
            "Repeated cohort input rows require condition_label or contrast_name "
            "for unambiguous report labels."
        )
    qualifiers = manifest[qualifier_columns[0]].astype(str)
    labels = tuple(f"{subject} · {qualifier}" for subject, qualifier in zip(subjects, qualifiers))
    if len(labels) != len(set(labels)):
        raise ValueError("Cohort input-map report labels must be unique.")
    return labels


def _input_map_concordance(
    *,
    manifest: pd.DataFrame,
    mask_img: Any,
) -> tuple[Any, pd.DataFrame]:
    """Compute and plot spatial Pearson correlations among fitted input maps."""
    import nibabel as nib
    from nilearn.plotting import plot_matrix

    paths = _manifest_map_paths(manifest)
    if len(paths) < 2:
        raise ValueError("Input-map concordance requires at least two fitted maps.")
    labels = _manifest_map_labels(manifest)
    mask = validated_binary_mask(mask_img)
    map_values = []
    for path in paths:
        image = nib.load(str(path))
        validate_cohort_map(image=image, mask_image=mask_img, label=f"input map {path.name}")
        map_values.append(np.asarray(image.dataobj, dtype=float)[mask])
    values = np.vstack(map_values)
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    if np.any(norms == 0):
        raise ValueError("Input-map concordance is undefined for a spatially constant map.")
    correlations = (centered @ centered.T) / np.outer(norms, norms)
    correlations = np.clip(correlations, -1.0, 1.0)
    frame = pd.DataFrame(correlations, index=labels, columns=labels)

    with plot_context():
        side = max(5.0, min(12.0, 2.4 + 0.32 * len(labels)))
        display = plot_matrix(
            correlations,
            title="Input-map spatial concordance",
            labels=labels if len(labels) <= 40 else False,
            figure=(side, side),
            colorbar=True,
            cmap=SIGNED_CMAP,
            tri="lower",
            grid="0.90",
            # Hierarchically ordered rather than left in manifest order. The panel exists
            # to show heterogeneity, and a participant who agrees with a subgroup rather
            # than with the cohort is a block on the diagonal -- invisible when the rows
            # are in the arbitrary order the manifest happened to list them. The labels
            # travel with the rows, so a row is still attributable.
            reorder="average",
            vmin=-1.0,
            vmax=1.0,
        )
        display.figure.axes[-1].set_ylabel("Pearson r")
        return display.figure, frame


def _permutation_cross_check(
    *, permutation_t_path: Path, parametric_stat_path: Path, mask_img: Any
) -> html.KeyValues:
    """Compare the two t maps the two inference paths each computed independently.

    ``non_parametric_inference`` returns its own t map beside the corrected p-maps, and
    ``SecondLevelModel.compute_contrast`` returns one for the same contrast on the same
    data. They are two implementations of one quantity, so they must agree; a divergence
    means the paths were not handed the same inputs -- a different mask, a different
    subject order, a stale map -- and every panel downstream of either would be wrong
    while looking entirely reasonable.
    """
    import nibabel as nib

    permutation = nib.load(str(permutation_t_path))
    parametric = nib.load(str(parametric_stat_path))
    mask = validate_cohort_map(image=permutation, mask_image=mask_img, label="permutation t")
    validate_cohort_map(image=parametric, mask_image=mask_img, label="parametric t")
    left = np.asarray(permutation.dataobj, dtype=float)[mask]
    right = np.asarray(parametric.dataobj, dtype=float)[mask]
    correlation = float(np.corrcoef(left, right)[0, 1])
    return html.KeyValues(
        title="Permutation cross-check",
        items=(
            ("Correlation of permutation and parametric t", f"{correlation:.6f}"),
            ("Largest absolute difference", f"{float(np.max(np.abs(left - right))):.4g}"),
            ("Voxels compared", f"{left.size:,}"),
        ),
    )


@dataclass(frozen=True)
class ResidualDiagnostics:
    """Second-level residuals, pooled and per participant."""

    normalized: np.ndarray
    per_subject: pd.DataFrame
    r_square: float


def _residual_diagnostics(
    *,
    manifest: pd.DataFrame,
    design_matrix: pd.DataFrame,
    mask_img: Any,
) -> ResidualDiagnostics:
    """Fit the reported design with Nilearn's OLS and keep what it leaves over.

    The report's assumption note already claims "a correctly specified second-level GLM
    with Gaussian errors". At this cohort size that claim carries the whole parametric
    arm, and nothing in the report bore on it. Nilearn's ``OLSModel`` fits the same design
    the contrast was tested on, so the claim can be shown instead of asserted.

    ``normalized_residuals`` is Nilearn's ``e / sqrt(MSE)``; under the stated assumption
    it is standard normal, which is the curve the panel draws it against.
    """
    from nilearn.glm import OLSModel
    from nilearn.maskers import NiftiMasker

    paths = _manifest_map_paths(manifest)
    subjects = _manifest_map_labels(manifest)
    values = NiftiMasker(mask_img=mask_img, standardize=None).fit_transform(
        [str(path) for path in paths]
    )
    results = OLSModel(np.asarray(design_matrix, dtype=float)).fit(values)
    normalized = np.asarray(results.normalized_residuals, dtype=float)
    residuals = np.asarray(results.residuals, dtype=float)
    return ResidualDiagnostics(
        normalized=normalized,
        per_subject=pd.DataFrame(
            {
                "subject": list(subjects),
                "residual RMS": np.sqrt((residuals**2).mean(axis=1)),
            }
        ),
        r_square=float(np.mean(np.asarray(results.r_square, dtype=float))),
    )


def _leave_one_out_influence(
    *,
    manifest: pd.DataFrame,
    design_matrix: pd.DataFrame,
    contrast_spec: Any,
    mask_img: Any,
    threshold_config: CohortThresholdConfig,
    stat_type: str,
) -> pd.DataFrame:
    """Refit the cohort without each participant and rethreshold under the same rule.

    The concordance matrix beside this panel measures spatial agreement, and Pearson
    correlation is scale-invariant: a participant whose map has the cohort's shape at
    several times its amplitude correlates with everyone while carrying the group mean
    on their own. On this cohort the most influential participant sat fourth from the
    bottom on mean pairwise r, and the participant whose removal *grew* the map sat third
    from the bottom -- so the existing diagnostic ranked neither.

    Refitting is what answers the question actually being asked. It costs one extra
    Nilearn fit per participant, and no permutations.
    """
    from nilearn.glm.second_level import SecondLevelModel

    paths = _manifest_map_paths(manifest)
    subjects = _manifest_map_labels(manifest)
    rows: list[dict[str, Any]] = []
    for index, subject in enumerate(subjects):
        remaining = design_matrix.drop(index=design_matrix.index[index]).reset_index(drop=True)
        if np.linalg.matrix_rank(np.asarray(remaining, dtype=float)) < remaining.shape[1]:
            # Dropping the only member of a group leaves a column that no longer
            # identifies anything. That is a fact about the design, not a failure.
            rows.append({"subject": subject, "surviving voxels": None, "estimable": False})
            continue
        model = SecondLevelModel(mask_img=mask_img).fit(
            [str(path) for i, path in enumerate(paths) if i != index],
            design_matrix=remaining,
        )
        z_img = model.compute_contrast(
            second_level_contrast=contrast_spec,
            second_level_stat_type=stat_type,
            output_type="z_score",
        )
        threshold = resolve_group_threshold(
            stat_img=z_img, mask_img=mask_img, config=threshold_config, stat_type=stat_type
        )
        surviving = int(np.count_nonzero(np.asarray(threshold.image.get_fdata())))
        rows.append({"subject": subject, "surviving voxels": surviving, "estimable": True})
    return pd.DataFrame(rows)


def _sample_world_coordinates(image: Any, coordinates: Sequence[Sequence[float]]) -> np.ndarray:
    """Sample image values at the nearest voxels to MNI coordinates."""
    from nibabel.affines import apply_affine

    world = np.asarray(coordinates, dtype=float)
    voxels = np.rint(apply_affine(np.linalg.inv(image.affine), world)).astype(int)
    bounds = np.asarray(image.shape, dtype=int)
    if np.any(voxels < 0) or np.any(voxels >= bounds):
        raise ValueError("A reported Nilearn peak lies outside its source image grid.")
    data = np.asarray(image.dataobj, dtype=float)
    return data[tuple(voxels.T)]


def _constant_valued_clusters(evidence_img: Any, label_map: Any) -> tuple[int, int]:
    """Count surviving clusters whose reported coordinate is a centroid, not a peak.

    A cluster of one voxel is constant too, but its centre of mass is that voxel, so
    there is nothing for a reader to be warned about; counting it would raise the number
    on the max-T panel, whose clusters are mostly single voxels, and spend the warning on
    the one case where the coordinate is exact.
    """
    values = np.asarray(evidence_img.get_fdata(), dtype=float)
    labels = np.asarray(label_map.get_fdata()).astype(int)
    present = [index for index in np.unique(labels) if index != 0]
    constant = sum(
        1
        for index in present
        if (cluster := values[labels == index]).size > 1 and np.unique(cluster).size == 1
    )
    return constant, len(present)


def _max_t_peak_table(
    *,
    evidence_img: Any,
    z_img: Any,
    effect_img: Any,
    cutoff: float,
    min_distance_mm: float,
    max_rows: int,
    output_path: Path,
    labeller: Any = None,
    label: str = "Max-T",
) -> ClusterTableResult:
    """Summarize one corrected -log10(p FWE) map with Nilearn's cluster table."""
    from nilearn.reporting import get_clusters_table

    # Nilearn binarises on a strict `>`, so the cutoff is nudged down to recover the
    # `p <= alpha` it was derived from. The label maps come back from the same call:
    # they are what says whether a reported coordinate is a peak at all.
    frame, label_maps = get_clusters_table(
        evidence_img,
        stat_threshold=float(np.nextafter(cutoff, -np.inf)),
        cluster_threshold=0,
        two_sided=False,
        min_distance=float(min_distance_mm),
        return_label_maps=True,
    )
    if frame.empty:
        raise ValueError(f"{label} evidence survived the cutoff but yielded no Nilearn peaks.")
    coordinates = _peak_coordinates(frame)
    peak_logp = frame["Peak Stat"].to_numpy(dtype=float)
    frame = frame.rename(columns={"Peak Stat": "−log10(p FWE)"})
    frame["p FWE"] = np.power(10.0, -peak_logp)
    frame["signed z"] = _sample_world_coordinates(z_img, coordinates)
    frame["effect estimate"] = _sample_world_coordinates(effect_img, coordinates)
    frame.to_csv(output_path, sep="\t", index=False)

    primary = _primary_cluster_rows(frame)
    display = primary.head(max_rows).copy()
    for column in ("X", "Y", "Z"):
        display[column] = display[column].map(lambda value: f"{float(value):.1f}")
    display["−log10(p FWE)"] = display["−log10(p FWE)"].map(lambda value: f"{float(value):.3g}")
    display["p FWE"] = display["p FWE"].map(lambda value: f"{float(value):.3g}")
    display["signed z"] = display["signed z"].map(lambda value: f"{float(value):.3f}")
    display["effect estimate"] = display["effect estimate"].map(lambda value: f"{float(value):.4g}")
    constant, total = _constant_valued_clusters(evidence_img, label_maps[0])
    caption = (
        f"One primary row per {label} FWE cluster; selection uses only corrected "
        "−log10(p FWE), while signed z and effect estimate supply direction and magnitude; "
        "coordinates are MNI152 millimetres"
    )
    if constant:
        # A cluster-level p-value belongs to the cluster, not to any voxel in it, so
        # cluster-extent and cluster-mass clusters are constant throughout and have no
        # peak to find. Nilearn answers with the centre of mass instead; saying "peak"
        # here would claim a maximum that does not exist, and would hide that the signed
        # z beside it was read at a centroid rather than at the cluster's strongest voxel.
        caption += (
            f"; {constant} of {total} clusters carry one corrected value throughout, so "
            "their coordinate is Nilearn's centre of mass rather than a peak"
        )
    if labeller is not None:
        display = _label_peaks(display, labeller)
        caption += f"; regions from {labeller.source}"
    if len(primary) > len(display):
        caption += f"; showing {len(display)} of {len(primary)} clusters"
    if len(frame) > len(primary):
        caption += f"; the TSV retains all {len(frame)} peak and subpeak rows"
    return ClusterTableResult(
        block=html.Table(
            title=f"{label} corrected peaks",
            html=display.to_html(index=False, border=0, classes=""),
            tsv_path=output_path,
            caption=caption,
        ),
        peak_coordinates=_peak_coordinates(primary.head(max_rows)),
        peak_labels=_cluster_identifiers(primary.head(max_rows)),
    )


def _summary_items(
    inputs: CohortReportInputs, threshold: CohortThresholdConfig
) -> tuple[tuple[str, str], ...]:
    subjects = inputs.input_manifest["subject"].astype(str)
    design = np.asarray(inputs.design_matrix, dtype=float)
    rank = int(np.linalg.matrix_rank(design))
    residual_dof = len(inputs.design_matrix) - rank
    two_sided = effective_two_sided(
        stat_type=inputs.stat_type,
        configured=threshold.two_sided,
    )
    items = [
        ("Task", inputs.task),
        ("Model", inputs.model),
        ("Contrast", inputs.output_name),
        ("Contrast test", f"{inputs.stat_type} (z-score map displayed)"),
        ("Unique subjects", str(subjects.nunique())),
        ("Input maps", str(len(inputs.input_manifest))),
        ("Design rank", f"{rank} / {design.shape[1]} columns"),
        ("Residual degrees of freedom", str(residual_dof)),
        ("Coordinate space", "MNI152NLin2009cAsym"),
        ("Parametric threshold", _configured_threshold_level(threshold)),
        ("Parametric test", "two-sided" if two_sided else "one-sided"),
    ]
    if threshold.cluster_min_voxels:
        items.append(
            (
                "Display-only cluster extent",
                f"{threshold.cluster_min_voxels} voxels",
            )
        )
    input_contrasts = inputs.metadata.get("input_contrast_names")
    if input_contrasts:
        items.append(("Input contrasts", ", ".join(str(value) for value in input_contrasts)))
    if "group_label" in inputs.input_manifest:
        counts = inputs.input_manifest.groupby("group_label")["subject"].nunique()
        items.append(
            ("Groups", ", ".join(f"{label}: n={count}" for label, count in counts.items()))
        )
    items.append(
        (
            "Second-level smoothing",
            "none" if inputs.smoothing_fwhm is None else f"{inputs.smoothing_fwhm:g} mm FWHM",
        )
    )
    if inputs.permutation_cluster_forming_p is not None:
        items.append(
            (
                "Cluster-forming threshold",
                f"p < {inputs.permutation_cluster_forming_p:g} "
                "(cluster-extent and cluster-mass FWE)",
            )
        )
    if inputs.n_permutations is not None:
        items.append(("Max-T permutations", str(inputs.n_permutations)))
        tail = "two-sided" if inputs.permutation_two_sided else "one-sided"
        items.append(("Permutation test", f"{tail} max-T FWE"))
        items.append(("Permutation random seed", str(inputs.permutation_random_state)))
    return tuple(items)


def cohort_assumption_note(inputs: CohortReportInputs) -> str:
    """Describe the GLM and exact Nilearn permutation scheme being reported."""
    base = (
        "Interpretation assumes independent participant units and a correctly "
        "specified second-level GLM with Gaussian errors."
    )
    if inputs.n_permutations is None:
        return base
    from nilearn.glm.contrasts import expression_to_contrast_vector

    if isinstance(inputs.contrast_spec, str):
        contrast = expression_to_contrast_vector(
            inputs.contrast_spec,
            list(inputs.design_matrix.columns),
        )
    else:
        contrast = np.asarray(inputs.contrast_spec, dtype=float)
    if contrast.ndim != 1 or contrast.shape[0] != inputs.design_matrix.shape[1]:
        raise ValueError("Max-T reporting requires one estimable t-contrast vector.")
    tested_variate = np.asarray(inputs.design_matrix, dtype=float) @ contrast
    intercept_test = np.allclose(
        tested_variate,
        tested_variate[0],
        rtol=0.0,
        atol=1e-12,
    )
    if intercept_test and inputs.model == "paired":
        permutation = (
            " Nilearn's intercept max-T test sign-flips the within-participant "
            "difference maps; validity additionally requires exchangeability under "
            "sign flipping (a difference distribution symmetric about zero)."
        )
    elif intercept_test:
        permutation = (
            " Nilearn's intercept max-T test sign-flips the tested input maps; "
            "validity additionally requires exchangeability under sign flipping "
            "(a distribution symmetric about zero)."
        )
    else:
        permutation = (
            " Nilearn's max-T test permutes design rows relative to the imaging "
            "data; validity additionally requires observations to be exchangeable "
            "under the null after accounting for the remaining design variables."
        )
    return base + permutation


def _design_diagnostic_items(inputs: CohortReportInputs) -> tuple[tuple[str, str], ...]:
    labels = {
        "design_condition_number": "Condition number",
        "design_max_vif": "Largest VIF",
        "design_max_vif_regressor": "Regressor with largest VIF",
        "design_contrast_efficiency": "Contrast efficiency",
    }
    return tuple(
        (label, str(inputs.design_outputs[key]))
        for key, label in labels.items()
        if key in inputs.design_outputs
    )


def _report_filename(inputs: CohortReportInputs) -> str:
    model = safe_slug(inputs.model, default="model").replace("_", "-")
    contrast = safe_slug(inputs.output_name, default="contrast").replace("_", "-")
    task = safe_slug(inputs.task, default="task").replace("_", "-")
    return f"group_task-{task}_model-{model}_contrast-{contrast}_report.html"


def build_cohort_report(
    *,
    inputs: CohortReportInputs,
    report_config: CohortReportConfig,
    threshold_config: CohortThresholdConfig,
) -> Path:
    """Render a clean, self-contained report from persisted group derivatives."""
    import nibabel as nib
    from nilearn.datasets import load_mni152_template

    report_config.validate()
    threshold_config.validate()
    if not report_config.enabled or not report_config.html_report:
        raise ValueError("Cohort report generation requires enabled=true and html_report=true.")
    if "subject" not in inputs.input_manifest:
        raise ValueError("Cohort input manifest requires a subject column.")

    output_dir = Path(inputs.output_dir)
    report_dir = output_dir / "report"
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "input_manifest.tsv"
    inputs.input_manifest.to_csv(manifest_path, sep="\t", index=False)

    z_img = nib.load(str(_required_map(inputs, "z_score")))
    mask_img = nib.load(str(_required_map(inputs, "analysis_mask")))
    validate_cohort_map(image=z_img, mask_image=mask_img, label="z score")
    effect_path: Path | None = None
    variance_path: Path | None = None
    effect_img: Any | None = None
    if inputs.stat_type == "t":
        effect_path = _required_map(inputs, "effect_size")
        variance_path = _required_map(inputs, "effect_variance")
    background = load_mni152_template(resolution=2)
    # Second-level coordinates are MNI, so the space gate the subject report has to clear
    # per contrast is satisfied here by construction; a missing or unreadable atlas costs
    # the tables a column, not the report.
    labeller = atlas.load_atlas(
        labels_img=report_config.atlas_labels_img,
        labels_tsv=report_config.atlas_labels_tsv,
    )
    threshold = resolve_group_threshold(
        stat_img=z_img,
        mask_img=mask_img,
        config=threshold_config,
        stat_type=inputs.stat_type,
    )
    # Formed once, at the display height, and shared by the cluster table and its panel:
    # one partition of the brain, so the bound on a row describes the cluster on that row.
    # A threshold Nilearn could not resolve leaves nothing to form clusters from.
    true_discovery_img = (
        _true_discovery_image(
            z_img=z_img,
            mask_img=mask_img,
            threshold=threshold.threshold,
            alpha=threshold_config.alpha,
        )
        if report_config.include_true_discovery_proportion and np.isfinite(threshold.threshold)
        else None
    )

    design_path = save_report_figure(
        _design_matrix_figure(inputs.design_matrix, contrast_spec=inputs.contrast_spec),
        out_dir=plots_dir,
        stem="second_level_design_matrix",
        formats=report_config.formats,
        dense=False,
    )
    correlation_figure = (
        _design_correlation_figure(inputs.design_matrix)
        if report_config.include_design_correlation
        else None
    )
    correlation_path = (
        save_report_figure(
            correlation_figure,
            out_dir=plots_dir,
            stem="second_level_design_correlation",
            formats=report_config.formats,
            dense=False,
        )
        if correlation_figure is not None
        else None
    )
    mask_path = save_report_figure(
        coverage_figures.coverage_figure(
            mask_img,
            bg_img=background,
            extent_note="intersection of valid voxels across every fitted cohort input",
            radiological=False,
            title="Second-level analysis mask",
        ),
        out_dir=plots_dir,
        stem="analysis_mask",
        formats=report_config.formats,
        dense=True,
    )

    consistency_blocks: list[html.Block] = []
    if report_config.include_input_map_concordance:
        concordance_figure, concordance = _input_map_concordance(
            manifest=inputs.input_manifest,
            mask_img=mask_img,
        )
        concordance_tsv = plots_dir / "input_map_concordance.tsv"
        concordance.to_csv(concordance_tsv, sep="\t", index=True, index_label="input map")
        concordance_path = save_report_figure(
            concordance_figure,
            out_dir=plots_dir,
            stem="input_map_concordance",
            formats=report_config.formats,
            dense=False,
        )
        consistency_blocks.append(
            html.Figure(
                title="Input-map spatial concordance",
                path=concordance_path,
                dense=False,
                tsv_path=concordance_tsv,
                caption=(
                    "Pearson correlations between fitted first-level effect maps over "
                    "the exact second-level analysis mask, hierarchically reordered so "
                    "that participants agreeing with one another form a block. This is a "
                    "descriptive heterogeneity diagnostic; it does not alter inclusion or "
                    "inference, and being scale-invariant it cannot see a participant who "
                    "carries the group mean by amplitude -- the influence panel below can."
                ),
            )
        )

    if report_config.include_residual_diagnostics:
        diagnostics = _residual_diagnostics(
            manifest=inputs.input_manifest,
            design_matrix=inputs.design_matrix,
            mask_img=mask_img,
        )
        residual_tsv = plots_dir / "residual_per_subject.tsv"
        diagnostics.per_subject.to_csv(residual_tsv, sep="\t", index=False)
        consistency_blocks.append(
            html.Figure(
                title="Second-level residuals",
                path=save_report_figure(
                    residual_figures.residual_figure(
                        diagnostics.normalized,
                        diagnostics.per_subject,
                        r_square=diagnostics.r_square,
                        rank=int(np.linalg.matrix_rank(np.asarray(inputs.design_matrix, float))),
                        extra_provenance=("Nilearn OLSModel on the reported design",),
                    ),
                    out_dir=plots_dir,
                    stem="second_level_residuals",
                    formats=report_config.formats,
                    dense=False,
                ),
                dense=False,
                tsv_path=residual_tsv,
                caption=(
                    "Residuals of the design this contrast was tested on, from Nilearn's "
                    "OLSModel. The summary above assumes Gaussian errors; this is the panel "
                    "that bears on it. Standardised residuals are Nilearn's e / sqrt(MSE), "
                    "drawn against the standard normal they are assumed to follow, beside "
                    "the residual spread per participant. Descriptive: whether a departure "
                    "matters depends on the design and the question."
                ),
            )
        )

    if report_config.include_leave_one_out_influence:
        influence = _leave_one_out_influence(
            manifest=inputs.input_manifest,
            design_matrix=inputs.design_matrix,
            contrast_spec=inputs.contrast_spec,
            mask_img=mask_img,
            threshold_config=threshold_config,
            stat_type=inputs.stat_type,
        )
        influence_tsv = plots_dir / "leave_one_out_influence.tsv"
        influence.to_csv(influence_tsv, sep="\t", index=False)
        full_survivors = int(np.count_nonzero(np.asarray(threshold.image.get_fdata())))
        consistency_blocks.append(
            html.Figure(
                title="Leave-one-participant-out influence",
                path=save_report_figure(
                    influence_figures.leave_one_out_figure(
                        influence,
                        full_survivors=full_survivors,
                        threshold_label=threshold.label,
                    ),
                    out_dir=plots_dir,
                    stem="leave_one_out_influence",
                    formats=report_config.formats,
                    dense=False,
                ),
                dense=False,
                tsv_path=influence_tsv,
                caption=(
                    "The model refitted without each participant in turn and rethresholded "
                    "under the same rule, counting the voxels that survive. The concordance "
                    "matrix above measures spatial agreement, and Pearson correlation is "
                    "scale-invariant: a participant whose map has the cohort's shape at a "
                    "larger amplitude correlates with everyone while either carrying the "
                    "group mean or inflating its variance. This panel measures that; it is "
                    "descriptive, and it does not remove anyone."
                ),
            )
        )

    design_blocks: list[html.Block] = [
        html.Figure(
            title="Second-level design matrix",
            path=design_path,
            dense=False,
            caption=(
                "Rows follow the input-manifest order. Each column is scaled to its own "
                "peak, so a one-column design reads as one uniform block; the colorbar "
                "gives the scale. The strip beneath carries the weights passed to Nilearn "
                "SecondLevelModel.compute_contrast, on the same axis as the columns they "
                "act on."
            ),
        ),
    ]
    diagnostic_items = _design_diagnostic_items(inputs)
    if correlation_path is not None:
        design_blocks.append(
            html.Figure(
                title="Regressor correlation",
                path=correlation_path,
                dense=False,
                caption=(
                    "Nilearn pairwise correlation of non-constant regressors. "
                    "The unit diagonal lies beyond the off-diagonal colour range. "
                    "This diagnoses collinearity; it does not select or remove regressors."
                ),
            )
        )
    if diagnostic_items:
        design_blocks.append(
            html.KeyValues(title="Estimability diagnostics", items=diagnostic_items)
        )

    estimate_blocks: list[html.Block] = []
    if inputs.stat_type == "t" and effect_path is not None and variance_path is not None:
        effect_img = nib.load(str(effect_path))
        variance_img = nib.load(str(variance_path))
        validate_cohort_map(
            image=effect_img,
            mask_image=mask_img,
            label="effect estimate",
        )
        validate_cohort_map(
            image=variance_img,
            mask_image=mask_img,
            label="effect variance",
        )
        group_effect_path = save_report_figure(
            stat_map_figures.stat_map_mosaic(
                effect_img,
                bg_img=background,
                mask_img=mask_img,
                title="Group contrast estimate",
                cbar_label="effect-size units",
                two_sided=True,
            ),
            out_dir=plots_dir,
            stem="group_effect",
            formats=report_config.formats,
            dense=True,
        )
        standard_error_nifti = plots_dir / "group_standard_error.nii.gz"
        standard_error_img = _standard_error_image(
            variance_img=variance_img,
            mask_img=mask_img,
            output_path=standard_error_nifti,
        )
        standard_error_path = save_report_figure(
            stat_map_figures.magnitude_mosaic(
                standard_error_img,
                bg_img=background,
                mask_img=mask_img,
                title="Standard error of the group contrast",
                cbar_label="standard error",
                extra_provenance=("sqrt(Nilearn effect_variance)",),
            ),
            out_dir=plots_dir,
            stem="group_standard_error",
            formats=report_config.formats,
            dense=True,
        )
        estimate_blocks.extend(
            [
                html.Figure(
                    title="Group effect",
                    path=group_effect_path,
                    caption=(
                        "The estimated second-level contrast in the first-level "
                        "effect-size units. This panel is descriptive, not thresholded evidence."
                    ),
                ),
                html.Figure(
                    title="Standard error",
                    path=standard_error_path,
                    caption=(
                        "Square root of Nilearn's effect-variance map. A larger value "
                        "means less precise cohort estimation."
                    ),
                ),
            ]
        )

    inference_blocks: list[html.Block] = []
    cluster_path = plots_dir / "clusters.tsv"
    thresholded_values = np.asarray(threshold.image.get_fdata(), dtype=float)
    has_parametric_survivors = bool(
        np.any(np.isfinite(thresholded_values) & (thresholded_values != 0))
    )
    if np.isfinite(threshold.threshold) and has_parametric_survivors:
        thresholded_path = save_report_figure(
            stat_map_figures.stat_map_mosaic(
                threshold.image,
                bg_img=background,
                mask_img=mask_img,
                threshold=threshold.threshold,
                two_sided=threshold.two_sided,
                title="Thresholded second-level z statistic",
                cbar_label="z",
            ),
            out_dir=plots_dir,
            stem="z_thresholded",
            formats=report_config.formats,
            dense=True,
        )
        cluster_result = _cluster_table(
            stat_img=z_img,
            threshold=threshold,
            config=threshold_config,
            output_path=cluster_path,
            max_rows=report_config.cluster_table_max_rows,
            labeller=labeller,
            true_discovery_img=true_discovery_img,
        )
        glass_path = save_report_figure(
            stat_map_figures.glass_brain(
                threshold.image,
                mask_img=mask_img,
                peak_coords=cluster_result.peak_coordinates,
                peak_labels=cluster_result.peak_labels,
                threshold=threshold.threshold,
                two_sided=threshold.two_sided,
                title="Whole-brain thresholded overview",
                cbar_label="z",
            ),
            out_dir=plots_dir,
            stem="z_glass_brain",
            formats=report_config.formats,
            dense=True,
        )
        inference_blocks.extend(
            [
                html.Figure(
                    title="Thresholded z statistic",
                    path=thresholded_path,
                    caption=(
                        f"{threshold.label}. The procedure and level were fixed in YAML; "
                        "the report does not tune them to this cohort."
                    ),
                ),
                html.Figure(
                    title="Glass-brain overview",
                    path=glass_path,
                    caption=(
                        "Maximum-intensity projection for spatial overview. Numbered "
                        f"markers identify the first {len(cluster_result.peak_coordinates)} "
                        "primary peaks in the table; use the slice mosaic and coordinates "
                        "for anatomical interpretation."
                    ),
                ),
                cluster_result.block,
            ]
        )
        if report_config.surface_mesh:
            surface_block = _surface_projection_block(
                stat_img=z_img,
                mask_img=mask_img,
                threshold=threshold,
                mesh=report_config.surface_mesh,
                plots_dir=plots_dir,
                formats=report_config.formats,
            )
            if surface_block is not None:
                inference_blocks.append(surface_block)
    else:
        pd.DataFrame().to_csv(cluster_path, sep="\t", index=False)
        inference_blocks.append(
            html.Note(text=f"No voxel survived {threshold.label}; no thresholded map is shown.")
        )

    if true_discovery_img is not None:
        inference_blocks.append(
            _true_discovery_proportion_block(
                proportion_img=true_discovery_img,
                mask_img=mask_img,
                background=background,
                declared=_true_discovery_declaration(
                    threshold=threshold.threshold, alpha=threshold_config.alpha
                ),
                plots_dir=plots_dir,
                formats=report_config.formats,
            )
        )

    permutation_maps = [
        (key, label, stem)
        for key, label, stem in _PERMUTATION_CORRECTIONS
        if _optional_map(inputs, f"permutation_{key}") is not None
    ]
    if permutation_maps:
        if type(inputs.n_permutations) is not int or inputs.n_permutations <= 0:
            raise ValueError("Max-T report provenance requires a positive integer n_permutations.")
        if type(inputs.permutation_two_sided) is not bool:
            raise TypeError("Max-T report provenance requires boolean permutation_two_sided.")
        if type(inputs.permutation_random_state) is not int or inputs.permutation_random_state < 0:
            raise TypeError(
                "Max-T report provenance requires a non-negative integer permutation_random_state."
            )
        cluster_shaped = {"logp_max_size", "logp_max_mass"}
        if inputs.permutation_cluster_forming_p is None and any(
            key in cluster_shaped for key, _label, _stem in permutation_maps
        ):
            raise ValueError(
                "Cluster-extent and cluster-mass evidence are defined by the "
                "cluster-forming threshold they were computed at; reporting them "
                "requires permutation_cluster_forming_p."
            )
        for key, label, stem in permutation_maps:
            inference_blocks.extend(
                _permutation_evidence_blocks(
                    evidence_path=_optional_map(inputs, f"permutation_{key}"),
                    key=key,
                    label=label,
                    tsv_stem=stem,
                    inputs=inputs,
                    z_img=z_img,
                    effect_img=effect_img,
                    mask_img=mask_img,
                    background=background,
                    threshold_config=threshold_config,
                    report_config=report_config,
                    labeller=labeller,
                    plots_dir=plots_dir,
                )
            )

    permutation_t_path = _optional_map(inputs, "permutation_t")
    parametric_stat_path = _optional_map(inputs, "stat")
    if permutation_t_path is not None and parametric_stat_path is not None:
        inference_blocks.append(
            _permutation_cross_check(
                permutation_t_path=permutation_t_path,
                parametric_stat_path=parametric_stat_path,
                mask_img=mask_img,
            )
        )

    if report_config.include_interactive_viewer:
        from nilearn.plotting import view_img

        viewer = view_img(
            z_img,
            threshold=threshold.threshold if np.isfinite(threshold.threshold) else None,
            bg_img=background,
            black_bg=False,
            cmap=SIGNED_CMAP,
            symmetric_cmap=True,
            title=None,
        )
        inference_blocks.append(
            html.Embed(
                title="Interactive volume viewer",
                html=viewer.get_iframe(),
                caption=(
                    f"The unthresholded z map, displayed from {threshold.label}. Nilearn's "
                    "view_img with its data and script inlined, so it works offline and "
                    "travels with the file. Every panel above shows the slices it was told "
                    "to; this is the one that reaches a cluster the fixed cuts miss."
                ),
            )
        )

    diagnostic_blocks: list[html.Block] = []
    if report_config.include_unthresholded:
        unthresholded_path = save_report_figure(
            stat_map_figures.stat_map_mosaic(
                z_img,
                bg_img=background,
                mask_img=mask_img,
                threshold=None,
                two_sided=threshold.two_sided,
                title="Unthresholded second-level z statistic",
                cbar_label="z",
            ),
            out_dir=plots_dir,
            stem="z_unthresholded",
            formats=report_config.formats,
            dense=True,
        )
        diagnostic_blocks.append(
            html.Figure(
                title="Unthresholded z statistic",
                path=unthresholded_path,
                caption=(
                    "Diagnostic view of the complete statistic field. It is not "
                    "presented as multiplicity-controlled evidence."
                ),
            )
        )

    document = html.Document(
        title="Cohort fMRI report",
        subtitle=(
            f"task-{inputs.task} · {inputs.model} second-level model · "
            f"contrast {inputs.output_name}"
        ),
        sections=(
            html.Section(
                slug="summary",
                title="Cohort and model",
                blocks=(
                    html.KeyValues(
                        title="Analysis summary",
                        items=_summary_items(inputs, threshold_config),
                    ),
                    html.Note(text=cohort_assumption_note(inputs)),
                    html.Table(
                        title="Exact input manifest",
                        tsv_path=manifest_path,
                        caption=(
                            "One row per fitted input map. Subject and map order are "
                            "the order supplied to Nilearn."
                        ),
                    ),
                ),
            ),
            html.Section(slug="design", title="Design and contrast", blocks=tuple(design_blocks)),
            html.Section(
                slug="coverage",
                title="Spatial coverage",
                blocks=(
                    html.Figure(
                        title="Analysis mask",
                        path=mask_path,
                        caption="Voxels outside this mask were not tested at second level.",
                    ),
                ),
            ),
            html.Section(
                slug="estimates",
                title="Effect and uncertainty",
                blocks=tuple(estimate_blocks)
                or (
                    html.Note(
                        text=(
                            "An omnibus F contrast has no single signed effect or "
                            "standard-error map; those panels are intentionally omitted."
                        )
                    ),
                ),
            ),
            *(
                (
                    html.Section(
                        slug="consistency",
                        title="Cohort consistency",
                        blocks=tuple(consistency_blocks),
                    ),
                )
                if consistency_blocks
                else ()
            ),
            html.Section(
                slug="inference",
                title="Statistical inference",
                blocks=tuple(inference_blocks),
            ),
            *(
                (
                    html.Section(
                        slug="diagnostics",
                        title="Unthresholded diagnostic",
                        blocks=tuple(diagnostic_blocks),
                        collapsed=True,
                    ),
                )
                if diagnostic_blocks
                else ()
            ),
        ),
    )
    report_path = report_dir / _report_filename(inputs)
    report_path.write_text(
        html.render(document, base_dir=report_dir, embed=report_config.embed_images),
        encoding="utf-8",
    )
    return report_path


__all__ = [
    "CohortReportConfig",
    "CohortReportInputs",
    "CohortThresholdConfig",
    "ResolvedGroupThreshold",
    "build_cohort_report",
    "cohort_report_config_from_mapping",
    "cohort_assumption_note",
    "cohort_threshold_config_from_mapping",
    "effective_two_sided",
    "resolve_group_threshold",
    "validate_cohort_map",
]
