"""Volume renderings of unsigned magnitude maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from fmri_pipeline.analysis.report.figures._display import figure_of, label_colorbar
from fmri_pipeline.analysis.report.style import (
    MAGNITUDE_CMAP,
    OKABE_ITO,
    annotate_provenance,
    clipped_fraction,
    plot_context,
    robust_upper_limit,
)


@dataclass(frozen=True)
class TsnrResult:
    """Mean tSNR map plus the per-run detail an average would hide."""

    mean_img: nib.Nifti1Image
    per_run_median: Tuple[float, ...]
    frames_used: Tuple[int, ...]
    frames_dropped: Tuple[int, ...]


def compute_tsnr(
    bold_imgs: Sequence[Any],
    *,
    mask_img: Any = None,
    sample_masks: Optional[Sequence[np.ndarray]] = None,
) -> TsnrResult:
    """Return mean temporal SNR across runs, and each run's median separately.

    ``sample_masks`` is one boolean array per run marking the frames to keep. Pass
    the same censoring the GLM used. Non-steady-state frames in particular must be
    excluded: they sit at much higher intensity before longitudinal magnetisation
    saturates, and leaving them in inflates the temporal standard deviation, biasing
    every tSNR value low. fMRIPrep flags them as ``non_steady_state_outlier_XX``.

    Per-run medians are returned alongside the mean map because averaging maps
    across runs makes a single bad run disappear, which is the one thing this
    measurement exists to catch.
    """
    if not bold_imgs:
        raise ValueError("compute_tsnr requires at least one BOLD image.")
    if sample_masks is not None and len(sample_masks) != len(bold_imgs):
        raise ValueError(
            f"Got {len(sample_masks)} sample masks for {len(bold_imgs)} runs."
        )

    mask = None
    if mask_img is not None:
        mask = np.asanyarray(mask_img.dataobj).astype(bool)

    total: Optional[np.ndarray] = None
    affine = None
    medians: List[float] = []
    used: List[int] = []
    dropped: List[int] = []

    for index, img in enumerate(bold_imgs):
        data = np.asanyarray(img.dataobj)
        if data.ndim != 4:
            raise ValueError(f"compute_tsnr requires 4D images, got shape {data.shape}.")
        if affine is None:
            affine = img.affine

        n_frames = data.shape[3]
        if sample_masks is not None:
            keep = np.asarray(sample_masks[index], dtype=bool)
            if keep.size != n_frames:
                raise ValueError(
                    f"Run {index} has {n_frames} frames but its sample mask "
                    f"has {keep.size}."
                )
            data = data[..., keep]
        used.append(int(data.shape[3]))
        dropped.append(int(n_frames - data.shape[3]))
        if data.shape[3] < 2:
            raise ValueError(f"Run {index} has fewer than two frames after censoring.")

        mean = np.mean(data, axis=3)
        std = np.std(data, axis=3)
        # A zero-variance voxel has undefined tSNR, not infinite tSNR.
        tsnr = np.divide(mean, std, out=np.zeros_like(mean, dtype=float), where=std > 0)
        if mask is not None and mask.shape == tsnr.shape:
            tsnr = np.where(mask, tsnr, 0.0)

        inside = tsnr[tsnr > 0]
        medians.append(float(np.median(inside)) if inside.size else 0.0)

        if total is None:
            total = tsnr.astype(float)
        elif total.shape == tsnr.shape:
            total += tsnr
        else:
            raise ValueError(f"Runs disagree on shape: {total.shape} vs {tsnr.shape}.")

    assert total is not None  # guarded by the empty check above
    return TsnrResult(
        mean_img=nib.Nifti1Image(
            (total / float(len(bold_imgs))).astype("float32"), affine
        ),
        per_run_median=tuple(medians),
        frames_used=tuple(used),
        frames_dropped=tuple(dropped),
    )


def per_run_tsnr_figure(
    result: TsnrResult,
    *,
    run_labels: Sequence[str],
    title: str = "",
) -> plt.Figure:
    """Draw each run's median tSNR, with the frames censored from each.

    Exists because the mean map cannot show that one run was bad. Drawn as bars
    against a run axis with every run named, since the labels are what let a reader
    act on the figure -- and the fill colour alone must not carry identity.
    """
    medians = np.asarray(result.per_run_median, dtype=float)
    positions = np.arange(len(medians))
    with plot_context():
        figure, axis = plt.subplots(figsize=(6.5, 0.4 * len(medians) + 1.6))
        axis.barh(positions, medians, color=OKABE_ITO["sky_blue"])
        axis.set_yticks(positions)
        axis.set_yticklabels(list(run_labels)[: len(medians)], fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel("Median tSNR (masked voxels)")
        if title:
            axis.set_title(title)
        for index, (value, drop) in enumerate(zip(medians, result.frames_dropped)):
            note = f"{value:.1f}" + (f"  ({drop} frames censored)" if drop else "")
            axis.annotate(
                note,
                xy=(value, index),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
            )
        annotate_provenance(
            figure,
            [
                f"{len(medians)} run(s)",
                f"{sum(result.frames_used):,} frames used, "
                f"{sum(result.frames_dropped):,} censored",
            ],
        )
        figure.tight_layout()
        return figure


def tsnr_volume(
    result: TsnrResult,
    *,
    bg_img: Any = None,
    title: str = "",
    vmax: Optional[float] = None,
) -> Any:
    """Draw a tSNR map in anatomical orientation.

    Rendered through nilearn so the affine determines what "sagittal" means. Slicing
    the voxel array directly and labelling the panels by anatomy is correct only for
    RAS-canonical data and silently mislabels -- including left/right -- otherwise.
    """
    from nilearn import plotting

    tsnr_img = result.mean_img
    data = np.asarray(tsnr_img.get_fdata())
    positive = data[np.isfinite(data) & (data > 0)]
    resolved_vmax = (
        float(vmax)
        if vmax is not None
        else (robust_upper_limit(positive) if positive.size else 1.0)
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
        label_colorbar(display, "tSNR")
        figure = figure_of(display)
        if positive.size:
            annotate_provenance(
                figure,
                [
                    f"n = {positive.size:,} voxels",
                    f"median tSNR {float(np.median(positive)):.1f}",
                    f"mean of {len(result.per_run_median)} run(s); "
                    f"{sum(result.frames_dropped):,} frames censored",
                    f"colour limit {resolved_vmax:.1f} "
                    f"({clipped_fraction(positive, limit=resolved_vmax):.1%} clipped)",
                ],
            )
        return figure


__all__ = ["TsnrResult", "compute_tsnr", "per_run_tsnr_figure", "tsnr_volume"]
