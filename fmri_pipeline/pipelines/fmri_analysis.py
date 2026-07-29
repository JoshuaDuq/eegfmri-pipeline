"""fMRI first-level analysis pipeline (GLM + contrasts).

This pipeline computes subject-level (first-level) contrasts between conditions
from BIDS events files using nilearn's FirstLevelModel multi-run support.

Outputs are written under:
  <deriv_root>/sub-<ID>/fmri/first_level/<task>/<contrast_name>/
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, Optional

from eeg_pipeline.pipelines.base import PipelineBase
from fmri_pipeline.utils.signature_paths import discover_signature_root_and_specs
from fmri_pipeline.utils.text import safe_slug


def _safe_slug(text: str, default: str = "contrast") -> str:
    """Backward-compatible alias for tests/imports."""
    return safe_slug(text, default=default)


def _contrast_hash(cfg: Any) -> str:
    """Stable hash of key contrast settings for cache-friendly filenames."""
    try:
        payload = asdict(cfg)
    except Exception:
        payload = {"repr": repr(cfg)}
    raw = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:8]


def _optional_positive_float(value: Any) -> Optional[float]:
    """Float when the value is a usable positive number, else None.

    Local rather than imported from ``contrast_builder``: reaching across for a
    private helper couples the two modules and breaks every test that stubs the
    contrast builder out.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _contrast_arg_for_model_runs(flm: Any, contrast_def: Any) -> Any:
    """Provide an explicit per-run contrast list for multi-run models."""
    if isinstance(contrast_def, (list, tuple, dict)):
        return contrast_def
    n_runs = len(getattr(flm, "design_matrices_", []) or [])
    if n_runs > 1:
        return [contrast_def] * n_runs
    return contrast_def


def _contrast_vector_for_design(
    *, glm_result: Any, contrast_def: Any
) -> tuple[Optional[list[float]], list[str]]:
    """Expand a contrast expression into weights against the design's own columns.

    The report draws the contrast as a strip beneath the design matrix, and computes
    the design's efficiency for it. Both need numbers per column; the manifest carried
    only the expression string, so neither could ever be produced. Matching is by
    column name downstream, so the two lists are returned together.

    Best-effort: an expression nilearn cannot parse against these columns costs the
    contrast strip and nothing else, and the map is already on disk by now.
    """
    import logging

    import numpy as np

    logger = logging.getLogger(__name__)
    design_matrices = getattr(getattr(glm_result, "flm", None), "design_matrices_", None)
    if not design_matrices:
        return None, []

    columns = [str(c) for c in design_matrices[0].columns]
    if not isinstance(contrast_def, str):
        # An explicit vector, already aligned to the design nilearn was given.
        try:
            values = np.asarray(contrast_def, dtype=float).ravel()
        except (TypeError, ValueError):
            return None, columns
        return (list(map(float, values)), columns) if values.size == len(columns) else (None, columns)

    try:
        from nilearn.glm.contrasts import expression_to_contrast_vector

        vector = expression_to_contrast_vector(contrast_def, columns)
    except Exception as exc:
        logger.info(
            "Could not expand contrast %r against the design columns (%s); the "
            "report will show the design matrix without its contrast strip.",
            contrast_def,
            exc,
        )
        return None, columns
    return [float(v) for v in np.asarray(vector).ravel()], columns


class FmriAnalysisPipeline(PipelineBase):
    """Compute first-level fMRI contrasts for each subject."""

    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="fmri_analysis", config=config)

    def _discover_signature_root_and_specs(self) -> tuple[Optional[Path], list]:
        """
        Resolve signature weight-map root directory and spec list from config.

        The root is config: paths.signature_dir. It is required when
        paths.signature_maps is non-empty.
        Specs are read from config: paths.signature_maps (list of {name, path} dicts).
        """
        return discover_signature_root_and_specs(self.config, self.deriv_root)

    def _discover_plot_assets(
        self,
        *,
        sub_label: str,
        task: str,
        space: str,
    ) -> tuple[Optional[Path], Optional[Path]]:
        """
        Best-effort discovery of a background image + brain mask for plotting.

        Preference order:
        - Brain mask: use fMRIPrep func-space brain masks (match BOLD/stat resolution).
        - Background: prefer anatomical preproc T1w (nice-looking) when available, else use func boldref.
        """
        deriv_root = self.deriv_root
        space = (space or "").strip().lower()

        def _first_existing(candidates: list[Path]) -> Optional[Path]:
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return None

        # 1) Preferred: func-level brain mask + boldref in the same space
        func_dirs = [
            deriv_root / "preprocessed" / "fmri" / sub_label / "func",
            deriv_root / "preprocessed" / "fmri" / "fmriprep" / sub_label / "func",
            deriv_root / "fmriprep" / sub_label / "func",
        ]
        space_tok = "MNI152NLin2009cAsym" if space == "mni" else "T1w"

        func_mask: Optional[Path] = None
        func_bg: Optional[Path] = None
        for d in func_dirs:
            if not d.exists():
                continue
            if func_mask is None:
                masks = sorted(d.glob(f"{sub_label}_task-{task}_run-*_space-{space_tok}_desc-brain_mask.nii.gz"))
                func_mask = masks[0] if masks else None
            if func_bg is None:
                # fMRIPrep commonly writes *_desc-preproc_boldref.nii.gz.
                boldrefs = sorted(d.glob(f"{sub_label}_task-{task}_run-*_space-{space_tok}_desc-preproc_boldref.nii.gz"))
                if not boldrefs:
                    boldrefs = sorted(d.glob(f"{sub_label}_task-{task}_run-*_space-{space_tok}_boldref.nii.gz"))
                func_bg = boldrefs[0] if boldrefs else None
            if func_mask is not None and func_bg is not None:
                break

        # 2) Optional: anatomical preproc background (better-looking overlays)
        search_dirs = [
            deriv_root / "preprocessed" / "fmri" / sub_label / "anat",
            deriv_root / "preprocessed" / "fmri" / "fmriprep" / sub_label / "anat",
            deriv_root / "fmriprep" / sub_label / "anat",
        ]

        anat_bg: Optional[Path] = None
        anat_mask: Optional[Path] = None
        for d in search_dirs:
            if not d.exists():
                continue
            if space == "mni":
                anat_bg = _first_existing(
                    [
                        d / f"{sub_label}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz",
                        d / f"{sub_label}_space-MNI152NLin6Asym_desc-preproc_T1w.nii.gz",
                    ]
                )
                anat_mask = _first_existing(
                    [d / f"{sub_label}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"]
                )
            else:
                anat_bg = _first_existing(
                    [
                        d / f"{sub_label}_desc-preproc_T1w.nii.gz",
                        d / f"{sub_label}_space-T1w_desc-preproc_T1w.nii.gz",
                    ]
                )
                anat_mask = _first_existing([d / f"{sub_label}_desc-brain_mask.nii.gz"])
            if anat_bg is not None:
                break

        # fMRIPrep's native-space T1w is a whole head: using it directly puts the
        # subject's face and neck into a report meant to be shared, and shrinks the
        # brain to a fraction of each panel. Skull-strip it with the brain mask that
        # sits beside it.
        if anat_bg is not None and anat_mask is not None:
            anat_bg = self._write_skull_stripped_background(
                anat_bg, anat_mask, sub_label=sub_label, space=space
            )

        bg_out = anat_bg or func_bg
        mask_out = func_mask
        return bg_out, mask_out

    def _write_skull_stripped_background(
        self,
        anat_path: Path,
        mask_path: Path,
        *,
        sub_label: str,
        space: str,
    ) -> Path:
        """Cache a brain-extracted copy of the anatomical background."""
        import nibabel as nib
        import numpy as np

        cache_dir = self.deriv_root / "logs" / "fmri_report_backgrounds"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"{sub_label}_space-{space}_desc-brain_T1w.nii.gz"
        if out_path.exists():
            return out_path

        anat = nib.load(str(anat_path))
        mask = nib.load(str(mask_path))
        anat_data = np.asanyarray(anat.dataobj)
        mask_data = np.asanyarray(mask.dataobj).astype(bool)
        if mask_data.shape != anat_data.shape:
            return anat_path

        nib.save(
            nib.Nifti1Image(np.where(mask_data, anat_data, 0), anat.affine, anat.header),
            str(out_path),
        )
        return out_path

    def _save_optional(self, img: Any, path: Path) -> Optional[Path]:
        """Write an image if there is one, and return where it went.

        Returns ``None`` on failure rather than raising. By the time this runs the GLM
        is fitted and the contrast is on disk; losing that to a problem writing an
        ancillary map would be the most expensive failure available.
        """
        if img is None:
            return None
        try:
            import nibabel as nib

            nib.save(img, str(path))
        except Exception as exc:
            self.logger.warning("Could not write %s (%s)", path.name, exc)
            return None
        self.logger.info("Saved %s", path.name)
        return path

    def _contrast_detail_maps(
        self, *, glm_result: Any, contrast_def: Any, plotting_cfg: Any
    ) -> tuple[Any, Any]:
        """Compute the effect and variance maps behind a contrast, if wanted.

        Gated on ``include_effect_size`` / ``include_standard_error`` rather than on
        whether plotting is enabled, which is where these used to live. Plotting is a
        separate step now: gating a *derivative* on it meant the report could not be
        rendered later without refitting the model, which is the one thing the
        manifest seam exists to prevent.
        """
        want_effect = bool(getattr(plotting_cfg, "include_effect_size", True))
        want_variance = bool(getattr(plotting_cfg, "include_standard_error", True))
        if not (want_effect or want_variance):
            return None, None

        flm = getattr(glm_result, "flm", None)
        if flm is None:
            return None, None

        try:
            argument = _contrast_arg_for_model_runs(flm, contrast_def)
            effect = flm.compute_contrast(argument, output_type="effect_size") if want_effect else None
            variance = (
                flm.compute_contrast(argument, output_type="effect_variance")
                if want_variance
                else None
            )
        except Exception as exc:
            # Two extra contrasts off an already-fitted model. Failing here must not
            # cost the fitted contrast that has already succeeded.
            self.logger.warning(
                "Could not compute the effect/variance maps for this contrast (%s); "
                "the report will omit the dual-coded and standard-error panels.",
                exc,
            )
            return None, None
        return effect, variance

    def _discover_tissue_segmentation(self, *, sub_label: str, space: str) -> Optional[Path]:
        """Discrete GM/WM/CSF segmentation used to order carpet-plot rows."""
        deriv_root = self.deriv_root
        search_dirs = [
            deriv_root / "preprocessed" / "fmri" / sub_label / "anat",
            deriv_root / "preprocessed" / "fmri" / "fmriprep" / sub_label / "anat",
            deriv_root / "fmriprep" / sub_label / "anat",
        ]
        if str(space or "").strip().lower().startswith("mni"):
            name = f"{sub_label}_space-MNI152NLin2009cAsym_dseg.nii.gz"
        else:
            name = f"{sub_label}_dseg.nii.gz"
        for directory in search_dirs:
            candidate = directory / name
            if candidate.exists():
                return candidate
        return None

    def process_subject(
        self,
        subject: str,
        task: str,
        *,
        contrast_cfg: Any,
        plotting_cfg: Optional[Any] = None,
        output_dir: Optional[Path] = None,
        freesurfer_subjects_dir: Optional[Path] = None,
        dry_run: bool = False,
        progress: Any = None,
        **_kwargs: Any,
    ) -> None:
        import nibabel as nib

        from fmri_pipeline.analysis.contrast_builder import (
            build_contrast_from_runs_detailed,
            resample_to_freesurfer,
        )

        sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"

        bids_fmri_root = self.config.get("paths.bids_fmri_root")
        if not bids_fmri_root:
            raise ValueError("Missing required config value: paths.bids_fmri_root")

        deriv_root = self.deriv_root
        out_base = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else deriv_root / sub_label / "fmri" / "first_level" / f"task-{task}"
        )

        contrast_name = safe_slug(
            str(getattr(contrast_cfg, "name", "contrast") or "contrast"),
            default="contrast",
        )
        out_dir = out_base / f"contrast-{contrast_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg_hash = _contrast_hash(contrast_cfg)
        # Use the *actual* nilearn output type for filenames to avoid ambiguity
        # (e.g., "beta" is represented by nilearn as "effect_size").
        output_type_req = str(getattr(contrast_cfg, "output_type", "z-score") or "z-score")
        output_type_actual = "z_score"
        nifti_path = out_dir / f"{sub_label}_task-{task}_contrast-{contrast_name}_stat-{output_type_req}_{cfg_hash}.nii.gz"
        sidecar_path = nifti_path.with_suffix("").with_suffix(".json")

        if dry_run:
            self.logger.info("Dry-run: would write %s", nifti_path)
            return

        import time as _time

        self.logger.info(
            "=== fMRI first-level: %s, task-%s, contrast='%s' ===",
            sub_label, task, contrast_name,
        )
        self.logger.info(
            "Output type: %s, space: %s",
            output_type_req,
            getattr(contrast_cfg, "fmriprep_space", "T1w"),
        )

        if progress is not None and hasattr(progress, "subject_start"):
            progress.subject_start(sub_label)
        if progress is not None and hasattr(progress, "step"):
            progress.step("Fit multi-run GLM + compute contrast")

        t_glm = _time.perf_counter()
        contrast_img, run_meta, glm_result, contrast_def, _ = build_contrast_from_runs_detailed(
            bids_fmri_root=Path(str(bids_fmri_root)).expanduser().resolve(),
            bids_derivatives=deriv_root,
            subject=subject,
            task=task,
            cfg=contrast_cfg,
            output_dir=out_dir,
        )
        glm_elapsed = _time.perf_counter() - t_glm

        if isinstance(run_meta, dict) and run_meta.get("output_type"):
            output_type_actual = str(run_meta.get("output_type"))
            nifti_path = out_dir / f"{sub_label}_task-{task}_contrast-{contrast_name}_stat-{output_type_actual}_{cfg_hash}.nii.gz"
            sidecar_path = nifti_path.with_suffix("").with_suffix(".json")

        n_runs = run_meta.get("n_runs", "?") if isinstance(run_meta, dict) else "?"
        shape_repr = getattr(contrast_img, "shape", None)
        if shape_repr is None:
            shape_repr = "unknown"
        self.logger.info(
            "GLM fit + contrast: %s runs, shape=%s (%.1fs)",
            n_runs, shape_repr, glm_elapsed,
        )

        contrast_img_for_plotting = contrast_img

        # The effect and the variance behind the same contrast, taken off the model
        # that is already fitted. Two compute_contrast calls, no refit.
        #
        # These used to be computed only when plotting was enabled, and then discarded
        # without being written. The report is a separate step now and cannot ask for
        # them retroactively, so without this the dual-coded panel and the standard
        # error panel were unreachable from any real run -- while the code that draws
        # them was fully written and tested.
        native_effect, native_variance = self._contrast_detail_maps(
            glm_result=glm_result, contrast_def=contrast_def, plotting_cfg=plotting_cfg
        )
        analysis_mask_img = getattr(glm_result, "mask_img", None)

        # Optional: resample to FreeSurfer subject space for downstream EEG integration.
        if bool(getattr(contrast_cfg, "resample_to_freesurfer", False)):
            fs_dir = freesurfer_subjects_dir
            if fs_dir is None:
                fs_dir = self.config.get("paths.freesurfer_dir")
                fs_dir = Path(str(fs_dir)).expanduser().resolve() if fs_dir else None
            if fs_dir is None:
                raise ValueError(
                    "resample_to_freesurfer=true requires paths.freesurfer_dir "
                    "(or --freesurfer-dir override)."
                )
            fs_subject_dir = fs_dir / sub_label
            if not fs_subject_dir.exists():
                raise FileNotFoundError(f"FreeSurfer subject directory not found: {fs_subject_dir}")
            contrast_img = resample_to_freesurfer(contrast_img, fs_subject_dir)
            # Everything that has to stay on the stat map's grid moves with it. The
            # dual-coded panel refuses a mismatched pair outright, and a mask on the
            # wrong grid is silently ignored -- which costs the colour limits their
            # brain and the report its coverage claim, with no error anywhere.
            if native_effect is not None:
                native_effect = resample_to_freesurfer(native_effect, fs_subject_dir)
            if native_variance is not None:
                native_variance = resample_to_freesurfer(native_variance, fs_subject_dir)
            if analysis_mask_img is not None:
                analysis_mask_img = resample_to_freesurfer(
                    analysis_mask_img, fs_subject_dir, interpolation="nearest"
                )

        nib.save(contrast_img, str(nifti_path))
        self.logger.info("Saved contrast map: %s", nifti_path.name)

        stem = f"{sub_label}_task-{task}_contrast-{contrast_name}"
        effect_path = self._save_optional(
            native_effect, out_dir / f"{stem}_stat-effect_size_{cfg_hash}.nii.gz"
        )
        variance_path = self._save_optional(
            native_variance, out_dir / f"{stem}_stat-effect_variance_{cfg_hash}.nii.gz"
        )
        # The mask the GLM was actually fitted inside: the intersection across runs.
        # The report previously recorded a mask *discovered* from the preprocessing
        # derivatives, which is a single run's brain mask. The two differ, and the
        # difference reached the reader as a coverage panel claiming an intersection
        # it was not showing and colour limits taken over voxels the model never fit.
        mask_path = self._save_optional(
            analysis_mask_img, out_dir / f"{stem}_desc-analysis_mask_{cfg_hash}.nii.gz"
        )

        # Record what was fit, beside what was fit. This is what lets `fmri-analysis
        # report` render from the derivatives tree without touching the model.
        from fmri_pipeline.analysis.report.manifest import write_report_manifest

        plot_cfg_for_manifest = plotting_cfg.normalized() if hasattr(plotting_cfg, "normalized") else None
        manifest_space = (
            "mni"
            if str(run_meta.get("analysis_space", "") if isinstance(run_meta, dict) else "")
            .lower()
            .startswith("mni")
            else "native"
        )
        if mask_path is None:
            # Falling back to a discovered mask is better than none, but it is not the
            # fitted mask and the manifest must not let it pass as one.
            _bg, mask_path = self._discover_plot_assets(
                sub_label=sub_label, task=task, space=manifest_space
            )

        contrast_vector, contrast_columns = _contrast_vector_for_design(
            glm_result=glm_result, contrast_def=contrast_def
        )
        manifest_path = write_report_manifest(
            contrast_dir=out_dir,
            subject=sub_label,
            task=task,
            contrast_name=contrast_name,
            stat_map=nifti_path,
            run_meta=run_meta,
            effect_map=effect_path,
            variance_map=variance_path,
            mask=mask_path,
            mask_is_analysis_mask=analysis_mask_img is not None and mask_path is not None,
            design_matrices=[
                Path(p) for p in (run_meta.get("design_matrix_tsv_paths") or [])
            ] if isinstance(run_meta, dict) else [],
            contrast_vector=contrast_vector,
            contrast_columns=contrast_columns,
            smoothing_fwhm=_optional_positive_float(
                getattr(contrast_cfg, "smoothing_fwhm", None)
            ),
            signal_scaling=bool(getattr(contrast_cfg, "signal_scaling", False)),
            threshold_mode=getattr(plot_cfg_for_manifest, "threshold_mode", "z"),
            z_threshold=getattr(plot_cfg_for_manifest, "z_threshold", 2.3),
            fdr_q=getattr(plot_cfg_for_manifest, "fdr_q", 0.05),
            cluster_min_voxels=getattr(plot_cfg_for_manifest, "cluster_min_voxels", 0),
            two_sided=getattr(plot_cfg_for_manifest, "two_sided", True),
            radiological=getattr(plot_cfg_for_manifest, "radiological", False),
        )
        if manifest_path is not None:
            self.logger.info("Wrote report manifest: %s", manifest_path.name)

        plotting_meta: Optional[dict[str, Any]] = None  # reporting now runs separately
        from fmri_pipeline.analysis.plotting_config import FmriPlottingConfig

        cfg_obj = plotting_cfg if isinstance(plotting_cfg, FmriPlottingConfig) else None
        if cfg_obj is not None and cfg_obj.normalized().enabled:
            # Optionally generate an MNI-space contrast in-memory (for plots only).
            mni_img = None
            mni_effect = None
            mni_variance = None
            want_mni = cfg_obj.normalized().space in {"mni", "both"}
            if want_mni:
                from fmri_pipeline.analysis.contrast_builder import ContrastBuilderConfig

                if isinstance(contrast_cfg, ContrastBuilderConfig):
                    cfg_mni = ContrastBuilderConfig(
                        **{**asdict(contrast_cfg), "fmriprep_space": "MNI152NLin2009cAsym"}
                    )
                else:
                    cfg_mni = replace(contrast_cfg) if is_dataclass(contrast_cfg) else copy.deepcopy(contrast_cfg)
                    if hasattr(cfg_mni, "fmriprep_space"):
                        setattr(cfg_mni, "fmriprep_space", "MNI152NLin2009cAsym")

                # Cache MNI map to disk for reproducibility and to avoid refitting when rerunning plots.
                mni_nifti_path = out_dir / (
                    f"{sub_label}_task-{task}_contrast-{contrast_name}"
                    f"_space-MNI152NLin2009cAsym_stat-{output_type_actual}_{cfg_hash}.nii.gz"
                )
                mni_effect_path = out_dir / (
                    f"{sub_label}_task-{task}_contrast-{contrast_name}"
                    f"_space-MNI152NLin2009cAsym_stat-effect_size_{cfg_hash}.nii.gz"
                )
                mni_variance_path = out_dir / (
                    f"{sub_label}_task-{task}_contrast-{contrast_name}"
                    f"_space-MNI152NLin2009cAsym_stat-effect_variance_{cfg_hash}.nii.gz"
                )
                need_mni_effect = bool(getattr(cfg_obj, "include_effect_size", True)) or bool(
                    getattr(cfg_obj, "include_signatures", True)
                )
                need_mni_variance = bool(getattr(cfg_obj, "include_standard_error", True))
                have_complete_mni_cache = mni_nifti_path.exists()
                if need_mni_effect:
                    have_complete_mni_cache = have_complete_mni_cache and mni_effect_path.exists()
                if need_mni_variance:
                    have_complete_mni_cache = have_complete_mni_cache and mni_variance_path.exists()

                if have_complete_mni_cache:
                    mni_img = nib.load(str(mni_nifti_path))
                    if need_mni_effect:
                        mni_effect = nib.load(str(mni_effect_path))
                    if need_mni_variance:
                        mni_variance = nib.load(str(mni_variance_path))
                else:
                    mni_img, _mni_meta, mni_glm, mni_contrast_def, _mni_out_type = build_contrast_from_runs_detailed(
                        bids_fmri_root=Path(str(bids_fmri_root)).expanduser().resolve(),
                        bids_derivatives=deriv_root,
                        subject=subject,
                        task=task,
                        cfg=cfg_mni,
                        output_dir=out_dir,
                    )
                    nib.save(mni_img, str(mni_nifti_path))

                    if need_mni_effect or need_mni_variance:
                        mni_contrast_arg = _contrast_arg_for_model_runs(mni_glm.flm, mni_contrast_def)
                        if need_mni_effect:
                            mni_effect = mni_glm.flm.compute_contrast(
                                mni_contrast_arg,
                                output_type="effect_size",
                            )
                            nib.save(mni_effect, str(mni_effect_path))
                        if need_mni_variance:
                            mni_variance = mni_glm.flm.compute_contrast(
                                mni_contrast_arg,
                                output_type="effect_variance",
                            )
                            nib.save(mni_variance, str(mni_variance_path))

            native_bg, native_mask = self._discover_plot_assets(sub_label=sub_label, task=task, space="native")
            mni_bg, mni_mask = self._discover_plot_assets(sub_label=sub_label, task=task, space="mni")
            tissue_seg = self._discover_tissue_segmentation(
                sub_label=sub_label,
                space=str(getattr(contrast_cfg, "fmriprep_space", "T1w") or "T1w"),
            )
            sig_root, sig_specs = self._discover_signature_root_and_specs()

            # native_effect and native_variance were recomputed here. They are now
            # computed once above, before the FreeSurfer resample, and written to
            # disk -- which is what makes them available to the report at all.

            # Signature expression is a computation, not a rendering step: it needs
            # the weight maps and the study's signature configuration. It is written
            # beside the contrast's maps, and the report reads it from there.
            if sig_root is not None and sig_specs and mni_effect is not None:
                try:
                    from fmri_pipeline.analysis.multivariate_signatures import (
                        compute_signature_expression,
                        write_signature_expression_tsv,
                    )

                    signature_results = compute_signature_expression(
                        stat_or_effect_img=mni_effect,
                        signature_root=sig_root,
                        signature_specs=sig_specs,
                        mask_img=(
                            nib.load(str(mni_mask)) if mni_mask is not None else None
                        ),
                    )
                    tsv_path = write_signature_expression_tsv(
                        signature_results, out_dir / "signature_expression.tsv"
                    )
                    self.logger.info("Wrote signature expression: %s", tsv_path.name)
                except Exception as exc:
                    # A signature that cannot be expressed is a measurement that did
                    # not resolve, not a reason to lose the fitted contrast.
                    self.logger.warning(
                        "Signature expression failed for %s (%s)", contrast_name, exc
                    )

        import json

        payload = {
            "subject": sub_label,
            "task": task,
            "contrast_name": getattr(contrast_cfg, "name", None),
            "output_type_requested": output_type_req,
            "output_type_actual": output_type_actual,
            "run_meta": run_meta,
            "contrast_cfg": asdict(contrast_cfg) if hasattr(contrast_cfg, "__dataclass_fields__") else repr(contrast_cfg),
            "plotting": {
                "cfg": asdict(plotting_cfg) if hasattr(plotting_cfg, "__dataclass_fields__") else repr(plotting_cfg),
                "outputs": plotting_meta,
            },
        }
        sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

        total_elapsed = _time.perf_counter() - t_glm
        self.logger.info(
            "fMRI analysis complete for %s: contrast='%s', stat=%s (%.1fs total)",
            sub_label, contrast_name, output_type_actual, total_elapsed,
        )

        if progress is not None and hasattr(progress, "subject_done"):
            progress.subject_done(sub_label, success=True)
