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
from fmri_pipeline.utils.bold_discovery import fitted_signal_scaling_mode
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
) -> tuple[list[float], list[str]]:
    """Expand a contrast expression into weights against the design's own columns.

    The report draws the contrast as a strip beneath the design matrix, and computes
    the design's efficiency for it. Both need numbers per column; the manifest carried
    only the expression string, so neither could ever be produced. Matching is by
    column name downstream, so the two lists are returned together.

    Invalid or unavailable design metadata is an incomplete scientific derivative,
    so it surfaces rather than silently dropping the report's contrast strip.
    """
    import numpy as np

    design_matrices = getattr(getattr(glm_result, "flm", None), "design_matrices_", None)
    if not design_matrices:
        raise ValueError("The fitted model has no design matrices to record.")

    columns = [str(c) for c in design_matrices[0].columns]
    if not isinstance(contrast_def, str):
        # An explicit vector, already aligned to the design nilearn was given.
        values = np.asarray(contrast_def, dtype=float).ravel()
        if values.size != len(columns):
            raise ValueError(
                f"Contrast vector has {values.size} weights for {len(columns)} design columns."
            )
        return list(map(float, values)), columns

    if contrast_def in columns:
        return [1.0 if column == contrast_def else 0.0 for column in columns], columns

    from nilearn.glm.contrasts import expression_to_contrast_vector

    vector = expression_to_contrast_vector(contrast_def, columns)
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
                masks = sorted(
                    d.glob(
                        f"{sub_label}_task-{task}_run-*_space-{space_tok}_desc-brain_mask.nii.gz"
                    )
                )
                func_mask = masks[0] if masks else None
            if func_bg is None:
                # fMRIPrep commonly writes *_desc-preproc_boldref.nii.gz.
                boldrefs = sorted(
                    d.glob(
                        f"{sub_label}_task-{task}_run-*_space-{space_tok}_desc-preproc_boldref.nii.gz"
                    )
                )
                if not boldrefs:
                    boldrefs = sorted(
                        d.glob(f"{sub_label}_task-{task}_run-*_space-{space_tok}_boldref.nii.gz")
                    )
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
        """Write an image when requested; only an absent optional image is skipped."""
        if img is None:
            return None

        import nibabel as nib

        nib.save(img, str(path))
        self.logger.info("Saved %s", path.name)
        return path

    def _save_required(self, img: Any, path: Path, *, artifact_name: str) -> Path:
        """Write a required image, preserving the original write error."""
        if img is None:
            raise ValueError(f"Cannot write absent required artifact: {artifact_name}.")

        import nibabel as nib

        nib.save(img, str(path))
        self.logger.info("Saved %s", path.name)
        return path

    def _contrast_detail_maps(
        self, *, glm_result: Any, contrast_def: Any, stats_cfg: Any
    ) -> tuple[Any, Any]:
        """Compute the effect and variance maps behind a contrast, if wanted.

        Gated on ``include_effect_size`` / ``include_standard_error`` rather than on
        whether plotting is enabled, which is where these used to live. Plotting is a
        separate step now: gating a *derivative* on it meant the report could not be
        rendered later without refitting the model, which is the one thing the
        manifest seam exists to prevent.
        """
        want_effect = bool(getattr(stats_cfg, "include_effect_size", True))
        want_variance = bool(getattr(stats_cfg, "include_standard_error", True))
        if not (want_effect or want_variance):
            return None, None

        flm = getattr(glm_result, "flm", None)
        if flm is None:
            raise ValueError("Cannot compute detail maps without the fitted model.")

        argument = _contrast_arg_for_model_runs(flm, contrast_def)
        effect = flm.compute_contrast(argument, output_type="effect_size") if want_effect else None
        variance = (
            flm.compute_contrast(argument, output_type="effect_variance") if want_variance else None
        )
        return effect, variance

    def _run_level_maps(
        self,
        *,
        glm_result: Any,
        contrast_def: Any,
        run_meta: Any,
        out_dir: Path,
        stem: str,
        cfg_hash: str,
        z_threshold: float = 2.3,
    ) -> dict[str, Any]:
        """Write what each run contributes to this contrast, from the fitted model.

        Returns manifest keyword arguments rather than a tuple: this produces three
        related artifacts and seven scalars, and a tuple that long makes the call site
        depend on positions nobody can read.

        Three measurements, all from the model already fitted:

        - each run's own estimate of the contrast, which the forest panel draws;
        - the run sign-flip null, which is the only familywise correction available
          here that does not assume every voxel is N(0, 1) -- a single-subject map
          combined across runs is routinely over-dispersed relative to that;
        - leave-one-run-out influence, which answers a question the forest panel
          cannot: a run can carry the largest peak estimates while a different run
          moves the map more.

        A fixed-effects combination across runs is weighted equally per run, so an
        effect resting on one run and an effect present in all of them produce the same
        map and the same cluster table.

        No refit: nilearn keeps ``labels_`` and ``results_`` per run and combines them
        at ``compute_contrast`` time, so the per-run estimates already exist inside the
        fitted object. Failing here costs one diagnostic panel, never the contrast --
        which is on disk by the time this runs.
        """
        from fmri_pipeline.analysis.run_level import (
            compute_run_influence,
            compute_run_level_contrast,
            compute_sign_flip_null,
            write_run_influence,
            write_run_level_maps,
            write_sign_flip_null,
        )

        flm = getattr(glm_result, "flm", None)
        if flm is None:
            raise ValueError("Cannot compute run-level diagnostics without the fitted model.")

        # The manifest's own labeller, so the forest plot's rows carry the same run
        # names as the motion table and the design section.
        from fmri_pipeline.analysis.report.manifest import run_labels_from_bold_paths

        included = run_meta.get("included_bold_paths") or [] if isinstance(run_meta, dict) else []
        labels = list(run_labels_from_bold_paths(included))

        result = compute_run_level_contrast(flm, contrast_def, run_labels=labels)
        if result is None:
            return {}

        effect_path, variance_path = write_run_level_maps(
            result, out_dir=out_dir, stem=stem, cfg_hash=cfg_hash
        )
        if effect_path is not None:
            self.logger.info(
                "Saved run-level maps for %d run(s): %s", result.n_runs, effect_path.name
            )
        fields: dict[str, Any] = {
            "run_effect_map": effect_path,
            "run_variance_map": variance_path,
        }

        null = compute_sign_flip_null(flm, contrast_def)
        if null is not None:
            sign_flip_path = write_sign_flip_null(
                null, out_dir=out_dir, stem=stem, cfg_hash=cfg_hash
            )
            if sign_flip_path is not None:
                fields.update(
                    sign_flip_null_tsv=sign_flip_path,
                    sign_flip_fwe_height=null.fwe_height,
                    sign_flip_fwe_survivors=null.fwe_survivors,
                    sign_flip_global_p=null.global_p,
                    sign_flip_p_floor=null.p_floor,
                    sign_flip_n_patterns=null.n_patterns,
                    sign_flip_n_runs=null.n_runs,
                    sign_flip_observed_max=null.observed_max,
                )
                self.logger.info(
                    "Sign-flip null over %d run(s): FWE height %.2f, %d voxel(s), "
                    "p = %.3f (floor %.3f)",
                    null.n_runs,
                    null.fwe_height,
                    null.fwe_survivors,
                    null.global_p,
                    null.p_floor,
                )

        influence = compute_run_influence(
            flm, contrast_def, run_labels=labels, threshold=float(z_threshold)
        )
        if influence:
            influence_path = write_run_influence(
                influence, out_dir=out_dir, stem=stem, cfg_hash=cfg_hash
            )
            if influence_path is not None:
                fields["run_influence_tsv"] = influence_path
                worst = max(influence, key=lambda row: abs(row.delta))
                self.logger.info(
                    "Run influence: dropping %s moves the survivor count by %+d",
                    worst.dropped_run,
                    worst.delta,
                )

        return fields

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
        stats_cfg: Optional[Any] = None,
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

        if stats_cfg is None:
            from fmri_pipeline.analysis.plotting_config import FmriStatsConfig

            stats_cfg = FmriStatsConfig()
        if hasattr(stats_cfg, "validate"):
            stats_cfg.validate()
        configured_signatures = self.config.get("paths.signature_maps", []) or []
        if stats_cfg.include_signatures and configured_signatures and stats_cfg.space == "native":
            raise ValueError(
                "Configured multivariate signatures require fmri_stats.space='mni' "
                "or 'both' because their weight maps are defined in standard space."
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
        nifti_path = (
            out_dir
            / f"{sub_label}_task-{task}_contrast-{contrast_name}_stat-{output_type_req}_{cfg_hash}.nii.gz"
        )
        sidecar_path = nifti_path.with_suffix("").with_suffix(".json")

        if dry_run:
            self.logger.info("Dry-run: would write %s", nifti_path)
            return

        import time as _time

        self.logger.info(
            "=== fMRI first-level: %s, task-%s, contrast='%s' ===",
            sub_label,
            task,
            contrast_name,
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
            nifti_path = (
                out_dir
                / f"{sub_label}_task-{task}_contrast-{contrast_name}_stat-{output_type_actual}_{cfg_hash}.nii.gz"
            )
            sidecar_path = nifti_path.with_suffix("").with_suffix(".json")

        n_runs = run_meta.get("n_runs", "?") if isinstance(run_meta, dict) else "?"
        shape_repr = getattr(contrast_img, "shape", None)
        if shape_repr is None:
            shape_repr = "unknown"
        self.logger.info(
            "GLM fit + contrast: %s runs, shape=%s (%.1fs)",
            n_runs,
            shape_repr,
            glm_elapsed,
        )

        # The effect and the variance behind the same contrast, taken off the model
        # that is already fitted. Two compute_contrast calls, no refit.
        #
        # These used to be computed only when plotting was enabled, and then discarded
        # without being written. The report is a separate step now and cannot ask for
        # them retroactively, so without this the dual-coded panel and the standard
        # error panel were unreachable from any real run -- while the code that draws
        # them was fully written and tested.
        native_effect, native_variance = self._contrast_detail_maps(
            glm_result=glm_result, contrast_def=contrast_def, stats_cfg=stats_cfg
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
        mask_path = self._save_required(
            analysis_mask_img,
            out_dir / f"{stem}_desc-analysis_mask_{cfg_hash}.nii.gz",
            artifact_name="fitted analysis mask",
        )

        from fmri_pipeline.analysis.model_fit import (
            extract_model_fit_images,
            write_model_fit_images,
        )
        from fmri_pipeline.analysis.report.manifest import (
            run_labels_from_bold_paths,
        )

        fitted_model = getattr(glm_result, "flm", None)
        if fitted_model is None:
            raise ValueError("Cannot write model-fit evidence without the fitted model.")
        run_labels = run_labels_from_bold_paths(run_meta["included_bold_paths"])
        model_fit_images = extract_model_fit_images(fitted_model)
        model_fit_paths = write_model_fit_images(
            model_fit_images,
            out_dir=out_dir,
            stem=stem,
            cfg_hash=cfg_hash,
            run_labels=run_labels,
        )
        self.logger.info(
            "Saved residual and predicted model series for %d run(s)",
            len(model_fit_paths.residuals),
        )

        run_level_fields = self._run_level_maps(
            glm_result=glm_result,
            contrast_def=contrast_def,
            run_meta=run_meta,
            out_dir=out_dir,
            stem=stem,
            cfg_hash=cfg_hash,
            z_threshold=float(stats_cfg.z_threshold),
        )

        # Record what was fit, beside what was fit. This is what lets `fmri-analysis
        # report` render from the derivatives tree without touching the model.
        from fmri_pipeline.analysis.report.manifest import write_report_manifest

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
            residual_paths=model_fit_paths.residuals,
            predicted_paths=model_fit_paths.predicted,
            effect_map=effect_path,
            variance_map=variance_path,
            **run_level_fields,
            mask=mask_path,
            mask_is_analysis_mask=analysis_mask_img is not None and mask_path is not None,
            design_matrices=(
                [Path(p) for p in (run_meta.get("design_matrix_tsv_paths") or [])]
                if isinstance(run_meta, dict)
                else []
            ),
            contrast_vector=contrast_vector,
            contrast_columns=contrast_columns,
            smoothing_fwhm=_optional_positive_float(getattr(contrast_cfg, "smoothing_fwhm", None)),
            # Off the fitted model, not off the config. `contrast_cfg` has no
            # `signal_scaling` field, so reading one recorded "not scaled" for every
            # contrast ever produced -- while the model scales unconditionally -- and
            # the report labelled percent-signal-change maps "arbitrary BOLD units".
            signal_scaling_mode=fitted_signal_scaling_mode(getattr(glm_result, "flm", None)),
            threshold_mode=stats_cfg.threshold_mode,
            z_threshold=stats_cfg.z_threshold,
            fdr_q=stats_cfg.fdr_q,
            cluster_min_voxels=stats_cfg.cluster_min_voxels,
            two_sided=stats_cfg.two_sided,
            radiological=False,
            contrast_cfg=contrast_cfg,
        )
        self.logger.info("Wrote report manifest: %s", manifest_path.name)

        sig_root, sig_specs = (None, [])
        if stats_cfg.include_signatures:
            sig_root, sig_specs = self._discover_signature_root_and_specs()
        signatures_requested = bool(sig_specs)
        if stats_cfg.space in {"mni", "both"}:
            # Generate explicitly configured standard-space statistical artifacts.
            mni_img = None
            mni_effect = None
            mni_variance = None
            want_mni = True
            if want_mni:
                from fmri_pipeline.analysis.contrast_builder import ContrastBuilderConfig

                if isinstance(contrast_cfg, ContrastBuilderConfig):
                    cfg_mni = ContrastBuilderConfig(
                        **{**asdict(contrast_cfg), "fmriprep_space": "MNI152NLin2009cAsym"}
                    )
                else:
                    cfg_mni = (
                        replace(contrast_cfg)
                        if is_dataclass(contrast_cfg)
                        else copy.deepcopy(contrast_cfg)
                    )
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
                need_mni_effect = stats_cfg.include_effect_size or stats_cfg.include_signatures
                need_mni_variance = stats_cfg.include_standard_error
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
                    mni_img, _mni_meta, mni_glm, mni_contrast_def, _mni_out_type = (
                        build_contrast_from_runs_detailed(
                            bids_fmri_root=Path(str(bids_fmri_root)).expanduser().resolve(),
                            bids_derivatives=deriv_root,
                            subject=subject,
                            task=task,
                            cfg=cfg_mni,
                            output_dir=out_dir,
                        )
                    )
                    nib.save(mni_img, str(mni_nifti_path))

                    if need_mni_effect or need_mni_variance:
                        mni_contrast_arg = _contrast_arg_for_model_runs(
                            mni_glm.flm, mni_contrast_def
                        )
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

            # native_effect and native_variance were recomputed here. They are now
            # computed once above, before the FreeSurfer resample, and written to
            # disk -- which is what makes them available to the report at all.

            # Signature expression is a computation, not a rendering step: it needs
            # the weight maps and the study's signature configuration. It is written
            # beside the contrast's maps, and the report reads it from there.
            if signatures_requested and sig_root is not None:
                if mni_effect is None:
                    raise ValueError(
                        "fmri_stats.include_signatures requires space='mni' or 'both' "
                        "and an MNI effect-size map."
                    )
                _mni_bg, mni_mask = self._discover_plot_assets(
                    sub_label=sub_label, task=task, space="mni"
                )
                from fmri_pipeline.analysis.multivariate_signatures import (
                    compute_signature_expression,
                    write_signature_expression_tsv,
                )

                signature_results = compute_signature_expression(
                    stat_or_effect_img=mni_effect,
                    signature_root=sig_root,
                    signature_specs=sig_specs,
                    mask_img=(nib.load(str(mni_mask)) if mni_mask is not None else None),
                )
                tsv_path = write_signature_expression_tsv(
                    signature_results, out_dir / "signature_expression.tsv"
                )
                self.logger.info("Wrote signature expression: %s", tsv_path.name)

        import json

        payload = {
            "subject": sub_label,
            "task": task,
            "contrast_name": getattr(contrast_cfg, "name", None),
            "output_type_requested": output_type_req,
            "output_type_actual": output_type_actual,
            "run_meta": run_meta,
            "contrast_cfg": (
                asdict(contrast_cfg)
                if hasattr(contrast_cfg, "__dataclass_fields__")
                else repr(contrast_cfg)
            ),
            "fmri_stats": (
                asdict(stats_cfg) if hasattr(stats_cfg, "__dataclass_fields__") else repr(stats_cfg)
            ),
        }
        sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

        total_elapsed = _time.perf_counter() - t_glm
        self.logger.info(
            "fMRI analysis complete for %s: contrast='%s', stat=%s (%.1fs total)",
            sub_label,
            contrast_name,
            output_type_actual,
            total_elapsed,
        )

        if progress is not None and hasattr(progress, "subject_done"):
            progress.subject_done(sub_label, success=True)
