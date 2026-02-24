"""fMRI first-level analysis pipeline (GLM + contrasts).

This pipeline computes subject-level (first-level) contrasts between conditions
from BIDS events files using nilearn's FirstLevelModel multi-run support.

Outputs are written under:
  <deriv_root>/sub-<ID>/fmri/first_level/<task>/<contrast_name>/
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from eeg_pipeline.pipelines.base import PipelineBase
from fmri_pipeline.utils.signature_paths import discover_signature_root
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


class FmriAnalysisPipeline(PipelineBase):
    """Compute first-level fMRI contrasts for each subject."""

    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="fmri_analysis", config=config)

    def _discover_signature_root(self) -> Optional[Path]:
        """
        Best-effort path discovery for multivariate signature weight maps.

        Preference:
        1) config: paths.signature_dir (explicit override)
        2) sibling directory of derivatives: <deriv_root>/../external
        """
        return discover_signature_root(self.config, self.deriv_root)

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

        # 1) Preferred: func-level brain mask + boldref in the same space
        func_dirs = [
            deriv_root / "preprocessed" / "fmri" / sub_label / "func",
            deriv_root / "preprocessed" / "fmri" / "fmriprep" / sub_label / "func",
            deriv_root / "fmriprep" / sub_label / "func",
        ]
        space_tok = "MNI152NLin2009cAsym" if space == "mni" else "T1w"
        # Use run-01 as the representative reference (typically consistent across runs).
        func_mask_patterns = [
            f"{sub_label}_task-{task}_run-01_space-{space_tok}_desc-brain_mask.nii.gz",
            f"{sub_label}_task-{task}_run-1_space-{space_tok}_desc-brain_mask.nii.gz",
        ]
        func_boldref_patterns = [
            f"{sub_label}_task-{task}_run-01_space-{space_tok}_boldref.nii.gz",
            f"{sub_label}_task-{task}_run-1_space-{space_tok}_boldref.nii.gz",
        ]

        func_mask: Optional[Path] = None
        func_bg: Optional[Path] = None
        for d in func_dirs:
            if not d.exists():
                continue
            if func_mask is None:
                func_mask = next((d / n for n in func_mask_patterns if (d / n).exists()), None)
            if func_bg is None:
                func_bg = next((d / n for n in func_boldref_patterns if (d / n).exists()), None)
            if func_mask is not None and func_bg is not None:
                break

        # 2) Optional: anatomical preproc background (better-looking overlays)
        search_dirs = [
            deriv_root / "preprocessed" / "fmri" / sub_label / "anat",
            deriv_root / "preprocessed" / "fmri" / "fmriprep" / sub_label / "anat",
            deriv_root / "fmriprep" / sub_label / "anat",
        ]

        if space == "mni":
            bg_names = [
                f"{sub_label}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz",
            ]
        else:
            bg_names = [
                f"{sub_label}_desc-preproc_T1w.nii.gz",
            ]

        anat_bg: Optional[Path] = None
        for d in search_dirs:
            if not d.exists():
                continue
            anat_bg = next((d / n for n in bg_names if (d / n).exists()), None)
            if anat_bg is not None:
                break

        bg_out = anat_bg or func_bg
        mask_out = func_mask
        return bg_out, mask_out

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

        nib.save(contrast_img, str(nifti_path))
        self.logger.info("Saved contrast map: %s", nifti_path.name)

        plotting_meta: Optional[dict[str, Any]] = None
        try:
            from fmri_pipeline.analysis.plotting_config import FmriPlottingConfig
            from fmri_pipeline.analysis.reporting import run_fmri_plotting_and_report

            cfg_obj = plotting_cfg if isinstance(plotting_cfg, FmriPlottingConfig) else None
            if cfg_obj is not None and cfg_obj.normalized().enabled:
                # Optionally generate an MNI-space contrast in-memory (for plots only).
                mni_img = None
                mni_effect = None
                mni_variance = None
                want_mni = cfg_obj.normalized().space in {"mni", "both"}
                if want_mni:
                    try:
                        from fmri_pipeline.analysis.contrast_builder import ContrastBuilderConfig

                        if isinstance(contrast_cfg, ContrastBuilderConfig):
                            cfg_mni = ContrastBuilderConfig(
                                **{**asdict(contrast_cfg), "fmriprep_space": "MNI152NLin2009cAsym"}
                            )
                        else:
                            # Best-effort: reuse the cfg and override fmriprep_space when it's attribute-like.
                            cfg_mni = contrast_cfg
                            if hasattr(cfg_mni, "fmriprep_space"):
                                setattr(cfg_mni, "fmriprep_space", "MNI152NLin2009cAsym")

                        # Cache MNI map to disk for reproducibility and to avoid refitting when rerunning plots.
                        mni_nifti_path = out_dir / (
                            f"{sub_label}_task-{task}_contrast-{contrast_name}"
                            f"_space-MNI152NLin2009cAsym_stat-{output_type_actual}_{cfg_hash}.nii.gz"
                        )
                        if mni_nifti_path.exists():
                            mni_img = nib.load(str(mni_nifti_path))
                        else:
                            mni_img, _mni_meta, mni_glm, mni_contrast_def, _mni_out_type = build_contrast_from_runs_detailed(
                                bids_fmri_root=Path(str(bids_fmri_root)).expanduser().resolve(),
                                bids_derivatives=deriv_root,
                                subject=subject,
                                task=task,
                                cfg=cfg_mni,
                                output_dir=out_dir,
                            )
                            try:
                                nib.save(mni_img, str(mni_nifti_path))
                            except Exception as exc:
                                self.logger.warning(
                                    "Failed to cache MNI contrast NIfTI at %s: %s",
                                    mni_nifti_path,
                                    exc,
                                )

                            try:
                                mni_effect = mni_glm.flm.compute_contrast(mni_contrast_def, output_type="effect_size")
                                mni_variance = mni_glm.flm.compute_contrast(mni_contrast_def, output_type="variance")
                            except Exception as exc:
                                self.logger.warning(
                                    "Failed to compute MNI effect/variance maps for plotting: %s",
                                    exc,
                                )
                    except Exception as exc:
                        self.logger.warning("Skipping MNI plotting (failed to build MNI contrast): %s", exc)

                native_bg, native_mask = self._discover_plot_assets(sub_label=sub_label, task=task, space="native")
                mni_bg, mni_mask = self._discover_plot_assets(sub_label=sub_label, task=task, space="mni")

                native_effect = None
                native_variance = None
                try:
                    if bool(getattr(cfg_obj, "include_effect_size", True)) or bool(
                        getattr(cfg_obj, "include_standard_error", True)
                    ):
                        native_effect = glm_result.flm.compute_contrast(contrast_def, output_type="effect_size")
                        native_variance = glm_result.flm.compute_contrast(contrast_def, output_type="variance")
                except Exception as exc:
                    self.logger.warning(
                        "Failed to compute native effect/variance maps for plotting: %s",
                        exc,
                    )

                plotting_meta = run_fmri_plotting_and_report(
                    contrast_dir=out_dir,
                    subject=sub_label,
                    task=task,
                    contrast_name=contrast_name,
                    cfg=cfg_obj,
                    run_meta=run_meta if isinstance(run_meta, dict) else None,
                    native_stat_img=contrast_img_for_plotting,
                    mni_stat_img=mni_img,
                    native_effect_img=native_effect,
                    native_variance_img=native_variance,
                    mni_effect_img=mni_effect,
                    mni_variance_img=mni_variance,
                    native_bg_img_path=native_bg,
                    mni_bg_img_path=mni_bg,
                    native_mask_img_path=native_mask,
                    mni_mask_img_path=mni_mask,
                    signature_root=self._discover_signature_root(),
                )
        except Exception as exc:
            # Best-effort: plotting/reporting should never fail the fMRI analysis step.
            self.logger.warning("Failed to generate fMRI plots/report (continuing): %s", exc)

        try:
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
        except Exception:
            # Sidecar is best-effort; do not fail the analysis.
            pass

        total_elapsed = _time.perf_counter() - t_glm
        self.logger.info(
            "fMRI analysis complete for %s: contrast='%s', stat=%s (%.1fs total)",
            sub_label, contrast_name, output_type_actual, total_elapsed,
        )

        if progress is not None and hasattr(progress, "subject_done"):
            progress.subject_done(sub_label, success=True)
