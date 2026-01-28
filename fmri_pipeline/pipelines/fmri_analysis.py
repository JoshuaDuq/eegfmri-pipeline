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


def _safe_slug(value: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    value = "_".join(part for part in value.split("_") if part)
    return value or "contrast"


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

    def process_subject(
        self,
        subject: str,
        task: str,
        *,
        contrast_cfg: Any,
        output_dir: Optional[Path] = None,
        freesurfer_subjects_dir: Optional[Path] = None,
        dry_run: bool = False,
        progress: Any = None,
        **_kwargs: Any,
    ) -> None:
        import nibabel as nib

        from fmri_pipeline.analysis.contrast_builder import (
            build_contrast_from_runs,
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

        contrast_name = _safe_slug(str(getattr(contrast_cfg, "name", "contrast") or "contrast"))
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

        if progress is not None and hasattr(progress, "subject_start"):
            progress.subject_start(sub_label)
        if progress is not None and hasattr(progress, "step"):
            progress.step("Fit multi-run GLM + compute contrast")

        contrast_img, run_meta = build_contrast_from_runs(
            bids_fmri_root=Path(str(bids_fmri_root)).expanduser().resolve(),
            bids_derivatives=deriv_root,
            subject=subject,
            task=task,
            cfg=contrast_cfg,
            output_dir=out_dir,
        )

        if isinstance(run_meta, dict) and run_meta.get("output_type"):
            output_type_actual = str(run_meta.get("output_type"))
            nifti_path = out_dir / f"{sub_label}_task-{task}_contrast-{contrast_name}_stat-{output_type_actual}_{cfg_hash}.nii.gz"
            sidecar_path = nifti_path.with_suffix("").with_suffix(".json")

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
            }
            sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        except Exception:
            # Sidecar is best-effort; do not fail the analysis.
            pass

        if progress is not None and hasattr(progress, "subject_done"):
            progress.subject_done(sub_label, success=True)
