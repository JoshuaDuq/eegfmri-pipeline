from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.utils.config.roots import (
    resolve_fmri_bids_root,
    resolve_fmri_deriv_root,
)
from fmri_pipeline.analysis.resting_state import (
    RestingStateAnalysisConfig,
    atlas_output_name,
    run_resting_state_analysis_for_subject,
)


class FmriRestingStatePipeline(PipelineBase):
    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="fmri_resting_state", config=config)

    def _resolve_task_is_rest(self) -> bool:
        return True

    def _resolve_pipeline_deriv_root(self) -> Path:
        return resolve_fmri_deriv_root(
            self.config,
            task_is_rest=self._resolve_task_is_rest(),
        )

    def process_subject(
        self,
        subject: str,
        task: str,
        *,
        rest_cfg: RestingStateAnalysisConfig,
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
        progress: Any = None,
        **_kwargs: Any,
    ) -> None:
        sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"
        normalized_cfg = rest_cfg.normalized()
        bids_root = resolve_fmri_bids_root(
            self.config,
            task_is_rest=self._resolve_task_is_rest(),
        )
        deriv_root = resolve_fmri_deriv_root(
            self.config,
            task_is_rest=self._resolve_task_is_rest(),
        )
        atlas_name = atlas_output_name(normalized_cfg.atlas_labels_img)
        resolved_output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else deriv_root / sub_label / "fmri" / "rest" / f"task-{task}" / f"atlas-{atlas_name}"
        )

        if progress is not None and hasattr(progress, "subject_start"):
            progress.subject_start(sub_label)
        success = False
        try:
            if dry_run:
                resolved_output_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info("Dry-run: would write resting-state outputs to %s", resolved_output_dir)
                success = True
                return

            outputs = run_resting_state_analysis_for_subject(
                bids_fmri_root=Path(bids_root).expanduser().resolve(),
                bids_derivatives=Path(deriv_root).expanduser().resolve(),
                deriv_root=Path(deriv_root).expanduser().resolve(),
                subject=subject,
                task=task,
                cfg=normalized_cfg,
                output_dir=resolved_output_dir,
            )
            self.logger.info(
                "fMRI resting-state analysis complete for %s: %s",
                sub_label,
                outputs.get("connectivity_path"),
            )
            success = True
        finally:
            if progress is not None and hasattr(progress, "subject_done"):
                progress.subject_done(sub_label, success=success)
