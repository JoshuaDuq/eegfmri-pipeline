from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from eeg_pipeline.pipelines.base import PipelineBase
from fmri_pipeline.analysis.second_level import (
    SecondLevelConfig,
    run_second_level_analysis,
)


class FmriSecondLevelPipeline(PipelineBase):
    """Run explicit group-level fMRI inference from first-level contrast maps."""

    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="fmri_second_level", config=config)

    def process_subject(self, subject: str, task: str, **kwargs: Any) -> None:
        raise NotImplementedError(
            "FmriSecondLevelPipeline does not process single subjects."
        )

    def run_batch(
        self,
        subjects: List[str],
        task: Optional[str] = None,
        *,
        second_level_cfg: SecondLevelConfig,
        dry_run: bool = False,
        progress: Any = None,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        resolved_task = self._validate_batch_inputs(subjects, task)
        if len(subjects) < 2:
            raise ValueError(
                "Second-level fMRI analysis requires at least two selected subjects."
            )

        run_context = self._create_run_metadata_context(
            subjects=subjects,
            task=resolved_task,
            kwargs={
                "second_level_cfg": second_level_cfg,
                "dry_run": dry_run,
            },
        )
        run_status = "failed"
        run_error: Optional[str] = None
        outputs: Dict[str, Any] = {}
        summary = {"n_subjects": len(subjects)}

        if progress is not None:
            progress.start(self.name, subjects)

        try:
            outputs = self.run_group_level(
                subjects,
                task=resolved_task,
                second_level_cfg=second_level_cfg,
                dry_run=dry_run,
                progress=progress,
            )
            run_status = "success"
            if progress is not None:
                progress.complete(success=True)
            return outputs
        except Exception as exc:
            run_error = str(exc)
            if progress is not None:
                progress.complete(success=False)
            raise
        finally:
            self._write_run_metadata(
                run_context,
                status=run_status,
                error=run_error,
                outputs=outputs,
                summary=summary,
            )

    def run_group_level(
        self,
        subjects: List[str],
        task: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        second_level_cfg = kwargs.get("second_level_cfg")
        if not isinstance(second_level_cfg, SecondLevelConfig):
            raise TypeError(
                "second_level_cfg must be a SecondLevelConfig instance."
            )

        return run_second_level_analysis(
            config=second_level_cfg,
            subjects=subjects,
            task=task,
            deriv_root=Path(self.deriv_root),
            dry_run=bool(kwargs.get("dry_run", False)),
            progress=kwargs.get("progress"),
        )
