"""Trial-wise fMRI beta estimation + multivariate signature readout.

Modes:
- beta-series: one GLM per run with one regressor per trial (LSA-style beta series)
- lss: one GLM per trial (Least Squares Separate)

Outputs are written under:
  <deriv_root>/sub-XX/fmri/beta_series/task-*/contrast-*/
  <deriv_root>/sub-XX/fmri/lss/task-*/contrast-*/
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from eeg_pipeline.pipelines.base import PipelineBase


class FmriTrialSignaturePipeline(PipelineBase):
    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="fmri_trial_signatures", config=config)

    def _discover_signature_root(self) -> Optional[Path]:
        try:
            cfg_path = self.config.get("paths.signature_dir")
        except Exception:
            cfg_path = None
        if cfg_path:
            p = Path(str(cfg_path)).expanduser()
            return p if p.exists() else None

        try:
            candidate = Path(self.deriv_root).expanduser().resolve().parent / "external"
            return candidate if candidate.exists() else None
        except Exception:
            return None

    def process_subject(
        self,
        subject: str,
        *,
        task: str,
        bids_fmri_root: Path,
        trial_cfg: Any,
        output_dir: Optional[Path] = None,
        signature_root: Optional[Path] = None,
        progress: Any = None,
        dry_run: bool = False,
        **_kwargs: Any,
    ) -> None:
        from fmri_pipeline.analysis.trial_signatures import (
            TrialSignatureExtractionConfig,
            run_trial_signature_extraction_for_subject,
        )

        cfg = trial_cfg if isinstance(trial_cfg, TrialSignatureExtractionConfig) else None
        if cfg is None:
            raise TypeError("trial_cfg must be a TrialSignatureExtractionConfig")

        sig_root = signature_root or self._discover_signature_root()
        if sig_root is None:
            self.logger.warning("No signature_root found; signature readouts will be skipped.")

        if dry_run:
            self.logger.info(
                "Dry-run: would compute %s trial signatures for sub-%s task-%s (out=%s)",
                cfg.method,
                subject,
                task,
                str(output_dir) if output_dir else "(default)",
            )
            return

        if progress is not None and hasattr(progress, "subject_start"):
            progress.subject_start(f"sub-{subject}" if not str(subject).startswith("sub-") else str(subject))
        if progress is not None and hasattr(progress, "step"):
            progress.step(f"Trial signatures ({cfg.method})")

        res = run_trial_signature_extraction_for_subject(
            bids_fmri_root=Path(bids_fmri_root).expanduser().resolve(),
            bids_derivatives=Path(self.deriv_root).expanduser().resolve(),
            deriv_root=Path(self.deriv_root).expanduser().resolve(),
            subject=subject,
            cfg=TrialSignatureExtractionConfig(**{**asdict(cfg), "task": task}),
            signature_root=sig_root,
            output_dir=output_dir,
        )
        self.logger.info(
            "Trial signatures done for %s: %s", subject, res.get("output_dir", "")
        )

    def run_group_level(self, subjects, task=None, **kwargs):  # type: ignore[override]
        # No group-level inference here; downstream EEG ML can aggregate subject TSVs.
        return

