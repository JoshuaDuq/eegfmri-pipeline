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
from fmri_pipeline.utils.signature_paths import discover_signature_root_and_specs


class FmriTrialSignaturePipeline(PipelineBase):
    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="fmri_trial_signatures", config=config)

    def _discover_signature_root_and_specs(self) -> tuple[Optional[Path], list]:
        return discover_signature_root_and_specs(self.config, self.deriv_root)

    def process_subject(
        self,
        subject: str,
        *,
        task: str,
        bids_fmri_root: Path,
        trial_cfg: Any,
        output_dir: Optional[Path] = None,
        signature_root: Optional[Path] = None,
        signature_specs: Optional[list] = None,
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

        discovered_root, discovered_specs = self._discover_signature_root_and_specs()
        sig_root = signature_root or discovered_root
        sig_specs = signature_specs if signature_specs is not None else discovered_specs

        if sig_root is None or not sig_specs:
            self.logger.warning("No signature_root or signature_specs configured; signature readouts will be skipped.")

        if dry_run:
            self.logger.info(
                "Dry-run: would compute %s trial signatures for sub-%s task-%s (out=%s)",
                cfg.method,
                subject,
                task,
                str(output_dir) if output_dir else "(default)",
            )
            return

        import time as _time

        sub_label = f"sub-{subject}" if not str(subject).startswith("sub-") else str(subject)
        self.logger.info(
            "=== Trial signatures: %s, task-%s, method=%s ===",
            sub_label, task, cfg.method,
        )

        if progress is not None and hasattr(progress, "subject_start"):
            progress.subject_start(sub_label)
        if progress is not None and hasattr(progress, "step"):
            progress.step(f"Trial signatures ({cfg.method})")

        t0 = _time.perf_counter()
        res = run_trial_signature_extraction_for_subject(
            bids_fmri_root=Path(bids_fmri_root).expanduser().resolve(),
            bids_derivatives=Path(self.deriv_root).expanduser().resolve(),
            deriv_root=Path(self.deriv_root).expanduser().resolve(),
            subject=subject,
            cfg=TrialSignatureExtractionConfig(**{**asdict(cfg), "task": task}),
            signature_root=sig_root,
            signature_specs=sig_specs,
            output_dir=output_dir,
        )
        elapsed = _time.perf_counter() - t0

        out_path = res.get("output_dir", "")
        n_trials = res.get("n_trials", "?")
        n_sigs = res.get("n_signatures", "?")
        self.logger.info(
            "Trial signatures complete for %s: %s trials, %s signatures (%.1fs) -> %s",
            sub_label, n_trials, n_sigs, elapsed, out_path,
        )

    def run_group_level(self, subjects, task=None, **kwargs):  # type: ignore[override]
        return
