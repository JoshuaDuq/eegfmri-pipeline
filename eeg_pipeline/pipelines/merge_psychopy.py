"""Merge PsychoPy TrialSummary.csv into BIDS *_events.tsv pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional

from eeg_pipeline.analysis.utilities.merge_psychopy import run_merge_psychopy
from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.infra.tsv import read_tsv


class MergePsychopyPipeline(PipelineBase):
    """Merge behavioral TrialSummary CSVs into BIDS events.tsv files."""

    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="merge_psychopy", config=config)
        self.bids_root = Path(self.config.bids_root)
        default_source = "data/source_data"
        self.source_root = Path(self.config.get("paths.source_data", default_source))

    def run_group_level(self, subjects: List[str], task: str, **kwargs: Any) -> None:
        return

    def process_subject(self, subject: str, task: str, **kwargs: Any) -> None:
        allow = kwargs.get(
            "allow_misaligned_trim",
            bool(self.config.get("alignment.allow_misaligned_trim", False)),
        )
        run_merge_psychopy(
            bids_root=self.bids_root,
            source_root=self.source_root,
            task=task,
            subjects=[subject],
            event_prefixes=kwargs.get("event_prefixes"),
            event_types=kwargs.get("event_types"),
            dry_run=bool(kwargs.get("dry_run", False)),
            allow_misaligned_trim=bool(allow),
            _logger=self.logger,
        )

        if not bool(kwargs.get("dry_run", False)):
            self._validate_against_fmri_events(
                subject,
                task,
                qc_columns=kwargs.get("qc_columns"),
            )

    def run_batch(
        self,
        subjects: List[str],
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        resolved_task = self._validate_batch_inputs(subjects, task)
        run_context = self._create_run_metadata_context(
            subjects=subjects,
            task=resolved_task,
            kwargs=kwargs,
        )
        run_status = "failed"
        run_error: Optional[str] = None
        n_merged = 0
        allow = kwargs.get(
            "allow_misaligned_trim",
            bool(self.config.get("alignment.allow_misaligned_trim", False)),
        )
        try:
            n_merged = run_merge_psychopy(
                bids_root=self.bids_root,
                source_root=self.source_root,
                task=resolved_task,
                subjects=subjects,
                event_prefixes=kwargs.get("event_prefixes"),
                event_types=kwargs.get("event_types"),
                dry_run=bool(kwargs.get("dry_run", False)),
                allow_misaligned_trim=bool(allow),
                _logger=self.logger,
            )

            if not bool(kwargs.get("dry_run", False)):
                for subj in subjects:
                    self._validate_against_fmri_events(
                        subj,
                        resolved_task,
                        qc_columns=kwargs.get("qc_columns"),
                    )
            run_status = "success"
            return n_merged
        except Exception as exc:
            run_error = str(exc)
            raise
        finally:
            self._write_run_metadata(
                run_context,
                status=run_status,
                error=run_error,
                outputs={},
                summary={
                    "n_subjects": len(subjects),
                    "n_merged": n_merged,
                },
            )

    def _validate_against_fmri_events(
        self,
        subject: str,
        task: str,
        qc_columns: Optional[List[str]] = None,
    ) -> None:
        fmri_root = self.config.get("paths.bids_fmri_root")
        if not fmri_root:
            return
        fmri_root = Path(str(fmri_root))
        if not fmri_root.exists():
            return

        eeg_dir = self.bids_root / f"sub-{subject}" / "eeg"
        fmri_dir = fmri_root / f"sub-{subject}" / "func"
        if not eeg_dir.exists() or not fmri_dir.exists():
            return

        eeg_paths = sorted(eeg_dir.glob(f"sub-{subject}_task-{task}_run-*_events.tsv"))
        fmri_paths = [
            p
            for p in sorted(fmri_dir.glob(f"sub-{subject}_task-{task}_run-*_events.tsv"))
            if not p.name.endswith("_bold_events.tsv")
        ]
        if not fmri_paths:
            # Backward-compat: older outputs used a non-BIDS `_bold_events.tsv` suffix.
            fmri_paths = sorted(fmri_dir.glob(f"sub-{subject}_task-{task}_run-*_bold_events.tsv"))
        if not eeg_paths or not fmri_paths:
            return

        configured_cols: Any = qc_columns
        if configured_cols is None:
            configured_cols = self.config.get("alignment.cross_modal_qc_columns", None)
        if isinstance(configured_cols, str):
            configured_cols = [c.strip() for c in configured_cols.replace(";", ",").split(",")]
        cols = [str(c).strip() for c in (configured_cols or []) if str(c).strip()]
        if not cols:
            return

        def run_num(p: Path) -> Optional[int]:
            m = re.search(r"_run-([0-9]+)_", p.name)
            if not m:
                return None
            try:
                return int(m.group(1))
            except ValueError:
                return None

        fmri_by_run = {run_num(p): p for p in fmri_paths if run_num(p) is not None}

        for eeg_ev in eeg_paths:
            r = run_num(eeg_ev)
            if r is None:
                continue
            fmri_ev = fmri_by_run.get(r)
            if fmri_ev is None:
                self.logger.info("No matching fMRI events for sub-%s run-%s (skip cross-modal QC).", subject, r)
                continue

            try:
                eeg_df = read_tsv(eeg_ev)
                fmri_df = read_tsv(fmri_ev)
            except Exception as exc:
                self.logger.warning("Cross-modal QC failed reading events: %s", exc)
                continue

            # Keep only trial rows (drop volume triggers etc.)
            for df in (eeg_df, fmri_df):
                if "trial_number" in df.columns:
                    df["trial_number"] = df["trial_number"].astype("Int64")

            if "trial_number" not in eeg_df.columns or "trial_number" not in fmri_df.columns:
                continue

            eeg_trials = eeg_df.dropna(subset=["trial_number"]).copy()
            fmri_trials = fmri_df.dropna(subset=["trial_number"]).copy()
            if eeg_trials.empty or fmri_trials.empty:
                continue

            eeg_trials = eeg_trials.sort_values("trial_number", kind="mergesort")
            fmri_trials = fmri_trials.sort_values("trial_number", kind="mergesort")

            # Compare configured behavioral columns if present in both.
            for col in cols:
                if col not in eeg_trials.columns or col not in fmri_trials.columns:
                    continue
                merged = eeg_trials[["trial_number", col]].merge(
                    fmri_trials[["trial_number", col]],
                    on="trial_number",
                    suffixes=("_eeg", "_fmri"),
                    how="inner",
                )
                if merged.empty:
                    continue
                if col == "stimulus_temp":
                    diff = (merged[f"{col}_eeg"].astype(float) - merged[f"{col}_fmri"].astype(float)).abs()
                    n_bad = int((diff > 1e-3).sum())
                else:
                    n_bad = int((merged[f"{col}_eeg"].astype(str) != merged[f"{col}_fmri"].astype(str)).sum())
                if n_bad:
                    self.logger.warning(
                        "Cross-modal QC mismatch sub-%s run-%s column=%s (n_bad=%d/%d).",
                        subject,
                        r,
                        col,
                        n_bad,
                        len(merged),
                    )
