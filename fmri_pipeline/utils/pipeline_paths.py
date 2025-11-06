#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


def _coerce_path(value: Optional[str | Path], fallback: Path) -> Path:
    if value is None:
        return fallback
    return Path(value)


@dataclass(frozen=True)
class PipelinePaths:
    work_root: Path
    outputs_root: Path
    qc_root: Path
    firstlevel_root: Path
    signatures_root: Path
    group_root: Path
    figures_root: Path
    reports_root: Path

    @classmethod
    def from_config(
        cls,
        config: Dict,
        work_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        qc_dir: Optional[str] = None,
    ) -> "PipelinePaths":
        outputs_cfg = config.get("outputs", {})

        pipeline_root = _coerce_path(
            output_dir,
            Path(outputs_cfg.get("pipeline_root_dir", outputs_cfg.get("glm_dir", "outputs"))),
        )
        work_root = _coerce_path(
            work_dir,
            Path(outputs_cfg.get("work_dir", pipeline_root.parent / "work")),
        )
        qc_root = _coerce_path(
            qc_dir,
            Path(outputs_cfg.get("qc_dir", pipeline_root / "qc")),
        )

        firstlevel_root = _coerce_path(
            outputs_cfg.get("glm_dir"), pipeline_root / "firstlevel"
        )
        signatures_root = _coerce_path(
            outputs_cfg.get("signatures_root_dir"), pipeline_root / "signatures"
        )
        group_root = _coerce_path(
            outputs_cfg.get("group_dir"), pipeline_root / "group"
        )
        figures_root = _coerce_path(
            outputs_cfg.get("figures_dir"), pipeline_root / "figures"
        )
        reports_root = _coerce_path(
            outputs_cfg.get("reports_dir"), pipeline_root / "reports"
        )

        return cls(
            work_root=work_root.resolve(),
            outputs_root=pipeline_root.resolve(),
            qc_root=qc_root.resolve(),
            firstlevel_root=firstlevel_root.resolve(),
            signatures_root=signatures_root.resolve(),
            group_root=group_root.resolve(),
            figures_root=figures_root.resolve(),
            reports_root=reports_root.resolve(),
        )

    ###################################################################
    # Basic Utilities
    ###################################################################
    
    @staticmethod
    def ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_core_roots(self) -> None:
        for path in {
            self.work_root,
            self.outputs_root,
            self.qc_root,
            self.firstlevel_root,
            self.signatures_root,
            self.group_root,
            self.figures_root,
            self.reports_root,
        }:
            self.ensure_dir(path)

    ###################################################################
    # Work Directories
    ###################################################################
    
    @property
    def index_dir(self) -> Path:
        return self.work_root / "index"

    def work_firstlevel_dir(self, subject: Optional[str] = None) -> Path:
        base = self.work_root / "firstlevel"
        return base if subject is None else base / subject

    ###################################################################
    # First-Level Outputs
    ###################################################################
    
    def firstlevel_subject_dir(self, subject: str) -> Path:
        return self.firstlevel_root / subject

    ###################################################################
    # Signature Outputs
    ###################################################################
    
    def signature_root(self, signature: str) -> Path:
        return self.signatures_root / signature

    def signature_stage_dir(self, signature: str, stage: str) -> Path:
        return self.signature_root(signature) / stage

    def harmonized_dir(self, signature: str, subject: Optional[str] = None) -> Path:
        base = self.signature_stage_dir(signature, "harmonized")
        return base if subject is None else base / subject

    def signature_scores_dir(self, signature: str, subject: Optional[str] = None) -> Path:
        base = self.signature_stage_dir(signature, "scores")
        return base if subject is None else base / subject

    def signature_metrics_dir(self, signature: str, subject: Optional[str] = None) -> Path:
        base = self.signature_stage_dir(signature, "metrics")
        return base if subject is None else base / subject

    ###################################################################
    # QC Helpers
    ###################################################################
    
    def qc_stage_dir(self, stage: str) -> Path:
        return self.qc_root / stage

