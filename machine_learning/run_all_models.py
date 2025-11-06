#!/usr/bin/env python
"""Orchestrate EEG->NPS model training scripts under a common configuration.

This utility launches all requested model training scripts (shallow/tabular regressors and
deep neural architectures) with consistent CLI arguments so
results are directly comparable. Each run receives a dedicated output directory,
and the orchestrator collates model-level metrics into a comparison table and
summary manifest for downstream analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from machine_learning.utils import (  # noqa: E402
    config_loader,
    feature_utils,
    io_utils,
    target_signatures,
)

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for a single downstream training script."""

    key: str
    script_name: str
    model_argument: Optional[str]
    description: str
    default_models: Optional[Sequence[str]] = None
    permutation_flag: Optional[str] = "--permutation-count"
    supports_device: bool = False
    supports_n_jobs: bool = True

    @property
    def script_path(self) -> Path:
        path = THIS_DIR / self.script_name
        if not path.exists():
            raise FileNotFoundError(f"Expected script not found: {path}")
        return path

    def build_command(self, args: argparse.Namespace, output_dir: Path) -> List[str]:
        """Assemble the subprocess command for this model specification."""

        cmd: List[str] = [sys.executable, str(self.script_path)]
        if self.model_argument:
            cmd.append(self.model_argument)

        if args.subjects:
            cmd += ["--subjects", *args.subjects]
        if args.bands:
            cmd += ["--bands", *args.bands]
        if args.include_temperature:
            cmd.append("--include-temperature")
        if args.random_state is not None:
            cmd += ["--random-state", str(args.random_state)]
        if self.supports_n_jobs and args.n_jobs is not None:
            cmd += ["--n-jobs", str(args.n_jobs)]
        if self.permutation_flag and args.permutation_count is not None:
            cmd += [self.permutation_flag, str(args.permutation_count)]
        if self.supports_device and args.device:
            cmd += ["--device", args.device]
        if self.default_models:
            cmd += ["--models", *self.default_models]
        if args.target_signature:
            cmd += ["--target-signature", args.target_signature]

        cmd += ["--output-dir", str(output_dir)]
        return cmd


MODEL_SPECS: Dict[str, ModelSpec] = {
    "baseline": ModelSpec(
        key="baseline",
        script_name="eeg_to_signature_shallow_models.py",
        model_argument="baseline",
        description="ElasticNet and Random Forest baseline regressors",
        default_models=("elasticnet", "random_forest"),
        permutation_flag="--permutation-per-model",
        supports_device=False,
        supports_n_jobs=True,
    ),
    "svm_rbf": ModelSpec(
        key="svm_rbf",
        script_name="eeg_to_signature_shallow_models.py",
        model_argument="svm_rbf",
        description="RBF-kernel support vector regression",
        supports_device=False,
        supports_n_jobs=True,
    ),
    "covariance": ModelSpec(
        key="covariance",
        script_name="eeg_to_signature_shallow_models.py",
        model_argument="covariance",
        description="Covariance tangent-space regressors",
        default_models=("elasticnet", "ridge"),
        supports_device=False,
        supports_n_jobs=True,
    ),
    "cnn": ModelSpec(
        key="cnn",
        script_name="eeg_to_signature_deep_models.py",
        model_argument="cnn",
        description="1D convolutional neural network",
        supports_device=True,
        supports_n_jobs=True,
    ),
    "cnn_transformer": ModelSpec(
        key="cnn_transformer",
        script_name="eeg_to_signature_deep_models.py",
        model_argument="cnn_transformer",
        description="Hybrid CNN + Transformer architecture",
        supports_device=True,
        supports_n_jobs=True,
    ),
    "graph": ModelSpec(
        key="graph",
        script_name="eeg_to_signature_deep_models.py",
        model_argument="graph",
        description="Graph neural network on electrode connectivity",
        supports_device=True,
        supports_n_jobs=True,
    ),
}


def _setup_logger(output_root: Path) -> logging.Logger:
    logger = logging.getLogger("eeg_to_signature_orchestrator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_root / "orchestrator.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _validate_bands(bands: Sequence[str]) -> List[str]:
    cleaned = []
    for band in bands:
        band_lower = band.lower()
        if band_lower not in feature_utils.SUPPORTED_BANDS:
            raise ValueError(f"Unsupported band '{band}'. Supported bands: {feature_utils.SUPPORTED_BANDS}.")
        cleaned.append(band_lower)
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple EEG->NPS training scripts with shared configuration."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(MODEL_SPECS.keys()),
        default=list(MODEL_SPECS.keys()),
        help="Model groups to execute (default: all).",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Subject identifiers (with or without sub- prefix). Defaults to all available.",
    )
    parser.add_argument(
        "--bands",
        nargs="*",
        default=list(feature_utils.DEFAULT_BANDS),
        help="EEG frequency bands to include (subset of %s)." % (feature_utils.SUPPORTED_BANDS,),
    )
    parser.add_argument(
        "--target-signature",
        type=str,
        default=target_signatures.DEFAULT_TARGET_KEY,
        choices=sorted(target_signatures.TARGET_SIGNATURES),
        help="fMRI signature to decode (default: %(default)s).",
    )
    parser.add_argument(
        "--include-temperature",
        action="store_true",
        help="Include stimulus temperature as an additional predictor/covariate.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible CV splits.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for GridSearchCV (passed to compatible scripts).",
    )
    parser.add_argument(
        "--permutation-count",
        type=int,
        default=0,
        help="Number of label permutations for significance testing (0 disables).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device for PyTorch-based models (auto selects CUDA when available).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Directory to collect orchestrator outputs. Defaults to machine_learning/outputs/comparison_<timestamp>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing downstream scripts.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file. Defaults to utils/ml_config.yaml if not specified.",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Disable loading configuration from YAML file.",
    )
    args = parser.parse_args()
    
    if not args.no_config:
        config = config_loader.load_config(args.config)
        config_loader.apply_config_to_args(config, args)
    
    return args


def main() -> None:
    args = parse_args()
    target = target_signatures.get_target_signature(args.target_signature)
    bands = _validate_bands(args.bands)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = THIS_DIR / "outputs" / f"comparison_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(output_root)
    logger.info("Orchestrator output root: %s", output_root)
    logger.info("Target signature: %s (%s)", target.display_name, target.short_name)
    logger.info(
        "Running models: %s | Bands: %s | Include temperature: %s | Random state: %d | Permutations: %d",
        ", ".join(args.models),
        ", ".join(bands),
        args.include_temperature,
        args.random_state,
        args.permutation_count,
    )

    run_records: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    metric_names: set[str] = set()

    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_output_dir = output_root / f"{spec.key}_{run_stamp}"
        command = spec.build_command(args, model_output_dir)
        command_str = shlex.join(command)

        record: Dict[str, Any] = {
            "model_key": spec.key,
            "description": spec.description,
            "script": spec.script_name,
            "command": command,
            "command_str": command_str,
            "target_signature": target.key,
            "target_display_name": target.display_name,
            "output_dir": str(model_output_dir),
        }

        if args.dry_run:
            logger.info("[DRY RUN] %s", command_str)
            record["status"] = "dry_run"
            run_records.append(record)
            continue

        model_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Launching %s: %s", spec.key, command_str)
        start_time = datetime.now()
        completed = subprocess.run(command, cwd=REPO_ROOT)
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()
        record.update(
            {
                "status": "success" if completed.returncode == 0 else "failed",
                "return_code": completed.returncode,
                "started_at": start_time.isoformat(),
                "finished_at": end_time.isoformat(),
                "duration_seconds": duration,
            }
        )

        summary_path = model_output_dir / "summary.json"
        summary_data: Optional[Dict[str, Any]] = None
        if completed.returncode == 0 and summary_path.exists():
            try:
                summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
                record["summary_path"] = str(summary_path)
                record["summary"] = summary_data
            except Exception as exc:  # pragma: no cover - best-effort logging only
                logger.warning("Failed to parse summary for %s: %s", spec.key, exc)
        elif completed.returncode == 0:
            logger.warning("Summary file missing for %s at %s", spec.key, summary_path)

        if summary_data:
            output_dir_display = str(model_output_dir.relative_to(REPO_ROOT))
            for model_entry in summary_data.get("models", []):
                metrics = model_entry.get("metrics", {}) or {}
                row: Dict[str, Any] = {
                    "model_group": spec.key,
                    "model_name": model_entry.get("name"),
                    "script": spec.script_name,
                    "output_dir": output_dir_display,
                }
                for metric_name, metric_value in metrics.items():
                    row[metric_name] = metric_value
                    metric_names.add(metric_name)
                comparison_rows.append(row)

        run_records.append(record)

    comparison_table_path: Optional[Path] = None
    if comparison_rows:
        base_fields = ["model_group", "model_name", "script", "output_dir"]
        header = base_fields + sorted(metric_names)
        comparison_table_path = output_root / "comparison_metrics.tsv"
        with comparison_table_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for row in comparison_rows:
                writer.writerow({field: row.get(field) for field in header})
        logger.info("Wrote comparison metrics table: %s", comparison_table_path)

    comparison_summary = {
        "created_at": datetime.now().isoformat(),
        "output_root": str(output_root),
        "models_requested": args.models,
        "subjects": args.subjects,
        "target_signature": target.key,
        "target_display_name": target.display_name,
        "target_short_name": target.short_name,
        "bands": bands,
        "include_temperature": args.include_temperature,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
        "permutation_count": args.permutation_count,
        "device": args.device,
        "runs": run_records,
        "comparison_metrics_file": comparison_table_path.name if comparison_table_path else None,
    }

    comparison_summary_path = output_root / "comparison_summary.json"
    io_utils.write_json(comparison_summary_path, comparison_summary)
    logger.info("Comparison summary written to %s", comparison_summary_path)

    io_utils.create_run_manifest(
        output_dir=output_root,
        args=args,
        repo_root=REPO_ROOT,
        additional={
            "comparison_summary_file": comparison_summary_path.name,
            "target_signature": target.key,
            "target_display_name": target.display_name,
            "target_short_name": target.short_name,
        },
    )

    logger.info("All requested runs processed. Status recap: %s", ", ".join(f"{r['model_key']}={r['status']}" for r in run_records))


if __name__ == "__main__":
    main()
