#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from utils import get_log_function, load_config


def _stdout_log(message: str, level: str = "INFO") -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}", flush=True)


log: Callable[[str, str], None] = _stdout_log
LOG_FILE: Optional[Path] = None


def run_step(script: str, step_args: List[str], description: str, env: Optional[Dict[str, str]] = None) -> None:
    result = subprocess.run(["python", script, *step_args], check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed (exit {result.returncode})")


def main():
    parser = argparse.ArgumentParser(description="Run the fMRI NPS analysis pipeline end-to-end")
    parser.add_argument("--config", default="utils/config.yaml", help="Configuration file (default: utils/config.yaml)")
    parser.add_argument("--subject", default=None, help="Process single subject (default: all subjects in config)")
    parser.add_argument("--work-dir", default="work", help="Working directory (default: work)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--qc-dir", default="qc", help="QC directory (default: qc)")
    parser.add_argument("--log-dir", default=None, help="Directory for log files (default: ./logs or $NPS_LOG_DIR)")
    args = parser.parse_args()

    log_dir_input = args.log_dir or os.environ.get("NPS_LOG_DIR")
    log_dir = Path(log_dir_input).expanduser() if log_dir_input else Path.cwd() / "logs"
    os.environ["NPS_LOG_DIR"] = str(log_dir)

    global log, LOG_FILE
    log, LOG_FILE = get_log_function(Path(__file__).stem, log_dir=str(log_dir))

    common_env = os.environ.copy()
    start_time = datetime.now()
    try:
        config = load_config(args.config)
    except Exception as exc:
        log(f"Config: {exc}", "ERROR")
        return 1

    subjects: List[str] = [args.subject] if args.subject else list(config["subjects"])

    bids_root = Path(config["bids_root"])
    eeg_root = Path(config.get("eeg_derivatives_root", "")).resolve() if config.get("eeg_derivatives_root") else None

    subject_args = ["--subject", args.subject] if args.subject else []

    steps = [
        ("01_prepare_data.py", ["--config", args.config, "--work-dir", args.work_dir, "--qc-dir", args.qc_dir, *subject_args], "Prepare data: split events and discover inputs"),
        ("02_prepare_glm_inputs.py", ["--config", args.config, "--work-dir", args.work_dir, "--qc-dir", args.qc_dir, *subject_args], "Prepare GLM inputs: confounds and design matrices"),
        ("03_first_level_glm.py", ["--config", args.config, "--work-dir", args.work_dir, "--output-dir", args.output_dir, "--qc-dir", args.qc_dir, *subject_args], "First-level GLM: fit, combine runs, harmonize"),
        ("04_score_signatures.py", ["--config", args.config, "--work-dir", args.work_dir, "--output-dir", args.output_dir, "--qc-dir", args.qc_dir, *subject_args], "Score signatures: condition and trial-level"),
        ("05_metrics_and_stats.py", ["--config", args.config, "--output-dir", args.output_dir, "--qc-dir", args.qc_dir, *subject_args], "Metrics and statistics: subject and group"),
        (
            "06_visualization.py",
            [
                "--config",
                args.config,
                "--scores-dir",
                f"{args.output_dir}/nps_scores",
                "--figures-dir",
                f"{args.output_dir}/figures",
                "--group-dir",
                f"{args.output_dir}/group",
            ],
            "Generate figures",
        ),
        ("07_qc_collation.py", ["--config", args.config, "--work-dir", args.work_dir, "--outputs-dir", args.output_dir], "Collate QC"),
    ]

    for script, step_args, description in steps:
        log(description)
        try:
            run_step(script, step_args, description, env=common_env)
        except Exception as exc:
            log(f"{description}: {exc}", "ERROR")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
