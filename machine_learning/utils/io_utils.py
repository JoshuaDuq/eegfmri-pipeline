import argparse
import importlib
import json
import logging
import math
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

###################################################################
# Logging Setup
###################################################################

def setup_logging(output_dir: Path, logger_name: str = "eeg_to_signature") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "train.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


###################################################################
# JSON Serialization
###################################################################

def sanitize_for_json(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(sanitize_for_json(payload), indent=2), encoding="utf-8")


def _stringify_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _stringify_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_stringify_for_json(v) for v in value]
    return value


###################################################################
# Package Version Collection
###################################################################

PACKAGE_VERSION_TARGETS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "torch": "torch",
}

THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]


def _collect_package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for module_name, display_name in PACKAGE_VERSION_TARGETS.items():
        try:
            module = importlib.import_module(module_name)
        except Exception:
            versions[display_name] = None
            continue
        version = getattr(module, "__version__", None)
        if version is None and hasattr(module, "version"):
            attr = getattr(module, "version")
            try:
                version = attr() if callable(attr) else attr
            except Exception:
                version = None
        versions[display_name] = str(version) if version is not None else None
    return versions


def _collect_thread_limits() -> Dict[str, Optional[str]]:
    return {var: os.environ.get(var) for var in THREAD_ENV_VARS}


###################################################################
# Git Metadata
###################################################################

def _gather_git_metadata(repo_root: Path) -> Dict[str, Optional[str]]:
    metadata: Dict[str, Optional[str]] = {
        "commit": None,
        "branch": None,
        "dirty": None,
    }

    def _run_git(args: Sequence[str]) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return None
        if completed.returncode != 0:
            return None
        output = completed.stdout.strip()
        return output or None

    metadata["commit"] = _run_git(["rev-parse", "HEAD"])
    metadata["branch"] = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    metadata["dirty"] = "true" if status else ("false" if status == "" else None)
    return metadata


###################################################################
# Run Manifest Creation
###################################################################

def create_run_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    repo_root: Path,
    *,
    additional: Optional[Dict[str, Any]] = None,
) -> Path:
    cli_arguments = {k: _stringify_for_json(v) for k, v in vars(args).items()}
    manifest = {
        "created_at": datetime.now().isoformat(),
        "script": str(Path(__file__).resolve().parent.parent),
        "working_directory": str(Path.cwd()),
        "cli_arguments": cli_arguments,
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(),
            "thread_env": _collect_thread_limits(),
            "random_state": getattr(args, "random_state", None),
            "permutation_seed": getattr(args, "permutation_seed", None),
        },
        "git": _gather_git_metadata(repo_root),
        "packages": _collect_package_versions(),
    }
    if additional:
        manifest["analysis"] = _stringify_for_json(additional)

    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path

