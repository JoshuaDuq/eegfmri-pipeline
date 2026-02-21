"""
Docker-based BEM/Trans Generation
=================================

Automatically generates BEM model, solution, and coregistration transform
files using Docker with FreeSurfer and MNE-Python.

Requires:
- Docker installed and running
- FreeSurfer license file
- freesurfer-mne:7.4.1 Docker image (or custom image with FreeSurfer + MNE)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from eeg_pipeline.utils.config.loader import get_config_value

logger = logging.getLogger(__name__)
FS_LICENSE_ENV_VAR = "EEG_PIPELINE_FREESURFER_LICENSE"


def check_docker_available() -> bool:
    """Check if Docker is installed and running."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _auto_detect_fs_license() -> Optional[Path]:
    """Auto-detect FreeSurfer license from common locations."""
    home = Path.home()
    candidates = [
        home / "license.txt",
        home / ".freesurfer" / "license.txt",
        home / "freesurfer" / "license.txt",
        Path("/usr/local/freesurfer/license.txt"),
        Path("/opt/freesurfer/license.txt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Auto-detected FreeSurfer license: {candidate}")
            return candidate
    return None


def get_fs_license_path(config: Any) -> Optional[Path]:
    """Get FreeSurfer license path from env, config, or auto-detect."""
    env_license = os.getenv(FS_LICENSE_ENV_VAR)
    if env_license and str(env_license).strip():
        return Path(env_license).expanduser()

    paths_cfg = get_config_value(config, "paths", {})
    if isinstance(paths_cfg, dict):
        fs_license = paths_cfg.get("freesurfer_license")
    else:
        fs_license = None

    if fs_license:
        return Path(fs_license).expanduser()

    return _auto_detect_fs_license()


def get_bem_generation_config(config: Any) -> Dict[str, Any]:
    """Extract BEM/Trans generation config from main config."""
    src_cfg = get_config_value(config, "feature_engineering", {})
    if isinstance(src_cfg, dict):
        src_cfg = src_cfg.get("sourcelocalization", {}) or {}
    else:
        src_cfg = {}

    bem_cfg = src_cfg.get("bem_generation", {}) or {}

    return {
        "create_trans": bool(bem_cfg.get("create_trans", False)),
        "create_model": bool(bem_cfg.get("create_model", False)),
        "create_solution": bool(bem_cfg.get("create_solution", False)),
        "allow_identity_trans": bool(bem_cfg.get("allow_identity_trans", False)),
        "docker_image": bem_cfg.get("docker_image", "freesurfer-mne:7.4.1"),
        "ico": int(bem_cfg.get("ico", 4)),
        "conductivity": bem_cfg.get("conductivity", [0.3, 0.006, 0.3]),
    }


def generate_coregistration_transform(
    subject: str,
    subjects_dir: Path,
    fs_license_path: Path,
    eeg_info_path: Optional[Path] = None,
    docker_image: str = "freesurfer-mne:7.4.1",
    overwrite: bool = True,
    *,
    allow_identity_trans: bool = False,
) -> Optional[Path]:
    """
    Generate coregistration transform (EEG↔MRI alignment) using Docker.

    Uses MNE's coregistration with fiducials from the FreeSurfer subject.
    For EEG-only (no digitization), uses fsaverage fiducials scaled to subject.

    Parameters
    ----------
    subject : str
        FreeSurfer subject name (e.g., "sub-0000")
    subjects_dir : Path
        Path to FreeSurfer SUBJECTS_DIR containing the subject folder
    fs_license_path : Path
        Path to FreeSurfer license.txt file
    eeg_info_path : Path, optional
        Path to EEG info file (.fif) with digitization points (if available)
    docker_image : str
        Docker image name with FreeSurfer + MNE installed
    overwrite : bool
        Whether to overwrite existing files

    Returns
    -------
    trans_path : Path or None
        Path to generated transform file
    """
    # Reserved for future support of subject-specific EEG digitization points.
    _ = eeg_info_path
    if not check_docker_available():
        raise RuntimeError(
            "Docker is not available. Please install Docker and ensure it is running. "
            "See https://docs.docker.com/get-docker/"
        )

    subjects_dir = Path(subjects_dir).resolve()
    fs_license_path = Path(fs_license_path).resolve()

    if not subjects_dir.exists():
        raise FileNotFoundError(f"SUBJECTS_DIR not found: {subjects_dir}")
    if not fs_license_path.exists():
        raise FileNotFoundError(f"FreeSurfer license not found: {fs_license_path}")

    subject_dir = subjects_dir / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    bem_dir = subject_dir / "bem"
    bem_dir.mkdir(parents=True, exist_ok=True)
    trans_path = bem_dir / f"{subject}-trans.fif"

    # Check if trans file already exists
    if trans_path.exists() and not overwrite:
        logger.info(f"Using existing trans file: {trans_path}")
        return trans_path

    # Safety check: refuse to auto-generate identity transform unless explicitly allowed
    if not allow_identity_trans:
        raise RuntimeError(
            "Refusing to auto-generate an EEG↔MRI transform because doing so without digitization/coregistration "
            "is scientifically invalid. Provide an existing `*-trans.fif` (recommended), or set "
            "`feature_engineering.sourcelocalization.bem_generation.allow_identity_trans=true` to force creation "
            "of an identity transform for debugging only."
        )

    python_script = f"""import mne
import numpy as np

rsubject = '{subject}'
subjects_dir = '/subjects'

# Get fiducials from FreeSurfer subject
fiducials_path = f'/subjects/{subject}/bem/{subject}-fiducials.fif'
try:
    fiducials, coord_frame = mne.io.read_fiducials(fiducials_path)
except Exception:
    # Create fiducials from fsaverage scaled to subject
    from mne.coreg import get_mni_fiducials
    fiducials = get_mni_fiducials(subject, subjects_dir=subjects_dir)

# Create identity transform (DEBUG ONLY; not valid for anatomy).
trans = mne.transforms.Transform('head', 'mri', trans=np.eye(4))
mne.write_trans(f'/subjects/{subject}/bem/{subject}-trans.fif', trans, overwrite={overwrite})
print(f'Created transform: /subjects/{subject}/bem/{subject}-trans.fif')
"""

    docker_cmd = [
        "docker", "run", "--rm",
        "--platform", "linux/amd64",
        "-v", f"{subjects_dir}:/subjects",
        "-v", f"{fs_license_path}:/usr/local/freesurfer/.license",
        docker_image,
        "bash", "-lc",
        f"set -e; set +u; source $FREESURFER_HOME/SetUpFreeSurfer.sh; set -u; python -c \"{python_script}\"",
    ]

    logger.info(f"Running Docker trans generation for subject {subject}")
    logger.debug(f"Docker command: {' '.join(docker_cmd)}")

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            logger.error(f"Docker trans generation failed:\n{result.stderr}")
            raise RuntimeError(
                f"Docker trans generation failed with exit code {result.returncode}:\n{result.stderr}"
            )

        logger.info("Trans generation completed successfully")
        if result.stdout:
            logger.debug(f"Docker output:\n{result.stdout}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("Docker trans generation timed out after 10 minutes")

    return trans_path if trans_path.exists() else None


def generate_bem_model_and_solution(
    subject: str,
    subjects_dir: Path,
    fs_license_path: Path,
    docker_image: str = "freesurfer-mne:7.4.1",
    ico: int = 4,
    conductivity: Tuple[float, float, float] = (0.3, 0.006, 0.3),
    create_model: bool = True,
    create_solution: bool = True,
    overwrite: bool = True,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Generate BEM model and solution using Docker.

    Parameters
    ----------
    subject : str
        FreeSurfer subject name (e.g., "sub-0000")
    subjects_dir : Path
        Path to FreeSurfer SUBJECTS_DIR containing the subject folder
    fs_license_path : Path
        Path to FreeSurfer license.txt file
    docker_image : str
        Docker image name with FreeSurfer + MNE installed
    ico : int
        ICO downsampling level (4 = 5120 triangles per surface)
    conductivity : tuple
        Conductivity values for [scalp, skull, brain]
    create_model : bool
        Whether to create the BEM model (watershed surfaces)
    create_solution : bool
        Whether to create the BEM solution matrix
    overwrite : bool
        Whether to overwrite existing files

    Returns
    -------
    bem_model_path : Path or None
        Path to generated BEM model file
    bem_solution_path : Path or None
        Path to generated BEM solution file
    """
    if not check_docker_available():
        raise RuntimeError(
            "Docker is not available. Please install Docker and ensure it is running. "
            "See https://docs.docker.com/get-docker/"
        )

    subjects_dir = Path(subjects_dir).resolve()
    fs_license_path = Path(fs_license_path).resolve()

    if not subjects_dir.exists():
        raise FileNotFoundError(f"SUBJECTS_DIR not found: {subjects_dir}")
    if not fs_license_path.exists():
        raise FileNotFoundError(f"FreeSurfer license not found: {fs_license_path}")

    subject_dir = subjects_dir / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    bem_dir = subject_dir / "bem"
    n_tris = 5120 if ico == 4 else (20480 if ico == 5 else 1280)
    bem_model_name = f"{subject}-{n_tris}-{n_tris}-{n_tris}-bem.fif"
    bem_solution_name = f"{subject}-{n_tris}-{n_tris}-{n_tris}-bem-sol.fif"

    bem_model_path = bem_dir / bem_model_name
    bem_solution_path = bem_dir / bem_solution_name

    # Check if BEM files already exist
    model_exists = bem_model_path.exists() if create_model else False
    solution_exists = bem_solution_path.exists() if create_solution else False

    if not overwrite:
        if create_model and model_exists:
            logger.info(f"Using existing BEM model: {bem_model_path}")
        if create_solution and solution_exists:
            logger.info(f"Using existing BEM solution: {bem_solution_path}")

        if (not create_model or model_exists) and (not create_solution or solution_exists):
            return (
                bem_model_path if model_exists else None,
                bem_solution_path if solution_exists else None,
            )

    if not create_model and not create_solution:
        logger.info("Neither create_model nor create_solution requested, skipping BEM generation")
        return (
            bem_model_path if bem_model_path.exists() else None,
            bem_solution_path if bem_solution_path.exists() else None,
        )

    cond_str = f"[{conductivity[0]}, {conductivity[1]}, {conductivity[2]}]"

    python_script = f"""import mne
bem_model = mne.make_bem_model(
    subject="{subject}",
    ico={ico},
    subjects_dir="/subjects",
    conductivity={cond_str}
)
mne.write_bem_surfaces(
    "/subjects/{subject}/bem/{bem_model_name}",
    bem_model,
    overwrite={overwrite}
)
bem_solution = mne.make_bem_solution(
    "/subjects/{subject}/bem/{bem_model_name}"
)
mne.write_bem_solution(
    "/subjects/{subject}/bem/{bem_solution_name}",
    bem_solution,
    overwrite={overwrite}
)
"""

    bash_commands = []
    if create_model:
        bash_commands.append(f"mne watershed_bem --subject {subject} --overwrite")

    if create_model or create_solution:
        escaped_script = python_script.replace('"', '\\"')
        bash_commands.append(f'python -c "{escaped_script}"')

    full_script = " && ".join(bash_commands)

    docker_cmd = [
        "docker", "run", "--rm",
        "--platform", "linux/amd64",
        "-v", f"{subjects_dir}:/subjects",
        "-v", f"{fs_license_path}:/usr/local/freesurfer/.license",
        docker_image,
        "bash", "-lc",
        f"set -e; set +u; source $FREESURFER_HOME/SetUpFreeSurfer.sh; set -u; {full_script}",
    ]

    logger.info(f"Running Docker BEM generation for subject {subject}")
    logger.debug(f"Docker command: {' '.join(docker_cmd)}")

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            logger.error(f"Docker BEM generation failed:\n{result.stderr}")
            raise RuntimeError(
                f"Docker BEM generation failed with exit code {result.returncode}:\n{result.stderr}"
            )

        logger.info(f"BEM generation completed successfully")
        if result.stdout:
            logger.debug(f"Docker output:\n{result.stdout}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("Docker BEM generation timed out after 1 hour")

    return (
        bem_model_path if bem_model_path.exists() else None,
        bem_solution_path if bem_solution_path.exists() else None,
    )


def ensure_bem_and_trans_files(
    subject: str,
    subjects_dir: Path,
    config: Any,
    logger_instance: Optional[logging.Logger] = None,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Ensure BEM model, solution, and trans files exist, creating them if configured.

    This is the main entry point for automatic BEM/trans generation.

    Parameters
    ----------
    subject : str
        FreeSurfer subject name
    subjects_dir : Path
        Path to FreeSurfer SUBJECTS_DIR
    config : Any
        Pipeline configuration containing bem_generation settings
    logger_instance : Logger, optional
        Logger instance to use

    Returns
    -------
    trans_path : Path or None
        Path to coregistration transform file (None if not available)
    bem_model_path : Path or None
        Path to BEM model file (None if not available)
    bem_solution_path : Path or None
        Path to BEM solution file (None if not available)
    """
    log = logger_instance or logger

    bem_cfg = get_bem_generation_config(config)
    create_trans = bem_cfg["create_trans"]
    create_model = bem_cfg["create_model"]
    create_solution = bem_cfg["create_solution"]

    if not create_trans and not create_model and not create_solution:
        log.debug("BEM/Trans auto-generation disabled in config")
        return None, None, None

    # Get FS license from global config or auto-detect
    fs_license_path = get_fs_license_path(config)
    if not fs_license_path or not fs_license_path.exists():
        raise ValueError(
            "FreeSurfer license not found. Set paths.freesurfer_license in eeg_config.yaml "
            f"or {FS_LICENSE_ENV_VAR}.\n"
            "Get a free license at: https://surfer.nmr.mgh.harvard.edu/registration.html"
        )

    subjects_dir = Path(subjects_dir)
    docker_image = bem_cfg.get("docker_image", "freesurfer-mne:7.4.1")

    trans_path = None
    bem_model_path = None
    bem_solution_path = None

    if create_trans:
        trans_path = generate_coregistration_transform(
            subject=subject,
            subjects_dir=subjects_dir,
            fs_license_path=fs_license_path,
            docker_image=docker_image,
            overwrite=False,
            allow_identity_trans=bool(bem_cfg.get("allow_identity_trans", False)),
        )

    if create_model or create_solution:
        bem_model_path, bem_solution_path = generate_bem_model_and_solution(
            subject=subject,
            subjects_dir=subjects_dir,
            fs_license_path=fs_license_path,
            docker_image=docker_image,
            ico=bem_cfg.get("ico", 4),
            conductivity=tuple(bem_cfg.get("conductivity", [0.3, 0.006, 0.3])),
            create_model=create_model,
            create_solution=create_solution,
            overwrite=False,
        )

    return trans_path, bem_model_path, bem_solution_path


def ensure_bem_files(
    subject: str,
    subjects_dir: Path,
    config: Any,
    logger_instance: Optional[logging.Logger] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Ensure BEM model and solution files exist, creating them if configured.

    Backward-compatible wrapper around ensure_bem_and_trans_files.

    Parameters
    ----------
    subject : str
        FreeSurfer subject name
    subjects_dir : Path
        Path to FreeSurfer SUBJECTS_DIR
    config : Any
        Pipeline configuration containing bem_generation settings
    logger_instance : Logger, optional
        Logger instance to use

    Returns
    -------
    bem_model_path : Path or None
        Path to BEM model file (None if not available)
    bem_solution_path : Path or None
        Path to BEM solution file (None if not available)
    """
    _, bem_model_path, bem_solution_path = ensure_bem_and_trans_files(
        subject=subject,
        subjects_dir=subjects_dir,
        config=config,
        logger_instance=logger_instance,
    )
    return bem_model_path, bem_solution_path
