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

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
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


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate_text(value: str, max_chars: int = 12000) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _surface_geometry_report(path: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "path": str(path),
        "exists": bool(path.exists()),
    }
    if not path.exists():
        return report

    try:
        import mne
        import numpy as np

        rr, tris = mne.read_surface(str(path))
        rr = np.asarray(rr, dtype=float)
        tris = np.asarray(tris, dtype=int)
        report["n_vertices"] = int(rr.shape[0])
        report["n_faces"] = int(tris.shape[0])
        report["bbox_min"] = [float(v) for v in np.min(rr, axis=0).tolist()]
        report["bbox_max"] = [float(v) for v in np.max(rr, axis=0).tolist()]
    except Exception as exc:
        report["read_error"] = str(exc)
    return report


def _collect_bem_surface_reports(subjects_dir: Path, subject: str) -> Dict[str, Dict[str, Any]]:
    subject_bem_dir = Path(subjects_dir) / str(subject) / "bem"
    reports: Dict[str, Dict[str, Any]] = {}
    for surf_name in ("inner_skull", "outer_skull", "outer_skin"):
        primary = subject_bem_dir / f"{surf_name}.surf"
        watershed = subject_bem_dir / "watershed" / f"{subject}_{surf_name}_surface"
        reports[surf_name] = {
            "primary": _surface_geometry_report(primary),
            "watershed": _surface_geometry_report(watershed),
        }
    return reports


def _write_bem_failure_qc_artifact(
    *,
    subject: str,
    subjects_dir: Path,
    docker_cmd: list[str],
    returncode: Optional[int],
    started_at_utc: str,
    ended_at_utc: str,
    stderr: str,
    stdout: str,
) -> Path:
    subject_dir = Path(subjects_dir) / str(subject)
    qc_dir = subject_dir / "bem" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    qc_path = qc_dir / f"{subject}_bem_failure_{stamp}.json"

    payload: Dict[str, Any] = {
        "subject": str(subject),
        "subjects_dir": str(subjects_dir),
        "started_at_utc": str(started_at_utc),
        "ended_at_utc": str(ended_at_utc),
        "docker_command": list(docker_cmd),
        "returncode": int(returncode) if returncode is not None else None,
        "stderr_tail": _truncate_text(stderr),
        "stdout_tail": _truncate_text(stdout),
        "surface_reports": _collect_bem_surface_reports(subjects_dir, subject),
    }
    qc_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return qc_path


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


def _load_eeg_info_for_coregistration(
    *,
    eeg_info: Optional[Any],
    eeg_info_path: Optional[Path],
) -> Optional[Any]:
    """Load EEG info with digitization for coregistration if available."""
    if eeg_info is not None:
        return eeg_info
    if eeg_info_path is None:
        return None

    import mne

    info_path = Path(eeg_info_path).expanduser().resolve()
    if not info_path.exists():
        raise FileNotFoundError(f"EEG info path not found: {info_path}")

    suffixes = "".join(info_path.suffixes).lower()
    if suffixes.endswith("_epo.fif") or info_path.name.endswith("-epo.fif"):
        epochs = mne.read_epochs(info_path, preload=False, verbose="ERROR")
        return epochs.info
    return mne.io.read_info(info_path, verbose="ERROR")


def _count_headshape_points(info: Any) -> tuple[int, int, int]:
    """Count fiducial, extra headshape, and EEG dig points."""
    import mne

    dig = list(info.get("dig") or [])
    n_fids = sum(1 for d in dig if int(d.get("kind", -1)) == int(mne.io.constants.FIFF.FIFFV_POINT_CARDINAL))
    n_extra = sum(1 for d in dig if int(d.get("kind", -1)) == int(mne.io.constants.FIFF.FIFFV_POINT_EXTRA))
    n_eeg = sum(1 for d in dig if int(d.get("kind", -1)) == int(mne.io.constants.FIFF.FIFFV_POINT_EEG))
    return int(n_fids), int(n_extra), int(n_eeg)


def _generate_trans_from_eeg_info(
    *,
    subject: str,
    subjects_dir: Path,
    info: Any,
    trans_path: Path,
    overwrite: bool,
) -> Optional[Path]:
    """Generate head↔MRI transform from EEG digitization and MRI anatomy."""
    import numpy as np
    import mne

    n_fids, n_extra, n_eeg = _count_headshape_points(info)
    if n_fids < 3:
        raise RuntimeError(
            f"Insufficient fiducials for coregistration (found {n_fids}, need 3)."
        )

    coreg = mne.coreg.Coregistration(
        info=info,
        subject=subject,
        subjects_dir=subjects_dir,
        fiducials="auto",
    )
    coreg.fit_fiducials(verbose=False)

    if n_extra > 10:
        # Refine with head-shape ICP when enough head points are available.
        coreg.fit_icp(n_iterations=50, nasion_weight=10.0, verbose=False)
        coreg.omit_head_shape_points(distance=0.005)
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=False)
    elif n_eeg > 0:
        logger.warning(
            "Coregistration for %s is fiducial-only (%d fiducials, %d headshape points, %d EEG points). "
            "No extra headshape points were found; transform quality depends on fiducials.",
            subject,
            n_fids,
            n_extra,
            n_eeg,
        )

    trans_candidate = coreg.trans
    matrix: np.ndarray
    if isinstance(trans_candidate, dict) and "trans" in trans_candidate:
        matrix = np.asarray(trans_candidate["trans"], dtype=float)
    elif hasattr(trans_candidate, "trans"):
        matrix = np.asarray(getattr(trans_candidate, "trans"), dtype=float)
    else:
        matrix = np.asarray(trans_candidate, dtype=float)
    if matrix.shape != (4, 4):
        raise RuntimeError("Coregistration produced an invalid transform matrix shape.")
    if np.allclose(matrix, np.eye(4), atol=1e-9):
        raise RuntimeError(
            "Coregistration produced an identity-like transform, which is not valid for source localization."
        )

    trans = mne.transforms.Transform("head", "mri", trans=matrix)
    mne.write_trans(trans_path, trans, overwrite=overwrite)
    return trans_path if trans_path.exists() else None


def generate_coregistration_transform(
    subject: str,
    subjects_dir: Path,
    fs_license_path: Path,
    eeg_info_path: Optional[Path] = None,
    eeg_info: Optional[Any] = None,
    docker_image: str = "freesurfer-mne:7.4.1",
    overwrite: bool = True,
    *,
    allow_identity_trans: bool = False,
) -> Optional[Path]:
    """
    Generate coregistration transform (EEG↔MRI alignment).

    Preferred path uses MNE coregistration from EEG digitization/fiducials.
    Identity transforms are only allowed as explicit debug fallback.

    Parameters
    ----------
    subject : str
        FreeSurfer subject name (e.g., "sub-0000")
    subjects_dir : Path
        Path to FreeSurfer SUBJECTS_DIR containing the subject folder
    fs_license_path : Path
        Path to FreeSurfer license.txt file
    eeg_info_path : Path, optional
        Path to EEG info/epochs FIF containing digitization points
    eeg_info : Any, optional
        In-memory MNE Info object with digitization points
    docker_image : str
        Docker image name with FreeSurfer + MNE installed
    overwrite : bool
        Whether to overwrite existing files

    Returns
    -------
    trans_path : Path or None
        Path to generated transform file
    """
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

    # Preferred: generate a real transform from EEG digitization/coregistration data.
    info = _load_eeg_info_for_coregistration(
        eeg_info=eeg_info,
        eeg_info_path=eeg_info_path,
    )
    if info is not None:
        logger.info("Generating EEG↔MRI transform from EEG digitization for subject %s", subject)
        try:
            generated = _generate_trans_from_eeg_info(
                subject=subject,
                subjects_dir=subjects_dir,
                info=info,
                trans_path=trans_path,
                overwrite=overwrite,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to generate EEG↔MRI transform from digitization for {subject}: {exc}"
            ) from exc
        if generated is not None:
            logger.info("Coregistration transform created: %s", generated)
            return generated

    # Safety check: refuse to auto-generate identity transform unless explicitly allowed
    if not allow_identity_trans:
        raise RuntimeError(
            "Refusing to auto-generate an EEG↔MRI transform because no usable digitization/coregistration "
            "data were provided. Provide an existing `*-trans.fif` (recommended), pass EEG info with "
            "fiducials/digitization, or set "
            "`feature_engineering.sourcelocalization.bem_generation.allow_identity_trans=true` to force creation "
            "of an identity transform for debugging only."
        )

    if not check_docker_available():
        raise RuntimeError(
            "Docker is not available. Please install Docker and ensure it is running. "
            "See https://docs.docker.com/get-docker/"
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
import os
import shutil

subject = "{subject}"
ico = {ico}
cond_str = {cond_str}
subjects_dir = "/subjects"

def get_surf_path(name):
    return f"{{subjects_dir}}/{{subject}}/bem/{{name}}.surf"

for name in ["inner_skull", "outer_skull", "outer_skin"]:
    surf_path = get_surf_path(name)
    if not os.path.exists(surf_path):
        alt_path = f"{{subjects_dir}}/{{subject}}/bem/watershed/{{subject}}_{{name}}_surface"
        if os.path.exists(alt_path):
            shutil.copy(alt_path, surf_path)

bem_model = mne.make_bem_model(
    subject=subject,
    ico=ico,
    subjects_dir=subjects_dir,
    conductivity=cond_str,
)
mne.write_bem_surfaces(
    f"/subjects/{{subject}}/bem/{bem_model_name}",
    bem_model,
    overwrite={overwrite},
)
bem_solution = mne.make_bem_solution(
    f"/subjects/{{subject}}/bem/{bem_model_name}"
)
mne.write_bem_solution(
    f"/subjects/{{subject}}/bem/{bem_solution_name}",
    bem_solution,
    overwrite={overwrite},
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

    run_started_utc = _iso_utc_now()
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            logger.error(f"Docker BEM generation failed:\n{result.stderr}")
            qc_path = _write_bem_failure_qc_artifact(
                subject=subject,
                subjects_dir=subjects_dir,
                docker_cmd=docker_cmd,
                returncode=result.returncode,
                started_at_utc=run_started_utc,
                ended_at_utc=_iso_utc_now(),
                stderr=result.stderr,
                stdout=result.stdout,
            )
            raise RuntimeError(
                "Docker BEM generation failed "
                f"(exit code {result.returncode}). QC report: {qc_path}\n"
                f"{result.stderr}"
            )

        logger.info("BEM generation completed successfully")
        if result.stdout:
            logger.debug(f"Docker output:\n{result.stdout}")

    except subprocess.TimeoutExpired as exc:
        qc_path = _write_bem_failure_qc_artifact(
            subject=subject,
            subjects_dir=subjects_dir,
            docker_cmd=docker_cmd,
            returncode=None,
            started_at_utc=run_started_utc,
            ended_at_utc=_iso_utc_now(),
            stderr=f"Docker BEM generation timed out after 1 hour: {exc}",
            stdout="",
        )
        raise RuntimeError(
            f"Docker BEM generation timed out after 1 hour. QC report: {qc_path}"
        ) from exc

    return (
        bem_model_path if bem_model_path.exists() else None,
        bem_solution_path if bem_solution_path.exists() else None,
    )


def ensure_bem_and_trans_files(
    subject: str,
    subjects_dir: Path,
    config: Any,
    logger_instance: Optional[logging.Logger] = None,
    eeg_info_path: Optional[Path] = None,
    eeg_info: Optional[Any] = None,
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
    eeg_info_path : Path, optional
        Path to EEG info/epochs FIF for trans coregistration
    eeg_info : Any, optional
        In-memory MNE Info object for trans coregistration

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

    if create_trans:
        trans_path = generate_coregistration_transform(
            subject=subject,
            subjects_dir=subjects_dir,
            fs_license_path=fs_license_path,
            eeg_info_path=eeg_info_path,
            eeg_info=eeg_info,
            docker_image=docker_image,
            overwrite=False,
            allow_identity_trans=bool(bem_cfg.get("allow_identity_trans", False)),
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
