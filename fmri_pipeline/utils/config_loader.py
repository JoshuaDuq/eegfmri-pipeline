#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path is None:
        utils_dir = Path(__file__).parent
        config_path = str(utils_dir / "config.yaml")
    config_file = Path(config_path)
    with open(config_file) as f:
        return _resolve_paths(yaml.safe_load(f), config_file.parent)


def _resolve_paths(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    pipeline_root = base_dir.parent
    
    def _resolve(value: Any, use_pipeline_root: bool = False) -> Any:
        if not isinstance(value, str):
            return value
        root = pipeline_root if use_pipeline_root else base_dir
        return str((root / value).resolve())

    root_keys = ("bids_root", "fmriprep_root", "derivatives_root")
    for key in root_keys:
        if key in config:
            config[key] = _resolve(config[key])
    
    if "resources" in config:
        config["resources"] = {
            k: _resolve(v, use_pipeline_root=True) if k.endswith(("_path", "_dir")) else v
            for k, v in config["resources"].items()
        }
    
    if "outputs" in config:
        config["outputs"] = {
            k: _resolve(v) if k.endswith(("_path", "_dir")) else v
            for k, v in config["outputs"].items()
        }
    
    if "logging" in config and "log_dir" in config["logging"]:
        config["logging"]["log_dir"] = _resolve(config["logging"]["log_dir"])
    
    return config


def validate_config(config: Dict[str, Any], check_files: bool = True) -> None:
    required = {"bids_root", "fmriprep_root", "subjects", "task", "runs", "glm", "confounds"}
    if missing := required - config.keys():
        raise KeyError(f"Missing config keys: {sorted(missing)}")

    glm = config["glm"]
    for key in ("tr", "temp_labels", "nuisance_events"):
        if key not in glm:
            raise KeyError(f"Missing GLM setting: {key}")

    if len(config["confounds"].get("motion_24_params", [])) != 24:
        raise ValueError("Expected 24 motion parameters")

    if check_files:
        for name, root in [("BIDS", config["bids_root"]), ("fMRIPrep", config["fmriprep_root"])]:
            if not Path(root).exists():
                raise FileNotFoundError(f"{name} root not found: {root}")


def _select_unique_existing(paths: Iterable[Path]) -> Path:
    existing = [p for p in paths if p.exists()]
    if len(existing) != 1:
        raise FileNotFoundError(f"Expected 1 file, found {len(existing)}: {existing or list(paths)}")
    return existing[0]


def get_subject_files(config: Dict[str, Any], subject: str, run: int, file_type: str = "bold") -> Path:
    task = config["task"]
    space = config["space"]
    res = config.get("resolution", "2")
    
    templates = {
        "bold": (config["fmriprep_root"], "func", 
                f"{subject}_task-{task}_run-{{run}}_space-{space}_res-{res}_desc-preproc_bold.nii.gz"),
        "mask": (config["fmriprep_root"], "func",
                f"{subject}_task-{task}_run-{{run}}_space-{space}_res-{res}_desc-brain_mask.nii.gz"),
        "confounds": (config["fmriprep_root"], "func",
                     f"{subject}_task-{task}_run-{{run}}_desc-confounds_timeseries.tsv"),
        "events": (config["bids_root"], "func",
                  f"{subject}_task-{task}_run-{{run}}_events.tsv"),
    }
    
    if file_type in templates:
        root, subdir, pattern = templates[file_type]
        candidate_paths = [
            Path(root) / subject / subdir / pattern.format(run=f"{run:02d}"),
            Path(root) / subject / subdir / pattern.format(run=str(run))
        ]
        return _select_unique_existing(candidate_paths)
    
    if file_type == "anat":
        anat_acq = config.get('anat_acquisition', 'mprageipat2')
        path = Path(config["fmriprep_root"]) / subject / "anat" / \
               f"{subject}_acq-{anat_acq}_space-{space}_res-{res}_desc-preproc_T1w.nii.gz"
        if not path.exists():
            raise FileNotFoundError(f"Anatomical not found: {path}")
        return path
    
    raise ValueError(f"Unknown file_type: {file_type}")


def get_confound_columns(config: Dict[str, Any], include_motion: bool = True, 
                         include_compcor: Optional[bool] = None, include_physio: bool = False) -> List[str]:
    columns: List[str] = []
    
    if include_motion:
        columns.extend(config["confounds"]["motion_24_params"])
    
    compcor_config = config["confounds"].get("compcor", {})
    if include_compcor is None:
        include_compcor = compcor_config.get("enabled", False)
    
    if include_compcor:
        n_components = compcor_config.get("n_components", 5)
        method = compcor_config.get("method", "acompcor")
        
        if method in {"acompcor", "both"}:
            columns.extend([f"a_comp_cor_{i:02d}" for i in range(n_components)])
        if method in {"tcompcor", "both"}:
            columns.extend([f"t_comp_cor_{i:02d}" for i in range(n_components)])
    
    if include_physio:
        columns.extend(config["confounds"].get("physiological", []))
    
    return columns


def print_config_summary(config: Dict[str, Any]) -> None:
    print(f"Subjects: {len(config['subjects'])}, Task: {config['task']}, Runs: {len(config['runs'])}")
    print(f"TR: {config['glm']['tr']}s, HRF: {config['glm']['hrf']['model']}, Space: {config['space']}")
    print(f"Temps: {len(config['glm']['temp_labels'])}, Nuisance: {len(config['glm']['nuisance_events'])}")


if __name__ == "__main__":
    config = load_config()
    print_config_summary(config)
    validate_config(config, check_files=True)
    print("Config validated")

