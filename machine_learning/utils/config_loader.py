from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    import yaml
except ImportError:
    yaml = None

###################################################################
# Configuration Loading
###################################################################

def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    if config_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / "utils" / "ml_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        return {}
    
    if yaml is None:
        raise ImportError("PyYAML is required to load configuration files. Install with: pip install pyyaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config if config is not None else {}


def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    keys = path.split(".")
    value = config
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
        if value is None:
            return default
    return value


def get_general_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("general", {})


def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("data", {})


def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    model_configs = {
        "baseline": config.get("baseline", {}),
        "svm_rbf": config.get("svm_rbf", {}),
        "covariance": config.get("covariance", {}),
        "cnn": config.get("cnn", {}),
        "cnn_transformer": config.get("cnn_transformer", {}),
        "graph": config.get("graph", {}),
    }
    return model_configs.get(model_name, {})


def get_cv_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("cross_validation", {})


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("training", {})


def get_output_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("output", {})


def get_permutation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("permutation", {})


def apply_config_to_args(
    config: Dict[str, Any],
    args: Any,
    *,
    use_config_defaults: bool = True,
) -> None:
    if not use_config_defaults:
        return
    
    general = get_general_config(config)
    data = get_data_config(config)
    
    if not hasattr(args, "random_state") or args.random_state is None:
        args.random_state = general.get("random_state", 42)
    
    if not hasattr(args, "n_jobs") or args.n_jobs is None:
        args.n_jobs = general.get("n_jobs", -1)
    
    if not hasattr(args, "device") or args.device is None or args.device == "auto":
        args.device = general.get("device", "auto")
    
    if not hasattr(args, "target_signature") or args.target_signature is None:
        args.target_signature = general.get("target_signature", "nps")
    
    if not hasattr(args, "include_temperature"):
        args.include_temperature = general.get("include_temperature", False)
    
    if not hasattr(args, "subjects") or args.subjects is None:
        args.subjects = general.get("subjects")
    
    if not hasattr(args, "bands") or not args.bands:
        args.bands = data.get("bands", ["delta", "theta", "alpha", "beta", "gamma"])
    
    if not hasattr(args, "permutation_count") or args.permutation_count is None:
        perm_config = get_permutation_config(config)
        if perm_config.get("enabled", False):
            args.permutation_count = perm_config.get("count", 0)
        else:
            args.permutation_count = 0
    
    if not hasattr(args, "permutation_seed") or args.permutation_seed is None:
        perm_config = get_permutation_config(config)
        args.permutation_seed = perm_config.get("seed")

