import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
import pandas as pd


###################################################################
# Inventory Management
###################################################################

def load_inventory(inventory_path: Path) -> Dict:
    with open(inventory_path) as f:
        return json.load(f)


def sanitize_for_json(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    return obj


def save_inventory(inventory: Dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_for_json(inventory)
    output_path.write_text(json.dumps(sanitized, indent=2))


###################################################################
# BOLD Data
###################################################################

def get_bold_n_volumes(bold_path: Path) -> int:
    img = nib.load(str(bold_path))
    return img.shape[3] if img.ndim == 4 else 1


def get_bold_info(bold_path: Path) -> Tuple[int, Optional[float]]:
    img = nib.load(str(bold_path))
    if img.ndim == 4:
        n_volumes = img.shape[3]
        tr = img.header.get_zooms()[3]
    else:
        n_volumes = 1
        tr = None
    return n_volumes, tr


###################################################################
# Confounds
###################################################################

def load_confounds(confounds_path: Path) -> pd.DataFrame:
    return pd.read_csv(confounds_path, sep="\t")


###################################################################
# Events
###################################################################

def get_events_paths(inventory: Dict) -> List[Path]:
    return [
        Path(run['files']['events']['path'])
        for run in inventory['runs'].values()
        if run['complete'] and 'events' in run['files']
    ]


def extract_vas_ratings(events_paths: List[Path], temp_label: str) -> Dict:
    vas_values = []
    for events_path in events_paths:
        if not events_path.exists():
            continue
        try:
            events = pd.read_csv(events_path, sep='\t')
            temp_events = events[events['trial_type'] == temp_label]
            vas_col = 'vas_0_200' if 'vas_0_200' in events.columns else 'rating'
            if vas_col in events.columns:
                vas_values.extend(temp_events[vas_col].dropna().tolist())
        except (pd.errors.EmptyDataError, KeyError, ValueError):
            continue
    
    if not vas_values:
        return {
            'n_trials': 0,
            'mean_vas': np.nan,
            'std_vas': np.nan,
            'median_vas': np.nan
        }
    
    return {
        'n_trials': len(vas_values),
        'mean_vas': float(np.mean(vas_values)),
        'std_vas': float(np.std(vas_values)),
        'median_vas': float(np.median(vas_values))
    }


###################################################################
# Subject ID
###################################################################

def normalize_subject_id(subject: Optional[str]) -> str:
    if subject is None:
        return "sub-0001"
    
    subject = subject.strip()
    if subject.startswith("sub-"):
        suffix = subject[4:].strip()
    else:
        suffix = subject.strip()
    
    if suffix.isdigit():
        suffix = f"{int(suffix):04d}"
    
    return f"sub-{suffix}"


###################################################################
# EEG Drop Log
###################################################################

def load_eeg_drop_log(drop_log_path: Path) -> pd.DataFrame:
    if not drop_log_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(drop_log_path, sep='\t')
        if 'run' in df.columns and 'trial_number' in df.columns:
            return df
        return pd.DataFrame()
    except (pd.errors.EmptyDataError, pd.errors.ParserError, IOError):
        return pd.DataFrame()

