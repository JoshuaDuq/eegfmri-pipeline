from pathlib import Path
from eeg_pipeline.cli.commands.base import detect_feature_availability

features_dir = Path("/Users/joduq24/Desktop/EEG_fMRI_Pipeline/eeg_pipeline/data/derivatives/sub-0000/eeg/features")
# Ensure directory exists for test
features_dir.mkdir(parents=True, exist_ok=True)
stats_dir = features_dir.parent / "stats"
stats_dir.mkdir(parents=True, exist_ok=True)

result = detect_feature_availability(features_dir)

import json
print(json.dumps(result, indent=2))
