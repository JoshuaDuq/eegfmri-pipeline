"""
EEG preprocessing modules.

Submodules:
- raw_to_bids: Convert BrainVision EEG to BIDS format
- merge_behavior_to_events: Merge behavioral data into BIDS events
"""

from eeg_pipeline.preprocessing.raw_to_bids import (
    convert_one,
    find_brainvision_vhdrs,
    parse_subject_id,
    ensure_dataset_description,
)
from eeg_pipeline.preprocessing.merge_behavior_to_events import (
    merge_behavior_events_all_subjects,
)

__all__ = [
    "convert_one",
    "find_brainvision_vhdrs",
    "parse_subject_id",
    "ensure_dataset_description",
    "merge_behavior_events_all_subjects",
]

