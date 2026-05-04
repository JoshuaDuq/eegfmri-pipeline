"""EEG raw (BrainVision) to BIDS conversion (analysis-layer)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import mne

from eeg_pipeline.analysis.utilities.bids_metadata import (
    ensure_participants_tsv,
    ensure_task_events_json,
)
from eeg_pipeline.utils.data.preprocessing import (
    ensure_dataset_description,
    filter_annotations,
    find_brainvision_vhdrs,
    get_run_index,
    parse_subject_id,
    set_channel_types,
    set_montage,
    trim_to_first_volume,
)

logger = logging.getLogger(__name__)


def _has_volume_triggers(raw: mne.io.BaseRaw) -> bool:
    if len(raw.annotations) == 0:
        return False
    for desc in raw.annotations.description:
        s = str(desc).strip()
        if s.startswith("Volume/") or s.startswith("Volume"):
            return True
        if s.startswith("Volume/V"):
            return True
    return False


def run_raw_to_bids(
    source_root: Path,
    bids_root: Path,
    task: str,
    subjects: Optional[List[str]] = None,
    montage: str = "easycap-M1",
    line_freq: float = 60.0,
    overwrite: bool = False,
    zero_base_onsets: bool = False,
    do_trim_to_first_volume: bool = False,
    event_prefixes: Optional[List[str]] = None,
    keep_all_annotations: bool = False,
    *,
    _logger: Optional[logging.Logger] = None,
) -> int:
    """Convert raw BrainVision files to BIDS format."""
    log = _logger or logger

    from mne_bids import BIDSPath, write_raw_bids

    log.info("Scanning for BrainVision files in: %s", source_root)
    vhdrs = find_brainvision_vhdrs(source_root)
    if not vhdrs:
        log.error("No .vhdr files found under sub-*/eeg/. Nothing to convert.")
        return 0

    if subjects:
        subj_set = set(subjects)
        vhdrs = [p for p in vhdrs if parse_subject_id(p) in subj_set]
        if not vhdrs:
            log.error("No matching .vhdr files for subjects: %s", sorted(subj_set))
            return 0

    ensure_dataset_description(bids_root, name=f"{task} EEG")
    ensure_task_events_json(bids_root, task=task)
    ensure_participants_tsv(bids_root, sorted({parse_subject_id(p) for p in vhdrs}))

    for i, vhdr in enumerate(vhdrs, 1):
        subject_label = parse_subject_id(vhdr)
        run_index = get_run_index(vhdr)

        raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose=False)
        set_channel_types(raw)

        if montage:
            set_montage(raw, montage)

        raw.info["line_freq"] = line_freq

        has_vol = _has_volume_triggers(raw)
        if do_trim_to_first_volume and not has_vol:
            log.warning(
                "trim_to_first_volume requested but no volume triggers detected in %s. "
                "EEG↔fMRI temporal anchoring will be limited.",
                vhdr.name,
            )
        if (not do_trim_to_first_volume) and has_vol:
            log.info(
                "Volume triggers detected in %s. For EEG↔fMRI alignment, consider enabling "
                "--trim-to-first-volume and --zero-base-onsets.",
                vhdr.name,
            )

        was_trimmed = False
        if do_trim_to_first_volume:
            was_trimmed = trim_to_first_volume(raw)

        if was_trimmed and not raw.preload:
            raw.load_data()

        filter_annotations(raw, event_prefixes, keep_all_annotations, zero_base_onsets)

        bids_path = BIDSPath(
            subject=subject_label,
            task=task,
            run=run_index,
            datatype="eeg",
            suffix="eeg",
            root=bids_root,
        )

        write_raw_bids(
            raw=raw,
            bids_path=bids_path,
            overwrite=overwrite,
            allow_preload=raw.preload,
            format="BrainVision",
            verbose=False,
        )

        log.info("[%d/%d] Wrote: sub-%s", i, len(vhdrs), subject_label)

    log.info("Done. Converted %d file(s) to BIDS in: %s", len(vhdrs), bids_root)
    return len(vhdrs)
