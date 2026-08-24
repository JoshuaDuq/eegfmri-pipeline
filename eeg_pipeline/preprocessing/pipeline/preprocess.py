import os
from collections import Counter
import warnings

import mne
import pyprep
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from mne_bids import get_bids_path_from_fname, get_entities_from_fname, read_raw_bids
from mne_bids_pipeline._logging import gen_log_kwargs, logger

from . import utils
from . import io

###################################################################
# Bad Channel Detection
###################################################################

#: ``description`` this step writes for a channel PyPREP flagged in this run.
PYPREP_BAD_DESCRIPTION = "Bad channel detected by pyprep"
#: ``description`` written for a channel carried in from ``custom_bad_dict``.
CUSTOM_BAD_DESCRIPTION = "Bad channel from custom bad channel list"
#: ``description`` written by :func:`synchronize_bad_channels_across_runs`.
SYNCHRONIZED_BAD_DESCRIPTION = "Bad channel from subject-union synchronization"

#: Descriptions this pipeline writes itself, and therefore re-derives on every run.
#:
#: Anything else in ``description`` came from outside the pipeline — a hand-marked
#: channel, an upstream conversion step — and is carried forward untouched when
#: ``consider_previous_bads`` is set. The distinction is what keeps re-running the step
#: idempotent: without it, a channel PyPREP flagged once is fed back in as an input,
#: excluded from PyPREP's own analysis (``NoisyChannels`` drops ``info['bads']`` before it
#: measures anything), and so can never be un-flagged. The bad-channel set would then
#: depend on how many times the step had been run rather than on the recording.
PIPELINE_BAD_DESCRIPTIONS = frozenset(
    {
        PYPREP_BAD_DESCRIPTION,
        CUSTOM_BAD_DESCRIPTION,
        SYNCHRONIZED_BAD_DESCRIPTION,
    }
)


def _pyprep_reject_by_annotation(*, delete_breaks):
    """Exclude deliberately marked breaks from PyPREP's channel statistics."""
    return "omit" if delete_breaks else None


def _is_eeg_row(chan_file):
    """Return a mask over ``chan_file`` rows typed as EEG."""
    return chan_file["type"].astype(str).str.lower() == "eeg"


def _pipeline_written_bad_mask(chan_file):
    """Return a mask over EEG rows this pipeline marked bad on an earlier run."""
    if "description" not in chan_file.columns:
        # No description column means nothing can be attributed, so every previous bad is
        # treated as curated. That is the conservative direction: it keeps a hand-marked
        # channel rather than silently re-deriving one the pipeline cannot account for.
        return pd.Series(False, index=chan_file.index)
    descriptions = chan_file["description"].astype(str).str.strip()
    return (
        _is_eeg_row(chan_file)
        & (chan_file["status"] == "bad")
        & descriptions.isin(PIPELINE_BAD_DESCRIPTIONS)
    )


def _split_previous_bads(chan_file):
    """Split previously marked EEG bads into curated and pipeline-derived names.

    Curated bads are inputs to this step; pipeline-derived bads are outputs of a previous
    invocation of it and must be re-measured rather than assumed.
    """
    is_bad_eeg = _is_eeg_row(chan_file) & (chan_file["status"] == "bad")
    pipeline_written = _pipeline_written_bad_mask(chan_file)
    curated = sorted(chan_file.loc[is_bad_eeg & ~pipeline_written, "name"].astype(str))
    derived = sorted(chan_file.loc[pipeline_written, "name"].astype(str))
    return curated, derived


def _majority_bad_channels(repeated_bads):
    """Return channels marked bad in a strict majority of independent PyPREP runs."""
    if not repeated_bads:
        return []

    threshold = (len(repeated_bads) // 2) + 1
    counts = Counter(channel for bads in repeated_bads for channel in set(bads))
    return sorted(channel for channel, count in counts.items() if count >= threshold)


def _find_bad_channels_by_ransac(noisy_channels):
    """Run PyPREP RANSAC without verified spurious matmul warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^(divide by zero|overflow|invalid value) encountered in matmul$",
            category=RuntimeWarning,
            module=(r"^(scipy\.linalg\._basic|mne\.channels\.interpolation|" r"pyprep\.ransac)$"),
        )
        noisy_channels.find_bad_by_ransac()

    try:
        correlations = np.asarray(
            noisy_channels._extra_info["bad_by_ransac"]["ransac_correlations"]
        )
    except (AttributeError, KeyError, TypeError) as error:
        raise RuntimeError("PyPREP did not return RANSAC correlations.") from error
    if correlations.ndim != 2 or correlations.size == 0:
        raise RuntimeError("PyPREP returned a malformed RANSAC correlation matrix.")
    if not np.all(np.isfinite(correlations)):
        raise FloatingPointError("PyPREP returned non-finite RANSAC correlations.")


def _mark_breaks_bad(
    raw,
    breaks_min_length,
    t_start_after_previous,
    t_stop_before_next,
):
    """Annotate break periods as bad spans without changing sample timing."""
    annot_breaks = mne.preprocessing.annotate_break(
        raw=raw,
        min_break_duration=breaks_min_length,
        t_start_after_previous=t_start_after_previous,
        t_stop_before_next=t_stop_before_next,
        ignore=(
            "bad",
            "edge",
            "New Segment",
        ),
    )
    removed_dur = float(np.sum(annot_breaks.duration)) if len(annot_breaks) else 0.0
    if len(annot_breaks):
        bad_breaks = mne.Annotations(
            onset=annot_breaks.onset,
            duration=annot_breaks.duration,
            description=["BAD_break"] * len(annot_breaks),
            orig_time=annot_breaks.orig_time,
        )
        raw.set_annotations(raw.annotations + bad_breaks)
    return annot_breaks, removed_dur


def run_bads_detection_single_file(
    file,
    bids_path=None,
    ransac=False,
    repeats=3,
    average_reref=False,
    montage="easycap-M1",
    delete_breaks=False,
    rename_anot_dict=None,
    overwrite_chans_tsv=True,
    breaks_min_length=20,
    t_start_after_previous=2,
    t_stop_before_next=2,
    consider_previous_bads=False,
    l_pass=100,
    notch=None,
    custom_bad_dict=None,
    random_state=42,
):
    bads_frame = pd.DataFrame(
        data=None,
        columns=[
            "file_name",
            "participant_id",
            "session",
            "n_bads",
            "bad_channels",
            "n_breaks_found",
            "recording_duration",
            "success",
            "error",
        ],
    )
    bads_frame.loc[file, "file_name"] = os.path.basename(file)

    try:
        with mne.utils.use_log_level(False):
            msg = "Finding bad channels using pyprep."
            logger.info(
                **gen_log_kwargs(
                    message=msg,
                    subject=get_entities_from_fname(file)["subject"],
                    session=get_entities_from_fname(file)["session"],
                )
            )

            channels_path = utils.get_channels_path_from_eeg_file(file)
            if not os.path.exists(channels_path):
                raise FileNotFoundError(
                    f"channels.tsv not found alongside EEG file: {channels_path}. "
                    f"Expected for subject {get_entities_from_fname(file).get('subject')} "
                    f"session {get_entities_from_fname(file).get('session')}."
                )
            chan_file = io.read_channels_tsv(channels_path)

            bads_frame.loc[file, "participant_id"] = get_entities_from_fname(file)["subject"]

            if get_entities_from_fname(file).get("session") is not None:
                bads_frame.loc[file, "session"] = get_entities_from_fname(file)["session"]

            curated_previous_bads, derived_previous_bads = _split_previous_bads(chan_file)
            previous_bads = sorted(set(curated_previous_bads + derived_previous_bads))

            if previous_bads:
                if not consider_previous_bads:
                    msg = (
                        f"Found {len(previous_bads)} bad channels already marked. THOSE WILL BE "
                        "IGNORED AND CLEARED BECAUSE consider_previous_bads=False."
                    )
                else:
                    msg = (
                        f"Found {len(previous_bads)} bad channels already marked. "
                        f"{len(curated_previous_bads)} were marked outside this step and will be "
                        f"CONSIDERED because consider_previous_bads=True: {curated_previous_bads}. "
                        f"{len(derived_previous_bads)} were written by this step on an earlier run "
                        f"and will be RE-MEASURED rather than assumed: {derived_previous_bads}."
                    )
                logger.info(
                    **gen_log_kwargs(
                        message=msg,
                        subject=get_entities_from_fname(file)["subject"],
                        session=get_entities_from_fname(file)["session"],
                        emoji="⚠️",
                    )
                )

            eog_chans = chan_file.loc[chan_file["type"].isin(["EOG", "eog"]), "name"].tolist()
            ecg_chans = chan_file.loc[chan_file["type"].isin(["ecg", "ECG"]), "name"].tolist()
            emg_chans = chan_file.loc[chan_file["type"].isin(["EMG", "emg"]), "name"].tolist()
            misc_chans = chan_file.loc[chan_file["type"].isin(["MISC", "misc"]), "name"].tolist()

            is_gsr = (chan_file["name"].astype(str).str.upper() == "GSR") | (
                chan_file["type"].astype(str).str.upper() == "GSR"
            )
            if is_gsr.any():
                chan_file.loc[is_gsr, "type"] = "MISC"
                gsr_names = chan_file.loc[is_gsr, "name"].astype(str).tolist()
                misc_chans = list(sorted(set(misc_chans + gsr_names)))

            if bids_path:
                bp = get_bids_path_from_fname(file, check=False)
                raw = read_raw_bids(bp, verbose=False)
                raw.load_data()
            else:
                raw = mne.io.read_raw(
                    file,
                    preload=True,
                    verbose=False,
                    eog=eog_chans,
                    misc=misc_chans + ecg_chans + emg_chans,
                )
            assert isinstance(raw, mne.io.BaseRaw)

            if raw.get_montage() is None and raw.info.get("dig") is None:
                raw.set_montage(montage)

            # A low-pass here is off by default, and should stay off unless something in
            # the recording demands it. PyPREP's high-frequency-noise criterion is the
            # ratio of a channel's >50 Hz amplitude to its <50 Hz amplitude, so the band it
            # measures is exactly the band a low-pass removes. On 1000 Hz data, cutting at
            # 100 Hz discards 100-500 Hz — where contact and electrode noise live — and
            # leaves the criterion reading a narrow 50-100 Hz sliver with the line notch
            # cut out of it. The detector still runs and still reports z-scores; it is just
            # far less sensitive than the numbers suggest.
            #
            # The notch is a different case and stays on: line noise is a genuine confound
            # for the deviation criterion, and MATLAB PREP removes it before detection too.
            if l_pass:
                raw.filter(None, l_pass, picks="eeg", verbose=False)

            if notch:
                raw.notch_filter(notch, picks="eeg", verbose=False)

            # ``read_raw_bids`` seeds ``info['bads']`` from every ``status == "bad"`` row,
            # including the ones this step wrote last time. ``NoisyChannels`` drops
            # ``info['bads']`` before it measures anything, so leaving those in place would
            # exempt them from detection permanently. Only curated marks are kept.
            raw.info["bads"] = sorted(
                (set(raw.info["bads"]) - set(derived_previous_bads))
                | (set(curated_previous_bads) if consider_previous_bads else set())
            )
            if not consider_previous_bads:
                raw.info["bads"] = sorted(set(raw.info["bads"]) - set(previous_bads))

            if delete_breaks:
                annot_breaks, removed_dur = _mark_breaks_bad(
                    raw=raw,
                    breaks_min_length=breaks_min_length,
                    t_start_after_previous=t_start_after_previous,
                    t_stop_before_next=t_stop_before_next,
                )
                msg = (
                    f"Found {len(annot_breaks)} breaks in the data; "
                    "marked as BAD_break annotations before PyPREP."
                )
                logger.info(
                    **gen_log_kwargs(
                        message=msg,
                        subject=get_entities_from_fname(file)["subject"],
                        session=get_entities_from_fname(file)["session"],
                        emoji="⚠️",
                    )
                )
                bads_frame.loc[file, "n_breaks_found"] = len(annot_breaks)
                bads_frame.loc[file, "removed_breaks_duration"] = removed_dur

            if rename_anot_dict:
                raw.annotations.rename(rename_anot_dict)

            if average_reref:
                raw.set_eeg_reference("average")

            repeat_count = int(repeats)
            if repeat_count < 1:
                raise ValueError(f"pyprep repeats must be >= 1, got {repeats!r}.")
            if repeat_count > 1 and not ransac:
                # Only RANSAC consumes the random state; the other detectors are
                # deterministic, so repeating them votes on identical results at N times
                # the cost.
                logger.info(
                    **gen_log_kwargs(
                        message=(
                            f"pyprep repeats={repeat_count} has no effect without RANSAC; "
                            "running the deterministic detectors once."
                        ),
                        subject=get_entities_from_fname(file)["subject"],
                        session=get_entities_from_fname(file)["session"],
                        emoji="⚠️",
                    )
                )
                repeat_count = 1

            initial_bads = sorted(set(raw.info["bads"]))
            repeated_bads = []

            for repeat_index in range(repeat_count):
                raw.info["bads"] = list(initial_bads)
                repeat_random_state = (
                    None if random_state is None else int(random_state) + repeat_index
                )
                nc = pyprep.NoisyChannels(
                    raw=raw,
                    random_state=repeat_random_state,
                    reject_by_annotation=_pyprep_reject_by_annotation(delete_breaks=delete_breaks),
                )
                # Flat and NaN channels first: they are not merely noisy, and leaving them
                # in place makes every correlation against them meaningless.
                nc.find_bad_by_nan_flat()
                nc.find_bad_by_deviation()
                nc.find_bad_by_hfnoise()
                nc.find_bad_by_correlation()
                if ransac:
                    _find_bad_channels_by_ransac(nc)
                repeated_bads.append(nc.get_bads())

            pyprep_bads = _majority_bad_channels(repeated_bads)
            all_bads = sorted(set(initial_bads + pyprep_bads))
            raw.info["bads"] = list(all_bads)

            if custom_bad_dict is not None:
                task = get_entities_from_fname(file)["task"]
                sub = get_entities_from_fname(file)["subject"]
                if task in custom_bad_dict:
                    if sub in custom_bad_dict[task]:
                        all_bads.extend(custom_bad_dict[task][sub])
                        all_bads = sorted(set(all_bads))
                        msg = f"Found {len(custom_bad_dict[task][sub])} custom bad channels: {custom_bad_dict[task][sub]}."
                        logger.info(
                            **gen_log_kwargs(
                                message=msg,
                                subject=get_entities_from_fname(file)["subject"],
                                session=get_entities_from_fname(file)["session"],
                                emoji="⚠️",
                            )
                        )
                        removed_custom_bads = [
                            ch for ch in custom_bad_dict[task][sub] if ch not in raw.info["bads"]
                        ]
                        raw.info["bads"] = list(all_bads)
                    else:
                        removed_custom_bads = []
                else:
                    removed_custom_bads = []
            else:
                removed_custom_bads = []

            bad_chans = sorted(all_bads)

            if "description" in chan_file.columns:
                chan_file["description"] = chan_file["description"].astype(str)

            # Rows this step wrote before are cleared whatever ``consider_previous_bads``
            # says, so that a channel PyPREP no longer flags actually loses its mark. The
            # flag is a measurement of this recording, and re-running the measurement has
            # to be able to move it in both directions.
            pipeline_written = _pipeline_written_bad_mask(chan_file)
            chan_file.loc[pipeline_written, "status"] = "good"
            if "description" in chan_file.columns:
                chan_file.loc[pipeline_written, "description"] = ""

            if not consider_previous_bads:
                chan_file.loc[_is_eeg_row(chan_file), "status"] = "good"
                if "description" in chan_file.columns:
                    chan_file.loc[_is_eeg_row(chan_file), "description"] = ""

            task = get_entities_from_fname(file)["task"]
            sub = get_entities_from_fname(file)["subject"]

            custom_bads_for_run = set()
            if custom_bad_dict is not None:
                custom_bads_for_run = set(custom_bad_dict.get(task, {}).get(sub, []))
            # Only meaningful when the curated marks survived the clearing above.
            curated_bad_set = set(curated_previous_bads) if consider_previous_bads else set()

            for ch in bad_chans:
                row = chan_file["name"] == ch
                chan_file.loc[row, "status"] = "bad"
                if "description" not in chan_file.columns:
                    continue
                if ch in custom_bads_for_run:
                    chan_file.loc[row, "description"] = CUSTOM_BAD_DESCRIPTION
                elif ch in curated_bad_set:
                    # Leave the curator's own wording in place. Overwriting it would
                    # re-attribute the mark to this step, and the next run would then
                    # re-derive — and potentially drop — a channel a human had set.
                    continue
                else:
                    chan_file.loc[row, "description"] = PYPREP_BAD_DESCRIPTION

            if overwrite_chans_tsv:
                io.write_channels_tsv(chan_file, channels_path, index=False)
            else:
                base, _ext = os.path.splitext(channels_path)
                bad_channels_path = base + "_bad_channels.tsv"
                io.write_channels_tsv(chan_file, bad_channels_path, index=False)

            msg = f"Found {len(raw.info['bads'])} bad channels using pyprep: {raw.info['bads']} and {len(removed_custom_bads)} custom bad channels that were not detected by pyprep: {removed_custom_bads} for a total of {len(all_bads)} bad channels."
            bads_frame.loc[file, "n_bads"] = len(all_bads)
            bads_frame.loc[file, "bad_channels"] = bad_chans

            logger.info(
                **gen_log_kwargs(
                    message=msg,
                    subject=get_entities_from_fname(file)["subject"],
                    session=get_entities_from_fname(file)["session"],
                    emoji="✅",
                )
            )
    except Exception as e:
        bads_frame.loc[file, "success"] = 0
        bads_frame.loc[file, "error"] = str(e)
        logger.error(
            **gen_log_kwargs(
                message=f"Error while finding bad channels in {file}: {e}",
                subject=get_entities_from_fname(file)["subject"],
                session=get_entities_from_fname(file)["session"],
                emoji="❌",
            )
        )
        raise

    bads_frame.loc[file, "ransac"] = ransac
    bads_frame.loc[file, "repeats"] = repeats
    bads_frame.loc[file, "average_reref"] = average_reref
    bads_frame.loc[file, "montage"] = montage
    bads_frame.loc[file, "delete_breaks"] = delete_breaks
    bads_frame.loc[file, "rename_anot_dict"] = str(rename_anot_dict)
    bads_frame.loc[file, "overwrite_chans_tsv"] = overwrite_chans_tsv
    bads_frame.loc[file, "breaks_min_length"] = breaks_min_length
    bads_frame.loc[file, "t_start_after_previous"] = t_start_after_previous
    bads_frame.loc[file, "t_stop_before_next"] = t_stop_before_next
    bads_frame.loc[file, "consider_previous_bads"] = consider_previous_bads
    bads_frame.loc[file, "l_pass"] = l_pass
    bads_frame.loc[file, "notch"] = notch
    if custom_bad_dict is not None:
        bads_frame.loc[file, "custom_bad_dict"] = str(custom_bad_dict)
    else:
        bads_frame.loc[file, "custom_bad_dict"] = "None"

    bads_frame.loc[file, "success"] = 1
    bads_frame.loc[file, "error_log"] = ""

    return bads_frame


def run_bads_detection(
    bids_path,
    pipeline_path,
    task,
    session=None,
    ransac=False,
    repeats=3,
    average_reref=False,
    file_extension=".vhdr",
    montage="easycap-M1",
    delete_breaks=False,
    rename_anot_dict=None,
    breaks_min_length=20,
    t_start_after_previous=2,
    t_stop_before_next=2,
    overwrite_chans_tsv=True,
    consider_previous_bads=False,
    n_jobs=1,
    l_pass=100,
    notch=None,
    subjects="all",
    custom_bad_dict=None,
    random_state=42,
):
    eeg_files = utils.find_bids_files(
        root=bids_path,
        task=task,
        session=session,
        datatype="eeg",
        suffix="eeg",
        extension=file_extension,
        subjects=subjects,
    )

    logger.title(f"Custom step - Find bad channels in {len(eeg_files)} files.")

    if len(eeg_files) == 0:
        raise ValueError(
            "No EEG files found for bad-channel detection "
            f"(bids_path={bids_path}, task={task}, session={session}, "
            f"subjects={subjects}, extension={file_extension})."
        )

    if n_jobs != 1:
        bads_frame_list = Parallel(n_jobs=n_jobs)(
            delayed(run_bads_detection_single_file)(
                file,
                bids_path=bids_path,
                ransac=ransac,
                repeats=repeats,
                average_reref=average_reref,
                montage=montage,
                delete_breaks=delete_breaks,
                rename_anot_dict=rename_anot_dict,
                overwrite_chans_tsv=overwrite_chans_tsv,
                breaks_min_length=breaks_min_length,
                t_start_after_previous=t_start_after_previous,
                t_stop_before_next=t_stop_before_next,
                consider_previous_bads=consider_previous_bads,
                l_pass=l_pass,
                notch=notch,
                custom_bad_dict=custom_bad_dict,
                random_state=random_state,
            )
            for file in eeg_files
        )
    else:
        bads_frame_list = []
        for file in eeg_files:
            bframe = run_bads_detection_single_file(
                file,
                bids_path=bids_path,
                ransac=ransac,
                repeats=repeats,
                average_reref=average_reref,
                montage=montage,
                delete_breaks=delete_breaks,
                rename_anot_dict=rename_anot_dict,
                overwrite_chans_tsv=overwrite_chans_tsv,
                breaks_min_length=breaks_min_length,
                t_start_after_previous=t_start_after_previous,
                t_stop_before_next=t_stop_before_next,
                consider_previous_bads=consider_previous_bads,
                l_pass=l_pass,
                notch=notch,
                custom_bad_dict=custom_bad_dict,
                random_state=random_state,
            )
            bads_frame_list.append(bframe)

    if len(bads_frame_list) > 1:
        bads_frame = pd.concat(bads_frame_list, ignore_index=False)
    else:
        bads_frame = bads_frame_list[0]

    if not os.path.exists(pipeline_path):
        os.makedirs(pipeline_path)
    bads_frame.to_csv(os.path.join(pipeline_path, f"pyprep_task_{task}_log.csv"), index=False)


###################################################################
# Bad Channel Synchronization
###################################################################


def synchronize_bad_channels_across_runs(bids_path, task, subjects="all"):
    import glob

    logger.info("🔄 Synchronizing bad channels across runs for each subject...")

    if subjects == "all":
        subject_dirs = glob.glob(os.path.join(bids_path, "sub-*"))
        subjects = [
            os.path.basename(d).replace("sub-", "") for d in subject_dirs if os.path.isdir(d)
        ]
        logger.info(f"📂 Discovered {len(subjects)} subjects: {subjects}")

    for subject in subjects:
        # Recursive, and matching the task entity wherever it falls in the name. A
        # pattern anchored at ``sub-X/eeg`` skips session-organized datasets; one
        # demanding ``_run-`` skips single-run datasets; and one that puts ``task-``
        # directly after the subject skips anything with a ``ses-`` entity, because that
        # entity sits between the two. Each failure mode matches nothing rather than
        # raising, so the sync silently does nothing at all.
        pattern = os.path.join(bids_path, f"sub-{subject}", "**", f"sub-{subject}_*channels.tsv")
        selector = f"_task-{task}_"
        channel_files = sorted(
            path
            for path in glob.glob(pattern, recursive=True)
            if not os.path.basename(path).startswith("._")
            and (
                selector in os.path.basename(path)
                or os.path.basename(path).endswith(f"_task-{task}_channels.tsv")
            )
        )

        if not channel_files:
            logger.warning(f"No channel files found for subject {subject}")
            continue

        files_by_session = {}
        for file_path in channel_files:
            session = get_entities_from_fname(file_path).get("session")
            files_by_session.setdefault(session, []).append(file_path)

        logger.info(
            f"📋 Processing {len(channel_files)} channel files for sub-{subject} "
            f"in {len(files_by_session)} session(s)"
        )
        for session, session_files in sorted(
            files_by_session.items(), key=lambda item: str(item[0] or "")
        ):
            _synchronize_bad_channels_within_session(
                session_files,
                subject=subject,
                session=session,
            )

    logger.info("✅ Bad channel synchronization completed")


def _synchronize_bad_channels_within_session(channel_files, *, subject, session):
    """Apply one bad-channel union to runs that share a recording session."""
    all_bad_channels = set()
    channel_data = {}
    for file_path in channel_files:
        frame = io.read_channels_tsv(file_path)
        is_eeg = _is_eeg_row(frame)
        bad_channels = frame.loc[is_eeg & (frame["status"] == "bad"), "name"].tolist()
        all_bad_channels.update(bad_channels)
        channel_data[file_path] = frame

        run_id = next(
            (part for part in os.path.basename(file_path).split("_") if part.startswith("run-")),
            "unknown",
        )
        logger.info(f"  📁 {run_id}: Found {len(bad_channels)} bad channels: {bad_channels}")

    unified_bad_channels = sorted(all_bad_channels)
    session_label = f"ses-{session}" if session is not None else "no-session"
    logger.info(
        f"🔗 Unified bad channels for sub-{subject} {session_label}: "
        f"{unified_bad_channels} (total: {len(unified_bad_channels)})"
    )

    for file_path, frame in channel_data.items():
        is_eeg = _is_eeg_row(frame)
        already_bad = is_eeg & (frame["status"] == "bad")
        frame.loc[is_eeg, "status"] = "good"
        frame.loc[is_eeg & frame["name"].isin(unified_bad_channels), "status"] = "bad"

        if "description" in frame.columns:
            frame["description"] = frame["description"].astype(str)
            propagated = is_eeg & (frame["status"] == "bad") & ~already_bad
            frame.loc[propagated, "description"] = SYNCHRONIZED_BAD_DESCRIPTION
            frame.loc[is_eeg & (frame["status"] != "bad"), "description"] = ""

        io.write_channels_tsv(frame, file_path, index=False)
        run_id = next(
            (part for part in os.path.basename(file_path).split("_") if part.startswith("run-")),
            "unknown",
        )
        logger.info(f"  ✅ Updated {run_id} with {len(unified_bad_channels)} bad channels")
