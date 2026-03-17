import os
import mne
import pyprep
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from mne_bids import BIDSPath, get_entities_from_fname, read_raw_bids
from mne_bids_pipeline._logging import gen_log_kwargs, logger

from . import utils
from . import io


###################################################################
# Bad Channel Detection
###################################################################

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
    file_extension=".vhdr",
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

            previous_bads = chan_file[(chan_file["status"] == "bad") & (chan_file["type"].isin(['eeg', 'EEG']))]["name"].tolist()
            
            if previous_bads:
                if not consider_previous_bads:
                    msg = f"Found {len(previous_bads)} bad channels already marked. THOSE WILL BE IGNORED AND CLEARED BECAUSE consider_previous_bads=False."
                else:
                    msg = f"Found {len(previous_bads)} bad channels already marked. THOSE WILL BE CONSIDERED BECAUSE consider_previous_bads=True."
                logger.info(
                    **gen_log_kwargs(
                        message=msg,
                        subject=get_entities_from_fname(file)["subject"],
                        session=get_entities_from_fname(file)["session"],
                        emoji="⚠️",
                    )
                )

            eog_chans = chan_file.loc[
                chan_file["type"].isin(["EOG", "eog"]), "name"
            ].tolist()
            ecg_chans = chan_file.loc[
                chan_file["type"].isin(["ecg", "ECG"]), "name"
            ].tolist()
            emg_chans = chan_file.loc[
                chan_file["type"].isin(["EMG", "emg"]), "name"
            ].tolist()
            misc_chans = chan_file.loc[
                chan_file["type"].isin(["MISC", "misc"]), "name"
            ].tolist()

            is_gsr = (chan_file["name"].astype(str).str.upper() == "GSR") | (
                chan_file["type"].astype(str).str.upper() == "GSR"
            )
            if is_gsr.any():
                chan_file.loc[is_gsr, "type"] = "MISC"
                gsr_names = chan_file.loc[is_gsr, "name"].astype(str).tolist()
                misc_chans = list(sorted(set(misc_chans + gsr_names)))

            if bids_path:
                try:
                    ents = get_entities_from_fname(file)
                    bp = BIDSPath(
                        root=bids_path,
                        subject=ents.get("subject"),
                        session=ents.get("session"),
                        task=ents.get("task"),
                        datatype="eeg",
                        suffix="eeg",
                        extension=file_extension,
                        check=False,
                    )
                    raw = read_raw_bids(bp, verbose=False)
                    raw.load_data()
                except Exception:
                    raw = mne.io.read_raw(
                        file,
                        preload=True,
                        verbose=False,
                        eog=eog_chans,
                        misc=misc_chans + ecg_chans + emg_chans,
                    )
            else:
                raw = mne.io.read_raw(
                    file,
                    preload=True,
                    verbose=False,
                    eog=eog_chans,
                    misc=misc_chans + ecg_chans + emg_chans,
                )
            assert isinstance(raw, mne.io.BaseRaw)

            try:
                if raw.get_montage() is None and raw.info.get("dig") is None:
                    raw.set_montage(montage)
            except Exception as exc:
                logger.warning(
                    **gen_log_kwargs(
                        message=f"Failed to apply montage '{montage}' for {file}: {exc}",
                        subject=get_entities_from_fname(file).get("subject"),
                        session=get_entities_from_fname(file).get("session"),
                        emoji="⚠️",
                    )
                )

            if l_pass:
                raw.filter(None, l_pass, picks="eeg", verbose=False)
            
            if notch:
                raw.notch_filter(notch, picks="eeg", verbose=False)

            if consider_previous_bads:
                raw.info["bads"] = list(set(raw.info["bads"] + previous_bads))

            if delete_breaks:
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
                msg = f"Found {len(annot_breaks)} breaks in the data; not cropping prior to PyPREP."
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

            all_bads = []

            for _ in range(repeats):
                nc = pyprep.NoisyChannels(raw=raw, random_state=42)
                nc.find_bad_by_deviation()
                nc.find_bad_by_correlation()
                if ransac:
                    nc.find_bad_by_ransac()
                bads = nc.get_bads()
                all_bads.extend(bads)
                all_bads = sorted(all_bads)
                raw.info["bads"] = all_bads

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
                            ch for ch in custom_bad_dict[task][sub]
                            if ch not in raw.info["bads"] 
                        ]
                    else:
                        removed_custom_bads = []
                else:
                    removed_custom_bads = []
            else:
                removed_custom_bads = []

            bad_chans = ", ".join(sorted(all_bads))
            bad_chans = bad_chans.replace(" ", "").split(",")

            if "description" in chan_file.columns:
                chan_file["description"] = chan_file["description"].astype(str)

            if not consider_previous_bads:
                chan_file.loc[chan_file["type"].isin(["EEG", "eeg"]), "status"] = "good"
                chan_file.loc[chan_file["type"].isin(["EEG", "eeg"]), "description"] = ""

            task = get_entities_from_fname(file)["task"]
            sub = get_entities_from_fname(file)["subject"]
            
            for ch in bad_chans:
                chan_file.loc[chan_file["name"] == ch, "status"] = "bad"
                chan_file.loc[chan_file["name"] == ch, "description"] = "Bad channel detected by pyprep"
                if custom_bad_dict is not None and ch in custom_bad_dict.get(task, {}).get(sub, []):
                    chan_file.loc[chan_file["name"] == ch, "description"] = "Bad channel from custom bad channel list"

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
    subjects='all',
    custom_bad_dict=None
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
                file_extension=file_extension,
            ) for file in eeg_files
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
                file_extension=file_extension,
            )
            bads_frame_list.append(bframe)

    if len(bads_frame_list) == 0:
        logger.warning(f"No EEG files processed for task {task}")
        return
    
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
        subjects = [os.path.basename(d).replace("sub-", "") for d in subject_dirs if os.path.isdir(d)]
        logger.info(f"📂 Discovered {len(subjects)} subjects: {subjects}")
    
    for subject in subjects:
        pattern = os.path.join(bids_path, f"sub-{subject}", "eeg", f"sub-{subject}_task-{task}_*_channels.tsv")
        channel_files = glob.glob(pattern)
        
        if not channel_files:
            logger.warning(f"No channel files found for subject {subject}")
            continue
            
        logger.info(f"📋 Processing {len(channel_files)} channel files for sub-{subject}")
        
        all_bad_channels = set()
        channel_data = {}
        
        for file_path in channel_files:
            df = io.read_channels_tsv(file_path)
            bad_channels = df[df['status'] == 'bad']['name'].tolist()
            all_bad_channels.update(bad_channels)
            channel_data[file_path] = df
            
            run_info = os.path.basename(file_path).split('_')
            run_id = next((part for part in run_info if part.startswith('run-')), 'unknown')
            logger.info(f"  📁 {run_id}: Found {len(bad_channels)} bad channels: {bad_channels}")
        
        unified_bad_channels = sorted(list(all_bad_channels))
        logger.info(f"🔗 Unified bad channels for sub-{subject}: {unified_bad_channels} (total: {len(unified_bad_channels)})")
        
        for file_path, df in channel_data.items():
            df['status'] = 'good'
            df.loc[df['name'].isin(unified_bad_channels), 'status'] = 'bad'
            io.write_channels_tsv(df, file_path, index=False)
            
            run_info = os.path.basename(file_path).split('_')
            run_id = next((part for part in run_info if part.startswith('run-')), 'unknown')
            logger.info(f"  ✅ Updated {run_id} with {len(unified_bad_channels)} bad channels")
    
    logger.info("✅ Bad channel synchronization completed")

