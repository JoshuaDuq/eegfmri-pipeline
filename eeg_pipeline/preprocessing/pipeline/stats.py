import os
import mne
import numpy as np
import pandas as pd
import scipy.stats
from mne_bids import BIDSPath, get_entities_from_fname
from mne_bids_pipeline._logging import gen_log_kwargs, logger

from . import utils
from . import io


###################################################################
# Preprocessing Statistics
###################################################################

def collect_preprocessing_stats(bids_path, pipeline_path, task):
    mne.set_log_level("ERROR")
    
    preprocessing_stats = pd.DataFrame(data=None, columns=["participant_id", "session"])
    preprocessing_stats.set_index(["participant_id", "session"], inplace=True)
    
    clean_epo_files = list(
        set(
            str(f)
            for f in BIDSPath(
                root=pipeline_path,
                task=task,
                session=None,
                suffix="epo",
                processing="clean",
                extension=".fif",
                check=False,
            ).match()
        )
    )
    
    for p in clean_epo_files:
        sess_num = get_entities_from_fname(p)["session"]
        if not sess_num:
            sess_num = "1"
        
        sub_num = 'sub-' + get_entities_from_fname(p)["subject"]
        
        chan_filename = utils.get_derived_path(p, "_proc-clean_epo.fif", "_bads.tsv")
        
        if not os.path.exists(chan_filename):
            import glob
            run_pattern = chan_filename.replace("_bads.tsv", "_run-*_bads.tsv")
            run_bads_files = glob.glob(run_pattern)
            
            if run_bads_files:
                all_bad_channels = set()
                for run_file in run_bads_files:
                    run_bads = io.read_channels_tsv(run_file)
                    all_bad_channels.update(run_bads['name'].tolist())
                chan_file = pd.DataFrame({'name': list(all_bad_channels)})
            else:
                logger.warning(f"No bads file found for {sub_num}, assuming no bad channels")
                chan_file = pd.DataFrame({'name': []})
        else:
            chan_file = io.read_channels_tsv(chan_filename)
        
        msg = f"Collecting preprocessing stats."
        logger.info(**gen_log_kwargs(message=msg, subject=sub_num.replace("sub", ""), session=sess_num))
        
        preprocessing_stats.loc[(sub_num, sess_num), "n_bad_channels"] = len(chan_file)
        
        components_path = utils.get_derived_path(p, "_proc-clean_epo.fif", "_proc-ica_components.tsv")
        ica_frame = io.read_components_tsv(components_path)
        
        if ica_frame is not None:
            preprocessing_stats.loc[(sub_num, sess_num), "n_bad_ica"] = len(
                ica_frame[ica_frame["status"] == "bad"]
            )
        
        epochs = io.load_epochs(p)
        events_type = list(epochs.event_id.keys())
        preprocessing_stats.loc[(sub_num, sess_num), "total_clean_epochs"] = len(epochs)
        preprocessing_stats.loc[(sub_num, sess_num), "n_removed_epochs"] = len(epochs.drop_log) - len(epochs)
        
        n_boundary = len(
            [i for i, x in enumerate(epochs.drop_log) if "BAD boundary" in x]
        )
        preprocessing_stats.loc[(sub_num, sess_num), "boundary_n_removed_epochs"] = n_boundary
        
        for event in events_type:
            epochs_cond = epochs[event]
            preprocessing_stats.loc[(sub_num, sess_num), event + "_total_clean_epochs"] = len(epochs_cond)
    
    preprocessing_stats.sort_values(by=["participant_id", "session"], inplace=True)
    preprocessing_stats.reset_index(inplace=True)
    
    stats_path = os.path.join(pipeline_path, f"task_{task}_preprocessing_stats.tsv")
    preprocessing_stats.to_csv(stats_path, sep="\t", index=False)
    
    desc_path = os.path.join(pipeline_path, f"task_{task}_preprocessing_stats_desc.tsv")
    preprocessing_stats.describe().to_csv(desc_path, sep="\t", index=False)


###################################################################
# TFR Statistical Testing
###################################################################

def rm_ttest_tfr(
    tfr_dict1,
    tfr_dict2,
    freq_range=None,
    time_range=None,
    channels=None,
    n_permutations=1000,
    clusters=None,
    tail=0,
    n_jobs=1,
    seed=None,
    decimate=None,
    threshold=None,
    adjacency_name='easycap-M1',
    return_stats_as_evoked=False,
    alpha=0.05
):
    if set(tfr_dict1.keys()) != set(tfr_dict2.keys()):
        raise ValueError("Both TFR dictionaries must have the same keys.")
    
    for key in tfr_dict1.keys():
        tfr_dict1[key].pick_types(eeg=True)
        if tfr_dict2 is not None:
            tfr_dict2[key].pick_types(eeg=True)
    
    if freq_range is not None:
        for tfr in tfr_dict1.values():
            tfr.crop(fmin=freq_range[0], fmax=freq_range[1])
        if tfr_dict2 is not None:
            for tfr in tfr_dict2.values():
                tfr.crop(fmin=freq_range[0], fmax=freq_range[1])
    
    if time_range is not None:
        for tfr in tfr_dict1.values():
            tfr.crop(tmin=time_range[0], tmax=time_range[1])
        if tfr_dict2 is not None:
            for tfr in tfr_dict2.values():
                tfr.crop(tmin=time_range[0], tmax=time_range[1])
    
    if channels is not None:
        for tfr in tfr_dict1.values():
            tfr.pick(channels)
        if tfr_dict2 is not None:
            for tfr in tfr_dict2.values():
                tfr.pick(channels)
    
    if decimate is not None:
        for tfr in tfr_dict1.values():
            tfr.decimate(decimate, verbose=False)
        if tfr_dict2 is not None:
            for tfr in tfr_dict2.values():
                tfr.decimate(decimate, verbose=False)
    
    data1 = np.array([tfr.data for tfr in tfr_dict1.values()])
    if tfr_dict2 is not None:
        data2 = np.array([tfr.data for tfr in tfr_dict2.values()])
        data_diff = data1 - data2
    else:
        data_diff = data1
    
    tfr = list(tfr_dict1.values())[0]
    
    assert np.argwhere(np.asarray(data_diff.shape) == len(tfr_dict1))[0][0] == 0
    assert np.argwhere(np.asarray(data_diff.shape) == len(tfr.info['ch_names']))[0][0] == 1
    assert np.argwhere(np.asarray(data_diff.shape) == len(tfr.freqs))[0][0] == 2
    assert np.argwhere(np.asarray(data_diff.shape) == len(tfr.times))[0][0] == 3
    
    if clusters is None:
        X = data_diff.reshape(data_diff.shape[0], -1)
        t_stat, p_val = mne.stats.permutation_t_test(
            X, n_permutations=n_permutations, tail=tail, n_jobs=n_jobs, seed=seed
        )
        t_stat = t_stat.reshape(data_diff.shape[1:])
        p_val = p_val.reshape(data_diff.shape[1:])
    else:
        if adjacency_name is not None:
            sensor_adjacency = mne.channels.read_ch_adjacency(adjacency_name, tfr.ch_names)
        else:
            sensor_adjacency = mne.channels.find_ch_adjacency(tfr.info, ch_type='eeg', verbose=False)
        
        adjacency = mne.stats.combine_adjacency(sensor_adjacency, len(tfr.freqs), len(tfr.times))
        
        if threshold is None:
            df = data_diff.shape[0] - 1
            if tail == 0:
                t_thresh = scipy.stats.distributions.t.ppf(1 - alpha / 2, df=df)
            elif tail == 1:
                t_thresh = scipy.stats.distributions.t.ppf(1 - alpha, df=df)
        elif threshold == 'TFCE':
            t_thresh = dict(start=0, step=0.2)
        
        t_stat, clusters, cluster_p_values, H0 = mne.stats.spatio_temporal_cluster_1samp_test(
            data_diff,
            n_permutations=n_permutations,
            tail=tail,
            n_jobs=n_jobs,
            seed=seed,
            adjacency=adjacency,
            threshold=t_thresh,
        )
        
        t_stat = t_stat.reshape(data_diff.shape[1:])
        p_val = np.ones_like(t_stat) * np.nan
        for cl, p in zip(clusters, cluster_p_values):
            p_val[cl] = p
    
    if return_stats_as_evoked:
        cp = tfr.copy()
        cp.data = t_stat[np.newaxis, :, :]
        t_stat = cp.copy()
        
        cp_p = tfr.copy()
        cp_p.data = p_val[np.newaxis, :, :]
        p_val = cp_p.copy()
    
    return t_stat, p_val, list(tfr_dict1.values())[0].info

