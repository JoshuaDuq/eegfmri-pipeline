import os
import mne
import pandas as pd
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

