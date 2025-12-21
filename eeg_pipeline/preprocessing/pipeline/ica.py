import os
import mne
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from mne_icalabel import label_components
from mne_bids import get_entities_from_fname
from mne_bids_pipeline._logging import gen_log_kwargs, logger

from . import utils
from . import io


###################################################################
# ICA Component Labeling
###################################################################

def run_ica_label_single_file(
    p,
    prob_threshold=0.8,
    labels_to_keep=["brain", "other"],
    keep_mnebids_bads=False,
):
    bad_ica_frame = pd.DataFrame(index=None, columns=["file_name", "participant_id", "session", "n_bad_icas", "bad_icas"])

    if "ses-" in p:
        ses_num = get_entities_from_fname(p)["session"]
    else:
        ses_num = None
    sub_num = get_entities_from_fname(p)["subject"]
    
    bad_ica_frame.loc[p, "file_name"] = os.path.basename(p)
    bad_ica_frame.loc[p, "participant_id"] = sub_num
    bad_ica_frame.loc[p, "session"] = ses_num

    with mne.utils.use_log_level(False):
        msg = f"Finding bad icas using mne-icalabel."
        logger.info(**gen_log_kwargs(message=msg, subject=sub_num, session=ses_num))

        ica = io.load_ica(p)
        epochs_path = utils.get_derived_path(p, "_proc-icafit_ica.fif", "_proc-icafit_epo.fif")
        ica_epo = io.load_epochs(epochs_path)

        ica_epo.set_eeg_reference("average")

        icalabel = label_components(ica_epo, ica, method="iclabel")
        icalabel["labels"] = np.asanyarray(icalabel["labels"])

        bad_comps = np.where(
            (icalabel["y_pred_proba"] > prob_threshold)
            & (~np.isin(icalabel["labels"], labels_to_keep))
        )[0]

        msg = f"Found {len(bad_comps)} bad components."
        logger.info(
            **gen_log_kwargs(
                message=msg, subject=sub_num, session=ses_num, emoji="✅"
            )
        )
        bad_ica_frame.loc[p, "n_bad_icas"] = len(bad_comps)
        bad_ica_frame.loc[p, "bad_icas"] = ", ".join([str(c) for c in bad_comps])

        try:
            components_path = utils.get_derived_path(p, "_proc-icafit_ica.fif", "_proc-ica_components.tsv")
            ica_frame = io.read_components_tsv(components_path)
            
            if ica_frame is None:
                n_components = ica.n_components_
                ica_frame = io.create_empty_components_tsv(n_components)
            
            if not keep_mnebids_bads:
                ica_frame["status"] = "good"
                ica_frame["status_description"] = ""

            for comp in bad_comps:
                ica_frame.loc[ica_frame["component"] == comp, "status"] = "bad"
                ica_frame.loc[ica_frame["component"] == comp, "status_description"] = "Bad component detected by mne_icalabel"

            ica_frame["mne_icalabel_labels"] = icalabel["labels"]
            ica_frame["mne_icalabel_proba"] = icalabel["y_pred_proba"]

            io.write_components_tsv(ica_frame, components_path, index=False)
            bad_ica_frame.loc[p, "success"] = 1
            bad_ica_frame.loc[p, "error_log"] = ""
            
            ica.exclude = bad_comps.tolist()
            ica_path = utils.get_derived_path(p, "_proc-icafit_ica.fif", "_proc-ica_ica.fif")
            io.save_ica(ica, ica_path, overwrite=True)
    
        except Exception as e:
            logger.error(
                **gen_log_kwargs(
                    message=f"Error while finding bad components in {p}: {e}, skipping this file",
                    subject=sub_num,
                    session=ses_num,
                    emoji="❌",
                )
            )
            bad_ica_frame.loc[p, "success"] = 0
            bad_ica_frame.loc[p, "error_log"] = str(e)
    
    return bad_ica_frame


def run_ica_label(
    pipeline_path,
    task,
    prob_threshold=0.8,
    labels_to_keep=["brain", "other"],
    n_jobs=1,
    keep_mnebids_bads=False,
    subjects='all',
):
    from mne_bids import BIDSPath
    
    ica_files = list(
        set(
            str(f)
            for f in BIDSPath(
                root=pipeline_path,
                task=task,
                session=None,
                suffix="ica",
                processing="icafit",
                extension=".fif",
                check=False,
            ).match()
        )
    )
    
    if subjects != "all":
        ica_files = [
            f for f in ica_files
            if get_entities_from_fname(f).get("subject") in subjects
        ]
    
    ica_files.sort()

    logger.title(
        "Custom step - Find bad ICs using mne_icalabel in %d files." % len(ica_files)
    )

    if n_jobs != 1:
        bad_ica_frames = Parallel(n_jobs=n_jobs)(
            delayed(run_ica_label_single_file)(
                p,
                prob_threshold=prob_threshold,
                labels_to_keep=labels_to_keep,
                keep_mnebids_bads=keep_mnebids_bads,
            ) for p in ica_files
        )
    else:
        bad_ica_frames = []
        for p in ica_files:
            bframe = run_ica_label_single_file(
                p,
                prob_threshold=prob_threshold,
                labels_to_keep=labels_to_keep,
                keep_mnebids_bads=keep_mnebids_bads,
            )
            bad_ica_frames.append(bframe)

    if len(bad_ica_frames) == 0:
        logger.warning(f"No ICA files processed for task {task}")
        return
    
    if len(bad_ica_frames) > 1:
        bad_ica_frame = pd.concat(bad_ica_frames, ignore_index=False)
    else:
        bad_ica_frame = bad_ica_frames[0]
    
    bad_ica_frame.to_csv(
        os.path.join(pipeline_path, f"icalabel_task_{task}_log.csv"), sep="\t", index=False
    )

