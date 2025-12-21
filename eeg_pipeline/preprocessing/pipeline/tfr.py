import os
import mne
import numpy as np
from mne_bids import BIDSPath, get_entities_from_fname
from mne_bids_pipeline._logging import gen_log_kwargs, logger

from . import utils
from . import io


###################################################################
# TFR Computation
###################################################################

def compute_tfr_morlet(epochs, freqs, n_cycles, decim, n_jobs, return_itc, average):
    return mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        decim=decim,
        n_jobs=n_jobs,
        return_itc=return_itc,
        average=average,
    )


###################################################################
# Single Subject TFR Computation
###################################################################

def compute_tfr_single_subject(
    p,
    freqs,
    n_cycles,
    decim,
    n_jobs,
    return_itc,
    interpolate_bads,
    average,
    crop,
    return_average,
    conditions=None,
):
    epo = io.load_epochs(p)
    
    if interpolate_bads:
        epo = epo.interpolate_bads()
    
    if conditions is not None:
        epo = epo[conditions]
    
    subject, session = utils.get_subject_session(p)
    
    msg = "Computing TFR"
    logger.info(**gen_log_kwargs(message=msg, subject=subject, session=session))
    
    if not average:
        power = compute_tfr_morlet(
            epo, freqs, n_cycles, decim, n_jobs, return_itc, average=False
        )
        
        if return_itc:
            power, itc = power
            if crop:
                itc.crop(tmin=crop[0], tmax=crop[1], include_tmax=True)
            
            itc_path = utils.get_derived_path(p, "_proc-clean_epo.fif", "_itc_epo-tfr.h5")
            itc.save(itc_path, overwrite=True)
        
        if crop:
            power.crop(tmin=crop[0], tmax=crop[1], include_tmax=True)
        
        power_path = utils.get_derived_path(p, "_proc-clean_epo.fif", "_power_epo-tfr.h5")
        power.save(power_path, overwrite=True)
        
        if return_average:
            for cond in epo.event_id:
                cond_save = utils.sanitize_condition_name(cond)
                power_cond = power[cond].average()
                power_cond_path = utils.get_derived_path(
                    p, "_proc-clean_epo.fif", f"_power_{cond_save}_avg-tfr.h5"
                )
                power_cond.save(power_cond_path, overwrite=True)
                
                if return_itc:
                    itc_cond = itc[cond].average()
                    itc_cond_path = utils.get_derived_path(
                        p, "_proc-clean_epo.fif", f"_itc_{cond_save}_avg-tfr.h5"
                    )
                    itc_cond.save(itc_cond_path, overwrite=True)
    else:
        for cond in epo.event_id:
            cond_save = utils.sanitize_condition_name(cond)
            epochs_cond = epo[cond]
            
            power_cond = compute_tfr_morlet(
                epochs_cond, freqs, n_cycles, decim, n_jobs, return_itc, average=True
            )
            
            if return_itc:
                power_cond, itc_cond = power_cond
                if crop:
                    itc_cond.crop(tmin=crop[0], tmax=crop[1], include_tmax=True)
                
                itc_cond_path = utils.get_derived_path(
                    p, "_proc-clean_epo.fif", f"_itc+{cond_save}_avg-tfr.h5"
                )
                itc_cond.save(itc_cond_path, overwrite=True)
            
            if crop:
                power_cond.crop(tmin=crop[0], tmax=crop[1], include_tmax=True)
            
            power_cond_path = utils.get_derived_path(
                p, "_proc-clean_epo.fif", f"_power+{cond_save}_avg-tfr.h5"
            )
            power_cond.save(power_cond_path, overwrite=True)
    
    msg = "Done computing TFR"
    logger.info(**gen_log_kwargs(message=msg, subject=subject, session=session, emoji="✅"))


###################################################################
# Main TFR Orchestrator
###################################################################

def custom_tfr(
    pipeline_path,
    task,
    freqs=np.arange(1, 100, 1),
    n_cycles=None,
    subjects='all',
    decim=1,
    n_jobs=1,
    return_itc=True,
    interpolate_bads=True,
    average=True,
    return_average=True,
    crop=None,
):
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
    clean_epo_files.sort()
    
    if subjects != 'all':
        clean_epo_files = [
            f for f in clean_epo_files
            if get_entities_from_fname(f)["subject"] in subjects
        ]
    
    if n_cycles is None:
        n_cycles = freqs / 3.0
    
    logger.title(f"Custom step - Computing TFR in {len(clean_epo_files)} files.")
    
    for p in clean_epo_files:
        compute_tfr_single_subject(
            p,
            freqs=freqs,
            n_cycles=n_cycles,
            decim=decim,
            n_jobs=n_jobs,
            return_itc=return_itc,
            interpolate_bads=interpolate_bads,
            average=average,
            crop=crop,
            return_average=return_average,
            conditions=None,
        )

