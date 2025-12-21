import os
import warnings
import mne
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from mne_bids import BIDSPath, get_entities_from_fname
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from mne_connectivity import spectral_connectivity_epochs, envelope_correlation
from mne.datasets import fetch_fsaverage
from specparam import SpectralModel
import bct

from . import utils
from . import io
from . import plotting


###################################################################
# PSD Computation
###################################################################

def compute_psd(epochs, freq_res, psd_freqmin, psd_freqmax):
    pad_s = 1 / freq_res
    epo_dur = epochs.get_data().shape[2] / epochs.info["sfreq"]
    n_pad = int(int((pad_s - epo_dur) * epochs.info["sfreq"]) / 2)
    
    epochs_padded = mne.EpochsArray(
        np.pad(
            epochs.get_data(),
            pad_width=((0, 0), (0, 0), (n_pad, n_pad)),
            mode="constant",
            constant_values=0,
        ),
        info=epochs.info,
        tmin=-pad_s / 2,
        verbose=False,
    )
    
    psd = epochs_padded.compute_psd(
        method="multitaper", n_jobs=-1, fmin=psd_freqmin, fmax=psd_freqmax
    ).average()
    
    return psd


###################################################################
# Peak Alpha Computation
###################################################################

def compute_peak_alpha(psd, freq_bands, somato_chans=None):
    freq_range = (psd.freqs >= freq_bands["alpha"][0]) & (
        psd.freqs <= freq_bands["alpha"][1]
    )
    
    results = {}
    
    avgpow = psd.get_data().mean(axis=0)
    peak, prop = scipy.signal.find_peaks(avgpow[freq_range])
    
    if len(peak) > 1:
        peak = peak[np.argmax(avgpow[freq_range][peak])]
        used_max_global = 0
    elif len(peak) == 0:
        peak = np.argmax(avgpow[freq_range])
        used_max_global = 1
    else:
        used_max_global = 0
    
    results["alpha_peak_global"] = psd.freqs[freq_range][peak]
    results["alpha_peak_global_usedmax"] = used_max_global
    results["alpha_cog_global"] = np.sum(
        np.multiply(avgpow[freq_range], psd.freqs[freq_range])
    ) / np.sum(avgpow[freq_range])
    
    if somato_chans:
        avgpow_somato = psd.copy().pick(somato_chans).get_data().mean(axis=0)
        peak, prop = scipy.signal.find_peaks(avgpow_somato[freq_range])
        if len(peak) > 1:
            peak = peak[np.argmax(avgpow_somato[freq_range][peak])]
            used_max_somato = 0
        elif len(peak) == 0:
            peak = np.argmax(avgpow_somato[freq_range])
            used_max_somato = 1
        else:
            used_max_somato = 0
        results["alpha_peak_somato"] = psd.freqs[freq_range][peak]
        results["alpha_peak_somato_usedmax"] = used_max_somato
        results["alpha_cog_somato"] = np.sum(
            np.multiply(avgpow_somato[freq_range], psd.freqs[freq_range])
        ) / np.sum(avgpow_somato[freq_range])
    
    peak_alpha_all_chans = pd.DataFrame(
        columns=["peak", "cog"], index=psd.ch_names
    )
    for ch in psd.ch_names:
        avgpow = psd.copy().pick(ch).get_data().mean(axis=0)
        peak, prop = scipy.signal.find_peaks(avgpow[freq_range])
        if len(peak) > 1:
            peak = peak[np.argmax(avgpow[freq_range][peak])]
            used_max = 0
        elif len(peak) == 0:
            peak = np.argmax(avgpow[freq_range])
            used_max = 1
        else:
            used_max = 0
        peak_alpha_all_chans.loc[ch, "peak"] = psd.freqs[freq_range][peak]
        peak_alpha_all_chans.loc[ch, "peak_usedmax"] = used_max
        peak_alpha_all_chans.loc[ch, "cog"] = np.sum(
            np.multiply(avgpow[freq_range], psd.freqs[freq_range])
        ) / np.sum(avgpow[freq_range])
    
    results["peak_alpha_all_chans"] = peak_alpha_all_chans
    return results


###################################################################
# Band Power Computation
###################################################################

def compute_band_power(psd, freq_bands, somato_chans=None):
    results = {}
    for band in freq_bands:
        curr_freq_range = (psd.freqs >= freq_bands[band][0]) & (
            psd.freqs <= freq_bands[band][1]
        )
        results[f"{band}_power_global"] = np.sum(
            psd.get_data()[:, curr_freq_range].mean(axis=1)
        )
        
        if somato_chans:
            results[f"{band}_power_somato"] = np.sum(
                psd.copy()
                .pick(somato_chans)
                .get_data()[:, curr_freq_range]
                .mean(axis=1)
            )
    return results


###################################################################
# FOOF Fitting
###################################################################

def fit_foof(psd, psd_freqmin, psd_freqmax):
    fm = SpectralModel(verbose=False)
    freq_range = [psd_freqmin, psd_freqmax]
    fm.fit(psd.freqs, psd.get_data().mean(axis=0), freq_range)
    return fm


###################################################################
# Source Space Connectivity
###################################################################

def compute_connectivity(epochs_band, forward, labels, freq_bands, band):
    cov = mne.compute_covariance(
        epochs_band,
        method="empirical",
        keep_sample_mean=False,
        verbose=False,
    )
    
    filters = make_lcmv(
        epochs_band.info,
        forward,
        cov,
        reg=0.05,
        pick_ori="max-power",
        rank=None,
    )
    
    stcs = apply_lcmv_epochs(epochs_band, filters)
    
    con_wpli = spectral_connectivity_epochs(
        data=stcs,
        method="wpli",
        mode="multitaper",
        sfreq=epochs_band.info["sfreq"],
        fmin=freq_bands[band][0],
        fmax=freq_bands[band][1],
        faverage=True,
        n_jobs=-1,
        verbose=False,
    )
    
    con_wpli_matrix = (
        con_wpli.get_data().squeeze().reshape(len(labels), len(labels))
    )
    con_wpli_matrix[np.triu_indices(len(labels), 0)] = np.nan
    
    vtcs = []
    for s in stcs:
        vtcs.append(s.copy().apply_hilbert(envelope=True).data)
    vtcs = np.stack(vtcs, axis=0)
    
    con_aec = envelope_correlation(
        data=vtcs, orthogonalize="pairwise", verbose=False
    )
    con_aec = con_aec.combine()
    
    con_aec_matrix = con_aec.get_data(output="dense")[:, :, 0]
    con_aec_matrix /= 0.577
    con_aec_matrix[np.triu_indices(len(labels), 0)] = np.nan
    
    return con_wpli_matrix, con_aec_matrix, stcs


###################################################################
# Graph Measures Computation
###################################################################

def compute_graph_measures(conn_matrix, adjacency_matrix):
    sorted_values = np.sort(np.abs(conn_matrix.flatten()))
    sorted_values = sorted_values[~np.isnan(sorted_values)]
    threshold = sorted_values[int(0.8 * len(sorted_values))]
    adjacency_matrix = np.abs(conn_matrix) >= threshold
    
    measures = {}
    measures["threshold"] = threshold
    measures["degree"] = bct.degree.degrees_und(adjacency_matrix)
    measures["cc"] = bct.clustering.clustering_coef_bu(adjacency_matrix)
    measures["gcc"] = np.mean(measures["cc"])
    
    distance = bct.distance.distance_bin(adjacency_matrix)
    cpl = bct.distance.charpath(distance, 0, 0)[0]
    measures["geff"] = bct.efficiency.efficiency_bin(adjacency_matrix)
    
    randN = bct.makerandCIJ_und(
        len(adjacency_matrix), int(np.floor(np.sum(adjacency_matrix) / 2))
    )
    gcc_rand = np.mean(bct.clustering.clustering_coef_bu(randN))
    cpl_rand = bct.distance.charpath(
        bct.distance.distance_bin(randN), 0, 0
    )[0]
    measures["smallworldness"] = (measures["gcc"] / gcc_rand) / (cpl / cpl_rand)
    
    return measures


###################################################################
# Source Space Features
###################################################################

def compute_source_space_features(
    epochs,
    sourcecoords_file,
    freq_bands,
    out_path,
    sub_num,
    session_out,
    task,
    session_file,
    features_frame,
    session,
    compute_connectivity=True,
    connectivity_methods=["aec", "wpli"],
    graph_metrics=["density", "clustering", "path_length", "efficiency"],
    connectivity_threshold=0.8,
):
    if not compute_connectivity:
        return
    
    pos = pd.read_csv(sourcecoords_file)
    
    pos_coord = {}
    pos_coord["rr"] = np.array([pos["R"], pos["A"], pos["S"]]).T / 1000
    pos_coord["nn"] = np.tile([0.0, 0.0, 1.0], (len(pos_coord["rr"]), 1))
    labels = pos["ROI Name"]
    
    src = mne.setup_volume_source_space("fsaverage", pos=pos_coord, verbose=False)
    
    bem = os.path.join(
        fetch_fsaverage(), "bem", "fsaverage-5120-5120-5120-bem-sol.fif"
    )
    
    forward = mne.make_forward_solution(
        epochs.info, src=src, trans="fsaverage", bem=bem, eeg=True
    )
    
    graph_measures = {}
    subject_eeg_dir = os.path.join(out_path, sub_num, session_out, "eeg")
    
    for band in freq_bands:
        epochs_band = epochs.copy().filter(
            freq_bands[band][0], freq_bands[band][1], n_jobs=-1
        )
        
        cov = mne.compute_covariance(
            epochs_band,
            method="empirical",
            keep_sample_mean=False,
            verbose=False,
        )
        
        filters = make_lcmv(
            epochs_band.info,
            forward,
            cov,
            reg=0.05,
            pick_ori="max-power",
            rank=None,
        )
        
        stcs = apply_lcmv_epochs(epochs_band, filters)
        
        con_wpli_matrix = None
        con_aec_matrix = None
        
        if "wpli" in connectivity_methods:
            con_wpli = spectral_connectivity_epochs(
                data=stcs,
                method="wpli",
                mode="multitaper",
                sfreq=epochs_band.info["sfreq"],
                fmin=freq_bands[band][0],
                fmax=freq_bands[band][1],
                faverage=True,
                n_jobs=-1,
                verbose=False,
            )
            con_wpli_matrix = (
                con_wpli.get_data().squeeze().reshape(len(labels), len(labels))
            )
            con_wpli_matrix[np.triu_indices(len(labels), 0)] = np.nan
            
            wpli_path = os.path.join(
                subject_eeg_dir,
                f"{sub_num}_task-{task}_{session_file}connectivity_wpli_{band}.npy"
            )
            np.save(wpli_path, con_wpli_matrix)
            
            wpli_plot_path = os.path.join(
                subject_eeg_dir,
                f"{sub_num}_task-{task}_{session_file}connectivitymatrix_wpli_{band}.jpg"
            )
            plotting.plot_connectivity_matrix(
                con_wpli_matrix, labels, wpli_plot_path,
                title=f"wPLI Connectivity Matrix - {band}"
            )
        
        vtcs = []
        for s in stcs:
            vtcs.append(s.copy().apply_hilbert(envelope=True).data)
        vtcs = np.stack(vtcs, axis=0)
        
        if "aec" in connectivity_methods:
            con_aec = envelope_correlation(
                data=vtcs, orthogonalize="pairwise", verbose=False
            )
            con_aec = con_aec.combine()
            con_aec_matrix = con_aec.get_data(output="dense")[:, :, 0]
            con_aec_matrix /= 0.577
            con_aec_matrix[np.triu_indices(len(labels), 0)] = np.nan
            
            aec_path = os.path.join(
                subject_eeg_dir,
                f"{sub_num}_task-{task}_{session_file}connectivity_aec_{band}.npy"
            )
            np.save(aec_path, con_aec_matrix)
            
            aec_plot_path = os.path.join(
                subject_eeg_dir,
                f"{sub_num}_task-{task}_{session_file}connectivitymatrix_aec_{band}.jpg"
            )
            plotting.plot_connectivity_matrix(
                con_aec_matrix, labels, aec_plot_path,
                title=f"AEC Connectivity Matrix - {band}"
            )
        
        if band == list(freq_bands.keys())[0]:
            labels_path = os.path.join(
                subject_eeg_dir,
                f"{sub_num}_task-{task}_{session_file}connectivity_labels.npy"
            )
            np.save(labels_path, labels)
        
        graph_measures[band] = {}
        for conn_measure in connectivity_methods:
            graph_measures[band][conn_measure] = {}
            if conn_measure == "wpli":
                if con_wpli_matrix is None:
                    continue
                conn_matrix = con_wpli_matrix.copy()
            else:
                if con_aec_matrix is None:
                    continue
                conn_matrix = con_aec_matrix.copy()
            
            sorted_values = np.sort(np.abs(conn_matrix.flatten()))
            sorted_values = sorted_values[~np.isnan(sorted_values)]
            threshold = sorted_values[int(connectivity_threshold * len(sorted_values))]
            adjacency_matrix = np.abs(conn_matrix) >= threshold
            
            measures = {}
            measures["threshold"] = threshold
            measures["degree"] = bct.degree.degrees_und(adjacency_matrix)
            measures["cc"] = bct.clustering.clustering_coef_bu(adjacency_matrix)
            measures["gcc"] = np.mean(measures["cc"])
            distance = bct.distance.distance_bin(adjacency_matrix)
            cpl = bct.distance.charpath(distance, 0, 0)[0]
            measures["geff"] = bct.efficiency.efficiency_bin(adjacency_matrix)
            
            randN = bct.makerandCIJ_und(
                len(adjacency_matrix), int(np.floor(np.sum(adjacency_matrix) / 2))
            )
            gcc_rand = np.mean(bct.clustering.clustering_coef_bu(randN))
            cpl_rand = bct.distance.charpath(
                bct.distance.distance_bin(randN), 0, 0
            )[0]
            measures["smallworldness"] = (measures["gcc"] / gcc_rand) / (cpl / cpl_rand)
            
            measures["density"] = np.mean(measures["degree"])
            measures["clustering"] = measures["gcc"]
            measures["path_length"] = cpl
            measures["efficiency"] = measures["geff"]
            
            graph_measures[band][conn_measure] = measures
            
            conn_matrix_plot = conn_matrix.copy()
            conn_matrix_plot[np.isnan(conn_matrix_plot)] = 0
            conn_matrix_plot = (
                conn_matrix_plot
                + conn_matrix_plot.T
                - np.diag(np.diag(conn_matrix_plot))
            )
            
            connectome_path = os.path.join(
                subject_eeg_dir,
                f"{sub_num}_task-{task}_{session_file}connectome_{conn_measure}_{band}.jpg"
            )
            plotting.plot_connectome(
                conn_matrix_plot,
                pos_coord["rr"] * 1000,
                threshold="99%",
                save_path=connectome_path,
                title=f"{band} {conn_measure} thresholded at 99%",
            )
            
            degree_path = os.path.join(
                subject_eeg_dir,
                f"{sub_num}_task-{task}_{session_file}degree_{band}_{conn_measure}.jpg"
            )
            plotting.plot_graph_measures(
                measures["degree"],
                pos_coord["rr"] * 1000,
                degree_path,
                title_prefix=f"Degree - {band} - {conn_measure}",
            )
            
            cc_path = os.path.join(
                subject_eeg_dir,
                f"{sub_num}_task-{task}_{session_file}cc_{band}_{conn_measure}.jpg"
            )
            plotting.plot_graph_measures(
                measures["cc"],
                pos_coord["rr"] * 1000,
                cc_path,
                title_prefix=f"Clustering coefficient - {band} - {conn_measure}",
            )
            
            if "density" in graph_metrics:
                features_frame.loc[(sub_num, session), f"{band}_{conn_measure}_density"] = (
                    measures["density"]
                )
            if "clustering" in graph_metrics:
                features_frame.loc[(sub_num, session), f"{band}_{conn_measure}_clustering"] = (
                    measures["clustering"]
                )
            if "path_length" in graph_metrics:
                features_frame.loc[(sub_num, session), f"{band}_{conn_measure}_path_length"] = (
                    measures["path_length"]
                )
            if "efficiency" in graph_metrics:
                features_frame.loc[(sub_num, session), f"{band}_{conn_measure}_efficiency"] = (
                    measures["efficiency"]
                )
            features_frame.loc[(sub_num, session), f"{band}_{conn_measure}_smallworldness"] = (
                measures["smallworldness"]
            )
    
    graph_measures_path = os.path.join(
        subject_eeg_dir,
        f"{sub_num}_task-{task}_graph_measures.npy"
    )
    np.save(graph_measures_path, graph_measures)


###################################################################
# Single Subject Feature Computation
###################################################################

def _get_default_roi_channels():
    return {
        "somato": [
            "C3", "C4", "CP3", "CP4", "C1", "C2", "C5", "C6",
            "CP1", "CP2", "CP5", "CP6", "Cz", "CPz",
        ],
        "prefrontal": ["Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "AFz"],
        "frontal": [
            "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
            "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
        ],
        "parietal": [
            "Pz", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8",
            "CP1", "CP2", "CP3", "CP4", "CPz",
        ],
        "temporal": [
            "T7", "T8", "TP7", "TP8", "TP9", "TP10",
            "FT7", "FT8", "FT9", "FT10",
        ],
        "occipital": ["O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8", "POz"],
        "midline": [
            "Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz",
        ],
    }


def _compute_trial_features(
    psd_data,
    psd,
    freq_bands,
    freqRange,
    trial_idx,
    somato_chans,
    roi_channels,
    psd_freqmin,
    psd_freqmax,
):
    trial_row = {}
    
    avgpow_trial = psd_data[trial_idx].mean(axis=0)
    peak, _ = scipy.signal.find_peaks(avgpow_trial[freqRange])
    
    if len(peak) > 1:
        peak = peak[np.argmax(avgpow_trial[freqRange][peak])]
        used_max_global = 0
    elif len(peak) == 0:
        peak = np.argmax(avgpow_trial[freqRange])
        used_max_global = 1
    else:
        used_max_global = 0
    
    trial_row["alpha_peak_global"] = psd.freqs[freqRange][peak]
    trial_row["alpha_peak_global_usedmax"] = used_max_global
    
    denom = np.sum(avgpow_trial[freqRange])
    if denom > 1e-20:
        trial_row["alpha_cog_global"] = (
            np.sum(np.multiply(avgpow_trial[freqRange], psd.freqs[freqRange])) / denom
        )
    else:
        trial_row["alpha_cog_global"] = np.nan
    
    if somato_chans:
        somato_data = psd.copy().pick(somato_chans).get_data()
        avgpow_somato_trial = somato_data[trial_idx].mean(axis=0)
        peak, _ = scipy.signal.find_peaks(avgpow_somato_trial[freqRange])
        
        if len(peak) > 1:
            peak = peak[np.argmax(avgpow_somato_trial[freqRange][peak])]
            used_max_somato = 0
        elif len(peak) == 0:
            peak = np.argmax(avgpow_somato_trial[freqRange])
            used_max_somato = 1
        else:
            used_max_somato = 0
        
        trial_row["alpha_peak_somato"] = psd.freqs[freqRange][peak]
        trial_row["alpha_peak_somato_usedmax"] = used_max_somato
        denom_s = np.sum(avgpow_somato_trial[freqRange])
        if denom_s > 1e-20:
            trial_row["alpha_cog_somato"] = (
                np.sum(np.multiply(avgpow_somato_trial[freqRange], psd.freqs[freqRange])) / denom_s
            )
        else:
            trial_row["alpha_cog_somato"] = np.nan
    
    for roi_name, roi_chans in roi_channels.items():
        available_roi_chans = [ch for ch in roi_chans if ch in psd.ch_names]
        if available_roi_chans:
            roi_data = psd.copy().pick(available_roi_chans).get_data()
            avgpow_roi_trial = roi_data[trial_idx].mean(axis=0)
            peak, _ = scipy.signal.find_peaks(avgpow_roi_trial[freqRange])
            
            if len(peak) > 1:
                peak = peak[np.argmax(avgpow_roi_trial[freqRange][peak])]
                used_max_roi = 0
            elif len(peak) == 0:
                peak = np.argmax(avgpow_roi_trial[freqRange])
                used_max_roi = 1
            else:
                used_max_roi = 0
            
            trial_row[f"alpha_peak_{roi_name}"] = psd.freqs[freqRange][peak]
            trial_row[f"alpha_peak_{roi_name}_usedmax"] = used_max_roi
            denom_r = np.sum(avgpow_roi_trial[freqRange])
            if denom_r > 1e-20:
                trial_row[f"alpha_cog_{roi_name}"] = (
                    np.sum(np.multiply(avgpow_roi_trial[freqRange], psd.freqs[freqRange])) / denom_r
                )
            else:
                trial_row[f"alpha_cog_{roi_name}"] = np.nan
    
    for band in freq_bands:
        curr_freqRange = (psd.freqs >= freq_bands[band][0]) & (
            psd.freqs <= freq_bands[band][1]
        )
        for roi_name, roi_chans in roi_channels.items():
            available_roi_chans = [ch for ch in roi_chans if ch in psd.ch_names]
            if available_roi_chans:
                roi_data = psd.copy().pick(available_roi_chans).get_data()
                band_power = np.sum(roi_data[trial_idx][:, curr_freqRange].mean(axis=0))
                trial_row[f"{band}_power_{roi_name}"] = band_power
    
    fm = SpectralModel(verbose=False)
    freq_range = [psd_freqmin, psd_freqmax]
    power_spectrum = psd_data[trial_idx].mean(axis=0)
    fm.fit(psd.freqs, power_spectrum, freq_range)
    
    trial_row["foof_offset"] = fm.aperiodic_params_[0]
    trial_row["foof_exponent"] = fm.aperiodic_params_[1]
    
    return trial_row, fm


def compute_features_single_subject(
    p,
    task,
    out_path,
    freq_bands,
    freq_res,
    somato_chans,
    psd_freqmax,
    psd_freqmin,
    sourcecoords_file,
    interpolate_bads,
    compute_sourcespace_features,
    task_is_rest=True,
    roi_channels=None,
    compute_connectivity=True,
    connectivity_methods=["aec", "wpli"],
    graph_metrics=["density", "clustering", "path_length", "efficiency"],
    connectivity_threshold=0.8,
):
    subject, session_entity = utils.get_subject_session(p)
    sub_num, session, session_out, session_file = utils.format_subject_session(
        subject, session_entity
    )
    
    if not session:
        session = "1"
        session_out = ""
        session_file = ""
    else:
        session_out = f"ses-{session}"
        session_file = f"ses-{session}_"
    
    subject_eeg_dir = os.path.join(out_path, sub_num, session_out, "eeg")
    os.makedirs(subject_eeg_dir, exist_ok=True)
    
    clean_epo = io.load_epochs(p)
    if interpolate_bads:
        clean_epo = clean_epo.interpolate_bads()
    
    clean_epo = clean_epo.pick_types(eeg=True)
    
    if task_is_rest:
        features_frame = pd.DataFrame({
            "participant_id": [sub_num],
            "session": [session],
            "task": [task],
            "file_name": [p],
        })
        features_frame["session"] = features_frame["session"].astype(str)
        features_frame.set_index(["participant_id", "session"], inplace=True)
        
        pad_s = 1 / freq_res
        epo_dur = clean_epo.get_data().shape[2] / clean_epo.info["sfreq"]
        if epo_dur < pad_s:
            n_pad = int(int((pad_s - epo_dur) * clean_epo.info["sfreq"]) / 2)
            clean_epo_padded = mne.EpochsArray(
                np.pad(
                    clean_epo.get_data(),
                    pad_width=((0, 0), (0, 0), (n_pad, n_pad)),
                    mode="constant",
                    constant_values=0,
                ),
                info=clean_epo.info,
                tmin=-pad_s / 2,
                verbose=False,
            )
        else:
            clean_epo_padded = clean_epo
        
        msg = "Computing features - PSD."
        logger.info(**gen_log_kwargs(message=msg, subject=subject, session=session))
        
        psd = clean_epo_padded.compute_psd(
            method="multitaper",
            n_jobs=-1,
            fmin=psd_freqmin,
            fmax=psd_freqmax,
            bandwidth=freq_res,
        ).average()
        
        psd_path = os.path.join(
            subject_eeg_dir, f"{sub_num}_task-{task}_{session_file}psd.h5"
        )
        psd.save(psd_path, overwrite=True)
        
        if compute_sourcespace_features and compute_connectivity:
            clean_epo_connectivity = clean_epo.copy().set_eeg_reference(projection=True)
            try:
                clean_epo_connectivity.apply_proj()
            except Exception:
                pass
        
        peak_alpha_results = compute_peak_alpha(psd, freq_bands, somato_chans)
        for key, value in peak_alpha_results.items():
            if key == "peak_alpha_all_chans":
                peak_alpha_path = os.path.join(
                    subject_eeg_dir,
                    f"{sub_num}_task-{task}_{session_file}peak_alpha_all_chans.tsv"
                )
                value.to_csv(peak_alpha_path, sep="\t", index=True)
            else:
                features_frame.loc[(sub_num, session), key] = value
        
        band_power_results = compute_band_power(psd, freq_bands, somato_chans)
        for key, value in band_power_results.items():
            features_frame.loc[(sub_num, session), key] = value
        
        fm = fit_foof(psd, psd_freqmin, psd_freqmax)
        features_frame.loc[(sub_num, session), "foof_offset"] = fm.aperiodic_params_[0]
        features_frame.loc[(sub_num, session), "foof_exponent"] = fm.aperiodic_params_[1]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            foof_path = os.path.join(
                subject_eeg_dir,
                f"{sub_num}_task-{task}_{session_file}foof.h5"
            )
            fm.save(foof_path)
            foof_report_path = os.path.join(
                subject_eeg_dir,
                f"{sub_num}_task-{task}_{session_file}foof_report.jpg"
            )
            fm.save_report(foof_report_path)
        
        plotting.plot_psd(psd, os.path.join(subject_eeg_dir, f"{sub_num}_task-{task}_{session_file}psd.jpg"))
        alpha_peak = features_frame.loc[(sub_num, session), "alpha_peak_global"]
        alpha_cog = features_frame.loc[(sub_num, session), "alpha_cog_global"]
        plotting.plot_psd_with_bands(
            psd,
            freq_bands,
            os.path.join(subject_eeg_dir, f"{sub_num}_task-{task}_{session_file}psd_bands.jpg"),
            alpha_peak=alpha_peak,
            alpha_cog=alpha_cog,
        )
        
        if roi_channels is None:
            roi_channels = _get_default_roi_channels()
        
        freqRange = (psd.freqs >= freq_bands["alpha"][0]) & (psd.freqs <= freq_bands["alpha"][1])
        
        for roi_name, roi_chans in roi_channels.items():
            available_roi_chans = [ch for ch in roi_chans if ch in psd.ch_names]
            if available_roi_chans:
                avgpow_roi = psd.copy().pick(available_roi_chans).get_data().mean(axis=0)
                peak, _ = scipy.signal.find_peaks(avgpow_roi[freqRange])
                if len(peak) > 1:
                    peak = peak[np.argmax(avgpow_roi[freqRange][peak])]
                    used_max_roi = 0
                elif len(peak) == 0:
                    peak = np.argmax(avgpow_roi[freqRange])
                    used_max_roi = 1
                else:
                    used_max_roi = 0
                
                features_frame.loc[(sub_num, session), f"alpha_peak_{roi_name}"] = psd.freqs[freqRange][peak]
                features_frame.loc[(sub_num, session), f"alpha_peak_{roi_name}_usedmax"] = used_max_roi
                denom_r = np.sum(avgpow_roi[freqRange])
                if denom_r > 1e-20:
                    features_frame.loc[(sub_num, session), f"alpha_cog_{roi_name}"] = (
                        np.sum(np.multiply(avgpow_roi[freqRange], psd.freqs[freqRange])) / denom_r
                    )
                else:
                    features_frame.loc[(sub_num, session), f"alpha_cog_{roi_name}"] = np.nan
        
        for band in freq_bands:
            curr_freqRange = (psd.freqs >= freq_bands[band][0]) & (psd.freqs <= freq_bands[band][1])
            for roi_name, roi_chans in roi_channels.items():
                available_roi_chans = [ch for ch in roi_chans if ch in psd.ch_names]
                if available_roi_chans:
                    avgpow_roi = psd.copy().pick(available_roi_chans).get_data().mean(axis=0)
                    band_power = np.sum(avgpow_roi[curr_freqRange])
                    features_frame.loc[(sub_num, session), f"{band}_power_{roi_name}"] = band_power
        
        if compute_sourcespace_features and compute_connectivity:
            clean_epo_for_source = io.load_epochs(p)
            clean_epo_for_source = clean_epo_for_source.set_eeg_reference(projection=True)
            try:
                clean_epo_for_source.apply_proj()
            except Exception:
                pass
            compute_source_space_features(
                clean_epo_for_source,
                sourcecoords_file,
                freq_bands,
                out_path,
                sub_num,
                session_out,
                task,
                session_file,
                features_frame,
                session,
                compute_connectivity=compute_connectivity,
                connectivity_methods=connectivity_methods,
                graph_metrics=graph_metrics,
                connectivity_threshold=connectivity_threshold,
            )
        
        features_path = os.path.join(
            subject_eeg_dir,
            f"{sub_num}_task-{task}_{session_file}features_frame.tsv"
        )
        io.save_features(features_frame, features_path, index=True)
        
        return features_frame
    
    else:
        features_frame = pd.DataFrame(
            columns=[
                "participant_id",
                "session",
                "trial_id",
                "condition",
                "task",
                "file_name",
            ]
        )
        
        original_event_id = clean_epo.event_id.copy()
        conditions_to_process = list(original_event_id.keys())
        
        msg = f"Computing features - PSD (task-based, {len(conditions_to_process)} conditions: {conditions_to_process})."
        logger.info(**gen_log_kwargs(message=msg, subject=subject, session=session))
        
        if roi_channels is None:
            roi_channels = _get_default_roi_channels()
        
        for condition in conditions_to_process:
            epochs_for_condition = clean_epo[condition].copy()
            sanitized_name = utils.sanitize_condition_name(condition)
            epochs_for_condition.event_id = {sanitized_name: original_event_id[condition]}
            
            logger.info(
                **gen_log_kwargs(
                    message=f"Processing condition: {condition} -> {sanitized_name}",
                    subject=subject,
                    session=session,
                )
            )
            
            condition_suffix = "_" + sanitized_name
            
            pad_s = 1 / freq_res
            epo_dur = epochs_for_condition.get_data().shape[2] / epochs_for_condition.info["sfreq"]
            
            if epo_dur < pad_s:
                n_pad = int(int((pad_s - epo_dur) * epochs_for_condition.info["sfreq"]) / 2)
                clean_epo_padded = mne.EpochsArray(
                    np.pad(
                        epochs_for_condition.get_data(),
                        pad_width=((0, 0), (0, 0), (n_pad, n_pad)),
                        mode="constant",
                        constant_values=0,
                    ),
                    info=epochs_for_condition.info,
                    tmin=-pad_s / 2,
                    event_id=epochs_for_condition.event_id,
                    verbose=False,
                )
            else:
                clean_epo_padded = epochs_for_condition
            
            psd = clean_epo_padded.compute_psd(
                method="multitaper",
                n_jobs=-1,
                fmin=psd_freqmin,
                fmax=psd_freqmax,
                bandwidth=freq_res,
            )
            
            psd.save(
                os.path.join(
                    subject_eeg_dir,
                    f"{sub_num}_task-{task}_{session_file}psd{condition_suffix}.h5",
                ),
                overwrite=True,
            )
            
            freqRange = (psd.freqs >= freq_bands["alpha"][0]) & (psd.freqs <= freq_bands["alpha"][1])
            psd_data = psd.get_data()
            n_trials = psd_data.shape[0]
            trial_features = []
            
            for trial_idx in range(n_trials):
                trial_row = {
                    "participant_id": sub_num,
                    "session": session,
                    "trial_id": trial_idx,
                    "condition": sanitized_name,
                    "task": task,
                    "file_name": p,
                }
                
                trial_row_data, fm = _compute_trial_features(
                    psd_data,
                    psd,
                    freq_bands,
                    freqRange,
                    trial_idx,
                    somato_chans,
                    roi_channels,
                    psd_freqmin,
                    psd_freqmax,
                )
                trial_row.update(trial_row_data)
                
                if trial_idx == 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fm.save(
                            os.path.join(
                                subject_eeg_dir,
                                f"{sub_num}_task-{task}_{session_file}foof_{sanitized_name}.h5",
                            )
                        )
                        fm.save_report(
                            os.path.join(
                                subject_eeg_dir,
                                f"{sub_num}_task-{task}_{session_file}foof_report_{sanitized_name}.jpg",
                            )
                        )
                
                trial_features.append(trial_row)
            
            trial_df = pd.DataFrame(trial_features)
            features_frame = pd.concat([features_frame, trial_df], ignore_index=True)
            
            if compute_sourcespace_features and compute_connectivity:
                msg = f"Computing features - source space and connectivity for condition: {sanitized_name}"
                logger.info(**gen_log_kwargs(message=msg, subject=subject, session=session))
                
                clean_epo_connectivity = epochs_for_condition.copy().set_eeg_reference(projection=True)
                try:
                    clean_epo_connectivity.apply_proj()
                except Exception:
                    pass
                
                pos = pd.read_csv(sourcecoords_file)
                pos_coord = {}
                pos_coord["rr"] = np.array([pos["R"], pos["A"], pos["S"]]).T / 1000
                pos_coord["nn"] = np.tile([0.0, 0.0, 1.0], (len(pos_coord["rr"]), 1))
                labels = pos["ROI Name"]
                
                src = mne.setup_volume_source_space("fsaverage", pos=pos_coord, verbose=False)
                bem = os.path.join(
                    fetch_fsaverage(), "bem", "fsaverage-5120-5120-5120-bem-sol.fif"
                )
                
                forward = mne.make_forward_solution(
                    clean_epo_connectivity.info,
                    src=src,
                    trans="fsaverage",
                    bem=bem,
                    eeg=True,
                )
                
                for band in freq_bands:
                    clean_epo_band = clean_epo_connectivity.copy().filter(
                        freq_bands[band][0], freq_bands[band][1], n_jobs=-1
                    )
                    
                    cov = mne.compute_covariance(
                        clean_epo_band,
                        method="empirical",
                        keep_sample_mean=False,
                        verbose=False,
                    )
                    
                    filters = make_lcmv(
                        clean_epo_band.info,
                        forward,
                        cov,
                        reg=0.05,
                        pick_ori="max-power",
                        rank='info',
                    )
                    
                    stcs = apply_lcmv_epochs(clean_epo_band, filters)
                    
                    all_aec_matrices = []
                    all_wpli_matrices = []
                    
                    for trial_idx in range(len(stcs)):
                        stc_trial = stcs[trial_idx]
                        
                        stc_env = stc_trial.copy().apply_hilbert(envelope=True)
                        vtcs_trial_env = stc_env.data[np.newaxis, :, :]
                        
                        con_aec_trial = envelope_correlation(
                            data=vtcs_trial_env, orthogonalize="pairwise", verbose=False
                        ).combine()
                        
                        aec_matrix = con_aec_trial.get_data(output="dense")[:, :, 0]
                        aec_matrix[np.triu_indices(len(labels), 0)] = np.nan
                        
                        if "aec" in connectivity_methods:
                            all_aec_matrices.append(aec_matrix)
                        
                        if "wpli" in connectivity_methods:
                            try:
                                stc_single = [stc_trial]
                                con_wpli = spectral_connectivity_epochs(
                                    data=stc_single,
                                    method="wpli",
                                    mode="multitaper",
                                    sfreq=clean_epo_band.info["sfreq"],
                                    fmin=freq_bands[band][0],
                                    fmax=freq_bands[band][1],
                                    faverage=True,
                                    n_jobs=1,
                                    verbose=False,
                                )
                                wpli_matrix = (
                                    con_wpli.get_data()
                                    .squeeze()
                                    .reshape(len(labels), len(labels))
                                )
                                wpli_matrix[np.triu_indices(len(labels), 0)] = np.nan
                                all_wpli_matrices.append(wpli_matrix)
                            except Exception as _e:
                                logger.warning(
                                    **gen_log_kwargs(
                                        message=f"wPLI computation failed for trial {trial_idx}: {_e}",
                                        subject=subject,
                                        session=session,
                                        emoji="⚠️",
                                    )
                                )
                        
                        for conn_measure in connectivity_methods:
                            if conn_measure == "wpli":
                                if not all_wpli_matrices:
                                    continue
                                conn_matrix = all_wpli_matrices[-1].copy()
                            else:
                                conn_matrix = all_aec_matrices[-1].copy()
                            
                            vals = np.abs(conn_matrix).flatten()
                            vals = vals[~np.isnan(vals)]
                            if vals.size == 0:
                                continue
                            
                            if isinstance(connectivity_threshold, (list, tuple, np.ndarray)):
                                thresholds_list = list(connectivity_threshold)
                            else:
                                thresholds_list = [connectivity_threshold]
                            
                            for thr in thresholds_list:
                                thr = float(thr)
                                thr = min(max(thr, 0.0), 1.0)
                                idx = int(thr * len(vals))
                                idx = min(max(idx, 0), len(vals) - 1)
                                cutoff = np.sort(vals)[idx]
                                
                                adjacency_matrix = np.abs(conn_matrix) >= cutoff
                                
                                degree = bct.degree.degrees_und(adjacency_matrix)
                                cc = bct.clustering.clustering_coef_bu(adjacency_matrix)
                                gcc = np.mean(cc)
                                distance = bct.distance.distance_bin(adjacency_matrix)
                                cpl = bct.distance.charpath(distance, 0, 0)[0]
                                geff = bct.efficiency.efficiency_bin(adjacency_matrix)
                                
                                trial_row_idx = len(features_frame) - len(trial_features) + trial_idx
                                dens_tag = f"dens{int(round(thr*100))}"
                                
                                if "density" in graph_metrics:
                                    features_frame.loc[
                                        trial_row_idx,
                                        f"{band}_{conn_measure}_{dens_tag}_density",
                                    ] = np.mean(degree)
                                if "clustering" in graph_metrics:
                                    features_frame.loc[
                                        trial_row_idx,
                                        f"{band}_{conn_measure}_{dens_tag}_clustering",
                                    ] = gcc
                                if "path_length" in graph_metrics:
                                    features_frame.loc[
                                        trial_row_idx,
                                        f"{band}_{conn_measure}_{dens_tag}_path_length",
                                    ] = cpl
                                if "efficiency" in graph_metrics:
                                    features_frame.loc[
                                        trial_row_idx,
                                        f"{band}_{conn_measure}_{dens_tag}_efficiency",
                                    ] = geff
                                
                                try:
                                    randN = bct.makerandCIJ_und(
                                        len(adjacency_matrix),
                                        int(np.floor(np.sum(adjacency_matrix) / 2)),
                                    )
                                    gcc_rand = np.mean(bct.clustering.clustering_coef_bu(randN))
                                    cpl_rand = bct.distance.charpath(
                                        bct.distance.distance_bin(randN), 0, 0
                                    )[0]
                                    smallworldness = (gcc / gcc_rand) / (cpl / cpl_rand)
                                    features_frame.loc[
                                        trial_row_idx,
                                        f"{band}_{conn_measure}_{dens_tag}_smallworldness",
                                    ] = smallworldness
                                except Exception:
                                    pass
                    
                    if "aec" in connectivity_methods and all_aec_matrices:
                        all_aec_matrices = np.stack(all_aec_matrices, axis=0)
                        np.save(
                            os.path.join(
                                subject_eeg_dir,
                                f"{sub_num}_task-{task}_{session_file}connectivity_aec_{band}{condition_suffix}_all_trials.npy",
                            ),
                            all_aec_matrices,
                        )
                    
                    if "wpli" in connectivity_methods:
                        if all_wpli_matrices:
                            all_wpli_matrices = np.stack(all_wpli_matrices, axis=0)
                            np.save(
                                os.path.join(
                                    subject_eeg_dir,
                                    f"{sub_num}_task-{task}_{session_file}connectivity_wpli_{band}{condition_suffix}_all_trials.npy",
                                ),
                                all_wpli_matrices,
                            )
                        else:
                            logger.warning(
                                **gen_log_kwargs(
                                    message=f"wPLI computation failed for all trials in band {band}.",
                                    subject=subject,
                                    session=session,
                                    emoji="⚠️",
                                )
                            )
                    
                    np.save(
                        os.path.join(
                            subject_eeg_dir,
                            f"{sub_num}_task-{task}_{session_file}connectivity_labels{condition_suffix}.npy",
                        ),
                        labels,
                    )
            
            fig = psd.plot(show=False)
            fig.savefig(
                os.path.join(
                    subject_eeg_dir,
                    f"{sub_num}_task-{task}_{session_file}psd{condition_suffix}.jpg",
                )
            )
            
            dat = psd.get_data().mean(axis=(0, 1))
            plt.figure()
            plt.plot(psd.freqs, dat, label=f"Average PSD - {sanitized_name}")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power")
            plt.title(f"Average PSD - {sanitized_name}")
            plt.legend()
            plt.savefig(
                os.path.join(
                    subject_eeg_dir,
                    f"{sub_num}_task-{task}_{session_file}psd_bands{condition_suffix}.jpg",
                )
            )
            plt.close("all")
        
        features_path = os.path.join(
            subject_eeg_dir,
            f"{sub_num}_task-{task}_{session_file}features_frame.tsv"
        )
        io.save_features(features_frame, features_path, index=False)
        
        return features_frame


###################################################################
# Configuration Validation
###################################################################

def validate_features_config(
    connectivity_methods,
    graph_metrics,
    connectivity_threshold,
    freq_bands,
    psd_freqmin,
    psd_freqmax,
    freq_res,
    compute_connectivity,
    compute_sourcespace_features,
):
    errors = []

    valid_connectivity_methods = ["aec", "wpli"]
    if connectivity_methods:
        invalid_methods = [
            m for m in connectivity_methods if m not in valid_connectivity_methods
        ]
        if invalid_methods:
            errors.append(
                f"Invalid connectivity methods: {invalid_methods}. Valid methods are: {valid_connectivity_methods}"
            )

    valid_graph_metrics = ["density", "clustering", "path_length", "efficiency"]
    if graph_metrics:
        invalid_metrics = [m for m in graph_metrics if m not in valid_graph_metrics]
        if invalid_metrics:
            errors.append(
                f"Invalid graph metrics: {invalid_metrics}. Valid metrics are: {valid_graph_metrics}"
            )

    if not (0.0 <= connectivity_threshold <= 1.0):
        errors.append(
            f"Connectivity threshold must be between 0.0 and 1.0, got: {connectivity_threshold}"
        )

    if psd_freqmin >= psd_freqmax:
        errors.append(
            f"PSD minimum frequency ({psd_freqmin}) must be less than maximum frequency ({psd_freqmax})"
        )

    if freq_res <= 0:
        errors.append(f"Frequency resolution must be positive, got: {freq_res}")

    if not freq_bands:
        errors.append("Frequency bands dictionary cannot be empty")
    else:
        for band_name, (low, high) in freq_bands.items():
            if low >= high:
                errors.append(
                    f"Invalid frequency band '{band_name}': low frequency ({low}) must be less than high frequency ({high})"
                )
            if low < 0:
                errors.append(
                    f"Invalid frequency band '{band_name}': frequencies must be non-negative, got low={low}"
                )

    if compute_connectivity and not compute_sourcespace_features:
        errors.append(
            "Connectivity analysis requires source space features to be enabled (compute_sourcespace_features=True)"
        )

    if compute_connectivity and not connectivity_methods:
        errors.append(
            "Connectivity analysis is enabled but no connectivity methods specified"
        )

    if connectivity_methods and not compute_connectivity:
        logger.warning(
            "Connectivity methods specified but connectivity analysis is disabled (compute_connectivity=False)"
        )

    if graph_metrics and not compute_connectivity:
        logger.warning(
            "Graph metrics specified but connectivity analysis is disabled (compute_connectivity=False)"
        )

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        raise ValueError(error_msg)

    logger.info("✅ Configuration validation passed")


###################################################################
# Main Feature Extraction Orchestrator
###################################################################

def compute_features(
    out_path,
    bids_path,
    task,
    freq_bands,
    sourcecoords_file,
    freq_res=1,
    somato_chans=None,
    psd_freqmax=100,
    psd_freqmin=1,
    n_jobs=1,
    subjects="all",
    interpolate_bads=True,
    compute_sourcespace_features=True,
    task_is_rest=True,
    roi_channels=None,
    compute_connectivity=True,
    connectivity_methods=["aec", "wpli"],
    graph_metrics=["density", "clustering", "path_length", "efficiency"],
    connectivity_threshold=0.8,
):
    logger.title("Custom step - Computing features")

    validate_features_config(
        connectivity_methods=connectivity_methods,
        graph_metrics=graph_metrics,
        connectivity_threshold=connectivity_threshold,
        freq_bands=freq_bands,
        psd_freqmin=psd_freqmin,
        psd_freqmax=psd_freqmax,
        freq_res=freq_res,
        compute_connectivity=compute_connectivity,
        compute_sourcespace_features=compute_sourcespace_features,
    )
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    clean_epo_files = list(
        set(
            str(f)
            for f in BIDSPath(
                root=bids_path,
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
    
    if subjects != "all":
        clean_epo_files = [
            f for f in clean_epo_files
            if get_entities_from_fname(f)["subject"] in subjects
        ]
    
    if n_jobs != 1 and task_is_rest:
        frames = Parallel(n_jobs=n_jobs)(
            delayed(compute_features_single_subject)(
                p,
                task,
                out_path,
                freq_bands,
                freq_res,
                somato_chans,
                psd_freqmax,
                psd_freqmin,
                sourcecoords_file,
                interpolate_bads,
                compute_sourcespace_features,
                task_is_rest,
                roi_channels,
                compute_connectivity,
                connectivity_methods,
                graph_metrics,
                connectivity_threshold,
            ) for p in clean_epo_files
        )
    else:
        frames = []
        for p in clean_epo_files:
            frames.append(compute_features_single_subject(
                p,
                task,
                out_path,
                freq_bands,
                freq_res,
                somato_chans,
                psd_freqmax,
                psd_freqmin,
                sourcecoords_file,
                interpolate_bads,
                compute_sourcespace_features,
                task_is_rest,
                roi_channels,
                compute_connectivity,
                connectivity_methods,
                graph_metrics,
                connectivity_threshold,
            ))
    
    if task_is_rest:
        if len(frames) == 1:
            all_frames = frames[0]
        else:
            all_frames = pd.concat(frames)
        all_frames_path = os.path.join(out_path, f"task-{task}_features_frame.tsv")
        io.save_features(all_frames, all_frames_path, index=True)
    else:
        for frame in frames:
            if frame is not None and not frame.empty:
                subject_id = frame.iloc[0]["participant_id"]
                frame.to_csv(
                    os.path.join(out_path, f"{subject_id}_task-{task}_features_frame.tsv"),
                    sep="\t",
                    index=False,
                )
        if len(frames) > 1:
            all_frames = pd.concat(frames, ignore_index=True)
            all_frames.to_csv(
                os.path.join(out_path, f"task-{task}_features_frame_pooled.tsv"),
                sep="\t",
                index=False,
            )

