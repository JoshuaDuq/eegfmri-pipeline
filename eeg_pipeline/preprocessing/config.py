"""
DEPRECATED: This legacy config is kept for compatibility with coll_lab_eeg_pipeline
and MNE-BIDS pipeline tooling. Most scripts now read configuration from
eeg_pipeline/eeg_config.yaml via config_loader.py. Avoid adding new settings here.
"""

import os

import numpy as np

# Bids root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bids_root = os.path.join(project_root, "eeg_pipeline", "bids_output")

log_type = "file"  # 'file' or 'console'

use_pyprep = True
use_icalabel = True
compute_features = True
custom_tfr = False

tasks_to_process = ["thermalactive"]
sessions_to_process = []

task = 'thermalactive'
deriv_root = os.path.join(bids_root, "derivatives")
config_validation = "warn"
# Subjects to analyze. If `'all'`, include all subjects.
# Otherwise, a list of subject IDs.
subjects = "all"

##################################
# Pyprep bad channels detection
##################################
# Path to the BIDS root directory.
pyprep_bids_path = bids_root
pyprep_subjects = subjects
pyprep_pipeline_path = deriv_root
pyprep_task = task  
pyprep_ransac = False
pyprep_repeats = 3
pyprep_average_reref = False
pyprep_file_extension = ".vhdr"
pyprep_montage = "easycap-M1"
pyprep_l_pass = 100
pyprep_notch = 60.0
pyprep_delete_breaks = False
pyprep_breaks_min_length = 20
pyprep_t_start_after_previous = 10
pyprep_t_stop_before_next = 2
pyprep_consider_previous_bads = False
pyprep_rename_anot_dict = None
pyprep_overwrite_chans_tsv = True
pyprep_n_jobs = -1


##################################
# ica label config
##################################

icalabel_pipeline_path = deriv_root
icalabel_task = task
icalabel_prob_threshold = 0.7
icalabel_labels_to_keep = ["brain", "other"]
icalabel_n_jobs = -1
icalabel_subjects = subjects


##################################
# MNE-BIDS preprocessing config
##################################
# General settings

# Number of jobs
n_jobs = -1
# The task to process.
# Whether the task should be treated as resting-state data.
task_is_rest = False

# The channel types to consider.
ch_types = ["eeg"]
# Specify EOG channels to use (prefer using BIDS channels.tsv typing). Leave empty to avoid mislabeling EEG as EOG.
eog_channels = ["Fp1", "Fp2"]
# The EEG reference to use. If `average`, will use the average reference,
eeg_reference = "average"
# eeg_template_montage
eeg_template_montage = "easycap-M1"
# You can specify the seed of the random number generator (RNG).
random_state = 42
# The low-frequency cut-off in the highpass filtering step.
l_freq = 1.0
# The high-frequency cut-off in the lowpass filtering step.
h_freq = 100
# Specifies frequency to remove using Zapline filtering. If None, zapline will not be used.
zapline_fline = 60.0
# Resampling
raw_resample_sfreq = 500

# ## Epoching
# Use the pain conditions for epoching. the -5 s will be in the Hz and the 10 s will be in the Hz+pain
# Also add additional conditions for the 10, 20, 40 Hz and CNHz time locked on the onset of the svr
conditions = ["Trig_therm/T  1"]
epochs_tmin = -5
epochs_tmax = 10.5

find_breaks = False

# if `None`, no baseline correction is applied.
baseline = None
# Whether to use a spatial filter to detect and remove artifacts. The BIDS
# Pipeline offers the use of signal-space projection (SSP) and independent
# component analysis (ICA).
spatial_filter = "ica"
# Peak-to-peak amplitude limits to exclude epochs from ICA fitting. This allows you to
# remove strong transient artifacts from the epochs used for fitting ICA, which could
# negatively affect ICA performance.
ica_reject = {"eeg": 500e-6}
# The ICA algorithm to use. `"picard-extended_infomax"` operates `picard` such that the
ica_algorithm = "extended_infomax"  # extended infomax for mne icalabel
# ICA high pass filter for mne icalabel
ica_l_freq = 1.0

# For this first pass, we stop after the ICA fitting and do not perform any further
# processing steps.
run_source_estimation = False

# Second pass settings
reject = "autoreject_local"


##################################
# Features extraction config
##################################
# features_bids_path: Path to the BIDS dataset for features extraction, do not change unless you want a different path from the bids files
features_bids_path = deriv_root
# features_out_path: Path to the derivatives folder where the preprocessed data will be saved for features extraction
features_out_path = deriv_root.replace("preprocessed", "features")
# features_task: Task to process for features extraction, do not change
features_task = task
# features_task_is_rest: Set to True if this is a resting-state task (features averaged across all epochs), False for task-based data (features computed per condition)
features_task_is_rest = False
# features_sourcecoords_file: Path to the source coordinates file for features extraction, default is Schaefer 2018 atlas
features_sourcecoords_file = os.path.join(
    project_root,
    "eeg_pipeline",
    "source_data",
    "Schaefer2018",
    "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv",
)
# features_freq_res: Frequency resolution for the PSD, default is 0.1 Hz
features_freq_res = 0.1
# features_freq_bands: Frequency bands to use for the PSD, default is theta, alpha, beta, gamma
features_freq_bands = {
    "theta": (4, 8 - features_freq_res),
    "alpha": (8, 13 - features_freq_res),
    "beta": (13, 30),
    "gamma": (30 + features_freq_res, 80),
}
# features_psd_freqmax: Maximum frequency for the PSD, default is 100 Hz
features_psd_freqmax = 100
# features_psd_freqmin: Minimum frequency for the PSD, default is 1 Hz
features_psd_freqmin = 1
# features_somato_chans: List of somatosensory channels to use for the features extraction, default is ['C3', 'C4', 'Cz']
features_somato_chans = ["C3", "C4", "Cz"]
# features_subjects: List of specific subjects to compute features for, default is same as rest of pipeline)
features_subjects = subjects
# features_compute_sourcespace_features: Set to True to compute source space features, False to skip this step
features_compute_sourcespace_features = (
    True  # Now enabled for full connectivity analysis
)
# features_n_jobs: Number of jobs to use for features extraction, default is 1
features_n_jobs = -1


##################################
# Custom tfr config
##################################
custom_tfr_pipeline_path = deriv_root
custom_tfr_task = task
custom_tfr_subjects = subjects
custom_tfr_n_jobs = -1  # Number of jobs for TFR computation
custom_tfr_freqs = np.arange(1, 101, 1)  # Frequencies to compute
custom_tfr_crop = False  # Time interval to crop the TFR
custom_tfr_n_cycles = (
    custom_tfr_freqs / 3.0
)  # Number of cycles for TFR computation (7 for now, should be adjusted based on the data)
custom_tfr_decim = 4  # Decimation factor for TFR computation
custom_tfr_return_itc = False  # Whether to return inter-trial coherence
custom_tfr_interpolate_bads = True  # Whether to interpolate bad channels
custom_tfr_average = False  # Whether to average TFR across epochs
custom_tfr_return_average = (
    False  # Do not save averaged TFR; only epoch-level TFRs will be created
)
