import os
import textwrap
from pathlib import Path
from mne_bids_pipeline._config_template import create_template_config
from mne_bids_pipeline._logging import logger


def generate_config(path='example_config.py', mne_bids_type="minimal"):
    with open(path, 'w') as f:
        f.write("########################################################\n# This is a custom config file for the labmp EEG pipeline.\n###########################################################")
        f.write("\n\n# It includes the default config from mne-bids pipeline and some custom settings.\n\n")
        f.write("# Imports")
        f.write("\nimport os\nfrom mne_bids import BIDSPath, get_entities_from_fname\n")
        f.write("# Global settings\n")
        f.write("# project_root: Path to the project root directory\n")
        f.write("project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n")
        f.write("# bids_root: Path to the BIDS dataset\n")
        f.write("bids_root = os.path.join(project_root, 'data', 'bids_output')\n")
        f.write("# external_root: Path to the external data folder (e.g. Schaefer atlas)\n")
        f.write("external_root = '/path/to/external/data'\n")
        f.write("# log_type: how to log the messages from the pipeline. Use 'console' to print everything to the console or 'file' to log all to a file in the preprocessed folder (recommended)\n")
        f.write("log_type = 'file'\n")
        f.write("# use_pyprep: Set to True to use pyprep for bad channels detection. If false, no automated bad channels dection\n")
        f.write("use_pyprep = True\n")
        f.write("# use_icalabel: Set to True to use mne-icalabel for ICA component classification. If false, the mne-bids-pipeline default classification based on eog and ecg is used\n")
        f.write("use_icalabel = True\n")
        f.write("# use_custom_tfr: Set to True to use the custom TFR function to compute time-frequency representations. If false, no TFRs are computed\n")
        f.write("custom_tfr = True\n")
        f.write("# compute_rest_features: Set to True to compute rest features after preprocessing. If false, no features are computed\n")
        f.write("compute_rest_features = True\n")
        f.write("# tasks_to_process: List of tasks to process.\n")
        f.write("tasks_to_process = []\n")
        f.write("#config_validation: mne-bids-pipeline config validation. Leave to False because we use custom options.\n")
        f.write("config_validation = False\n")
        f.write("# subjects: List of subjects to process or 'all' to process all subjects.\n")
        f.write("subjects = 'all'\n")
        f.write("# sessions: List of sessions to process or leave empty to pcrocess all sessions.\n")
        f.write("sessions = []\n")
        f.write("#task: Task to process. This will be updated iteratively by the pipeline for each task. No need to change.\n")
        f.write("task = ''\n")
        f.write("# deriv_root: Path to the derivatives folder where the preprocessed data will be saved.\n")
        f.write("deriv_root = os.path.join(project_root, 'data', 'derivatives', 'preprocessed')\n")
        f.write("# select_subjects: If True, only the subjects with a file for the current task will be processed. If False, pipeline will crash if missing task\n")
        f.write("select_subjects = True\n")
        f.write("if select_subjects:\n")
        f.write("    task_subs = list(set(str(f) for f in BIDSPath(root=bids_root, task=task, session=None, datatype='eeg', suffix='eeg', extension='vhdr').match()))\n")
        f.write("    task_subs = [get_entities_from_fname(f).get('subject') for f in task_subs]\n")
        f.write("    if subjects != 'all':\n")
        f.write("        # If subjects is not 'all', filter the task_subs list\n")
        f.write("        subjects = [sub for sub in task_subs if sub in subjects]\n")
        f.write("    else:\n")
        f.write("        # If subjects is 'all', use all available subjects\n")
        f.write("        subjects = task_subs\n")
        f.write("\n")
        f.write("########################################################\n# Options for pyprep bad channels detection\n###########################################################")
        f.write("\n# pyprep_bids_path: Path to the BIDS dataset for pyprep, do not change unless you want a different path from the bids files\n")
        f.write("pyprep_bids_path = bids_root\n")
        f.write("# pyprep_pipeline_path: Path to the derivatives folder where the preprocessed data will be saved for pyprep, do not change\n")
        f.write("pyprep_pipeline_path = deriv_root\n")
        f.write("# pyprep_task: Task to process for pyprep, do not change\n")
        f.write("pyprep_task = task\n")
        f.write("# pyprep_ransac: Set to True to use RANSAC for bad channels detection, False to use only the other methods\n")
        f.write("pyprep_ransac = False\n")
        f.write("# pyprep_repeats: Number of repeats for the bad channel detection. This can improve detection by removing very bad channels and iterating again\n")
        f.write("pyprep_repeats = 3\n")
        f.write("# pyprep_average_reref: Set to True to average rereference the data before bad channels detection, False to use the original data\n")
        f.write("pyprep_average_reref = False\n")
        f.write("# pyprep_file_extension: File extension to use for the data files, default is .vhdr for BrainVision files\n")
        f.write("pyprep_file_extension = '.vhdr'\n")
        f.write("# pyprep_montage: Montage to use for the data, default is easycap-M1 for BrainVision files\n")
        f.write("pyprep_montage = 'easycap-M1'\n")
        f.write("# pyprep_l_pass: Low pass filter frequency for the data, default is 100.0 Hz\n")
        f.write("pyprep_l_pass = 100.0\n")
        f.write("# pyprep_notch: Notch filter frequency for the data, default is 60.0 Hz\n")
        f.write("pyprep_notch = 60.0\n")
        f.write("# pyprep_consider_previous_bads: Set to True to consider previous bad channels in the data (e.g. visually identified), False to ignore and clear them (e.g. when re-running the pipeline)\n")
        f.write("pyprep_consider_previous_bads = False\n")
        f.write("# pyprep_rename_anot_dict: Dictionary to rename the annotations to the format expected by MNE  (e.g. BAD_)\n")
        f.write("pyprep_rename_anot_dict = None\n")
        f.write("# pyprep_overwrite_chans_tsv: Set to True to overwrite the channels.tsv file with the bad channels detected by pyprep, False to keep the original file and create a second file. mne-bids-pipeline will only use original file so not recommended to set to False\n")
        f.write("pyprep_overwrite_chans_tsv = True\n")
        f.write("# pyprep_n_jobs: Number of jobs to use for pyprep, default is 1\n")
        f.write("pyprep_n_jobs = 1\n")
        f.write("# pyprep_subjects: List of subjects to process for pyprep, default is same as the rest of the pipeline\n")
        f.write("pyprep_subjects = subjects\n")
        f.write("# pyprep_delete_breaks: Set to True to delete breaks in the data (only for this operation, the data file is not modified), False to keep them\n")
        f.write("pyprep_delete_breaks = False\n")
        f.write("pyprep_breaks_min_length = 20  # Minimum length of breaks in seconds to consider them as breaks\n")
        f.write("pyprep_t_start_after_previous = 2  # Time in seconds to start after the last event\n")
        f.write("pyprep_t_stop_before_next = 2  # Time in seconds to stop before the next event\n")
        f.write("# pyprep_custom_bad_dict: Dictionary to specify custom bad channels for each subject. The format should be {taskname :{subject:[bad_chan_list]}} for example: {'eegtask': {'001': ['TP8']}} If not specified, the bad channels will only be detected automatically.\n")
        f.write("pyprep_custom_bad_dict = None\n")
        f.write('\n\n\n')
        f.write("########################################################\n# Options for Icalabel \n###########################################################")
        f.write("\n# icalabel_bids_path: Path to the BIDS dataset for icalabel, do not change unless you want a different path from the bids files\n")
        f.write("icalabel_bids_path = bids_root\n")
        f.write("# icalabel_pipeline_path: Path to the derivatives folder where the preprocessed data will be saved for icalabel, do not change\n")
        f.write("icalabel_pipeline_path = deriv_root\n")
        f.write("# icalabel_task: Task to process for icalabel, do not change\n")
        f.write("icalabel_task = task\n")
        f.write("# icalabel_prob_threshold: Probability threshold to use for icalabel, default is 0.8\n")
        f.write("icalabel_prob_threshold = 0.8\n")
        f.write("# icalabel_labels_to_keep: List of labels to keep for icalabel, default is ['brain', 'other']\n")
        f.write("icalabel_labels_to_keep = ['brain', 'other']\n")
        f.write("# icalabel_n_jobs: Number of jobs to use for icalabel, default is 1\n")
        f.write("icalabel_n_jobs = 1\n")
        f.write("# icalabel_subjects: List of subjects to process for icalabel, default is same as the rest of the pipeline\n")
        f.write("icalabel_subjects = subjects\n")
        f.write("# icalabel_keep_mnebids_bads: Set to True to keep the bad ica already flagged in the components.tsv file (e.g. visual inspection)\n")
        f.write("icalabel_keep_mnebids_bads = False\n")
        f.write('\n\n\n')
        f.write("########################################################\n# Rest features extraction config\n###########################################################")
        f.write("\n# features_bids_path: Path to the BIDS dataset for features extraction, do not change unless you want a different path from the bids files\n")
        f.write("features_bids_path = deriv_root\n")
        f.write("# features_out_path: Path to the derivatives folder where the preprocessed data will be saved for features extraction\n")
        f.write("features_out_path = deriv_root.replace('preprocessed', 'features')\n")
        f.write("# features_task: Task to process for features extraction, do not change\n")
        f.write("features_task = task\n")
        f.write("# features_sourcecoords_file: Path to the source coordinates file for features extraction, default is Schaefer 2018 atlas\n")
        f.write("features_sourcecoords_file = os.path.join(external_root, 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv')\n")
        f.write("# features_freq_res: Frequency resolution for the PSD, default is 0.1 Hz\n")
        f.write("features_freq_res = 0.1\n")
        f.write("# features_freq_bands: Frequency bands to use for the PSD, default is theta, alpha, beta, gamma\n")
        f.write("features_freq_bands = {\n")
        f.write("    'theta': (4, 8 - features_freq_res),\n")
        f.write("    'alpha': (8, 13 - features_freq_res),\n")
        f.write("    'beta': (13, 30),\n")
        f.write("    'gamma': (30 + features_freq_res, 80),\n")
        f.write("}\n")
        f.write("# features_psd_freqmax: Maximum frequency for the PSD, default is 100 Hz\n")
        f.write("features_psd_freqmax = 100\n")
        f.write("# features_psd_freqmin: Minimum frequency for the PSD, default is 1 Hz\n")
        f.write("features_psd_freqmin = 1\n")
        f.write("# features_somato_chans: List of somatosensory channels to use for the features extraction, default is ['C3', 'C4', 'Cz']\n")
        f.write("features_somato_chans = ['C3', 'C4', 'Cz']\n")
        f.write("# features_subjects: List of specific subjects to compute features for, default is same as rest of pipeline)\n")
        f.write("features_subjects = subjects\n")
        f.write("# features_compute_sourcespace_features: Set to True to compute source space features, False to skip this step\n")
        f.write("features_compute_sourcespace_features = False\n")
        f.write("# features_n_jobs: Number of jobs to use for features extraction, default is 1\n")
        f.write("features_n_jobs = 1\n")
        f.write("# features_subjects: List of subjects to process for features extraction, default is same as the rest of the pipeline\n")
        f.write('\n\n\n')
        f.write("########################################################\n# custom tfr config\n###########################################################")
        f.write("\n# custom_tfr_pipeline_path: Path to the preprocessed epochs, do not change unless you want a different path from the preprocessed epochs files\n")
        f.write("custom_tfr_pipeline_path = deriv_root\n")
        f.write("# custom_tfr_task: Task to process for TFR, do not change\n")
        f.write("custom_tfr_task = task\n")
        f.write("# custom_tfr_n_jobs: Number of jobs to use for TFR computation, default is 5\n")
        f.write("custom_tfr_n_jobs = 5\n")
        f.write("# custom_tfr_freqs: Frequencies to compute for TFR, default is np.arange(1, 100, 1)\n")
        f.write("custom_tfr_freqs = np.arange(1, 100, 1)\n")
        f.write("# custom_tfr_crop: Time interval to crop the TFR, default is None (no cropping)\n")
        f.write("custom_tfr_crop = None\n")
        f.write("# custom_tfr_n_cycles: Number of cycles for TFR computation, default is freqs/3.0\n")
        f.write("custom_tfr_n_cycles = custom_tfr_freqs / 3.0\n")
        f.write("# custom_tfr_decim: Decimation factor for TFR computation, default is 1\n")
        f.write("custom_tfr_decim = 2\n")
        f.write("# custom_tfr_return_itc: Whether to return inter-trial coherence, default is False\n")
        f.write("custom_tfr_return_itc = False\n")
        f.write("# custom_tfr_interpolate_bads: Whether to interpolate bad channels before computing the TFR, default is True\n")
        f.write("custom_tfr_interpolate_bads = True\n")
        f.write("# custom_tfr_average: Whether to average TFR across epochs, default is False\n")
        f.write("custom_tfr_average = False\n")
        f.write("# custom_tfr_return_average: Whether to return the average TFR in addition to the single trials, default is True\n")
        f.write("custom_tfr_return_average = True\n")
        f.write('\n\n\n')
        
        if mne_bids_type == "minimal":
            f.write('\n\n\n')
            f.write("########################################################\n# Rest MNE-BIDS-PIPELINE OPTIONS (MININIMAL)\n###########################################################")
            
            config_text = """\
            # Number of jobs
            n_jobs = 10
            # The task to process.
            # Whether the task should be treated as resting-state data.
            task_is_rest = True
            # The channel types to consider.
            ch_types = ["eeg"]
            # Specify EOG channels to use, or create virtual EOG channels.
            eog_channels = ["Fp1", "Fp2"]
            # The EEG reference to use. If `average`, will use the average reference,
            eeg_reference = "average"
            # eeg_template_montage
            eeg_template_montage = "easycap-M1"
            # You can specify the seed of the random number generator (RNG).
            random_state = 42
            # The low-frequency cut-off in the highpass filtering step.
            l_freq = 0.5
            # The high-frequency cut-off in the lowpass filtering step.
            h_freq = 100.0
            # Specifies frequency to remove using Zapline filtering. If None, zapline will not
            # be used.
            zapline_fline = 60.0
            # Resampling
            raw_resample_sfreq = 500
            # ## Epoching
            # Duration of epochs in seconds.
            rest_epochs_duration = 5.0  # data are segmented into 5-second epochs
            # Overlap between epochs in seconds
            rest_epochs_overlap = 2.5  # with a 50% overlap
            epochs_tmin = 0.0
            epochs_tmax = 5.0
            # if `None`, no baseline correction is applied.
            baseline = None
            # Whether to use a spatial filter to detect and remove artifacts. The BIDS
            # Pipeline offers the use of signal-space projection (SSP) and independent
            # component analysis (ICA).
            spatial_filter = "ica"
            # Peak-to-peak amplitude limits to exclude epochs from ICA fitting. This allows you to
            # remove strong transient artifacts from the epochs used for fitting ICA, which could
            # negatively affect ICA performance.
            ica_reject = {"eeg": 300e-6}
            # The ICA algorithm to use. `"picard-extended_infomax"` operates `picard` such that the
            ica_algorithm = "extended_infomax"  # extended infomax for mne icalabel
            # ICA high pass filter for mne icalabel
            ica_l_freq = 1.0

            # Run source estimation or not
            run_source_estimation = False
            # How to handle bad epochs after ICA.
            reject = "autoreject_local"
            autoreject_n_interpolate = [4, 8, 16]"""
            
            clean_text = textwrap.dedent(config_text)
            f.write(clean_text)
            f.close()
        
        elif mne_bids_type == "full":
            f.write("########################################################\n# Rest MNE-BIDS-PIPELINE OPTIONS (Full)\n###########################################################")
            create_template_config(target_path=Path(path.replace('.py', '_temp.py')), overwrite=True)
            
            with open(Path(path.replace('.py', '_temp.py')), 'r') as temp_config:
                for line in temp_config:
                    f.write(line)
            f.close()
            
            os.remove(Path(path.replace('.py', '_temp.py')))
    
    logger.info(f"✅ Config file generated at {path}.")


def main():
    import sys
    if len(sys.argv) < 2:
        print("ERROR! Usage: python create_config.py <config_file> [minimal|full]")
        print("  minimal: Generate minimal config file (default)")
        print("  full: Generate full config file with all mne-bids-pipeline options")
        sys.exit(1)
    
    config_file = sys.argv[1]
    mne_bids_type = sys.argv[2] if len(sys.argv) > 2 else "minimal"
    
    if mne_bids_type not in ["minimal", "full"]:
        print(f"ERROR! Invalid type: {mne_bids_type}. Must be 'minimal' or 'full'")
        sys.exit(1)
    
    generate_config(config_file, mne_bids_type)


if __name__ == "__main__":
    main()

