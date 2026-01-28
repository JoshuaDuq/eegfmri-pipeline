import os
import mne
import pandas as pd
import numpy as np


###################################################################
# MNE Object I/O
###################################################################

def load_epochs(file_path, **kwargs):
    return mne.read_epochs(file_path, **kwargs)


def load_ica(file_path, **kwargs):
    return mne.preprocessing.read_ica(file_path, **kwargs)


def save_ica(ica, file_path, overwrite=True):
    ica.save(file_path, overwrite=overwrite)


###################################################################
# Channels TSV I/O
###################################################################

def read_channels_tsv(file_path):
    return pd.read_csv(file_path, sep="\t")


def write_channels_tsv(channels_df, file_path, index=False):
    channels_df.to_csv(file_path, sep="\t", index=index)


###################################################################
# Components TSV I/O
###################################################################

def read_components_tsv(file_path):
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path, sep="\t")


def create_empty_components_tsv(n_components):
    components_df = pd.DataFrame(
        columns=[
            "component",
            "type",
            "status",
            "status_description",
            "mne_icalabel_labels",
            "mne_icalabel_proba",
        ]
    )
    components_df["component"] = np.arange(n_components)
    return components_df


def write_components_tsv(components_df, file_path, index=False):
    components_df.to_csv(file_path, sep="\t", index=index)
