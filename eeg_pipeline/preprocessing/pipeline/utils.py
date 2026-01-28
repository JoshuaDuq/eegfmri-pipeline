import re
from mne_bids import BIDSPath, get_entities_from_fname


###################################################################
# BIDS File Finding
###################################################################

def find_bids_files(
    root,
    task,
    session=None,
    datatype="eeg",
    suffix="eeg",
    extension=".vhdr",
    processing=None,
    check=True,
    exclude_sourcedata=True,
    exclude_derivatives=True,
    subjects="all",
):
    bids_path = BIDSPath(
        root=root,
        task=task,
        session=session,
        datatype=datatype,
        suffix=suffix,
        extension=extension,
        processing=processing,
        check=check,
    )
    
    files = list(set(str(f) for f in bids_path.match()))
    files.sort()
    
    if exclude_sourcedata or exclude_derivatives:
        filtered_files = []
        for f in files:
            if exclude_sourcedata and "sourcedata" in f:
                continue
            if exclude_derivatives and "derivatives" in f:
                continue
            filtered_files.append(f)
        files = filtered_files
    
    if subjects != "all":
        files = [
            f for f in files
            if get_entities_from_fname(f).get("subject") in subjects
        ]
    
    return files


###################################################################
# Subject and Session Extraction
###################################################################

def get_subject_session(bids_path):
    entities = get_entities_from_fname(bids_path)
    subject = entities.get("subject", "")
    session = entities.get("session", "")
    
    if not subject:
        raise ValueError(f"Could not extract subject from path: {bids_path}")
    
    return subject, session


###################################################################
# Path Manipulation
###################################################################

def get_derived_path(file_path, old_suffix, new_suffix):
    return file_path.replace(old_suffix, new_suffix)


def get_channels_path_from_eeg_file(eeg_file):
    return re.sub(r"_eeg\.[^.]+$", "_channels.tsv", eeg_file)


###################################################################
# Condition Name Sanitization
###################################################################

def sanitize_condition_name(condition_name):
    sanitized = condition_name.replace(" ", "_")
    sanitized = sanitized.replace("/", "_")
    sanitized = sanitized.replace("\\", "_")
    sanitized = sanitized.replace(":", "_")
    sanitized = sanitized.replace("*", "_")
    sanitized = sanitized.replace("?", "_")
    sanitized = sanitized.replace('"', "_")
    sanitized = sanitized.replace("<", "_")
    sanitized = sanitized.replace(">", "_")
    sanitized = sanitized.replace("|", "_")
    return sanitized

