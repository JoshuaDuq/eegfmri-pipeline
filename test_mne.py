import mne
import numpy as np
raw = mne.io.RawArray(np.random.randn(3, 1000), mne.create_info(['eeg1', 'eeg2', 'ECG'], 100, ['eeg', 'eeg', 'eeg']))
events = np.array([[100, 0, 1], [200, 0, 1], [300, 0, 1]])
epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.1, tmax=0.1, preload=True, picks=None)
ica = mne.preprocessing.ICA(n_components=2)
ica.fit(epochs.copy().pick(["eeg1", "eeg2"]))
print("Finding bads...")
try:
    ica.find_bads_ecg(epochs, method="ctps", ch_name="ECG", verbose="ERROR")
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
