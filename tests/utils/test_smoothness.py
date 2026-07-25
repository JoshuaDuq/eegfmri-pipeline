import mne
import numpy as np


def compute_spatial_roughness(ica: mne.preprocessing.ICA) -> np.ndarray:
    try:
        adjacency, ch_names = mne.channels.find_ch_adjacency(ica.info, ch_type="eeg")
        adj_matrix = adjacency.toarray()
    except Exception:
        # fallback to identity if no adjacency
        return np.zeros(ica.n_components_)

    components = ica.get_components()  # (n_channels, n_components)
    n_components = components.shape[1]
    roughness = np.zeros(n_components)

    for c in range(n_components):
        weights = components[:, c]
        # Normalize weights to [0, 1] for scale-invariant roughness
        if np.ptp(weights) > 0:
            weights = (weights - np.min(weights)) / np.ptp(weights)

        diffs = []
        for i in range(len(weights)):
            neighbors = np.where(adj_matrix[i] == 1)[0]
            if len(neighbors) > 0:
                # mean absolute difference with neighbors
                diffs.append(np.mean(np.abs(weights[i] - weights[neighbors])))

        roughness[c] = np.mean(diffs) if diffs else 0.0

    return roughness


print("Smoothness test written")
