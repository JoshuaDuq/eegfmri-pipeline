import numpy as np
import matplotlib.pyplot as plt


def plot_autocorr(data, sfreq, component):
    # data is (epochs, times)
    # Compute autocorrelation per epoch and average
    n_epochs, n_times = data.shape
    max_lag = int(2.0 * sfreq)  # 2 seconds max lag
    if max_lag > n_times - 1:
        max_lag = n_times - 1

    autocorr = np.zeros(max_lag * 2 + 1)
    for i in range(n_epochs):
        corr = np.correlate(data[i], data[i], mode="full")
        # normalize
        corr /= corr[len(corr) // 2]
        # extract center
        center = len(corr) // 2
        autocorr += corr[center - max_lag : center + max_lag + 1]
    autocorr /= n_epochs

    lags = np.arange(-max_lag, max_lag + 1) / sfreq

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags, autocorr, color="#276B8A")
    ax.set_title(f"ICA{component:03d} Epoch-Averaged Autocorrelation")
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Autocorrelation")
    ax.grid(True, alpha=0.3)
    return fig


print("Script written")
