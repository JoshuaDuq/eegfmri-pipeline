# Third-Party Notices

## NeuXus EEG-fMRI QRS detector (removed 2026-07-30)

The native EEG-fMRI preprocessing package adapted the R-peak prediction algorithm and trained
model weights from NeuXus. That package was removed on 2026-07-30 when the native correction
approach was abandoned, so no NeuXus-derived code or model asset ships in the current tree.

This notice is kept because the adapted source and weights remain in this repository's history,
where they were distributed under GPL-3.0-only:

- Project: [LaSEEB/NeuXus](https://github.com/LaSEEB/NeuXus)
- Release: `v0.0.4`
- Source revision: `f13ff7157071731c9851d1d81eda0b36a5938ef0`
- Upstream model: `examples/mri-artifact-correction/weights-input-500.pkl`
- Upstream model SHA-256: `f6e9a0d06f8b4878e5207f9210bb6f8d6d0458720d943288791c1c7f5f1aa88b`
- License: GNU General Public License version 3

NeuXus is described in:

> Caetano G, Esteves I, Vourvopoulos A, Fleury M, Figueiredo P. NeuXus open-source tool for
> real-time artifact reduction in simultaneous EEG-fMRI. NeuroImage. 2023;280:120353.
> doi:10.1016/j.neuroimage.2023.120353.

The project modified the upstream implementation for deterministic offline processing. It removed
the streaming graph, loaded model parameters from a non-executable NumPy archive, filtered the full
ECG before windowing, consolidated overlapping predictions over the complete recording, and passed
the resulting R-peaks to MNE-Python PCA-OBS instead of NeuXus pulse average-artifact subtraction.
