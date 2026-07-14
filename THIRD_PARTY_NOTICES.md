# Third-Party Notices

## NeuXus EEG-fMRI QRS detector

The native EEG-fMRI preprocessing package adapts the R-peak prediction algorithm and trained model
weights from NeuXus:

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

This project modifies the upstream implementation for deterministic offline processing. It removes
the streaming graph, loads model parameters from a non-executable NumPy archive, filters the full
ECG before windowing, consolidates overlapping predictions over the complete recording, and passes
the resulting R-peaks to MNE-Python PCA-OBS instead of NeuXus pulse average-artifact subtraction.

The adapted source and model assets are distributed under GPL-3.0-only as part of this project.
