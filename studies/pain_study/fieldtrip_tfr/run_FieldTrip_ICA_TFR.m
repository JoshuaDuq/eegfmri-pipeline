clear;
clc;

pipelinePath = ...
    '/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/fieldtrip_tfr/matlab';
runtimeConfig = ...
    '/Volumes/KINGSTON/EEG_fMRI_data/derivatives/fieldtrip_manual_ica_tfr/fieldtrip_tfr_runtime.json';

addpath(pipelinePath);

prepareIcaBatch(runtimeConfig, Overwrite=true);
computeTfrBatch(runtimeConfig, Overwrite=true);
