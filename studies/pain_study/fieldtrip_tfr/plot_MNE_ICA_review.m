clear;
clc;

workflowDirectory = fileparts(mfilename("fullpath"));
matlabDirectory = fullfile(workflowDirectory, "matlab");
configPath = fullfile(workflowDirectory, "config", "mne_ica_review.yaml");

addpath(matlabDirectory);
runtimeConfig = prepareMneIcaReviewRuntime(configPath, ExportComponents=false);
plotMneIcaReviewBatch(runtimeConfig);
