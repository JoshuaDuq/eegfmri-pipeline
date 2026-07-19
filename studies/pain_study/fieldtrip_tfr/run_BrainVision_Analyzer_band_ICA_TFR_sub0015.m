clear;
clc;

pipelineRoot = "/Users/joduq24/Desktop/EEG_fMRI_Pipeline";
matlabFunctions = fullfile( ...
    pipelineRoot, "studies", "pain_study", "fieldtrip_tfr", "matlab");
exportConfig = fullfile( ...
    pipelineRoot, "studies", "pain_study", "fieldtrip_tfr", "config", ...
    "fieldtrip_tfr_brainvision_analyzer_sub0015.yaml");
pythonExecutable = fullfile(pipelineRoot, ".venv", "bin", "python");
outputRoot = ...
    "/Volumes/KINGSTON/EEG_fMRI_data/derivatives/fieldtrip_brainvision_analyzer_sub-0015";
runtimeConfig = fullfile(outputRoot, "fieldtrip_tfr_runtime.json");
subject = "sub-0015";

if ~isfile(pythonExecutable)
    error("fieldtripTfr:MissingPython", ...
        "Project Python executable does not exist: %s", pythonExecutable);
end
if ~isfile(exportConfig)
    error("fieldtripTfr:MissingConfig", ...
        "FieldTrip export configuration does not exist: %s", exportConfig);
end

exportCommand = sprintf( ...
    '"%s" -m studies.pain_study.fieldtrip_tfr.export_clean_epochs_fieldtrip --config "%s" --subject %s --overwrite', ...
    pythonExecutable, exportConfig, subject);

originalDirectory = pwd;
directoryCleanup = onCleanup(@() cd(originalDirectory));
cd(pipelineRoot);
[exportStatus, exportOutput] = system(exportCommand, "-echo");
if exportStatus ~= 0
    error("fieldtripTfr:ExportFailure", ...
        "FieldTrip export failed with status %d:\n%s", ...
        exportStatus, exportOutput);
end
if ~isfile(runtimeConfig)
    error("fieldtripTfr:MissingRuntimeConfig", ...
        "Exporter did not create the runtime configuration: %s", runtimeConfig);
end

addpath(matlabFunctions);
prepareBandIcaBatch( ...
    runtimeConfig, Subjects=subject, Overwrite=true);
computeBandTfrBatch( ...
    runtimeConfig, Subjects=subject, Overwrite=true);
computeSensorTfrBatch( ...
    runtimeConfig, Subjects=subject, Overwrite=true);

fprintf("Completed alpha, beta, gamma, and sensor TFR analysis for %s.\n", subject);
fprintf("Outputs: %s\n", outputRoot);
