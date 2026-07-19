clear;
clc;

fieldtripPath = "/Users/joduq24/Documents/MATLAB/fieldtrip-master";
pipelineRoot = "/Users/joduq24/Desktop/EEG_fMRI_Pipeline";
matlabFunctions = fullfile( ...
    pipelineRoot, "studies", "pain_study", "fieldtrip_tfr", "matlab");
outputRoot = ...
    "/Volumes/KINGSTON/EEG_fMRI_data/derivatives/fieldtrip_brainvision_analyzer_sub-0015";
subject = "sub-0015";

if ~isfile(fullfile(fieldtripPath, "ft_defaults.m"))
    error("fieldtripTfr:MissingFieldTrip", ...
        "FieldTrip does not exist at: %s", fieldtripPath);
end

addpath(fieldtripPath);
addpath(matlabFunctions);
ft_defaults;

tfrFile = fullfile( ...
    outputRoot, "pow", "sensor", subject + "_pow_EEG.mat");
if ~isfile(tfrFile)
    error("fieldtripTfr:MissingSensorTfr", ...
        "Sensor TFR output does not exist: %s", tfrFile);
end

plotVariables = tfrPlotVariables();
variableNames = cellstr(plotVariables.names);
results = load(tfrFile, "data", variableNames{:});
if ~isfield(results, "data") || ~isfield(results.data, "elec")
    error("fieldtripTfr:InvalidSensorTfr", ...
        "Sensor TFR output does not contain data.elec.");
end
if numel(results.data.label) ~= 63
    error("fieldtripTfr:InvalidSensorTfr", ...
        "Expected 63 EEG channels, found %d.", numel(results.data.label));
end

layoutConfig = [];
layoutConfig.elec = results.data.elec;
sensorLayout = ft_prepare_layout(layoutConfig);

for variableIndex = 1:height(plotVariables)
    variableName = plotVariables.names(variableIndex);
    if ~isfield(results, variableName)
        error("fieldtripTfr:MissingVariable", ...
            "Sensor TFR output does not contain %s.", variableName);
    end
    power = results.(variableName);
    if ~isequal(string(power.label(:)), string(results.data.label(:)))
        error("fieldtripTfr:ChannelMismatch", ...
            "%s channel order does not match the saved sensor data.", variableName);
    end
    if any(~isfinite(power.powspctrm), "all")
        error("fieldtripTfr:InvalidPower", ...
            "%s contains non-finite power.", variableName);
    end

    figure( ...
        "Name", subject + " sensor TFR — " + variableName, ...
        "Color", "w", ...
        "Units", "normalized", ...
        "OuterPosition", [0, 0, 1, 1]);
    cfg = [];
    cfg.parameter = "powspctrm";
    cfg.xlim = [power.time(1), power.time(end)];
    cfg.ylim = [power.freq(1), power.freq(end)];
    cfg.zlim = tfrPlotColorLimits( ...
        power.powspctrm, plotVariables.signed(variableIndex), variableName);
    cfg.interactive = "yes";
    cfg.layout = sensorLayout;
    cfg.colormap = turbo(256);
    cfg.colorbar = "yes";
    ft_multiplotTFR(cfg, power);
end
