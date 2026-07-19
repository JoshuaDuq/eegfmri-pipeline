clear;
clc;

fieldtripPath = "/Users/joduq24/Documents/MATLAB/fieldtrip-master";
pipelineRoot = "/Users/joduq24/Desktop/EEG_fMRI_Pipeline";
matlabFunctions = fullfile( ...
    pipelineRoot, "studies", "pain_study", "fieldtrip_tfr", "matlab");
outputRoot = ...
    "/Volumes/KINGSTON/EEG_fMRI_data/derivatives/fieldtrip_brainvision_analyzer_sub-0015";
subject = "sub-0015";
bandName = "alpha";  % "alpha", "beta", or "gamma"

if ~isfile(fullfile(fieldtripPath, "ft_defaults.m"))
    error("fieldtripTfr:MissingFieldTrip", ...
        "FieldTrip does not exist at: %s", fieldtripPath);
end
if ~ismember(bandName, ["alpha", "beta", "gamma"])
    error("fieldtripTfr:InvalidBand", ...
        "bandName must be alpha, beta, or gamma.");
end

addpath(fieldtripPath);
addpath(matlabFunctions);
ft_defaults;

icaFile = fullfile( ...
    outputRoot, "ica", bandName, subject, "eeg", ...
    subject + "_task-thermalactive_desc-fieldtripica_components.mat");
tfrFile = fullfile( ...
    outputRoot, "pow", bandName, subject + "_ICA_pow_EEG.mat");
if ~isfile(icaFile)
    error("fieldtripTfr:MissingIca", "ICA output does not exist: %s", icaFile);
end
if ~isfile(tfrFile)
    error("fieldtripTfr:MissingTfr", "TFR output does not exist: %s", tfrFile);
end

decomposition = load(icaFile, "componentFit");
if ~isfield(decomposition, "componentFit")
    error("fieldtripTfr:InvalidIca", ...
        "ICA output does not contain componentFit: %s", icaFile);
end
componentFit = decomposition.componentFit;
requiredIcaFields = ["label", "topo", "topolabel", "unmixing", "elec"];
for fieldName = requiredIcaFields
    if ~isfield(componentFit, fieldName)
        error("fieldtripTfr:InvalidIca", ...
            "componentFit does not contain %s.", fieldName);
    end
end
if any(~isfinite(componentFit.topo), "all")
    error("fieldtripTfr:InvalidIca", ...
        "ICA topographies contain non-finite values.");
end

plotVariables = tfrPlotVariables();
variableNames = cellstr(plotVariables.names);
results = load(tfrFile, "comp", variableNames{:});
if ~isfield(results, "comp") || ...
        ~isequal(componentFit.unmixing, results.comp.unmixing) || ...
        ~isequal(string(componentFit.topolabel(:)), ...
            string(results.comp.topolabel(:)))
    error("fieldtripTfr:ComponentMismatch", ...
        "The ICA maps and TFR components are not the same decomposition.");
end

componentCount = numel(componentFit.label);
if componentCount > 64
    error("fieldtripTfr:TooManyComponents", ...
        "Four 4-by-4 figures can display at most 64 components, found %d.", ...
        componentCount);
end

layoutConfig = [];
layoutConfig.elec = componentFit.elec;
componentLayout = ft_prepare_layout(layoutConfig);
componentGroups = {
    1:min(16, componentCount)
    17:min(32, componentCount)
    33:min(48, componentCount)
    49:componentCount
};

for groupIndex = 1:numel(componentGroups)
    components = componentGroups{groupIndex};
    if isempty(components)
        continue;
    end
    figure( ...
        "Name", sprintf("%s %s ICA maps — components %d-%d", ...
            subject, bandName, components(1), components(end)), ...
        "Color", "w", ...
        "Units", "normalized", ...
        "OuterPosition", [0, 0, 1, 1]);
    cfg = [];
    cfg.component = components;
    cfg.layout = componentLayout;
    cfg.colormap = turbo(256);
    cfg.comment = "no";
    cfg.colorbar = "no";
    ft_topoplotIC(cfg, componentFit);
end

for variableIndex = 1:height(plotVariables)
    variableName = plotVariables.names(variableIndex);
    if ~isfield(results, variableName)
        error("fieldtripTfr:MissingVariable", ...
            "TFR output does not contain %s.", variableName);
    end
    power = results.(variableName);
    if ~isequal(string(power.label(:)), string(results.comp.label(:))) || ...
            numel(power.label) ~= componentCount
        error("fieldtripTfr:ComponentMismatch", ...
            "%s component order does not match the ICA decomposition.", variableName);
    end
    if any(~isfinite(power.powspctrm), "all")
        error("fieldtripTfr:InvalidPower", ...
            "%s contains non-finite power.", variableName);
    end

    for groupIndex = 1:numel(componentGroups)
        components = componentGroups{groupIndex};
        if isempty(components)
            continue;
        end
        componentFigure = figure( ...
            "Name", sprintf("%s %s %s — components %d-%d", ...
                subject, bandName, variableName, components(1), components(end)), ...
            "Color", "w", ...
            "Units", "normalized", ...
            "OuterPosition", [0, 0, 1, 1]);

        for subplotIndex = 1:numel(components)
            componentNumber = components(subplotIndex);
            subplot(4, 4, subplotIndex, "Parent", componentFigure);
            componentPower = squeeze(power.powspctrm(componentNumber, :, :));
            colorLimits = tfrPlotColorLimits( ...
                componentPower, plotVariables.signed(variableIndex), variableName);
            imagesc(power.time, power.freq, componentPower, colorLimits);
            axis xy;
            xlim([power.time(1), power.time(end)]);
            ylim([power.freq(1), power.freq(end)]);
            colorbar;
            xline(0, "--w", LineWidth=0.75);
            title("ic" + componentNumber);
        end
        colormap(componentFigure, turbo(256));
        sgtitle(sprintf("%s — %s ICA — %s", ...
            subject, bandName, variableName), Interpreter="none");
    end
end
