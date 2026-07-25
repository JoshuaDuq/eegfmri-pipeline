function resultPath = fitGradientTroughIca(exportPath, outputDirectory, icaSettings)
%FITGRADIENTTROUGHICA Fit ICA on trough data and apply it to broadband data.

arguments
    exportPath (1, 1) string
    outputDirectory (1, 1) string
    icaSettings (1, 1) struct
end

if ~isfile(exportPath)
    error("gradientTrough:MissingExport", "MATLAB export does not exist: %s", exportPath);
end
if ~isfield(icaSettings, "seed") || ~isfield(icaSettings, "extended")
    error("gradientTrough:InvalidIcaSettings", "ICA seed and extended settings are required.");
end
if ~logical(icaSettings.extended)
    error("gradientTrough:InvalidIcaSettings", ...
        "This workflow requires extended Infomax ICA.");
end

loaded = load(exportPath, "gradient_trough");
if ~isfield(loaded, "gradient_trough")
    error("gradientTrough:InvalidExport", "Export lacks variable 'gradient_trough'.");
end
gradientTrough = loaded.gradient_trough;
requiredFields = ["participant", "ica_data", "broadband_data", "elec"];
for fieldName = requiredFields
    if ~isfield(gradientTrough, fieldName)
        error("gradientTrough:InvalidExport", ...
            "Export lacks gradient_trough.%s.", fieldName);
    end
end

dataRank = computeGradientTroughDataRank(gradientTrough.ica_data);
rng(double(icaSettings.seed), "twister");

cfg = [];
cfg.method = 'runica';
cfg.numcomponent = dataRank;
cfg.demean = 'no';
cfg.runica.extended = 1;
cfg.runica.maxsteps = 2000;
componentFit = ft_componentanalysis(cfg, gradientTrough.ica_data);
componentFit.elec = gradientTrough.elec;

cfg = [];
cfg.unmixing = componentFit.unmixing;
cfg.topolabel = componentFit.topolabel;
cfg.demean = 'no';
componentBroadband = ft_componentanalysis(cfg, gradientTrough.broadband_data);
componentBroadband.trialinfo = gradientTrough.broadband_data.trialinfo;
componentBroadband.trialinfo_labels = gradientTrough.broadband_data.trialinfo_labels;
componentBroadband.elec = gradientTrough.elec;

if size(componentBroadband.trialinfo, 1) ~= numel(componentBroadband.trial)
    error("gradientTrough:TrialInfoMismatch", ...
        "Broadband component trials and trialinfo are not aligned.");
end

if ~isfolder(outputDirectory)
    mkdir(outputDirectory);
end
resultPath = fullfile(outputDirectory, ...
    string(gradientTrough.participant) + "_desc-troughica_components.mat");
if isfile(resultPath)
    error("gradientTrough:ExistingResult", "Refusing to overwrite: %s", resultPath);
end
save(resultPath, "componentFit", "componentBroadband", "dataRank", ...
    "gradientTrough", "-v7.3");
end
