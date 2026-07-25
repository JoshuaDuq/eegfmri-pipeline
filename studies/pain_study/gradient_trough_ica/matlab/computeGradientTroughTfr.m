function resultPath = computeGradientTroughTfr(componentPath, outputDirectory, tfrSettings)
%COMPUTEGRADIENTTROUGHTFR Compute low/high component TFRs and their contrast.

arguments
    componentPath (1, 1) string
    outputDirectory (1, 1) string
    tfrSettings (1, 1) struct
end

loaded = load(componentPath, "componentBroadband", "gradientTrough");
if ~isfield(loaded, "componentBroadband") || ~isfield(loaded, "gradientTrough")
    error("gradientTrough:InvalidComponentFile", ...
        "Component file lacks componentBroadband or gradientTrough.");
end
componentBroadband = loaded.componentBroadband;
gradientTrough = loaded.gradientTrough;

labels = string(componentBroadband.trialinfo_labels(:));
temperatureColumn = find(labels == "stimulus_temp", 1);
if isempty(temperatureColumn)
    error("gradientTrough:MissingTemperature", ...
        "Component trialinfo lacks stimulus_temp.");
end
temperatures = componentBroadband.trialinfo(:, temperatureColumn);
lowMask = ismember(temperatures, double(tfrSettings.low_temperatures_c));
highMask = ismember(temperatures, double(tfrSettings.high_temperatures_c));
if any(lowMask & highMask) || ~any(lowMask) || ~any(highMask)
    error("gradientTrough:InvalidConditions", ...
        "Low/high condition masks must be non-empty and disjoint.");
end

cfg = [];
cfg.method = 'mtmconvol';
cfg.output = 'pow';
cfg.taper = 'dpss';
cfg.foi = double(tfrSettings.frequencies_hz(:))';
cfg.toi = double(tfrSettings.times_s(:))';
cfg.t_ftimwin = repmat(double(tfrSettings.window_s), size(cfg.foi));
cfg.tapsmofrq = repmat(double(tfrSettings.smoothing_hz), size(cfg.foi));
cfg.pad = double(tfrSettings.padding_s);
cfg.keeptrials = 'no';
cfg.trials = find(lowMask);
lowPower = ft_freqanalysis(cfg, componentBroadband);
cfg.trials = find(highMask);
highPower = ft_freqanalysis(cfg, componentBroadband);

baselineCfg = [];
baselineCfg.baseline = double(tfrSettings.baseline_s(:))';
baselineCfg.baselinetype = 'db';
lowDb = ft_freqbaseline(baselineCfg, lowPower);
highDb = ft_freqbaseline(baselineCfg, highPower);
contrastDb = highDb;
contrastDb.powspctrm = highDb.powspctrm - lowDb.powspctrm;
contrastDb.condition = "high_minus_low_db";

conditionCounts = struct("low", sum(lowMask), "high", sum(highMask));
lowTrialIndices = find(lowMask);
highTrialIndices = find(highMask);
if ~isfolder(outputDirectory)
    mkdir(outputDirectory);
end
resultPath = fullfile(outputDirectory, ...
    string(gradientTrough.participant) + "_desc-troughica_tfr.mat");
if isfile(resultPath)
    error("gradientTrough:ExistingResult", "Refusing to overwrite: %s", resultPath);
end
save(resultPath, "lowPower", "highPower", "lowDb", "highDb", ...
    "contrastDb", "conditionCounts", "lowTrialIndices", "highTrialIndices", ...
    "tfrSettings", "-v7.3");
end
