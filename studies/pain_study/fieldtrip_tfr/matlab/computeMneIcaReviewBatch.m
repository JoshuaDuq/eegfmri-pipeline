function outputs = computeMneIcaReviewBatch(runtimePath)
% computeMneIcaReviewBatch Compute configured TFRs of exact MNE components.
    config = loadMneIcaReviewConfig(runtimePath);
    initializeFieldTrip(config);

    subjects = config.resolved_participants;
    outputs = strings(numel(subjects), 1);
    for index = 1:numel(subjects)
        outputs(index) = computeSubject(config, subjects(index));
    end
end

function outputPath = computeSubject(config, subject)
    task = string(config.study.task);
    outputRoot = string(config.paths.output);
    exportPath = fullfile( ...
        outputRoot, "exports", subject, ...
        subject + "_task-" + task + "_desc-mneica_components.mat");
    if ~isfile(exportPath)
        error("mneIcaReview:MissingExport", ...
            "Component export does not exist for %s: %s", subject, exportPath);
    end

    outputDirectory = fullfile(outputRoot, "pow");
    outputPath = fullfile(outputDirectory, subject + "_MNE_ICA_TFR.mat");
    if isfile(outputPath) && ~logical(config.execution.overwrite_tfr)
        fprintf("Keeping existing component TFR for %s: %s\n", subject, outputPath);
        return;
    end

    package = load(exportPath, "component", "metadata");
    validatePackage(package, subject);
    component = package.component;
    component.trialinfo = double(package.metadata.trialinfo);
    masks = buildConditionMasks(config, package.metadata);
    reviewFields = [
        "mne_proposed_bad_indices"
        "mne_proposed_bad_mask"
        "mne_status"
        "mne_status_description"
    ];
    componentData = rmfield(component, reviewFields);
    conditionResults = computeConditions(config, componentData, masks);
    componentOverview = rmfield(component, ["trial", "time", "sampleinfo", "trialinfo"]);

    if ~isfolder(outputDirectory)
        mkdir(outputDirectory);
    end
    save(outputPath, "componentOverview", "conditionResults", "-v7.3");
    fprintf("Completed exact-MNE component TFR for %s: %s\n", subject, outputPath);
end

function validatePackage(package, subject)
    if ~isfield(package, "component") || ~isfield(package, "metadata")
        error("mneIcaReview:InvalidExport", ...
            "Export for %s must contain component and metadata.", subject);
    end
    componentFields = [
        "label"
        "trial"
        "time"
        "topo"
        "topolabel"
        "elec"
        "mne_proposed_bad_indices"
        "mne_proposed_bad_mask"
        "mne_status"
        "mne_status_description"
    ];
    for fieldName = componentFields'
        if ~isfield(package.component, fieldName)
            error("mneIcaReview:InvalidExport", ...
                "Component export for %s lacks '%s'.", subject, fieldName);
        end
    end
    trialCount = numel(package.component.trial);
    if size(package.metadata.trialinfo, 1) ~= trialCount
        error("mneIcaReview:InvalidExport", ...
            "Trial metadata and component trial counts differ for %s.", subject);
    end
    if any(~isfinite(package.component.topo), "all")
        error("mneIcaReview:InvalidExport", ...
            "Component topographies contain non-finite values for %s.", subject);
    end
end

function masks = buildConditionMasks(config, metadata)
    trialinfo = double(metadata.trialinfo);
    labels = string(metadata.trialinfo_labels);
    runColumn = requireColumn(labels, "run_id");
    trialColumn = requireColumn(labels, "trial_number");
    temperatureColumn = requireColumn(labels, "stimulus_temp");
    painColumn = requireColumn(labels, "pain_binary_coded");

    identifiers = trialinfo(:, [runColumn, trialColumn]);
    if size(unique(identifiers, "rows"), 1) ~= size(identifiers, 1)
        error("mneIcaReview:InvalidMetadata", ...
            "Trial metadata contain duplicated run/trial identifiers.");
    end

    masks.temperatures = double(config.study.temperature_levels_c(:));
    observedTemperatures = trialinfo(:, temperatureColumn);
    masks.temperature = cell(numel(masks.temperatures), 1);
    for index = 1:numel(masks.temperatures)
        masks.temperature{index} = observedTemperatures == masks.temperatures(index);
        requireTrials(masks.temperature{index}, ...
            sprintf("temperature %.1f C", masks.temperatures(index)));
    end

    painValues = trialinfo(:, painColumn);
    masks.painful = painValues == double(config.study.painful_value);
    masks.nonpainful = painValues == double(config.study.nonpainful_value);
    requireTrials(masks.painful, "painful");
    requireTrials(masks.nonpainful, "non-painful");

    lowLevels = double(config.study.low_temperature_levels_c(:));
    highLevels = double(config.study.high_temperature_levels_c(:));
    if ~all(ismember([lowLevels; highLevels], masks.temperatures))
        error("mneIcaReview:InvalidMetadata", ...
            "Low and high temperature levels must be configured temperature levels.");
    end
    masks.low = ismember(observedTemperatures, lowLevels);
    masks.high = ismember(observedTemperatures, highLevels);
    requireTrials(masks.low, "low temperature");
    requireTrials(masks.high, "high temperature");
    masks.all = true(size(observedTemperatures));
end

function column = requireColumn(labels, name)
    matches = find(labels == name);
    if numel(matches) ~= 1
        error("mneIcaReview:InvalidMetadata", ...
            "Trial metadata require exactly one '%s' column.", name);
    end
    column = matches;
end

function requireTrials(mask, conditionName)
    if ~any(mask)
        error("mneIcaReview:MissingCondition", ...
            "No retained trials exist for %s.", conditionName);
    end
end

function results = computeConditions(config, component, masks)
    requested = config.conditions;
    temperatureConditions = [
        "high_vs_low"
        "high_temperature"
        "low_temperature"
        "individual_temperatures"
        "temperature_slope"
    ];
    painConditions = ["painful_vs_nonpainful"; "painful"; "nonpainful"];
    needsTemperature = any(ismember(requested, temperatureConditions));
    needsPain = any(ismember(requested, painConditions));

    temperaturePower = {};
    temperatureCounts = [];
    temperatureDb = {};
    if needsTemperature
        count = numel(masks.temperatures);
        temperaturePower = cell(count, 1);
        temperatureCounts = zeros(count, 1);
        temperatureDb = cell(count, 1);
        for index = 1:count
            temperaturePower{index} = computeTfr(config, component, masks.temperature{index});
            temperatureCounts(index) = sum(masks.temperature{index});
            temperatureDb{index} = baselineTfr(config, temperaturePower{index});
        end
    end

    painfulDb = [];
    nonpainfulDb = [];
    if needsPain
        painfulDb = baselineTfr(config, computeTfr(config, component, masks.painful));
        nonpainfulDb = baselineTfr(config, computeTfr(config, component, masks.nonpainful));
    end

    individualTemperatureCount = sum(requested == "individual_temperatures") * ...
        (numel(masks.temperatures) - 1);
    resultCount = numel(requested) + individualTemperatureCount;
    results = repmat(result("", "", "", struct()), 1, resultCount);
    resultIndex = 0;
    for requestedIndex = 1:numel(requested)
        name = requested(requestedIndex);
        switch name
            case "high_vs_low"
                [lowDb, highDb] = groupedTemperatureDb( ...
                    config, masks, temperaturePower, temperatureCounts);
                resultIndex = resultIndex + 1;
                results(resultIndex) = result( ...
                    name, "High minus low temperature", "dB", subtractTfr(highDb, lowDb));
            case "high_temperature"
                [~, highDb] = groupedTemperatureDb( ...
                    config, masks, temperaturePower, temperatureCounts);
                resultIndex = resultIndex + 1;
                results(resultIndex) = result(name, "High temperature", "dB", highDb);
            case "low_temperature"
                [lowDb, ~] = groupedTemperatureDb( ...
                    config, masks, temperaturePower, temperatureCounts);
                resultIndex = resultIndex + 1;
                results(resultIndex) = result(name, "Low temperature", "dB", lowDb);
            case "painful_vs_nonpainful"
                resultIndex = resultIndex + 1;
                results(resultIndex) = result(name, ...
                    "Painful minus non-painful", "dB", ...
                    subtractTfr(painfulDb, nonpainfulDb));
            case "painful"
                resultIndex = resultIndex + 1;
                results(resultIndex) = result(name, "Painful", "dB", painfulDb);
            case "nonpainful"
                resultIndex = resultIndex + 1;
                results(resultIndex) = result(name, "Non-painful", "dB", nonpainfulDb);
            case "individual_temperatures"
                for temperatureIndex = 1:numel(masks.temperatures)
                    temperature = masks.temperatures(temperatureIndex);
                    fieldName = "temperature_" + replace(sprintf("%.1f", temperature), ".", "p");
                    title = sprintf("Temperature %.1f C", temperature);
                    resultIndex = resultIndex + 1;
                    results(resultIndex) = result( ...
                        fieldName, title, "dB", temperatureDb{temperatureIndex});
                end
            case "temperature_slope"
                slope = temperatureSlope(temperatureDb, masks.temperatures);
                resultIndex = resultIndex + 1;
                results(resultIndex) = result( ...
                    name, "Temperature slope", "dB / degree C", slope);
            case "grand_average"
                averageDb = baselineTfr(config, computeTfr(config, component, masks.all));
                resultIndex = resultIndex + 1;
                results(resultIndex) = result(name, "Grand average", "dB", averageDb);
        end
    end
end

function entry = result(name, title, units, power)
    entry = struct("name", name, "title", title, "units", units, "power", power);
end

function power = computeTfr(config, component, mask)
    tfrConfig = [];
    tfrConfig.method = "mtmconvol";
    tfrConfig.output = "pow";
    tfrConfig.pad = double(config.tfr.padding_s);
    tfrConfig.taper = char(string(config.tfr.taper));
    tfrConfig.foi = double(config.tfr.frequency_min_hz): ...
        double(config.tfr.frequency_step_hz):double(config.tfr.frequency_max_hz);
    tfrConfig.t_ftimwin = repmat(double(config.tfr.window_s), size(tfrConfig.foi));
    tfrConfig.toi = double(config.tfr.time_min_s): ...
        double(config.tfr.time_step_s):double(config.tfr.time_max_s);
    tfrConfig.tapsmofrq = repmat(double(config.tfr.smoothing_hz), size(tfrConfig.foi));
    tfrConfig.trials = find(mask);
    tfrConfig.keeptrials = "no";
    power = ft_freqanalysis(tfrConfig, component);
    if any(~isfinite(power.powspctrm), "all")
        error("mneIcaReview:InvalidPower", ...
            "FieldTrip produced non-finite component power.");
    end
    power.epoch_count = sum(mask);
end

function normalized = baselineTfr(config, power)
    baselineConfig = [];
    baselineConfig.baseline = [
        double(config.epochs.baseline_tmin_s)
        double(config.epochs.baseline_tmax_s)
    ];
    baselineConfig.baselinetype = "db";
    normalized = ft_freqbaseline(baselineConfig, power);
end

function [lowDb, highDb] = groupedTemperatureDb( ...
        config, masks, temperaturePower, temperatureCounts)
    lowLevels = double(config.study.low_temperature_levels_c(:));
    highLevels = double(config.study.high_temperature_levels_c(:));
    lowIndices = ismember(masks.temperatures, lowLevels);
    highIndices = ismember(masks.temperatures, highLevels);
    lowPower = weightedAverage(temperaturePower(lowIndices), temperatureCounts(lowIndices));
    highPower = weightedAverage(temperaturePower(highIndices), temperatureCounts(highIndices));
    lowDb = baselineTfr(config, lowPower);
    highDb = baselineTfr(config, highPower);
end

function average = weightedAverage(powerSets, weights)
    average = powerSets{1};
    accumulated = zeros(size(average.powspctrm), "like", average.powspctrm);
    weights = double(weights(:));
    for index = 1:numel(powerSets)
        requireCompatible(powerSets{1}, powerSets{index});
        accumulated = accumulated + weights(index) * powerSets{index}.powspctrm;
    end
    average.powspctrm = accumulated / sum(weights);
    average.epoch_count = sum(weights);
end

function difference = subtractTfr(first, second)
    requireCompatible(first, second);
    difference = first;
    difference.powspctrm = first.powspctrm - second.powspctrm;
end

function slope = temperatureSlope(powerSets, temperatures)
    temperatures = double(temperatures(:));
    centered = temperatures - mean(temperatures);
    slope = powerSets{1};
    slope.powspctrm = zeros(size(slope.powspctrm), "like", slope.powspctrm);
    for index = 1:numel(powerSets)
        requireCompatible(powerSets{1}, powerSets{index});
        slope.powspctrm = slope.powspctrm + ...
            centered(index) * powerSets{index}.powspctrm;
    end
    slope.powspctrm = slope.powspctrm / sum(centered .^ 2);
end

function requireCompatible(first, second)
    if ~isequal(first.label, second.label) || ...
            ~isequal(first.freq, second.freq) || ...
            ~isequal(first.time, second.time) || ...
            ~isequal(size(first.powspctrm), size(second.powspctrm))
        error("mneIcaReview:IncompatiblePower", ...
            "Component TFR structures do not have identical axes.");
    end
end
