function masks = buildTfrConditionMasks(config, metadata)
% buildTfrConditionMasks Validate trial metadata and build analysis masks.
    trialinfo = double(metadata.trialinfo);
    labels = string(metadata.trialinfo_labels);
    runColumn = requireColumn(labels, "run_id");
    trialColumn = requireColumn(labels, "trial_number");
    temperatureColumn = requireColumn(labels, "stimulus_temp");
    painColumn = requireColumn(labels, "pain_binary_coded");

    identifiers = trialinfo(:, [runColumn, trialColumn]);
    if size(unique(identifiers, "rows"), 1) ~= size(identifiers, 1)
        error("fieldtripTfr:InvalidMetadata", ...
            "Trial metadata contain duplicated run/trial identifiers.");
    end

    temperatures = double(config.study.temperature_levels_c(:)');
    observedTemperature = trialinfo(:, temperatureColumn);
    temperatureMasks = cell(numel(temperatures), 1);
    temperatureLabels = strings(numel(temperatures), 1);
    for index = 1:numel(temperatures)
        temperatureMasks{index} = observedTemperature == temperatures(index);
        temperatureLabels(index) = temperatureField(temperatures(index));
        if ~any(temperatureMasks{index})
            error("fieldtripTfr:MissingCondition", ...
                "No trials exist for temperature %.1f C.", temperatures(index));
        end
    end

    painValues = trialinfo(:, painColumn);
    painful = painValues == double(config.study.painful_value);
    nonpainful = painValues == double(config.study.nonpainful_value);
    if ~any(painful) || ~any(nonpainful)
        error("fieldtripTfr:MissingCondition", ...
            "Painful and non-painful conditions both require at least one trial.");
    end

    lowTemperatures = double(config.study.low_temperature_levels_c(:)');
    highTemperatures = double(config.study.high_temperature_levels_c(:)');
    if ~all(ismember(lowTemperatures, temperatures)) || ...
            ~all(ismember(highTemperatures, temperatures))
        error("fieldtripTfr:InvalidTemperatureGroups", ...
            "Low and high temperature levels must belong to temperature_levels_c.");
    end

    masks = struct( ...
        temperatures=temperatures, ...
        temperature_labels=temperatureLabels, ...
        temperature_masks={temperatureMasks}, ...
        painful=painful, ...
        nonpainful=nonpainful, ...
        low_temperature_levels=lowTemperatures, ...
        high_temperature_levels=highTemperatures);
end

function column = requireColumn(labels, name)
    matches = find(labels == name);
    if numel(matches) ~= 1
        error("fieldtripTfr:InvalidMetadata", ...
            "Trial metadata require exactly one '%s' column.", name);
    end
    column = matches;
end

function name = temperatureField(temperature)
    name = "temp_" + replace(sprintf("%.1f", temperature), ".", "p");
end
