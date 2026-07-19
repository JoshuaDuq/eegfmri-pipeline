function results = computeTfrResultSet(config, data, masks)
% computeTfrResultSet Compute all configured condition TFR results.
    temperaturePower = cell(numel(masks.temperatures), 1);
    temperatureCounts = zeros(numel(masks.temperatures), 1);
    results = struct();
    for index = 1:numel(masks.temperatures)
        fieldName = "pow_" + masks.temperature_labels(index);
        temperaturePower{index} = computeConditionTfr( ...
            config, data, masks.temperature_masks{index});
        temperatureCounts(index) = sum(masks.temperature_masks{index});
        results.(fieldName) = temperaturePower{index};
        results.(fieldName + "_db") = baselinePower(config, temperaturePower{index});
    end

    results.pow_painful = computeConditionTfr(config, data, masks.painful);
    results.pow_nonpainful = computeConditionTfr(config, data, masks.nonpainful);
    results.pow_painful_db = baselinePower(config, results.pow_painful);
    results.pow_nonpainful_db = baselinePower(config, results.pow_nonpainful);
    results.pow_painful_v_nonpainful = logPowerRatio( ...
        results.pow_painful, results.pow_nonpainful);

    lowMask = ismember(masks.temperatures, masks.low_temperature_levels);
    highMask = ismember(masks.temperatures, masks.high_temperature_levels);
    lowPower = averagePower(temperaturePower(lowMask), temperatureCounts(lowMask));
    highPower = averagePower(temperaturePower(highMask), temperatureCounts(highMask));
    results.pow_low_temperature = lowPower;
    results.pow_high_temperature = highPower;
    results.pow_high_v_low = logPowerRatio(highPower, lowPower);
    results.pow_temperature_slope = temperatureSlope( ...
        temperaturePower, masks.temperatures);
    results.pow_avg = averagePower(temperaturePower, temperatureCounts);
end

function power = computeConditionTfr(config, data, mask)
    trialIndices = find(mask);
    if isempty(trialIndices)
        error("fieldtripTfr:MissingCondition", ...
            "Cannot compute a TFR for an empty condition.");
    end

    windowSeconds = double(config.tfr.window_s);
    firstSampleTime = double(data.time{1}(1));
    lastSampleTime = double(data.time{1}(end));
    firstCenterTime = double(config.tfr.time_min_s);
    lastCenterTime = double(config.tfr.time_max_s);
    if firstCenterTime <= firstSampleTime + windowSeconds/2 || ...
            lastCenterTime >= lastSampleTime - windowSeconds/2
        error("fieldtripTfr:InvalidTimeWindow", ...
            "TFR centers must remain strictly inside the epoch by half a window.");
    end

    tfrConfig = [];
    tfrConfig.method = "mtmconvol";
    tfrConfig.output = "pow";
    tfrConfig.pad = double(config.tfr.padding_s);
    tfrConfig.taper = char(string(config.tfr.taper));
    tfrConfig.foi = double(config.tfr.frequency_min_hz): ...
        double(config.tfr.frequency_step_hz):double(config.tfr.frequency_max_hz);
    tfrConfig.t_ftimwin = repmat(windowSeconds, size(tfrConfig.foi));
    tfrConfig.toi = firstCenterTime:double(config.tfr.time_step_s):lastCenterTime;
    tfrConfig.tapsmofrq = repmat(double(config.tfr.smoothing_hz), size(tfrConfig.foi));
    tfrConfig.trials = trialIndices;
    tfrConfig.keeptrials = "no";
    power = ft_freqanalysis(tfrConfig, data);
    if any(~isfinite(power.powspctrm), "all")
        error("fieldtripTfr:InvalidPower", ...
            "FieldTrip produced non-finite power values.");
    end
    power.epoch_count = numel(trialIndices);
end

function normalized = baselinePower(config, power)
    baselineConfig = [];
    baselineConfig.baseline = [
        double(config.epochs.baseline_tmin_s)
        double(config.epochs.baseline_tmax_s)
    ];
    baselineConfig.baselinetype = "db";
    normalized = ft_freqbaseline(baselineConfig, power);
end

function contrast = logPowerRatio(numerator, denominator)
    requireCompatiblePower(numerator, denominator);
    if any(numerator.powspctrm <= 0, "all") || ...
            any(denominator.powspctrm <= 0, "all")
        error("fieldtripTfr:InvalidPower", ...
            "Log-power contrasts require strictly positive power values.");
    end
    contrast = numerator;
    contrast.powspctrm = log10(numerator.powspctrm./denominator.powspctrm);
    contrast.contrast = "log10_ratio";
end

function average = averagePower(powerSets, weights)
    if isempty(powerSets) || numel(powerSets) ~= numel(weights)
        error("fieldtripTfr:InvalidPower", ...
            "Power sets and averaging weights must be non-empty and equal in length.");
    end
    weights = double(weights(:));
    if any(weights <= 0) || any(~isfinite(weights))
        error("fieldtripTfr:InvalidPower", ...
            "Power averaging weights must be finite and positive.");
    end

    average = powerSets{1};
    accumulated = zeros(size(average.powspctrm), "like", average.powspctrm);
    for index = 1:numel(powerSets)
        requireCompatiblePower(average, powerSets{index});
        accumulated = accumulated + weights(index)*powerSets{index}.powspctrm;
    end
    average.powspctrm = accumulated/sum(weights);
    average.epoch_count = sum(weights);
end

function slope = temperatureSlope(powerSets, temperatures)
    temperatures = double(temperatures(:));
    if numel(powerSets) ~= numel(temperatures)
        error("fieldtripTfr:InvalidPower", ...
            "Temperature count must equal the number of temperature power sets.");
    end
    centeredTemperature = temperatures - mean(temperatures);
    denominator = sum(centeredTemperature.^2);
    slope = powerSets{1};
    slope.powspctrm = zeros(size(slope.powspctrm), "like", slope.powspctrm);
    for index = 1:numel(powerSets)
        requireCompatiblePower(slope, powerSets{index});
        if any(powerSets{index}.powspctrm <= 0, "all")
            error("fieldtripTfr:InvalidPower", ...
                "Temperature slopes require strictly positive power values.");
        end
        slope.powspctrm = slope.powspctrm + ...
            centeredTemperature(index)*log10(powerSets{index}.powspctrm);
    end
    slope.powspctrm = slope.powspctrm/denominator;
    slope.contrast = "log10_power_per_degree_celsius";
end

function requireCompatiblePower(first, second)
    if ~isequal(first.label, second.label) || ...
            ~isequal(first.freq, second.freq) || ...
            ~isequal(first.time, second.time) || ...
            ~isequal(size(first.powspctrm), size(second.powspctrm))
        error("fieldtripTfr:IncompatiblePower", ...
            "TFR structures do not share identical channel, frequency, time, and shape axes.");
    end
end
