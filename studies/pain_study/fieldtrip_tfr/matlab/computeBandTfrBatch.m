function outputs = computeBandTfrBatch(configPath, options)
% computeBandTfrBatch Compute TFR outputs for each band-specific ICA.
    arguments
        configPath (1, 1) string
        options.Subjects string = strings(0, 1)
        options.Overwrite (1, 1) logical = false
    end

    config = loadFieldTripConfig(configPath);
    initializeFieldTrip(config);
    bandNames = validateBandNames(config);
    exports = findSubjectExports(config, options.Subjects);
    outputs = strings(height(exports), numel(bandNames));

    for subjectIndex = 1:height(exports)
        for bandIndex = 1:numel(bandNames)
            outputs(subjectIndex, bandIndex) = computeSubjectBandTfr( ...
                config, exports.subject(subjectIndex), ...
                exports.path(subjectIndex), bandNames(bandIndex), options.Overwrite);
        end
    end
end

function outputPath = computeSubjectBandTfr( ...
        config, subject, exportPath, bandName, overwrite)
    task = string(config.study.task);
    icaDirectory = fullfile( ...
        string(config.paths.output), "ica", bandName, subject, "eeg");
    icaPath = fullfile(icaDirectory, ...
        subject + "_task-" + task + "_desc-fieldtripica_components.mat");
    if ~isfile(icaPath)
        error("fieldtripTfr:MissingBandIca", ...
            "%s ICA output does not exist: %s", bandName, icaPath);
    end

    outputDirectory = fullfile(string(config.paths.output), "pow", bandName);
    outputPath = fullfile(outputDirectory, subject + "_ICA_pow_EEG.mat");
    if isfile(outputPath) && ~overwrite
        error("fieldtripTfr:ExistingBandTfr", ...
            "%s TFR output already exists for %s: %s", ...
            bandName, subject, outputPath);
    end

    package = loadExportPackage(exportPath);
    decomposition = load(icaPath);
    requiredVariables = ["componentBroadband", "bandName", "bandFrequencyHz"];
    for variableName = requiredVariables
        if ~isfield(decomposition, variableName)
            error("fieldtripTfr:InvalidBandIca", ...
                "ICA output lacks %s: %s", variableName, icaPath);
        end
    end
    if string(decomposition.bandName) ~= bandName
        error("fieldtripTfr:InvalidBandIca", ...
            "ICA file band %s does not match requested band %s.", ...
            string(decomposition.bandName), bandName);
    end

    conditionMasks = buildTfrConditionMasks(config, package.metadata);
    results = computeTfrResultSet( ...
        config, decomposition.componentBroadband, conditionMasks);
    results.comp = decomposition.componentBroadband;
    results.ica_band = string(decomposition.bandName);
    results.ica_band_frequency_hz = double(decomposition.bandFrequencyHz);

    if ~isfolder(outputDirectory)
        mkdir(outputDirectory);
    end
    save(outputPath, "-struct", "results", "-v7.3");
    fprintf("Completed %s TFR for %s: %s\n", bandName, subject, outputPath);
end

function bandNames = validateBandNames(config)
    if ~isfield(config.ica, "bands") || ~isstruct(config.ica.bands)
        error("fieldtripTfr:MissingIcaBands", ...
            "Configuration ica.bands must define alpha, beta, and gamma.");
    end
    bandNames = ["alpha"; "beta"; "gamma"];
    actualNames = sort(string(fieldnames(config.ica.bands)));
    if ~isequal(actualNames, sort(bandNames))
        error("fieldtripTfr:InvalidIcaBands", ...
            "Configuration ica.bands must contain exactly alpha, beta, and gamma.");
    end
end
