function outputs = computeTfrBatch(configPath, options)
% computeTfrBatch Compute TFR outputs from the FieldTrip ICA components.
    arguments
        configPath (1, 1) string
        options.Subjects string = strings(0, 1)
        options.Overwrite (1, 1) logical = false
    end

    config = loadFieldTripConfig(configPath);
    initializeFieldTrip(config);
    exports = findSubjectExports(config, options.Subjects);
    outputs = strings(height(exports), 1);
    for index = 1:height(exports)
        outputs(index, :) = computeSubjectTfr( ...
            config, exports.subject(index), exports.path(index), options.Overwrite);
    end
end

function outputPath = computeSubjectTfr(config, subject, exportPath, overwrite)
    task = string(config.study.task);
    icaDirectory = fullfile(string(config.paths.output), "ica", subject, "eeg");
    icaPath = fullfile(icaDirectory, ...
        subject + "_task-" + task + "_desc-fieldtripica_components.mat");
    if ~isfile(icaPath)
        error("fieldtripTfr:MissingIca", "ICA output does not exist: %s", icaPath);
    end

    outputDirectory = fullfile(string(config.paths.output), "pow");
    outputPath = fullfile(outputDirectory, subject + "_ICA_pow_EEG.mat");
    if isfile(outputPath) && ~overwrite
        error("fieldtripTfr:ExistingTfr", ...
            "TFR output already exists for %s: %s", subject, outputPath);
    end

    package = loadExportPackage(exportPath);
    decomposition = load(icaPath);
    if ~isfield(decomposition, "componentBroadband")
        error("fieldtripTfr:InvalidIca", ...
            "ICA output does not contain componentBroadband: %s", icaPath);
    end

    conditionMasks = buildTfrConditionMasks(config, package.metadata);
    results = computeTfrResultSet( ...
        config, decomposition.componentBroadband, conditionMasks);
    results.comp = decomposition.componentBroadband;

    if ~isfolder(outputDirectory)
        mkdir(outputDirectory);
    end
    save(outputPath, "-struct", "results", "-v7.3");
    fprintf("Completed TFR for %s: %s\n", subject, outputPath);
end
