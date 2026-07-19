function outputs = computeSensorTfrBatch(configPath, options)
% computeSensorTfrBatch Compute one sensor-space TFR result set per subject.
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
        outputs(index) = computeSubjectSensorTfr( ...
            config, exports.subject(index), exports.path(index), options.Overwrite);
    end
end

function outputPath = computeSubjectSensorTfr( ...
        config, subject, exportPath, overwrite)
    outputDirectory = fullfile(string(config.paths.output), "pow", "sensor");
    outputPath = fullfile(outputDirectory, subject + "_pow_EEG.mat");
    if isfile(outputPath) && ~overwrite
        error("fieldtripTfr:ExistingSensorTfr", ...
            "Sensor TFR output already exists for %s: %s", subject, outputPath);
    end

    package = loadExportPackage(exportPath);
    data = package.broadband_data;
    conditionMasks = buildTfrConditionMasks(config, package.metadata);
    results = computeTfrResultSet(config, data, conditionMasks);
    results.data = data;

    if ~isfolder(outputDirectory)
        mkdir(outputDirectory);
    end
    save(outputPath, "-struct", "results", "-v7.3");
    fprintf("Completed sensor TFR for %s: %s\n", subject, outputPath);
end
