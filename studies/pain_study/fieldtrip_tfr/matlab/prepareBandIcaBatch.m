function outputs = prepareBandIcaBatch(configPath, options)
% prepareBandIcaBatch Fit band-specific ICA on pooled participant runs.
    arguments
        configPath (1, 1) string
        options.Subjects string = strings(0, 1)
        options.Overwrite (1, 1) logical = false
    end

    config = loadFieldTripConfig(configPath);
    initializeFieldTrip(config);
    bands = loadIcaBands(config);
    exports = findSubjectExports(config, options.Subjects);
    outputs = strings(height(exports), height(bands));

    for subjectIndex = 1:height(exports)
        for bandIndex = 1:height(bands)
            subject = exports.subject(subjectIndex);
            bandName = bands.name(bandIndex);
            fprintf("Preparing %s ICA for %s (%d/%d)\n", ...
                bandName, subject, subjectIndex, height(exports));
            outputs(subjectIndex, bandIndex) = prepareSubjectBandIca( ...
                config, subject, exports.path(subjectIndex), ...
                bandName, bands.frequency_hz(bandIndex, :), options.Overwrite);
            fprintf("Completed %s ICA for %s\n", bandName, subject);
        end
    end
end

function outputPath = prepareSubjectBandIca( ...
        config, subject, exportPath, bandName, bandFrequencyHz, overwrite)
    task = string(config.study.task);
    outputDirectory = fullfile( ...
        string(config.paths.output), "ica", bandName, subject, "eeg");
    outputPath = fullfile(outputDirectory, ...
        subject + "_task-" + task + "_desc-fieldtripica_components.mat");
    provenancePath = fullfile(outputDirectory, ...
        subject + "_task-" + task + "_desc-fieldtripica_provenance.json");
    if (isfile(outputPath) || isfile(provenancePath)) && ~overwrite
        error("fieldtripTfr:ExistingBandIca", ...
            "%s ICA output already exists for %s.", bandName, subject);
    end

    package = loadExportPackage(exportPath);
    icaChannels = cellstr(string(package.metadata.ica_channels));
    if numel(icaChannels) < 2
        error("fieldtripTfr:InvalidChannels", ...
            "%s has fewer than two valid ICA channels.", subject);
    end

    selectConfig = [];
    selectConfig.channel = icaChannels;
    broadbandData = ft_selectdata(selectConfig, package.broadband_data);

    filterConfig = [];
    filterConfig.bpfilter = "yes";
    filterConfig.bpfreq = bandFrequencyHz;
    bandData = ft_preprocessing(filterConfig, broadbandData);
    bandData.trialinfo = broadbandData.trialinfo;
    dataRank = computeDataRank(bandData);

    rng(double(config.ica.random_seed), "twister");
    analysisConfig = [];
    analysisConfig.method = char(string(config.ica.method));
    analysisConfig.demean = "no";
    analysisConfig.numcomponent = dataRank;
    analysisConfig.runica.extended = double(logical(config.ica.extended));
    analysisConfig.runica.maxsteps = double(config.ica.max_iterations);
    componentFit = ft_componentanalysis(analysisConfig, bandData);

    applicationConfig = [];
    applicationConfig.demean = "no";
    applicationConfig.unmixing = componentFit.unmixing;
    applicationConfig.topolabel = componentFit.topolabel;
    componentBroadband = ft_componentanalysis(applicationConfig, broadbandData);
    componentBroadband.trialinfo = broadbandData.trialinfo;

    if ~isfolder(outputDirectory)
        mkdir(outputDirectory);
    end
    exportId = string(package.metadata.export_id);
    save(outputPath, "componentFit", "componentBroadband", "broadbandData", ...
        "exportId", "dataRank", "bandName", "bandFrequencyHz", "-v7.3");

    provenance = struct( ...
        subject=subject, ...
        task=task, ...
        export_id=exportId, ...
        export_path=exportPath, ...
        ica_band=bandName, ...
        ica_band_frequency_hz=bandFrequencyHz, ...
        component_count=numel(componentBroadband.label), ...
        data_rank=dataRank, ...
        ica_channels={icaChannels}, ...
        created_at=string(datetime("now", TimeZone="UTC")));
    writeJson(provenancePath, provenance);
end

function bands = loadIcaBands(config)
    if ~isfield(config.ica, "bands") || ~isstruct(config.ica.bands)
        error("fieldtripTfr:MissingIcaBands", ...
            "Configuration ica.bands must define alpha, beta, and gamma.");
    end

    expectedNames = ["alpha"; "beta"; "gamma"];
    actualNames = sort(string(fieldnames(config.ica.bands)));
    if ~isequal(actualNames, sort(expectedNames))
        error("fieldtripTfr:InvalidIcaBands", ...
            "Configuration ica.bands must contain exactly alpha, beta, and gamma.");
    end

    frequencies = zeros(numel(expectedNames), 2);
    for index = 1:numel(expectedNames)
        frequencyHz = double(config.ica.bands.(expectedNames(index)));
        frequencyHz = frequencyHz(:)';
        if numel(frequencyHz) ~= 2 || any(~isfinite(frequencyHz)) || ...
                frequencyHz(1) >= frequencyHz(2)
            error("fieldtripTfr:InvalidIcaBand", ...
                "ICA band %s must contain two finite increasing frequencies.", ...
                expectedNames(index));
        end
        frequencies(index, :) = frequencyHz;
    end
    bands = table(expectedNames, frequencies, ...
        VariableNames=["name", "frequency_hz"]);
end

function writeJson(path, value)
    fileId = fopen(path, "w");
    if fileId == -1
        error("fieldtripTfr:WriteFailure", "Cannot write JSON file: %s", path);
    end
    cleanup = onCleanup(@() fclose(fileId));
    fprintf(fileId, "%s\n", jsonencode(value, PrettyPrint=true));
    clear cleanup;
end
