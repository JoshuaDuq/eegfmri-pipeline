function outputs = prepareIcaBatch(configPath, options)
% prepareIcaBatch Fit pooled-run ICA and apply it to broadband epochs.
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
        fprintf("Preparing ICA for %s (%d/%d)\n", ...
            exports.subject(index), index, height(exports));
        outputs(index) = prepareSubjectIca( ...
            config, exports.subject(index), exports.path(index), options.Overwrite);
        fprintf("Completed ICA for %s\n", exports.subject(index));
    end
end

function outputPath = prepareSubjectIca(config, subject, exportPath, overwrite)
    task = string(config.study.task);
    outputDirectory = fullfile(string(config.paths.output), "ica", subject, "eeg");
    outputPath = fullfile(outputDirectory, ...
        subject + "_task-" + task + "_desc-fieldtripica_components.mat");
    provenancePath = fullfile(outputDirectory, ...
        subject + "_task-" + task + "_desc-fieldtripica_provenance.json");
    if (isfile(outputPath) || isfile(provenancePath)) && ~overwrite
        error("fieldtripTfr:ExistingIca", ...
            "ICA output already exists for %s. Pass Overwrite=true to replace it.", subject);
    end

    package = loadExportPackage(exportPath);
    icaChannels = cellstr(string(package.metadata.ica_channels));
    if numel(icaChannels) < 2
        error("fieldtripTfr:InvalidChannels", ...
            "%s has fewer than two valid ICA channels.", subject);
    end

    selectConfig = [];
    selectConfig.channel = icaChannels;
    icaData = ft_selectdata(selectConfig, package.ica_data);
    broadbandData = ft_selectdata(selectConfig, package.broadband_data);
    dataRank = computeDataRank(icaData);

    rng(double(config.ica.random_seed), "twister");
    analysisConfig = [];
    analysisConfig.method = char(string(config.ica.method));
    analysisConfig.demean = "no";
    analysisConfig.numcomponent = dataRank;
    analysisConfig.runica.extended = double(logical(config.ica.extended));
    analysisConfig.runica.maxsteps = double(config.ica.max_iterations);
    componentFit = ft_componentanalysis(analysisConfig, icaData);

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
        "exportId", "dataRank", "-v7.3");

    provenance = struct( ...
        subject=subject, ...
        task=task, ...
        export_id=exportId, ...
        export_path=exportPath, ...
        component_count=numel(componentBroadband.label), ...
        data_rank=dataRank, ...
        ica_channels={icaChannels}, ...
        created_at=string(datetime("now", TimeZone="UTC")));
    writeJson(provenancePath, provenance);
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
