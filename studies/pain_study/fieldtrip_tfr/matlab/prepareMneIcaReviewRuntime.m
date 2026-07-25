function runtimePath = prepareMneIcaReviewRuntime(configPath, options)
% prepareMneIcaReviewRuntime Export MNE components and resolve MATLAB config.
    arguments
        configPath (1, 1) string
        options.ExportComponents (1, 1) logical = true
    end

    if ~isfile(configPath)
        error("mneIcaReview:MissingConfig", ...
            "Review configuration does not exist: %s", configPath);
    end

    matlabDirectory = fileparts(mfilename("fullpath"));
    workflowDirectory = fileparts(matlabDirectory);
    projectRoot = fileparts(fileparts(fileparts(workflowDirectory)));
    runtimeOnly = "";
    if ~options.ExportComponents
        runtimeOnly = " --runtime-only";
    end
    command = sprintf( ...
        'cd "%s" && python3 -m studies.pain_study.fieldtrip_tfr.export_mne_ica_review --config "%s"%s', ...
        projectRoot, configPath, runtimeOnly);
    [status, output] = system(command, "-echo");
    if status ~= 0
        error("mneIcaReview:ExportFailed", ...
            "MNE ICA export failed with status %d.", status);
    end

    match = regexp(output, "RUNTIME_CONFIG=([^\r\n]+)", "tokens", "once");
    if isempty(match)
        error("mneIcaReview:MissingRuntime", ...
            "The exporter did not report its runtime configuration path.");
    end
    runtimePath = string(strtrim(match{1}));
    if ~isfile(runtimePath)
        error("mneIcaReview:MissingRuntime", ...
            "Runtime configuration does not exist: %s", runtimePath);
    end
end
