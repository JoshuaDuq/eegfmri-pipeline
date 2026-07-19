function package = loadExportPackage(exportPath)
% loadExportPackage Load and validate one Python-generated FieldTrip export.
    arguments
        exportPath (1, 1) string
    end

    if ~isfile(exportPath)
        error("fieldtripTfr:MissingExport", ...
            "Participant export does not exist: %s", exportPath);
    end
    package = load(exportPath);
    requiredVariables = ["broadband_data", "ica_data", "metadata"];
    for variableName = requiredVariables
        if ~isfield(package, variableName)
            error("fieldtripTfr:InvalidExport", ...
                "Export %s lacks variable '%s'.", exportPath, variableName);
        end
    end

    metadataFields = [
        "subject"
        "export_id"
        "trialinfo"
        "trialinfo_labels"
        "bad_channels"
        "ica_channels"
    ];
    for fieldName = metadataFields'
        if ~isfield(package.metadata, fieldName)
            error("fieldtripTfr:InvalidExport", ...
                "Export %s metadata lack field '%s'.", exportPath, fieldName);
        end
    end

    trialCount = numel(package.broadband_data.trial);
    if trialCount ~= numel(package.ica_data.trial)
        error("fieldtripTfr:InvalidExport", ...
            "Broadband and ICA-fit trial counts differ in %s.", exportPath);
    end
    if size(package.metadata.trialinfo, 1) ~= trialCount
        error("fieldtripTfr:InvalidExport", ...
            "Trial metadata count does not match EEG trial count in %s.", exportPath);
    end
    if ~isequal(package.broadband_data.label, package.ica_data.label)
        error("fieldtripTfr:InvalidExport", ...
            "Broadband and ICA-fit channel orders differ in %s.", exportPath);
    end

    package.broadband_data.trialinfo = package.metadata.trialinfo;
    package.ica_data.trialinfo = package.metadata.trialinfo;
end
