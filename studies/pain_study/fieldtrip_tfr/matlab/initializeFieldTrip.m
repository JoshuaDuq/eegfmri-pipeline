function initializeFieldTrip(config)
% initializeFieldTrip Add the configured FieldTrip root and initialize it.
    arguments
        config (1, 1) struct
    end

    fieldtripPath = string(config.paths.fieldtrip);
    addpath(fieldtripPath);
    ft_defaults();
    bundledEeglabPath = fullfile(fieldtripPath, "external", "eeglab");
    addpath(bundledEeglabPath, "-begin");

    requiredFunctions = [
        "ft_componentanalysis"
        "ft_databrowser"
        "ft_freqanalysis"
        "ft_freqbaseline"
        "ft_rejectcomponent"
        "runica"
    ];
    for functionName = requiredFunctions'
        if isempty(which(functionName))
            error("fieldtripTfr:MissingFunction", ...
                "Required FieldTrip/EEGLAB function is unavailable: %s", functionName);
        end
    end

    expectedRunica = fullfile(bundledEeglabPath, "runica.m");
    if ~strcmp(which("runica"), expectedRunica)
        error("fieldtripTfr:WrongRunica", ...
            "FieldTrip's bundled runica is shadowed by: %s", which("runica"));
    end
end
