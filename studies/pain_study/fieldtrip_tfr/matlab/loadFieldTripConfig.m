function config = loadFieldTripConfig(configPath)
% loadFieldTripConfig Load and validate the resolved JSON configuration.
    arguments
        configPath (1, 1) string
    end

    if ~isfile(configPath)
        error("fieldtripTfr:MissingConfig", ...
            "Runtime configuration does not exist: %s", configPath);
    end

    config = jsondecode(fileread(configPath));
    requiredSections = ["paths", "study", "epochs", "ica", "tfr"];
    for section = requiredSections
        if ~isfield(config, section)
            error("fieldtripTfr:InvalidConfig", ...
                "Runtime configuration lacks the '%s' section.", section);
        end
    end

    requiredPaths = ["fieldtrip", "bids_eeg", "mne_derivatives", "output"];
    for pathName = requiredPaths
        pathValue = string(config.paths.(pathName));
        if strlength(pathValue) == 0
            error("fieldtripTfr:InvalidConfig", ...
                "Configuration path '%s' must not be empty.", pathName);
        end
    end

    fieldtripPath = string(config.paths.fieldtrip);
    if ~isfile(fullfile(fieldtripPath, "ft_defaults.m"))
        error("fieldtripTfr:MissingFieldTrip", ...
            "FieldTrip ft_defaults.m does not exist under %s.", fieldtripPath);
    end
end
