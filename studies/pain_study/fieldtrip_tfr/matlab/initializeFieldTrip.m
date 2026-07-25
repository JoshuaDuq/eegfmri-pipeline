function initializeFieldTrip(config)
% initializeFieldTrip Add and initialize the configured FieldTrip release.
    fieldtripPath = string(config.paths.fieldtrip);
    if ~isfile(fullfile(fieldtripPath, "ft_defaults.m"))
        error("mneIcaReview:MissingFieldTrip", ...
            "FieldTrip does not exist at: %s", fieldtripPath);
    end
    addpath(fieldtripPath);
    ft_defaults;
end
