function config = loadMneIcaReviewConfig(runtimePath)
% loadMneIcaReviewConfig Load and validate resolved review configuration.
    arguments
        runtimePath (1, 1) string
    end

    if ~isfile(runtimePath)
        error("mneIcaReview:MissingRuntime", ...
            "Runtime configuration does not exist: %s", runtimePath);
    end
    config = jsondecode(fileread(runtimePath));
    requiredSections = ["paths", "study", "epochs", "tfr", "plots", "execution"];
    for section = requiredSections
        if ~isfield(config, section)
            error("mneIcaReview:InvalidConfig", ...
                "Runtime configuration lacks '%s'.", section);
        end
    end
    if ~isfield(config, "conditions") || isempty(config.conditions)
        error("mneIcaReview:InvalidConfig", ...
            "At least one review condition is required.");
    end
    if ~isfield(config, "resolved_participants") || ...
            isempty(config.resolved_participants)
        error("mneIcaReview:InvalidConfig", ...
            "No resolved participants exist in the runtime configuration.");
    end

    config.conditions = string(config.conditions(:));
    config.resolved_participants = string(config.resolved_participants(:));
    supportedConditions = [
        "high_vs_low"
        "high_temperature"
        "low_temperature"
        "painful_vs_nonpainful"
        "painful"
        "nonpainful"
        "individual_temperatures"
        "temperature_slope"
        "grand_average"
    ];
    unsupported = setdiff(config.conditions, supportedConditions);
    if ~isempty(unsupported)
        error("mneIcaReview:InvalidCondition", ...
            "Unsupported review conditions: %s", strjoin(unsupported, ", "));
    end

    frequencyView = string(config.plots.frequency_view);
    if ~ismember(frequencyView, ["alpha", "beta", "gamma", "full"])
        error("mneIcaReview:InvalidBand", ...
            "plots.frequency_view must be alpha, beta, gamma, or full.");
    end
    if double(config.plots.components_per_figure) ~= 16
        error("mneIcaReview:InvalidLayout", ...
            "This review uses exactly 16 components per figure.");
    end
    if ~isfield(config.plots, "mark_mne_proposed_bad") || ...
            ~isscalar(config.plots.mark_mne_proposed_bad)
        error("mneIcaReview:InvalidConfig", ...
            "plots.mark_mne_proposed_bad must be true or false.");
    end
end
