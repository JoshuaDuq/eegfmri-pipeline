clear;
clc;

fieldtripPath = "/Users/joduq24/Documents/MATLAB/fieldtrip-master";
outputRoot = ...
    "/Volumes/KINGSTON/EEG_fMRI_data/derivatives/fieldtrip_brainvision_analyzer_sub-0015";
subject = "sub-0015";
bandName = "alpha";  % "alpha", "beta", or "gamma"
tfrVariables = [
    "pow_temp_44p3"
    "pow_temp_44p3_db"
    "pow_temp_45p3"
    "pow_temp_45p3_db"
    "pow_temp_46p3"
    "pow_temp_46p3_db"
    "pow_temp_47p3"
    "pow_temp_47p3_db"
    "pow_temp_48p3"
    "pow_temp_48p3_db"
    "pow_temp_49p3"
    "pow_temp_49p3_db"
    "pow_painful"
    "pow_painful_db"
    "pow_nonpainful"
    "pow_nonpainful_db"
    "pow_painful_v_nonpainful"
    "pow_low_temperature"
    "pow_high_temperature"
    "pow_high_v_low"
    "pow_temperature_slope"
    "pow_avg"
];

if ~isfile(fullfile(fieldtripPath, "ft_defaults.m"))
    error("fieldtripTfr:MissingFieldTrip", ...
        "FieldTrip does not exist at: %s", fieldtripPath);
end
if ~ismember(bandName, ["alpha", "beta", "gamma"])
    error("fieldtripTfr:InvalidBand", ...
        "bandName must be alpha, beta, or gamma.");
end

addpath(fieldtripPath);
ft_defaults;

tfrFile = fullfile( ...
    outputRoot, "pow", bandName, subject + "_ICA_pow_EEG.mat");
if ~isfile(tfrFile)
    error("fieldtripTfr:MissingTfr", ...
        "TFR output does not exist: %s", tfrFile);
end
icaFile = fullfile( ...
    outputRoot, "ica", bandName, subject, "eeg", ...
    subject + "_task-thermalactive_desc-fieldtripica_components.mat");
if ~isfile(icaFile)
    error("fieldtripTfr:MissingIca", ...
        "ICA output does not exist: %s", icaFile);
end

results = load(tfrFile);
if ~isfield(results, "comp")
    error("fieldtripTfr:InvalidTfr", ...
        "TFR output does not contain comp: %s", tfrFile);
end
decomposition = load(icaFile, "componentFit");
if ~isfield(decomposition, "componentFit")
    error("fieldtripTfr:InvalidIca", ...
        "ICA output does not contain componentFit: %s", icaFile);
end
componentFit = decomposition.componentFit;
requiredIcaFields = ["label", "topo", "topolabel", "elec"];
for fieldName = requiredIcaFields
    if ~isfield(componentFit, fieldName)
        error("fieldtripTfr:InvalidIca", ...
            "componentFit does not contain %s.", fieldName);
    end
end
if size(componentFit.topo, 1) ~= numel(componentFit.topolabel) || ...
        size(componentFit.topo, 2) ~= numel(componentFit.label)
    error("fieldtripTfr:InvalidIca", ...
        "ICA topography dimensions do not match its labels.");
end
if any(~isfinite(componentFit.topo), "all")
    error("fieldtripTfr:InvalidIca", ...
        "ICA topographies contain non-finite values.");
end
if ~isequal(componentFit.unmixing, results.comp.unmixing) || ...
        ~isequal(string(componentFit.topolabel(:)), ...
            string(results.comp.topolabel(:)))
    error("fieldtripTfr:ComponentMismatch", ...
        "The ICA maps and broadband component data are not the same decomposition.");
end

layoutConfig = [];
layoutConfig.elec = componentFit.elec;
componentLayout = ft_prepare_layout(layoutConfig);

for tfrVariable = tfrVariables'
    if ~isfield(results, tfrVariable)
        error("fieldtripTfr:MissingVariable", ...
            "TFR output does not contain %s.", tfrVariable);
    end
end

componentsPerFigure = 16;
figureCount = 4;
for tfrVariable = tfrVariables'
    power = results.(tfrVariable);
    componentCount = numel(power.label);
    if ~isequal(string(power.label(:)), string(results.comp.label(:))) || ...
            numel(power.label) ~= numel(componentFit.label)
        error("fieldtripTfr:ComponentMismatch", ...
            "%s component labels do not match the %s ICA model.", ...
            tfrVariable, bandName);
    end
    if componentCount > 64
        error("fieldtripTfr:TooManyComponents", ...
            "Four 4-by-4 figures can display at most 64 components, found %d.", ...
            componentCount);
    end

    powerMinimum = min(power.powspctrm, [], "all");
    powerMaximum = max(power.powspctrm, [], "all");
    if powerMinimum < 0 && powerMaximum > 0
        absoluteMaximum = max(abs([powerMinimum, powerMaximum]));
        colorLimits = [-absoluteMaximum, absoluteMaximum];
    else
        colorLimits = [powerMinimum, powerMaximum];
    end
    if colorLimits(1) == colorLimits(2)
        error("fieldtripTfr:ConstantPower", ...
            "%s contains identical power values.", tfrVariable);
    end

    for figureNumber = 1:figureCount
        firstComponent = (figureNumber - 1)*componentsPerFigure + 1;
        lastComponent = min(figureNumber*componentsPerFigure, componentCount);

        componentFigure = figure( ...
            "Name", sprintf("%s %s %s — components %d-%d", ...
                subject, bandName, tfrVariable, firstComponent, lastComponent), ...
            "Color", "w", ...
            "Units", "normalized", ...
            "OuterPosition", [0, 0, 1, 1]);

        for componentNumber = firstComponent:lastComponent
            subplotIndex = componentNumber - firstComponent + 1;
            tfrAxes = subplot( ...
                4, 4, subplotIndex, "Parent", componentFigure);

            componentLabel = string(power.label{componentNumber});
            cfg = [];
            cfg.channel = char(componentLabel);
            cfg.parameter = "powspctrm";
            cfg.xlim = [-5, 14.5];
            cfg.ylim = [1, 100];
            cfg.zlim = colorLimits;
            cfg.colorbar = "no";
            cfg.figure = tfrAxes;
            ft_singleplotTFR(cfg, power);

            title(componentLabel, Interpreter="none", FontSize=9);
            xline(0, "--w", LineWidth=0.75);

            tfrPosition = tfrAxes.Position;
            insetWidth = 0.32*tfrPosition(3);
            insetHeight = 0.32*tfrPosition(4);
            insetPosition = [
                tfrPosition(1) + 0.66*tfrPosition(3), ...
                tfrPosition(2) + 0.64*tfrPosition(4), ...
                insetWidth, ...
                insetHeight
            ];
            topographyAxes = axes( ...
                "Parent", componentFigure, ...
                "Position", insetPosition);

            topographyConfig = [];
            topographyConfig.component = componentNumber;
            topographyConfig.layout = componentLayout;
            topographyConfig.figure = topographyAxes;
            topographyConfig.zlim = "maxabs";
            topographyConfig.marker = "off";
            topographyConfig.comment = "no";
            topographyConfig.title = "off";
            topographyConfig.colorbar = "no";
            ft_topoplotIC(topographyConfig, componentFit);
        end

        sgtitle(sprintf("%s — %s ICA — %s — components %d-%d", ...
            subject, bandName, tfrVariable, firstComponent, lastComponent), ...
            Interpreter="none");
    end
end
