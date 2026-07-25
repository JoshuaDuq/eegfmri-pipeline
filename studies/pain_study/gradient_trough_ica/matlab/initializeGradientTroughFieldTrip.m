function initializeGradientTroughFieldTrip(fieldTripRoot)
%INITIALIZEGRADIENTTROUGHFIELDTRIP Add one explicit FieldTrip installation.

arguments
    fieldTripRoot (1, 1) string
end

if ~isfolder(fieldTripRoot)
    error("gradientTrough:MissingFieldTrip", ...
        "FieldTrip directory does not exist: %s", fieldTripRoot);
end

addpath(fieldTripRoot);
if exist("ft_defaults", "file") ~= 2
    error("gradientTrough:InvalidFieldTrip", ...
        "ft_defaults.m was not found in: %s", fieldTripRoot);
end
ft_defaults;

fieldTripEeglabRoot = fullfile(fieldTripRoot, "external", "eeglab");
if ~isfile(fullfile(fieldTripEeglabRoot, "runica.m"))
    error("gradientTrough:MissingRunica", ...
        "FieldTrip's bundled runica.m does not exist: %s", fieldTripEeglabRoot);
end
addpath(fieldTripEeglabRoot, "-begin");
clear runica;
rehash path;

resolvedRunica = string(which("runica"));
expectedRunica = string(fullfile(fieldTripEeglabRoot, "runica.m"));
if resolvedRunica ~= expectedRunica
    error("gradientTrough:RunicaPathConflict", ...
        "Expected runica at %s, but MATLAB resolved %s.", ...
        expectedRunica, resolvedRunica);
end
end
