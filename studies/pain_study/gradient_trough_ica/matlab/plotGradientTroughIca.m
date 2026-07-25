function plotGradientTroughIca(fieldTripRoot, runtimePath)
%PLOTGRADIENTTROUGHICA Recreate MATLAB figures from completed results.

arguments
    fieldTripRoot (1, 1) string
    runtimePath (1, 1) string
end

initializeGradientTroughFieldTrip(fieldTripRoot);
runtime = loadGradientTroughRuntime(runtimePath);
participants = string(runtime.participants(:));
for participantIndex = 1:numel(participants)
    participant = participants(participantIndex);
    derivativeDirectory = fullfile(string(runtime.output_root), "derivatives", participant);
    componentPath = fullfile(derivativeDirectory, ...
        participant + "_desc-troughica_components.mat");
    tfrPath = fullfile(derivativeDirectory, participant + "_desc-troughica_tfr.mat");
    plotGradientTroughResults(componentPath, tfrPath, ...
        fullfile(string(runtime.output_root), "figures"));
end
end
