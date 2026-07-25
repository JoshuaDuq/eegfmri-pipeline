function runGradientTroughIca(fieldTripRoot, runtimePath)
%RUNGRADIENTTROUGHICA Fit ICA, transfer weights, compute TFRs, and plot.

arguments
    fieldTripRoot (1, 1) string
    runtimePath (1, 1) string
end

initializeGradientTroughFieldTrip(fieldTripRoot);
runtime = loadGradientTroughRuntime(runtimePath);
participants = string(runtime.participants(:));

for participantIndex = 1:numel(participants)
    participant = participants(participantIndex);
    exportField = matlab.lang.makeValidName(participant);
    exportPath = string(runtime.exports.(exportField));
    derivativeDirectory = fullfile(string(runtime.output_root), "derivatives", participant);
    figureDirectory = fullfile(string(runtime.output_root), "figures");
    componentPath = fitGradientTroughIca(exportPath, derivativeDirectory, runtime.ica);
    tfrPath = computeGradientTroughTfr(componentPath, derivativeDirectory, runtime.tfr);
    plotGradientTroughResults(componentPath, tfrPath, figureDirectory);
end
end
