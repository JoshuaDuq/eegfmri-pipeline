function plotGradientTroughResults(componentPath, tfrPath, outputDirectory)
%PLOTGRADIENTTROUGHRESULTS Create all ICA/TFR figures in MATLAB.

arguments
    componentPath (1, 1) string
    tfrPath (1, 1) string
    outputDirectory (1, 1) string
end

components = load(componentPath, ...
    "componentFit", "componentBroadband", "gradientTrough");
tfr = load(tfrPath, "lowDb", "highDb", "contrastDb", ...
    "lowTrialIndices", "highTrialIndices");
participant = string(components.gradientTrough.participant);
figureDirectory = fullfile(outputDirectory, participant);
if ~isfolder(figureDirectory)
    mkdir(figureDirectory);
end

plotSelectionQc(components.gradientTrough.selection_qc, participant, figureDirectory);
plotComponentTopographies(components.componentFit, participant, figureDirectory);
plotComponentTimecourses(components.componentBroadband, ...
    tfr.lowTrialIndices, tfr.highTrialIndices, participant, figureDirectory);
conditionColorLimit = max(abs([tfr.lowDb.powspctrm(:); tfr.highDb.powspctrm(:)]));
contrastColorLimit = max(abs(tfr.contrastDb.powspctrm), [], "all");
plotTfrPages(tfr.lowDb, participant, "low", conditionColorLimit, figureDirectory);
plotTfrPages(tfr.highDb, participant, "high", conditionColorLimit, figureDirectory);
plotTfrPages(tfr.contrastDb, participant, "high-minus-low", ...
    contrastColorLimit, figureDirectory);
end


function plotSelectionQc(selectionQc, participant, outputDirectory)
figureHandle = figure("Color", "w", "Position", [100, 100, 1400, 700]);
layout = tiledlayout(2, 3, "TileSpacing", "compact", "Padding", "compact");
title(layout, participant + " volume-locked ECG plateau selections");
for runIndex = 1:numel(selectionQc.run_id)
    nexttile;
    times = selectionQc.times_ms{runIndex};
    trace = selectionQc.mean_rectified_ecg_uv{runIndex};
    plot(times, trace, "Color", [0.1, 0.3, 0.65], "LineWidth", 0.8);
    hold on;
    limits = ylim;
    starts = selectionQc.plateau_start_ms{runIndex};
    stops = selectionQc.plateau_stop_ms{runIndex};
    for windowIndex = 1:numel(starts)
        patch([starts(windowIndex), stops(windowIndex), stops(windowIndex), starts(windowIndex)], ...
            [limits(1), limits(1), limits(2), limits(2)], [0.9, 0.45, 0.1], ...
            "FaceAlpha", 0.16, "EdgeColor", "none");
    end
    plot(times, trace, "Color", [0.1, 0.3, 0.65], "LineWidth", 0.8);
    troughs = selectionQc.trough_latency_ms{runIndex};
    arrayfun(@(latency) xline(latency, ":", "Color", [0.35, 0.35, 0.35]), troughs);
    xlim([0, 900]);
    xlabel("Time from volume marker (ms)");
    ylabel("Mean rectified ECG (\muV)");
    title("Run " + selectionQc.run_id(runIndex));
    box off;
end
exportFigure(figureHandle, fullfile(outputDirectory, participant + "_selection-qc.png"));
end


function plotComponentTopographies(componentFit, participant, outputDirectory)
componentCount = size(componentFit.topo, 2);
componentsPerPage = 16;
layoutCfg = [];
layoutCfg.elec = componentFit.elec;
sensorLayout = ft_prepare_layout(layoutCfg);
for pageStart = 1:componentsPerPage:componentCount
    pageStop = min(pageStart + componentsPerPage - 1, componentCount);
    figureHandle = figure("Color", "w", "Position", [100, 100, 1200, 900]);
    cfg = [];
    cfg.component = pageStart:pageStop;
    cfg.layout = sensorLayout;
    cfg.comment = 'no';
    ft_topoplotIC(cfg, componentFit);
    sgtitle(participant + " ICA topographies " + pageStart + "-" + pageStop);
    exportFigure(figureHandle, fullfile(outputDirectory, ...
        participant + "_topographies_" + pageStart + "-" + pageStop + ".png"));
end
end


function plotComponentTimecourses(componentData, lowTrialIndices, highTrialIndices, ...
    participant, outputDirectory)
allTrials = cat(3, componentData.trial{:});
lowMean = mean(allTrials(:, :, lowTrialIndices), 3);
highMean = mean(allTrials(:, :, highTrialIndices), 3);
times = componentData.time{1};
componentCount = size(allTrials, 1);
componentsPerPage = 16;
for pageStart = 1:componentsPerPage:componentCount
    pageStop = min(pageStart + componentsPerPage - 1, componentCount);
    figureHandle = figure("Color", "w", "Position", [100, 100, 1400, 900]);
    layout = tiledlayout(4, 4, "TileSpacing", "compact", "Padding", "compact");
    title(layout, participant + " component time courses");
    for component = pageStart:pageStop
        nexttile;
        plot(times, lowMean(component, :), "Color", [0.1, 0.35, 0.75]);
        hold on;
        plot(times, highMean(component, :), "Color", [0.8, 0.2, 0.1]);
        xline(0, ":k");
        xlim([-5, 14.5]);
        title("IC " + component);
        if component == pageStart
            legend("Low", "High", "Location", "best");
        end
    end
    exportFigure(figureHandle, fullfile(outputDirectory, ...
        participant + "_timecourses_" + pageStart + "-" + pageStop + ".png"));
end
end


function plotTfrPages(frequencyData, participant, conditionName, colorLimit, outputDirectory)
componentCount = size(frequencyData.powspctrm, 1);
componentsPerPage = 16;
if ~isfinite(colorLimit) || colorLimit == 0
    error("gradientTrough:InvalidTfr", "TFR has no finite non-zero power values.");
end
for pageStart = 1:componentsPerPage:componentCount
    pageStop = min(pageStart + componentsPerPage - 1, componentCount);
    figureHandle = figure("Color", "w", "Position", [100, 100, 1400, 900]);
    layout = tiledlayout(4, 4, "TileSpacing", "compact", "Padding", "compact");
    title(layout, participant + " " + conditionName + " component TFR (dB)");
    for component = pageStart:pageStop
        nexttile;
        imagesc(frequencyData.time, frequencyData.freq, ...
            squeeze(frequencyData.powspctrm(component, :, :)));
        axis xy;
        xline(0, ":k");
        xlim([-5, 14.4]);
        ylim([frequencyData.freq(1), frequencyData.freq(end)]);
        clim([-colorLimit, colorLimit]);
        title("IC " + component);
    end
    colorbar;
    exportFigure(figureHandle, fullfile(outputDirectory, ...
        participant + "_tfr-" + conditionName + "_" + pageStart + "-" + pageStop + ".png"));
end
end


function exportFigure(figureHandle, outputPath)
if isfile(outputPath)
    error("gradientTrough:ExistingFigure", "Refusing to overwrite: %s", outputPath);
end
exportgraphics(figureHandle, outputPath, "Resolution", 300);
close(figureHandle);
end
