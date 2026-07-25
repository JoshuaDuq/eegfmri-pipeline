function plotMneIcaReviewBatch(runtimePath)
% plotMneIcaReviewBatch Plot all exact MNE components and configured TFRs.
    config = loadMneIcaReviewConfig(runtimePath);
    initializeFieldTrip(config);
    for subject = config.resolved_participants'
        plotSubject(config, subject);
    end
end

function plotSubject(config, subject)
    tfrPath = fullfile( ...
        string(config.paths.output), "pow", subject + "_MNE_ICA_TFR.mat");
    if ~isfile(tfrPath)
        error("mneIcaReview:MissingTfr", ...
            "Component TFR output does not exist for %s: %s", subject, tfrPath);
    end
    review = load(tfrPath, "componentOverview", "conditionResults");
    if ~isfield(review, "componentOverview") || ~isfield(review, "conditionResults")
        error("mneIcaReview:InvalidTfr", ...
            "Review output lacks componentOverview or conditionResults: %s", tfrPath);
    end

    component = review.componentOverview;
    proposedBad = double(component.mne_proposed_bad_indices(:));
    componentCount = numel(component.label);
    groups = componentGroups(componentCount);
    plotTopographies(config, subject, component, proposedBad, groups);
    for conditionIndex = 1:numel(review.conditionResults)
        plotCondition( ...
            config, subject, proposedBad, groups, ...
            review.conditionResults(conditionIndex));
    end
end

function groups = componentGroups(componentCount)
    if componentCount < 1
        error("mneIcaReview:NoComponents", "The MNE ICA contains no components.");
    end
    groups = cell(ceil(componentCount / 16), 1);
    for index = 1:numel(groups)
        first = (index - 1) * 16 + 1;
        groups{index} = first:min(first + 15, componentCount);
    end
end

function plotTopographies(config, subject, component, proposedBad, groups)
    layoutConfig = [];
    layoutConfig.elec = component.elec;
    componentLayout = ft_prepare_layout(layoutConfig);
    for groupIndex = 1:numel(groups)
        components = groups{groupIndex};
        componentFigure = figure( ...
            "Name", sprintf("%s MNE ICA topographies — components %d-%d", ...
                subject, components(1), components(end)), ...
            "Color", "w", ...
            "Units", "normalized", ...
            "OuterPosition", [0, 0, 1, 1]);
        cfg = [];
        cfg.component = components;
        cfg.layout = componentLayout;
        cfg.rows = 4;
        cfg.columns = 4;
        cfg.colormap = turbo(256);
        cfg.comment = "no";
        cfg.colorbar = "no";
        ft_topoplotIC(cfg, component);
        markTopographyTitles( ...
            componentFigure, proposedBad, logical(config.plots.mark_mne_proposed_bad));
        annotation(componentFigure, "textbox", [0.01, 0.955, 0.4, 0.035], ...
            "String", "Red title = automatically proposed as bad by MNE-ICALabel", ...
            "Color", [0.75, 0, 0], "EdgeColor", "none", "FontWeight", "bold");
    end
end

function markTopographyTitles(componentFigure, proposedBad, markProposedBad)
    titleHandles = findall(componentFigure, "Type", "text");
    for handle = reshape(titleHandles, 1, [])
        label = string(handle.String);
        match = regexp(label, "component\s+(\d+)$", "tokens", "once");
        if isempty(match)
            continue;
        end
        componentNumber = str2double(match{1});
        isProposedBad = markProposedBad && ismember(componentNumber, proposedBad);
        handle.String = componentTitle(componentNumber, isProposedBad);
        if isProposedBad
            handle.Color = [0.75, 0, 0];
            handle.FontWeight = "bold";
        end
    end
end

function plotCondition(config, subject, proposedBad, groups, condition)
    frequencyView = string(config.plots.frequency_view);
    frequencyRange = double(config.plots.bands_hz.(frequencyView));
    power = condition.power;
    frequencyMask = power.freq >= frequencyRange(1) & power.freq <= frequencyRange(2);
    if ~any(frequencyMask)
        error("mneIcaReview:MissingBand", ...
            "No TFR frequencies exist inside the %s range.", frequencyView);
    end
    bandPower = power.powspctrm(:, frequencyMask, :);
    colorLimit = sharedColorLimit(bandPower);

    for groupIndex = 1:numel(groups)
        components = groups{groupIndex};
        componentFigure = figure( ...
            "Name", sprintf("%s %s %s — components %d-%d", ...
                subject, condition.name, frequencyView, ...
                components(1), components(end)), ...
            "Color", "w", ...
            "Units", "normalized", ...
            "OuterPosition", [0, 0, 1, 1]);

        for subplotIndex = 1:numel(components)
            componentNumber = components(subplotIndex);
            axisHandle = subplot(4, 4, subplotIndex, "Parent", componentFigure);
            componentPower = squeeze(bandPower(componentNumber, :, :));
            imagesc(axisHandle, power.time, power.freq(frequencyMask), ...
                componentPower, [-colorLimit, colorLimit]);
            axis(axisHandle, "xy");
            xline(axisHandle, 0, "--k", LineWidth=0.75);
            colorbar(axisHandle);
            isProposedBad = logical(config.plots.mark_mne_proposed_bad) && ...
                ismember(componentNumber, proposedBad);
            title(axisHandle, componentTitle(componentNumber, isProposedBad), ...
                "Color", titleColor(isProposedBad), ...
                "FontWeight", titleWeight(isProposedBad));
            if isProposedBad
                axisHandle.XColor = [0.75, 0, 0];
                axisHandle.YColor = [0.75, 0, 0];
                axisHandle.LineWidth = 1.5;
            end
        end
        colormap(componentFigure, turbo(256));
        sgtitle(componentFigure, sprintf( ...
            "%s — %s — %s (%s) — red = MNE proposed bad", ...
            subject, condition.title, frequencyView, condition.units), ...
            "Interpreter", "none");
    end
end

function limit = sharedColorLimit(power)
    values = abs(power(isfinite(power)));
    if isempty(values)
        error("mneIcaReview:InvalidPower", ...
            "The selected component TFR contains no finite values.");
    end
    limit = prctile(values, 98);
    if limit <= 0
        error("mneIcaReview:InvalidPower", ...
            "The selected component TFR contains no non-zero values.");
    end
end

function label = componentTitle(componentNumber, proposedBad)
    label = string(sprintf("IC%03d", componentNumber));
    if proposedBad
        label = label + " — MNE PROPOSED BAD";
    end
end

function color = titleColor(proposedBad)
    if proposedBad
        color = [0.75, 0, 0];
    else
        color = [0, 0, 0];
    end
end

function weight = titleWeight(proposedBad)
    if proposedBad
        weight = "bold";
    else
        weight = "normal";
    end
end
