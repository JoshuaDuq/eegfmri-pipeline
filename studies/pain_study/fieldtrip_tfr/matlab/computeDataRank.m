function dataRank = computeDataRank(data)
% computeDataRank Compute the numerical rank from pooled trial covariance.
    arguments
        data (1, 1) struct
    end

    channelCount = numel(data.label);
    sampleCount = sum(cellfun(@(trial) size(trial, 2), data.trial));
    if sampleCount <= channelCount
        error("fieldtripTfr:InsufficientSamples", ...
            "ICA data need more samples than channels.");
    end

    sampleSum = zeros(channelCount, 1);
    crossProduct = zeros(channelCount, channelCount);
    for trialIndex = 1:numel(data.trial)
        trial = double(data.trial{trialIndex});
        if any(~isfinite(trial), "all")
            error("fieldtripTfr:NonfiniteData", ...
                "ICA trial %d contains non-finite samples.", trialIndex);
        end
        sampleSum = sampleSum + sum(trial, 2);
        crossProduct = crossProduct + trial*trial';
    end

    covariance = (crossProduct - (sampleSum*sampleSum')/sampleCount) ...
        / (sampleCount - 1);
    singularValues = svd(covariance);
    tolerance = max(size(covariance))*eps(max(singularValues));
    dataRank = sum(singularValues > tolerance);
    if dataRank < 2
        error("fieldtripTfr:InvalidRank", ...
            "ICA data rank is %d; at least two dimensions are required.", dataRank);
    end
end
