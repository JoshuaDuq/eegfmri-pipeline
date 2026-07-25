function dataRank = computeGradientTroughDataRank(data)
%COMPUTEGRADIENTTROUGHDATARANK Compute numerical rank across ICA trials.

arguments
    data (1, 1) struct
end

if ~isfield(data, "trial") || isempty(data.trial)
    error("gradientTrough:InvalidIcaData", "ICA data contains no trials.");
end

concatenated = double(cat(2, data.trial{:}));
if any(~isfinite(concatenated), "all")
    error("gradientTrough:InvalidIcaData", "ICA data contains non-finite values.");
end

singularValues = svd(concatenated, "econ");
tolerance = max(size(concatenated)) * eps(singularValues(1));
dataRank = sum(singularValues > tolerance);
if dataRank < 2
    error("gradientTrough:RankDeficient", ...
        "ICA data rank is %d; at least two dimensions are required.", dataRank);
end
end

