function limits = tfrPlotColorLimits(values, signed, variableName)
maximum = max(values, [], "all");
minimum = min(values, [], "all");

if signed
    absoluteMaximum = max(abs([minimum, maximum]));
    if absoluteMaximum == 0
        error("fieldtripTfr:ConstantPower", ...
            "%s contains only zero values.", variableName);
    end
    limits = [-absoluteMaximum, absoluteMaximum];
    return;
end

if minimum < 0
    error("fieldtripTfr:InvalidPower", ...
        "%s is declared nonnegative but contains negative values.", variableName);
end
if maximum == 0
    error("fieldtripTfr:ConstantPower", ...
        "%s contains only zero values.", variableName);
end
limits = [0, maximum];
end
