function runtime = loadGradientTroughRuntime(runtimePath)
%LOADGRADIENTTROUGHRUNTIME Load and validate the Python export manifest.

arguments
    runtimePath (1, 1) string
end

if ~isfile(runtimePath)
    error("gradientTrough:MissingRuntime", ...
        "Runtime manifest does not exist: %s", runtimePath);
end

runtime = jsondecode(fileread(runtimePath));
requiredFields = ["participants", "exports", "output_root", "ica", "tfr"];
for fieldName = requiredFields
    if ~isfield(runtime, fieldName)
        error("gradientTrough:InvalidRuntime", ...
            "Runtime manifest lacks field '%s'.", fieldName);
    end
end
end

