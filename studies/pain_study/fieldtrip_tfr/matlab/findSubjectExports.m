function exports = findSubjectExports(config, requestedSubjects)
% findSubjectExports Discover one validated export path per participant.
    arguments
        config (1, 1) struct
        requestedSubjects string = strings(0, 1)
    end

    exportRoot = fullfile(string(config.paths.output), "exports");
    if ~isfolder(exportRoot)
        error("fieldtripTfr:MissingExports", ...
            "Export directory does not exist: %s", exportRoot);
    end

    subjectDirectories = dir(fullfile(exportRoot, "sub-*"));
    subjectDirectories = subjectDirectories([subjectDirectories.isdir]);
    availableSubjects = string({subjectDirectories.name})';
    if isempty(availableSubjects)
        error("fieldtripTfr:MissingExports", ...
            "No participant export directories exist under %s.", exportRoot);
    end

    if isempty(requestedSubjects)
        subjects = sort(availableSubjects);
    else
        subjects = normalizeSubjects(requestedSubjects);
        missingSubjects = setdiff(subjects, availableSubjects);
        if ~isempty(missingSubjects)
            error("fieldtripTfr:MissingExports", ...
                "Requested participant exports are missing: %s", ...
                strjoin(missingSubjects, ", "));
        end
    end

    exportPaths = strings(numel(subjects), 1);
    task = string(config.study.task);
    for index = 1:numel(subjects)
        pattern = fullfile(exportRoot, subjects(index), ...
            subjects(index) + "_task-" + task + "_desc-preica_fieldtrip.mat");
        if ~isfile(pattern)
            error("fieldtripTfr:MissingExport", ...
                "Participant export does not exist: %s", pattern);
        end
        exportPaths(index) = pattern;
    end
    exports = table(subjects, exportPaths, ...
        VariableNames=["subject", "path"]);
end
function subjects = normalizeSubjects(values)
    subjects = strip(string(values(:)));
    if any(strlength(subjects) == 0)
        error("fieldtripTfr:InvalidSubject", ...
            "Participant identifiers must not be empty.");
    end
    for index = 1:numel(subjects)
        if ~startsWith(subjects(index), "sub-")
            subjects(index) = "sub-" + subjects(index);
        end
    end
    if numel(unique(subjects)) ~= numel(subjects)
        error("fieldtripTfr:InvalidSubject", ...
            "Participant identifiers must not be repeated.");
    end
end
