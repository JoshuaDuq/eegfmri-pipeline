package wizard

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	fmriSecondLevelRequiredOutputType = "effect_size"
	fmriSecondLevelRequiredSpace      = "MNI152NLin2009cAsym"
)

type fmriSecondLevelFirstLevelSidecar struct {
	Subject          string         `json:"subject"`
	Task             string         `json:"task"`
	ContrastName     string         `json:"contrast_name"`
	OutputTypeActual string         `json:"output_type_actual"`
	ContrastCfg      map[string]any `json:"contrast_cfg"`
}

func normalizeFmriSecondLevelSubjectLabel(subject string) string {
	clean := strings.TrimSpace(subject)
	if clean == "" {
		return ""
	}
	if strings.HasPrefix(clean, "sub-") {
		return clean
	}
	return "sub-" + clean
}

func (m Model) fmriSecondLevelSelectedSubjects() []string {
	selected := make([]string, 0, len(m.subjects))
	for _, subject := range m.subjects {
		if !m.subjectSelected[subject.ID] {
			continue
		}
		label := normalizeFmriSecondLevelSubjectLabel(subject.ID)
		if label != "" {
			selected = append(selected, label)
		}
	}
	return selected
}

func (m Model) fmriSecondLevelResolvedInputRoot() string {
	if value := strings.TrimSpace(m.fmriSecondLevelInputRoot); value != "" {
		return expandUserPath(value)
	}
	if value := strings.TrimSpace(m.derivRoot); value != "" {
		return expandUserPath(value)
	}
	return ""
}

func (m Model) nextFmriSecondLevelContrastDiscoveryKey() string {
	task := strings.TrimSpace(m.task)
	inputRoot := m.fmriSecondLevelResolvedInputRoot()
	subjects := m.fmriSecondLevelSelectedSubjects()
	if task == "" || inputRoot == "" || len(subjects) == 0 {
		return ""
	}
	return strings.Join(append([]string{task, inputRoot}, subjects...), "|")
}

func (m Model) nextFmriSecondLevelCovariatesDiscoveryKey() string {
	path := strings.TrimSpace(m.fmriSecondLevelCovariatesFile)
	if path == "" {
		return ""
	}
	return expandUserPath(path)
}

func (m Model) currentFmriSecondLevelContrastDiscovery() ([]string, string) {
	if m.fmriSecondLevelContrastDiscoveryKey != m.nextFmriSecondLevelContrastDiscoveryKey() {
		return nil, ""
	}
	return m.fmriSecondLevelDiscoveredContrasts, strings.TrimSpace(m.fmriSecondLevelContrastDiscoveryError)
}

func (m Model) currentFmriSecondLevelCovariatesDiscovery() ([]string, map[string][]string, string) {
	if m.fmriSecondLevelCovariatesDiscoveryKey != m.nextFmriSecondLevelCovariatesDiscoveryKey() {
		return nil, nil, ""
	}
	return m.fmriSecondLevelDiscoveredCovariatesColumns, m.fmriSecondLevelDiscoveredCovariatesValues, strings.TrimSpace(m.fmriSecondLevelCovariatesDiscoveryError)
}

func (m Model) GetFmriSecondLevelDiscoveredContrastNames() []string {
	contrasts, _ := m.currentFmriSecondLevelContrastDiscovery()
	return contrasts
}

func (m Model) GetFmriSecondLevelDiscoveredCovariateValues(column string) []string {
	_, values, _ := m.currentFmriSecondLevelCovariatesDiscovery()
	if values == nil {
		return nil
	}
	return values[column]
}

func (m *Model) ensureFmriSecondLevelContrastDiscovery() error {
	key := m.nextFmriSecondLevelContrastDiscoveryKey()
	if key == "" {
		return fmt.Errorf("Select subjects and set a task/input root before choosing contrasts")
	}
	if m.fmriSecondLevelContrastDiscoveryKey == key {
		if errText := strings.TrimSpace(m.fmriSecondLevelContrastDiscoveryError); errText != "" {
			return errors.New(errText)
		}
		if len(m.fmriSecondLevelDiscoveredContrasts) == 0 {
			return fmt.Errorf("No common first-level contrasts found for the selected subjects")
		}
		return nil
	}

	inputRoot := m.fmriSecondLevelResolvedInputRoot()
	info, err := os.Stat(inputRoot)
	if err != nil {
		return m.setFmriSecondLevelContrastDiscoveryError(
			key,
			fmt.Errorf("Second-level input root is not readable: %s", inputRoot),
		)
	}
	if !info.IsDir() {
		return m.setFmriSecondLevelContrastDiscoveryError(
			key,
			fmt.Errorf("Second-level input root is not a directory: %s", inputRoot),
		)
	}

	subjects := m.fmriSecondLevelSelectedSubjects()
	common := make(map[string]struct{})
	for index, subject := range subjects {
		available, discoverErr := discoverFmriSecondLevelSubjectContrasts(
			inputRoot,
			subject,
			strings.TrimSpace(m.task),
		)
		if discoverErr != nil {
			return m.setFmriSecondLevelContrastDiscoveryError(key, discoverErr)
		}
		if len(available) == 0 {
			return m.setFmriSecondLevelContrastDiscoveryError(
				key,
				fmt.Errorf(
					"No valid first-level effect-size contrasts found for %s under %s",
					subject,
					inputRoot,
				),
			)
		}
		if index == 0 {
			for contrast := range available {
				common[contrast] = struct{}{}
			}
			continue
		}
		for contrast := range common {
			if _, exists := available[contrast]; !exists {
				delete(common, contrast)
			}
		}
	}

	discovered := sortedStringKeys(common)
	if len(discovered) == 0 {
		return m.setFmriSecondLevelContrastDiscoveryError(
			key,
			fmt.Errorf(
				"No common first-level contrasts found across selected subjects under %s",
				inputRoot,
			),
		)
	}

	m.fmriSecondLevelContrastDiscoveryKey = key
	m.fmriSecondLevelDiscoveredContrasts = discovered
	m.fmriSecondLevelContrastDiscoveryError = ""
	return nil
}

func (m *Model) ensureFmriSecondLevelCovariatesDiscovery() error {
	key := m.nextFmriSecondLevelCovariatesDiscoveryKey()
	if key == "" {
		return fmt.Errorf("Set a covariates file before choosing columns")
	}
	if m.fmriSecondLevelCovariatesDiscoveryKey == key {
		if errText := strings.TrimSpace(m.fmriSecondLevelCovariatesDiscoveryError); errText != "" {
			return errors.New(errText)
		}
		return nil
	}

	columns, values, err := discoverFmriSecondLevelCovariatesFile(key)
	if err != nil {
		return m.setFmriSecondLevelCovariatesDiscoveryError(key, err)
	}

	m.fmriSecondLevelCovariatesDiscoveryKey = key
	m.fmriSecondLevelDiscoveredCovariatesColumns = columns
	m.fmriSecondLevelDiscoveredCovariatesValues = values
	m.fmriSecondLevelCovariatesDiscoveryError = ""
	return nil
}

func (m *Model) setFmriSecondLevelContrastDiscoveryError(key string, err error) error {
	m.fmriSecondLevelContrastDiscoveryKey = key
	m.fmriSecondLevelDiscoveredContrasts = nil
	if err == nil {
		m.fmriSecondLevelContrastDiscoveryError = ""
		return nil
	}
	m.fmriSecondLevelContrastDiscoveryError = err.Error()
	return err
}

func (m *Model) setFmriSecondLevelCovariatesDiscoveryError(key string, err error) error {
	m.fmriSecondLevelCovariatesDiscoveryKey = key
	m.fmriSecondLevelDiscoveredCovariatesColumns = nil
	m.fmriSecondLevelDiscoveredCovariatesValues = nil
	if err == nil {
		m.fmriSecondLevelCovariatesDiscoveryError = ""
		return nil
	}
	m.fmriSecondLevelCovariatesDiscoveryError = err.Error()
	return err
}

func discoverFmriSecondLevelSubjectContrasts(
	inputRoot string,
	subjectLabel string,
	task string,
) (map[string]struct{}, error) {
	dirs, err := fmriSecondLevelCandidateContrastDirs(inputRoot, subjectLabel, task)
	if err != nil {
		return nil, err
	}

	contrasts := make(map[string]struct{})
	for _, dir := range dirs {
		sidecars, discoverErr := filepath.Glob(filepath.Join(dir, "*.json"))
		if discoverErr != nil {
			return nil, fmt.Errorf("Failed to enumerate first-level sidecars in %s: %w", dir, discoverErr)
		}
		sort.Strings(sidecars)
		for _, sidecarPath := range sidecars {
			contrastName, matches, parseErr := parseFmriSecondLevelContrastSidecar(
				sidecarPath,
				subjectLabel,
				task,
			)
			if parseErr != nil {
				return nil, parseErr
			}
			if matches {
				contrasts[contrastName] = struct{}{}
			}
		}
	}
	return contrasts, nil
}

func fmriSecondLevelCandidateContrastDirs(
	inputRoot string,
	subjectLabel string,
	task string,
) ([]string, error) {
	taskRoot := filepath.Join(inputRoot, subjectLabel, "fmri", "first_level", "task-"+task)
	subjectDirs, err := discoverContrastDirs(taskRoot)
	if err != nil {
		return nil, err
	}
	sharedDirs, err := discoverContrastDirs(inputRoot)
	if err != nil {
		return nil, err
	}

	ordered := make([]string, 0, len(subjectDirs)+len(sharedDirs))
	ordered = append(ordered, subjectDirs...)
	ordered = append(ordered, sharedDirs...)
	return ordered, nil
}

func discoverContrastDirs(root string) ([]string, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("Failed to read %s: %w", root, err)
	}

	dirs := make([]string, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() || !strings.HasPrefix(entry.Name(), "contrast-") {
			continue
		}
		dirs = append(dirs, filepath.Join(root, entry.Name()))
	}
	sort.Strings(dirs)
	return dirs, nil
}

func parseFmriSecondLevelContrastSidecar(
	sidecarPath string,
	subjectLabel string,
	task string,
) (string, bool, error) {
	payloadBytes, err := os.ReadFile(sidecarPath)
	if err != nil {
		return "", false, fmt.Errorf("Failed to read first-level sidecar %s: %w", sidecarPath, err)
	}

	var payload fmriSecondLevelFirstLevelSidecar
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		return "", false, fmt.Errorf("Invalid first-level JSON sidecar %s: %w", sidecarPath, err)
	}

	if strings.TrimSpace(payload.Subject) != subjectLabel || strings.TrimSpace(payload.Task) != task {
		return "", false, nil
	}
	if strings.TrimSpace(payload.OutputTypeActual) != fmriSecondLevelRequiredOutputType {
		return "", false, nil
	}
	if strings.TrimSpace(payload.ContrastName) == "" {
		return "", false, fmt.Errorf("Missing contrast_name in first-level sidecar %s", sidecarPath)
	}
	if payload.ContrastCfg == nil {
		return "", false, fmt.Errorf("Missing contrast_cfg object in first-level sidecar %s", sidecarPath)
	}
	if strings.TrimSpace(fmt.Sprint(payload.ContrastCfg["fmriprep_space"])) != fmriSecondLevelRequiredSpace {
		return "", false, nil
	}

	niftiPath := strings.TrimSuffix(sidecarPath, ".json") + ".nii.gz"
	info, err := os.Stat(niftiPath)
	if err != nil {
		return "", false, fmt.Errorf("Missing first-level NIfTI for sidecar %s", sidecarPath)
	}
	if info.IsDir() {
		return "", false, fmt.Errorf("Expected NIfTI file but found directory: %s", niftiPath)
	}

	return strings.TrimSpace(payload.ContrastName), true, nil
}

func discoverFmriSecondLevelCovariatesFile(path string) ([]string, map[string][]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("Failed to open second-level covariates file %s: %w", path, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	switch strings.ToLower(filepath.Ext(path)) {
	case ".tsv":
		reader.Comma = '\t'
	case ".csv":
	default:
		return nil, nil, fmt.Errorf("Second-level covariates file must be .tsv or .csv: %s", path)
	}
	reader.FieldsPerRecord = -1

	header, err := reader.Read()
	if err != nil {
		if errors.Is(err, io.EOF) {
			return nil, nil, fmt.Errorf("Second-level covariates file is empty: %s", path)
		}
		return nil, nil, fmt.Errorf("Failed reading second-level covariates header from %s: %w", path, err)
	}

	columns := make([]string, 0, len(header))
	seenColumns := make(map[string]struct{}, len(header))
	valueSets := make(map[string]map[string]struct{}, len(header))
	for _, raw := range header {
		column := strings.TrimSpace(raw)
		if column == "" {
			return nil, nil, fmt.Errorf("Second-level covariates file has an empty column name: %s", path)
		}
		if _, exists := seenColumns[column]; exists {
			return nil, nil, fmt.Errorf("Second-level covariates file has duplicate column %q", column)
		}
		seenColumns[column] = struct{}{}
		columns = append(columns, column)
		valueSets[column] = make(map[string]struct{})
	}

	for {
		record, err := reader.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, nil, fmt.Errorf("Failed reading second-level covariates rows from %s: %w", path, err)
		}
		for index, column := range columns {
			if index >= len(record) {
				continue
			}
			value := strings.TrimSpace(record[index])
			if value == "" {
				continue
			}
			valueSets[column][value] = struct{}{}
		}
	}

	values := make(map[string][]string, len(columns))
	for _, column := range columns {
		values[column] = sortedStringKeys(valueSets[column])
	}
	return columns, values, nil
}

func sortedStringKeys(set map[string]struct{}) []string {
	if len(set) == 0 {
		return nil
	}
	values := make([]string, 0, len(set))
	for value := range set {
		values = append(values, value)
	}
	sort.Strings(values)
	return values
}
