package wizard

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"
)

// File layout notes:
// - `commands.go`: selection getters, shared arg builders, root BuildCommand flow.
// - `commands_build_*.go`: pipeline-specific advanced argument builders.

///////////////////////////////////////////////////////////////////
// Getters
///////////////////////////////////////////////////////////////////

func (m Model) SelectedCategories() []string {
	var result []string
	for i, sel := range m.selected {
		if sel && i < len(m.categories) {
			result = append(result, m.categories[i])
		}
	}
	sort.Strings(result)
	return result
}

func (m Model) isCategorySelected(category string) bool {
	for i, sel := range m.selected {
		if sel && i < len(m.categories) && m.categories[i] == category {
			return true
		}
	}
	return false
}

func (m Model) SelectedComputations() []string {
	var result []string

	for i, sel := range m.computationSelected {
		if sel && i < len(m.computations) {
			key := m.computations[i].Key

			result = append(result, key)
		}
	}

	if len(result) > 0 {
		result = append(result, "trial_table")
	}

	seen := make(map[string]bool)
	unique := make([]string, 0, len(result))
	for _, r := range result {
		if !seen[r] {
			seen[r] = true
			unique = append(unique, r)
		}
	}

	sort.Strings(unique)
	return unique
}

// isComputationSelected checks if a specific computation is currently selected.
func (m Model) isComputationSelected(computation string) bool {
	for i, sel := range m.computationSelected {
		if sel && i < len(m.computations) {
			key := m.computations[i].Key
			if key == computation {
				return true
			}
		}
	}
	return false
}

func (m Model) SelectedSubjectIDs() []string {
	var result []string
	for id, sel := range m.subjectSelected {
		if sel {
			result = append(result, id)
		}
	}
	sort.Strings(result)
	return result
}

func (m Model) SelectedBands() []string {
	var result []string
	for i, sel := range m.bandSelected {
		if sel && i < len(m.bands) {
			result = append(result, m.bands[i].Key)
		}
	}
	sort.Strings(result)
	return result
}

func (m Model) GetFrequencyBandDefinitions() []string {
	if len(m.bands) == len(frequencyBands) {
		allMatch := true
		for i, band := range m.bands {
			def := frequencyBands[i]
			if band.Key != def.Key || band.LowHz != def.LowHz || band.HighHz != def.HighHz {
				allMatch = false
				break
			}
		}
		if allMatch {
			return nil
		}
	}

	var result []string
	for _, band := range m.bands {
		result = append(result, fmt.Sprintf("%s:%.1f:%.1f", band.Key, band.LowHz, band.HighHz))
	}
	return result
}

func (m Model) SelectedSpatialModes() []string {
	var result []string
	for i, mode := range spatialModes {
		if m.spatialSelected[i] {
			result = append(result, mode.Key)
		}
	}
	return result
}

func (m Model) SelectedROIs() []string {
	var result []string
	for i := 0; i < len(m.rois); i++ {
		if m.roiSelected[i] {
			result = append(result, m.rois[i].Key)
		}
	}
	return result
}

func (m Model) GetROIDefinitions() []string {
	if len(m.rois) == len(defaultROIs) {
		allMatch := true
		for i, roi := range m.rois {
			def := defaultROIs[i]
			if roi.Key != def.Key || roi.Channels != def.Channels {
				allMatch = false
				break
			}
		}
		if allMatch {
			return nil
		}
	}

	// Build set of unavailable channels (case-insensitive) for filtering
	unavailableSet := make(map[string]bool)
	for _, ch := range m.unavailableChannels {
		unavailableSet[strings.ToUpper(strings.TrimSpace(ch))] = true
	}

	var result []string
	for i, roi := range m.rois {
		if m.roiSelected[i] {
			// Filter unavailable channels from ROI channels for CLI
			var filteredChannels []string
			for _, ch := range strings.Split(roi.Channels, ",") {
				ch = strings.TrimSpace(ch)
				if ch != "" && !unavailableSet[strings.ToUpper(ch)] {
					filteredChannels = append(filteredChannels, ch)
				}
			}
			filteredChannelsStr := strings.Join(filteredChannels, ",")
			if filteredChannelsStr != "" {
				result = append(result, fmt.Sprintf("%s:%s", roi.Key, filteredChannelsStr))
			}
		}
	}
	return result
}

func (m Model) SelectedFeatureFiles() []string {
	var result []string
	for _, file := range m.featureFiles {
		if m.featureFileSelected[file.Key] {
			result = append(result, file.Key)
		}
	}
	return result
}

func (m Model) GetApplicableFeatureFiles() []FeatureFile {
	applicableKeys := make(map[string]bool)

	for i, sel := range m.computationSelected {
		if !sel || i >= len(m.computations) {
			continue
		}
		compKey := m.computations[i].Key
		if features, ok := computationApplicableFeatures[compKey]; ok {
			for _, f := range features {
				applicableKeys[f] = true
			}
		}
	}

	if len(applicableKeys) == 0 {
		return featureFileOptions
	}

	// Filter feature files to only those that are applicable
	var result []FeatureFile
	for _, file := range featureFileOptions {
		if applicableKeys[file.Key] {
			result = append(result, file)
		}
	}
	return result
}

func (m Model) SelectedPreprocessingStages() []string {
	var result []string
	for i, sel := range m.prepStageSelected {
		if sel && i < len(m.prepStages) {
			result = append(result, m.prepStages[i].Key)
		}
	}
	sort.Strings(result)
	return result
}

func (m Model) selectedConnectivityMeasures() []string {
	var result []string
	for i, measure := range connectivityMeasures {
		if m.connectivityMeasures[i] {
			result = append(result, measure.Key)
		}
	}
	return result
}

func (m Model) selectedDirectedConnectivityMeasures() []string {
	var result []string
	for i, measure := range directedConnectivityMeasures {
		if m.directedConnMeasures[i] {
			result = append(result, measure.Key)
		}
	}
	return result
}

///////////////////////////////////////////////////////////////////
// Subject Filtering
///////////////////////////////////////////////////////////////////

func (m Model) getFilteredSubjects() []types.SubjectStatus {
	if m.subjectFilter == "" && !m.showOnlyValid {
		return m.subjects
	}

	var filtered []types.SubjectStatus
	filterLower := strings.ToLower(m.subjectFilter)

	for _, s := range m.subjects {
		if m.subjectFilter != "" && !strings.Contains(strings.ToLower(s.ID), filterLower) {
			continue
		}

		if m.showOnlyValid && !m.isSubjectValid(s) {
			continue
		}

		filtered = append(filtered, s)
	}

	return filtered
}

func (m Model) isSubjectValid(s types.SubjectStatus) bool {
	valid, _ := m.Pipeline.ValidateSubject(s)
	return valid
}

///////////////////////////////////////////////////////////////////
// Command Builder
///////////////////////////////////////////////////////////////////

// argBuilder provides helper methods for building command arguments
type argBuilder struct {
	args []string
}

func newArgBuilder() *argBuilder {
	return &argBuilder{args: make([]string, 0)}
}

func (ab *argBuilder) addIfNonZero(flag string, value float64, format string) {
	if value != 0 {
		ab.args = append(ab.args, flag, fmt.Sprintf(format, value))
	}
}

func (ab *argBuilder) addIfNonZeroInt(flag string, value int) {
	if value != 0 {
		ab.args = append(ab.args, flag, fmt.Sprintf("%d", value))
	}
}

func (ab *argBuilder) addIfNonEmpty(flag string, value string) {
	trimmed := strings.TrimSpace(value)
	if trimmed != "" {
		ab.args = append(ab.args, flag, trimmed)
	}
}

func (ab *argBuilder) addBoolFlag(flag string, value bool) {
	if value {
		ab.args = append(ab.args, flag)
	} else {
		flagName := strings.TrimPrefix(flag, "--")
		ab.args = append(ab.args, "--no-"+flagName)
	}
}

func (ab *argBuilder) addOptionalBoolFlag(flag string, value *bool) {
	if value != nil {
		ab.addBoolFlag(flag, *value)
	}
}

func (ab *argBuilder) addListFlag(flag string, values []string) {
	if len(values) > 0 {
		ab.args = append(ab.args, flag)
		ab.args = append(ab.args, values...)
	}
}

func (ab *argBuilder) addSpaceListFlag(flag string, spec string) {
	trimmed := strings.TrimSpace(spec)
	if trimmed != "" {
		ab.args = append(ab.args, flag)
		ab.args = append(ab.args, splitSpaceList(trimmed)...)
	}
}

func (ab *argBuilder) addSpaceListFlagWithLengthCheck(flag string, spec string, expectedLength int) {
	trimmed := strings.TrimSpace(spec)
	if trimmed != "" {
		vals := splitSpaceList(trimmed)
		if len(vals) == expectedLength {
			ab.args = append(ab.args, flag)
			ab.args = append(ab.args, vals...)
		}
	}
}

func (ab *argBuilder) build() []string {
	return ab.args
}

func (m Model) BuildCommand() string {
	return executor.JoinCommand(runtime.GOOS, m.BuildCommandArgs())
}

func (m Model) BuildCommandArgs() []string {
	parts := []string{"eeg-pipeline", m.Pipeline.CLICommand()}

	needsMode := m.Pipeline == types.PipelinePreprocessing ||
		m.Pipeline == types.PipelineFeatures ||
		m.Pipeline == types.PipelineBehavior ||
		m.Pipeline == types.PipelineML ||
		m.Pipeline == types.PipelineFmri ||
		m.Pipeline == types.PipelineFmriAnalysis

	hasValidModeIndex := len(m.modeOptions) > m.modeIndex
	modeToUse := ""
	if needsMode && hasValidModeIndex {
		modeToUse = m.modeOptions[m.modeIndex]

		cliMode := modeToUse
		if m.Pipeline == types.PipelineFmriAnalysis && modeToUse == "trial-signatures" {
			if m.fmriTrialSigMethodIndex%2 == 1 {
				cliMode = "lss"
			} else {
				cliMode = "beta-series"
			}
		}
		parts = append(parts, cliMode)
	}

	usesFmriRestRoots :=
		(m.Pipeline == types.PipelineFmri && m.fmriTaskIsRest) ||
			(m.Pipeline == types.PipelineFmriAnalysis && modeToUse == "rest")

	if m.Pipeline == types.PipelineML {
		parts = append(parts, "--cv-scope", m.mlScope.CLIValue())
	}

	if m.Pipeline == types.PipelineBehavior && m.modeOptions[m.modeIndex] == styles.ModeCompute {
		comps := m.SelectedComputations()
		if len(comps) > 0 {
			parts = append(parts, "--computations")
			parts = append(parts, comps...)
		}

		featureFiles := m.SelectedFeatureFiles()
		if len(featureFiles) > 0 && len(featureFiles) < len(m.featureFiles) {
			parts = append(parts, "--feature-files")
			parts = append(parts, featureFiles...)
		}

		bands := m.SelectedBands()
		if len(bands) > 0 && len(bands) < len(m.bands) {
			parts = append(parts, "--bands")
			parts = append(parts, bands...)
		}
	} else if m.Pipeline != types.PipelinePreprocessing {
		cats := m.SelectedCategories()
		if len(cats) > 0 && len(cats) < len(m.categories) {
			parts = append(parts, "--categories")
			parts = append(parts, cats...)
		}
	}

	if m.Pipeline == types.PipelineFeatures && m.modeOptions[m.modeIndex] == styles.ModeCompute {
		bands := m.SelectedBands()
		if len(bands) > 0 && len(bands) < len(m.bands) {
			parts = append(parts, "--bands")
			parts = append(parts, bands...)
		}

		freqBandDefs := m.GetFrequencyBandDefinitions()
		if len(freqBandDefs) > 0 {
			parts = append(parts, "--frequency-bands")
			parts = append(parts, freqBandDefs...)
		}

		roiDefs := m.GetROIDefinitions()
		if len(roiDefs) > 0 {
			parts = append(parts, "--rois")
			parts = append(parts, roiDefs...)
		}
	}

	needsPaths := m.Pipeline == types.PipelinePreprocessing ||
		m.Pipeline == types.PipelineFeatures ||
		m.Pipeline == types.PipelineBehavior ||
		m.Pipeline == types.PipelineML ||
		m.Pipeline == types.PipelineFmri ||
		m.Pipeline == types.PipelineFmriAnalysis

	if needsPaths {
		if m.Pipeline == types.PipelineFmri || m.Pipeline == types.PipelineFmriAnalysis {
			if m.bidsFmriRoot != "" {
				parts = append(parts, "--bids-fmri-root", expandUserPath(m.bidsFmriRoot))
			}
			if usesFmriRestRoots && m.bidsRestRoot != "" {
				parts = append(parts, "--bids-rest-root", expandUserPath(m.bidsRestRoot))
			}
		} else {
			if m.bidsRoot != "" {
				parts = append(parts, "--bids-root", expandUserPath(m.bidsRoot))
			}
			if m.bidsRestRoot != "" {
				parts = append(parts, "--bids-rest-root", expandUserPath(m.bidsRestRoot))
			}
		}
		if m.derivRoot != "" {
			parts = append(parts, "--deriv-root", expandUserPath(m.derivRoot))
		}
		if usesFmriRestRoots && m.derivRestRoot != "" {
			parts = append(parts, "--deriv-rest-root", expandUserPath(m.derivRestRoot))
		} else if m.Pipeline != types.PipelineFmri && m.Pipeline != types.PipelineFmriAnalysis && m.derivRestRoot != "" {
			parts = append(parts, "--deriv-rest-root", expandUserPath(m.derivRestRoot))
		}
	}

	if m.Pipeline == types.PipelineFeatures && m.modeOptions[m.modeIndex] == styles.ModeCompute {
		if m.prepTaskIsRest {
			parts = append(parts, "--task-is-rest", "--no-power-require-baseline", "--no-power-subtract-evoked")
		} else {
			parts = append(parts, "--no-task-is-rest")
		}

		spatial := m.SelectedSpatialModes()
		if len(spatial) > 0 && len(spatial) < len(spatialModes) {
			parts = append(parts, "--spatial")
			parts = append(parts, spatial...)
		}

		for _, tr := range m.TimeRanges {
			tmin := normalizeTimeRangeValue(tr.Tmin)
			tmax := normalizeTimeRangeValue(tr.Tmax)
			parts = append(parts, "--time-range", tr.Name, tmin, tmax)
		}
	}

	if !m.useDefaultAdvanced {
		switch m.Pipeline {
		case types.PipelineFeatures:
			parts = append(parts, m.buildFeaturesAdvancedArgs()...)
		case types.PipelineBehavior:
			parts = append(parts, m.buildBehaviorAdvancedArgs()...)
		case types.PipelineML:
			parts = append(parts, m.buildMLAdvancedArgs()...)
		case types.PipelinePreprocessing:
			parts = append(parts, m.buildPreprocessingAdvancedArgs()...)
		case types.PipelineFmri:
			parts = append(parts, m.buildFmriAdvancedArgs()...)
		case types.PipelineFmriAnalysis:
			parts = append(parts, m.buildFmriAnalysisAdvancedArgs()...)
		}
	}

	for _, override := range parseConfigSetOverrides(m.configSetOverrides) {
		parts = append(parts, "--set", override)
	}

	if m.task != "" {
		parts = append(parts, "--task", m.task)
	}

	subjs := m.SelectedSubjectIDs()
	allSubjectsSelected := len(subjs) == 0 || len(subjs) == len(m.subjects)
	if allSubjectsSelected {
		parts = append(parts, "--all-subjects")
	} else {
		for _, s := range subjs {
			parts = append(parts, "--subject", s)
		}
	}

	if m.DryRunMode {
		parts = append(parts, "--dry-run")
	}

	return parts
}

func splitCSVList(raw string) []string {
	parts := strings.FieldsFunc(raw, func(r rune) bool {
		return r == ',' || r == ';' || r == '\t' || r == '\n'
	})
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		s := strings.TrimSpace(p)
		if s == "" {
			continue
		}
		out = append(out, s)
	}
	return out
}

func parseConfigSetOverrides(raw string) []string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return nil
	}

	parts := strings.FieldsFunc(trimmed, func(r rune) bool {
		return r == ';' || r == '\n' || r == '\r'
	})
	overrides := make([]string, 0, len(parts))
	for _, part := range parts {
		override := strings.TrimSpace(part)
		if strings.HasPrefix(override, "--set ") {
			override = strings.TrimSpace(strings.TrimPrefix(override, "--set "))
		}
		if override == "" || !strings.Contains(override, "=") {
			continue
		}
		overrides = append(overrides, override)
	}
	if len(overrides) == 0 {
		return nil
	}
	return overrides
}

func splitSpaceList(raw string) []string {
	parsed, err := splitShellWords(raw)
	if err == nil && len(parsed) > 0 {
		return parsed
	}
	out := strings.Fields(raw)
	if len(out) == 0 {
		return nil
	}
	return out
}

func splitLooseList(raw string) []string {
	replacer := strings.NewReplacer(
		",", " ",
		";", " ",
		"\t", " ",
		"\n", " ",
		"\r", " ",
	)
	normalized := replacer.Replace(raw)
	return splitSpaceList(normalized)
}

func splitShellWords(raw string) ([]string, error) {
	type quoteState int
	const (
		stateNone quoteState = iota
		stateSingle
		stateDouble
	)

	var out []string
	var cur strings.Builder
	state := stateNone
	escaped := false

	flush := func() {
		if cur.Len() == 0 {
			return
		}
		out = append(out, cur.String())
		cur.Reset()
	}

	for _, r := range raw {
		if escaped {
			cur.WriteRune(r)
			escaped = false
			continue
		}

		switch state {
		case stateNone:
			if r == '\\' {
				escaped = true
				continue
			}
			if r == '\'' {
				state = stateSingle
				continue
			}
			if r == '"' {
				state = stateDouble
				continue
			}
			if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
				flush()
				continue
			}
			cur.WriteRune(r)
		case stateSingle:
			if r == '\'' {
				state = stateNone
				continue
			}
			cur.WriteRune(r)
		case stateDouble:
			if r == '\\' {
				escaped = true
				continue
			}
			if r == '"' {
				state = stateNone
				continue
			}
			cur.WriteRune(r)
		}
	}

	if escaped {
		return nil, fmt.Errorf("unfinished escape sequence")
	}
	if state != stateNone {
		return nil, fmt.Errorf("unterminated quote")
	}
	flush()
	return out, nil
}

// buildMLAdvancedArgs returns CLI args for machine learning pipeline advanced options
func expandUserPath(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return value
	}
	if strings.HasPrefix(value, "~") {
		home, err := os.UserHomeDir()
		if err == nil {
			return expandUserPathWithHome(value, home)
		}
	}
	return filepath.Clean(value)
}

func expandUserPathWithHome(value string, home string) string {
	if value == "~" {
		return filepath.Clean(home)
	}
	if len(value) >= 2 && (value[1] == '/' || value[1] == '\\') {
		relative := strings.NewReplacer(
			"/", string(filepath.Separator),
			"\\", string(filepath.Separator),
		).Replace(value[2:])
		return filepath.Clean(filepath.Join(home, relative))
	}
	return filepath.Clean(value)
}

func normalizeTimeRangeValue(value string) string {
	if value == "" {
		return "none"
	}
	return value
}

func splitListInput(value string) []string {
	parts := strings.FieldsFunc(value, func(r rune) bool {
		return r == ',' || r == ';'
	})
	var out []string
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}
