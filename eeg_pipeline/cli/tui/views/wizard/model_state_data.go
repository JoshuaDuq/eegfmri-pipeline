package wizard

import (
	"strings"

	"github.com/eeg-pipeline/tui/types"
)

// Subject and metadata state mutators/accessors.

func (m *Model) SetSubjects(subjects []types.SubjectStatus) {
	m.subjects = subjects
	m.subjectsLoading = false
	m.subjectLoadError = ""
	for _, s := range subjects {
		m.subjectSelected[s.ID] = true
	}

	// Calculate feature and computation availability based on all subjects
	m.updateFeatureAvailability()
	m.updateComputationAvailability()
}

// updateFeatureAvailability recalculates feature availability based on selected subjects
func (m *Model) updateFeatureAvailability() {
	m.featureAvailability = make(map[string]bool)
	m.featureLastModified = make(map[string]string)

	for _, s := range m.subjects {
		isSelected := m.subjectSelected[s.ID]
		if !isSelected {
			continue
		}

		if s.FeatureAvailability == nil {
			continue
		}

		for cat, info := range s.FeatureAvailability.Features {
			if info.Available {
				m.featureAvailability[cat] = true
				hasLastModified := info.LastModified != ""
				if hasLastModified {
					existing, exists := m.featureLastModified[cat]
					isNewer := !exists || info.LastModified > existing
					if isNewer {
						m.featureLastModified[cat] = info.LastModified
					}
				}
			}
		}
	}
}

// updateComputationAvailability recalculates computation availability based on selected subjects
func (m *Model) updateComputationAvailability() {
	m.computationAvailability = make(map[string]bool)
	m.computationLastModified = make(map[string]string)

	for _, s := range m.subjects {
		isSelected := m.subjectSelected[s.ID]
		if !isSelected {
			continue
		}

		if s.FeatureAvailability == nil || s.FeatureAvailability.Computations == nil {
			continue
		}

		for comp, info := range s.FeatureAvailability.Computations {
			if info.Available {
				m.computationAvailability[comp] = true
				hasLastModified := info.LastModified != ""
				if hasLastModified {
					existing, exists := m.computationLastModified[comp]
					isNewer := !exists || info.LastModified > existing
					if isNewer {
						m.computationLastModified[comp] = info.LastModified
					}
				}
			}
		}
	}
}

func (m *Model) SetSubjectsLoading() {
	m.subjectsLoading = true
	m.subjectLoadError = ""
}

func (m Model) IsOnSubjectSelectionStep() bool {
	return m.CurrentStep == types.StepSelectSubjects
}

func (m *Model) SetSubjectLoadError(message string) {
	m.subjects = nil
	m.subjectSelected = make(map[string]bool)
	m.subjectsLoading = false
	m.subjectLoadError = strings.TrimSpace(message)
	m.subjectCursor = 0
	m.featureAvailability = make(map[string]bool)
	m.featureLastModified = make(map[string]string)
	m.computationAvailability = make(map[string]bool)
	m.computationLastModified = make(map[string]string)
	m.availableWindows = nil
	m.availableColumns = nil
	m.availableWindowsByFeature = make(map[string][]string)
	m.availableChannels = nil
	m.unavailableChannels = nil
}

func (m *Model) SetTimeRanges(ranges []types.TimeRange) {
	if len(ranges) > 0 {
		m.TimeRanges = ranges
	}
}

// SetBands restores band definitions and selection states from persisted state.
func (m *Model) SetBands(bands []FrequencyBand, selected []bool) {
	if len(bands) > 0 {
		m.bands = bands
		m.bandSelected = make(map[int]bool)
		for i, sel := range selected {
			if i < len(bands) {
				m.bandSelected[i] = sel
			}
		}
	}
}

// GetBands returns the current band definitions for persistence.
func (m Model) GetBands() []FrequencyBand {
	return m.bands
}

// GetBandSelected returns the band selection states for persistence.
func (m Model) GetBandSelected() []bool {
	result := make([]bool, len(m.bands))
	for i := range m.bands {
		result[i] = m.bandSelected[i]
	}
	return result
}

// SetROIs restores ROI definitions and selection states from persisted state.
func (m *Model) SetROIs(rois []ROIDefinition, selected []bool) {
	if len(rois) > 0 {
		m.rois = rois
		m.roiSelected = make(map[int]bool)
		for i, sel := range selected {
			if i < len(rois) {
				m.roiSelected[i] = sel
			}
		}
	}
}

// GetROIs returns the current ROI definitions for persistence.
func (m Model) GetROIs() []ROIDefinition {
	return m.rois
}

// GetROISelected returns the ROI selection states for persistence.
func (m Model) GetROISelected() []bool {
	result := make([]bool, len(m.rois))
	for i := range m.rois {
		result[i] = m.roiSelected[i]
	}
	return result
}

// SetSpatialSelected restores spatial mode selection from persisted state.
func (m *Model) SetSpatialSelected(selected []bool) {
	if len(selected) > 0 {
		for i, sel := range selected {
			if i < len(spatialModes) {
				m.spatialSelected[i] = sel
			}
		}
	}
}

// SetSelectedCategories restores feature category selection from config keys.
func (m *Model) SetSelectedCategories(selected []string) {
	if m.selected == nil {
		m.selected = make(map[int]bool, len(m.categories))
	}

	selectedSet := make(map[string]bool, len(selected))
	for _, category := range selected {
		category = strings.TrimSpace(category)
		if category != "" {
			selectedSet[category] = true
		}
	}

	for i, category := range m.categories {
		m.selected[i] = selectedSet[category]
	}
}

// SetSelectedSpatialModes restores spatial mode selection from config keys.
func (m *Model) SetSelectedSpatialModes(selected []string) {
	if m.spatialSelected == nil {
		m.spatialSelected = make(map[int]bool, len(spatialModes))
	}

	selectedSet := make(map[string]bool, len(selected))
	for _, mode := range selected {
		mode = strings.TrimSpace(mode)
		if mode != "" {
			selectedSet[mode] = true
		}
	}

	for i, mode := range spatialModes {
		m.spatialSelected[i] = selectedSet[mode.Key]
	}
}

// GetSpatialSelected returns spatial selection states for persistence.
func (m Model) GetSpatialSelected() []bool {
	result := make([]bool, len(spatialModes))
	for i := range spatialModes {
		result[i] = m.spatialSelected[i]
	}
	return result
}

// SetAvailableMetadata stores runtime-derived metadata (e.g., discovered time
// windows / event columns) for use in UI hints and lightweight validation.
func (m *Model) SetAvailableMetadata(windows []string, eventColumns []string) {
	m.availableWindows = append([]string(nil), windows...)
	m.availableColumns = append([]string(nil), eventColumns...)
}

// SetAvailableWindowsByFeature stores windows discovered per feature group.
func (m *Model) SetAvailableWindowsByFeature(windowsByFeature map[string][]string) {
	if m.availableWindowsByFeature == nil {
		m.availableWindowsByFeature = make(map[string][]string)
	}
	for feature, windows := range windowsByFeature {
		m.availableWindowsByFeature[feature] = append([]string(nil), windows...)
	}
}

// SetChannelInfo stores available and unavailable EEG channels from BIDS data
// and preprocessing logs. Used by ROI selection to validate channel names.
func (m *Model) SetChannelInfo(available, unavailable []string) {
	m.availableChannels = append([]string(nil), available...)
	m.unavailableChannels = append([]string(nil), unavailable...)
}

// SetDiscoveredColumns sets the columns and values discovered from events/trial tables
func (m *Model) SetDiscoveredColumns(columns []string, values map[string][]string, source string) {
	m.discoveredColumns = columns
	m.discoveredColumnValues = values
	m.columnDiscoverySource = source
	m.columnDiscoveryDone = true
	m.columnDiscoveryError = ""
}

func (m *Model) SetTrialTableColumns(columns []string, values map[string][]string) {
	m.trialTableColumns = columns
	m.trialTableColumnValues = values
	m.trialTableDiscoveryDone = true
	m.trialTableDiscoveryError = ""
}

// SetColumnsDiscoveryError sets the error from column discovery
func (m *Model) SetColumnsDiscoveryError(err error) {
	if err == nil {
		return
	}
	m.columnDiscoveryError = err.Error()
	m.columnDiscoveryDone = true
}

func (m *Model) SetTrialTableDiscoveryError(err error) {
	if err == nil {
		return
	}
	m.trialTableDiscoveryError = err.Error()
	m.trialTableDiscoveryDone = true
}

// GetDiscoveredColumnValues returns the unique values for a column.
// Checks primary discovered columns first, then trial table columns as fallback.
func (m Model) GetDiscoveredColumnValues(column string) []string {
	// First check the primary discovery source
	if m.discoveredColumnValues != nil {
		if vals, ok := m.discoveredColumnValues[column]; ok && len(vals) > 0 {
			return vals
		}
	}
	// Fallback to trial table values (in case column came from trial table discovery)
	if m.trialTableColumnValues != nil {
		if vals, ok := m.trialTableColumnValues[column]; ok && len(vals) > 0 {
			return vals
		}
	}
	return nil
}

// GetAvailableColumns returns discovered event columns, falling back to subject metadata.
func (m Model) GetAvailableColumns() []string {
	if len(m.discoveredColumns) > 0 {
		return m.discoveredColumns
	}
	return m.availableColumns
}

func (m *Model) SetFmriDiscoveredColumns(columns []string, values map[string][]string, source string) {
	m.fmriDiscoveredColumns = columns
	m.fmriDiscoveredColumnValues = values
	m.fmriColumnDiscoveryDone = true
	m.fmriColumnDiscoveryError = ""
}

// SetFmriColumnsDiscoveryError sets the error from fMRI column discovery
func (m *Model) SetFmriColumnsDiscoveryError(err error) {
	if err == nil {
		return
	}
	m.fmriColumnDiscoveryError = err.Error()
	m.fmriColumnDiscoveryDone = true
}

// GetFmriDiscoveredColumnValues returns the unique values for an fMRI column
func (m Model) GetFmriDiscoveredColumnValues(column string) []string {
	if m.fmriDiscoveredColumnValues == nil {
		return nil
	}
	return m.fmriDiscoveredColumnValues[column]
}

// SetMultigroupStats sets the multigroup stats discovered from precomputed stats
func (m *Model) SetMultigroupStats(available bool, groups []string, nFeatures int, nSignificant int, file string) {
	m.multigroupStatsAvailable = available
	m.multigroupStatsGroups = groups
	m.multigroupStatsNFeatures = nFeatures
	m.multigroupStatsNSignificant = nSignificant
	m.multigroupStatsFile = file
	m.multigroupStatsDiscoveryDone = true
}

// HasMultigroupStats returns whether multigroup stats are available
func (m Model) HasMultigroupStats() bool {
	return m.multigroupStatsAvailable && len(m.multigroupStatsGroups) > 0
}

// GetMultigroupStatsGroups returns the group labels from precomputed multigroup stats
func (m Model) GetMultigroupStatsGroups() []string {
	return m.multigroupStatsGroups
}

// SetDiscoveredROIs sets the ROIs discovered from feature parquet files
func (m *Model) SetDiscoveredROIs(rois []string) {
	m.discoveredROIs = rois
	m.roiDiscoveryDone = true
	m.roiDiscoveryError = ""
}

// SetROIDiscoveryError sets the error from ROI discovery
func (m *Model) SetROIDiscoveryError(err error) {
	if err == nil {
		return
	}
	m.roiDiscoveryError = err.Error()
	m.roiDiscoveryDone = true
}
