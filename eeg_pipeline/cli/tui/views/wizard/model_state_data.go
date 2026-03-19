package wizard

import (
	"sort"
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

func (m *Model) SetFeaturePlotters(plotters map[string][]PlotterInfo) {
	m.featurePlotters = plotters
	m.featurePlotterError = ""
	if m.featurePlotterSelected == nil {
		m.featurePlotterSelected = make(map[string]bool)
	}
	// Default: select all discovered plotters (the selection step can narrow).
	for _, entries := range plotters {
		for _, p := range entries {
			if p.ID != "" {
				m.featurePlotterSelected[p.ID] = true
			}
		}
	}
}

func (m *Model) SetFeaturePlottersError(err error) {
	if err == nil {
		return
	}
	m.featurePlotterError = err.Error()
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
	m.trialTableFeatureCategories = detectFeatureCategoriesFromColumns(columns)
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

// SetConditionEffectsColumns sets the columns, values, and windows discovered from condition effects files
func (m *Model) SetConditionEffectsColumns(columns []string, values map[string][]string, windows []string) {
	m.conditionEffectsColumns = columns
	m.conditionEffectsColumnValues = values
	m.conditionEffectsWindows = windows
	m.conditionEffectsDiscoveryDone = true
	m.conditionEffectsDiscoveryError = ""
}

// SetConditionEffectsDiscoveryError sets the error from condition effects discovery
func (m *Model) SetConditionEffectsDiscoveryError(err error) {
	if err == nil {
		return
	}
	m.conditionEffectsDiscoveryError = err.Error()
	m.conditionEffectsDiscoveryDone = true
}

// GetConditionEffectsColumnValues returns the unique values for a condition effects column
func (m Model) GetConditionEffectsColumnValues(column string) []string {
	if m.conditionEffectsColumnValues == nil {
		return nil
	}
	return m.conditionEffectsColumnValues[column]
}

// GetAvailableColumns returns the best available columns for dropdowns.
// Prefers discovered event columns, then falls back to lightweight columns
// surfaced by the subjects endpoint.
func (m Model) GetAvailableColumns() []string {
	if len(m.discoveredColumns) > 0 {
		return m.discoveredColumns
	}
	return m.availableColumns
}

// GetPlottingComparisonColumns returns columns for plotting comparison from trial table (events.tsv).
// Feature plots (connectivity, power, etc.) use trial-level columns for condition comparisons,
// not condition effects stats columns.
func (m Model) GetPlottingComparisonColumns() []string {
	// Prefer columns discovered from trial tables (richer, includes value maps),
	// but fall back to the lightweight columns surfaced by the subjects endpoint
	// so dropdowns still work even if discovery hasn't completed yet.
	if len(m.discoveredColumns) > 0 {
		return m.discoveredColumns
	}
	return m.availableColumns
}

func (m Model) GetTrialTableFeatureColumns() []string {
	if len(m.trialTableColumns) == 0 {
		return nil
	}

	// Keep in sync with eeg_pipeline.analysis.behavior.orchestration.FEATURE_COLUMN_PREFIXES
	featurePrefixes := []string{
		"power_",
		"connectivity_",
		"directedconnectivity_",
		"sourcelocalization_",
		"aperiodic_",
		"erp_",
		"itpc_",
		"pac_",
		"complexity_",
		"bursts_",
		"quality_",
		"erds_",
		"spectral_",
		"ratios_",
		"asymmetry_",
		"microstates_",
		"temporal_",
	}

	out := make([]string, 0, len(m.trialTableColumns))
	for _, col := range m.trialTableColumns {
		c := strings.TrimSpace(col)
		if c == "" {
			continue
		}
		for _, p := range featurePrefixes {
			if strings.HasPrefix(c, p) {
				out = append(out, c)
				break
			}
		}
	}
	return out
}

func (m Model) GetTrialTableFeatureCategories() []string {
	if len(m.trialTableFeatureCategories) == 0 {
		return nil
	}
	out := make([]string, 0, len(m.trialTableFeatureCategories))
	out = append(out, m.trialTableFeatureCategories...)
	return out
}

var knownFeatureScopes = map[string]struct{}{
	"global": {},
	"roi":    {},
	"ch":     {},
	"chpair": {},
}

var knownFeatureStatsByLength = []string{
	"peak_latency",
	"peak_residual",
	"rebound_magnitude",
	"rebound_latency",
	"center_freq",
	"edge_freq_95",
	"erd_magnitude",
	"erd_duration",
	"ers_magnitude",
	"ers_duration",
	"percent_mean",
	"percent_std",
	"logratio_mean",
	"logratio_std",
	"db_mean",
	"db_std",
	"peak_freq",
	"peak_power",
	"peak_ratio",
	"peak_height",
	"power_ratio",
	"log_ratio",
	"latency_diff",
	"smallworld",
	"logratio",
	"bandwidth",
	"entropy",
	"slope",
	"mean",
	"std",
	"auc",
	"ptp",
	"geff",
	"clust",
	"lzc",
	"pe",
	"db",
}

type doseResponseFeatureMetadata struct {
	bands  map[string]struct{}
	scopes map[string]struct{}
	rois   map[string]struct{}
	stats  map[string]struct{}
}

func newDoseResponseFeatureMetadata() doseResponseFeatureMetadata {
	return doseResponseFeatureMetadata{
		bands:  make(map[string]struct{}),
		scopes: make(map[string]struct{}),
		rois:   make(map[string]struct{}),
		stats:  make(map[string]struct{}),
	}
}

func sortedKeys(set map[string]struct{}) []string {
	out := make([]string, 0, len(set))
	for k := range set {
		if strings.TrimSpace(k) == "" {
			continue
		}
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func parseDoseResponseFeatureColumn(category string, column string, knownSegments map[string]struct{}) (scope string, band string, roi string, stat string, ok bool) {
	prefix := strings.TrimSpace(category) + "_"
	if !strings.HasPrefix(column, prefix) {
		return "", "", "", "", false
	}
	parts := strings.Split(strings.TrimPrefix(column, prefix), "_")
	if len(parts) < 3 {
		return "", "", "", "", false
	}

	scopeIdx := -1
	for i, token := range parts {
		if _, exists := knownFeatureScopes[token]; exists {
			scopeIdx = i
			scope = token
			break
		}
	}
	if scopeIdx < 0 {
		return "", "", "", "", false
	}

	segIdx := -1
	for i := 0; i < scopeIdx; i++ {
		if _, exists := knownSegments[parts[i]]; exists {
			segIdx = i
			break
		}
	}
	if segIdx >= 0 && segIdx+1 < scopeIdx {
		band = strings.Join(parts[segIdx+1:scopeIdx], "_")
	} else if scopeIdx > 0 {
		band = parts[scopeIdx-1]
	}

	tailParts := parts[scopeIdx+1:]
	if len(tailParts) == 0 {
		return scope, band, "", "", true
	}
	tail := strings.Join(tailParts, "_")
	if scope == "global" {
		return scope, band, "", tail, true
	}

	for _, candidate := range knownFeatureStatsByLength {
		if tail == candidate {
			return scope, band, "", candidate, true
		}
		suffix := "_" + candidate
		if strings.HasSuffix(tail, suffix) {
			identifier := strings.TrimSuffix(tail, suffix)
			if scope == "roi" {
				roi = identifier
			}
			return scope, band, roi, candidate, true
		}
	}

	// Fallback when stat suffix is unknown: assume final token is stat.
	lastUnderscore := strings.LastIndex(tail, "_")
	if lastUnderscore <= 0 || lastUnderscore >= len(tail)-1 {
		return scope, band, "", tail, true
	}
	identifier := tail[:lastUnderscore]
	stat = tail[lastUnderscore+1:]
	if scope == "roi" {
		roi = identifier
	}
	return scope, band, roi, stat, true
}

func (m Model) doseResponseFeatureMetadata(categories []string) doseResponseFeatureMetadata {
	meta := newDoseResponseFeatureMetadata()
	if len(m.trialTableColumns) == 0 {
		return meta
	}

	catSet := make(map[string]struct{}, len(categories))
	for _, cat := range categories {
		c := strings.TrimSpace(cat)
		if c == "" {
			continue
		}
		catSet[c] = struct{}{}
	}

	segments := m.GetPlottingComparisonWindows()
	segmentSet := make(map[string]struct{}, len(segments))
	for _, seg := range segments {
		s := strings.TrimSpace(seg)
		if s != "" {
			segmentSet[s] = struct{}{}
		}
	}

	for _, col := range m.trialTableColumns {
		c := strings.TrimSpace(col)
		if c == "" {
			continue
		}
		catIdx := strings.Index(c, "_")
		if catIdx <= 0 {
			continue
		}
		category := c[:catIdx]
		if _, include := catSet[category]; !include {
			continue
		}
		scope, band, roi, stat, ok := parseDoseResponseFeatureColumn(category, c, segmentSet)
		if !ok {
			continue
		}
		if strings.TrimSpace(scope) != "" {
			meta.scopes[scope] = struct{}{}
		}
		if strings.TrimSpace(band) != "" {
			meta.bands[band] = struct{}{}
		}
		if strings.TrimSpace(roi) != "" {
			meta.rois[roi] = struct{}{}
		}
		if strings.TrimSpace(stat) != "" {
			meta.stats[stat] = struct{}{}
		}
	}
	return meta
}

func (m Model) GetDoseResponseBands(categories []string) []string {
	return sortedKeys(m.doseResponseFeatureMetadata(categories).bands)
}

func (m Model) GetDoseResponseScopes(categories []string) []string {
	return sortedKeys(m.doseResponseFeatureMetadata(categories).scopes)
}

func (m Model) GetDoseResponseROIs(categories []string) []string {
	meta := m.doseResponseFeatureMetadata(categories)
	if len(meta.rois) == 0 && len(m.discoveredROIs) > 0 {
		return append([]string{}, m.discoveredROIs...)
	}
	return sortedKeys(meta.rois)
}

func (m Model) GetDoseResponseStats(categories []string) []string {
	return sortedKeys(m.doseResponseFeatureMetadata(categories).stats)
}

func detectFeatureCategoriesFromColumns(columns []string) []string {
	// Keep in sync with eeg_pipeline.analysis.behavior.orchestration.FEATURE_COLUMN_PREFIXES
	featurePrefixes := []string{
		"power_",
		"connectivity_",
		"directedconnectivity_",
		"sourcelocalization_",
		"aperiodic_",
		"erp_",
		"itpc_",
		"pac_",
		"complexity_",
		"bursts_",
		"quality_",
		"erds_",
		"spectral_",
		"ratios_",
		"asymmetry_",
		"microstates_",
		"temporal_",
	}

	prefixSet := make(map[string]bool, len(featurePrefixes))
	for _, p := range featurePrefixes {
		prefixSet[p] = true
	}

	seen := make(map[string]bool, len(featurePrefixes))
	for _, col := range columns {
		c := strings.TrimSpace(col)
		if c == "" {
			continue
		}
		idx := strings.Index(c, "_")
		if idx <= 0 {
			continue
		}
		prefix := c[:idx+1]
		if !prefixSet[prefix] {
			continue
		}
		seen[prefix] = true
	}

	out := make([]string, 0, len(featurePrefixes))
	for _, p := range featurePrefixes {
		if seen[p] {
			out = append(out, strings.TrimSuffix(p, "_"))
		}
	}
	return out
}

// GetPlottingComparisonWindows returns windows for plotting comparison from computed feature data.
// If featureGroup is provided, returns only windows for that feature group.
func (m Model) GetPlottingComparisonWindows(featureGroup ...string) []string {
	if len(featureGroup) > 0 && featureGroup[0] != "" {
		// Prefer feature-specific windows (more accurate), but fall back to the global
		// discovered windows list so selection dropdowns still work even when the
		// executor didn't provide per-feature windows.
		//
		// This is a UI convenience fallback only; downstream plotting code will still
		// validate window availability when reading feature files.
		if m.availableWindowsByFeature != nil {
			if windows, ok := m.availableWindowsByFeature[featureGroup[0]]; ok && len(windows) > 0 {
				return windows
			}
		}
		return m.availableWindows
	}
	return m.availableWindows
}

// getPlotByID returns the PlotItem for the given plot ID.
func (m Model) getPlotByID(plotID string) PlotItem {
	for _, plot := range m.plotItems {
		if plot.ID == plotID {
			return plot
		}
	}
	return PlotItem{Group: ""}
}

// getFeatureGroupForPlot returns the feature group name used in column names for a given plot.
// Maps plot Group/ID to the actual feature group in NamingSchema (e.g., "phase" -> "itpc" or "pac").
func (m Model) getFeatureGroupForPlot(plotID string) string {
	plot := m.getPlotByID(plotID)

	// Map plot IDs to their feature groups
	switch plotID {
	case "itpc_topomaps", "itpc_by_condition":
		return "itpc"
	case "pac_by_condition":
		return "pac"
	}

	// For most plots, Group matches the feature group name
	// But handle special cases
	switch plot.Group {
	case "phase":
		// Phase group contains both ITPC and PAC - determine from plot ID
		if strings.HasPrefix(plotID, "itpc_") {
			return "itpc"
		}
		if strings.HasPrefix(plotID, "pac_") {
			return "pac"
		}
		return "itpc" // Default for phase plots
	}

	// Default: use Group as feature group (works for power, connectivity, aperiodic, etc.)
	return plot.Group
}

// GetPlottingComparisonColumnValues returns values for a column in plotting comparison from trial table.
// Feature plots use trial-level columns from events.tsv for condition comparisons.
func (m Model) GetPlottingComparisonColumnValues(column string) []string {
	return m.GetDiscoveredColumnValues(column)
}

// GetSourcePlotConditions returns detected condition labels for source plot filtering.
func (m Model) GetSourcePlotConditions() []string {
	conditionColumn := strings.TrimSpace(m.sourceLocContrastCondition)
	if conditionColumn == "" {
		return nil
	}
	return m.GetDiscoveredColumnValues(conditionColumn)
}

// GetSourcePlotBands returns the available frequency band keys for source plot band selection.
// Derived from the configured bands (same set used by all other pipelines).
func (m Model) GetSourcePlotBands() []string {
	out := make([]string, 0, len(m.bands))
	for _, b := range m.bands {
		if key := strings.TrimSpace(b.Key); key != "" {
			out = append(out, key)
		}
	}
	return out
}

// SetFmriDiscoveredColumns sets the columns and values discovered from fMRI events files
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
