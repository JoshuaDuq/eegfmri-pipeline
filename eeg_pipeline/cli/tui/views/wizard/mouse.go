package wizard

import (
	"regexp"
	"strings"

	"github.com/eeg-pipeline/tui/types"

	tea "github.com/charmbracelet/bubbletea"
)

var mouseANSIPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func stripANSI(s string) string {
	return mouseANSIPattern.ReplaceAllString(s, "")
}

func (m Model) handleMouse(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	if m.showHelp || m.IsEditing() {
		return m, nil
	}

	switch msg.Button {
	case tea.MouseButtonWheelUp:
		m.handleUp()
		return m, nil
	case tea.MouseButtonWheelDown:
		m.handleDown()
		return m, nil
	}

	if msg.Action != tea.MouseActionMotion &&
		(msg.Action != tea.MouseActionPress || msg.Button != tea.MouseButtonLeft) {
		return m, nil
	}

	activate := msg.Action == tea.MouseActionPress && msg.Button == tea.MouseButtonLeft
	w, h := m.effectiveDimensions()
	containerW, containerH := m.containerDimensions(w, h)
	innerW := containerW - containerPadH*2 - containerBorder
	if innerW < 1 {
		innerW = 1
	}
	m.contentWidth = innerW
	_ = containerH

	line := m.viewLineAt(msg.Y)
	if strings.TrimSpace(line) == "" {
		return m, nil
	}

	contentLines := strings.Split(stripANSI(m.renderStepContent()), "\n")
	if subIdx := m.matchExpandedListLine(line, contentLines); subIdx >= 0 && m.expandedOption >= 0 {
		m.subCursor = subIdx
		if activate {
			m.handleSpace()
		}
		return m, nil
	}

	switch m.CurrentStep {
	case types.StepSelectMode:
		if idx := m.matchModeLine(line); idx >= 0 {
			m.modeIndex = idx
			if activate {
				return m.handleEnter()
			}
			return m, nil
		}
	case types.StepSelectComputations:
		if idx := m.matchComputationLine(line); idx >= 0 {
			m.computationCursor = idx
			if activate {
				m.handleSpace()
			}
		}
	case types.StepConfigureOptions, types.StepSelectPlotCategories:
		if m.showGlobalStyling && m.CurrentStep == types.StepSelectPlotCategories {
			if idx := m.matchAdvancedRowIndex(line, contentLines); idx >= 0 {
				m.globalStylingCursor = idx
				if activate {
					m.handleSpace()
				}
			}
			return m, nil
		}
		if idx := m.matchCategoryLine(line); idx >= 0 {
			m.categoryIndex = idx
			if activate {
				m.handleSpace()
			}
		}
	case types.StepSelectSubjects:
		if m.matchSubjectFilterLine(line) {
			if activate {
				m.filteringSubject = true
			}
			return m, nil
		}
		if scope := m.matchSubjectScopeSelection(line, msg.X); scope >= 0 {
			switch m.Pipeline {
			case types.PipelineML:
				m.mlScope = MLCVScope(scope)
			case types.PipelinePlotting:
				m.plottingScope = PlottingScope(scope)
			}
			return m, nil
		}
		if idx := m.matchSubjectLine(line); idx >= 0 {
			m.subjectCursor = idx
			if activate {
				m.handleSpace()
			}
		}
	case types.StepSelectBands:
		if idx := m.matchBandLine(line); idx >= 0 {
			m.bandCursor = idx
			if activate {
				m.handleSpace()
			}
		}
	case types.StepSelectROIs:
		if idx := m.matchROILine(line); idx >= 0 {
			m.roiCursor = idx
			if activate {
				m.handleSpace()
			}
		}
	case types.StepSelectSpatial:
		if idx := m.matchSpatialLine(line); idx >= 0 {
			m.spatialCursor = idx
			if activate {
				m.handleSpace()
			}
		}
	case types.StepSelectPreprocessingStages:
		if idx := m.matchPreprocessingStageLine(line); idx >= 0 {
			m.prepStageCursor = idx
			if activate {
				m.prepStageSelected[idx] = !m.prepStageSelected[idx]
			}
		}
	case types.StepSelectFeatureFiles:
		if idx := m.matchFeatureFileLine(line); idx >= 0 {
			m.featureFileCursor = idx
			if activate {
				m.handleSpace()
			}
		}
	case types.StepSelectPlots:
		if idx := m.matchPlotLine(line); idx >= 0 {
			m.plotCursor = idx
			if activate {
				m.handleSpace()
			}
		}
	case types.StepSelectFeaturePlotters:
		if idx := m.matchFeaturePlotterLine(line); idx >= 0 {
			m.featurePlotterCursor = idx
			if activate {
				m.handleSpace()
			}
		}
	case types.StepPlotConfig:
		if idx := m.matchPlotConfigLine(line); idx >= 0 {
			m.plotConfigCursor = idx
			if activate {
				m.handleSpace()
			}
		}
	case types.StepTimeRange:
		if idx := m.matchTimeRangeLine(line); idx >= 0 {
			m.timeRangeCursor = idx
			if activate {
				m.handleSpace()
			}
		}
	case types.StepPreprocessingFiltering:
		if idx := m.matchPreprocessingFilteringLine(line); idx >= 0 {
			m.advancedCursor = idx
			if activate {
				m.startNumberEdit()
			}
		}
	case types.StepPreprocessingICA:
		if idx := m.matchPreprocessingICALine(line); idx >= 0 {
			m.advancedCursor = idx
			if activate {
				if handled := m.handlePreprocessingICAMouse(line); handled {
					return m, nil
				}
			}
		}
	case types.StepPreprocessingEpochs:
		if idx := m.matchPreprocessingEpochLine(line); idx >= 0 {
			m.advancedCursor = idx
			if activate {
				if handled := m.handlePreprocessingEpochMouse(line); handled {
					return m, nil
				}
			}
		}
	case types.StepAdvancedConfig:
		if idx, subIdx := m.matchAdvancedLine(line, contentLines); idx >= 0 {
			m.advancedCursor = idx
			if subIdx >= 0 {
				m.subCursor = subIdx
			}
			if activate {
				m.handleSpace()
			}
		}
	default:
		return m, nil
	}

	return m, nil
}

func (m Model) viewLineAt(y int) string {
	if y < 0 {
		return ""
	}
	lines := strings.Split(stripANSI(m.View()), "\n")
	if y >= len(lines) {
		return ""
	}
	return lines[y]
}

func (m Model) matchModeLine(line string) int {
	return indexOfContains(line, m.modeOptions)
}

func (m Model) matchComputationLine(line string) int {
	names := make([]string, len(m.computations))
	for i, comp := range m.computations {
		names[i] = comp.Name
	}
	return indexOfContains(line, names)
}

func (m Model) matchCategoryLine(line string) int {
	names := make([]string, len(m.categories))
	for i, cat := range m.categories {
		names[i] = featureCategoryLabel(cat)
	}
	return indexOfContains(line, names)
}

func (m Model) matchSubjectLine(line string) int {
	names := make([]string, len(m.subjects))
	for i, subj := range m.subjects {
		names[i] = subj.ID
	}
	return indexOfContains(line, names)
}

func (m Model) matchSubjectFilterLine(line string) bool {
	text := stripANSI(line)
	return strings.Contains(text, "Filter:") || strings.Contains(text, "filter:")
}

func (m Model) matchSubjectScopeSelection(line string, x int) int {
	text := stripANSI(line)
	switch m.Pipeline {
	case types.PipelineML:
		if hitRange(text, x, "Group (LOSO)") {
			return int(MLCVScopeGroup)
		}
		if hitRange(text, x, "Subject (within)") {
			return int(MLCVScopeSubject)
		}
	case types.PipelinePlotting:
		if hitRange(text, x, "Group") {
			return int(PlottingScopeGroup)
		}
		if hitRange(text, x, "Subject") {
			return int(PlottingScopeSubject)
		}
	}
	return -1
}

func (m Model) matchBandLine(line string) int {
	names := make([]string, len(m.bands))
	for i, band := range m.bands {
		names[i] = band.Name
	}
	return indexOfContains(line, names)
}

func (m Model) matchROILine(line string) int {
	names := make([]string, len(m.rois))
	for i, roi := range m.rois {
		names[i] = roi.Name
	}
	return indexOfContains(line, names)
}

func (m Model) matchSpatialLine(line string) int {
	names := make([]string, len(spatialModes))
	for i, mode := range spatialModes {
		names[i] = mode.Name
	}
	return indexOfContains(line, names)
}

func (m Model) matchPreprocessingStageLine(line string) int {
	names := make([]string, len(m.prepStages))
	for i, stage := range m.prepStages {
		names[i] = stage.Name
	}
	return indexOfContains(line, names)
}

func (m Model) matchFeatureFileLine(line string) int {
	names := make([]string, len(m.GetApplicableFeatureFiles()))
	for i, file := range m.GetApplicableFeatureFiles() {
		names[i] = file.Name
	}
	return indexOfContains(line, names)
}

func (m Model) matchPlotLine(line string) int {
	names := make([]string, len(m.plotItems))
	for i, plot := range m.plotItems {
		names[i] = plot.Name
	}
	return indexOfContains(line, names)
}

func (m Model) matchFeaturePlotterLine(line string) int {
	items := m.featurePlotterItems()
	names := make([]string, len(items))
	for i, item := range items {
		names[i] = item.Name
	}
	return indexOfContains(line, names)
}

func (m Model) matchPlotConfigLine(line string) int {
	labels := []string{
		"PNG",
		"SVG",
		"PDF",
		"Figure DPI",
		"Savefig DPI",
		"Shared Colorbar",
		"Overwrite",
	}
	return indexOfContains(line, labels)
}

func (m Model) matchTimeRangeLine(line string) int {
	text := strings.TrimSpace(stripANSI(line))
	if strings.Contains(text, "Resting State") {
		return 0
	}

	needle := strings.TrimSpace(text)
	for i, tr := range m.TimeRanges {
		names := []string{tr.Name, "(none)"}
		for _, name := range names {
			if strings.TrimSpace(name) == "" {
				continue
			}
			if strings.Contains(needle, name) {
				if m.timeRangeShowsRestToggle() {
					return i + 1
				}
				return i
			}
		}
	}
	return -1
}

func (m Model) matchPreprocessingFilteringLine(line string) int {
	labels := []string{
		"Resample Freq",
		"High-Pass Freq",
		"Low-Pass Freq",
		"Notch Freq",
		"Line Freq",
	}
	return indexOfContains(line, labels)
}

func (m Model) matchPreprocessingICALine(line string) int {
	labels := []string{
		"Use ICALabel",
		"ICA Method",
		"Components",
		"Prob Threshold",
		"Labels to Keep",
		"Keep MNE-BIDS Bads",
	}
	return indexOfContains(line, labels)
}

func (m Model) matchPreprocessingEpochLine(line string) int {
	labels := []string{
		"Tmin",
		"Tmax",
		"Correction",
		"Window",
		"Reject Threshold",
	}
	return indexOfContains(line, labels)
}

func (m *Model) handlePreprocessingICAMouse(line string) bool {
	label := strings.TrimSpace(stripANSI(line))
	switch {
	case strings.Contains(label, "Use ICALabel"):
		m.prepUseIcalabel = !m.prepUseIcalabel
		return true
	case strings.Contains(label, "ICA Method"):
		m.prepICAAlgorithm = (m.prepICAAlgorithm + 1) % 4
		return true
	case strings.Contains(label, "Components"):
		m.advancedCursor = 2
		m.startNumberEdit()
		return true
	case strings.Contains(label, "Prob Threshold"):
		m.advancedCursor = 3
		m.startNumberEdit()
		return true
	case strings.Contains(label, "Labels to Keep"):
		m.advancedCursor = 4
		m.startTextEdit(textFieldIcaLabelsToKeep)
		return true
	case strings.Contains(label, "Keep MNE-BIDS Bads"):
		m.prepKeepMnebidsBads = !m.prepKeepMnebidsBads
		return true
	default:
		return false
	}
}

func (m *Model) handlePreprocessingEpochMouse(line string) bool {
	label := strings.TrimSpace(stripANSI(line))
	switch {
	case strings.Contains(label, "Tmin"):
		m.advancedCursor = 0
		m.startNumberEdit()
		return true
	case strings.Contains(label, "Tmax"):
		m.advancedCursor = 1
		m.startNumberEdit()
		return true
	case strings.Contains(label, "Correction"):
		m.prepEpochsNoBaseline = !m.prepEpochsNoBaseline
		return true
	case strings.Contains(label, "Window"):
		m.advancedCursor = 3
		m.startNumberEdit()
		return true
	case strings.Contains(label, "Reject Threshold"):
		m.advancedCursor = 4
		m.startNumberEdit()
		return true
	default:
		return false
	}
}

func (m Model) matchAdvancedRowIndex(line string, contentLines []string) int {
	rowIdx, _ := m.matchAdvancedLine(line, contentLines)
	return rowIdx
}

func (m Model) matchExpandedListLine(line string, contentLines []string) int {
	_, subIdx := m.matchAdvancedLine(line, contentLines)
	return subIdx
}

func (m Model) matchAdvancedLine(line string, contentLines []string) (int, int) {
	clicked := strings.TrimSpace(stripANSI(line))
	if clicked == "" {
		return -1, -1
	}

	rowIndex := -1
	subIndex := -1
	for _, contentLine := range contentLines {
		raw := stripANSI(contentLine)
		if strings.TrimSpace(raw) == "" {
			continue
		}
		if isExpandedListLine(raw) {
			subIndex++
			if sameMouseLine(raw, clicked) {
				if rowIndex < 0 {
					return -1, -1
				}
				return rowIndex, subIndex
			}
			continue
		}
		subIndex = -1
		if isAdvancedInteractiveLine(raw) {
			rowIndex++
		}
		if sameMouseLine(raw, clicked) {
			if !isAdvancedInteractiveLine(raw) {
				return -1, -1
			}
			return rowIndex, -1
		}
	}
	return -1, -1
}

func isExpandedListLine(raw string) bool {
	return strings.HasPrefix(raw, "      ")
}

func isAdvancedInteractiveLine(raw string) bool {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return false
	}
	if strings.Contains(trimmed, "toggle/expand") ||
		strings.Contains(trimmed, "proceed") ||
		strings.Contains(trimmed, "Navigate") ||
		strings.Contains(trimmed, "Back") {
		return false
	}
	if strings.HasPrefix(trimmed, "│") || strings.HasPrefix(trimmed, "▏") {
		return false
	}
	if strings.HasPrefix(raw, "      ") {
		return false
	}
	// `▸` (legacy) and `›` (current SelectedMark) both indicate the
	// cursor/selection affordance; `▾` signals an expanded group. Any of
	// these marks — along with `:` / `·` used in key/value lines — flag
	// the row as interactive for mouse hit-testing.
	if strings.ContainsAny(trimmed, ":·▸▾›") {
		return true
	}
	return strings.HasPrefix(trimmed, "›")
}

func sameMouseLine(a, b string) bool {
	a = normalizeMouseText(a)
	b = normalizeMouseText(b)
	if a == b {
		return true
	}
	return strings.Contains(a, b) || strings.Contains(b, a)
}

func indexOfContains(line string, needles []string) int {
	target := normalizeMouseText(line)
	for i, needle := range needles {
		if needle == "" {
			continue
		}
		if strings.Contains(target, normalizeMouseText(needle)) {
			return i
		}
	}
	return -1
}

func normalizeMouseText(s string) string {
	s = stripANSI(s)
	s = strings.NewReplacer(
		"│", " ",
		"─", " ",
		"━", " ",
		"·", " ",
		"•", " ",
		"›", " ",
		"‹", " ",
	).Replace(s)
	return strings.Join(strings.Fields(s), " ")
}

func hitRange(text string, x int, needle string) bool {
	start := strings.Index(text, needle)
	if start < 0 {
		return false
	}
	return x >= start && x < start+len(needle)
}
