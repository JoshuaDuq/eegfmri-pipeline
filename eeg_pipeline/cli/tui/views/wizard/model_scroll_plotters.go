package wizard

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/eeg-pipeline/tui/types"
)

// Scroll calculations, plotting availability summaries, and plotter discovery.

func (m *Model) UpdateComputationOffset() {
	// Match overhead with renderComputationSelection (12 lines)
	overheadLines := 12
	maxVisibleLines := m.height - overheadLines
	if maxVisibleLines < minVisibleLines {
		maxVisibleLines = minVisibleLines
	}

	totalLines := len(m.computations)
	m.computationOffset = calculateScrollOffset(
		m.computationCursor,
		m.computationOffset,
		totalLines,
		maxVisibleLines,
	)
}

// UpdateAdvancedOffset calculates and updates the scrolling offset for advanced config lists.
func (m *Model) UpdateAdvancedOffset() {
	// Use a fallback height if terminal size not yet received
	effectiveHeight := m.height
	if effectiveHeight <= 0 {
		effectiveHeight = defaultTerminalHeight
	}

	// Overhead must match each pipeline's render function.
	overheadLines := configOverhead
	maxLines := effectiveHeight - overheadLines
	if maxLines < minVisibleLines {
		maxLines = minVisibleLines
	}

	totalLines := 0
	cursorLine := 0

	switch m.Pipeline {
	case types.PipelineBehavior:
		options := m.getBehaviorOptions()
		totalLines = len(options)
		// Note: expanded list items are rendered inline, so cursorLine stays as advancedCursor
		cursorLine = m.advancedCursor

	case types.PipelineFeatures:
		options := m.getFeaturesOptions()
		totalLines = len(options)
		cursorLine = m.advancedCursor

		if m.expandedOption == expandedConnectivityMeasures {
			expandedIdx := -1
			for i, opt := range options {
				if opt == optConnectivity {
					expandedIdx = i
					break
				}
			}
			if expandedIdx >= 0 {
				totalLines += len(connectivityMeasures)
				cursorLine = expandedIdx + 1 + m.subCursor
			}
		}

	case types.PipelinePlotting:
		rows := m.getPlottingAdvancedRows()
		totalLines = len(rows)
		cursorLine = m.advancedCursor

	case types.PipelinePreprocessing:
		options := m.getPreprocessingOptions()
		totalLines = len(options)
		cursorLine = m.advancedCursor

	case types.PipelineFmri:
		options := m.getFmriPreprocessingOptions()
		totalLines = len(options)
		cursorLine = m.advancedCursor

	case types.PipelineFmriAnalysis:
		options := m.getFmriAnalysisOptions()
		totalLines = len(options)
		cursorLine = m.advancedCursor

	case types.PipelineML:
		options := m.getMLOptions()
		lineIdx := 0
		for i, opt := range options {
			if !isMLRenderedOption(opt) {
				continue
			}
			if i == m.advancedCursor {
				cursorLine = lineIdx
			}
			lineIdx++
			if m.shouldRenderExpandedListAfterOption(opt) {
				lineIdx += m.getExpandedListLength()
			}
		}
		totalLines = lineIdx

	default:
		totalLines = 0
		cursorLine = 0
	}

	if totalLines <= 0 {
		m.advancedOffset = 0
		return
	}

	m.advancedOffset = calculateScrollOffset(
		cursorLine,
		m.advancedOffset,
		totalLines,
		maxLines,
	)
}

// UpdatePlotOffset calculates and updates the scrolling offset for the plots list
func (m *Model) UpdatePlotOffset() {
	// Match overhead with renderPlotSelection (10-14 lines)
	overheadLines := 10
	maxLines := m.height - overheadLines
	if maxLines < minVisibleLines {
		maxLines = minVisibleLines
	}

	// Reconstruct the list logic to find cursor position
	currentGroup := ""
	lineIdx := 0
	cursorLine := -1

	for i, plot := range m.plotItems {
		if !m.IsPlotVisibleForSelection(plot) {
			continue
		}

		if plot.Group != currentGroup {
			lineIdx++ // Group header
			currentGroup = plot.Group
		}

		if i == m.plotCursor {
			cursorLine = lineIdx
		}
		lineIdx++ // Item line
	}

	if cursorLine == -1 {
		return
	}

	m.plotOffset = calculateScrollOffset(
		cursorLine,
		m.plotOffset,
		lineIdx,
		maxLines,
	)
}

func (m Model) selectedFeaturePlotterCategories() []string {
	var ordered []string
	seen := make(map[string]bool)
	for i, plot := range m.plotItems {
		if !m.plotSelected[i] || !m.IsPlotVisibleForSelection(plot) {
			continue
		}
		if plot.Group != "features" {
			continue
		}
		id := strings.TrimSpace(plot.ID)
		if !strings.HasPrefix(id, "features_") {
			continue
		}
		cat := strings.TrimPrefix(id, "features_")
		if cat == "" || seen[cat] {
			continue
		}
		seen[cat] = true
		ordered = append(ordered, cat)
	}
	return ordered
}

func (m Model) featurePlotterItems() []PlotterInfo {
	if m.featurePlotters == nil {
		return nil
	}
	var items []PlotterInfo
	for _, category := range m.selectedFeaturePlotterCategories() {
		items = append(items, m.featurePlotters[category]...)
	}
	return items
}

func (m *Model) UpdateFeaturePlotterOffset() {
	// Match overhead with renderFeaturePlotterSelection (10 lines)
	overheadLines := 10
	maxLines := m.height - overheadLines
	if maxLines < minVisibleLines {
		maxLines = minVisibleLines
	}

	items := m.featurePlotterItems()
	if len(items) == 0 {
		m.featurePlotterOffset = 0
		return
	}

	currentCategory := ""
	lineIdx := 0
	cursorLine := -1
	for i, p := range items {
		if p.Category != currentCategory {
			lineIdx++
			currentCategory = p.Category
		}
		if i == m.featurePlotterCursor {
			cursorLine = lineIdx
		}
		lineIdx++
	}
	if cursorLine < 0 {
		return
	}

	m.featurePlotterOffset = calculateScrollOffset(
		cursorLine,
		m.featurePlotterOffset,
		lineIdx,
		maxLines,
	)
}

// calculateScrollOffset computes the scroll offset to keep the cursor visible.
// It ensures the cursor stays within the visible area when scrolling.
func calculateScrollOffset(cursorLine, currentOffset, totalLines, maxVisibleLines int) int {
	if totalLines <= 0 {
		return 0
	}

	// Everything fits — no scrolling needed, reset any stale offset
	if totalLines <= maxVisibleLines {
		return 0
	}

	// Clamp cursor to valid range
	if cursorLine < 0 {
		cursorLine = 0
	}
	if cursorLine >= totalLines {
		cursorLine = totalLines - 1
	}

	// Adjust offset to keep cursor visible
	if cursorLine < currentOffset {
		currentOffset = cursorLine
	} else if cursorLine >= currentOffset+maxVisibleLines {
		currentOffset = cursorLine - maxVisibleLines + 1
	}

	// Ensure offset is non-negative
	if currentOffset < 0 {
		currentOffset = 0
	}

	// Ensure offset doesn't exceed maximum
	maxOffset := totalLines - maxVisibleLines
	if maxOffset > 0 && currentOffset > maxOffset {
		currentOffset = maxOffset
	}

	return currentOffset
}

func (m Model) plotCountsForGroup(group string) (total int, selected int) {
	for i, plot := range m.plotItems {
		if strings.EqualFold(plot.Group, group) {
			total++
			if m.plotSelected[i] {
				selected++
			}
		}
	}
	return total, selected
}

func (m Model) plotAvailabilitySummary(plot PlotItem) (int, int, map[string]int) {
	missing := make(map[string]int)
	total := 0
	available := 0

	for _, s := range m.subjects {
		if !m.subjectSelected[s.ID] {
			continue
		}
		total++

		hasEpochs := !plot.RequiresEpochs || s.HasEpochs
		hasFeatures := !plot.RequiresFeatures || s.HasFeatures
		hasStats := !plot.RequiresStats || s.HasStats
		isAvailable := hasEpochs && hasFeatures && hasStats

		if !hasEpochs {
			missing["epochs"]++
		}
		if !hasFeatures {
			missing["features"]++
		}
		if !hasStats {
			missing["stats"]++
		}

		if isAvailable {
			available++
		}
	}

	return available, total, missing
}

func (m Model) discoverTemporalTopomapsStatsFeatureFolders() ([]string, error) {
	derivRoot := strings.TrimSpace(m.derivRoot)
	if derivRoot == "" {
		return nil, fmt.Errorf("deriv_root is not set")
	}

	selectedSubjects := make([]string, 0, len(m.subjects))
	for _, s := range m.subjects {
		if m.subjectSelected[s.ID] {
			selectedSubjects = append(selectedSubjects, s.ID)
		}
	}
	if len(selectedSubjects) == 0 {
		return nil, fmt.Errorf("no subjects selected")
	}

	var intersection map[string]struct{}
	for _, subjID := range selectedSubjects {
		statsDir := filepath.Join(derivRoot, fmt.Sprintf("sub-%s", subjID), "eeg", "stats")
		kindDirs, _ := filepath.Glob(filepath.Join(statsDir, "temporal_correlations*"))
		perSubject := make(map[string]struct{})

		for _, kindDir := range kindDirs {
			entries, err := os.ReadDir(kindDir)
			if err != nil {
				continue
			}
			for _, entry := range entries {
				if !entry.IsDir() {
					continue
				}
				featureFolder := entry.Name()
				featureDir := filepath.Join(kindDir, featureFolder)
				matches, _ := filepath.Glob(filepath.Join(featureDir, "temporal_correlations_by_condition*.npz"))
				if len(matches) > 0 {
					perSubject[featureFolder] = struct{}{}
				}
			}
		}

		if intersection == nil {
			intersection = perSubject
			continue
		}
		for k := range intersection {
			if _, ok := perSubject[k]; !ok {
				delete(intersection, k)
			}
		}
	}

	if len(intersection) == 0 {
		return nil, fmt.Errorf("no temporal_correlations feature folders found (expected NPZ in stats/temporal_correlations*/<feature>/)")
	}

	out := make([]string, 0, len(intersection))
	for k := range intersection {
		out = append(out, k)
	}
	sort.Strings(out)
	return out, nil
}

///////////////////////////////////////////////////////////////////
// Setters
///////////////////////////////////////////////////////////////////
