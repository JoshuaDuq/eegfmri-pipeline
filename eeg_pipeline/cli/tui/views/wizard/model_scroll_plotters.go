package wizard

import (
	"github.com/eeg-pipeline/tui/types"
)

// Scroll calculations for wizard option lists.

func (m Model) advancedRenderedLines(options []optionType, shouldRender func(optionType) bool) (totalLines, cursorLine int) {
	lineIdx := 0
	cursorLine = 0

	for i, opt := range options {
		if !shouldRender(opt) {
			continue
		}

		if i == m.advancedCursor {
			cursorLine = lineIdx
			if m.shouldRenderExpandedListAfterOption(opt) && m.subCursor >= 0 {
				maxSubCursor := m.getExpandedListLength() - 1
				if maxSubCursor >= 0 {
					cursorLine += 1 + min(m.subCursor, maxSubCursor)
				}
			}
		}

		lineIdx++
		if m.shouldRenderExpandedListAfterOption(opt) {
			lineIdx += m.getExpandedListLength()
		}
	}

	return lineIdx, cursorLine
}

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
	totalLines := 0
	cursorLine := 0

	switch m.Pipeline {
	case types.PipelineBehavior:
		options := m.getBehaviorOptions()
		totalLines, cursorLine = m.advancedRenderedLines(options, func(optionType) bool { return true })

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

	case types.PipelinePreprocessing:
		options := m.getPreprocessingOptions()
		totalLines, cursorLine = m.advancedRenderedLines(options, func(optionType) bool { return true })

	case types.PipelineFmri:
		options := m.getFmriPreprocessingOptions()
		totalLines, cursorLine = m.advancedRenderedLines(options, func(optionType) bool { return true })

	case types.PipelineFmriAnalysis:
		options := m.getFmriAnalysisOptions()
		totalLines, cursorLine = m.advancedRenderedLines(options, func(optionType) bool { return true })

	case types.PipelineML:
		options := m.getMLOptions()
		totalLines, cursorLine = m.advancedRenderedLines(options, isMLRenderedOption)

	default:
		totalLines = 0
		cursorLine = 0
	}

	if totalLines <= 0 {
		m.advancedOffset = 0
		return
	}

	maxLines := scrollableVisibleLines(totalLines, m.availableAdvancedContentHeight())
	m.advancedOffset = calculateScrollOffset(
		cursorLine,
		m.advancedOffset,
		totalLines,
		maxLines,
	)
}

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
