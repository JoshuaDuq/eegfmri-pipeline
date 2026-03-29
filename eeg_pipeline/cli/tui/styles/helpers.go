package styles

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/reflow/truncate"
)

type ListLayout struct {
	MaxItems     int
	StartIdx     int
	EndIdx       int
	ShowScrollUp bool
	ShowScrollDn bool
	TotalItems   int
	CursorIdx    int
}

func CalculateListLayout(termHeight, cursorIdx, totalItems, headerRows int) ListLayout {
	availableRows := calculateAvailableRows(termHeight, headerRows)

	if totalItems <= availableRows {
		return createFullViewLayout(availableRows, cursorIdx, totalItems)
	}

	startIdx := calculateStartIndex(cursorIdx, totalItems, availableRows)
	startIdx = clampStartIndex(startIdx, totalItems, availableRows)
	endIdx := clampEndIndex(startIdx, availableRows, totalItems)

	return ListLayout{
		MaxItems:     availableRows,
		StartIdx:     startIdx,
		EndIdx:       endIdx,
		ShowScrollUp: startIdx > 0,
		ShowScrollDn: endIdx < totalItems,
		TotalItems:   totalItems,
		CursorIdx:    cursorIdx,
	}
}

func calculateAvailableRows(termHeight, headerRows int) int {
	availableRows := termHeight - headerRows
	if availableRows < MinListItems {
		return MinListItems
	}
	return availableRows
}

func createFullViewLayout(availableRows, cursorIdx, totalItems int) ListLayout {
	return ListLayout{
		MaxItems:     availableRows,
		StartIdx:     0,
		EndIdx:       totalItems,
		ShowScrollUp: false,
		ShowScrollDn: false,
		TotalItems:   totalItems,
		CursorIdx:    cursorIdx,
	}
}

func calculateStartIndex(cursorIdx, totalItems, availableRows int) int {
	if cursorIdx < ListScrollMargin {
		return 0
	}
	if cursorIdx >= totalItems-ListScrollMargin {
		return totalItems - availableRows
	}
	return cursorIdx - availableRows/2
}

func clampStartIndex(startIdx, totalItems, availableRows int) int {
	if startIdx < 0 {
		return 0
	}
	maxStart := totalItems - availableRows
	if startIdx > maxStart {
		return maxStart
	}
	return startIdx
}

func clampEndIndex(startIdx, availableRows, totalItems int) int {
	endIdx := startIdx + availableRows
	if endIdx > totalItems {
		return totalItems
	}
	return endIdx
}

func RenderScrollUpIndicator(count int) string {
	if count <= 0 {
		return ""
	}
	arrow := lipgloss.NewStyle().Foreground(Primary).Bold(true).Render("  ▲")
	text := lipgloss.NewStyle().Foreground(TextDim).Render(fmt.Sprintf(" %d more", count))
	return arrow + text
}

func RenderScrollDownIndicator(count int) string {
	if count <= 0 {
		return ""
	}
	arrow := lipgloss.NewStyle().Foreground(Primary).Bold(true).Render("  ▼")
	text := lipgloss.NewStyle().Foreground(TextDim).Render(fmt.Sprintf(" %d more", count))
	return arrow + text
}

// RenderScrollTrack renders a compact vertical scrollbar track showing thumb position.
// current is the top visible item index, visible is the viewport height in items,
// total is the total item count. Track is trackHeight chars tall.
func RenderScrollTrack(current, visible, total, trackHeight int) string {
	if total <= visible || trackHeight < 2 {
		return ""
	}

	thumbSize := max(1, trackHeight*visible/total)
	maxOffset := total - visible
	thumbPos := 0
	if maxOffset > 0 {
		thumbPos = current * (trackHeight - thumbSize) / maxOffset
	}
	if thumbPos+thumbSize > trackHeight {
		thumbPos = trackHeight - thumbSize
	}

	trackStyle := lipgloss.NewStyle().Foreground(Border)
	thumbStyle := lipgloss.NewStyle().Foreground(Primary)

	var sb strings.Builder
	for i := 0; i < trackHeight; i++ {
		if i >= thumbPos && i < thumbPos+thumbSize {
			sb.WriteString(thumbStyle.Render("┃"))
		} else {
			sb.WriteString(trackStyle.Render("│"))
		}
		if i < trackHeight-1 {
			sb.WriteString("\n")
		}
	}
	return sb.String()
}

func IsTerminalTooSmall(width, height int) bool {
	return width < MinTerminalWidth || height < MinTerminalHeight
}

func RenderTerminalTooSmall(width, height int) string {
	msg := lipgloss.NewStyle().
		Foreground(Warning).
		Bold(true).
		Render("⚠ Terminal too small")

	hint := lipgloss.NewStyle().
		Foreground(TextDim).
		Render(fmt.Sprintf("\nResize to at least %dx%d", MinTerminalWidth, MinTerminalHeight))

	current := lipgloss.NewStyle().
		Foreground(Muted).
		Render(fmt.Sprintf("\nCurrent: %dx%d", width, height))

	return msg + hint + current
}

// RenderCursor returns the styled list/focus cursor.
func RenderCursor() string {
	return lipgloss.NewStyle().Foreground(Primary).Bold(true).Render(SelectedMark + " ")
}

// RenderCursorOptional returns the cursor when visible, or matching whitespace for alignment.
func RenderCursorOptional(visible bool) string {
	if visible {
		return RenderCursor()
	}
	return "  "
}

// RenderFooterSeparator returns the styled footer hint separator (e.g. "  │  ").
func RenderFooterSeparator() string {
	return lipgloss.NewStyle().Foreground(Secondary).Render(FooterHintSeparator)
}

func RenderKeyHint(key, label string) string {
	return FooterKeyPrimaryStyle.Render(key) + " " + FooterLabelPrimaryStyle.Render(label)
}

func RenderKeyHintSecondary(key, label string) string {
	return FooterKeySecondaryStyle.Render(key) + " " + FooterLabelSecondaryStyle.Render(label)
}

// RenderHeaderSeparator returns a styled horizontal line (e.g. for under titles).
func RenderHeaderSeparator(width int) string {
	if width <= 0 {
		return ""
	}
	return HeaderLineStyle.Render(strings.Repeat(HeaderSeparatorChar, width))
}

func RenderCheckbox(checked, focused bool) string {
	if checked && focused {
		return lipgloss.NewStyle().Foreground(Primary).Bold(true).Render("▣")
	}
	if checked {
		return lipgloss.NewStyle().Foreground(Success).Render("▣")
	}
	if focused {
		return lipgloss.NewStyle().Foreground(Primary).Bold(true).Render("□")
	}
	return lipgloss.NewStyle().Foreground(Muted).Render("□")
}

func RenderRadio(selected, focused bool) string {
	if selected && focused {
		return lipgloss.NewStyle().Foreground(Primary).Bold(true).Render("◉")
	}
	if selected {
		return lipgloss.NewStyle().Foreground(Primary).Render("◉")
	}
	if focused {
		return lipgloss.NewStyle().Foreground(Primary).Render("○")
	}
	return lipgloss.NewStyle().Foreground(Muted).Render("○")
}

// RenderSectionLabel renders a section label with a thick steel-blue bar and bold white title.
func RenderSectionLabel(title string) string {
	bar := lipgloss.NewStyle().Foreground(Primary).Render(SectionIconActive)
	label := lipgloss.NewStyle().Bold(true).Foreground(Text).Render(" " + title)
	return bar + label
}

// RenderActiveSectionLabel delegates to RenderSectionLabel (active = same strong style).
func RenderActiveSectionLabel(title string) string {
	return RenderSectionLabel(title)
}

// RenderDimSectionLabel renders an inactive section label: thin muted bar and text.
func RenderDimSectionLabel(title string) string {
	bar := lipgloss.NewStyle().Foreground(Secondary).Render(SectionIcon)
	label := lipgloss.NewStyle().Foreground(Muted).Render(" " + title)
	return bar + label
}

// RenderPreviewSubHeader renders a bold dim section label with a trailing rule.
func RenderPreviewSubHeader(title string) string {
	label := lipgloss.NewStyle().Foreground(TextDim).Bold(true).Render(title)
	rule := lipgloss.NewStyle().Foreground(Border).Render(" " + strings.Repeat(SectionDividerChar, 6))
	return label + rule
}

// RenderPreviewSubHeaderWithRule renders a titled rule spanning the given width.
func RenderPreviewSubHeaderWithRule(title string, width int) string {
	if width <= 0 {
		return RenderPreviewSubHeader(title)
	}
	label := lipgloss.NewStyle().Foreground(TextDim).Bold(true).Render(title)
	ruleStyle := lipgloss.NewStyle().Foreground(Border)
	trailing := width - lipgloss.Width(label) - 1
	if trailing < 1 {
		trailing = 1
	}
	suffix := ruleStyle.Render(" " + strings.Repeat(SectionDividerChar, trailing))
	return label + suffix
}

// RenderSectionBlock renders a section label followed by a thin separator line.
func RenderSectionBlock(title string, width int) string {
	label := RenderSectionLabel(title)
	if width <= 0 {
		return label
	}
	sep := lipgloss.NewStyle().Foreground(Border).Render(strings.Repeat(SectionDividerChar, width))
	return label + "\n" + sep
}

// RenderKeyValue renders a label-value pair with consistent alignment.
func RenderKeyValue(label, value string, labelWidth int) string {
	lbl := lipgloss.NewStyle().Foreground(TextDim).Width(labelWidth).Render(label)
	val := lipgloss.NewStyle().Foreground(Text).Render(value)
	return lbl + val
}

// RenderKeyValueAccent renders a label-value pair with the value in accent color.
func RenderKeyValueAccent(label, value string, labelWidth int) string {
	lbl := lipgloss.NewStyle().Foreground(TextDim).Width(labelWidth).Render(label)
	val := lipgloss.NewStyle().Foreground(Accent).Bold(true).Render(value)
	return lbl + val
}

// RenderDivider renders a subtle horizontal divider at the given width.
func RenderDivider(width int) string {
	if width <= 0 {
		return ""
	}
	return lipgloss.NewStyle().Foreground(Secondary).Render(strings.Repeat(SectionDividerChar, width))
}

// TruncateLine truncates a string to maxWidth visible characters, appending
// "..." if truncated. Uses ANSI-aware truncation to avoid splitting escape sequences.
func TruncateLine(s string, maxWidth int) string {
	if maxWidth <= 0 {
		return ""
	}
	if lipgloss.Width(s) <= maxWidth {
		return s
	}
	if maxWidth <= 3 {
		return truncate.String(s, uint(maxWidth))
	}
	return truncate.StringWithTail(s, uint(maxWidth), "...")
}

// PadRight pads a string with spaces to reach the target visual width.
// If the string is already wider, it is returned as-is (no truncation).
func PadRight(s string, targetWidth int) string {
	w := lipgloss.Width(s)
	if w >= targetWidth {
		return s
	}
	return s + strings.Repeat(" ", targetWidth-w)
}

// FitLine truncates a line to width and pads it back to that width.
func FitLine(s string, width int) string {
	if width <= 0 {
		return ""
	}
	return PadRight(TruncateLine(s, width), width)
}

// ClampBlock truncates each line in a multi-line block without wrapping.
func ClampBlock(content string, maxWidth int) string {
	if maxWidth <= 0 {
		return ""
	}
	lines := strings.Split(content, "\n")
	for i := range lines {
		lines[i] = TruncateLine(lines[i], maxWidth)
	}
	return strings.Join(lines, "\n")
}

// RenderNoWrapBlock renders a fixed-width styled block after truncating each
// content line to the style's inner width, preventing lipgloss from wrapping.
func RenderNoWrapBlock(style lipgloss.Style, content string, outerWidth int) string {
	if outerWidth <= 0 {
		return style.Render(content)
	}
	innerWidth := outerWidth - style.GetHorizontalFrameSize()
	if innerWidth < 1 {
		innerWidth = 1
	}
	return style.Width(outerWidth).Render(ClampBlock(content, innerWidth))
}

// RenderConfigLine builds a single config option line: cursor + label + value + hint,
// using manual padding instead of lipgloss .Width() to avoid internal wrapping.
// The result is truncated to maxWidth.
func RenderConfigLine(cursor, label, value, hint string, labelWidth, maxWidth int) string {
	paddedLabel := PadRight(label, labelWidth)
	line := cursor + paddedLabel + " " + value
	if hint != "" {
		line += "  " + hint
	}
	return TruncateLine(line, maxWidth)
}

// RenderStepHeader renders a step section title with a primary accent bar and divider.
func RenderStepHeader(title string, width int) string {
	bar := lipgloss.NewStyle().Foreground(Primary).Bold(true).Render(SectionIcon)
	header := bar + " " + lipgloss.NewStyle().Bold(true).Foreground(Text).Render(title)
	if width > 0 {
		return header + "\n" + RenderDivider(width)
	}
	return header
}

// RenderProgressBar renders a static filled/empty progress bar with a percentage label.
// Color shifts warning→primary→success as fill progresses; a sub-block partial cell
// gives a smoother leading edge.
// progress is clamped to [0.0, 1.0]; width is clamped to [MinProgressBarWidth, MaxProgressBarWidth].
func RenderProgressBar(progress float64, width int) string {
	if width < MinProgressBarWidth {
		width = MinProgressBarWidth
	}
	if width > MaxProgressBarWidth {
		width = MaxProgressBarWidth
	}
	if progress < 0 {
		progress = 0
	}
	if progress > 1 {
		progress = 1
	}

	var fillColor lipgloss.Color
	switch {
	case progress >= 1.0:
		fillColor = Success
	case progress >= 0.6:
		fillColor = Primary
	case progress >= 0.25:
		fillColor = Accent
	default:
		fillColor = Warning
	}

	subBlocks := []string{"", "▏", "▎", "▍", "▌", "▋", "▊", "▉"}
	exact := progress * float64(width)
	filled := int(exact)
	subIdx := int((exact - float64(filled)) * float64(len(subBlocks)))
	if subIdx >= len(subBlocks) {
		subIdx = len(subBlocks) - 1
	}
	hasPartial := subIdx > 0 && filled < width
	emptyWidth := width - filled
	if hasPartial {
		emptyWidth--
	}

	fillStyle := lipgloss.NewStyle().Foreground(fillColor)
	emptyStyle := lipgloss.NewStyle().Foreground(Muted)

	var sb strings.Builder
	if filled > 0 {
		sb.WriteString(fillStyle.Render(strings.Repeat("█", filled)))
	}
	if hasPartial {
		sb.WriteString(fillStyle.Render(subBlocks[subIdx]))
	}
	if emptyWidth > 0 {
		sb.WriteString(emptyStyle.Render(strings.Repeat("░", emptyWidth)))
	}

	pct := lipgloss.NewStyle().Foreground(fillColor).Bold(true).Render(fmt.Sprintf(" %3.0f%%", progress*100))
	return sb.String() + pct
}

// RenderStatusCount renders a count + noun summary line.
func RenderStatusCount(count, total int, noun string) string {
	color := Success
	if count == 0 {
		color = Warning
	}
	countText := lipgloss.NewStyle().Foreground(color).Bold(true).
		Render(fmt.Sprintf("%d/%d", count, total))
	nounText := lipgloss.NewStyle().Foreground(TextDim).Render(" " + noun)
	result := countText + nounText
	if count == 0 {
		result += "  " + lipgloss.NewStyle().Foreground(Warning).Italic(true).Render("select at least 1")
	}
	return result
}
