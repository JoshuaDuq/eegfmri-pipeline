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
	text := fmt.Sprintf("  ↑ %d more above", count)
	return lipgloss.NewStyle().Foreground(TextDim).Render(text)
}

func RenderScrollDownIndicator(count int) string {
	if count <= 0 {
		return ""
	}
	text := fmt.Sprintf("  ↓ %d more below", count)
	return lipgloss.NewStyle().Foreground(TextDim).Render(text)
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

// RenderSectionLabel renders a section label with a steel-blue bar and bold white title.
func RenderSectionLabel(title string) string {
	bar := lipgloss.NewStyle().Foreground(Primary).Render(SectionIcon)
	label := lipgloss.NewStyle().Bold(true).Foreground(Text).Render(" " + title)
	return bar + label
}

// RenderActiveSectionLabel delegates to RenderSectionLabel (active = same strong style).
func RenderActiveSectionLabel(title string) string {
	return RenderSectionLabel(title)
}

// RenderDimSectionLabel renders an inactive section label: muted bar and text.
func RenderDimSectionLabel(title string) string {
	bar := lipgloss.NewStyle().Foreground(Border).Render(SectionIcon)
	label := lipgloss.NewStyle().Foreground(Muted).Render(" " + title)
	return bar + label
}

// RenderPreviewSubHeader renders a lightweight sub-section label for preview panes.
func RenderPreviewSubHeader(title string) string {
	bar := lipgloss.NewStyle().Foreground(Secondary).Render(SectionIcon)
	label := lipgloss.NewStyle().Foreground(TextDim).Render(" " + title)
	return bar + label
}

// RenderPreviewSubHeaderWithRule renders a preview sub-header followed by a thin rule.
func RenderPreviewSubHeaderWithRule(title string, width int) string {
	header := RenderPreviewSubHeader(title)
	if width <= 0 {
		return header
	}
	rule := lipgloss.NewStyle().Foreground(Border).Render(strings.Repeat(SectionDividerChar, width))
	return header + "\n" + rule
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
	return lipgloss.NewStyle().Foreground(Border).Render(strings.Repeat(SectionDividerChar, width))
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

// RenderStatusCount renders a count badge + summary line.
func RenderStatusCount(count, total int, noun string) string {
	var pillFg, pillBg lipgloss.Color
	if count >= 1 {
		pillFg, pillBg = BgDark, Success
	} else {
		pillFg, pillBg = BgDark, Warning
	}
	pill := lipgloss.NewStyle().Foreground(pillFg).Background(pillBg).Bold(true).Padding(0, 1).
		Render(fmt.Sprintf("%d/%d", count, total))
	nounStyle := lipgloss.NewStyle().Foreground(TextDim)
	result := pill + " " + nounStyle.Render(noun)
	if count == 0 {
		result += "  " + lipgloss.NewStyle().Foreground(Warning).Render("select at least 1")
	}
	return result
}
