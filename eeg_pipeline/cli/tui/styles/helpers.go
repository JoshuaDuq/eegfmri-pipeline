package styles

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
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
	keyStyle := lipgloss.NewStyle().
		Foreground(Text).
		Background(Border).
		Bold(true).
		Padding(0, 1)
	labelStyle := lipgloss.NewStyle().Foreground(TextDim)
	return keyStyle.Render(key) + " " + labelStyle.Render(label)
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

// RenderSectionLabel renders a styled section label with a left accent bar.
func RenderSectionLabel(title string) string {
	bar := lipgloss.NewStyle().Foreground(Primary).Render(SectionIcon)
	label := lipgloss.NewStyle().Bold(true).Foreground(Text).Render(" " + title)
	return bar + label
}

// RenderDimSectionLabel renders a section label for inactive sections.
func RenderDimSectionLabel(title string) string {
	bar := lipgloss.NewStyle().Foreground(Border).Render(SectionIcon)
	label := lipgloss.NewStyle().Bold(true).Foreground(TextDim).Render(" " + title)
	return bar + label
}
