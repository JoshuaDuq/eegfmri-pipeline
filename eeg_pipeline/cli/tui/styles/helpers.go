package styles

import (
	"fmt"

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

func RenderKeyHint(key, label string) string {
	keyStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#000000")).
		Background(Accent).
		Bold(true).
		Padding(0, 1)
	labelStyle := lipgloss.NewStyle().Foreground(TextDim).PaddingLeft(1)
	return keyStyle.Render(key) + labelStyle.Render(label)
}

func RenderCheckbox(checked, focused bool) string {
	if checked && focused {
		return lipgloss.NewStyle().Foreground(Success).Bold(true).Render("◉")
	}
	if checked {
		return lipgloss.NewStyle().Foreground(Success).Render("●")
	}
	if focused {
		return lipgloss.NewStyle().Foreground(Primary).Bold(true).Render("○")
	}
	return lipgloss.NewStyle().Foreground(Muted).Render("○")
}

func RenderRadio(selected, focused bool) string {
	if selected && focused {
		return lipgloss.NewStyle().Foreground(Accent).Bold(true).Render("◉")
	}
	if selected {
		return lipgloss.NewStyle().Foreground(Primary).Render("●")
	}
	if focused {
		return lipgloss.NewStyle().Foreground(Primary).Render("○")
	}
	return lipgloss.NewStyle().Foreground(Muted).Render("○")
}
