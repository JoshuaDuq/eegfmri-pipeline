package styles

import (
	"fmt"

	"github.com/charmbracelet/lipgloss"
)

///////////////////////////////////////////////////////////////////
// List Layout Helpers
///////////////////////////////////////////////////////////////////

// ListLayout contains calculated layout parameters for a scrollable list
type ListLayout struct {
	MaxItems     int // Maximum items visible at once
	StartIdx     int // First visible item index
	EndIdx       int // Last visible item index (exclusive)
	ShowScrollUp bool
	ShowScrollDn bool
	TotalItems   int
	CursorIdx    int
}

// CalculateListLayout determines the visible range for a scrollable list
// based on terminal height, cursor position, and total items.
// headerRows is the number of rows reserved for headers/footers.
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

// RenderScrollIndicator returns scroll indicator text for a list
func RenderScrollIndicator(layout ListLayout, isUp bool) string {
	if isUp && layout.ShowScrollUp {
		return RenderScrollUpIndicator(layout.StartIdx)
	}
	if !isUp && layout.ShowScrollDn {
		return RenderScrollDownIndicator(layout.TotalItems - layout.EndIdx)
	}
	return ""
}

// RenderScrollUpIndicator returns a formatted "more above" indicator
func RenderScrollUpIndicator(count int) string {
	if count <= 0 {
		return ""
	}
	text := fmt.Sprintf("  ↑ %d more above", count)
	return lipgloss.NewStyle().Foreground(TextDim).Render(text)
}

// RenderScrollDownIndicator returns a formatted "more below" indicator
func RenderScrollDownIndicator(count int) string {
	if count <= 0 {
		return ""
	}
	text := fmt.Sprintf("  ↓ %d more below", count)
	return lipgloss.NewStyle().Foreground(TextDim).Render(text)
}

// IsTerminalTooSmall returns true if terminal dimensions are too small to render properly
func IsTerminalTooSmall(width, height int) bool {
	return width < MinTerminalWidth || height < MinTerminalHeight
}

// RenderTerminalTooSmall renders a warning message when terminal is too small
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

///////////////////////////////////////////////////////////////////
// Section Header Helpers
///////////////////////////////////////////////////////////////////

// RenderSectionHeader creates a professional animated section title with accent indicator.
// The ticker parameter controls the animation frame (pass model.ticker/3 for standard speed).
func RenderSectionHeader(title string, ticker int) string {
	accentFrames := []string{"◆", "◇", "◆", "◈"}
	accent := lipgloss.NewStyle().
		Foreground(Accent).
		Bold(true).
		Render(accentFrames[ticker%len(accentFrames)])
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(Primary).
		MarginLeft(1)
	return accent + titleStyle.Render(" "+title)
}

// RenderSectionHeaderWithSubtitle creates a section header with an optional subtitle.
func RenderSectionHeaderWithSubtitle(title, subtitle string, ticker int) string {
	header := RenderSectionHeader(title, ticker)
	if subtitle == "" {
		return header
	}
	subtitleStyle := lipgloss.NewStyle().
		Foreground(TextDim).
		Italic(true).
		PaddingLeft(4)
	return header + "\n" + subtitleStyle.Render(subtitle)
}

///////////////////////////////////////////////////////////////////
// Helper Functions
///////////////////////////////////////////////////////////////////

// RenderKeyHint formats a keyboard shortcut hint with pill-style key badge
func RenderKeyHint(key, label string) string {
	// Pill-style key badge
	keyStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#000000")).
		Background(Accent).
		Bold(true).
		Padding(0, 1)
	labelStyle := lipgloss.NewStyle().Foreground(TextDim).PaddingLeft(1)
	return keyStyle.Render(key) + labelStyle.Render(label)
}

// RenderFooterHelp formats multiple keyboard hints for footer
func RenderFooterHelp(hints ...string) string {
	if len(hints) == 0 {
		return ""
	}
	separator := lipgloss.NewStyle().Foreground(Secondary).Render("  │  ")
	result := hints[0]
	for i := 1; i < len(hints); i++ {
		result = lipgloss.JoinHorizontal(lipgloss.Center, result, separator, hints[i])
	}
	return result
}

// RenderStatusBadge returns a styled status badge
func RenderStatusBadge(status string) string {
	switch status {
	case "success", "running", "connected":
		return BadgeSuccessStyle.Render(status)
	case "error", "failed":
		return BadgeErrorStyle.Render(status)
	case "warning", "stopped":
		return BadgeWarningStyle.Render(status)
	case "pending", "checking":
		return BadgeAccentStyle.Render(status)
	default:
		return BadgeMutedStyle.Render(status)
	}
}

// RenderCheckbox returns a styled checkbox with modern circle icons
func RenderCheckbox(checked, focused bool) string {
	if checked && focused {
		// Selected + focused: filled circle with glow effect
		return lipgloss.NewStyle().Foreground(Success).Bold(true).Render("◉")
	} else if checked {
		// Selected: solid filled circle
		return lipgloss.NewStyle().Foreground(Success).Render("●")
	} else if focused {
		// Unselected + focused: ring with primary color
		return lipgloss.NewStyle().Foreground(Primary).Bold(true).Render("○")
	}
	// Unselected: subtle ring
	return lipgloss.NewStyle().Foreground(Muted).Render("○")
}

// RenderRadio returns a styled radio button with modern circle icons
func RenderRadio(selected, focused bool) string {
	if selected && focused {
		// Selected + focused: accent colored filled circle
		return lipgloss.NewStyle().Foreground(Accent).Bold(true).Render("◉")
	} else if selected {
		// Selected: primary filled circle
		return lipgloss.NewStyle().Foreground(Primary).Render("●")
	} else if focused {
		// Unselected + focused: primary ring
		return lipgloss.NewStyle().Foreground(Primary).Render("○")
	}
	// Unselected: muted ring
	return lipgloss.NewStyle().Foreground(Muted).Render("○")
}

// RenderValidationIndicator returns a validation status indicator
func RenderValidationIndicator(valid bool, reason string) string {
	if valid {
		return ValidIndicatorStyle.Render(CheckMark)
	}
	return InvalidIndicatorStyle.Render(WarningMark + " " + reason)
}

// RenderProgressBar creates a visual progress bar with polished edge caps
func RenderProgressBar(progress float64, width int, showPct bool) string {
	width = clampProgressBarWidth(width, 5)
	innerWidth := width - 2
	filled := clampFilledAmount(progress, innerWidth)

	leftCap := lipgloss.NewStyle().Foreground(Primary).Render("▐")
	rightCap := lipgloss.NewStyle().Foreground(Secondary).Render("▌")
	filledBar := ProgressFilledStyle.Render(repeatString("█", filled))
	emptyBar := ProgressEmptyStyle.Render(repeatString("░", innerWidth-filled))

	bar := leftCap + filledBar + emptyBar + rightCap

	if showPct {
		percentStyle := lipgloss.NewStyle().Bold(true).Foreground(Primary).Width(5).Align(lipgloss.Right)
		bar += percentStyle.Render(formatPercent(progress))
	}

	return bar
}

// RenderMiniProgressBar creates a compact progress bar
func RenderMiniProgressBar(progress float64, width int, color lipgloss.Color) string {
	filled := clampFilledAmount(progress, width)
	filledBar := lipgloss.NewStyle().Foreground(color).Render(repeatString("█", filled))
	emptyBar := lipgloss.NewStyle().Foreground(Secondary).Render(repeatString("░", width-filled))
	return filledBar + emptyBar
}

// RenderDivider creates a horizontal divider
func RenderDivider(width int, style DividerStyle) string {
	var char string
	switch style {
	case DividerDouble:
		char = "═"
	case DividerDashed:
		char = "╌"
	case DividerDots:
		char = "·"
	default:
		char = "─"
	}
	return lipgloss.NewStyle().Foreground(Secondary).Render(repeatString(char, width))
}

// RenderTitle creates a styled title
func RenderTitle(text string) string {
	return lipgloss.NewStyle().
		Bold(true).
		Foreground(Primary).
		Render(text)
}

// RenderTitleWithUnderline creates a styled title with an underline
func RenderTitleWithUnderline(text string) string {
	title := RenderTitle(text)
	underlineWidth := len(text)
	underline := lipgloss.NewStyle().Foreground(Secondary).Render(repeatString("─", underlineWidth))
	return title + "\n" + underline
}

// RenderSubtitle creates a styled subtitle
func RenderSubtitle(text string) string {
	return lipgloss.NewStyle().
		Foreground(TextDim).
		Italic(true).
		Render(text)
}

// RenderLabel creates a styled label for key-value pairs
func RenderLabel(text string, width int) string {
	return lipgloss.NewStyle().
		Foreground(TextDim).
		Width(width).
		Render(text + ":")
}

// RenderValue creates a styled value for key-value pairs
func RenderValue(text string) string {
	return lipgloss.NewStyle().
		Foreground(Text).
		Render(text)
}

// RenderHighlight creates highlighted text
func RenderHighlight(text string) string {
	return lipgloss.NewStyle().
		Bold(true).
		Foreground(Accent).
		Render(text)
}

// RenderCode renders text in a code style
func RenderCode(text string) string {
	return lipgloss.NewStyle().
		Foreground(Text).
		Bold(true).
		Render(text)
}

// RenderBadge renders a small status indicator
func RenderBadge(text string, color lipgloss.Color) string {
	return lipgloss.NewStyle().
		Foreground(color).
		Bold(true).
		Render("[" + text + "]")
}

// RenderSpinner returns a spinner frame
func RenderSpinner(tick int) string {
	frames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	return lipgloss.NewStyle().Foreground(Accent).Render(frames[tick%len(frames)])
}

// RenderDotSpinner returns a dot spinner frame
func RenderDotSpinner(tick int) string {
	frames := []string{"◐", "◓", "◑", "◒"}
	return lipgloss.NewStyle().Foreground(Accent).Render(frames[tick%len(frames)])
}

// RenderPulseSpinner returns a pulsing animation frame
func RenderPulseSpinner(tick int) string {
	frames := []string{"◉", "◎", "○", "◎"}
	return lipgloss.NewStyle().Foreground(Primary).Render(frames[tick%len(frames)])
}

// RenderBlockSpinner returns a block spinner animation
func RenderBlockSpinner(tick int) string {
	frames := []string{"▖", "▘", "▝", "▗"}
	return lipgloss.NewStyle().Foreground(Accent).Render(frames[tick%len(frames)])
}

// RenderGradientProgressBar creates a gradient-styled progress bar with color transitions
func RenderGradientProgressBar(progress float64, width int) string {
	width = clampProgressBarWidth(width, 10)
	filled := clampFilledAmount(progress, width)

	bar := buildGradientBar(filled, width)
	emptyBar := lipgloss.NewStyle().Foreground(Secondary).Render(repeatString("░", width-filled))
	percentStyle := lipgloss.NewStyle().Bold(true).Foreground(Primary).Width(5).Align(lipgloss.Right)

	return bar + emptyBar + percentStyle.Render(formatPercent(progress))
}

const (
	gradientFirstThreshold  = 0.33
	gradientSecondThreshold = 0.66
)

func buildGradientBar(filled, totalWidth int) string {
	var bar string
	for i := 0; i < filled; i++ {
		position := float64(i) / float64(totalWidth)
		color := selectGradientColor(position)
		bar += lipgloss.NewStyle().Foreground(color).Render("█")
	}
	return bar
}

func selectGradientColor(position float64) lipgloss.Color {
	if position < gradientFirstThreshold {
		return Accent
	}
	if position < gradientSecondThreshold {
		return Primary
	}
	return Success
}

// RenderAnimatedProgressBar creates an animated progress bar with a leading edge
func RenderAnimatedProgressBar(progress float64, width int, tick int) string {
	width = clampProgressBarWidth(width, 10)
	filled := clampFilledAmount(progress, width)

	leadChars := []string{"▓", "▒", "░"}
	leadIdx := tick % len(leadChars)

	filledBar := lipgloss.NewStyle().Foreground(Primary).Render(repeatString("█", filled))
	emptyBar := buildAnimatedEmptyBar(filled, width, leadChars[leadIdx])
	percentStyle := lipgloss.NewStyle().Bold(true).Foreground(Primary).Width(5).Align(lipgloss.Right)

	return filledBar + emptyBar + percentStyle.Render(formatPercent(progress))
}

func buildAnimatedEmptyBar(filled, width int, leadChar string) string {
	remaining := width - filled
	if remaining == 0 {
		return ""
	}
	if filled > 0 {
		lead := lipgloss.NewStyle().Foreground(Accent).Render(leadChar)
		empty := lipgloss.NewStyle().Foreground(Secondary).Render(repeatString("░", remaining-1))
		return lead + empty
	}
	return lipgloss.NewStyle().Foreground(Secondary).Render(repeatString("░", remaining))
}

// RenderCompactStats renders a compact key-value pair line
func RenderCompactStats(items map[string]string) string {
	var parts []string
	labelStyle := lipgloss.NewStyle().Foreground(TextDim)
	valueStyle := lipgloss.NewStyle().Foreground(Text).Bold(true)

	for label, value := range items {
		parts = append(parts, labelStyle.Render(label+":")+valueStyle.Render(" "+value))
	}

	if len(parts) == 0 {
		return ""
	}

	sep := lipgloss.NewStyle().Foreground(Secondary).Render(" │ ")
	result := parts[0]
	for i := 1; i < len(parts); i++ {
		result = lipgloss.JoinHorizontal(lipgloss.Center, result, sep, parts[i])
	}
	return result
}

// RenderStepIndicator creates a visual step indicator with connecting lines
func RenderStepIndicator(current, total int, tick int) string {
	steps := buildStepIndicators(current, total)
	connector := lipgloss.NewStyle().Foreground(Secondary).Render("──")
	return lipgloss.JoinHorizontal(lipgloss.Center, intersperse(steps, connector)...)
}

func buildStepIndicators(current, total int) []string {
	steps := make([]string, 0, total)
	for i := 1; i <= total; i++ {
		step := renderStepIndicator(i, current)
		steps = append(steps, step)
	}
	return steps
}

func renderStepIndicator(stepNum, currentStep int) string {
	stepText := string(rune('0' + stepNum))
	if stepNum < currentStep {
		return renderCompletedStep()
	}
	if stepNum == currentStep {
		return renderCurrentStep(stepText)
	}
	return renderPendingStep(stepText)
}

func renderCompletedStep() string {
	return lipgloss.NewStyle().
		Foreground(lipgloss.Color("#000000")).
		Background(Success).
		Bold(true).
		Padding(0, 1).
		Render(CheckMark)
}

func renderCurrentStep(stepText string) string {
	return lipgloss.NewStyle().
		Foreground(lipgloss.Color("#FFFFFF")).
		Background(Primary).
		Bold(true).
		Padding(0, 1).
		Render(stepText)
}

func renderPendingStep(stepText string) string {
	return lipgloss.NewStyle().
		Foreground(TextDim).
		Padding(0, 1).
		Render(stepText)
}

func intersperse(elems []string, sep string) []string {
	if len(elems) == 0 {
		return elems
	}
	result := make([]string, 0, len(elems)*2-1)
	for i, elem := range elems {
		result = append(result, elem)
		if i < len(elems)-1 {
			result = append(result, sep)
		}
	}
	return result
}

///////////////////////////////////////////////////////////////////
// Box Rendering
///////////////////////////////////////////////////////////////////

// RenderInfoBox renders content in a styled info box
func RenderInfoBox(title, content string, width int) string {
	titleBar := lipgloss.NewStyle().
		Bold(true).
		Foreground(Primary).
		Underline(true).
		Render(" " + title + " ")

	box := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(Muted).
		Padding(0, 1).
		Width(width)

	return titleBar + "\n" + box.Render(content)
}

// RenderSuccessBox renders content in a success-styled box
func RenderSuccessBox(content string, width int) string {
	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(Success).
		Padding(0, 1).
		Width(width).
		Render(content)
}

// RenderErrorBox renders content in an error-styled box
func RenderErrorBox(content string, width int) string {
	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(Error).
		Padding(0, 1).
		Width(width).
		Render(content)
}

// RenderWarningBox renders content in a warning-styled box
func RenderWarningBox(content string, width int) string {
	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(Warning).
		Padding(0, 1).
		Width(width).
		Render(content)
}

///////////////////////////////////////////////////////////////////
// Layout Helpers
///////////////////////////////////////////////////////////////////

// CenterText centers text within a given width
func CenterText(text string, width int) string {
	return lipgloss.PlaceHorizontal(width, lipgloss.Center, text)
}

// RightAlignText right-aligns text within a given width
func RightAlignText(text string, width int) string {
	return lipgloss.PlaceHorizontal(width, lipgloss.Right, text)
}

///////////////////////////////////////////////////////////////////
// Internal Helpers
///////////////////////////////////////////////////////////////////

func repeatString(s string, count int) string {
	if count <= 0 {
		return ""
	}
	result := ""
	for i := 0; i < count; i++ {
		result += s
	}
	return result
}

func clampProgressBarWidth(width, minWidth int) int {
	if width < minWidth {
		return minWidth
	}
	if width > MaxProgressBarWidth {
		return MaxProgressBarWidth
	}
	return width
}

func clampFilledAmount(progress float64, maxWidth int) int {
	filled := int(progress * float64(maxWidth))
	if filled < 0 {
		return 0
	}
	if filled > maxWidth {
		return maxWidth
	}
	return filled
}

func formatPercent(progress float64) string {
	percent := clampPercent(int(progress * 100))
	return formatPercentString(percent)
}

func clampPercent(percent int) int {
	if percent < 0 {
		return 0
	}
	if percent > 100 {
		return 100
	}
	return percent
}

func formatPercentString(percent int) string {
	if percent < 10 {
		return fmt.Sprintf("  %d%%", percent)
	}
	if percent < 100 {
		tens := percent / 10
		ones := percent % 10
		return fmt.Sprintf(" %d%d%%", tens, ones)
	}
	return "100%"
}
