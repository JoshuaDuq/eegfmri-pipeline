package styles

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

type BreadcrumbStep struct {
	Name      string
	Completed bool
	Current   bool
	Skipped   bool
}

func RenderBreadcrumb(steps []BreadcrumbStep, width int, ticker int) string {
	if len(steps) == 0 {
		return ""
	}

	compact := width < 80

	var parts []string

	for i, step := range steps {
		var icon string
		var nameStyle lipgloss.Style

		if step.Completed {
			icon = lipgloss.NewStyle().Foreground(Success).Render("●")
			nameStyle = lipgloss.NewStyle().Foreground(TextDim)
		} else if step.Current {
			frames := []string{"◉", "●", "◉", "◎"}
			frame := frames[(ticker/3)%len(frames)]
			icon = lipgloss.NewStyle().Foreground(Primary).Bold(true).Render(frame)
			nameStyle = lipgloss.NewStyle().Foreground(Primary).Bold(true)
		} else if step.Skipped {
			icon = lipgloss.NewStyle().Foreground(Muted).Render("○")
			nameStyle = lipgloss.NewStyle().Foreground(Muted).Strikethrough(true)
		} else {
			icon = lipgloss.NewStyle().Foreground(Muted).Render("○")
			nameStyle = lipgloss.NewStyle().Foreground(Muted)
		}

		if compact {
			parts = append(parts, icon)
		} else {
			parts = append(parts, icon+" "+nameStyle.Render(step.Name))
		}

		if i < len(steps)-1 {
			connector := lipgloss.NewStyle().Foreground(Secondary).Render(" → ")
			if compact {
				connector = " "
			}
			parts = append(parts, connector)
		}
	}

	return strings.Join(parts, "")
}

func RenderStepCounter(current, total int) string {
	counterStyle := lipgloss.NewStyle().Foreground(TextDim)
	return counterStyle.Render(fmt.Sprintf("Step %d of %d", current, total))
}

func RenderProgressDots(current, total int, ticker int) string {
	var dots []string

	for i := 0; i < total; i++ {
		var dot string
		if i < current {
			dot = lipgloss.NewStyle().Foreground(Success).Render("●")
		} else if i == current {
			frames := []string{"◉", "●", "◉", "◎"}
			frame := frames[(ticker/3)%len(frames)]
			dot = lipgloss.NewStyle().Foreground(Primary).Bold(true).Render(frame)
		} else {
			dot = lipgloss.NewStyle().Foreground(Muted).Render("○")
		}
		dots = append(dots, dot)
	}

	return strings.Join(dots, " ")
}

type ValidationStatus int

const (
	ValidationPending ValidationStatus = iota
	ValidationValid
	ValidationWarning
	ValidationError
)

func RenderStepValidation(status ValidationStatus, message string) string {
	var icon string
	var style lipgloss.Style

	switch status {
	case ValidationValid:
		icon = CheckMark
		style = lipgloss.NewStyle().Foreground(Success)
	case ValidationWarning:
		icon = WarningMark
		style = lipgloss.NewStyle().Foreground(Warning)
	case ValidationError:
		icon = CrossMark
		style = lipgloss.NewStyle().Foreground(Error)
	default:
		icon = PendingMark
		style = lipgloss.NewStyle().Foreground(Muted)
	}

	if message == "" {
		return style.Render(icon)
	}
	return style.Render(icon + " " + message)
}

func RenderContextualHint(hints []string, width int) string {
	if len(hints) == 0 {
		return ""
	}

	separator := lipgloss.NewStyle().Foreground(Secondary).Render("  │  ")
	content := strings.Join(hints, separator)

	return FooterStyle.Width(width).Render(content)
}

func RenderToast(message string, toastType string, width int) string {
	var bgColor, fgColor lipgloss.Color
	var icon string

	switch toastType {
	case "success":
		bgColor = Success
		fgColor = lipgloss.Color("#000000")
		icon = CheckMark
	case "error":
		bgColor = Error
		fgColor = lipgloss.Color("#FFFFFF")
		icon = CrossMark
	case "warning":
		bgColor = Warning
		fgColor = lipgloss.Color("#000000")
		icon = WarningMark
	default:
		bgColor = Primary
		fgColor = lipgloss.Color("#FFFFFF")
		icon = "ℹ"
	}

	toastStyle := lipgloss.NewStyle().
		Background(bgColor).
		Foreground(fgColor).
		Bold(true).
		Padding(0, 2)

	return toastStyle.Render(icon + " " + message)
}

func RenderInlineValidation(valid bool, count int, required int, itemName string) string {
	var indicator string
	var message string

	if count >= required {
		indicator = lipgloss.NewStyle().Foreground(Success).Render(CheckMark + " ")
		message = fmt.Sprintf("%d %s selected", count, itemName)
	} else if count > 0 {
		indicator = lipgloss.NewStyle().Foreground(Warning).Render(WarningMark + " ")
		message = fmt.Sprintf("%d %s selected (need %d)", count, itemName, required)
	} else {
		indicator = lipgloss.NewStyle().Foreground(Error).Render(CrossMark + " ")
		message = fmt.Sprintf("Select at least %d %s", required, itemName)
	}

	return indicator + lipgloss.NewStyle().Foreground(TextDim).Render(message)
}

func RenderSelectionSummary(selected, total int) string {
	pct := 0.0
	if total > 0 {
		pct = float64(selected) / float64(total) * 100
	}

	var color lipgloss.Color
	if selected == 0 {
		color = Error
	} else if selected == total {
		color = Success
	} else {
		color = Warning
	}

	countStyle := lipgloss.NewStyle().Foreground(color).Bold(true)
	dimStyle := lipgloss.NewStyle().Foreground(TextDim)

	return countStyle.Render(fmt.Sprintf("%d", selected)) +
		dimStyle.Render(fmt.Sprintf("/%d (%.0f%%)", total, pct))
}
