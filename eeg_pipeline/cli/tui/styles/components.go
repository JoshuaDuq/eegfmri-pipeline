package styles

import "github.com/charmbracelet/lipgloss"

// Common component styles for the TUI, built on the shared color palette.

var (
	BrandStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(Primary).
			Padding(0, 1)

	SectionTitleStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(Primary).
				Underline(true).
				MarginBottom(1)

	CardStyle = lipgloss.NewStyle().
			Padding(1, 2).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(Muted)

	BoxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(Muted).
			Padding(0, 1)

	FooterStyle = lipgloss.NewStyle().
			Foreground(TextDim).
			MarginTop(1).
			Border(lipgloss.NormalBorder(), true, false, false, false).
			BorderForeground(Muted).
			PaddingTop(1)

	ProgressFilledStyle = lipgloss.NewStyle().Foreground(Primary)
	ProgressEmptyStyle  = lipgloss.NewStyle().Foreground(Muted)
)

var (
	badgeDarkForeground  = lipgloss.Color("#000000")
	badgeLightForeground = lipgloss.Color("#FFFFFF")
)

func newBadgeStyle(foreground, background lipgloss.Color) lipgloss.Style {
	return lipgloss.NewStyle().
		Foreground(foreground).
		Background(background).
		Bold(true).
		Padding(0, 1)
}

var (
	BadgeSuccessStyle = newBadgeStyle(badgeDarkForeground, Success)
	BadgeErrorStyle   = newBadgeStyle(badgeLightForeground, Error)
	BadgeWarningStyle = newBadgeStyle(badgeDarkForeground, Warning)
	BadgeAccentStyle  = newBadgeStyle(badgeDarkForeground, Accent)
	BadgeMutedStyle   = newBadgeStyle(badgeLightForeground, Muted)
)

var (
	ValidIndicatorStyle   = lipgloss.NewStyle().Foreground(Success)
	InvalidIndicatorStyle = lipgloss.NewStyle().Foreground(Warning)
)
