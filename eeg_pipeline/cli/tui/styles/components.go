package styles

import "github.com/charmbracelet/lipgloss"

// Component styles: Modern Research Dashboard.
var (
	BrandStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(Primary).
			Padding(0, 1)

	SectionTitleStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(Text).
				MarginBottom(1)

	CardStyle = lipgloss.NewStyle().
			Padding(1, 3).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(Border)

	CardStyleFocused = lipgloss.NewStyle().
				Padding(1, 3).
				Border(lipgloss.RoundedBorder()).
				BorderForeground(Primary)

	BoxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(Border).
			Padding(1, 2)

	FooterStyle = lipgloss.NewStyle().
			Foreground(TextDim).
			MarginTop(1).
			Border(lipgloss.NormalBorder(), true, false, false, false).
			BorderForeground(Border).
			PaddingTop(1)

	HeaderLineStyle = lipgloss.NewStyle().Foreground(Border)

	SectionDividerStyle = lipgloss.NewStyle().Foreground(Border)

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
