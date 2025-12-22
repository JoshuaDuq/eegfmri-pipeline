package styles

import "github.com/charmbracelet/lipgloss"

///////////////////////////////////////////////////////////////////
// Common Styles
///////////////////////////////////////////////////////////////////

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

	SelectedStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(Primary).
			BorderStyle(lipgloss.NormalBorder()).
			BorderLeft(true).
			BorderForeground(Primary).
			PaddingLeft(1)

	HelpStyle = lipgloss.NewStyle().
			Foreground(Muted)

	SuccessStyle = lipgloss.NewStyle().Foreground(Success).Bold(true)
	WarningStyle = lipgloss.NewStyle().Foreground(Warning).Bold(true)
	ErrorStyle   = lipgloss.NewStyle().Foreground(Error).Bold(true)
	AccentStyle  = lipgloss.NewStyle().Foreground(Accent).Bold(true)

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

	ListItemStyle = lipgloss.NewStyle().
			Foreground(Text).
			PaddingLeft(2)

	ListItemDescStyle = lipgloss.NewStyle().
				Foreground(TextDim).
				Italic(true).
				Faint(true)
)

///////////////////////////////////////////////////////////////////
// Badge Styles
///////////////////////////////////////////////////////////////////

var (
	BadgeSuccessStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#000000")).
				Background(Success).
				Bold(true).
				Padding(0, 1)

	BadgeErrorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FFFFFF")).
			Background(Error).
			Bold(true).
			Padding(0, 1)

	BadgeWarningStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#000000")).
				Background(Warning).
				Bold(true).
				Padding(0, 1)

	BadgeAccentStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#000000")).
				Background(Accent).
				Bold(true).
				Padding(0, 1)

	BadgeMutedStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FFFFFF")).
			Background(Muted).
			Bold(true).
			Padding(0, 1)
)

///////////////////////////////////////////////////////////////////
// Validation & Dialog Styles
///////////////////////////////////////////////////////////////////

var (
	ValidIndicatorStyle   = lipgloss.NewStyle().Foreground(Success)
	InvalidIndicatorStyle = lipgloss.NewStyle().Foreground(Warning)

	ConfirmBoxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(Warning).
			Padding(1, 2)

	ErrorPanelStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(Error).
			Padding(1, 2)
)
