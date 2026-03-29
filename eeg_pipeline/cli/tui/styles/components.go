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
			Padding(1, 2).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(Border)

	CardStyleFocused = lipgloss.NewStyle().
				Padding(1, 2).
				Border(lipgloss.RoundedBorder()).
				BorderForeground(Primary)

	BoxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(Border).
			Padding(1, 2)

	FooterStyle = lipgloss.NewStyle().
			Foreground(TextDim)

	PanelStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(Secondary).
			Padding(1, 2)

	HeaderLineStyle = lipgloss.NewStyle().Foreground(Secondary)

	SectionDividerStyle = lipgloss.NewStyle().Foreground(Border)

	ProgressFilledStyle = lipgloss.NewStyle().Foreground(Primary)
	ProgressEmptyStyle  = lipgloss.NewStyle().Foreground(Muted)
)

var (
	BadgeSuccessStyle = lipgloss.NewStyle().Foreground(Success).Bold(true)
	BadgeErrorStyle   = lipgloss.NewStyle().Foreground(Error).Bold(true)
	BadgeWarningStyle = lipgloss.NewStyle().Foreground(Warning).Bold(true)
	BadgeAccentStyle  = lipgloss.NewStyle().Foreground(Accent).Bold(true)
	BadgeMutedStyle   = lipgloss.NewStyle().Foreground(Muted)
	InlineKindStyle   = lipgloss.NewStyle().Foreground(Muted)
	PreviewBlockLabel = lipgloss.NewStyle().Foreground(Primary).Bold(true)
	PreviewSepStyle   = lipgloss.NewStyle().Foreground(Border)
)

var (
	ValidIndicatorStyle   = lipgloss.NewStyle().Foreground(Success)
	InvalidIndicatorStyle = lipgloss.NewStyle().Foreground(Warning)
)

var (
	FocusedInputStyle     = lipgloss.NewStyle().Foreground(Primary).Bold(true).Underline(true)
	InputPlaceholderStyle = lipgloss.NewStyle().Foreground(Muted).Italic(true)
	PanelFocusedStyle     = lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(Primary).
				Padding(1, 2)
)

var (
	FooterKeyPrimaryStyle     = lipgloss.NewStyle().Foreground(BgDark).Background(Primary).Bold(true).Padding(0, 1)
	FooterKeySecondaryStyle   = lipgloss.NewStyle().Foreground(TextDim).Background(Surface).Padding(0, 1)
	FooterLabelPrimaryStyle   = lipgloss.NewStyle().Foreground(Text)
	FooterLabelSecondaryStyle = lipgloss.NewStyle().Foreground(Muted)
)
