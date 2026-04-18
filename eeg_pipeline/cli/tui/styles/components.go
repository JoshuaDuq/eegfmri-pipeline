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

	// Focused card uses a slightly brighter border (not full white) so the
	// chrome recedes and the focused content inside can own attention via
	// bold + bright-white text. A fully white border reads as a heavy frame
	// and competes with the content it's meant to highlight.
	CardStyleFocused = lipgloss.NewStyle().
				Padding(1, 2).
				Border(lipgloss.RoundedBorder()).
				BorderForeground(BorderBright)

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
				BorderForeground(BorderBright).
				Padding(1, 2)
)

var (
	// Legacy "chip"-style footer keys, kept for inline affordances (e.g. the
	// "B browse" hint in globalsetup, the selected shortcut in quickactions).
	// New footer hints use the bracket-style key below, which is quieter and
	// better suited to a research-app aesthetic.
	FooterKeyPrimaryStyle     = lipgloss.NewStyle().Foreground(BgDark).Background(Primary).Bold(true).Padding(0, 1)
	FooterKeySecondaryStyle   = lipgloss.NewStyle().Foreground(TextDim).Background(Surface).Padding(0, 1)
	FooterLabelPrimaryStyle   = lipgloss.NewStyle().Foreground(Text)
	FooterLabelSecondaryStyle = lipgloss.NewStyle().Foreground(Muted)

	// Bracket-style key hints: "[Enter] Open". Quieter and more CLI-native
	// than background-filled chips.
	FooterKeyBracketStyle  = lipgloss.NewStyle().Foreground(Border)
	FooterKeyTextPrimary   = lipgloss.NewStyle().Foreground(Primary).Bold(true)
	FooterKeyTextSecondary = lipgloss.NewStyle().Foreground(TextDim).Bold(true)
)

// Typography: reusable, semantic text styles.
// Prefer these over ad-hoc lipgloss.NewStyle() chains in views.
var (
	TitleStyle       = lipgloss.NewStyle().Foreground(Text).Bold(true)
	TitleAccentStyle = lipgloss.NewStyle().Foreground(Primary).Bold(true)
	SubtitleStyle    = lipgloss.NewStyle().Foreground(TextDim)
	LabelStyle       = lipgloss.NewStyle().Foreground(TextDim)
	ValueStyle       = lipgloss.NewStyle().Foreground(Text)
	ValueAccentStyle = lipgloss.NewStyle().Foreground(Accent).Bold(true)
	HintStyle        = lipgloss.NewStyle().Foreground(Muted).Italic(true)
	MutedTextStyle   = lipgloss.NewStyle().Foreground(Muted)
)

// Pill styles: small rounded badges used for counters, statuses, step indicators.
var (
	PillPrimaryStyle = lipgloss.NewStyle().Foreground(BgDark).Background(Primary).Bold(true).Padding(0, 1)
	PillAccentStyle  = lipgloss.NewStyle().Foreground(BgDark).Background(Accent).Bold(true).Padding(0, 1)
	PillSuccessStyle = lipgloss.NewStyle().Foreground(BgDark).Background(Success).Bold(true).Padding(0, 1)
	PillWarningStyle = lipgloss.NewStyle().Foreground(BgDark).Background(Warning).Bold(true).Padding(0, 1)
	PillErrorStyle   = lipgloss.NewStyle().Foreground(BgDark).Background(Error).Bold(true).Padding(0, 1)
	PillMutedStyle   = lipgloss.NewStyle().Foreground(TextDim).Background(Surface).Padding(0, 1)
	PillOutlineStyle = lipgloss.NewStyle().Foreground(TextDim).Border(lipgloss.RoundedBorder()).BorderForeground(Border).Padding(0, 1)
)
