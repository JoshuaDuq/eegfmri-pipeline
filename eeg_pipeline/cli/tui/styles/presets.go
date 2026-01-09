package styles

import "github.com/charmbracelet/lipgloss"

const (
	presetSelectorPrefix = "▸ "
	presetSelectorIndent = "  "
	presetDescriptionSep = " — "
	presetBadgePrefix    = "Preset: "
)

const (
	shortcutTextColor = lipgloss.Color("#000000")
)

type Preset struct {
	Key         string
	Name        string
	Shortcut    string
	Description string
	Icon        string
}

var FeaturePresets = []Preset{
	{
		Key:         "quick",
		Name:        "Quick Run",
		Shortcut:    "Q",
		Description: "Power + Aperiodic + Complexity (fast)",
		Icon:        "*",
	},
	{
		Key:         "full",
		Name:        "Full Analysis",
		Shortcut:    "F",
		Description: "All feature categories",
		Icon:        "+",
	},
	{
		Key:         "connectivity",
		Name:        "Connectivity Focus",
		Shortcut:    "C",
		Description: "Power + Connectivity + PAC",
		Icon:        "~",
	},
	{
		Key:         "spectral",
		Name:        "Spectral Focus",
		Shortcut:    "S",
		Description: "Power + Spectral + Aperiodic + Ratios",
		Icon:        "^",
	},
}

var BehaviorPresets = []Preset{
	{
		Key:         "quick",
		Name:        "Quick Analysis",
		Shortcut:    "Q",
		Description: "Trial table + Correlations + Report",
		Icon:        "*",
	},
	{
		Key:         "full",
		Name:        "Full Analysis",
		Shortcut:    "F",
		Description: "All computations enabled",
		Icon:        "+",
	},
	{
		Key:         "regression",
		Name:        "Regression Focus",
		Shortcut:    "R",
		Description: "Trial table + Regression + Models + Stability",
		Icon:        "~",
	},
	{
		Key:         "temporal",
		Name:        "Temporal Focus",
		Shortcut:    "T",
		Description: "Trial table + Temporal + Cluster + Mediation",
		Icon:        "^",
	},
}

func shortcutBadgeStyle() lipgloss.Style {
	return lipgloss.NewStyle().
		Foreground(shortcutTextColor).
		Background(Accent).
		Bold(true).
		Padding(0, 1)
}

func presetNameStyle(isSelected bool) lipgloss.Style {
	style := lipgloss.NewStyle().Foreground(Text).PaddingLeft(1)
	if isSelected {
		style = style.Foreground(Primary).Bold(true)
	}
	return style
}

func presetDescriptionStyle() lipgloss.Style {
	return lipgloss.NewStyle().Foreground(TextDim).Italic(true)
}

func presetHeaderStyle() lipgloss.Style {
	return lipgloss.NewStyle().
		Foreground(TextDim).
		Italic(true).
		MarginBottom(1)
}

func renderPresetLine(preset Preset, isSelected bool) string {
	shortcutStyle := shortcutBadgeStyle()
	nameStyle := presetNameStyle(isSelected)
	descStyle := presetDescriptionStyle()

	shortcutText := shortcutStyle.Render(preset.Shortcut)
	nameText := nameStyle.Render(preset.Icon + " " + preset.Name)
	descText := descStyle.Render(presetDescriptionSep + preset.Description)

	return shortcutText + nameText + " " + descText
}

func renderPresetPrefix(isSelected bool) string {
	if isSelected {
		return lipgloss.NewStyle().Foreground(Primary).Render(presetSelectorPrefix)
	}
	return presetSelectorIndent
}

func RenderPresetSelector(presets []Preset, selectedIdx int, width int) string {
	header := presetHeaderStyle().Render("Quick Presets (press key to apply):") + "\n\n"

	var output string
	output += header

	for i, preset := range presets {
		isSelected := i == selectedIdx
		prefix := renderPresetPrefix(isSelected)
		line := renderPresetLine(preset, isSelected)
		output += prefix + line + "\n"
	}

	return output
}

func RenderPresetBadge(presetName string) string {
	if presetName == "" {
		return ""
	}

	return shortcutBadgeStyle().
		Render(presetBadgePrefix + presetName)
}
