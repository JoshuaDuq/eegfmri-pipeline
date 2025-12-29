package styles

import "github.com/charmbracelet/lipgloss"

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

func RenderPresetSelector(presets []Preset, selectedIdx int, width int) string {
	var content string

	headerStyle := lipgloss.NewStyle().
		Foreground(TextDim).
		Italic(true).
		MarginBottom(1)

	content += headerStyle.Render("Quick Presets (press key to apply):") + "\n\n"

	for i, preset := range presets {
		isSelected := i == selectedIdx

		shortcutStyle := lipgloss.NewStyle().
			Foreground(lipgloss.Color("#000000")).
			Background(Accent).
			Bold(true).
			Padding(0, 1)

		nameStyle := lipgloss.NewStyle().Foreground(Text).PaddingLeft(1)
		descStyle := lipgloss.NewStyle().Foreground(TextDim).Italic(true)

		if isSelected {
			nameStyle = nameStyle.Foreground(Primary).Bold(true)
		}

		line := shortcutStyle.Render(preset.Shortcut) +
			nameStyle.Render(preset.Icon+" "+preset.Name) +
			" " + descStyle.Render("— "+preset.Description)

		if isSelected {
			content += lipgloss.NewStyle().Foreground(Primary).Render("▸ ") + line + "\n"
		} else {
			content += "  " + line + "\n"
		}
	}

	return content
}

func RenderPresetBadge(presetName string) string {
	if presetName == "" {
		return ""
	}

	return lipgloss.NewStyle().
		Foreground(lipgloss.Color("#000000")).
		Background(Accent).
		Bold(true).
		Padding(0, 1).
		Render("Preset: " + presetName)
}
