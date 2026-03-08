package styles

import "github.com/charmbracelet/lipgloss"

// Theme: Modern Research Dashboard.
// Warm dark background, signature teal brand, generous whitespace feel.
var (
	Primary   = lipgloss.Color("#2DD4BF") // Teal-400 — brand, focus rings, active items
	Secondary = lipgloss.Color("#475569") // Slate-600 — borders, dividers
	Accent    = lipgloss.Color("#38BDF8") // Sky-400 — highlights, badges, key hints
	Success   = lipgloss.Color("#34D399") // Emerald-400
	Warning   = lipgloss.Color("#FBBF24") // Amber-400
	Error     = lipgloss.Color("#F87171") // Red-400
	Muted     = lipgloss.Color("#64748B") // Slate-500
	Text      = lipgloss.Color("#E2E8F0") // Slate-200 — primary text
	TextDim   = lipgloss.Color("#94A3B8") // Slate-400 — secondary text
	BgDark    = lipgloss.Color("#0F172A") // Slate-900 — app background
	Surface   = lipgloss.Color("#1E293B") // Slate-800 — card/panel background
	Highlight = lipgloss.Color("#0F3D38") // Teal-950 — selected row background
	Border    = lipgloss.Color("#334155") // Slate-700 — subtle borders
)
