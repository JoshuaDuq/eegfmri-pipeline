package styles

import "github.com/charmbracelet/lipgloss"

// Theme: lab-instrument, refined. One dominant accent (teal), sharp highlight (cyan), clear hierarchy.
var (
	Primary   = lipgloss.Color("#0D9488") // Teal-600 — primary actions, focus, brand
	Secondary = lipgloss.Color("#475569")  // Slate-600 — section titles, borders
	Accent    = lipgloss.Color("#22D3EE")  // Cyan-400 — labels, badges, highlights
	Success   = lipgloss.Color("#10B981")  // Emerald-500
	Warning   = lipgloss.Color("#F59E0B")  // Amber-500
	Error     = lipgloss.Color("#EF4444")  // Red-500
	Muted     = lipgloss.Color("#64748B")  // Slate-500
	Text      = lipgloss.Color("#F1F5F9")  // Slate-100
	TextDim   = lipgloss.Color("#94A3B8")  // Slate-400
	BgDark    = lipgloss.Color("#0F172A")  // Slate-900
)
