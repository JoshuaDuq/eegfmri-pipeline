package styles

import "github.com/charmbracelet/lipgloss"

// Theme: Academic Research Terminal.
// Cool dark background, steel-blue brand, high readability, zero gimmicks.
var (
	Primary   = lipgloss.Color("#5B8DB8") // Steel-blue — brand, focus rings, active items
	Secondary = lipgloss.Color("#4A5568") // Cool-gray-600 — borders, dividers
	Accent    = lipgloss.Color("#7C9CBF") // Dusty-blue — secondary highlights
	Success   = lipgloss.Color("#68A87A") // Muted-green — success states
	Warning   = lipgloss.Color("#C9974A") // Muted-amber — warnings
	Error     = lipgloss.Color("#C47070") // Muted-red — errors
	Muted     = lipgloss.Color("#546378") // Blue-gray-500 — de-emphasized text
	Text      = lipgloss.Color("#D8E0EA") // Cool-white — primary text
	TextDim   = lipgloss.Color("#8A9BB0") // Cool-gray-400 — secondary text
	BgDark    = lipgloss.Color("#0D1117") // Near-black — app background
	Surface   = lipgloss.Color("#161C26") // Dark-navy — card/panel background
	Highlight = lipgloss.Color("#1A2D42") // Dark-steel — selected row background
	Border    = lipgloss.Color("#2D3748") // Cool-gray-700 — subtle borders
)
