package styles

import (
	"os"
	"sync"

	"github.com/charmbracelet/lipgloss"
)

// Theme: Monochrome Research Terminal.
// Pure grayscale palette. All emphasis is communicated through weight (bold
// vs. regular) and value (bright vs. dim), never hue. The only colored tones
// are reserved for semantically critical states that would be unsafe to
// convey through typography alone — Warning (amber) and Error (muted red).
// Everything else — Primary/Accent/Success/Info — resolves to grayscale,
// deliberately neutralizing the previous steel-blue accent.
var (
	// "Primary" is bright white; used with Bold to mark focus and active
	// selection. The selection is recognized by its weight, not its hue.
	Primary = lipgloss.Color("#E8EAED")
	// Used for subtle structural accents (thin bar indicators, small rules).
	Secondary = lipgloss.Color("#3A3B3D")
	// A slightly dimmer white used for secondary emphasis.
	Accent = lipgloss.Color("#A5A8AC")
	// Success resolves to the same bright white as Primary: meaning is
	// carried by the ✓ mark, not by color.
	Success = lipgloss.Color("#E8EAED")
	// Warning and Error are kept as desaturated tones so warning/error
	// states remain distinguishable in monochrome contexts and for users
	// with custom palettes.
	Warning = lipgloss.Color("#C9974A")
	Error   = lipgloss.Color("#C47070")
	// Info is grayscale; informational highlights rely on Bold instead of
	// a colored accent.
	Info  = lipgloss.Color("#A5A8AC")
	Muted = lipgloss.Color("#5F6163") // De-emphasized text.
	Text  = lipgloss.Color("#E8EAED") // Primary text — warm near-white.
	TextDim = lipgloss.Color("#9AA0A6") // Secondary text.
	BgDark  = lipgloss.Color("#0D0E10") // App background — near-black.
	Surface = lipgloss.Color("#161719") // Card/panel background.
	SurfaceAlt = lipgloss.Color("#1D1E21") // Slightly-raised panel surface.
	Border = lipgloss.Color("#2A2B2D")    // Subtle panel borders.
	BorderBright = lipgloss.Color("#3A3B3D") // Focus-adjacent borders.
)

var noColorOnce sync.Once

// ApplyNoColorProfile respects the NO_COLOR environment variable (https://no-color.org).
// Safe to call repeatedly; only applies the first time.
func ApplyNoColorProfile() {
	noColorOnce.Do(func() {
		if v := os.Getenv("NO_COLOR"); v != "" {
			lipgloss.SetColorProfile(0) // 0 == termenv.Ascii (monochrome)
		}
	})
}
