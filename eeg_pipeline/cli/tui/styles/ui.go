package styles

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// FooterHint is a key-binding hint rendered in a view's footer bar.
//
// Fields:
//   - Key: the key chord to show (e.g. "Enter", "↑/↓", "Ctrl+K").
//   - Label: full, descriptive label (e.g. "Navigate", "Open wizard").
//   - Compact: short label used when horizontal space is tight; falls back to Label when empty.
//   - Priority: 0 = essential (primary chip), 1 = useful (secondary chip), 2 = optional.
//     Higher-priority hints are dropped first when the footer must shrink.
type FooterHint struct {
	Key      string
	Label    string
	Compact  string
	Priority int
}

// RenderFooterHints renders a list of FooterHint separated by the footer hint
// separator, gracefully degrading when the target width is exceeded:
//  1. Full labels (all priorities).
//  2. Compact labels (all priorities).
//  3. Compact labels, dropping priority 2 first, then 1.
//
// If every strategy still overflows, the most-essential (priority 0) compact
// strip is returned even if wider than `width` (clamp is the caller's job).
func RenderFooterHints(width int, hints []FooterHint) string {
	if len(hints) == 0 {
		return ""
	}

	render := func(useCompact bool, maxPriority int) string {
		parts := make([]string, 0, len(hints))
		for _, h := range hints {
			if h.Priority > maxPriority {
				continue
			}
			label := h.Label
			if useCompact && h.Compact != "" {
				label = h.Compact
			}
			if h.Priority == 0 {
				parts = append(parts, RenderKeyHint(h.Key, label))
			} else {
				parts = append(parts, RenderKeyHintSecondary(h.Key, label))
			}
		}
		return strings.Join(parts, RenderFooterSeparator())
	}

	if full := render(false, 2); lipgloss.Width(full) <= width {
		return full
	}
	if compact := render(true, 2); lipgloss.Width(compact) <= width {
		return compact
	}
	for maxPriority := 1; maxPriority >= 0; maxPriority-- {
		if compact := render(true, maxPriority); compact != "" && lipgloss.Width(compact) <= width {
			return compact
		}
	}
	return render(true, 0)
}

// PillKind selects a pill color scheme.
type PillKind int

const (
	PillPrimary PillKind = iota
	PillAccent
	PillSuccess
	PillWarning
	PillError
	PillMuted
	PillOutline
)

// RenderPill renders a small styled badge with the given text.
func RenderPill(text string, kind PillKind) string {
	switch kind {
	case PillAccent:
		return PillAccentStyle.Render(text)
	case PillSuccess:
		return PillSuccessStyle.Render(text)
	case PillWarning:
		return PillWarningStyle.Render(text)
	case PillError:
		return PillErrorStyle.Render(text)
	case PillMuted:
		return PillMutedStyle.Render(text)
	case PillOutline:
		return PillOutlineStyle.Render(text)
	default:
		return PillPrimaryStyle.Render(text)
	}
}

// RenderStepPill renders a "N/Total" step counter as muted bracketed text
// (e.g. `[3/8]`). Bracketed form keeps the header on a single line and reads
// as CLI metadata rather than a UI chip.
func RenderStepPill(current, total int) string {
	if total <= 0 {
		return ""
	}
	if current < 1 {
		current = 1
	}
	if current > total {
		current = total
	}
	bracket := lipgloss.NewStyle().Foreground(Border)
	counter := lipgloss.NewStyle().Foreground(TextDim).Bold(true)
	return bracket.Render("[") +
		counter.Render(fmt.Sprintf("%d/%d", current, total)) +
		bracket.Render("]")
}

// RenderUppercaseHeading renders a section heading as plain uppercase bold
// text. Matches the convention already used by sub-section labels like
// "DETAILS" / "WORKSPACE" / "FOCUS" — quiet, compact, unambiguously a header.
func RenderUppercaseHeading(title string) string {
	if title == "" {
		return ""
	}
	return lipgloss.NewStyle().Foreground(Text).Bold(true).Render(strings.ToUpper(title))
}

// RenderLabelValueInline renders "label · value" with label in muted text
// and value in bold. Useful for header metadata like "task · thermalactive".
func RenderLabelValueInline(label, value string) string {
	if value == "" {
		return ""
	}
	lbl := lipgloss.NewStyle().Foreground(Muted).Render(label)
	sep := lipgloss.NewStyle().Foreground(Border).Render(" · ")
	val := lipgloss.NewStyle().Foreground(Text).Bold(true).Render(value)
	return lbl + sep + val
}

// RenderAccentBar returns a thin vertical accent glyph rendered in the given
// color. Used on selected list rows to signal focus with a quiet, left-edge
// highlight instead of a full-width background.
func RenderAccentBar(focused bool) string {
	if focused {
		return lipgloss.NewStyle().Foreground(Primary).Bold(true).Render("▎")
	}
	return " "
}

// RenderKeyBadge renders a bracketed single-key affordance like `[1]` or
// `[B]`. When `focused` is true the key is drawn in the primary accent color
// and bold; otherwise it's in a dim tone. Used as an inline shortcut chip
// (e.g., next to quick-action rows) in place of background-filled pills.
func RenderKeyBadge(key string, focused bool) string {
	textStyle := FooterKeyTextSecondary
	if focused {
		textStyle = FooterKeyTextPrimary
	}
	return FooterKeyBracketStyle.Render("[") +
		textStyle.Render(key) +
		FooterKeyBracketStyle.Render("]")
}

// RenderStepperBar renders a thin horizontal progress indicator that fills
// proportionally to `filled / total`. Uses a single monochrome fill (bright
// white on the progressed segment, subtle border color on the remainder) —
// progress is communicated by width alone, never by shifting hue.
func RenderStepperBar(filled, total, width int) string {
	if width <= 0 || total <= 0 {
		return ""
	}
	if filled < 0 {
		filled = 0
	}
	if filled > total {
		filled = total
	}
	if width < 4 {
		width = 4
	}

	filledW := width * filled / total
	emptyW := width - filledW
	filledStr := lipgloss.NewStyle().Foreground(Text).Render(strings.Repeat("━", filledW))
	emptyStr := lipgloss.NewStyle().Foreground(Border).Render(strings.Repeat("─", emptyW))
	return filledStr + emptyStr
}

// BreadcrumbStep is one segment of a wizard stepper/breadcrumb.
type BreadcrumbStep struct {
	Name string
}

// RenderBreadcrumb renders a step rail like
//
//	✓ Mode  ·  ✓ Subjects  ·  Bands  ·  ROIs  ·  Time
//
// with the current step bold+underlined, completed steps checked and muted,
// and upcoming steps rendered in the subtle border color. When `compact` is
// true, completed steps collapse to just the check mark and separators become
// single spaces.
func RenderBreadcrumb(steps []BreadcrumbStep, currentIdx int, compact bool) string {
	if len(steps) == 0 {
		return ""
	}

	checkStyle := lipgloss.NewStyle().Foreground(Success)
	doneStyle := lipgloss.NewStyle().Foreground(Muted)
	currentStyle := lipgloss.NewStyle().Foreground(Primary).Bold(true)
	futureStyle := lipgloss.NewStyle().Foreground(Border)
	// Chevron connector reads as a path/progression (Mode › Subjects › Bands),
	// which fits the wizard metaphor better than a generic middle dot.
	connector := lipgloss.NewStyle().Foreground(Border).Render("  ›  ")
	if compact {
		connector = " "
	}

	parts := make([]string, 0, len(steps))
	for i, s := range steps {
		var seg string
		switch {
		case i < currentIdx:
			if compact {
				seg = checkStyle.Render(CheckMark)
			} else {
				seg = checkStyle.Render(CheckMark) + doneStyle.Render(" "+s.Name)
			}
		case i == currentIdx:
			seg = currentStyle.Render(s.Name)
		default:
			seg = futureStyle.Render(s.Name)
		}
		parts = append(parts, seg)
	}
	return strings.Join(parts, connector)
}
