package components

import (
	"sort"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/eeg-pipeline/tui/styles"
)

type ToastType int

const (
	ToastInfo ToastType = iota
	ToastSuccess
	ToastWarning
	ToastError
)

const (
	ToastDurationShort  = 16 // ~1.6s at 100ms tick interval
	ToastDurationMedium = 24 // ~2.4s
	ToastDurationLong   = 40 // ~4.0s
)

type Toast struct {
	Message   string
	Type      ToastType
	Visible   bool
	TicksLeft int
}

func NewToast(message string, toastType ToastType, durationTicks int) Toast {
	return Toast{
		Message:   message,
		Type:      toastType,
		Visible:   true,
		TicksLeft: durationTicks,
	}
}

func (t *Toast) Tick() {
	if t.TicksLeft > 0 {
		t.TicksLeft--
	}
	if t.TicksLeft == 0 {
		t.Visible = false
	}
}

func (t Toast) toastStyle() (icon string, color lipgloss.Color) {
	switch t.Type {
	case ToastSuccess:
		return styles.CheckMark, styles.Success
	case ToastWarning:
		return styles.WarningMark, styles.Warning
	case ToastError:
		return styles.CrossMark, styles.Error
	default:
		return styles.ActiveMark, styles.Accent
	}
}

func (t Toast) View() string {
	if !t.Visible {
		return ""
	}
	icon, color := t.toastStyle()
	return lipgloss.NewStyle().
		Foreground(color).Bold(true).
		Render(icon + "  " + t.Message)
}

const helpKeyWidth = 12

type HelpItem struct {
	Key         string
	Description string
}

type HelpOverlay struct {
	Title    string
	Sections map[string][]HelpItem
	Visible  bool
	Width    int
}

func NewHelpOverlay(title string, width int) HelpOverlay {
	return HelpOverlay{
		Title:    title,
		Sections: make(map[string][]HelpItem),
		Visible:  false,
		Width:    width,
	}
}

func (h *HelpOverlay) AddSection(name string, items []HelpItem) {
	h.Sections[name] = items
}

func (h *HelpOverlay) Toggle() {
	h.Visible = !h.Visible
}

func (h HelpOverlay) View() string {
	if !h.Visible {
		return ""
	}

	var content strings.Builder
	innerWidth := h.Width - styles.PanelStyle.GetHorizontalFrameSize()
	if innerWidth < 1 {
		innerWidth = 1
	}

	titleStyle := lipgloss.NewStyle().Bold(true).Foreground(styles.Text)
	content.WriteString(titleStyle.Render(h.Title) + "\n")
	content.WriteString(styles.RenderHeaderSeparator(innerWidth) + "\n\n")

	// Bracket-style key glyphs (e.g. `[Enter]`) rather than background-filled
	// chips — cleaner on a research terminal and consistent with the footer
	// hint treatment used elsewhere.
	renderKey := func(k string) string {
		return styles.FooterKeyBracketStyle.Render("[") +
			styles.FooterKeyTextPrimary.Render(k) +
			styles.FooterKeyBracketStyle.Render("]")
	}
	descStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	sectionStyle := lipgloss.NewStyle().
		Foreground(styles.Muted).
		Bold(true)

	sectionOrder := []string{"Navigation", "Selection", "Actions", "General"}
	rendered := make(map[string]bool)
	for _, sectionName := range sectionOrder {
		items, exists := h.Sections[sectionName]
		if !exists || len(items) == 0 {
			continue
		}
		content.WriteString(sectionStyle.Render(strings.ToUpper(sectionName)) + "\n")
		for _, item := range items {
			keyText := styles.FitLine(renderKey(item.Key), helpKeyWidth)
			line := keyText + " " + descStyle.Render(item.Description)
			content.WriteString(styles.TruncateLine(line, innerWidth) + "\n")
		}
		content.WriteString("\n")
		rendered[sectionName] = true
	}

	var extra []string
	for name := range h.Sections {
		if !rendered[name] && len(h.Sections[name]) > 0 {
			extra = append(extra, name)
		}
	}
	sort.Strings(extra)
	for _, sectionName := range extra {
		content.WriteString(sectionStyle.Render(strings.ToUpper(sectionName)) + "\n")
		for _, item := range h.Sections[sectionName] {
			keyText := styles.FitLine(renderKey(item.Key), helpKeyWidth)
			line := keyText + " " + descStyle.Render(item.Description)
			content.WriteString(styles.TruncateLine(line, innerWidth) + "\n")
		}
		content.WriteString("\n")
	}

	closeHint := lipgloss.NewStyle().Foreground(styles.Muted).
		Render("? or Esc to close")
	content.WriteString(closeHint)

	return styles.RenderNoWrapBlock(styles.PanelStyle, content.String(), h.Width)
}

// spinnerFrames is a single shared braille animation. Braille gives a smooth
// indeterminate motion at low visual weight — the glyph never changes width
// or height, so it sits cleanly inline with surrounding text.
var spinnerFrames = []string{
	"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
}

// spinnerFrameTicks slows the animation cadence (frames advance every N ticks)
// so the spinner reads as a calm pulse rather than a frantic strobe at the
// 100 ms tick interval used app-wide.
const spinnerFrameTicks = 2

type Spinner struct {
	Label string
	tick  int
}

func NewSpinner(label string) Spinner {
	return Spinner{Label: label}
}

func (s *Spinner) Tick() {
	s.tick++
}

// Glyph returns just the rotating braille glyph, styled in Primary at bold
// weight. Use this when composing the spinner with caller-owned label/message
// text (e.g. wizard loading banners) to avoid duplicating the label. The
// Primary tone (rather than Accent) gives the glyph enough contrast against
// the dark surface for the motion to read as the focal status indicator
// without becoming chromatic noise — colour is reserved for state, brightness
// is the emphasis dial.
func (s Spinner) Glyph() string {
	frame := spinnerFrames[(s.tick/spinnerFrameTicks)%len(spinnerFrames)]
	return lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(frame)
}

// View renders the spinner glyph followed by the label, both at low weight.
// Used by stand-alone loading screens where the spinner is its own
// self-contained line.
func (s Spinner) View() string {
	labelStyle := lipgloss.NewStyle().Foreground(styles.Muted)
	if s.Label == "" {
		return s.Glyph()
	}
	return s.Glyph() + "  " + labelStyle.Render(s.Label)
}

type ScrollIndicator struct {
	Current    int
	Total      int
	ViewHeight int
}

func (s ScrollIndicator) CanScrollUp() bool {
	return s.Current > 0
}

func (s ScrollIndicator) CanScrollDown() bool {
	return s.Current+s.ViewHeight < s.Total
}

func (s ScrollIndicator) View() string {
	baseStyle := lipgloss.NewStyle().Foreground(styles.Border)
	activeStyle := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)

	upArrow := baseStyle.Render("▲")
	if s.CanScrollUp() {
		upArrow = activeStyle.Render("▲")
	}

	downArrow := baseStyle.Render("▼")
	if s.CanScrollDown() {
		downArrow = activeStyle.Render("▼")
	}

	return upArrow + " " + downArrow
}

type InfoRow struct {
	Label    string
	Value    string
	Style    lipgloss.Style
	HasStyle bool
}

func (r InfoRow) hasCustomStyle() bool {
	return r.HasStyle
}

type InfoPanel struct {
	Title      string
	Rows       []InfoRow
	LabelWidth int
}

func NewInfoPanel(title string, labelWidth int) InfoPanel {
	return InfoPanel{
		Title:      title,
		LabelWidth: labelWidth,
	}
}

func (p *InfoPanel) AddRow(label, value string) {
	p.Rows = append(p.Rows, InfoRow{Label: label, Value: value})
}

func (p *InfoPanel) AddStyledRow(label, value string, style lipgloss.Style) {
	p.Rows = append(p.Rows, InfoRow{Label: label, Value: value, Style: style, HasStyle: true})
}

func (p InfoPanel) View() string {
	var content strings.Builder

	if p.Title != "" {
		content.WriteString(styles.RenderSectionLabel(p.Title) + "\n")
	}

	labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(p.LabelWidth)
	defaultValueStyle := lipgloss.NewStyle().Foreground(styles.Text)

	for _, row := range p.Rows {
		valueStyle := defaultValueStyle
		if row.hasCustomStyle() {
			valueStyle = row.Style
		}

		labelText := labelStyle.Render(row.Label)
		valueText := valueStyle.Render(row.Value)
		content.WriteString(labelText + " " + valueText + "\n")
	}

	return content.String()
}

// DotsLoader is a lightweight animated "·  / ··  / ···" loader.
// Call Advance() on each tick (driven by the parent model's tick loop).
type DotsLoader struct {
	Label string
	tick  int
}

const dotsLoaderFrameTicks = 3

var dotsLoaderFrames = [4]string{"   ", "·  ", "·· ", "···"}

func NewDotsLoader(label string) DotsLoader {
	return DotsLoader{Label: label}
}

func (d *DotsLoader) Advance() {
	d.tick++
}

// View renders the current animation frame: label followed by 0–3 animated dots.
// Frames advance every dotsLoaderFrameTicks ticks (~300 ms at 100 ms/tick).
func (d DotsLoader) View() string {
	frame := dotsLoaderFrames[(d.tick/dotsLoaderFrameTicks)%len(dotsLoaderFrames)]
	dotsStyle := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
	labelStyle := lipgloss.NewStyle().Foreground(styles.Muted)
	return labelStyle.Render(d.Label) + dotsStyle.Render(frame)
}
