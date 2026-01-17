package components

import (
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

func (t Toast) toastColors() (icon string, bgColor, fgColor lipgloss.Color) {
	switch t.Type {
	case ToastSuccess:
		return styles.CheckMark, styles.Success, lipgloss.Color("#000000")
	case ToastWarning:
		return styles.WarningMark, styles.Warning, lipgloss.Color("#000000")
	case ToastError:
		return styles.CrossMark, styles.Error, lipgloss.Color("#FFFFFF")
	default:
		return "ℹ", styles.Accent, lipgloss.Color("#000000")
	}
}

func (t Toast) View() string {
	if !t.Visible {
		return ""
	}

	icon, bgColor, fgColor := t.toastColors()
	style := lipgloss.NewStyle().
		Foreground(fgColor).
		Background(bgColor).
		Padding(0, 2).
		Bold(true)

	return style.Render(icon + " " + t.Message)
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

	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		MarginBottom(1)
	content.WriteString(titleStyle.Render(h.Title) + "\n\n")

	keyStyle := lipgloss.NewStyle().
		Foreground(styles.Accent).
		Bold(true).
		Width(helpKeyWidth)
	descStyle := lipgloss.NewStyle().Foreground(styles.Text)
	sectionStyle := lipgloss.NewStyle().
		Foreground(styles.TextDim).
		Bold(true).
		MarginTop(1)

	sectionOrder := []string{"Navigation", "Selection", "Actions", "General"}
	for _, sectionName := range sectionOrder {
		items, exists := h.Sections[sectionName]
		if !exists || len(items) == 0 {
			continue
		}

		content.WriteString(sectionStyle.Render(sectionName) + "\n")
		for _, item := range items {
			keyText := keyStyle.Render(item.Key)
			descText := descStyle.Render(item.Description)
			content.WriteString(keyText + descText + "\n")
		}
		content.WriteString("\n")
	}

	dismissHint := lipgloss.NewStyle().Foreground(styles.Muted).Render("Press ? or Esc to close")
	content.WriteString(dismissHint)

	box := lipgloss.NewStyle().
		Border(lipgloss.DoubleBorder()).
		BorderForeground(styles.Primary).
		Padding(1, 2).
		Width(h.Width)

	return box.Render(content.String())
}

type Spinner struct {
	Frames []string
	Index  int
	Label  string
}

func NewSpinner(label string) Spinner {
	return Spinner{
		Frames: []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"},
		Index:  0,
		Label:  label,
	}
}

func (s *Spinner) Tick() {
	s.Index = (s.Index + 1) % len(s.Frames)
}

func (s Spinner) View() string {
	frameStyle := lipgloss.NewStyle().Foreground(styles.Accent)
	labelStyle := lipgloss.NewStyle().Foreground(styles.Text)
	return frameStyle.Render(s.Frames[s.Index]) + " " + labelStyle.Render(s.Label)
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
	baseStyle := lipgloss.NewStyle().Foreground(styles.Muted)
	activeStyle := lipgloss.NewStyle().Foreground(styles.Primary)

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
	Label string
	Value string
	Style lipgloss.Style
}

func (r InfoRow) hasCustomStyle() bool {
	emptyStyle := lipgloss.Style{}
	return r.Style.String() != emptyStyle.String()
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
	p.Rows = append(p.Rows, InfoRow{Label: label, Value: value, Style: style})
}

func (p InfoPanel) View() string {
	var content strings.Builder

	if p.Title != "" {
		titleText := " " + p.Title + " "
		content.WriteString(styles.SectionTitleStyle.Render(titleText) + "\n\n")
	}

	labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(p.LabelWidth)
	defaultValueStyle := lipgloss.NewStyle().Foreground(styles.Text)

	for _, row := range p.Rows {
		valueStyle := defaultValueStyle
		if row.hasCustomStyle() {
			valueStyle = row.Style
		}

		labelText := labelStyle.Render(row.Label + ":")
		valueText := valueStyle.Render(row.Value)
		content.WriteString(labelText + " " + valueText + "\n")
	}

	return content.String()
}
