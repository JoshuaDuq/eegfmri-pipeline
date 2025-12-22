package components

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/eeg-pipeline/tui/styles"
)

///////////////////////////////////////////////////////////////////
// Breadcrumb Component
///////////////////////////////////////////////////////////////////

// Breadcrumb renders a navigation breadcrumb trail
type Breadcrumb struct {
	Items     []string
	Separator string
}

func NewBreadcrumb(items ...string) Breadcrumb {
	return Breadcrumb{
		Items:     items,
		Separator: " › ",
	}
}

func (b Breadcrumb) View() string {
	if len(b.Items) == 0 {
		return ""
	}

	var parts []string
	for i, item := range b.Items {
		style := lipgloss.NewStyle().Foreground(styles.Muted)
		if i == len(b.Items)-1 {
			style = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		}
		parts = append(parts, style.Render(item))
	}

	sep := lipgloss.NewStyle().Foreground(styles.Secondary).Render(b.Separator)
	return strings.Join(parts, sep)
}

///////////////////////////////////////////////////////////////////
// Toast/Notification Component
///////////////////////////////////////////////////////////////////

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
	if t.TicksLeft <= 0 {
		t.Visible = false
	}
}

func (t Toast) View() string {
	if !t.Visible {
		return ""
	}

	var icon string
	var bgColor, fgColor lipgloss.Color

	switch t.Type {
	case ToastSuccess:
		icon = styles.CheckMark
		bgColor = styles.Success
		fgColor = lipgloss.Color("#000000")
	case ToastWarning:
		icon = styles.WarningMark
		bgColor = styles.Warning
		fgColor = lipgloss.Color("#000000")
	case ToastError:
		icon = styles.CrossMark
		bgColor = styles.Error
		fgColor = lipgloss.Color("#FFFFFF")
	default:
		icon = "ℹ"
		bgColor = styles.Accent
		fgColor = lipgloss.Color("#000000")
	}

	style := lipgloss.NewStyle().
		Foreground(fgColor).
		Background(bgColor).
		Padding(0, 2).
		Bold(true)

	return style.Render(icon + " " + t.Message)
}

///////////////////////////////////////////////////////////////////
// Help Overlay Component
///////////////////////////////////////////////////////////////////

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

	// Title
	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		MarginBottom(1).
		Render(h.Title)
	content.WriteString(title + "\n\n")

	keyStyle := lipgloss.NewStyle().
		Foreground(styles.Accent).
		Bold(true).
		Width(12)
	descStyle := lipgloss.NewStyle().
		Foreground(styles.Text)
	sectionStyle := lipgloss.NewStyle().
		Foreground(styles.TextDim).
		Bold(true).
		MarginTop(1)

	sectionOrder := []string{"Navigation", "Selection", "Actions", "General"}
	for _, sectionName := range sectionOrder {
		items, ok := h.Sections[sectionName]
		if !ok || len(items) == 0 {
			continue
		}

		content.WriteString(sectionStyle.Render(sectionName) + "\n")
		for _, item := range items {
			content.WriteString(keyStyle.Render(item.Key) + descStyle.Render(item.Description) + "\n")
		}
		content.WriteString("\n")
	}

	// Dismiss hint
	content.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Render("Press ? or Esc to close"))

	box := lipgloss.NewStyle().
		Border(lipgloss.DoubleBorder()).
		BorderForeground(styles.Primary).
		Padding(1, 2).
		Width(h.Width)

	return box.Render(content.String())
}

///////////////////////////////////////////////////////////////////
// Spinner Component
///////////////////////////////////////////////////////////////////

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

func NewDotsSpinner(label string) Spinner {
	return Spinner{
		Frames: []string{"⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"},
		Index:  0,
		Label:  label,
	}
}

func NewBlockSpinner(label string) Spinner {
	return Spinner{
		Frames: []string{"▖", "▘", "▝", "▗"},
		Index:  0,
		Label:  label,
	}
}

func (s *Spinner) Tick() {
	s.Index = (s.Index + 1) % len(s.Frames)
}

func (s Spinner) View() string {
	frame := lipgloss.NewStyle().Foreground(styles.Accent).Render(s.Frames[s.Index])
	label := lipgloss.NewStyle().Foreground(styles.Text).Render(s.Label)
	return frame + " " + label
}

///////////////////////////////////////////////////////////////////
// Progress Indicator Component
///////////////////////////////////////////////////////////////////

type ProgressStyle int

const (
	ProgressStyleBlock ProgressStyle = iota
	ProgressStyleLine
	ProgressStyleDots
)

type Progress struct {
	Current int
	Total   int
	Width   int
	Style   ProgressStyle
	Label   string
}

func NewProgress(current, total, width int) Progress {
	return Progress{
		Current: current,
		Total:   total,
		Width:   width,
		Style:   ProgressStyleBlock,
	}
}

func (p *Progress) SetProgress(current int) {
	p.Current = current
}

func (p Progress) Percent() float64 {
	if p.Total == 0 {
		return 0
	}
	return float64(p.Current) / float64(p.Total)
}

func (p Progress) View() string {
	pct := p.Percent()
	filled := int(pct * float64(p.Width))
	if filled > p.Width {
		filled = p.Width
	}

	var bar string
	switch p.Style {
	case ProgressStyleLine:
		bar = lipgloss.NewStyle().Foreground(styles.Primary).Render(strings.Repeat("━", filled))
		bar += lipgloss.NewStyle().Foreground(styles.Secondary).Render(strings.Repeat("─", p.Width-filled))
	case ProgressStyleDots:
		bar = lipgloss.NewStyle().Foreground(styles.Primary).Render(strings.Repeat("●", filled))
		bar += lipgloss.NewStyle().Foreground(styles.Secondary).Render(strings.Repeat("○", p.Width-filled))
	default: // Block
		bar = lipgloss.NewStyle().Foreground(styles.Primary).Render(strings.Repeat("█", filled))
		bar += lipgloss.NewStyle().Foreground(styles.Secondary).Render(strings.Repeat("░", p.Width-filled))
	}

	pctStr := lipgloss.NewStyle().Bold(true).Foreground(styles.Primary).Render(
		strings.Repeat(" ", 3-len(string(rune(int(pct*100))))) + string(rune(int(pct*100))) + "%")

	return bar + " " + pctStr
}

///////////////////////////////////////////////////////////////////
// Scroll Indicator Component
///////////////////////////////////////////////////////////////////

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
	var parts []string

	upStyle := lipgloss.NewStyle().Foreground(styles.Muted)
	downStyle := lipgloss.NewStyle().Foreground(styles.Muted)

	if s.CanScrollUp() {
		upStyle = upStyle.Foreground(styles.Primary)
	}
	if s.CanScrollDown() {
		downStyle = downStyle.Foreground(styles.Primary)
	}

	parts = append(parts, upStyle.Render("▲"))
	parts = append(parts, downStyle.Render("▼"))

	return strings.Join(parts, " ")
}

///////////////////////////////////////////////////////////////////
// Info Panel Component
///////////////////////////////////////////////////////////////////

type InfoRow struct {
	Label string
	Value string
	Style lipgloss.Style
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
		content.WriteString(styles.SectionTitleStyle.Render(" "+p.Title+" ") + "\n\n")
	}

	labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(p.LabelWidth)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Text)

	for _, row := range p.Rows {
		vs := valueStyle
		// Check if a custom style was provided by checking if it has any foreground set
		if row.Style.String() != "" {
			vs = row.Style
		}
		content.WriteString(labelStyle.Render(row.Label+":") + " " + vs.Render(row.Value) + "\n")
	}

	return content.String()
}

///////////////////////////////////////////////////////////////////
// Status Bar Component
///////////////////////////////////////////////////////////////////

type StatusBar struct {
	LeftItems   []string
	CenterItems []string
	RightItems  []string
	Width       int
}

func NewStatusBar(width int) StatusBar {
	return StatusBar{Width: width}
}

func (s *StatusBar) SetLeft(items ...string) {
	s.LeftItems = items
}

func (s *StatusBar) SetCenter(items ...string) {
	s.CenterItems = items
}

func (s *StatusBar) SetRight(items ...string) {
	s.RightItems = items
}

func (s StatusBar) View() string {
	style := lipgloss.NewStyle().
		Foreground(styles.TextDim).
		Width(s.Width)

	left := strings.Join(s.LeftItems, " │ ")
	center := strings.Join(s.CenterItems, " │ ")
	right := strings.Join(s.RightItems, " │ ")

	// Calculate spacing
	leftLen := lipgloss.Width(left)
	centerLen := lipgloss.Width(center)
	rightLen := lipgloss.Width(right)

	totalContent := leftLen + centerLen + rightLen
	remainingSpace := s.Width - totalContent
	if remainingSpace < 0 {
		remainingSpace = 0
	}

	leftPad := remainingSpace / 2
	rightPad := remainingSpace - leftPad

	content := left + strings.Repeat(" ", leftPad) + center + strings.Repeat(" ", rightPad) + right

	return style.Render(content)
}

///////////////////////////////////////////////////////////////////
// Confirmation Dialog Component
///////////////////////////////////////////////////////////////////

type ConfirmDialog struct {
	Title       string
	Message     string
	ConfirmText string
	CancelText  string
	Visible     bool
	Confirmed   bool
	Width       int
}

func NewConfirmDialog(title, message string, width int) ConfirmDialog {
	return ConfirmDialog{
		Title:       title,
		Message:     message,
		ConfirmText: "Yes",
		CancelText:  "No",
		Width:       width,
	}
}

func (c *ConfirmDialog) Show() {
	c.Visible = true
	c.Confirmed = false
}

func (c *ConfirmDialog) Hide() {
	c.Visible = false
}

func (c *ConfirmDialog) Confirm() {
	c.Confirmed = true
	c.Visible = false
}

func (c *ConfirmDialog) Cancel() {
	c.Confirmed = false
	c.Visible = false
}

func (c ConfirmDialog) View() string {
	if !c.Visible {
		return ""
	}

	var content strings.Builder

	// Title
	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Warning).
		Render(c.Title)
	content.WriteString(title + "\n\n")

	// Message
	content.WriteString(lipgloss.NewStyle().Foreground(styles.Text).Render(c.Message) + "\n\n")

	// Buttons
	confirmBtn := styles.BadgeSuccessStyle.Render(" [Y] " + c.ConfirmText + " ")
	cancelBtn := styles.BadgeErrorStyle.Render(" [N] " + c.CancelText + " ")
	content.WriteString(confirmBtn + "  " + cancelBtn)

	box := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(styles.Warning).
		Padding(1, 2).
		Width(c.Width)

	return box.Render(content.String())
}

///////////////////////////////////////////////////////////////////
// Table Component
///////////////////////////////////////////////////////////////////

type TableColumn struct {
	Header string
	Width  int
}

type TableRow struct {
	Cells    []string
	Selected bool
}

type Table struct {
	Columns     []TableColumn
	Rows        []TableRow
	CursorIndex int
	ScrollTop   int
	VisibleRows int
}

func NewTable(columns []TableColumn, visibleRows int) Table {
	return Table{
		Columns:     columns,
		VisibleRows: visibleRows,
	}
}

func (t *Table) AddRow(cells ...string) {
	t.Rows = append(t.Rows, TableRow{Cells: cells})
}

func (t *Table) CursorUp() {
	if t.CursorIndex > 0 {
		t.CursorIndex--
		if t.CursorIndex < t.ScrollTop {
			t.ScrollTop = t.CursorIndex
		}
	}
}

func (t *Table) CursorDown() {
	if t.CursorIndex < len(t.Rows)-1 {
		t.CursorIndex++
		if t.CursorIndex >= t.ScrollTop+t.VisibleRows {
			t.ScrollTop = t.CursorIndex - t.VisibleRows + 1
		}
	}
}

func (t *Table) ToggleSelected() {
	if t.CursorIndex < len(t.Rows) {
		t.Rows[t.CursorIndex].Selected = !t.Rows[t.CursorIndex].Selected
	}
}

func (t Table) View() string {
	var content strings.Builder

	// Header
	headerStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		Underline(true)

	var headerCells []string
	for _, col := range t.Columns {
		cell := headerStyle.Width(col.Width).Render(col.Header)
		headerCells = append(headerCells, cell)
	}
	content.WriteString(strings.Join(headerCells, "") + "\n")

	// Rows
	endIdx := t.ScrollTop + t.VisibleRows
	if endIdx > len(t.Rows) {
		endIdx = len(t.Rows)
	}

	for i := t.ScrollTop; i < endIdx; i++ {
		row := t.Rows[i]
		isCursor := i == t.CursorIndex

		rowStyle := lipgloss.NewStyle().Foreground(styles.Text)
		if isCursor {
			rowStyle = rowStyle.Bold(true).Foreground(styles.Primary)
		}

		var rowCells []string
		for j, cell := range row.Cells {
			if j < len(t.Columns) {
				cellContent := cell
				if j == 0 && row.Selected {
					cellContent = styles.CheckMark + " " + cellContent
				} else if j == 0 {
					cellContent = "  " + cellContent
				}
				c := rowStyle.Width(t.Columns[j].Width).Render(cellContent)
				rowCells = append(rowCells, c)
			}
		}
		content.WriteString(strings.Join(rowCells, "") + "\n")
	}

	return content.String()
}

///////////////////////////////////////////////////////////////////
// ASCII Art/Logo Component
///////////////////////////////////////////////////////////////////

func RenderLogo(style lipgloss.Style) string {
	logo := `
 ███████╗███████╗ ██████╗ 
 ██╔════╝██╔════╝██╔════╝ 
 █████╗  █████╗  ██║  ███╗
 ██╔══╝  ██╔══╝  ██║   ██║
 ███████╗███████╗╚██████╔╝
 ╚══════╝╚══════╝ ╚═════╝ `
	return style.Render(logo)
}

func RenderMiniLogo(style lipgloss.Style) string {
	return style.Render("◆ EEG Pipeline")
}
