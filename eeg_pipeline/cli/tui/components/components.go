package components

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/eeg-pipeline/tui/styles"
)

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
	lastIndex := len(b.Items) - 1
	for i, item := range b.Items {
		style := lipgloss.NewStyle().Foreground(styles.Muted)
		if i == lastIndex {
			style = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		}
		parts = append(parts, style.Render(item))
	}

	separator := lipgloss.NewStyle().Foreground(styles.Secondary).Render(b.Separator)
	return strings.Join(parts, separator)
}

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
	frameStyle := lipgloss.NewStyle().Foreground(styles.Accent)
	labelStyle := lipgloss.NewStyle().Foreground(styles.Text)
	return frameStyle.Render(s.Frames[s.Index]) + " " + labelStyle.Render(s.Label)
}

const percentagePaddingWidth = 3

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

func (p Progress) renderBar(filled, empty int) string {
	filledStyle := lipgloss.NewStyle().Foreground(styles.Primary)
	emptyStyle := lipgloss.NewStyle().Foreground(styles.Secondary)

	var filledChar, emptyChar string
	switch p.Style {
	case ProgressStyleLine:
		filledChar = "━"
		emptyChar = "─"
	case ProgressStyleDots:
		filledChar = "●"
		emptyChar = "○"
	default:
		filledChar = "█"
		emptyChar = "░"
	}

	return filledStyle.Render(strings.Repeat(filledChar, filled)) +
		emptyStyle.Render(strings.Repeat(emptyChar, empty))
}

func (p Progress) renderPercentage(percent float64) string {
	percentInt := int(percent * 100)
	percentStr := fmt.Sprintf("%d%%", percentInt)
	paddingNeeded := percentagePaddingWidth - len(percentStr)
	if paddingNeeded < 0 {
		paddingNeeded = 0
	}

	paddedPercent := strings.Repeat(" ", paddingNeeded) + percentStr
	return lipgloss.NewStyle().Bold(true).Foreground(styles.Primary).Render(paddedPercent)
}

func (p Progress) View() string {
	percent := p.Percent()
	filled := int(percent * float64(p.Width))
	if filled > p.Width {
		filled = p.Width
	}
	empty := p.Width - filled

	bar := p.renderBar(filled, empty)
	percentText := p.renderPercentage(percent)
	return bar + " " + percentText
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
	return r.Style.String() != ""
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

const statusBarSeparator = " │ "

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

func (s StatusBar) calculateSpacing() (leftPad, rightPad int) {
	leftText := strings.Join(s.LeftItems, statusBarSeparator)
	centerText := strings.Join(s.CenterItems, statusBarSeparator)
	rightText := strings.Join(s.RightItems, statusBarSeparator)

	leftLen := lipgloss.Width(leftText)
	centerLen := lipgloss.Width(centerText)
	rightLen := lipgloss.Width(rightText)

	totalContentLen := leftLen + centerLen + rightLen
	remainingSpace := s.Width - totalContentLen
	if remainingSpace < 0 {
		remainingSpace = 0
	}

	leftPad = remainingSpace / 2
	rightPad = remainingSpace - leftPad
	return
}

func (s StatusBar) View() string {
	style := lipgloss.NewStyle().
		Foreground(styles.TextDim).
		Width(s.Width)

	leftText := strings.Join(s.LeftItems, statusBarSeparator)
	centerText := strings.Join(s.CenterItems, statusBarSeparator)
	rightText := strings.Join(s.RightItems, statusBarSeparator)

	leftPad, rightPad := s.calculateSpacing()
	content := leftText +
		strings.Repeat(" ", leftPad) +
		centerText +
		strings.Repeat(" ", rightPad) +
		rightText

	return style.Render(content)
}

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

	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Warning)
	content.WriteString(titleStyle.Render(c.Title) + "\n\n")

	messageStyle := lipgloss.NewStyle().Foreground(styles.Text)
	content.WriteString(messageStyle.Render(c.Message) + "\n\n")

	confirmButton := styles.BadgeSuccessStyle.Render(" [Y] " + c.ConfirmText + " ")
	cancelButton := styles.BadgeErrorStyle.Render(" [N] " + c.CancelText + " ")
	content.WriteString(confirmButton + "  " + cancelButton)

	box := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(styles.Warning).
		Padding(1, 2).
		Width(c.Width)

	return box.Render(content.String())
}

const tableSelectionPadding = "  "
const tableSelectedPrefix = styles.CheckMark + " "

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

func (t Table) renderHeader() string {
	headerStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		Underline(true)

	var headerCells []string
	for _, col := range t.Columns {
		cell := headerStyle.Width(col.Width).Render(col.Header)
		headerCells = append(headerCells, cell)
	}
	return strings.Join(headerCells, "") + "\n"
}

func (t Table) renderRow(row TableRow, rowIndex int) string {
	isCursor := rowIndex == t.CursorIndex
	rowStyle := lipgloss.NewStyle().Foreground(styles.Text)
	if isCursor {
		rowStyle = rowStyle.Bold(true).Foreground(styles.Primary)
	}

	var rowCells []string
	for j, cell := range row.Cells {
		if j >= len(t.Columns) {
			break
		}

		cellContent := cell
		if j == 0 {
			if row.Selected {
				cellContent = tableSelectedPrefix + cellContent
			} else {
				cellContent = tableSelectionPadding + cellContent
			}
		}

		renderedCell := rowStyle.Width(t.Columns[j].Width).Render(cellContent)
		rowCells = append(rowCells, renderedCell)
	}
	return strings.Join(rowCells, "") + "\n"
}

func (t Table) View() string {
	var content strings.Builder

	content.WriteString(t.renderHeader())

	endIndex := t.ScrollTop + t.VisibleRows
	if endIndex > len(t.Rows) {
		endIndex = len(t.Rows)
	}

	for i := t.ScrollTop; i < endIndex; i++ {
		content.WriteString(t.renderRow(t.Rows[i], i))
	}

	return content.String()
}

const logoASCII = `
 ███████╗███████╗ ██████╗ 
 ██╔════╝██╔════╝██╔════╝ 
 █████╗  █████╗  ██║  ███╗
 ██╔══╝  ██╔══╝  ██║   ██║
 ███████╗███████╗╚██████╔╝
 ╚══════╝╚══════╝ ╚═════╝ `

const miniLogoText = "◆ EEG Pipeline"

func RenderLogo(style lipgloss.Style) string {
	return style.Render(logoASCII)
}

func RenderMiniLogo(style lipgloss.Style) string {
	return style.Render(miniLogoText)
}
