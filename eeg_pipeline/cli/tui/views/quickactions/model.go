package quickactions

import (
	"strings"
	"time"

	"github.com/eeg-pipeline/tui/animation"
	"github.com/eeg-pipeline/tui/styles"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type ActionType int

const (
	ActionStats ActionType = iota
	ActionHistory
	ActionValidate
	ActionExport
	ActionConfig
	ActionRefresh
)

type Action struct {
	Type        ActionType
	Name        string
	Description string
	Icon        string
	Shortcut    string
}

var (
	quickActions = []Action{
		{Type: ActionStats, Name: "Project Stats", Description: "View subject & feature analytics", Icon: styles.SelectedMark, Shortcut: "S"},
		{Type: ActionHistory, Name: "History", Description: "View recent pipeline executions", Icon: styles.SelectedMark, Shortcut: "H"},
		{Type: ActionValidate, Name: "Validate", Description: "Check data integrity", Icon: styles.SelectedMark, Shortcut: "V"},
		{Type: ActionExport, Name: "Export", Description: "Export features to CSV", Icon: styles.SelectedMark, Shortcut: "X"},
		{Type: ActionConfig, Name: "Config", Description: "View configuration", Icon: styles.SelectedMark, Shortcut: "C"},
		{Type: ActionRefresh, Name: "Refresh", Description: "Reload subject data", Icon: styles.SelectedMark, Shortcut: "R"},
	}
	shortcutMap = map[string]ActionType{
		"s": ActionStats,
		"h": ActionHistory,
		"v": ActionValidate,
		"x": ActionExport,
		"c": ActionConfig,
		"r": ActionRefresh,
	}
)

type tickMsg struct{}

type Model struct {
	cursor         int
	Visible        bool
	SelectedAction ActionType
	Done           bool
	ticker         int
	animQueue      animation.Queue
}

func New() Model {
	m := Model{
		cursor:  0,
		Visible: false,
	}
	m.animQueue.Push(animation.CursorBlinkLoop())
	return m
}

func (m Model) Init() tea.Cmd {
	return m.tick()
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(time.Millisecond*styles.TickIntervalMs, func(t time.Time) tea.Msg {
		return tickMsg{}
	})
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tickMsg:
		m.ticker++
		m.animQueue.Tick()
		return m, m.tick()

	case tea.KeyMsg:
		switch key := msg.String(); key {
		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			} else {
				m.cursor = len(quickActions) - 1
			}
		case "down", "j":
			if m.cursor < len(quickActions)-1 {
				m.cursor++
			} else {
				m.cursor = 0
			}
		case "enter", " ":
			m.SelectedAction = quickActions[m.cursor].Type
			m.Done = true
		case "esc", "q":
			m.Visible = false
		default:
			if action, ok := shortcutMap[key]; ok {
				m.SelectedAction = action
				m.Done = true
			}
		}
	}

	return m, nil
}

func (m Model) View() string {
	if !m.Visible {
		return ""
	}

	var b strings.Builder

	header := styles.RenderSectionLabel("Quick Actions")
	b.WriteString(header + "\n")
	b.WriteString(styles.RenderHeaderSeparator(35) + "\n\n")

	for i, action := range quickActions {
		b.WriteString(m.renderAction(action, i == m.cursor) + "\n")
	}

	b.WriteString("\n")

	footer := lipgloss.NewStyle().Foreground(styles.Muted).Render("shortcuts or ") +
		styles.RenderKeyHint("\u23ce", "select")
	b.WriteString(footer)

	return styles.CardStyle.Width(42).Render(b.String())
}

func (m Model) renderAction(action Action, isCursor bool) string {
	cursor := "  "
	if isCursor {
		cursor = styles.RenderCursorOptional(m.animQueue.CursorVisible())
	}

	shortcutStyle := lipgloss.NewStyle().
		Foreground(styles.Text).
		Background(styles.Border).
		Bold(true).
		Padding(0, 1)
	if isCursor {
		shortcutStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#0F172A")).
			Background(styles.Primary).
			Bold(true).
			Padding(0, 1)
	}
	shortcut := shortcutStyle.Render(action.Shortcut)

	nameStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	if isCursor {
		nameStyle = nameStyle.Foreground(styles.Text).Bold(true)
	}
	name := nameStyle.Render(action.Name)

	desc := ""
	if isCursor {
		descStyle := lipgloss.NewStyle().
			Foreground(styles.Muted).
			PaddingLeft(6)
		desc = "\n" + descStyle.Render(action.Description)
	}

	return cursor + shortcut + " " + name + desc
}

func (m *Model) Show() {
	m.Visible = true
	m.Done = false
	m.cursor = 0
}

func (m *Model) Hide() {
	m.Visible = false
}

func (m *Model) Reset() {
	m.Done = false
	m.SelectedAction = ActionStats
}
