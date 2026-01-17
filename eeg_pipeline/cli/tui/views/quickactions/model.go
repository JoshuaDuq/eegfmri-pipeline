package quickactions

import (
	"strings"
	"time"

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
	headerFrames = []string{"◆", "◇", "◆", "◈"}
	cursorFrames = []string{"▸", "▹", "▸", "▹"}
	quickActions = []Action{
		{Type: ActionStats, Name: "Project Stats", Description: "View subject & feature analytics", Icon: "◆", Shortcut: "S"},
		{Type: ActionHistory, Name: "History", Description: "View recent pipeline executions", Icon: "◇", Shortcut: "H"},
		{Type: ActionValidate, Name: "Validate", Description: "Check data integrity", Icon: "◈", Shortcut: "V"},
		{Type: ActionExport, Name: "Export", Description: "Export features to CSV", Icon: "◇", Shortcut: "X"},
		{Type: ActionConfig, Name: "Config", Description: "View configuration", Icon: "◆", Shortcut: "C"},
		{Type: ActionRefresh, Name: "Refresh", Description: "Reload subject data", Icon: "◈", Shortcut: "R"},
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
}

func New() Model {
	return Model{
		cursor:  0,
		Visible: false,
	}
}

func (m Model) Init() tea.Cmd {
	return m.tick()
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(time.Millisecond*100, func(t time.Time) tea.Msg {
		return tickMsg{}
	})
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tickMsg:
		m.ticker++
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

	headerIcon := lipgloss.NewStyle().
		Foreground(styles.Accent).
		Bold(true).
		Render(headerFrames[(m.ticker/2)%len(headerFrames)])

	header := headerIcon + " " + lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		Render("QUICK ACTIONS")
	b.WriteString(header + "\n")

	divWidth := 35
	div1 := lipgloss.NewStyle().Foreground(styles.Accent).Render(strings.Repeat("─", divWidth/3))
	div2 := lipgloss.NewStyle().Foreground(styles.Primary).Render(strings.Repeat("─", divWidth/3))
	div3 := lipgloss.NewStyle().Foreground(styles.Secondary).Render(strings.Repeat("─", divWidth/3+divWidth%3))
	b.WriteString(div1 + div2 + div3 + "\n\n")

	for i, action := range quickActions {
		b.WriteString(m.renderAction(action, i == m.cursor) + "\n")
	}

	b.WriteString("\n")

	footer := lipgloss.NewStyle().Foreground(styles.Muted).Render("Use shortcuts or ") +
		lipgloss.NewStyle().Foreground(styles.Primary).Render("Enter") +
		lipgloss.NewStyle().Foreground(styles.Muted).Render(" to select")
	b.WriteString(footer)

	box := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(styles.Primary).
		Padding(1, 2).
		Width(42)

	return box.Render(b.String())
}

func (m Model) renderAction(action Action, isCursor bool) string {
	cursor := "  "
	if isCursor {
		cursor = lipgloss.NewStyle().
			Foreground(styles.Primary).
			Bold(true).
			Render(cursorFrames[(m.ticker/2)%len(cursorFrames)] + " ")
	}

	shortcutStyle := lipgloss.NewStyle()
	if isCursor {
		shortcutStyle = shortcutStyle.
			Foreground(lipgloss.Color("#000000")).
			Background(styles.Accent).
			Bold(true).
			Padding(0, 1)
	} else {
		shortcutStyle = shortcutStyle.
			Foreground(styles.Accent).
			Bold(true)
	}
	shortcut := shortcutStyle.Render(action.Shortcut)

	nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
	iconStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	if isCursor {
		nameStyle = nameStyle.Foreground(styles.Primary).Bold(true)
		iconStyle = iconStyle.Foreground(styles.Accent)
	}
	name := iconStyle.Render(action.Icon) + " " + nameStyle.Render(action.Name)

	desc := ""
	if isCursor {
		descStyle := lipgloss.NewStyle().
			Foreground(styles.TextDim).
			Italic(true).
			PaddingLeft(6)
		desc = "\n" + descStyle.Render("→ "+action.Description)
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
