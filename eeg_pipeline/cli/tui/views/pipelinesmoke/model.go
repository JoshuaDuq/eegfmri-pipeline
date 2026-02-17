package pipelinesmoke

import (
	"fmt"
	"strings"
	"time"

	"github.com/eeg-pipeline/tui/animation"
	"github.com/eeg-pipeline/tui/styles"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type tickMsg struct{}

type smokeItem struct {
	ID          string
	Name        string
	Description string
}

var smokeItems = []smokeItem{
	{ID: "preprocessing", Name: "EEG Preprocessing", Description: "CLI parser/help smoke"},
	{ID: "features", Name: "Features", Description: "CLI parser/help smoke"},
	{ID: "behavior", Name: "Behavior", Description: "CLI parser/help smoke"},
	{ID: "machine_learning", Name: "Machine Learning", Description: "CLI parser/help smoke"},
	{ID: "plotting", Name: "Plotting", Description: "CLI parser/help smoke"},
	{ID: "fmri_preprocessing", Name: "fMRI Preprocessing", Description: "CLI parser/help smoke"},
	{ID: "fmri_analysis", Name: "fMRI Analysis", Description: "CLI parser/help smoke"},
	{ID: "utilities", Name: "Utilities", Description: "CLI parser/help smoke"},
	{ID: "validate", Name: "Validate", Description: "CLI parser/help smoke"},
	{ID: "info", Name: "Info", Description: "CLI parser/help smoke"},
	{ID: "stats", Name: "Stats", Description: "CLI parser/help smoke"},
	{ID: "runtime_version", Name: "Runtime Version", Description: "End-to-end dispatch smoke"},
}

type Model struct {
	task       string
	cursor     int
	selected   map[string]bool
	width      int
	height     int
	statusLine string
	Done       bool
	Cancelled  bool
	RunCommand string
	animQueue  animation.Queue
}

func New(task string) Model {
	selected := make(map[string]bool, len(smokeItems))
	for _, item := range smokeItems {
		selected[item.ID] = true
	}

	m := Model{
		task:     strings.TrimSpace(task),
		selected: selected,
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
		m.animQueue.Tick()
		return m, m.tick()
	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			} else {
				m.cursor = len(smokeItems) - 1
			}
		case "down", "j":
			if m.cursor < len(smokeItems)-1 {
				m.cursor++
			} else {
				m.cursor = 0
			}
		case " ":
			m.toggleCursor()
		case "a", "A":
			m.ToggleAll(!m.allSelected())
		case "enter":
			if len(m.selectedIDs()) == 0 {
				m.statusLine = "Select at least one smoke check"
				return m, nil
			}
			m.Done = true
			m.Cancelled = false
			m.RunCommand = m.BuildCommand()
			return m, nil
		case "esc":
			m.Done = true
			m.Cancelled = true
			m.RunCommand = ""
			return m, nil
		}
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
	}

	return m, nil
}

func (m *Model) toggleCursor() {
	if m.cursor < 0 || m.cursor >= len(smokeItems) {
		return
	}
	id := smokeItems[m.cursor].ID
	m.ToggleByID(id)
}

func (m Model) allSelected() bool {
	for _, item := range smokeItems {
		if !m.selected[item.ID] {
			return false
		}
	}
	return true
}

func (m *Model) ToggleAll(enabled bool) {
	for _, item := range smokeItems {
		m.selected[item.ID] = enabled
	}
}

func (m *Model) ToggleByID(id string) {
	_, ok := m.selected[id]
	if !ok {
		return
	}
	m.selected[id] = !m.selected[id]
}

func (m Model) selectedIDs() []string {
	ids := make([]string, 0, len(smokeItems))
	for _, item := range smokeItems {
		if m.selected[item.ID] {
			ids = append(ids, item.ID)
		}
	}
	return ids
}

func (m *Model) SetTask(task string) {
	m.task = strings.TrimSpace(task)
}

func (m Model) BuildCommand() string {
	cmd := "scripts/tui_pipeline_smoke.py"
	if m.task != "" {
		cmd += " --task " + m.task
	}
	ids := m.selectedIDs()
	if len(ids) > 0 && len(ids) < len(smokeItems) {
		cmd += " --pipelines " + strings.Join(ids, ",")
	}
	return cmd
}

func (m *Model) Reset() {
	m.Done = false
	m.Cancelled = false
	m.RunCommand = ""
	m.statusLine = ""
}

func (m Model) View() string {
	var b strings.Builder

	b.WriteString(styles.RenderSectionLabel("Pipeline Smoke Test") + "\n")
	b.WriteString(styles.RenderDivider(70) + "\n\n")
	task := m.task
	if task == "" {
		task = "(default)"
	}
	b.WriteString("Task: " + lipgloss.NewStyle().Foreground(styles.Accent).Render(task) + "\n")
	b.WriteString(fmt.Sprintf("Selected: %d/%d\n\n", len(m.selectedIDs()), len(smokeItems)))

	for i, item := range smokeItems {
		focused := i == m.cursor
		checked := m.selected[item.ID]
		cursor := "  "
		if focused {
			cursor = styles.RenderCursorOptional(m.animQueue.CursorVisible())
		}

		nameStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		descStyle := lipgloss.NewStyle().Foreground(styles.Muted)
		if focused {
			nameStyle = nameStyle.Foreground(styles.Text).Bold(true)
			descStyle = descStyle.Foreground(styles.TextDim)
		}

		line := fmt.Sprintf(
			"%s%s %s  %s",
			cursor,
			styles.RenderCheckbox(checked, focused),
			nameStyle.Render(item.Name),
			descStyle.Render(item.Description),
		)
		b.WriteString(line + "\n")
	}

	if m.statusLine != "" {
		b.WriteString("\n")
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Render(m.statusLine))
	}

	b.WriteString("\n")
	b.WriteString(styles.RenderDivider(70) + "\n")
	hints := []string{
		styles.RenderKeyHint("↑↓", "Navigate"),
		styles.RenderKeyHint("Space", "Toggle"),
		styles.RenderKeyHint("A", "Toggle all"),
		styles.RenderKeyHint("Enter", "Run selected"),
		styles.RenderKeyHint("Esc", "Back"),
	}
	b.WriteString(styles.FooterStyle.Render(strings.Join(hints, styles.RenderFooterSeparator())))

	cardWidth := m.width - 10
	if cardWidth < 72 {
		cardWidth = 72
	}
	return styles.CardStyle.Width(cardWidth).Render(b.String())
}
