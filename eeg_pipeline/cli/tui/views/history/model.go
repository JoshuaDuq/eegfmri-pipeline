package history

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/eeg-pipeline/tui/animation"
	"github.com/eeg-pipeline/tui/components"
	"github.com/eeg-pipeline/tui/styles"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

const (
	historyFileName          = "history.json"
	historyMaxEntries        = 50
	maxVisibleHistoryRecords = 10
)

///////////////////////////////////////////////////////////////////
// Types
///////////////////////////////////////////////////////////////////

type ExecutionRecord struct {
	ID           string    `json:"id"`
	Command      string    `json:"command"`
	Pipeline     string    `json:"pipeline"`
	Mode         string    `json:"mode"`
	StartTime    time.Time `json:"start_time"`
	EndTime      time.Time `json:"end_time"`
	Duration     float64   `json:"duration_secs"`
	ExitCode     int       `json:"exit_code"`
	Success      bool      `json:"success"`
	SubjectCount int       `json:"subject_count"`
	ErrorCount   int       `json:"error_count"`
}

type historyData struct {
	Executions []ExecutionRecord `json:"executions"`
	MaxEntries int               `json:"max_entries"`
}

///////////////////////////////////////////////////////////////////
// Model
///////////////////////////////////////////////////////////////////

type Model struct {
	records     []ExecutionRecord
	cursor      int
	historyPath string
	loading     bool
	loadError   error
	animQueue   animation.Queue
	spinner     components.Spinner
	width       int
}

///////////////////////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////////////////////

func New(repoRoot string) Model {
	historyPath := buildHistoryPath(repoRoot)
	m := Model{
		historyPath: historyPath,
		loading:     true,
		spinner:     components.NewSpinner("Loading history..."),
	}
	m.animQueue.Push(animation.CursorBlinkLoop())
	return m
}

///////////////////////////////////////////////////////////////////
// Persistence
///////////////////////////////////////////////////////////////////

func buildHistoryPath(repoRoot string) string {
	return filepath.Join(repoRoot, "eeg_pipeline", "cli", "tui", ".cache", historyFileName)
}

func loadHistory(path string) ([]ExecutionRecord, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return []ExecutionRecord{}, nil
		}
		return nil, err
	}

	var history historyData
	if err := json.Unmarshal(data, &history); err != nil {
		return nil, err
	}

	return history.Executions, nil
}

func LoadRecentRecords(repoRoot string, limit int) ([]ExecutionRecord, error) {
	records, err := loadHistory(buildHistoryPath(repoRoot))
	if err != nil {
		return nil, err
	}

	sort.Slice(records, func(i, j int) bool {
		return records[i].StartTime.After(records[j].StartTime)
	})

	if limit > 0 && len(records) > limit {
		records = records[:limit]
	}

	return records, nil
}

func saveHistory(path string, records []ExecutionRecord) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	if len(records) > historyMaxEntries {
		records = records[len(records)-historyMaxEntries:]
	}

	history := historyData{
		Executions: records,
		MaxEntries: historyMaxEntries,
	}

	data, err := json.MarshalIndent(history, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// AddRecord adds a new execution record to history
func AddRecord(repoRoot string, record ExecutionRecord) error {
	historyPath := buildHistoryPath(repoRoot)

	records, _ := loadHistory(historyPath)
	records = append(records, record)

	return saveHistory(historyPath, records)
}

///////////////////////////////////////////////////////////////////
// Tea Model Implementation
///////////////////////////////////////////////////////////////////

type loadHistoryMsg struct {
	Records []ExecutionRecord
	Error   error
}

func (m Model) Init() tea.Cmd {
	return tea.Batch(m.loadHistoryCmd(), m.tick(), m.immediateTick())
}

func (m Model) immediateTick() tea.Cmd {
	return tea.Tick(0, func(t time.Time) tea.Msg { return tickMsg{} })
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(time.Millisecond*styles.TickIntervalMs, func(t time.Time) tea.Msg {
		return tickMsg{}
	})
}

type tickMsg struct{}

func (m Model) loadHistoryCmd() tea.Cmd {
	return func() tea.Msg {
		records, err := loadHistory(m.historyPath)
		return loadHistoryMsg{Records: records, Error: err}
	}
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tickMsg:
		m.animQueue.Tick()
		m.spinner.Tick()
		return m, m.tick()
	case loadHistoryMsg:
		m.loading = false
		if msg.Error != nil {
			m.loadError = msg.Error
		} else {
			m.records = msg.Records
			sort.Slice(m.records, func(i, j int) bool {
				return m.records[i].StartTime.After(m.records[j].StartTime)
			})
		}
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			}
		case "down", "j":
			if m.cursor < len(m.records)-1 {
				m.cursor++
			}
		case "d":
			if len(m.records) > 0 && m.cursor < len(m.records) {
				m.records = append(m.records[:m.cursor], m.records[m.cursor+1:]...)
				saveHistory(m.historyPath, m.records)
				if m.cursor >= len(m.records) && m.cursor > 0 {
					m.cursor--
				}
			}
		case "c":
			m.records = []ExecutionRecord{}
			saveHistory(m.historyPath, m.records)
			m.cursor = 0
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
	}

	return m, nil
}

///////////////////////////////////////////////////////////////////
// View
///////////////////////////////////////////////////////////////////

func (m Model) View() string {
	var b strings.Builder

	b.WriteString(m.renderHeader())
	b.WriteString("\n\n")

	if m.loading {
		b.WriteString(m.renderLoading())
	} else if m.loadError != nil {
		b.WriteString(m.renderError())
	} else if len(m.records) == 0 {
		b.WriteString(m.renderEmpty())
	} else {
		b.WriteString(m.renderHistory())
	}

	b.WriteString("\n")
	b.WriteString(m.renderFooter())

	return styles.BoxStyle.Render(b.String())
}

func (m Model) innerWidth() int {
	w := m.width - 8
	if w < 50 {
		return 50
	}
	return w
}

func (m Model) renderHeader() string {
	title := styles.RenderSectionLabel("Execution History")
	count := lipgloss.NewStyle().Foreground(styles.Muted).Render(fmt.Sprintf("  %d records", len(m.records)))
	return title + count + "\n" + styles.RenderDivider(m.innerWidth())
}

func (m Model) renderLoading() string {
	return "\n  " + m.spinner.View() + "\n"
}

func (m Model) renderError() string {
	return "\n  " + lipgloss.NewStyle().Foreground(styles.Error).Render(styles.CrossMark+" Failed to load history")
}

func (m Model) renderEmpty() string {
	return "\n  " + lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).Render("No execution history yet. Run a pipeline to see it here.") + "\n"
}

func (m Model) renderHistory() string {
	var b strings.Builder

	maxShow := maxVisibleHistoryRecords
	if maxShow > len(m.records) {
		maxShow = len(m.records)
	}

	for i := 0; i < maxShow; i++ {
		record := m.records[i]
		isCursor := i == m.cursor

		b.WriteString(m.renderRecord(record, isCursor))
		b.WriteString("\n")
	}

	if len(m.records) > maxShow {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Render(
			fmt.Sprintf("  ... and %d more", len(m.records)-maxShow)))
	}

	return b.String()
}

func (m Model) renderRecord(record ExecutionRecord, isCursor bool) string {
	cursor := "  "
	if isCursor {
		cursor = styles.RenderCursorOptional(m.animQueue.CursorVisible())
	}

	var statusIcon string
	if record.Success {
		statusIcon = lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark)
	} else {
		statusIcon = lipgloss.NewStyle().Foreground(styles.Error).Render(styles.CrossMark)
	}

	pipelineStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(14)
	if isCursor {
		pipelineStyle = pipelineStyle.Foreground(styles.Primary).Bold(true)
	}

	modeStyle := lipgloss.NewStyle().Foreground(styles.Muted).Width(10)
	durationStyle := lipgloss.NewStyle().Foreground(styles.Muted).Width(10)
	timeStyle := lipgloss.NewStyle().Foreground(styles.Muted)

	return cursor + statusIcon + " " +
		pipelineStyle.Render(record.Pipeline) +
		modeStyle.Render(record.Mode) +
		durationStyle.Render(FormatDurationSeconds(record.Duration)) +
		timeStyle.Render(FormatTimeAgo(record.StartTime))
}

func FormatDurationSeconds(secs float64) string {
	if secs < 60 {
		return fmt.Sprintf("%.0fs", secs)
	} else if secs < 3600 {
		mins := int(secs) / 60
		return fmt.Sprintf("%dm", mins)
	}
	hours := int(secs) / 3600
	return fmt.Sprintf("%dh", hours)
}

func FormatTimeAgo(t time.Time) string {
	diff := time.Since(t)

	if diff < time.Minute {
		return "just now"
	} else if diff < time.Hour {
		mins := int(diff.Minutes())
		return fmt.Sprintf("%d min ago", mins)
	} else if diff < 24*time.Hour {
		hours := int(diff.Hours())
		return fmt.Sprintf("%d hr ago", hours)
	} else if diff < 7*24*time.Hour {
		days := int(diff.Hours() / 24)
		return fmt.Sprintf("%d days ago", days)
	}
	return t.Format("Jan 2")
}

func (m Model) renderFooter() string {
	hints := []string{
		styles.RenderKeyHint("↑↓", "Navigate"),
		styles.RenderKeyHint("D", "Delete"),
		styles.RenderKeyHint("C", "Clear All"),
		styles.RenderKeyHint("Esc", "Back"),
	}

	w := m.innerWidth()
	divider := styles.RenderDivider(w)
	bar := styles.FooterStyle.Width(w).Render(strings.Join(hints, styles.RenderFooterSeparator()))
	return divider + "\n" + bar
}
