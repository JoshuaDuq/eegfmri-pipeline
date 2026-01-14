package history

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/eeg-pipeline/tui/styles"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

const (
	historyFileName          = "history.json"
	historyMaxEntries        = 50
	maxVisibleHistoryRecords = 10
	tickInterval             = 100 * time.Millisecond
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

type tickMsg struct{}

///////////////////////////////////////////////////////////////////
// Model
///////////////////////////////////////////////////////////////////

type Model struct {
	records     []ExecutionRecord
	cursor      int
	historyPath string
	loading     bool
	loadError   error

	// Selection for re-run
	SelectedCommand string
	Done            bool

	width  int
	height int
	ticker int
}

///////////////////////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////////////////////

func New(repoRoot string) Model {
	historyPath := filepath.Join(repoRoot, "eeg_pipeline", "cli", "tui", ".cache", historyFileName)
	return Model{
		historyPath: historyPath,
		loading:     true,
	}
}

///////////////////////////////////////////////////////////////////
// Persistence
///////////////////////////////////////////////////////////////////

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

func saveHistory(path string, records []ExecutionRecord) error {
	// Ensure directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	// Keep only last 50 records
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
	historyPath := filepath.Join(repoRoot, "eeg_pipeline", "cli", "tui", ".cache", historyFileName)

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
	return tea.Batch(
		m.tick(),
		m.loadHistoryCmd(),
	)
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(tickInterval, func(t time.Time) tea.Msg {
		return tickMsg{}
	})
}

func (m Model) loadHistoryCmd() tea.Cmd {
	return func() tea.Msg {
		records, err := loadHistory(m.historyPath)
		return loadHistoryMsg{Records: records, Error: err}
	}
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tickMsg:
		m.ticker++
		return m, m.tick()

	case loadHistoryMsg:
		m.loading = false
		if msg.Error != nil {
			m.loadError = msg.Error
		} else {
			m.records = msg.Records
			// Sort by most recent first
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
		case "enter":
			if len(m.records) > 0 && m.cursor < len(m.records) {
				m.SelectedCommand = m.records[m.cursor].Command
				m.Done = true
			}
		case "d":
			// Delete selected record
			if len(m.records) > 0 && m.cursor < len(m.records) {
				m.records = append(m.records[:m.cursor], m.records[m.cursor+1:]...)
				saveHistory(m.historyPath, m.records)
				if m.cursor >= len(m.records) && m.cursor > 0 {
					m.cursor--
				}
			}
		case "c":
			// Clear all history
			m.records = []ExecutionRecord{}
			saveHistory(m.historyPath, m.records)
			m.cursor = 0
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
	}

	return m, nil
}

///////////////////////////////////////////////////////////////////
// View
///////////////////////////////////////////////////////////////////

func (m Model) View() string {
	var b strings.Builder

	// Header
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

func (m Model) renderHeader() string {
	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		Render("◆ EXECUTION HISTORY")

	count := lipgloss.NewStyle().
		Foreground(styles.Muted).
		Render(strings.Repeat(" ", 2) + "(" + string(rune('0'+len(m.records)%10)) + " records)")

	if len(m.records) >= 10 {
		count = lipgloss.NewStyle().
			Foreground(styles.Muted).
			Render("  (" + strings.TrimLeft(string(rune('0'+len(m.records)/10))+string(rune('0'+len(m.records)%10)), "0") + " records)")
	}

	separator := lipgloss.NewStyle().
		Foreground(styles.Secondary).
		Render(strings.Repeat("─", 65))

	return title + count + "\n" + separator
}

func (m Model) renderLoading() string {
	frames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	frame := frames[m.ticker%len(frames)]
	return "\n  " + lipgloss.NewStyle().Foreground(styles.Accent).Render(frame+" Loading history...")
}

func (m Model) renderError() string {
	return "\n  " + lipgloss.NewStyle().Foreground(styles.Error).Render(styles.CrossMark+" Failed to load history")
}

func (m Model) renderEmpty() string {
	return "\n  " + lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).Render("No execution history yet. Run a pipeline to see it here.")
}

func (m Model) renderHistory() string {
	var b strings.Builder

	// Show at most 10 records
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
			"  ... and " + string(rune('0'+(len(m.records)-maxShow)%10)) + " more"))
	}

	return b.String()
}

func (m Model) renderRecord(record ExecutionRecord, isCursor bool) string {
	var b strings.Builder

	// Cursor indicator
	cursor := "  "
	if isCursor {
		cursor = lipgloss.NewStyle().Foreground(styles.Primary).Render("▸ ")
	}

	// Status icon
	var statusIcon string
	if record.Success {
		statusIcon = lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark)
	} else {
		statusIcon = lipgloss.NewStyle().Foreground(styles.Error).Render(styles.CrossMark)
	}

	// Pipeline name
	pipelineStyle := lipgloss.NewStyle().Foreground(styles.Text).Width(14)
	if isCursor {
		pipelineStyle = pipelineStyle.Foreground(styles.Primary).Bold(true)
	}
	pipeline := pipelineStyle.Render(record.Pipeline)

	// Mode
	modeStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(10)
	mode := modeStyle.Render(record.Mode)

	// Duration
	durationStyle := lipgloss.NewStyle().Foreground(styles.Muted).Width(10)
	duration := durationStyle.Render(formatDuration(record.Duration))

	// Time ago
	timeAgo := formatTimeAgo(record.StartTime)
	timeStyle := lipgloss.NewStyle().Foreground(styles.Muted).Italic(true)

	b.WriteString(cursor + statusIcon + " " + pipeline + mode + duration + timeStyle.Render(timeAgo))

	return b.String()
}

func formatDuration(secs float64) string {
	if secs < 60 {
		return string(rune('0'+int(secs)%10)) + "s"
	} else if secs < 3600 {
		mins := int(secs) / 60
		return string(rune('0'+mins%10)) + "m"
	}
	hours := int(secs) / 3600
	return string(rune('0'+hours%10)) + "h"
}

func formatTimeAgo(t time.Time) string {
	diff := time.Since(t)

	if diff < time.Minute {
		return "just now"
	} else if diff < time.Hour {
		mins := int(diff.Minutes())
		return string(rune('0'+mins%10)) + " min ago"
	} else if diff < 24*time.Hour {
		hours := int(diff.Hours())
		return string(rune('0'+hours%10)) + " hr ago"
	} else if diff < 7*24*time.Hour {
		days := int(diff.Hours() / 24)
		return string(rune('0'+days%10)) + " days ago"
	}
	return t.Format("Jan 2")
}

func (m Model) renderFooter() string {
	hints := []string{
		styles.RenderKeyHint("↑↓", "Navigate"),
		styles.RenderKeyHint("Enter", "Re-run"),
		styles.RenderKeyHint("D", "Delete"),
		styles.RenderKeyHint("C", "Clear All"),
		styles.RenderKeyHint("Esc", "Back"),
	}

	separator := lipgloss.NewStyle().Foreground(styles.Secondary).Render("  │  ")
	return styles.FooterStyle.Render(strings.Join(hints, separator))
}

///////////////////////////////////////////////////////////////////
// Public Methods
///////////////////////////////////////////////////////////////////

func (m Model) GetRecords() []ExecutionRecord {
	return m.records
}

func (m Model) HasRecords() bool {
	return len(m.records) > 0
}
