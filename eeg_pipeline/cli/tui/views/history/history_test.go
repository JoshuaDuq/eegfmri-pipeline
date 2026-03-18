package history

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

func TestLoadHistory_MissingFileReturnsEmpty(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "missing.json")

	records, err := loadHistory(path)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(records) != 0 {
		t.Fatalf("expected empty records, got %d", len(records))
	}
}

func TestSaveHistory_TrimsToMaxEntries(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "history.json")

	records := make([]ExecutionRecord, historyMaxEntries+2)
	for i := range records {
		records[i] = ExecutionRecord{
			ID:        "id",
			Pipeline:  "pipe",
			StartTime: time.Now().Add(time.Duration(i) * time.Minute),
		}
	}

	if err := saveHistory(path, records); err != nil {
		t.Fatalf("saveHistory error: %v", err)
	}

	loaded, err := loadHistory(path)
	if err != nil {
		t.Fatalf("loadHistory error: %v", err)
	}
	if len(loaded) != historyMaxEntries {
		t.Fatalf("expected %d records, got %d", historyMaxEntries, len(loaded))
	}
}

func TestAddRecord_WritesToHistoryPath(t *testing.T) {
	tmpDir := t.TempDir()
	record := ExecutionRecord{
		ID:        "one",
		Pipeline:  "preprocessing",
		StartTime: time.Now(),
	}

	if err := AddRecord(tmpDir, record); err != nil {
		t.Fatalf("AddRecord error: %v", err)
	}

	loaded, err := loadHistory(buildHistoryPath(tmpDir))
	if err != nil {
		t.Fatalf("loadHistory error: %v", err)
	}
	if len(loaded) != 1 {
		t.Fatalf("expected 1 record, got %d", len(loaded))
	}
	if loaded[0].ID != "one" {
		t.Fatalf("expected record id one, got %q", loaded[0].ID)
	}
}

func TestLoadRecentRecords_SortsAndLimits(t *testing.T) {
	tmpDir := t.TempDir()
	path := buildHistoryPath(tmpDir)

	oldest := time.Now().Add(-2 * time.Hour)
	newest := time.Now().Add(-10 * time.Minute)
	records := []ExecutionRecord{
		{ID: "a", StartTime: oldest},
		{ID: "b", StartTime: newest},
	}

	if err := saveHistory(path, records); err != nil {
		t.Fatalf("saveHistory error: %v", err)
	}

	limited, err := LoadRecentRecords(tmpDir, 1)
	if err != nil {
		t.Fatalf("LoadRecentRecords error: %v", err)
	}
	if len(limited) != 1 {
		t.Fatalf("expected 1 record, got %d", len(limited))
	}
	if limited[0].ID != "b" {
		t.Fatalf("expected newest record, got %q", limited[0].ID)
	}
}

func TestFormatDurationSeconds(t *testing.T) {
	if got := FormatDurationSeconds(42); got != "42s" {
		t.Fatalf("expected 42s, got %q", got)
	}
	if got := FormatDurationSeconds(120); got != "2m" {
		t.Fatalf("expected 2m, got %q", got)
	}
	if got := FormatDurationSeconds(7200); got != "2h" {
		t.Fatalf("expected 2h, got %q", got)
	}
}

func TestFormatTimeAgo(t *testing.T) {
	now := time.Now()
	cases := []struct {
		t    time.Time
		want string
	}{
		{now.Add(-30 * time.Second), "just now"},
		{now.Add(-5 * time.Minute), "5 min ago"},
		{now.Add(-2 * time.Hour), "2 hr ago"},
		{now.Add(-3 * 24 * time.Hour), "3 days ago"},
	}
	for _, c := range cases {
		if got := FormatTimeAgo(c.t); got != c.want {
			t.Fatalf("expected %q, got %q", c.want, got)
		}
	}

	older := now.Add(-10 * 24 * time.Hour)
	if got := FormatTimeAgo(older); got != older.Format("Jan 2") {
		t.Fatalf("expected date %q, got %q", older.Format("Jan 2"), got)
	}
}

func TestUpdate_SortsOnLoadMessage(t *testing.T) {
	tmpDir := t.TempDir()
	m := New(tmpDir)
	records := []ExecutionRecord{
		{ID: "old", StartTime: time.Now().Add(-2 * time.Hour)},
		{ID: "new", StartTime: time.Now().Add(-10 * time.Minute)},
	}

	model, _ := m.Update(loadHistoryMsg{Records: records})
	updated := model.(Model)
	if len(updated.records) != 2 {
		t.Fatalf("expected 2 records, got %d", len(updated.records))
	}
	if updated.records[0].ID != "new" {
		t.Fatalf("expected records sorted desc, got %q", updated.records[0].ID)
	}
}

func TestUpdate_DeleteAndClear(t *testing.T) {
	tmpDir := t.TempDir()
	historyPath := buildHistoryPath(tmpDir)
	records := []ExecutionRecord{
		{ID: "a", StartTime: time.Now().Add(-2 * time.Hour)},
		{ID: "b", StartTime: time.Now().Add(-1 * time.Hour)},
	}
	if err := saveHistory(historyPath, records); err != nil {
		t.Fatalf("saveHistory error: %v", err)
	}

	m := New(tmpDir)
	m.records = append([]ExecutionRecord{}, records...)
	m.cursor = 1

	model, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("d")})
	updated := model.(Model)
	if len(updated.records) != 1 {
		t.Fatalf("expected 1 record after delete, got %d", len(updated.records))
	}
	loaded, err := loadHistory(historyPath)
	if err != nil {
		t.Fatalf("loadHistory error: %v", err)
	}
	if len(loaded) != 1 {
		t.Fatalf("expected 1 saved record, got %d", len(loaded))
	}

	model, _ = updated.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("c")})
	cleared := model.(Model)
	if len(cleared.records) != 0 {
		t.Fatalf("expected 0 records after clear, got %d", len(cleared.records))
	}
}

func TestView_Branches(t *testing.T) {
	tmpDir := t.TempDir()
	m := New(tmpDir)
	m.loading = true
	if view := m.View(); !strings.Contains(view, "Loading history") {
		t.Fatalf("expected loading view, got %q", view)
	}

	m.loading = false
	m.loadError = os.ErrNotExist
	if view := m.View(); !strings.Contains(view, "Failed to load history") {
		t.Fatalf("expected error view, got %q", view)
	}

	m.loadError = nil
	m.records = nil
	if view := m.View(); !strings.Contains(view, "No execution history yet") {
		t.Fatalf("expected empty view, got %q", view)
	}

	m.records = make([]ExecutionRecord, maxVisibleHistoryRecords+2)
	for i := range m.records {
		m.records[i] = ExecutionRecord{
			ID:        "id",
			Pipeline:  "preprocessing",
			Mode:      "cli",
			StartTime: time.Now().Add(-10 * 24 * time.Hour),
			Success:   true,
		}
	}
	sort.Slice(m.records, func(i, j int) bool {
		return m.records[i].StartTime.After(m.records[j].StartTime)
	})
	view := m.View()
	if !strings.Contains(view, "preprocessing") {
		t.Fatalf("expected history view, got %q", view)
	}
	if !strings.Contains(view, "and 2 more") {
		t.Fatalf("expected overflow indicator, got %q", view)
	}
}
