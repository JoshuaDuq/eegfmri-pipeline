package history

import (
	"path/filepath"
	"testing"
	"time"
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
