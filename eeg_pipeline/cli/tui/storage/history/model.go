package history

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"time"
)

const (
	historyFileName   = "history.json"
	historyMaxEntries = 50
)

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

func AddRecord(repoRoot string, record ExecutionRecord) error {
	historyPath := buildHistoryPath(repoRoot)

	records, _ := loadHistory(historyPath)
	records = append(records, record)

	return saveHistory(historyPath, records)
}
