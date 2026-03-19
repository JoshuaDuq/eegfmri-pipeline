package app

import (
	"github.com/eeg-pipeline/tui/views/history"
	"github.com/eeg-pipeline/tui/views/mainmenu"
)

const homeRecentRunsLimit = 3

func (m *Model) syncMainMenuConfigSummary() {
	m.mainMenu.SetConfigSummary(mainmenu.HomeConfigSummary{
		Task:               m.config.Task,
		BidsRoot:           m.config.BidsRoot,
		BidsFmriRoot:       m.config.BidsFmriRoot,
		DerivRoot:          m.config.DerivRoot,
		SourceRoot:         m.config.SourceRoot,
		PreprocessingNJobs: m.config.PreprocessingNJobs,
	})
}

func (m *Model) syncMainMenuSessionData() {
	m.mainMenu.SetLastPipeline(m.persistentState.LastPipeline)
}

func (m *Model) refreshMainMenuRecentRuns() {
	m.mainMenu.SetRecentRuns(m.loadRecentRuns())
}

func (m Model) loadRecentRuns() []mainmenu.RecentRunSummary {
	records, err := history.LoadRecentRecords(m.repoRoot, homeRecentRunsLimit)
	if err != nil {
		return nil
	}

	runs := make([]mainmenu.RecentRunSummary, 0, len(records))
	for _, record := range records {
		runs = append(runs, mainmenu.RecentRunSummary{
			Pipeline: record.Pipeline,
			Mode:     record.Mode,
			Age:      history.FormatTimeAgo(record.StartTime),
			Duration: history.FormatDurationSeconds(record.Duration),
			Success:  record.Success,
		})
	}
	return runs
}
