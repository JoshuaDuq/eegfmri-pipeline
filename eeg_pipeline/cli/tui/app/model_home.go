package app

import "github.com/eeg-pipeline/tui/views/mainmenu"

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
