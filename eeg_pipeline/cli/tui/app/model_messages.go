package app

import (
	"fmt"

	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/types"
	"github.com/eeg-pipeline/tui/views/wizard"

	tea "github.com/charmbracelet/bubbletea"
)

// Asynchronous message handlers and data conversion helpers.

func (m Model) handleSubjectsLoaded(msg messages.SubjectsLoadedMsg) (tea.Model, tea.Cmd) {
	if msg.Error != nil {
		m.execution.AddOutput("Error loading subjects: " + msg.Error.Error())
		m.wizard.SetSubjectLoadError(msg.Error.Error())
		return m, nil
	}

	if m.pendingSubjectsCacheKey != "" {
		m.subjectsCache[m.pendingSubjectsCacheKey] = msg
		m.pendingSubjectsCacheKey = ""
	}

	subjects := m.convertSubjects(msg.Subjects)
	m.wizard.SetSubjects(subjects)
	m.wizard.SetAvailableMetadata(msg.AvailableWindows, msg.AvailableEventColumns)
	if msg.AvailableWindowsByFeature != nil {
		m.wizard.SetAvailableWindowsByFeature(msg.AvailableWindowsByFeature)
	}
	m.wizard.SetChannelInfo(msg.AvailableChannels, msg.UnavailableChannels)

	// Trigger condition effects discovery for plotting pipeline
	if m.selectedPipeline == types.PipelinePlotting && len(subjects) > 0 {
		// Use first subject for discovery
		subjectID := ""
		for _, subj := range subjects {
			if subj.ID != "" {
				subjectID = subj.ID
				break
			}
		}
		return m, executor.DiscoverConditionEffectsColumns(m.repoRoot, m.task, subjectID)
	}

	// Trigger fMRI condition discovery for fMRI analysis pipeline (for contrast pickers)
	if m.selectedPipeline == types.PipelineFmriAnalysis && len(subjects) > 0 {
		subjectID := ""
		for _, subj := range subjects {
			if subj.ID != "" {
				subjectID = subj.ID
				break
			}
		}
		if subjectID != "" {
			return m, executor.DiscoverFmriConditions(
				m.repoRoot,
				subjectID,
				m.task,
				m.wizard.FmriDiscoveryConditionColumn(),
			)
		}
	}

	return m, nil
}

func (m *Model) convertSubjects(sourceSubjects []messages.SubjectInfo) []types.SubjectStatus {
	subjects := make([]types.SubjectStatus, len(sourceSubjects))
	for i, s := range sourceSubjects {
		subjects[i] = types.SubjectStatus{
			ID:                  s.ID,
			HasSourceData:       s.HasSourceData,
			HasBids:             s.HasBids,
			HasDerivatives:      s.HasDerivatives,
			HasEpochs:           s.HasEpochs,
			HasFeatures:         s.HasFeatures,
			HasStats:            s.HasStats,
			AvailableBands:      s.AvailableBands,
			FeatureAvailability: m.convertFeatureAvailability(s.FeatureAvailability),
			EpochMetadata:       s.EpochMetadata,
		}
	}
	return subjects
}

func (m *Model) convertFeatureAvailability(source *messages.FeatureAvailability) *types.FeatureAvailability {
	if source == nil {
		return nil
	}

	featAvail := &types.FeatureAvailability{
		Features:     make(map[string]types.AvailabilityInfo),
		Bands:        make(map[string]types.AvailabilityInfo),
		Computations: make(map[string]types.AvailabilityInfo),
	}

	for k, v := range source.Features {
		featAvail.Features[k] = m.convertAvailabilityInfo(v)
	}
	for k, v := range source.Bands {
		featAvail.Bands[k] = m.convertAvailabilityInfo(v)
	}
	for k, v := range source.Computations {
		featAvail.Computations[k] = m.convertAvailabilityInfo(v)
	}

	return featAvail
}

func (m *Model) convertAvailabilityInfo(v messages.AvailabilityInfo) types.AvailabilityInfo {
	lastModified := ""
	if v.LastModified != nil {
		lastModified = *v.LastModified
	}
	return types.AvailabilityInfo{
		Available:    v.Available,
		LastModified: lastModified,
	}
}

func (m *Model) handlePlottersLoaded(msg messages.PlottersLoadedMsg) {
	if msg.Error != nil {
		m.wizard.SetFeaturePlottersError(msg.Error)
		return
	}

	if msg.FeaturePlotters == nil {
		return
	}

	converted := m.convertPlotters(msg.FeaturePlotters)
	m.wizard.SetFeaturePlotters(converted)
}

func (m *Model) convertPlotters(source map[string][]messages.PlotterInfo) map[string][]wizard.PlotterInfo {
	converted := make(map[string][]wizard.PlotterInfo, len(source))
	for category, entries := range source {
		list := make([]wizard.PlotterInfo, 0, len(entries))
		for _, p := range entries {
			list = append(list, wizard.PlotterInfo{
				ID:       p.ID,
				Category: p.Category,
				Name:     p.Name,
			})
		}
		converted[category] = list
	}
	return converted
}

func (m *Model) handleColumnsDiscovered(msg messages.ColumnsDiscoveredMsg) {
	// Check if this is condition effects discovery (source will be "condition_effects")
	if msg.Source == "condition_effects" {
		if msg.Error != nil {
			m.wizard.SetConditionEffectsDiscoveryError(msg.Error)
			return
		}
		// Extract windows from the response if available
		windows := []string{}
		if msg.Windows != nil {
			windows = msg.Windows
		}
		m.wizard.SetConditionEffectsColumns(msg.Columns, msg.Values, windows)
		return
	}

	// Trial-table discovery is used for feature column dropdowns (e.g., dose-response response_column).
	// Keep it separate from the primary discovered columns (events) to avoid polluting event-column dropdowns.
	if msg.Source == "trial_table" {
		if msg.Error != nil {
			m.wizard.SetTrialTableDiscoveryError(msg.Error)
			return
		}
		m.wizard.SetTrialTableColumns(msg.Columns, msg.Values)
		return
	} else {
		if msg.Error != nil {
			m.wizard.SetColumnsDiscoveryError(msg.Error)
			return
		}
		m.wizard.SetDiscoveredColumns(msg.Columns, msg.Values, msg.Source)
	}
}

func (m *Model) handleFmriColumnsDiscovered(msg messages.FmriColumnsDiscoveredMsg) {
	if msg.Error != nil {
		m.wizard.SetFmriColumnsDiscoveryError(msg.Error)
		return
	}
	m.wizard.SetFmriDiscoveredColumns(msg.Columns, msg.Values, msg.Source)
}

func (m *Model) handleROIsDiscovered(msg messages.ROIsDiscoveredMsg) {
	if msg.Error != nil {
		m.wizard.SetROIDiscoveryError(msg.Error)
		return
	}
	m.wizard.SetDiscoveredROIs(msg.ROIs)
}

func (m *Model) handleMultigroupStatsDiscovered(msg messages.MultigroupStatsDiscoveredMsg) {
	if msg.Error != nil {
		return
	}
	m.wizard.SetMultigroupStats(msg.Available, msg.Groups, msg.NFeatures, msg.NSignificant, msg.File)
}

func (m Model) handleRefreshSubjects() tea.Cmd {
	if m.state != StatePipelineWizard {
		return nil
	}
	m.wizard.SetSubjectsLoading()
	cacheKey := fmt.Sprintf("%s|%s", m.task, m.selectedPipeline.GetDataSource())
	m.pendingSubjectsCacheKey = cacheKey
	return executor.LoadSubjectsRefresh(m.repoRoot, m.task, m.selectedPipeline)
}

func (m Model) handleConfigLoaded(msg messages.ConfigLoadedMsg) (tea.Model, tea.Cmd) {
	if msg.Error != nil {
		return m, nil
	}

	m.config = msg.Summary
	m.syncMainMenuConfigSummary()
	m.wizard.SetConfigSummary(msg.Summary)
	if !m.shouldUpdateTask(msg.Summary.Task) {
		return m, nil
	}

	m.updateTask(msg.Summary.Task)
	if m.state == StatePipelineWizard {
		m.wizard.SetSubjectsLoading()
		cacheKey := fmt.Sprintf("%s|%s", m.task, m.selectedPipeline.GetDataSource())
		m.pendingSubjectsCacheKey = cacheKey
		return m, executor.LoadSubjectsRefresh(m.repoRoot, m.task, m.selectedPipeline)
	}
	return m, nil
}

func (m Model) handleTaskUpdated(msg messages.TaskUpdatedMsg) (tea.Model, tea.Cmd) {
	if !m.shouldUpdateTask(msg.Task) {
		return m, nil
	}

	m.updateTask(msg.Task)
	if m.state == StatePipelineWizard {
		m.wizard.SetSubjectsLoading()
		cacheKey := fmt.Sprintf("%s|%s", m.task, m.selectedPipeline.GetDataSource())
		m.pendingSubjectsCacheKey = cacheKey
		return m, executor.LoadSubjectsRefresh(m.repoRoot, m.task, m.selectedPipeline)
	}
	return m, nil
}

func (m *Model) shouldUpdateTask(newTask string) bool {
	return newTask != "" && newTask != m.task
}

func (m *Model) updateTask(newTask string) {
	m.task = newTask
	m.config.Task = newTask
	m.mainMenu.Task = m.task
	m.syncMainMenuConfigSummary()
	m.pipelineSmoke.SetTask(m.task)
	m.wizard.SetTask(m.task)
}

func (m *Model) handleConfigKeysLoaded(msg messages.ConfigKeysLoadedMsg) {
	if msg.Error != nil {
		return
	}

	switch m.state {
	case StateGlobalSetup:
		m.global.SetConfigValues(msg.Values)
	case StatePipelineWizard:
		m.wizard.ApplyConfigKeys(msg.Values)
		// ApplyConfigKeys overwrites bands from YAML defaults. Restore only
		// bands from persisted state so user-customised bands survive. Do NOT
		// call the full restoreWizardConfig() here — it would also reapply
		// persisted selections (categories, plots, etc.) and overwrite any
		// changes the user has already made in the current session, causing
		// plot options to flash and disappear.
		m.restoreWizardConfigValues()
	}
}
