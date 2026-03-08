package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/types"
)

func (m *Model) validateCurrentStep() []string {
	validator := m.stepDefinition(m.CurrentStep).validate
	if validator == nil {
		return nil
	}
	return validator(m)
}

func (m *Model) validateSubjectSelectionStep() []string {
	var errors []string

	selectedCount := countSelectedStringItems(m.subjectSelected)
	validCount := 0
	for subjID, selected := range m.subjectSelected {
		if !selected {
			continue
		}
		for _, subject := range m.subjects {
			if subject.ID != subjID {
				continue
			}
			valid, _ := m.Pipeline.ValidateSubject(subject)
			if m.Pipeline == types.PipelinePlotting {
				valid, _ = m.validatePlottingSubject(subject)
			}
			if valid {
				validCount++
			}
			break
		}
	}

	minRequired := minSubjectsRequired
	if m.Pipeline == types.PipelineML && m.mlScope == MLCVScopeGroup {
		minRequired = minSubjectsForGroupCV
	}
	if m.Pipeline == types.PipelinePlotting && m.plottingScope == PlottingScopeGroup {
		minRequired = minSubjectsForGroupCV
	}
	if selectedCount < minRequired {
		errors = append(errors, fmt.Sprintf("Select at least %d subject(s)", minRequired))
	}
	if validCount < minRequired {
		errors = append(errors, fmt.Sprintf("Select at least %d valid subject(s)", minRequired))
	}

	return errors
}

func (m *Model) validateComputationSelectionStep() []string {
	if countSelectedItems(m.computationSelected) == 0 {
		return []string{"Select at least one analysis to run"}
	}
	return nil
}

func (m *Model) validateFeatureFileSelectionStep() []string {
	if countSelectedStringItems(m.featureFileSelected) == 0 {
		return []string{"Select at least one feature file to load"}
	}
	return nil
}

func (m *Model) validateCategorySelectionStep() []string {
	if countSelectedItems(m.selected) == 0 {
		return []string{"Select at least one category"}
	}
	return nil
}

func (m *Model) validateBandSelectionStep() []string {
	if countSelectedItems(m.bandSelected) == 0 {
		return []string{"Select at least one frequency band"}
	}
	return nil
}

func (m *Model) validatePlotSelectionStep() []string {
	if m.countSelectedVisiblePlots() == 0 {
		return []string{"Select at least one plot to generate"}
	}
	return nil
}

func (m *Model) validateFeaturePlotterSelectionStep() []string {
	if len(m.selectedFeaturePlotterCategories()) == 0 {
		return nil
	}
	if m.featurePlotters == nil && strings.TrimSpace(m.featurePlotterError) == "" {
		return []string{"Feature plot list is still loading"}
	}
	if m.featurePlotters == nil && strings.TrimSpace(m.featurePlotterError) != "" {
		return nil
	}

	count := 0
	for _, plotter := range m.featurePlotterItems() {
		if m.featurePlotterSelected[plotter.ID] {
			count++
		}
	}
	if count == 0 {
		return []string{"Select at least one feature plot"}
	}
	return nil
}

func (m *Model) validateSpatialSelectionStep() []string {
	if countSelectedItems(m.spatialSelected) == 0 {
		return []string{"Select at least one spatial mode"}
	}
	return nil
}

func (m *Model) validatePreprocessingStageSelectionStep() []string {
	if countSelectedItems(m.prepStageSelected) == 0 {
		return []string{"Select at least one preprocessing stage"}
	}
	return nil
}

func (m *Model) validateTimeRangeStep() []string {
	return m.validateTimeRanges()
}

func (m *Model) validatePlotConfigStep() []string {
	if countSelectedStringItems(m.plotFormatSelected) == 0 {
		return []string{"Select at least one output format (PNG, SVG, or PDF)"}
	}
	return nil
}
