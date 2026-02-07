// Advanced config dispatcher and default fallback.
package wizard

import (
	"strings"

	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"

	"github.com/charmbracelet/lipgloss"
)

// renderAdvancedConfig dispatches to the pipeline-specific advanced config renderer.
func (m Model) renderAdvancedConfig() string {
	switch m.Pipeline {
	case types.PipelineFeatures:
		return m.renderFeaturesAdvancedConfig()
	case types.PipelineBehavior:
		return m.renderBehaviorAdvancedConfig()
	case types.PipelinePlotting:
		return m.renderPlottingAdvancedConfigV2()
	case types.PipelineML:
		return m.renderMLAdvancedConfig()
	case types.PipelinePreprocessing:
		return m.renderPreprocessingAdvancedConfig()
	case types.PipelineFmri:
		return m.renderFmriAdvancedConfig()
	case types.PipelineFmriAnalysis:
		return m.renderFmriAnalysisAdvancedConfig()
	case types.PipelineRawToBIDS:
		return m.renderRawToBidsAdvancedConfig()
	case types.PipelineFmriRawToBIDS:
		return m.renderFmriRawToBidsAdvancedConfig()
	case types.PipelineMergePsychoPyData:
		return m.renderMergeBehaviorAdvancedConfig()
	default:
		return m.renderDefaultAdvancedConfig()
	}
}

func (m Model) renderDefaultAdvancedConfig() string {
	var b strings.Builder
	b.WriteString(styles.RenderStepHeader("Advanced", m.contentWidth) + "\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render("  No advanced options for this pipeline. Press Enter to continue.") + "\n")
	return b.String()
}
