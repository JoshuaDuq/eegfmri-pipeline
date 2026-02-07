// fMRI analysis pipeline advanced configuration.
package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

func (m Model) renderFmriAnalysisAdvancedConfig() string {
	var b strings.Builder

	b.WriteString(styles.RenderStepHeader("Advanced", m.contentWidth) + "\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
	if m.editingNumber {
		b.WriteString(infoStyle.Render("Enter value, Enter to confirm, Esc to cancel") + "\n")
	} else if m.editingText {
		b.WriteString(infoStyle.Render("Type text, Enter to confirm, Esc to cancel") + "\n")
	} else if m.expandedOption >= 0 {
		b.WriteString(infoStyle.Render("Space: select  Esc: close list") + "\n")
	} else {
		b.WriteString(infoStyle.Render("Space: toggle/expand  Enter: proceed") + "\n")
	}

	if m.useDefaultAdvanced {
		return m.renderDefaultConfigView("fMRI analysis")
	}

	labelWidth := defaultLabelWidthWide
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	// Display values
	inputSourceOptions := []string{"fmriprep", "bids_raw"}
	inputSourceVal := inputSourceOptions[m.fmriAnalysisInputSourceIndex%len(inputSourceOptions)]

	spaceVal := strings.TrimSpace(m.fmriAnalysisFmriprepSpace)
	if m.editingText && m.editingTextField == textFieldFmriAnalysisFmriprepSpace {
		spaceVal = m.textBuffer + "█"
	}
	if spaceVal == "" {
		spaceVal = "T1w"
	}

	runsVal := strings.TrimSpace(m.fmriAnalysisRunsSpec)
	if m.editingText && m.editingTextField == textFieldFmriAnalysisRuns {
		runsVal = m.textBuffer + "█"
	}
	if runsVal == "" {
		runsVal = "(auto)"
	}

	contrastTypeOptions := []string{"t-test", "custom"}
	contrastTypeVal := contrastTypeOptions[m.fmriAnalysisContrastType%len(contrastTypeOptions)]

	condAColumnVal := strings.TrimSpace(m.fmriAnalysisCondAColumn)
	if m.editingText && m.editingTextField == textFieldFmriAnalysisCondAColumn {
		condAColumnVal = m.textBuffer + "█"
	}
	if condAColumnVal == "" {
		condAColumnVal = "(select column)"
	}

	condAVal := strings.TrimSpace(m.fmriAnalysisCondAValue)
	if m.editingText && m.editingTextField == textFieldFmriAnalysisCondAValue {
		condAVal = m.textBuffer + "█"
	}
	if condAVal == "" {
		condAVal = "(select value)"
	}

	condBColumnVal := strings.TrimSpace(m.fmriAnalysisCondBColumn)
	if m.editingText && m.editingTextField == textFieldFmriAnalysisCondBColumn {
		condBColumnVal = m.textBuffer + "█"
	}
	if condBColumnVal == "" {
		condBColumnVal = "(select column)"
	}

	condBVal := strings.TrimSpace(m.fmriAnalysisCondBValue)
	if m.editingText && m.editingTextField == textFieldFmriAnalysisCondBValue {
		condBVal = m.textBuffer + "█"
	}
	if condBVal == "" {
		condBVal = "(optional)"
	}

	contrastNameVal := strings.TrimSpace(m.fmriAnalysisContrastName)
	if m.editingText && m.editingTextField == textFieldFmriAnalysisContrastName {
		contrastNameVal = m.textBuffer + "█"
	}
	if contrastNameVal == "" {
		contrastNameVal = "(required)"
	}

	formulaVal := strings.TrimSpace(m.fmriAnalysisFormula)
	if m.editingText && m.editingTextField == textFieldFmriAnalysisFormula {
		formulaVal = m.textBuffer + "█"
	}
	if formulaVal == "" {
		formulaVal = "(none)"
	}

	eventsToModelVal := strings.TrimSpace(m.fmriAnalysisEventsToModel)
	if m.editingText && m.editingTextField == textFieldFmriAnalysisEventsToModel {
		eventsToModelVal = m.textBuffer + "█"
	}
	if eventsToModelVal == "" {
		eventsToModelVal = "(all)"
	}

	scopeTrialTypesVal := strings.TrimSpace(m.fmriAnalysisScopeTrialTypes)
	if m.editingText && m.editingTextField == textFieldFmriAnalysisScopeTrialTypes {
		scopeTrialTypesVal = m.textBuffer + "█"
	}
	if scopeTrialTypesVal == "" {
		scopeTrialTypesVal = "(none)"
	}

	hrfOptions := []string{"spm", "flobs", "fir"}
	hrfVal := hrfOptions[m.fmriAnalysisHrfModel%len(hrfOptions)]
	driftOptions := []string{"none", "cosine", "polynomial"}
	driftVal := driftOptions[m.fmriAnalysisDriftModel%len(driftOptions)]

	confoundsOptions := []string{
		"auto",
		"none",
		"motion6",
		"motion12",
		"motion24",
		"motion24+wmcsf",
		"motion24+wmcsf+fd",
	}
	confoundsVal := confoundsOptions[m.fmriAnalysisConfoundsStrategy%len(confoundsOptions)]

	highPassVal := fmt.Sprintf("%.3f", m.fmriAnalysisHighPassHz)
	lowPassVal := fmt.Sprintf("%.3f", m.fmriAnalysisLowPassHz)
	if m.fmriAnalysisLowPassHz <= 0 {
		lowPassVal = "0 (disabled)"
	}
	smoothingVal := fmt.Sprintf("%.1f", m.fmriAnalysisSmoothingFwhm)
	if m.fmriAnalysisSmoothingFwhm <= 0 {
		smoothingVal = "0 (disabled)"
	}

	// Plotting / report display values
	plotSpaceOptions := []string{"both", "native", "mni"}
	plotSpaceVal := plotSpaceOptions[m.fmriAnalysisPlotSpaceIndex%len(plotSpaceOptions)]
	plotZVal := fmt.Sprintf("%.2f", m.fmriAnalysisPlotZThreshold)
	plotThresholdModeOptions := []string{"z", "fdr", "none"}
	plotThresholdModeVal := plotThresholdModeOptions[m.fmriAnalysisPlotThresholdModeIndex%len(plotThresholdModeOptions)]
	plotFdrQVal := fmt.Sprintf("%.3f", m.fmriAnalysisPlotFdrQ)
	plotClusterMinVal := fmt.Sprintf("%d", m.fmriAnalysisPlotClusterMinVoxels)
	plotVmaxModeOptions := []string{"per-space-robust", "shared-robust", "manual"}
	plotVmaxModeVal := plotVmaxModeOptions[m.fmriAnalysisPlotVmaxModeIndex%len(plotVmaxModeOptions)]
	plotVmaxManualVal := fmt.Sprintf("%.2f", m.fmriAnalysisPlotVmaxManual)

	if m.editingNumber {
		buffer := m.numberBuffer + "█"
		switch {
		case m.isCurrentlyEditing(optFmriAnalysisHighPassHz):
			highPassVal = buffer
		case m.isCurrentlyEditing(optFmriAnalysisLowPassHz):
			lowPassVal = buffer
		case m.isCurrentlyEditing(optFmriAnalysisSmoothingFwhm):
			smoothingVal = buffer
		case m.isCurrentlyEditing(optFmriAnalysisPlotZThreshold):
			plotZVal = buffer
		case m.isCurrentlyEditing(optFmriAnalysisPlotFdrQ):
			plotFdrQVal = buffer
		case m.isCurrentlyEditing(optFmriAnalysisPlotClusterMinVoxels):
			plotClusterMinVal = buffer
		case m.isCurrentlyEditing(optFmriAnalysisPlotVmaxManual):
			plotVmaxManualVal = buffer
		}
	}

	outTypeOptions := []string{"z-score", "t-stat", "cope", "beta"}
	outTypeVal := outTypeOptions[m.fmriAnalysisOutputType%len(outTypeOptions)]

	outDirVal := strings.TrimSpace(m.fmriAnalysisOutputDir)
	if m.editingText && m.editingTextField == textFieldFmriAnalysisOutputDir {
		outDirVal = m.textBuffer + "█"
	}
	if outDirVal == "" {
		outDirVal = "(default)"
	}

	fsDirVal := strings.TrimSpace(m.fmriAnalysisFreesurferDir)
	if m.editingText && m.editingTextField == textFieldFmriAnalysisFreesurferDir {
		fsDirVal = m.textBuffer + "█"
	}
	if fsDirVal == "" {
		fsDirVal = "(from config)"
	}

	options := m.getFmriAnalysisOptions()

	// Scrolling support: count option lines + expanded list lines
	totalLines := 0
	for _, opt := range options {
		totalLines++
		if m.shouldRenderExpandedListAfterOption(opt) {
			totalLines += m.getExpandedListLength()
		}
	}
	effectiveHeight := m.height
	if effectiveHeight <= 0 {
		effectiveHeight = defaultTerminalHeight
	}
	startLine, endLine, showScrollIndicators := calculateScrollWindow(
		totalLines, m.advancedOffset, effectiveHeight, configOverhead)

	if showScrollIndicators && startLine > 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↑ %d more above", startLine)) + "\n")
	}

	lineIdx := 0
	for i, opt := range options {
		if lineIdx >= endLine {
			break
		}
		inRange := lineIdx >= startLine

		isFocused := i == m.advancedCursor
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursorOptional(m.CursorBlinkVisible())
		}

		var labelStyle, valueStyle lipgloss.Style
		if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text)
		}
		valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)

		label := ""
		value := ""
		hint := ""
		expandIndicator := ""

		switch opt {
		case optUseDefaults:
			label = "Configuration"
			value = "Custom"
			hint = "Space to use defaults"

		// Group headers
		case optFmriAnalysisGroupInput:
			if m.fmriAnalysisGroupInputExpanded {
				label = "▾ Input"
			} else {
				label = "▸ Input"
			}
			hint = "BOLD source + runs"
			if !isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFmriAnalysisGroupContrast:
			if m.fmriAnalysisGroupContrastExpanded {
				label = "▾ Contrast"
			} else {
				label = "▸ Contrast"
			}
			hint = "Conditions + name"
			if !isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFmriAnalysisGroupGLM:
			if m.fmriAnalysisGroupGLMExpanded {
				label = "▾ GLM"
			} else {
				label = "▸ GLM"
			}
			hint = "HRF, drift, filters"
			if !isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFmriAnalysisGroupConfounds:
			if m.fmriAnalysisGroupConfoundsExpanded {
				label = "▾ Confounds/QC"
			} else {
				label = "▸ Confounds/QC"
			}
			hint = "Nuisance + QC outputs"
			if !isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFmriAnalysisGroupOutput:
			if m.fmriAnalysisGroupOutputExpanded {
				label = "▾ Output"
			} else {
				label = "▸ Output"
			}
			hint = "Map type + paths"
			if !isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFmriAnalysisGroupPlotting:
			if m.fmriAnalysisGroupPlottingExpanded {
				label = "▾ Plotting"
			} else {
				label = "▸ Plotting"
			}
			hint = "Figures + HTML report"
			if !isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFmriTrialSigGroup:
			if m.fmriTrialSigGroupExpanded {
				label = "▾ Trial Signatures"
			} else {
				label = "▸ Trial Signatures"
			}
			hint = "Beta-series / LSS + signature readouts"
			if !isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		// Input
		case optFmriAnalysisInputSource:
			label = "Input Source"
			value = inputSourceVal
			hint = "Space to toggle"
		case optFmriAnalysisFmriprepSpace:
			label = "fMRIPrep Space"
			value = spaceVal
			hint = "Space to edit"
		case optFmriAnalysisRequireFmriprep:
			label = "Require fMRIPrep"
			if m.fmriAnalysisRequireFmriprep {
				value = "true"
			} else {
				value = "false"
			}
			hint = "Space to toggle"
		case optFmriAnalysisRuns:
			label = "Runs"
			value = runsVal
			hint = "Space to edit"

		// Contrast
		case optFmriAnalysisContrastType:
			label = "Type"
			value = contrastTypeVal
			hint = "Space to toggle"
		case optFmriAnalysisCondAColumn:
			label = "Cond A Column"
			value = condAColumnVal
			if len(m.fmriDiscoveredColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns", len(m.fmriDiscoveredColumns))
			} else {
				hint = "Space to edit"
			}
		case optFmriAnalysisCondAValue:
			label = "Cond A Value"
			value = condAVal
			if m.fmriAnalysisCondAColumn == "" {
				hint = "Select column first"
			} else if vals := m.GetFmriDiscoveredColumnValues(m.fmriAnalysisCondAColumn); len(vals) > 0 {
				hint = fmt.Sprintf("Space to select · %d values in %s", len(vals), m.fmriAnalysisCondAColumn)
			} else {
				hint = "Space to edit"
			}
		case optFmriAnalysisCondBColumn:
			label = "Cond B Column"
			value = condBColumnVal
			if len(m.fmriDiscoveredColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns", len(m.fmriDiscoveredColumns))
			} else {
				hint = "Space to edit"
			}
		case optFmriAnalysisCondBValue:
			label = "Cond B Value"
			value = condBVal
			if m.fmriAnalysisCondBColumn == "" {
				hint = "Select column first"
			} else if vals := m.GetFmriDiscoveredColumnValues(m.fmriAnalysisCondBColumn); len(vals) > 0 {
				hint = fmt.Sprintf("Space to select · %d values in %s", len(vals), m.fmriAnalysisCondBColumn)
			} else {
				hint = "Space to edit"
			}
		case optFmriAnalysisContrastName:
			label = "Contrast Name"
			value = contrastNameVal
			hint = "Space to edit"
		case optFmriAnalysisFormula:
			label = "Formula"
			value = formulaVal
			hint = "Space to edit"

		// GLM
		case optFmriAnalysisHrfModel:
			label = "HRF"
			value = hrfVal
			hint = "Space to toggle"
		case optFmriAnalysisDriftModel:
			label = "Drift"
			value = driftVal
			hint = "Space to toggle"
		case optFmriAnalysisHighPassHz:
			label = "High-pass (Hz)"
			value = highPassVal
			hint = "Space to edit"
		case optFmriAnalysisLowPassHz:
			label = "Low-pass (Hz)"
			value = lowPassVal
			hint = "Space to edit (0 disables)"
		case optFmriAnalysisSmoothingFwhm:
			label = "Smoothing (FWHM mm)"
			value = smoothingVal
			hint = "Space to edit (0 disables)"

		// Confounds / QC
		case optFmriAnalysisEventsToModel:
			label = "Events to Model"
			value = eventsToModelVal
			hint = "Space to edit (comma-separated; empty = all)"
		case optFmriAnalysisScopeTrialTypes:
			label = "Condition trial_type Scope"
			value = scopeTrialTypesVal
			expandIndicatorHint := ""
			if vals := m.GetFmriDiscoveredColumnValues("trial_type"); len(vals) > 0 {
				expandIndicatorHint = fmt.Sprintf(" · %d values in trial_type", len(vals))
			}
			hint = "Space to select" + expandIndicatorHint
			if m.expandedOption == expandedFmriAnalysisScopeTrialTypes {
				expandIndicator = " [-]"
			} else {
				expandIndicator = " [+]"
			}
		case optFmriAnalysisStimPhasesToModel:
			label = "Stim Phase Scope"
			val := strings.TrimSpace(m.fmriAnalysisStimPhasesToModel)
			if val == "" {
				val = "(none)"
			}
			if m.editingText && m.editingTextField == textFieldFmriAnalysisStimPhasesToModel {
				val = m.textBuffer + "█"
			}
			value = val
			expandIndicatorHint := ""
			if vals := m.GetFmriDiscoveredColumnValues("stim_phase"); len(vals) > 0 {
				expandIndicatorHint = fmt.Sprintf(" · %d values in stim_phase", len(vals))
			}
			hint = "Space to select" + expandIndicatorHint
			if m.expandedOption == expandedFmriAnalysisStimPhases {
				expandIndicator = " [-]"
			} else {
				expandIndicator = " [+]"
			}
		case optFmriAnalysisConfoundsStrategy:
			label = "Confounds"
			value = confoundsVal
			hint = "Space to cycle"
		case optFmriAnalysisWriteDesignMatrix:
			label = "Write Design Matrix"
			if m.fmriAnalysisWriteDesignMatrix {
				value = "true"
			} else {
				value = "false"
			}
			hint = "Space to toggle"

		// Output
		case optFmriAnalysisOutputType:
			label = "Output Type"
			value = outTypeVal
			hint = "Space to toggle"
		case optFmriAnalysisOutputDir:
			label = "Output Dir"
			value = outDirVal
			hint = "Space to edit"
		case optFmriAnalysisResampleToFS:
			label = "Resample to FS"
			if m.fmriAnalysisResampleToFS {
				value = "true"
			} else {
				value = "false"
			}
			hint = "Space to toggle"
		case optFmriAnalysisFreesurferDir:
			label = "FreeSurfer Dir"
			value = fsDirVal
			hint = "Space to edit"

		// Plotting / report
		case optFmriAnalysisPlotsEnabled:
			label = "Generate Plots"
			value = m.boolToOnOff(m.fmriAnalysisPlotsEnabled)
			hint = "Space to toggle"
		case optFmriAnalysisPlotHTML:
			label = "HTML Report"
			value = m.boolToOnOff(m.fmriAnalysisPlotHTML)
			hint = "Space to toggle"
		case optFmriAnalysisPlotSpace:
			label = "Plot Space"
			value = plotSpaceVal
			hint = "Space to cycle"
		case optFmriAnalysisPlotThresholdMode:
			label = "Threshold Mode"
			value = plotThresholdModeVal
			hint = "Space to cycle"
		case optFmriAnalysisPlotZThreshold:
			label = "Z Threshold"
			value = plotZVal
			hint = "Space to edit (used when mode=z)"
		case optFmriAnalysisPlotFdrQ:
			label = "FDR q"
			value = plotFdrQVal
			hint = "Space to edit (used when mode=fdr)"
		case optFmriAnalysisPlotClusterMinVoxels:
			label = "Min Cluster Vox"
			value = plotClusterMinVal
			hint = "Space to edit (0 disables)"
		case optFmriAnalysisPlotVmaxMode:
			label = "Vmax Mode"
			value = plotVmaxModeVal
			hint = "Space to cycle"
		case optFmriAnalysisPlotVmaxManual:
			label = "Vmax Manual"
			value = plotVmaxManualVal
			hint = "Space to edit"
		case optFmriAnalysisPlotIncludeUnthresholded:
			label = "Include Unthresholded"
			value = m.boolToOnOff(m.fmriAnalysisPlotIncludeUnthresholded)
			hint = "Space to toggle"
		case optFmriAnalysisPlotFormatPNG:
			label = "Format: PNG"
			value = m.boolToOnOff(m.fmriAnalysisPlotFormatPNG)
			hint = "Space to toggle"
		case optFmriAnalysisPlotFormatSVG:
			label = "Format: SVG"
			value = m.boolToOnOff(m.fmriAnalysisPlotFormatSVG)
			hint = "Space to toggle"
		case optFmriAnalysisPlotTypeSlices:
			label = "Plot: Slices"
			value = m.boolToOnOff(m.fmriAnalysisPlotTypeSlices)
			hint = "Space to toggle"
		case optFmriAnalysisPlotTypeGlass:
			label = "Plot: Glass"
			value = m.boolToOnOff(m.fmriAnalysisPlotTypeGlass)
			hint = "Space to toggle"
		case optFmriAnalysisPlotTypeHist:
			label = "Plot: Histogram"
			value = m.boolToOnOff(m.fmriAnalysisPlotTypeHist)
			hint = "Space to toggle"
		case optFmriAnalysisPlotTypeClusters:
			label = "Plot: Clusters"
			value = m.boolToOnOff(m.fmriAnalysisPlotTypeClusters)
			hint = "Space to toggle"
		case optFmriAnalysisPlotEffectSize:
			label = "Plot: Effect Size"
			value = m.boolToOnOff(m.fmriAnalysisPlotEffectSize)
			hint = "Space to toggle"
		case optFmriAnalysisPlotStandardError:
			label = "Plot: Std Error"
			value = m.boolToOnOff(m.fmriAnalysisPlotStandardError)
			hint = "Space to toggle"
		case optFmriAnalysisPlotMotionQC:
			label = "QC: Motion"
			value = m.boolToOnOff(m.fmriAnalysisPlotMotionQC)
			hint = "Space to toggle"
		case optFmriAnalysisPlotCarpetQC:
			label = "QC: Carpet"
			value = m.boolToOnOff(m.fmriAnalysisPlotCarpetQC)
			hint = "Space to toggle"
		case optFmriAnalysisPlotTSNRQC:
			label = "QC: tSNR"
			value = m.boolToOnOff(m.fmriAnalysisPlotTSNRQC)
			hint = "Space to toggle"
		case optFmriAnalysisPlotDesignQC:
			label = "QC: Design"
			value = m.boolToOnOff(m.fmriAnalysisPlotDesignQC)
			hint = "Space to toggle"
		case optFmriAnalysisPlotEmbedImages:
			label = "Embed Images"
			value = m.boolToOnOff(m.fmriAnalysisPlotEmbedImages)
			hint = "Space to toggle"
		case optFmriAnalysisPlotSignatures:
			label = "Pain Signatures (NPS/SIIPS1)"
			value = m.boolToOnOff(m.fmriAnalysisPlotSignatures)
			hint = "Space to toggle"
		case optFmriAnalysisSignatureDir:
			label = "Signature Dir"
			value = strings.TrimSpace(m.fmriAnalysisSignatureDir)
			if m.editingText && m.editingTextField == textFieldFmriAnalysisSignatureDir {
				value = m.textBuffer + "█"
			}
			if value == "" {
				value = "(auto: <deriv_root>/../external)"
			}
			hint = "Space to edit"
		case optFmriTrialSigScopeStimPhases:
			label = "Stim Phase Scope"
			val := strings.TrimSpace(m.fmriTrialSigScopeStimPhases)
			if val == "" {
				val = "(none)"
			}
			if m.editingText && m.editingTextField == textFieldFmriTrialSigScopeStimPhases {
				val = m.textBuffer + "█"
			}
			value = val
			expandIndicatorHint := ""
			if vals := m.GetFmriDiscoveredColumnValues("stim_phase"); len(vals) > 0 {
				expandIndicatorHint = fmt.Sprintf(" · %d values in stim_phase", len(vals))
			}
			hint = "Space to select" + expandIndicatorHint
			if m.expandedOption == expandedFmriTrialSigStimPhases {
				expandIndicator = " [-]"
			} else {
				expandIndicator = " [+]"
			}
		case optFmriTrialSigScopeTrialTypes:
			label = "trial_type Scope"
			val := strings.TrimSpace(m.fmriTrialSigScopeTrialTypes)
			if val == "" {
				val = "(none)"
			}
			if m.editingText && m.editingTextField == textFieldFmriTrialSigScopeTrialTypes {
				val = m.textBuffer + "█"
			}
			value = val
			expandIndicatorHint := ""
			if vals := m.GetFmriDiscoveredColumnValues("trial_type"); len(vals) > 0 {
				expandIndicatorHint = fmt.Sprintf(" · %d values in trial_type", len(vals))
			}
			hint = "Space to select" + expandIndicatorHint
			if m.expandedOption == expandedFmriTrialSigScopeTrialTypes {
				expandIndicator = " [-]"
			} else {
				expandIndicator = " [+]"
			}
		case optFmriTrialSigGroupColumn:
			label = "Group Column"
			value = strings.TrimSpace(m.fmriTrialSigGroupColumn)
			if m.editingText && m.editingTextField == textFieldFmriTrialSigGroupColumn {
				value = m.textBuffer + "█"
			}
			if value == "" {
				value = "(select column)"
			}
			if len(m.fmriDiscoveredColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns", len(m.fmriDiscoveredColumns))
			} else {
				hint = "Space to edit"
			}
		case optFmriTrialSigGroupValues:
			label = "Group Values"
			value = strings.TrimSpace(m.fmriTrialSigGroupValuesSpec)
			if m.editingText && m.editingTextField == textFieldFmriTrialSigGroupValues {
				value = m.textBuffer + "█"
			}
			if value == "" {
				value = "(select values)"
			}
			if strings.TrimSpace(m.fmriTrialSigGroupColumn) == "" {
				hint = "Select column first"
			} else if vals := m.GetFmriDiscoveredColumnValues(m.fmriTrialSigGroupColumn); len(vals) > 0 {
				hint = fmt.Sprintf("Space to select · %d values in %s", len(vals), m.fmriTrialSigGroupColumn)
			} else {
				hint = "Space to edit"
			}
		case optFmriTrialSigGroupScope:
			label = "Group Scope"
			if m.fmriTrialSigGroupScopeIndex%2 == 1 {
				value = "per-run"
			} else {
				value = "across-runs (avg)"
			}
			hint = "Space to toggle"

		// Trial-wise signatures
		case optFmriTrialSigMethod:
			label = "Trial Method"
			if m.fmriTrialSigMethodIndex%2 == 1 {
				value = "lss"
				hint = "Space to toggle (one GLM per trial)"
			} else {
				value = "beta-series"
				hint = "Space to toggle (one GLM per run)"
			}
		case optFmriTrialSigIncludeOtherEvents:
			label = "Include Other Events"
			value = m.boolToOnOff(m.fmriTrialSigIncludeOtherEvents)
			hint = "Space to toggle"
		case optFmriTrialSigMaxTrialsPerRun:
			label = "Max Trials/Run"
			if m.editingNumber && m.isCurrentlyEditing(optFmriTrialSigMaxTrialsPerRun) {
				value = m.numberBuffer + "█"
			} else if m.fmriTrialSigMaxTrialsPerRun <= 0 {
				value = "(none)"
			} else {
				value = fmt.Sprintf("%d", m.fmriTrialSigMaxTrialsPerRun)
			}
			hint = "Space to edit (0 disables)"
		case optFmriTrialSigFixedEffectsWeighting:
			label = "Fixed-Effects Weighting"
			if m.fmriTrialSigFixedEffectsWeighting%2 == 1 {
				value = "mean"
			} else {
				value = "variance"
			}
			hint = "Space to toggle"
		case optFmriTrialSigWriteTrialBetas:
			label = "Write Trial Betas"
			value = m.boolToOnOff(m.fmriTrialSigWriteTrialBetas)
			hint = "Space to toggle"
		case optFmriTrialSigWriteTrialVariances:
			label = "Write Trial Variances"
			value = m.boolToOnOff(m.fmriTrialSigWriteTrialVariances)
			hint = "Space to toggle"
		case optFmriTrialSigWriteConditionBetas:
			label = "Write Condition Betas"
			value = m.boolToOnOff(m.fmriTrialSigWriteConditionBetas)
			hint = "Space to toggle"
		case optFmriTrialSigSignatureNPS:
			label = "Signature: NPS"
			value = m.boolToOnOff(m.fmriTrialSigSignatureNPS)
			hint = "Space to toggle"
		case optFmriTrialSigSignatureSIIPS1:
			label = "Signature: SIIPS1"
			value = m.boolToOnOff(m.fmriTrialSigSignatureSIIPS1)
			hint = "Space to toggle"
		case optFmriTrialSigLssOtherRegressors:
			label = "LSS Other Trials"
			if m.fmriTrialSigLssOtherRegressorsIndex%2 == 1 {
				value = "all"
			} else {
				value = "per-condition"
			}
			hint = "Space to toggle"
		}

		styledHint := ""
		if hint != "" {
			styledHint = hintStyle.Render(hint)
		}
		if inRange {
			b.WriteString(styles.RenderConfigLine(cursor, labelStyle.Render(label), valueStyle.Render(value+expandIndicator), styledHint, labelWidth, m.contentWidth) + "\n")
		}
		lineIdx++
		if m.shouldRenderExpandedListAfterOption(opt) {
			items := m.getExpandedListItems()
			subIndent := "      "
			for j, item := range items {
				if lineIdx >= startLine && lineIdx < endLine {
					isSubFocused := j == m.subCursor
					isSelected := m.isExpandedItemSelected(j, item)
					checkbox := styles.RenderCheckbox(isSelected, isSubFocused)
					itemStyle := lipgloss.NewStyle().Foreground(styles.Text)
					if isSubFocused {
						itemStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
					}
					b.WriteString(subIndent + checkbox + " " + itemStyle.Render(item) + "\n")
				}
				lineIdx++
			}
		}
	}

	if showScrollIndicators && endLine < totalLines {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↓ %d more below", totalLines-endLine)) + "\n")
	}

	return b.String()
}
