// Features pipeline advanced configuration.
package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

func (m Model) renderFeaturesAdvancedConfig() string {
	var b strings.Builder

	b.WriteString(styles.RenderStepHeader("Advanced", m.contentWidth) + "\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)

	if m.useDefaultAdvanced {
		return m.renderDefaultConfigView("feature parameters")
	}

	if m.editingNumber {
		b.WriteString(infoStyle.Render("Enter value, Enter to confirm, Esc to cancel") + "\n")
	} else if m.editingText {
		b.WriteString(infoStyle.Render("Type text, Enter to confirm, Esc to cancel") + "\n")
	} else if m.expandedOption >= 0 {
		b.WriteString(infoStyle.Render("Space: select  Esc: close submenu") + "\n")
	} else {
		b.WriteString(infoStyle.Render("Space: toggle/expand  Enter: proceed") + "\n")
	}

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	// Prepare values for display
	peOrderVal := fmt.Sprintf("%d", m.complexityPEOrder)
	peDelayVal := fmt.Sprintf("%d", m.complexityPEDelay)
	sampEnOrderVal := fmt.Sprintf("%d", m.complexitySampEnOrder)
	sampEnRVal := fmt.Sprintf("%.2f", m.complexitySampEnR)
	mseScaleMinVal := fmt.Sprintf("%d", m.complexityMSEScaleMin)
	mseScaleMaxVal := fmt.Sprintf("%d", m.complexityMSEScaleMax)
	burstThreshVal := fmt.Sprintf("%.1f z", m.burstThresholdZ)
	burstMinDurVal := fmt.Sprintf("%d ms", m.burstMinDuration)
	erpBaselineVal := m.boolToOnOff(m.erpBaselineCorrection)
	erpAllowNoBaselineVal := m.boolToOnOff(m.erpAllowNoBaseline)
	erpComponentsVal := m.erpComponentsSpec
	if strings.TrimSpace(erpComponentsVal) == "" {
		erpComponentsVal = "(default)"
	}
	powerBaselineVal := []string{"logratio", "mean", "ratio", "zscore", "zlogratio"}[m.powerBaselineMode]
	powerRequireBaselineVal := m.boolToOnOff(m.powerRequireBaseline)
	spectralEdgeVal := fmt.Sprintf("%.0f%%", m.spectralEdgePercentile*100)
	spectralRatioPairsVal := m.spectralRatioPairsSpec
	if strings.TrimSpace(spectralRatioPairsVal) == "" {
		spectralRatioPairsVal = "(default)"
	}
	connOutputVal := []string{"full", "global_only"}[m.connOutputLevel]
	connGraphVal := m.boolToOnOff(m.connGraphMetrics)
	connGraphPropVal := fmt.Sprintf("%.2f", m.connGraphProp)
	connWindowLenVal := fmt.Sprintf("%.1f s", m.connWindowLen)
	connWindowStepVal := fmt.Sprintf("%.1f s", m.connWindowStep)
	connAecVal := []string{"orth", "none", "sym"}[m.connAECMode]
	pacPhaseVal := fmt.Sprintf("%.1f-%.1f Hz", m.pacPhaseMin, m.pacPhaseMax)
	pacAmpVal := fmt.Sprintf("%.1f-%.1f Hz", m.pacAmpMin, m.pacAmpMax)
	pacMethodVal := []string{"mvl", "kl", "tort", "ozkurt"}[m.pacMethod]
	pacMinEpochsVal := fmt.Sprintf("%d", m.pacMinEpochs)
	pacPairsVal := m.pacPairsSpec
	if strings.TrimSpace(pacPairsVal) == "" {
		pacPairsVal = "(default)"
	}
	burstBandsVal := m.burstBandsSpec
	if strings.TrimSpace(burstBandsVal) == "" {
		burstBandsVal = "(default)"
	}
	asymPairsVal := m.asymmetryChannelPairsSpec
	if strings.TrimSpace(asymPairsVal) == "" {
		asymPairsVal = "(default)"
	}
	aperiodicPeakZVal := fmt.Sprintf("%.1f", m.aperiodicPeakZ)
	aperiodicR2Val := fmt.Sprintf("%.2f", m.aperiodicMinR2)
	aperiodicPointsVal := fmt.Sprintf("%d", m.aperiodicMinPoints)
	minEpochsVal := fmt.Sprintf("%d", m.minEpochsForFeatures)

	// Input overrides
	if m.editingNumber {
		buffer := m.numberBuffer + "█"
		switch {
		case m.isCurrentlyEditing(optPEOrder):
			peOrderVal = buffer
		case m.isCurrentlyEditing(optPEDelay):
			peDelayVal = buffer
		case m.isCurrentlyEditing(optComplexitySampleEntropyOrder):
			sampEnOrderVal = buffer
		case m.isCurrentlyEditing(optComplexitySampleEntropyR):
			sampEnRVal = buffer
		case m.isCurrentlyEditing(optComplexityMSEScaleMin):
			mseScaleMinVal = buffer
		case m.isCurrentlyEditing(optComplexityMSEScaleMax):
			mseScaleMaxVal = buffer
		case m.isCurrentlyEditing(optBurstThreshold):
			burstThreshVal = buffer
		case m.isCurrentlyEditing(optBurstMinDuration):
			burstMinDurVal = buffer
		case m.isCurrentlyEditing(optConnGraphProp):
			connGraphPropVal = buffer
		case m.isCurrentlyEditing(optConnWindowLen):
			connWindowLenVal = buffer
		case m.isCurrentlyEditing(optConnWindowStep):
			connWindowStepVal = buffer
		case m.isCurrentlyEditing(optPACMinEpochs):
			pacMinEpochsVal = buffer
		case m.isCurrentlyEditing(optAperiodicPeakZ):
			aperiodicPeakZVal = buffer
		case m.isCurrentlyEditing(optAperiodicMinR2):
			aperiodicR2Val = buffer
		case m.isCurrentlyEditing(optAperiodicMinPoints):
			aperiodicPointsVal = buffer
		case m.isCurrentlyEditing(optMinEpochs):
			minEpochsVal = buffer
		}
	}
	if m.editingText {
		buffer := m.textBuffer + "█"
		switch m.editingTextField {
		case textFieldPACPairs:
			pacPairsVal = buffer
		case textFieldBurstBands:
			burstBandsVal = buffer
		case textFieldSpectralRatioPairs:
			spectralRatioPairsVal = buffer
		case textFieldAsymmetryChannelPairs:
			asymPairsVal = buffer
		case textFieldERPComponents:
			erpComponentsVal = buffer
		}
	}

	options := m.getFeaturesOptions()

	// Calculate total lines including expanded connectivity measures
	totalLines := len(options)
	if m.expandedOption == expandedConnectivityMeasures {
		totalLines += len(connectivityMeasures)
	}
	if m.expandedOption == expandedDirectedConnMeasures {
		totalLines += len(directedConnectivityMeasures)
	}
	if m.expandedOption == expandedFmriCondAColumn {
		totalLines += len(m.fmriDiscoveredColumns)
	}
	if m.expandedOption == expandedFmriCondAValue {
		totalLines += len(m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondAColumn))
	}
	if m.expandedOption == expandedFmriCondBColumn {
		totalLines += len(m.fmriDiscoveredColumns)
	}
	if m.expandedOption == expandedFmriCondBValue {
		totalLines += len(m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondBColumn))
	}
	if m.expandedOption == expandedSourceLocFmriScopeTrialTypes {
		totalLines += len(m.getExpandedListItems())
	}
	if m.expandedOption == expandedSourceLocFmriStimPhases {
		totalLines += len(m.getExpandedListItems())
	}
	if m.expandedOption == expandedItpcConditionColumn {
		totalLines += len(m.GetAvailableColumns())
	}
	if m.expandedOption == expandedItpcConditionValues {
		totalLines += len(m.GetDiscoveredColumnValues(m.itpcConditionColumn))
	}
	if m.expandedOption == expandedConnConditionColumn {
		totalLines += len(m.GetAvailableColumns())
	}
	if m.expandedOption == expandedConnConditionValues {
		totalLines += len(m.GetDiscoveredColumnValues(m.connConditionColumn))
	}

	effectiveHeight := m.height
	if effectiveHeight <= 0 {
		effectiveHeight = defaultTerminalHeight
	}

	startLine, endLine, showScrollIndicators := calculateScrollWindow(
		totalLines, m.advancedOffset, effectiveHeight, configOverhead)

	// Show scroll indicator for items above
	if showScrollIndicators && startLine > 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↑ %d more items above", startLine)) + "\n")
	}

	lineIdx := 0

	for i, opt := range options {
		isFocused := i == m.advancedCursor && m.expandedOption < 0
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
			if m.useDefaultAdvanced {
				value = "Using Defaults"
				hint = "Press Space to customize"
				expandIndicator = " " + styles.SelectedMark
			} else {
				value = "Custom"
				hint = "Press Space to use defaults"
				expandIndicator = " ▼"
			}

		// Section headers - styled as distinct groups with chevron indicators
		case optFeatGroupConnectivity:
			label = "▸ Connectivity"
			hint = "Space to toggle"
			if m.featGroupConnectivityExpanded {
				label = "▾ Connectivity"
				value = ""
				expandIndicator = ""
			} else {
				value = ""
				expandIndicator = ""
			}
			// Use distinct styling for section headers
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFeatGroupDirectedConnectivity:
			label = "▸ Directed Connectivity"
			hint = "Space to toggle"
			if m.featGroupDirectedConnExpanded {
				label = "▾ Directed Connectivity"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFeatGroupPAC:
			label = "▸ PAC / CFC"
			hint = "Space to toggle"
			if m.featGroupPACExpanded {
				label = "▾ PAC / CFC"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFeatGroupAperiodic:
			label = "▸ Aperiodic"
			hint = "Space to toggle"
			if m.featGroupAperiodicExpanded {
				label = "▾ Aperiodic"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFeatGroupComplexity:
			label = "▸ Complexity"
			hint = "Space to toggle"
			if m.featGroupComplexityExpanded {
				label = "▾ Complexity"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFeatGroupBursts:
			label = "▸ Bursts"
			hint = "Space to toggle"
			if m.featGroupBurstsExpanded {
				label = "▾ Bursts"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFeatGroupPower:
			label = "▸ Power"
			hint = "Space to toggle"
			if m.featGroupPowerExpanded {
				label = "▾ Power"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFeatGroupSpectral:
			label = "▸ Spectral"
			hint = "Space to toggle"
			if m.featGroupSpectralExpanded {
				label = "▾ Spectral"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFeatGroupERP:
			label = "▸ ERP"
			hint = "Space to toggle"
			if m.featGroupERPExpanded {
				label = "▾ ERP"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFeatGroupRatios:
			label = "▸ Ratios"
			hint = "Space to toggle"
			if m.featGroupRatiosExpanded {
				label = "▾ Ratios"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFeatGroupAsymmetry:
			label = "▸ Asymmetry"
			hint = "Space to toggle"
			if m.featGroupAsymmetryExpanded {
				label = "▾ Asymmetry"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFeatGroupSpatialTransform:
			label = "▸ Spatial Transform"
			hint = "Space to toggle"
			if m.featGroupSpatialTransformExpanded {
				label = "▾ Spatial Transform"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optSpatialTransform:
			label = "Transform"
			transforms := []string{"none", "CSD", "Laplacian"}
			value = transforms[m.spatialTransform]
			hint = "volume conduction reduction"
		case optSpatialTransformLambda2:
			label = "Lambda2"
			value = fmt.Sprintf("%.2e", m.spatialTransformLambda2)
			if m.editingNumber && m.isCurrentlyEditing(optSpatialTransformLambda2) {
				value = m.numberBuffer + "█"
			}
			hint = "regularization parameter"
		case optSpatialTransformStiffness:
			label = "Stiffness"
			value = fmt.Sprintf("%.1f", m.spatialTransformStiffness)
			if m.editingNumber && m.isCurrentlyEditing(optSpatialTransformStiffness) {
				value = m.numberBuffer + "█"
			}
			hint = "spline stiffness"
		case optFeatGroupTFR:
			label = "▸ Time-Frequency"
			hint = "Space to toggle"
			if m.featGroupTFRExpanded {
				label = "▾ Time-Frequency"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optTfrFreqMin:
			label = "Freq Min"
			value = fmt.Sprintf("%.1f Hz", m.tfrFreqMin)
			if m.editingNumber && m.isCurrentlyEditing(optTfrFreqMin) {
				value = m.numberBuffer + "█"
			}
			hint = "min TFR frequency"
		case optTfrFreqMax:
			label = "Freq Max"
			value = fmt.Sprintf("%.1f Hz", m.tfrFreqMax)
			if m.editingNumber && m.isCurrentlyEditing(optTfrFreqMax) {
				value = m.numberBuffer + "█"
			}
			hint = "max TFR frequency"
		case optTfrNFreqs:
			label = "N Frequencies"
			value = fmt.Sprintf("%d", m.tfrNFreqs)
			if m.editingNumber && m.isCurrentlyEditing(optTfrNFreqs) {
				value = m.numberBuffer + "█"
			}
			hint = "number of freq bins"
		case optTfrMinCycles:
			label = "Min Cycles"
			value = fmt.Sprintf("%.1f", m.tfrMinCycles)
			if m.editingNumber && m.isCurrentlyEditing(optTfrMinCycles) {
				value = m.numberBuffer + "█"
			}
			hint = "wavelet cycles floor"
		case optTfrNCyclesFactor:
			label = "Cycles Factor"
			value = fmt.Sprintf("%.1f", m.tfrNCyclesFactor)
			if m.editingNumber && m.isCurrentlyEditing(optTfrNCyclesFactor) {
				value = m.numberBuffer + "█"
			}
			hint = "cycles per Hz"
		case optTfrWorkers:
			label = "Workers"
			value = fmt.Sprintf("%d", m.tfrWorkers)
			if m.editingNumber && m.isCurrentlyEditing(optTfrWorkers) {
				value = m.numberBuffer + "█"
			}
			hint = "-1 = all cores"
		case optBandEnvelopePadSec:
			label = "Envelope pad (sec)"
			value = fmt.Sprintf("%.2f", m.bandEnvelopePadSec)
			if m.editingNumber && m.isCurrentlyEditing(optBandEnvelopePadSec) {
				value = m.numberBuffer + "█"
			}
			hint = "padding for Hilbert envelopes"
		case optBandEnvelopePadCycles:
			label = "Envelope pad (cycles)"
			value = fmt.Sprintf("%.1f", m.bandEnvelopePadCycles)
			if m.editingNumber && m.isCurrentlyEditing(optBandEnvelopePadCycles) {
				value = m.numberBuffer + "█"
			}
			hint = "padding scaled by fmin"
		case optIAFEnabled:
			label = "IAF enabled"
			value = m.boolToOnOff(m.iafEnabled)
			hint = "individualized alpha frequency"
		case optIAFAlphaWidthHz:
			label = "IAF alpha width"
			value = fmt.Sprintf("%.1f Hz", m.iafAlphaWidthHz)
			if m.editingNumber && m.isCurrentlyEditing(optIAFAlphaWidthHz) {
				value = m.numberBuffer + "█"
			}
			hint = "alpha band half-width"
		case optIAFSearchRangeMin:
			label = "IAF search min"
			value = fmt.Sprintf("%.1f Hz", m.iafSearchRangeMin)
			if m.editingNumber && m.isCurrentlyEditing(optIAFSearchRangeMin) {
				value = m.numberBuffer + "█"
			}
			hint = "search range lower bound"
		case optIAFSearchRangeMax:
			label = "IAF search max"
			value = fmt.Sprintf("%.1f Hz", m.iafSearchRangeMax)
			if m.editingNumber && m.isCurrentlyEditing(optIAFSearchRangeMax) {
				value = m.numberBuffer + "█"
			}
			hint = "search range upper bound"
		case optIAFMinProminence:
			label = "IAF prominence"
			value = fmt.Sprintf("%.3f", m.iafMinProminence)
			if m.editingNumber && m.isCurrentlyEditing(optIAFMinProminence) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum PSD peak prominence"
		case optIAFRois:
			label = "IAF ROIs"
			rois := splitCSVList(m.iafRoisSpec)
			if len(rois) == 0 {
				value = "(none selected)"
			} else {
				value = strings.Join(rois, ",")
			}
			hint = fmt.Sprintf("Space to select \u00b7 %d ROI options", len(m.rois))
		case optIAFMinCyclesAtFmin:
			label = "IAF min cycles"
			value = fmt.Sprintf("%.1f", m.iafMinCyclesAtFmin)
			if m.editingNumber && m.isCurrentlyEditing(optIAFMinCyclesAtFmin) {
				value = m.numberBuffer + "█"
			}
			hint = "cycles at iaf search fmin"
		case optIAFMinBaselineSec:
			label = "IAF min baseline"
			value = fmt.Sprintf("%.2f s", m.iafMinBaselineSec)
			if m.editingNumber && m.isCurrentlyEditing(optIAFMinBaselineSec) {
				value = m.numberBuffer + "█"
			}
			hint = "additional baseline duration"
		case optIAFAllowFullFallback:
			label = "Allow full fallback"
			value = m.boolToOnOff(m.iafAllowFullFallback)
			hint = "use full segment if baseline missing"
		case optIAFAllowAllChannelsFallback:
			label = "Allow channels fallback"
			value = m.boolToOnOff(m.iafAllowAllChannelsFallback)
			hint = "use all channels if ROIs missing"
		case optFeatGroupStorage:
			label = "▸ Storage"
			hint = "Space to toggle"
			if m.featGroupStorageExpanded {
				label = "▾ Storage"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optSaveSubjectLevelFeatures:
			label = "Save Subject-Level"
			value = m.boolToOnOff(m.saveSubjectLevelFeatures)
			hint = "Space to toggle"
		case optFeatAlsoSaveCsv:
			label = "Also Save CSV"
			value = m.boolToOnOff(m.featAlsoSaveCsv)
			hint = "save feature tables as both parquet and CSV"
		case optFeatGroupExecution:
			label = "▸ Execution"
			hint = "Space to toggle"
			if m.featGroupExecutionExpanded {
				label = "▾ Execution"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFeatGroupSourceLoc:
			label = "▸ Source Localization"
			hint = "Space to toggle"
			if m.featGroupSourceLocExpanded {
				label = "▾ Source Localization"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		// Connectivity settings
		case optConnectivity:
			label = "Measures"
			value = m.selectedConnectivityDisplay()
			hint = "Select measures"
			if m.expandedOption != expandedConnectivityMeasures {
				expandIndicator = " [+]"
			} else {
				expandIndicator = " [-]"
			}
		case optConnOutputLevel:
			label = "Output Level"
			value = connOutputVal
			hint = "full / global_only"
		case optConnGranularity:
			label = "Granularity"
			granularities := []string{"trial", "condition", "subject"}
			value = granularities[m.connGranularity]
			hint = "trial / condition / subject"
		case optConnConditionColumn:
			label = "Condition Column"
			val := m.connConditionColumn
			if val == "" {
				val = "(select column)"
			}
			if m.editingText && m.editingTextField == textFieldConnConditionColumn {
				val = m.textBuffer + "█"
			}
			value = val
			expandIndicatorHint := ""
			if len(m.GetAvailableColumns()) > 0 {
				expandIndicatorHint = fmt.Sprintf(" · %d columns available", len(m.GetAvailableColumns()))
			}
			hint = "Space to select" + expandIndicatorHint
			if m.expandedOption == expandedConnConditionColumn {
				expandIndicator = " [-]"
			} else {
				expandIndicator = " [+]"
			}
		case optConnConditionValues:
			label = "Condition Values"
			if m.connConditionColumn == "" {
				value = "(select column first)"
				hint = "requires column selection"
			} else {
				val := m.connConditionValues
				if val == "" {
					val = "(select values)"
				}
				if m.editingText && m.editingTextField == textFieldConnConditionValues {
					val = m.textBuffer + "█"
				}
				value = val
				expandIndicatorHint := ""
				if vals := m.GetDiscoveredColumnValues(m.connConditionColumn); len(vals) > 0 {
					expandIndicatorHint = fmt.Sprintf(" · %d values in %s", len(vals), m.connConditionColumn)
				}
				hint = "Space to select (others excluded)" + expandIndicatorHint
				if m.expandedOption == expandedConnConditionValues {
					expandIndicator = " [-]"
				} else {
					expandIndicator = " [+]"
				}
			}
		case optConnPhaseEstimator:
			label = "Phase estimator"
			estimators := []string{"within_epoch", "across_epochs"}
			value = estimators[m.connPhaseEstimator]
			hint = "within_epoch / across_epochs"
		case optConnMinEpochsPerGroup:
			label = "Min epochs/group"
			value = fmt.Sprintf("%d", m.connMinEpochsPerGroup)
			if m.editingNumber && m.isCurrentlyEditing(optConnMinEpochsPerGroup) {
				value = m.numberBuffer + "█"
			}
			hint = "for condition/subject granularity"
		case optConnMinCyclesPerBand:
			label = "Min cycles/band"
			value = fmt.Sprintf("%.1f", m.connMinCyclesPerBand)
			if m.editingNumber && m.isCurrentlyEditing(optConnMinCyclesPerBand) {
				value = m.numberBuffer + "█"
			}
			hint = "recommended cycles at band fmin"
		case optConnMinSegmentSec:
			label = "Min segment (s)"
			value = fmt.Sprintf("%.1f", m.connMinSegmentSec)
			if m.editingNumber && m.isCurrentlyEditing(optConnMinSegmentSec) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum duration"
		case optConnWarnNoSpatialTransform:
			label = "Warn (no transform)"
			value = m.boolToOnOff(m.connWarnNoSpatialTransform)
			hint = "warn when phase measures lack CSD/Laplacian"
		case optConnGraphMetrics:
			label = "Graph Metrics"
			value = connGraphVal
			hint = "toggles metric computation"
		case optConnGraphProp:
			label = "Graph Threshold"
			value = connGraphPropVal
			hint = "top edges density"
		case optConnWindowLen:
			label = "Window length"
			value = connWindowLenVal
			hint = "seconds per slice"
		case optConnWindowStep:
			label = "Window step"
			value = connWindowStepVal
			hint = "overlap amount"
		case optConnAECMode:
			label = "AEC Mode"
			value = connAecVal
			hint = "orth/none/sym"
		case optConnAECOutput:
			label = "AEC Output"
			switch m.connAECOutput {
			case 1:
				value = "z"
			case 2:
				value = "r+z"
			default:
				value = "r"
			}
			hint = "r (raw), z (Fisher-z), or both"
		case optConnForceWithinEpochML:
			label = "Force within_epoch"
			value = m.boolToOnOff(m.connForceWithinEpochML)
			hint = "CV/ML leakage safety"
		case optConnDynamicEnabled:
			label = "Dynamic features"
			value = m.boolToOnOff(m.connDynamicEnabled)
			hint = "enable sliding-window connectivity dynamics"
		case optConnDynamicMeasures:
			label = "Dynamic measures"
			switch m.connDynamicMeasures {
			case 1:
				value = "wpli"
			case 2:
				value = "aec"
			default:
				value = "wpli+aec"
			}
			hint = "measures for dynamic summaries"
		case optConnDynamicAutocorrLag:
			label = "Dynamic AC lag"
			value = fmt.Sprintf("%d", m.connDynamicAutocorrLag)
			if m.editingNumber && m.isCurrentlyEditing(optConnDynamicAutocorrLag) {
				value = m.numberBuffer + "█"
			}
			hint = "autocorr lag in windows"
		case optConnDynamicMinWindows:
			label = "Dynamic min windows"
			value = fmt.Sprintf("%d", m.connDynamicMinWindows)
			if m.editingNumber && m.isCurrentlyEditing(optConnDynamicMinWindows) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum windows for dynamic features"
		case optConnDynamicIncludeROIPairs:
			label = "Dynamic ROI pairs"
			value = m.boolToOnOff(m.connDynamicIncludeROIPairs)
			hint = "emit ROI-pair dynamic summaries"
		case optConnDynamicStateEnabled:
			label = "Dynamic states"
			value = m.boolToOnOff(m.connDynamicStateEnabled)
			hint = "k-means state transitions"
		case optConnDynamicStateNStates:
			label = "State count"
			value = fmt.Sprintf("%d", m.connDynamicStateNStates)
			if m.editingNumber && m.isCurrentlyEditing(optConnDynamicStateNStates) {
				value = m.numberBuffer + "█"
			}
			hint = "number of dynamic connectivity states"
		case optConnDynamicStateMinWindows:
			label = "State min windows"
			value = fmt.Sprintf("%d", m.connDynamicStateMinWindows)
			if m.editingNumber && m.isCurrentlyEditing(optConnDynamicStateMinWindows) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum windows for state metrics"
		case optConnDynamicStateRandomSeed:
			label = "State random seed"
			if m.connDynamicStateRandomSeed < 0 {
				value = "(auto)"
			} else {
				value = fmt.Sprintf("%d", m.connDynamicStateRandomSeed)
			}
			if m.editingNumber && m.isCurrentlyEditing(optConnDynamicStateRandomSeed) {
				value = m.numberBuffer + "█"
			}
			hint = "-1 auto, >=0 fixed seed"

		// Directed Connectivity (PSI, DTF, PDC)
		case optDirectedConnMeasures:
			label = "Measures"
			value = m.selectedDirectedConnectivityDisplay()
			hint = "Select directed measures"
			if m.expandedOption != expandedDirectedConnMeasures {
				expandIndicator = " [+]"
			} else {
				expandIndicator = " [-]"
			}
		case optDirectedConnOutputLevel:
			label = "Output Level"
			outputLevels := []string{"full", "global_only"}
			value = outputLevels[m.directedConnOutputLevel]
			hint = "full / global_only"
		case optDirectedConnMvarOrder:
			label = "MVAR Order"
			value = fmt.Sprintf("%d", m.directedConnMvarOrder)
			if m.editingNumber && m.isCurrentlyEditing(optDirectedConnMvarOrder) {
				value = m.numberBuffer + "█"
			}
			hint = "model order for DTF/PDC"
		case optDirectedConnNFreqs:
			label = "N Freqs"
			value = fmt.Sprintf("%d", m.directedConnNFreqs)
			if m.editingNumber && m.isCurrentlyEditing(optDirectedConnNFreqs) {
				value = m.numberBuffer + "█"
			}
			hint = "frequency bins"
		case optDirectedConnMinSegSamples:
			label = "Min Seg Samples"
			value = fmt.Sprintf("%d", m.directedConnMinSegSamples)
			if m.editingNumber && m.isCurrentlyEditing(optDirectedConnMinSegSamples) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum samples per segment"

		// Source Localization (LCMV, eLORETA)
		case optSourceLocMode:
			label = "Mode"
			modes := []string{"EEG-only (template)", "fMRI-informed"}
			value = modes[m.sourceLocMode]
			hint = "template or subject-specific (requires fmri-prep)"
		case optSourceLocMethod:
			label = "Method"
			methods := []string{"LCMV", "eLORETA"}
			value = methods[m.sourceLocMethod]
			hint = "beamformer type"
		case optSourceLocSpacing:
			label = "Spacing"
			spacings := []string{"oct5", "oct6", "ico4", "ico5"}
			value = spacings[m.sourceLocSpacing]
			hint = "source space resolution"
		case optSourceLocParc:
			label = "Parcellation"
			parcs := []string{"aparc", "aparc.a2009s", "HCPMMP1"}
			value = parcs[m.sourceLocParc]
			hint = "cortical atlas"
		case optSourceLocReg:
			label = "Regularization"
			value = fmt.Sprintf("%.3f", m.sourceLocReg)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocReg) {
				value = m.numberBuffer + "█"
			}
			hint = "LCMV regularization"
		case optSourceLocSnr:
			label = "SNR"
			value = fmt.Sprintf("%.1f", m.sourceLocSnr)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocSnr) {
				value = m.numberBuffer + "█"
			}
			hint = "eLORETA signal-to-noise"
		case optSourceLocLoose:
			label = "Loose"
			value = fmt.Sprintf("%.2f", m.sourceLocLoose)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocLoose) {
				value = m.numberBuffer + "█"
			}
			hint = "eLORETA loose constraint"
		case optSourceLocDepth:
			label = "Depth"
			value = fmt.Sprintf("%.2f", m.sourceLocDepth)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocDepth) {
				value = m.numberBuffer + "█"
			}
			hint = "eLORETA depth weighting"
		case optSourceLocConnMethod:
			label = "Connectivity"
			connMethods := []string{"AEC", "wPLI", "PLV"}
			value = connMethods[m.sourceLocConnMethod]
			hint = "source-space connectivity"
		case optSourceLocSubject:
			label = "FS Subject"
			if strings.TrimSpace(m.sourceLocSubject) == "" {
				value = "(auto)"
			} else {
				value = m.sourceLocSubject
			}
			if m.editingText && m.editingTextField == textFieldSourceLocSubject {
				value = m.textBuffer + "█"
			}
			hint = "FreeSurfer subject name"
		case optSourceLocTrans:
			label = "Coreg trans"
			if strings.TrimSpace(m.sourceLocTrans) == "" {
				value = "(unset)"
			} else {
				value = m.sourceLocTrans
			}
			hint = "EEG↔MRI transform .fif"
		case optSourceLocBem:
			label = "BEM sol"
			if strings.TrimSpace(m.sourceLocBem) == "" {
				value = "(unset)"
			} else {
				value = m.sourceLocBem
			}
			hint = "BEM solution .fif"
		case optSourceLocMindistMm:
			label = "Mindist (mm)"
			value = fmt.Sprintf("%.1f", m.sourceLocMindistMm)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocMindistMm) {
				value = m.numberBuffer + "█"
			}
			hint = "min distance to inner skull"
		case optSourceLocFmriEnabled:
			label = "fMRI Prior"
			if m.sourceLocFmriEnabled {
				value = "on"
			} else {
				value = "off"
			}
			hint = "restrict volume sources using fMRI"
		case optSourceLocFmriStatsMap:
			label = "fMRI Stats Map"
			if strings.TrimSpace(m.sourceLocFmriStatsMap) == "" {
				value = "(unset)"
			} else {
				value = m.sourceLocFmriStatsMap
			}
			hint = "NIfTI map in FS MRI space"
		case optSourceLocFmriProvenance:
			label = "fMRI Provenance"
			prov := []string{"independent", "same_dataset"}
			if m.sourceLocFmriProvenance >= 0 && m.sourceLocFmriProvenance < len(prov) {
				value = prov[m.sourceLocFmriProvenance]
			} else {
				value = "independent"
			}
			hint = "independent (recommended) vs same_dataset"
		case optSourceLocFmriRequireProvenance:
			label = "Require provenance"
			if m.sourceLocFmriRequireProv {
				value = "on"
			} else {
				value = "off"
			}
			hint = "error if provenance unknown"
		case optSourceLocFmriThreshold:
			label = "fMRI Threshold"
			value = fmt.Sprintf("%.2f", m.sourceLocFmriThreshold)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriThreshold) {
				value = m.numberBuffer + "█"
			}
			hint = "e.g., z>=3.10"
		case optSourceLocFmriTail:
			label = "fMRI Tail"
			tails := []string{"pos", "abs"}
			value = tails[m.sourceLocFmriTail]
			hint = "pos or abs"
		case optSourceLocFmriMinClusterMM3:
			label = "fMRI Min Volume (mm^3)"
			value = fmt.Sprintf("%.0f", m.sourceLocFmriMinClusterMM3)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriMinClusterMM3) {
				value = m.numberBuffer + "█"
			}
			hint = "preferred; stable across voxel sizes (0 disables)"
		case optSourceLocFmriMinClusterVox:
			label = "fMRI Min Voxels"
			value = fmt.Sprintf("%d", m.sourceLocFmriMinClusterVox)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriMinClusterVox) {
				value = m.numberBuffer + "█"
			}
			hint = "legacy; used only when Min Volume=0"
		case optSourceLocFmriMaxClusters:
			label = "fMRI Max Clusters"
			value = fmt.Sprintf("%d", m.sourceLocFmriMaxClusters)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriMaxClusters) {
				value = m.numberBuffer + "█"
			}
			hint = "limit ROI clusters"
		case optSourceLocFmriMaxVoxPerClus:
			label = "fMRI Max Vox/Cluster"
			value = fmt.Sprintf("%d", m.sourceLocFmriMaxVoxPerClus)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriMaxVoxPerClus) {
				value = m.numberBuffer + "█"
			}
			hint = "subsample per cluster"
		case optSourceLocFmriMaxTotalVox:
			label = "fMRI Max Total Vox"
			value = fmt.Sprintf("%d", m.sourceLocFmriMaxTotalVox)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriMaxTotalVox) {
				value = m.numberBuffer + "█"
			}
			hint = "subsample total"
		case optSourceLocFmriRandomSeed:
			label = "fMRI Random Seed"
			value = fmt.Sprintf("%d", m.sourceLocFmriRandomSeed)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriRandomSeed) {
				value = m.numberBuffer + "█"
			}
			hint = "0 = nondeterministic"

		// BEM/Trans generation options (Docker-based)
		case optSourceLocCreateTrans:
			label = "Create Trans"
			if m.sourceLocCreateTrans {
				value = "on"
			} else {
				value = "off"
			}
			hint = "auto-generate via Docker (requires Docker)"
		case optSourceLocCreateBemModel:
			label = "Create BEM Model"
			if m.sourceLocCreateBemModel {
				value = "on"
			} else {
				value = "off"
			}
			hint = "auto-generate via Docker (requires Docker)"
		case optSourceLocCreateBemSolution:
			label = "Create BEM Solution"
			if m.sourceLocCreateBemSolution {
				value = "on"
			} else {
				value = "off"
			}
			hint = "auto-generate via Docker (requires Docker)"

		// fMRI GLM Contrast Builder options
		case optSourceLocFmriContrastEnabled:
			label = "Build Contrast"
			if m.sourceLocFmriContrastEnabled {
				value = "on"
			} else {
				value = "off"
			}
			hint = "build from BOLD vs. load pre-computed"
		case optSourceLocFmriContrastType:
			label = "Contrast Type"
			contrastTypes := []string{"t-test", "paired t-test", "F-test", "custom formula"}
			value = contrastTypes[m.sourceLocFmriContrastType]
			hint = "statistical test type"
		case optSourceLocFmriCondAColumn:
			label = "Cond A Column"
			val := m.sourceLocFmriCondAColumn
			if val == "" {
				val = "(select column)"
			}
			if m.editingText && m.editingTextField == textFieldSourceLocFmriCondAColumn {
				val = m.textBuffer + "█"
			}
			value = val
			expandIndicatorHint := ""
			if len(m.fmriDiscoveredColumns) > 0 {
				expandIndicatorHint = fmt.Sprintf(" · %d columns available", len(m.fmriDiscoveredColumns))
			}
			hint = "Space to select" + expandIndicatorHint
			if m.expandedOption == expandedFmriCondAColumn {
				expandIndicator = " [-]"
			} else {
				expandIndicator = " [+]"
			}
		case optSourceLocFmriCondAValue:
			label = "Cond A Value"
			if m.sourceLocFmriCondAColumn == "" {
				value = "(select column first)"
				hint = "requires column selection"
			} else {
				val := m.sourceLocFmriCondAValue
				if val == "" {
					val = "(select value)"
				}
				if m.editingText && m.editingTextField == textFieldSourceLocFmriCondAValue {
					val = m.textBuffer + "█"
				}
				value = val
				expandIndicatorHint := ""
				if vals := m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondAColumn); len(vals) > 0 {
					expandIndicatorHint = fmt.Sprintf(" · %d values in %s", len(vals), m.sourceLocFmriCondAColumn)
				}
				hint = "Space to select" + expandIndicatorHint
				if m.expandedOption == expandedFmriCondAValue {
					expandIndicator = " [-]"
				} else {
					expandIndicator = " [+]"
				}
			}
		case optSourceLocFmriCondBColumn:
			label = "Cond B Column"
			val := m.sourceLocFmriCondBColumn
			if val == "" {
				val = "(select column)"
			}
			if m.editingText && m.editingTextField == textFieldSourceLocFmriCondBColumn {
				val = m.textBuffer + "█"
			}
			value = val
			expandIndicatorHint := ""
			if len(m.fmriDiscoveredColumns) > 0 {
				expandIndicatorHint = fmt.Sprintf(" · %d columns available", len(m.fmriDiscoveredColumns))
			}
			hint = "Space to select" + expandIndicatorHint
			if m.expandedOption == expandedFmriCondBColumn {
				expandIndicator = " [-]"
			} else {
				expandIndicator = " [+]"
			}
		case optSourceLocFmriCondBValue:
			label = "Cond B Value"
			if m.sourceLocFmriCondBColumn == "" {
				value = "(select column first)"
				hint = "requires column selection"
			} else {
				val := m.sourceLocFmriCondBValue
				if val == "" {
					val = "(select value)"
				}
				if m.editingText && m.editingTextField == textFieldSourceLocFmriCondBValue {
					val = m.textBuffer + "█"
				}
				value = val
				expandIndicatorHint := ""
				if vals := m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondBColumn); len(vals) > 0 {
					expandIndicatorHint = fmt.Sprintf(" · %d values in %s", len(vals), m.sourceLocFmriCondBColumn)
				}
				hint = "Space to select" + expandIndicatorHint
				if m.expandedOption == expandedFmriCondBValue {
					expandIndicator = " [-]"
				} else {
					expandIndicator = " [+]"
				}
			}
		case optSourceLocFmriContrastFormula:
			label = "Formula"
			if strings.TrimSpace(m.sourceLocFmriContrastFormula) == "" {
				value = "(e.g., pain_high - pain_low)"
			} else {
				value = m.sourceLocFmriContrastFormula
			}
			if m.editingText && m.editingTextField == textFieldSourceLocFmriContrastFormula {
				value = m.textBuffer + "█"
			}
			hint = "custom contrast formula"
		case optSourceLocFmriContrastName:
			label = "Contrast Name"
			if strings.TrimSpace(m.sourceLocFmriContrastName) == "" {
				value = "pain_vs_baseline"
			} else {
				value = m.sourceLocFmriContrastName
			}
			if m.editingText && m.editingTextField == textFieldSourceLocFmriContrastName {
				value = m.textBuffer + "█"
			}
			hint = "output name for contrast"
		case optSourceLocFmriRunsToInclude:
			label = "Runs"
			if strings.TrimSpace(m.sourceLocFmriRunsToInclude) == "" {
				value = "(auto-detect)"
			} else {
				value = m.sourceLocFmriRunsToInclude
			}
			if m.editingText && m.editingTextField == textFieldSourceLocFmriRunsToInclude {
				value = m.textBuffer + "█"
			}
			hint = "comma-separated run numbers"
		case optSourceLocFmriAutoDetectRuns:
			label = "Auto-detect Runs"
			if m.sourceLocFmriAutoDetectRuns {
				value = "on"
			} else {
				value = "off"
			}
			hint = "scan BIDS for BOLD runs"
		case optSourceLocFmriHrfModel:
			label = "HRF Model"
			hrfModels := []string{"SPM", "FLOBS", "FIR"}
			value = hrfModels[m.sourceLocFmriHrfModel]
			hint = "hemodynamic response function"
		case optSourceLocFmriDriftModel:
			label = "Drift Model"
			driftModels := []string{"none", "cosine", "polynomial"}
			value = driftModels[m.sourceLocFmriDriftModel]
			hint = "low-frequency drift removal"
		case optSourceLocFmriHighPassHz:
			label = "High-pass (Hz)"
			value = fmt.Sprintf("%.4f", m.sourceLocFmriHighPassHz)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriHighPassHz) {
				value = m.numberBuffer + "█"
			}
			hint = "e.g., 0.008 = 128s period"
		case optSourceLocFmriLowPassHz:
			label = "Low-pass (Hz)"
			if m.sourceLocFmriLowPassHz <= 0 {
				value = "(disabled)"
			} else {
				value = fmt.Sprintf("%.2f", m.sourceLocFmriLowPassHz)
			}
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriLowPassHz) {
				value = m.numberBuffer + "█"
			}
			hint = "optional; generally avoid for task GLM"
		case optSourceLocFmriConditionScopeTrialTypes:
			label = "Condition trial_type Scope"
			val := strings.TrimSpace(m.sourceLocFmriConditionScopeTrialTypes)
			if val == "" {
				val = "(none)"
			}
			if m.editingText && m.editingTextField == textFieldSourceLocFmriConditionScopeTrialTypes {
				val = m.textBuffer + "█"
			}
			value = val
			expandIndicatorHint := ""
			if vals := m.GetFmriDiscoveredColumnValues("trial_type"); len(vals) > 0 {
				expandIndicatorHint = fmt.Sprintf(" · %d values in trial_type", len(vals))
			}
			hint = "Space to select" + expandIndicatorHint
			if m.expandedOption == expandedSourceLocFmriScopeTrialTypes {
				expandIndicator = " [-]"
			} else {
				expandIndicator = " [+]"
			}
		case optSourceLocFmriStimPhasesToModel:
			label = "Stim Phase Scope"
			val := strings.TrimSpace(m.sourceLocFmriStimPhasesToModel)
			if val == "" {
				val = "(none)"
			}
			if m.editingText && m.editingTextField == textFieldSourceLocFmriStimPhasesToModel {
				val = m.textBuffer + "█"
			}
			value = val
			expandIndicatorHint := ""
			if vals := m.GetFmriDiscoveredColumnValues("stim_phase"); len(vals) > 0 {
				expandIndicatorHint = fmt.Sprintf(" · %d values in stim_phase", len(vals))
			}
			hint = "Space to select" + expandIndicatorHint
			if m.expandedOption == expandedSourceLocFmriStimPhases {
				expandIndicator = " [-]"
			} else {
				expandIndicator = " [+]"
			}
		case optSourceLocFmriClusterCorrection:
			label = "Cluster Correction"
			if m.sourceLocFmriClusterCorrection {
				value = "on"
			} else {
				value = "off"
			}
			hint = "cluster-extent heuristic (not FWE)"
		case optSourceLocFmriClusterPThreshold:
			label = "Cluster p-threshold"
			value = fmt.Sprintf("%.4f", m.sourceLocFmriClusterPThreshold)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriClusterPThreshold) {
				value = m.numberBuffer + "█"
			}
			hint = "cluster-forming threshold"
		case optSourceLocFmriOutputType:
			label = "Output Type"
			outputTypes := []string{"z-score", "t-stat", "cope", "beta"}
			value = outputTypes[m.sourceLocFmriOutputType]
			hint = "statistical map type"
		case optSourceLocFmriResampleToFS:
			label = "Resample to FS"
			if m.sourceLocFmriResampleToFS {
				value = "on"
			} else {
				value = "off"
			}
			hint = "auto-resample to FreeSurfer space"
		case optSourceLocFmriWindowAName:
			label = "Window A Name"
			if m.sourceLocFmriWindowAName == "" {
				value = "(not set)"
			} else {
				value = m.sourceLocFmriWindowAName
			}
			if m.editingText && m.editingTextField == textFieldSourceLocFmriWindowAName {
				value = m.textBuffer + "█"
			}
			hint = "e.g., plateau"
		case optSourceLocFmriWindowATmin:
			label = "Window A Tmin"
			value = fmt.Sprintf("%.2f s", m.sourceLocFmriWindowATmin)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriWindowATmin) {
				value = m.numberBuffer + "█"
			}
			hint = "start time in seconds"
		case optSourceLocFmriWindowATmax:
			label = "Window A Tmax"
			value = fmt.Sprintf("%.2f s", m.sourceLocFmriWindowATmax)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriWindowATmax) {
				value = m.numberBuffer + "█"
			}
			hint = "end time in seconds"
		case optSourceLocFmriWindowBName:
			label = "Window B Name"
			if m.sourceLocFmriWindowBName == "" {
				value = "(not set)"
			} else {
				value = m.sourceLocFmriWindowBName
			}
			if m.editingText && m.editingTextField == textFieldSourceLocFmriWindowBName {
				value = m.textBuffer + "█"
			}
			hint = "e.g., baseline (optional)"
		case optSourceLocFmriWindowBTmin:
			label = "Window B Tmin"
			value = fmt.Sprintf("%.2f s", m.sourceLocFmriWindowBTmin)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriWindowBTmin) {
				value = m.numberBuffer + "█"
			}
			hint = "start time in seconds"
		case optSourceLocFmriWindowBTmax:
			label = "Window B Tmax"
			value = fmt.Sprintf("%.2f s", m.sourceLocFmriWindowBTmax)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriWindowBTmax) {
				value = m.numberBuffer + "█"
			}
			hint = "end time in seconds"

		// ITPC options
		case optFeatGroupITPC:
			label = "▸ ITPC"
			hint = "Space to toggle"
			if m.featGroupITPCExpanded {
				label = "▾ ITPC"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optItpcMethod:
			label = "ITPC Method"
			itpcMethods := []string{"global", "fold_global", "loo", "condition"}
			value = itpcMethods[m.itpcMethod]
			hint = "global/fold_global/loo/condition"
		case optItpcAllowUnsafeLoo:
			label = "Allow unsafe LOO"
			value = m.boolToOnOff(m.itpcAllowUnsafeLoo)
			hint = "unsafe unless computed within CV folds"
		case optItpcBaselineCorrection:
			label = "Baseline correction"
			modes := []string{"none", "subtract"}
			value = modes[m.itpcBaselineCorrection]
			hint = "ITPC baseline correction"
		case optItpcConditionColumn:
			label = "Condition Column"
			val := m.itpcConditionColumn
			if val == "" {
				val = "(select column)"
			}
			if m.editingText && m.editingTextField == textFieldItpcConditionColumn {
				val = m.textBuffer + "█"
			}
			value = val
			expandIndicatorHint := ""
			if len(m.GetAvailableColumns()) > 0 {
				expandIndicatorHint = fmt.Sprintf(" · %d columns available", len(m.GetAvailableColumns()))
			}
			hint = "Space to select" + expandIndicatorHint
			if m.expandedOption == expandedItpcConditionColumn {
				expandIndicator = " [-]"
			} else {
				expandIndicator = " [+]"
			}
		case optItpcConditionValues:
			label = "Condition Values"
			if m.itpcConditionColumn == "" {
				value = "(select column first)"
				hint = "requires column selection"
			} else {
				val := m.itpcConditionValues
				if val == "" {
					val = "(select values)"
				}
				if m.editingText && m.editingTextField == textFieldItpcConditionValues {
					val = m.textBuffer + "█"
				}
				value = val
				expandIndicatorHint := ""
				if vals := m.GetDiscoveredColumnValues(m.itpcConditionColumn); len(vals) > 0 {
					expandIndicatorHint = fmt.Sprintf(" · %d values in %s", len(vals), m.itpcConditionColumn)
				}
				hint = "Space to select (others excluded)" + expandIndicatorHint
				if m.expandedOption == expandedItpcConditionValues {
					expandIndicator = " [-]"
				} else {
					expandIndicator = " [+]"
				}
			}
		case optItpcMinTrialsPerCondition:
			label = "Min Trials/Condition"
			value = fmt.Sprintf("%d", m.itpcMinTrialsPerCondition)
			if m.editingNumber && m.isCurrentlyEditing(optItpcMinTrialsPerCondition) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum trials per condition"
		case optItpcNJobs:
			label = "Parallel Jobs"
			value = fmt.Sprintf("%d", m.itpcNJobs)
			if m.editingNumber && m.isCurrentlyEditing(optItpcNJobs) {
				value = m.numberBuffer + "█"
			}
			hint = "-1 = all CPUs"

		// PAC
		case optPACPhaseRange:
			label = "Phase range"
			value = pacPhaseVal
			hint = "frequencies for phase"
		case optPACAmpRange:
			label = "Amp range"
			value = pacAmpVal
			hint = "frequencies for amplitude"
		case optPACMethod:
			label = "PAC Method"
			value = pacMethodVal
			hint = "algorithm type"
		case optPACMinEpochs:
			label = "Min Epochs"
			value = pacMinEpochsVal
			hint = "minimum required"
		case optPACPairs:
			label = "Band pairs"
			value = pacPairsVal
			hint = "e.g. theta:gamma,alpha:gamma"
		case optPACSource:
			label = "Source"
			sources := []string{"precomputed", "tfr"}
			value = sources[m.pacSource]
			hint = "Hilbert vs wavelet"
		case optPACNormalize:
			label = "Normalize"
			value = m.boolToOnOff(m.pacNormalize)
			hint = "normalize PAC values"
		case optPACNSurrogates:
			label = "N Surrogates"
			value = fmt.Sprintf("%d", m.pacNSurrogates)
			if m.editingNumber && m.isCurrentlyEditing(optPACNSurrogates) {
				value = m.numberBuffer + "█"
			}
			hint = "0=none, >0 for z-scores"
		case optPACAllowHarmonicOverlap:
			label = "Allow Harmonics"
			value = m.boolToOnOff(m.pacAllowHarmonicOvrlap)
			hint = "allow harmonic overlap"
		case optPACMaxHarmonic:
			label = "Max Harmonic"
			value = fmt.Sprintf("%d", m.pacMaxHarmonic)
			if m.editingNumber && m.isCurrentlyEditing(optPACMaxHarmonic) {
				value = m.numberBuffer + "█"
			}
			hint = "upper harmonic to check"
		case optPACHarmonicToleranceHz:
			label = "Harmonic Tol Hz"
			value = fmt.Sprintf("%.1f", m.pacHarmonicToleranceHz)
			if m.editingNumber && m.isCurrentlyEditing(optPACHarmonicToleranceHz) {
				value = m.numberBuffer + "█"
			}
			hint = "tolerance for overlap check"
		case optPACRandomSeed:
			label = "Random Seed"
			value = fmt.Sprintf("%d", m.pacRandomSeed)
			if m.editingNumber && m.isCurrentlyEditing(optPACRandomSeed) {
				value = m.numberBuffer + "█"
			}
			hint = "seed for surrogate testing"
		case optPACComputeWaveformQC:
			label = "Waveform QC"
			value = m.boolToOnOff(m.pacComputeWaveformQC)
			hint = "compute waveform quality"
		case optPACWaveformOffsetMs:
			label = "Waveform Offset"
			value = fmt.Sprintf("%.1f ms", m.pacWaveformOffsetMs)
			if m.editingNumber && m.isCurrentlyEditing(optPACWaveformOffsetMs) {
				value = m.numberBuffer + "█"
			}
			hint = "offset in milliseconds"

		// Aperiodic
		case optAperiodicModel:
			label = "Model"
			models := []string{"fixed", "knee"}
			value = models[m.aperiodicModel]
			hint = "aperiodic model type"
		case optAperiodicPsdMethod:
			label = "PSD method"
			methods := []string{"multitaper", "welch"}
			value = methods[m.aperiodicPsdMethod]
			hint = "PSD estimation method"
		case optAperiodicFmin:
			aperiodicFminVal := fmt.Sprintf("%.1f", m.aperiodicFmin)
			if m.editingNumber && m.isCurrentlyEditing(optAperiodicFmin) {
				aperiodicFminVal = m.numberBuffer + "█"
			}
			label = "Fit range (min)"
			value = aperiodicFminVal
			hint = "minimum frequency (Hz)"
		case optAperiodicFmax:
			aperiodicFmaxVal := fmt.Sprintf("%.1f", m.aperiodicFmax)
			if m.editingNumber && m.isCurrentlyEditing(optAperiodicFmax) {
				aperiodicFmaxVal = m.numberBuffer + "█"
			}
			label = "Fit range (max)"
			value = aperiodicFmaxVal
			hint = "maximum frequency (Hz)"
		case optAperiodicMinSegmentSec:
			label = "Min segment (s)"
			value = fmt.Sprintf("%.1f", m.aperiodicMinSegmentSec)
			if m.editingNumber && m.isCurrentlyEditing(optAperiodicMinSegmentSec) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum duration for stable fits"
		case optAperiodicExcludeLineNoise:
			label = "Exclude line noise"
			value = m.boolToOnOff(m.aperiodicExcludeLineNoise)
			hint = "exclude line-noise bins before fitting"
		case optAperiodicPeakZ:
			label = "Peak Z-thresh"
			value = aperiodicPeakZVal
			hint = "peak rejection"
		case optAperiodicMinR2:
			label = "Min R2"
			value = aperiodicR2Val
			hint = "minimum fit quality"
		case optAperiodicMinPoints:
			label = "Min Points"
			value = aperiodicPointsVal
			hint = "minimum bins required"
		case optAperiodicPsdBandwidth:
			aperiodicBandwidthVal := fmt.Sprintf("%.1f", m.aperiodicPsdBandwidth)
			if m.aperiodicPsdBandwidth == 0 {
				aperiodicBandwidthVal = "(default)"
			}
			if m.editingNumber && m.isCurrentlyEditing(optAperiodicPsdBandwidth) {
				aperiodicBandwidthVal = m.numberBuffer + "█"
			}
			label = "PSD Bandwidth"
			value = aperiodicBandwidthVal
			hint = "Hz (0=default)"
		case optAperiodicMaxRms:
			aperiodicMaxRmsVal := fmt.Sprintf("%.3f", m.aperiodicMaxRms)
			if m.aperiodicMaxRms == 0 {
				aperiodicMaxRmsVal = "(no limit)"
			}
			if m.editingNumber && m.isCurrentlyEditing(optAperiodicMaxRms) {
				aperiodicMaxRmsVal = m.numberBuffer + "█"
			}
			label = "Max RMS"
			value = aperiodicMaxRmsVal
			hint = "fit quality limit (0=no limit)"
		case optAperiodicLineNoiseFreq:
			label = "Line noise freq"
			value = fmt.Sprintf("%.0f", m.aperiodicLineNoiseFreq)
			if m.editingNumber && m.isCurrentlyEditing(optAperiodicLineNoiseFreq) {
				value = m.numberBuffer + "█"
			}
			hint = "Hz (50 or 60)"
		case optAperiodicLineNoiseWidthHz:
			label = "Line noise width"
			value = fmt.Sprintf("%.1f", m.aperiodicLineNoiseWidthHz)
			if m.editingNumber && m.isCurrentlyEditing(optAperiodicLineNoiseWidthHz) {
				value = m.numberBuffer + "█"
			}
			hint = "Hz bandwidth to exclude"
		case optAperiodicLineNoiseHarmonics:
			label = "Line noise harmonics"
			value = fmt.Sprintf("%d", m.aperiodicLineNoiseHarmonics)
			if m.editingNumber && m.isCurrentlyEditing(optAperiodicLineNoiseHarmonics) {
				value = m.numberBuffer + "█"
			}
			hint = "number of harmonics"

		// Complexity
		case optPEOrder:
			label = "PE Order"
			value = peOrderVal
			hint = "symbol length (3-7)"
		case optPEDelay:
			label = "PE Delay"
			value = peDelayVal
			hint = "sample lag"
		case optComplexitySampleEntropyOrder:
			label = "SampEn Order"
			value = sampEnOrderVal
			hint = "embedding dimension (m)"
		case optComplexitySampleEntropyR:
			label = "SampEn r"
			value = sampEnRVal
			hint = "tolerance as SD fraction"
		case optComplexityMSEScaleMin:
			label = "MSE scale min"
			value = mseScaleMinVal
			hint = "coarse-graining start"
		case optComplexityMSEScaleMax:
			label = "MSE scale max"
			value = mseScaleMaxVal
			hint = "coarse-graining end"
		case optComplexitySignalBasis:
			label = "Signal basis"
			bases := []string{"filtered", "envelope"}
			if m.complexitySignalBasis >= 0 && m.complexitySignalBasis < len(bases) {
				value = bases[m.complexitySignalBasis]
			} else {
				value = "filtered"
			}
			hint = "filtered or envelope"
		case optComplexityMinSegmentSec:
			label = "Min segment (s)"
			value = fmt.Sprintf("%.2f", m.complexityMinSegmentSec)
			if m.editingNumber && m.isCurrentlyEditing(optComplexityMinSegmentSec) {
				value = m.numberBuffer + "█"
			}
			hint = "skip short segments"
		case optComplexityMinSamples:
			label = "Min samples"
			value = fmt.Sprintf("%d", m.complexityMinSamples)
			if m.editingNumber && m.isCurrentlyEditing(optComplexityMinSamples) {
				value = m.numberBuffer + "█"
			}
			hint = "skip low-sample segments"
		case optComplexityZscore:
			label = "Z-score"
			if m.complexityZscore {
				value = "on"
			} else {
				value = "off"
			}
			hint = "normalize per-channel"

		// Bursts
		case optBurstThresholdMethod:
			methods := []string{"percentile", "zscore", "mad"}
			methodVal := "percentile"
			if m.burstThresholdMethod >= 0 && m.burstThresholdMethod < len(methods) {
				methodVal = methods[m.burstThresholdMethod]
			}
			label = "Threshold Method"
			value = methodVal
			hint = "percentile / zscore / mad"
		case optBurstThresholdPercentile:
			burstPercentileVal := fmt.Sprintf("%.1f", m.burstThresholdPercentile)
			if m.editingNumber && m.isCurrentlyEditing(optBurstThresholdPercentile) {
				burstPercentileVal = m.numberBuffer + "█"
			}
			label = "Threshold Percentile"
			value = burstPercentileVal
			hint = "percentile threshold"
		case optBurstThreshold:
			label = "Threshold Z"
			value = burstThreshVal
			hint = "amplitude trigger"
		case optBurstThresholdReference:
			label = "Threshold ref"
			refs := []string{"trial", "subject", "condition"}
			value = refs[m.burstThresholdReference]
			hint = "trial/subject/condition"
		case optBurstMinTrialsPerCondition:
			label = "Min trials/cond"
			value = fmt.Sprintf("%d", m.burstMinTrialsPerCondition)
			if m.editingNumber && m.isCurrentlyEditing(optBurstMinTrialsPerCondition) {
				value = m.numberBuffer + "█"
			}
			hint = "used for condition reference"
		case optBurstMinSegmentSec:
			label = "Min segment (s)"
			value = fmt.Sprintf("%.2f", m.burstMinSegmentSec)
			if m.editingNumber && m.isCurrentlyEditing(optBurstMinSegmentSec) {
				value = m.numberBuffer + "█"
			}
			hint = "skip short segments"
		case optBurstSkipInvalidSegments:
			label = "Skip invalid"
			value = m.boolToOnOff(m.burstSkipInvalidSegments)
			hint = "skip invalid segments"
		case optBurstMinDuration:
			label = "Min Duration"
			value = burstMinDurVal
			hint = "minimum length"
		case optBurstBands:
			label = "Burst bands"
			value = burstBandsVal
			hint = "e.g. beta,gamma"

		// Power
		case optPowerRequireBaseline:
			label = "Require baseline"
			value = powerRequireBaselineVal
			hint = "allow raw log power if OFF"
		case optPowerSubtractEvoked:
			label = "Subtract evoked"
			value = m.boolToOnOff(m.powerSubtractEvoked)
			hint = "induced power; CV-safe only with train_mask"
		case optPowerMinTrialsPerCondition:
			label = "Min trials/cond"
			value = fmt.Sprintf("%d", m.powerMinTrialsPerCondition)
			if m.editingNumber && m.isCurrentlyEditing(optPowerMinTrialsPerCondition) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum trials per condition"
		case optPowerExcludeLineNoise:
			label = "Exclude line noise"
			value = m.boolToOnOff(m.powerExcludeLineNoise)
			hint = "exclude line-noise bins"
		case optPowerLineNoiseFreq:
			label = "Line noise freq"
			value = fmt.Sprintf("%.0f", m.powerLineNoiseFreq)
			if m.editingNumber && m.isCurrentlyEditing(optPowerLineNoiseFreq) {
				value = m.numberBuffer + "█"
			}
			hint = "Hz (50 or 60)"
		case optPowerLineNoiseWidthHz:
			label = "Line noise width"
			value = fmt.Sprintf("%.1f", m.powerLineNoiseWidthHz)
			if m.editingNumber && m.isCurrentlyEditing(optPowerLineNoiseWidthHz) {
				value = m.numberBuffer + "█"
			}
			hint = "Hz bandwidth to exclude"
		case optPowerLineNoiseHarmonics:
			label = "Line noise harmonics"
			value = fmt.Sprintf("%d", m.powerLineNoiseHarmonics)
			if m.editingNumber && m.isCurrentlyEditing(optPowerLineNoiseHarmonics) {
				value = m.numberBuffer + "█"
			}
			hint = "number of harmonics"
		case optPowerEmitDb:
			label = "Emit dB"
			value = m.boolToOnOff(m.powerEmitDb)
			hint = "emit 10*log10 ratios"
		case optPowerBaselineMode:
			label = "Baseline mode"
			value = powerBaselineVal
			hint = "normalization type"

		// Spectral
		case optSpectralEdge:
			label = "Spectral Edge"
			value = spectralEdgeVal
			hint = "percentile for SEF"

		// ERP
		case optERPBaseline:
			label = "ERP baseline"
			value = erpBaselineVal
			hint = "subtract baseline mean"
		case optERPAllowNoBaseline:
			label = "Allow no baseline"
			value = erpAllowNoBaselineVal
			hint = "only used if baseline ON"
		case optERPComponents:
			label = "Components"
			value = erpComponentsVal
			hint = "e.g. n1=0.10-0.20,n2=0.20-0.35"
		case optERPSmoothMs:
			erpSmoothVal := fmt.Sprintf("%.1f ms", m.erpSmoothMs)
			if m.erpSmoothMs == 0 {
				erpSmoothVal = "(no smoothing)"
			}
			if m.editingNumber && m.isCurrentlyEditing(optERPSmoothMs) {
				erpSmoothVal = m.numberBuffer + "█"
			}
			label = "Smooth Window"
			value = erpSmoothVal
			hint = "ms (0=no smoothing)"
		case optERPPeakProminenceUv:
			erpProminenceVal := fmt.Sprintf("%.1f µV", m.erpPeakProminenceUv)
			if m.erpPeakProminenceUv == 0 {
				erpProminenceVal = "(default)"
			}
			if m.editingNumber && m.isCurrentlyEditing(optERPPeakProminenceUv) {
				erpProminenceVal = m.numberBuffer + "█"
			}
			label = "Peak Prominence"
			value = erpProminenceVal
			hint = "µV (0=use default)"
		case optERPLowpassHz:
			erpLowpassVal := fmt.Sprintf("%.1f Hz", m.erpLowpassHz)
			if m.editingNumber && m.isCurrentlyEditing(optERPLowpassHz) {
				erpLowpassVal = m.numberBuffer + "█"
			}
			label = "Low-Pass Filter"
			value = erpLowpassVal
			hint = "Hz before peak detection"

		// Ratios / asymmetry
		case optSpectralRatioPairs:
			label = "Ratio pairs"
			value = spectralRatioPairsVal
			hint = ""
		case optAperiodicSubtractEvoked:
			label = "Induced spectra"
			value = m.boolToOnOff(m.aperiodicSubtractEvoked)
			hint = "subtract evoked for pain"
		case optAsymmetryChannelPairs:
			label = "Channel pairs"
			value = asymPairsVal
			hint = "e.g. F3:F4,C3:C4"
		case optAsymmetryMinSegmentSec:
			label = "Min segment (s)"
			value = fmt.Sprintf("%.2f", m.asymmetryMinSegmentSec)
			if m.editingNumber && m.isCurrentlyEditing(optAsymmetryMinSegmentSec) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum segment duration"
		case optAsymmetryMinCyclesAtFmin:
			label = "Min cycles"
			value = fmt.Sprintf("%.1f", m.asymmetryMinCyclesAtFmin)
			if m.editingNumber && m.isCurrentlyEditing(optAsymmetryMinCyclesAtFmin) {
				value = m.numberBuffer + "█"
			}
			hint = "cycles at lowest freq"
		case optAsymmetrySkipInvalidSegments:
			label = "Skip invalid"
			value = m.boolToOnOff(m.asymmetrySkipInvalidSegments)
			hint = "skip invalid segments"
		case optAsymmetryEmitActivationConvention:
			label = "Emit activation"
			value = m.boolToOnOff(m.asymmetryEmitActivationConvention)
			hint = "(R-L)/(R+L) for activation bands"
		case optAsymmetryActivationBands:
			label = "Activation bands"
			val := m.asymmetryActivationBandsSpec
			if strings.TrimSpace(val) == "" {
				val = "(default: alpha)"
			}
			if m.editingText && m.editingTextField == textFieldAsymmetryActivationBands {
				val = m.textBuffer + "█"
			}
			value = val
			hint = "e.g. alpha,beta"

		// Ratios advanced options
		case optRatiosMinSegmentSec:
			label = "Min segment (s)"
			value = fmt.Sprintf("%.2f", m.ratiosMinSegmentSec)
			if m.editingNumber && m.isCurrentlyEditing(optRatiosMinSegmentSec) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum segment duration"
		case optRatiosMinCyclesAtFmin:
			label = "Min cycles"
			value = fmt.Sprintf("%.1f", m.ratiosMinCyclesAtFmin)
			if m.editingNumber && m.isCurrentlyEditing(optRatiosMinCyclesAtFmin) {
				value = m.numberBuffer + "█"
			}
			hint = "cycles at lowest freq"
		case optRatiosSkipInvalidSegments:
			label = "Skip invalid"
			value = m.boolToOnOff(m.ratiosSkipInvalidSegments)
			hint = "skip invalid segments"

		// Spectral advanced options
		case optSpectralIncludeLogRatios:
			label = "Log ratios"
			value = m.boolToOnOff(m.spectralIncludeLogRatios)
			hint = "include log ratios"
		case optSpectralPsdMethod:
			label = "PSD method"
			methods := []string{"multitaper", "welch"}
			value = methods[m.spectralPsdMethod]
			hint = "multitaper or welch"
		case optSpectralPsdAdaptive:
			label = "PSD adaptive"
			value = m.boolToOnOff(m.spectralPsdAdaptive)
			hint = "adaptive PSD settings"
		case optSpectralMultitaperAdaptive:
			label = "Multitaper adaptive"
			value = m.boolToOnOff(m.spectralMultitaperAdaptive)
			hint = "adaptive multitaper (if supported)"
		case optSpectralFmin:
			label = "Freq min"
			value = fmt.Sprintf("%.1f Hz", m.spectralFmin)
			if m.editingNumber && m.isCurrentlyEditing(optSpectralFmin) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum frequency"
		case optSpectralFmax:
			label = "Freq max"
			value = fmt.Sprintf("%.1f Hz", m.spectralFmax)
			if m.editingNumber && m.isCurrentlyEditing(optSpectralFmax) {
				value = m.numberBuffer + "█"
			}
			hint = "maximum frequency"
		case optSpectralMinSegmentSec:
			label = "Min segment (s)"
			value = fmt.Sprintf("%.2f", m.spectralMinSegmentSec)
			if m.editingNumber && m.isCurrentlyEditing(optSpectralMinSegmentSec) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum segment duration"
		case optSpectralMinCyclesAtFmin:
			label = "Min cycles"
			value = fmt.Sprintf("%.1f", m.spectralMinCyclesAtFmin)
			if m.editingNumber && m.isCurrentlyEditing(optSpectralMinCyclesAtFmin) {
				value = m.numberBuffer + "█"
			}
			hint = "cycles at lowest freq"
		case optSpectralExcludeLineNoise:
			label = "Exclude line noise"
			value = m.boolToOnOff(m.spectralExcludeLineNoise)
			hint = "exclude line-noise bins"
		case optSpectralLineNoiseFreq:
			label = "Line noise freq"
			value = fmt.Sprintf("%.0f", m.spectralLineNoiseFreq)
			if m.editingNumber && m.isCurrentlyEditing(optSpectralLineNoiseFreq) {
				value = m.numberBuffer + "█"
			}
			hint = "Hz (50 or 60)"
		case optSpectralLineNoiseWidthHz:
			label = "Line noise width"
			value = fmt.Sprintf("%.1f", m.spectralLineNoiseWidthHz)
			if m.editingNumber && m.isCurrentlyEditing(optSpectralLineNoiseWidthHz) {
				value = m.numberBuffer + "█"
			}
			hint = "Hz bandwidth to exclude"
		case optSpectralLineNoiseHarmonics:
			label = "Line noise harmonics"
			value = fmt.Sprintf("%d", m.spectralLineNoiseHarmonics)
			if m.editingNumber && m.isCurrentlyEditing(optSpectralLineNoiseHarmonics) {
				value = m.numberBuffer + "█"
			}
			hint = "number of harmonics"

		// Quality group header
		case optFeatGroupQuality:
			label = "▸ Quality"
			hint = "Space to toggle"
			if m.featGroupQualityExpanded {
				label = "▾ Quality"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optQualityPsdMethod:
			label = "PSD method"
			methods := []string{"welch", "multitaper"}
			value = methods[m.qualityPsdMethod]
			hint = "welch or multitaper"
		case optQualityFmin:
			label = "Freq min"
			value = fmt.Sprintf("%.1f Hz", m.qualityFmin)
			if m.editingNumber && m.isCurrentlyEditing(optQualityFmin) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum frequency"
		case optQualityFmax:
			label = "Freq max"
			value = fmt.Sprintf("%.1f Hz", m.qualityFmax)
			if m.editingNumber && m.isCurrentlyEditing(optQualityFmax) {
				value = m.numberBuffer + "█"
			}
			hint = "maximum frequency"
		case optQualityNFft:
			label = "N FFT"
			value = fmt.Sprintf("%d", m.qualityNfft)
			if m.editingNumber && m.isCurrentlyEditing(optQualityNFft) {
				value = m.numberBuffer + "█"
			}
			hint = "FFT size"
		case optQualityExcludeLineNoise:
			label = "Exclude line noise"
			value = m.boolToOnOff(m.qualityExcludeLineNoise)
			hint = "remove line noise bins"
		case optQualityLineNoiseFreq:
			label = "Line noise freq"
			value = fmt.Sprintf("%.0f Hz", m.qualityLineNoiseFreq)
			if m.editingNumber && m.isCurrentlyEditing(optQualityLineNoiseFreq) {
				value = m.numberBuffer + "█"
			}
			hint = "50 or 60 Hz"
		case optQualityLineNoiseWidthHz:
			label = "Line noise width"
			value = fmt.Sprintf("%.1f Hz", m.qualityLineNoiseWidthHz)
			if m.editingNumber && m.isCurrentlyEditing(optQualityLineNoiseWidthHz) {
				value = m.numberBuffer + "█"
			}
			hint = "exclusion bandwidth"
		case optQualityLineNoiseHarmonics:
			label = "Line noise harmonics"
			value = fmt.Sprintf("%d", m.qualityLineNoiseHarmonics)
			if m.editingNumber && m.isCurrentlyEditing(optQualityLineNoiseHarmonics) {
				value = m.numberBuffer + "█"
			}
			hint = "number of harmonics"
		case optQualitySnrSignalBandMin:
			label = "SNR signal min"
			value = fmt.Sprintf("%.1f Hz", m.qualitySnrSignalBandMin)
			if m.editingNumber && m.isCurrentlyEditing(optQualitySnrSignalBandMin) {
				value = m.numberBuffer + "█"
			}
			hint = "signal band lower bound"
		case optQualitySnrSignalBandMax:
			label = "SNR signal max"
			value = fmt.Sprintf("%.1f Hz", m.qualitySnrSignalBandMax)
			if m.editingNumber && m.isCurrentlyEditing(optQualitySnrSignalBandMax) {
				value = m.numberBuffer + "█"
			}
			hint = "signal band upper bound"
		case optQualitySnrNoiseBandMin:
			label = "SNR noise min"
			value = fmt.Sprintf("%.1f Hz", m.qualitySnrNoiseBandMin)
			if m.editingNumber && m.isCurrentlyEditing(optQualitySnrNoiseBandMin) {
				value = m.numberBuffer + "█"
			}
			hint = "noise band lower bound"
		case optQualitySnrNoiseBandMax:
			label = "SNR noise max"
			value = fmt.Sprintf("%.1f Hz", m.qualitySnrNoiseBandMax)
			if m.editingNumber && m.isCurrentlyEditing(optQualitySnrNoiseBandMax) {
				value = m.numberBuffer + "█"
			}
			hint = "noise band upper bound"
		case optQualityMuscleBandMin:
			label = "Muscle band min"
			value = fmt.Sprintf("%.1f Hz", m.qualityMuscleBandMin)
			if m.editingNumber && m.isCurrentlyEditing(optQualityMuscleBandMin) {
				value = m.numberBuffer + "█"
			}
			hint = "muscle artifact lower"
		case optQualityMuscleBandMax:
			label = "Muscle band max"
			value = fmt.Sprintf("%.1f Hz", m.qualityMuscleBandMax)
			if m.editingNumber && m.isCurrentlyEditing(optQualityMuscleBandMax) {
				value = m.numberBuffer + "█"
			}
			hint = "muscle artifact upper"

		// ERDS group header
		case optFeatGroupERDS:
			label = "▸ ERDS"
			hint = "Space to toggle"
			if m.featGroupERDSExpanded {
				label = "▾ ERDS"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optERDSUseLogRatio:
			label = "Use log ratio"
			value = m.boolToOnOff(m.erdsUseLogRatio)
			hint = "dB (on) vs percent (off)"
		case optERDSMinBaselinePower:
			label = "Min baseline power"
			value = fmt.Sprintf("%.2e", m.erdsMinBaselinePower)
			if m.editingNumber && m.isCurrentlyEditing(optERDSMinBaselinePower) {
				value = m.numberBuffer + "█"
			}
			hint = "clamp baseline power"
		case optERDSMinActivePower:
			label = "Min active power"
			value = fmt.Sprintf("%.2e", m.erdsMinActivePower)
			if m.editingNumber && m.isCurrentlyEditing(optERDSMinActivePower) {
				value = m.numberBuffer + "█"
			}
			hint = "clamp active power"
		case optERDSMinSegmentSec:
			label = "Min segment (s)"
			value = fmt.Sprintf("%.2f", m.erdsMinSegmentSec)
			if m.editingNumber && m.isCurrentlyEditing(optERDSMinSegmentSec) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum segment duration"
		case optERDSBands:
			erdsBandsVal := m.erdsBandsSpec
			if strings.TrimSpace(erdsBandsVal) == "" {
				erdsBandsVal = "(default: alpha,beta)"
			}
			if m.editingText && m.editingTextField == textFieldERDSBands {
				erdsBandsVal = m.textBuffer + "█"
			}
			label = "Bands"
			value = erdsBandsVal
			hint = "e.g. alpha,beta"
		case optERDSOnsetThresholdSigma:
			label = "Onset threshold (sigma)"
			value = fmt.Sprintf("%.2f", m.erdsOnsetThresholdSigma)
			if m.editingNumber && m.isCurrentlyEditing(optERDSOnsetThresholdSigma) {
				value = m.numberBuffer + "█"
			}
			hint = "ERD onset threshold in baseline SD units"
		case optERDSOnsetMinDurationMs:
			label = "Onset min duration (ms)"
			value = fmt.Sprintf("%.1f", m.erdsOnsetMinDurationMs)
			if m.editingNumber && m.isCurrentlyEditing(optERDSOnsetMinDurationMs) {
				value = m.numberBuffer + "█"
			}
			hint = "sustained threshold crossing duration"
		case optERDSReboundMinLatencyMs:
			label = "Rebound min latency (ms)"
			value = fmt.Sprintf("%.1f", m.erdsReboundMinLatencyMs)
			if m.editingNumber && m.isCurrentlyEditing(optERDSReboundMinLatencyMs) {
				value = m.numberBuffer + "█"
			}
			hint = "delay from ERD peak to rebound search"
		case optERDSInferContralateral:
			label = "Infer contralateral side"
			value = m.boolToOnOff(m.erdsInferContralateral)
			hint = "use stronger somatosensory ERD if side metadata missing"

		// Generic
		case optMinEpochs:
			label = "Min Epochs"
			value = minEpochsVal
			hint = "global minimum required"
		case optFeatAnalysisMode:
			label = "Analysis mode"
			modes := []string{"group_stats", "trial_ml_safe"}
			value = modes[m.featAnalysisMode]
			hint = "group_stats / trial_ml_safe"
		case optFeatComputeChangeScores:
			label = "Change scores"
			value = m.boolToOnOff(m.featComputeChangeScores)
			hint = "add within-subject change columns"
		case optFeatSaveTfrWithSidecar:
			label = "Save TFR sidecar"
			value = m.boolToOnOff(m.featSaveTfrWithSidecar)
			hint = "write TFR arrays for inspection"
		case optFeatNJobsBands:
			label = "n_jobs (bands)"
			value = fmt.Sprintf("%d", m.featNJobsBands)
			if m.editingNumber && m.isCurrentlyEditing(optFeatNJobsBands) {
				value = m.numberBuffer + "█"
			}
			hint = "-1 = all cores"
		case optFeatNJobsConnectivity:
			label = "n_jobs (connectivity)"
			value = fmt.Sprintf("%d", m.featNJobsConnectivity)
			if m.editingNumber && m.isCurrentlyEditing(optFeatNJobsConnectivity) {
				value = m.numberBuffer + "█"
			}
			hint = "-1 = all cores"
		case optFeatNJobsAperiodic:
			label = "n_jobs (aperiodic)"
			value = fmt.Sprintf("%d", m.featNJobsAperiodic)
			if m.editingNumber && m.isCurrentlyEditing(optFeatNJobsAperiodic) {
				value = m.numberBuffer + "█"
			}
			hint = "-1 = all cores"
		case optFeatNJobsComplexity:
			label = "n_jobs (complexity)"
			value = fmt.Sprintf("%d", m.featNJobsComplexity)
			if m.editingNumber && m.isCurrentlyEditing(optFeatNJobsComplexity) {
				value = m.numberBuffer + "█"
			}
			hint = "-1 = all cores"

		default:
			label = "Unknown"
			value = ""
			hint = ""
		}

		if lineIdx >= startLine && lineIdx < endLine {
			styledLabel := labelStyle.Render(label + ":")
			styledValue := valueStyle.Render(value + expandIndicator)
			styledHint := hintStyle.Render(hint)
			b.WriteString(styles.RenderConfigLine(cursor, styledLabel, styledValue, styledHint, labelWidth, m.contentWidth) + "\n")
		}
		lineIdx++

		// Expanded items (connectivity measures)
		if opt == optConnectivity && m.expandedOption == expandedConnectivityMeasures {
			subIndent := "      " // 6 spaces for sub-items
			for j, measure := range connectivityMeasures {
				isSelected := m.connectivityMeasures[j]
				isSubFocused := j == m.subCursor

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				desc := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + measure.Description)
				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(measure.Name) + desc + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (directed connectivity measures)
		if opt == optDirectedConnMeasures && m.expandedOption == expandedDirectedConnMeasures {
			subIndent := "      " // 6 spaces for sub-items
			for j, measure := range directedConnectivityMeasures {
				isSelected := m.directedConnMeasures[j]
				isSubFocused := j == m.subCursor

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				desc := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + measure.Description)
				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(measure.Name) + desc + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (fMRI condition A column)
		if opt == optSourceLocFmriCondAColumn && m.expandedOption == expandedFmriCondAColumn {
			subIndent := "      " // 6 spaces for sub-items
			for j, col := range m.fmriDiscoveredColumns {
				isSubFocused := j == m.subCursor
				isSelected := m.sourceLocFmriCondAColumn == col

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(col) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (fMRI condition A value)
		if opt == optSourceLocFmriCondAValue && m.expandedOption == expandedFmriCondAValue {
			subIndent := "      " // 6 spaces for sub-items
			vals := m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondAColumn)
			for j, v := range vals {
				isSubFocused := j == m.subCursor
				isSelected := m.sourceLocFmriCondAValue == v

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(v) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (fMRI condition B column)
		if opt == optSourceLocFmriCondBColumn && m.expandedOption == expandedFmriCondBColumn {
			subIndent := "      " // 6 spaces for sub-items
			for j, col := range m.fmriDiscoveredColumns {
				isSubFocused := j == m.subCursor
				isSelected := m.sourceLocFmriCondBColumn == col

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(col) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (fMRI condition B value)
		if opt == optSourceLocFmriCondBValue && m.expandedOption == expandedFmriCondBValue {
			subIndent := "      " // 6 spaces for sub-items
			vals := m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondBColumn)
			for j, v := range vals {
				isSubFocused := j == m.subCursor
				isSelected := m.sourceLocFmriCondBValue == v

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(v) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (fMRI condition trial_type scope)
		if opt == optSourceLocFmriConditionScopeTrialTypes && m.expandedOption == expandedSourceLocFmriScopeTrialTypes {
			subIndent := "      " // 6 spaces for sub-items
			items := m.getExpandedListItems()
			for j, item := range items {
				isSubFocused := j == m.subCursor
				isSelected := m.isExpandedItemSelected(j, item)

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(item) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (fMRI stimulation phase scope)
		if opt == optSourceLocFmriStimPhasesToModel && m.expandedOption == expandedSourceLocFmriStimPhases {
			subIndent := "      " // 6 spaces for sub-items
			items := m.getExpandedListItems()
			for j, item := range items {
				isSubFocused := j == m.subCursor
				isSelected := m.isExpandedItemSelected(j, item)

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(item) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (ITPC condition column)
		if opt == optItpcConditionColumn && m.expandedOption == expandedItpcConditionColumn {
			subIndent := "      " // 6 spaces for sub-items
			for j, col := range m.GetAvailableColumns() {
				isSubFocused := j == m.subCursor
				isSelected := m.itpcConditionColumn == col

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(col) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (ITPC condition values)
		if opt == optItpcConditionValues && m.expandedOption == expandedItpcConditionValues {
			subIndent := "      " // 6 spaces for sub-items
			vals := m.GetDiscoveredColumnValues(m.itpcConditionColumn)
			for j, v := range vals {
				isSubFocused := j == m.subCursor
				isSelected := m.isColumnValueSelected(v)

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(v) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (Connectivity condition column)
		if opt == optConnConditionColumn && m.expandedOption == expandedConnConditionColumn {
			subIndent := "      " // 6 spaces for sub-items
			for j, col := range m.GetAvailableColumns() {
				isSubFocused := j == m.subCursor
				isSelected := m.connConditionColumn == col

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(col) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (Connectivity condition values)
		if opt == optConnConditionValues && m.expandedOption == expandedConnConditionValues {
			subIndent := "      " // 6 spaces for sub-items
			vals := m.GetDiscoveredColumnValues(m.connConditionColumn)
			for j, v := range vals {
				isSubFocused := j == m.subCursor
				isSelected := m.isColumnValueSelected(v)

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(v) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (IAF ROIs)
		if opt == optIAFRois && m.expandedOption == expandedIAFRois {
			subIndent := "      " // 6 spaces for sub-items
			for j, roi := range m.rois {
				isSubFocused := j == m.subCursor
				isSelected := m.isExpandedItemSelected(j, roi.Key)
				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(roi.Key) + "\n")
				}
				lineIdx++
			}
		}
	}

	// Show scroll indicator for items below
	if showScrollIndicators && lineIdx > endLine {
		remaining := lineIdx - endLine
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↓ %d more items below", remaining)) + "\n")
	}

	return b.String()
}
