// Preprocessing pipeline advanced configuration.
package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

func (m Model) renderPreprocessingAdvancedConfig() string {
	var b strings.Builder

	b.WriteString(styles.RenderStepHeader("Advanced", m.contentWidth) + "\n")
	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)

	if m.useDefaultAdvanced {
		return m.renderDefaultConfigView("preprocessing")
	}

	if m.editingNumber {
		b.WriteString(infoStyle.Render("Enter value, Enter to confirm, Esc to cancel") + "\n")
	} else if m.editingText {
		b.WriteString(infoStyle.Render("Type text, Enter to confirm, Esc to cancel") + "\n")
	} else {
		b.WriteString(infoStyle.Render("Space: toggle/expand  Enter: proceed") + "\n")
	}

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	// Build values for display
	nJobsVal := fmt.Sprintf("%d", m.prepNJobs)
	montageVal := m.prepMontage
	if m.editingText && m.editingTextField == textFieldPrepMontage {
		montageVal = m.textBuffer + "█"
	}
	chTypesVal := m.prepChTypes
	if m.editingText && m.editingTextField == textFieldPrepChTypes {
		chTypesVal = m.textBuffer + "█"
	}
	eegRefVal := m.prepEegReference
	if m.editingText && m.editingTextField == textFieldPrepEegReference {
		eegRefVal = m.textBuffer + "█"
	}
	fileExtVal := m.prepFileExtension
	if m.editingText && m.editingTextField == textFieldPrepFileExtension {
		fileExtVal = m.textBuffer + "█"
	}
	renameAnotDictVal := m.prepRenameAnotDict
	if m.editingText && m.editingTextField == textFieldPrepRenameAnotDict {
		renameAnotDictVal = m.textBuffer + "█"
	}
	if strings.TrimSpace(renameAnotDictVal) == "" {
		renameAnotDictVal = "(none)"
	}
	customBadDictVal := m.prepCustomBadDict
	if m.editingText && m.editingTextField == textFieldPrepCustomBadDict {
		customBadDictVal = m.textBuffer + "█"
	}
	if strings.TrimSpace(customBadDictVal) == "" {
		customBadDictVal = "(none)"
	}
	icaLabelsVal := m.icaLabelsToKeep
	if m.editingText && m.editingTextField == textFieldIcaLabelsToKeep {
		icaLabelsVal = m.textBuffer + "█"
	}
	if strings.TrimSpace(icaLabelsVal) == "" {
		icaLabelsVal = "(default)"
	}
	eogChannelsVal := m.prepEogChannels
	if m.editingText && m.editingTextField == textFieldPrepEogChannels {
		eogChannelsVal = m.textBuffer + "█"
	}
	conditionsVal := m.prepConditions
	if m.editingText && m.editingTextField == textFieldPrepConditions {
		conditionsVal = m.textBuffer + "█"
	}
	resampleVal := fmt.Sprintf("%d Hz", m.prepResample)
	lFreqVal := fmt.Sprintf("%.1f Hz", m.prepLFreq)
	hFreqVal := fmt.Sprintf("%.1f Hz", m.prepHFreq)
	notchVal := fmt.Sprintf("%d Hz", m.prepNotch)
	lineFreqVal := fmt.Sprintf("%d Hz", m.prepLineFreq)
	icaCompVal := fmt.Sprintf("%.2f", m.prepICAComp)
	probThreshVal := fmt.Sprintf("%.1f", m.prepProbThresh)
	tminVal := fmt.Sprintf("%.1f s", m.prepEpochsTmin)
	tmaxVal := fmt.Sprintf("%.1f s", m.prepEpochsTmax)
	repeatsVal := fmt.Sprintf("%d", m.prepRepeats)
	breaksMinLenVal := fmt.Sprintf("%d s", m.prepBreaksMinLength)
	tStartPrevVal := fmt.Sprintf("%d s", m.prepTStartAfterPrevious)
	tStopNextVal := fmt.Sprintf("%d s", m.prepTStopBeforeNext)
	randomStateVal := fmt.Sprintf("%d", m.prepRandomState)
	zaplineFlineVal := fmt.Sprintf("%.1f Hz", m.prepZaplineFline)
	icaLFreqVal := fmt.Sprintf("%.1f Hz", m.prepICALFreq)
	icaRejThreshVal := fmt.Sprintf("%.0f µV", m.prepICARejThresh)

	var baselineVal string
	if m.prepEpochsNoBaseline {
		baselineVal = "N/A"
	} else if m.prepEpochsBaselineStart == 0 && m.prepEpochsBaselineEnd == 0 {
		baselineVal = "(default)"
	} else {
		baselineVal = fmt.Sprintf("%.2f to %.2f s", m.prepEpochsBaselineStart, m.prepEpochsBaselineEnd)
	}
	var rejectVal string
	if m.prepEpochsReject == 0 {
		rejectVal = "(none)"
	} else {
		rejectVal = fmt.Sprintf("%.0f µV", m.prepEpochsReject)
	}

	// Input overrides for number editing
	if m.editingNumber {
		buffer := m.numberBuffer + "█"
		switch {
		case m.isCurrentlyEditing(optPrepNJobs):
			nJobsVal = buffer
		case m.isCurrentlyEditing(optPrepResample):
			resampleVal = buffer
		case m.isCurrentlyEditing(optPrepLFreq):
			lFreqVal = buffer
		case m.isCurrentlyEditing(optPrepHFreq):
			hFreqVal = buffer
		case m.isCurrentlyEditing(optPrepNotch):
			notchVal = buffer
		case m.isCurrentlyEditing(optPrepLineFreq):
			lineFreqVal = buffer
		case m.isCurrentlyEditing(optPrepICAComp):
			icaCompVal = buffer
		case m.isCurrentlyEditing(optPrepProbThresh):
			probThreshVal = buffer
		case m.isCurrentlyEditing(optPrepEpochsTmin):
			tminVal = buffer
		case m.isCurrentlyEditing(optPrepEpochsTmax):
			tmaxVal = buffer
		case m.isCurrentlyEditing(optPrepEpochsBaseline):
			baselineVal = buffer
		case m.isCurrentlyEditing(optPrepEpochsReject):
			rejectVal = buffer
		case m.isCurrentlyEditing(optPrepRepeats):
			repeatsVal = buffer
		case m.isCurrentlyEditing(optPrepBreaksMinLength):
			breaksMinLenVal = buffer
		case m.isCurrentlyEditing(optPrepTStartAfterPrevious):
			tStartPrevVal = buffer
		case m.isCurrentlyEditing(optPrepTStopBeforeNext):
			tStopNextVal = buffer
		case m.isCurrentlyEditing(optPrepRandomState):
			randomStateVal = buffer
		case m.isCurrentlyEditing(optPrepZaplineFline):
			zaplineFlineVal = buffer
		case m.isCurrentlyEditing(optPrepICALFreq):
			icaLFreqVal = buffer
		case m.isCurrentlyEditing(optPrepICARejThresh):
			icaRejThreshVal = buffer
		}
	}

	options := m.getPreprocessingOptions()

	// Scrolling support
	totalLines := len(options)
	effectiveHeight := m.height
	if effectiveHeight <= 0 {
		effectiveHeight = defaultTerminalHeight
	}
	startLine, endLine, showScrollIndicators := calculateScrollWindow(
		totalLines, m.advancedOffset, effectiveHeight, configOverhead)

	if showScrollIndicators && startLine > 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↑ %d more above", startLine)) + "\n")
	}

	for i, opt := range options {
		if i < startLine || i >= endLine {
			continue
		}

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

		switch opt {
		case optUseDefaults:
			label = "Configuration"
			if m.useDefaultAdvanced {
				value = "Using Defaults"
				hint = "Space to customize"
			} else {
				value = "Custom"
				hint = "Space to use defaults"
			}

		// Group headers with chevron indicators
		case optPrepGroupStages:
			if m.prepGroupStagesExpanded {
				label = "▾ Stages"
			} else {
				label = "▸ Stages"
			}
			hint = "Choose preprocessing steps"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		case optPrepGroupGeneral:
			if m.prepGroupGeneralExpanded {
				label = "▾ General"
			} else {
				label = "▸ General"
			}
			hint = "Montage, parallel jobs, random seed"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		case optPrepGroupFiltering:
			if m.prepGroupFilteringExpanded {
				label = "▾ Filtering"
			} else {
				label = "▸ Filtering"
			}
			hint = "Resampling, bandpass, notch filters"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		case optPrepGroupPyprep:
			if m.prepGroupPyprepExpanded {
				label = "▾ PyPREP"
			} else {
				label = "▸ PyPREP"
			}
			hint = "Bad channel detection parameters"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		case optPrepGroupICA:
			if m.prepGroupICAExpanded {
				label = "▾ ICA"
			} else {
				label = "▸ ICA"
			}
			hint = "ICA algorithm and parameters"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		case optPrepGroupEpoching:
			if m.prepGroupEpochingExpanded {
				label = "▾ Epoching"
			} else {
				label = "▸ Epoching"
			}
			hint = "Epoch timing and rejection criteria"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		// Stage toggles (indented under Stages group)
		case optPrepStageBadChannels:
			label = "Bad Channels"
			value = m.boolToOnOff(m.prepStageSelected[0])
			hint = "Automatically detect noisy channels"
		case optPrepStageFiltering:
			label = "Filtering"
			value = m.boolToOnOff(m.prepStageSelected[1])
			hint = "Apply frequency filters to data"
		case optPrepStageICA:
			label = "ICA"
			value = m.boolToOnOff(m.prepStageSelected[2])
			hint = "Remove artifacts via ICA decomposition"
		case optPrepStageEpoching:
			label = "Epoching"
			value = m.boolToOnOff(m.prepStageSelected[3])
			hint = "Extract time-locked epochs from events"

		// General settings (indented)
		case optPrepMontage:
			label = "Montage"
			value = montageVal
			hint = "EEG electrode layout (e.g., easycap-M1)"
		case optPrepChTypes:
			label = "Ch Types"
			value = chTypesVal
			hint = "Channel types to process (e.g., eeg)"
		case optPrepEegReference:
			label = "Reference"
			value = eegRefVal
			hint = "Re-referencing scheme (e.g., average)"
		case optPrepEogChannels:
			label = "EOG Chans"
			value = eogChannelsVal
			if strings.TrimSpace(value) == "" {
				value = "(none)"
			}
			hint = "Eye movement channels (e.g., Fp1,Fp2)"
		case optPrepRandomState:
			label = "Random Seed"
			value = randomStateVal
			hint = "Random seed for reproducible results"
		case optPrepTaskIsRest:
			label = "Resting State"
			value = m.boolToOnOff(m.prepTaskIsRest)
			hint = "Data is resting-state (no events)"
		case optPrepNJobs:
			label = "N Jobs"
			value = nJobsVal
			hint = "Number of parallel processes to use"
		case optPrepUsePyprep:
			label = "Use PyPREP"
			value = m.boolToOnOff(m.prepUsePyprep)
			hint = "Enable PyPREP for bad channel detection"
		case optPrepUseIcalabel:
			label = "Use ICA-Label"
			value = m.boolToOnOff(m.prepUseIcalabel)
			hint = "Auto-classify ICA components as brain/artifact"

		// Filtering settings (indented)
		case optPrepResample:
			label = "Resample"
			value = resampleVal
			hint = "Downsample to this sampling rate"
		case optPrepLFreq:
			label = "High-Pass"
			value = lFreqVal
			hint = "High-pass filter cutoff frequency"
		case optPrepHFreq:
			label = "Low-Pass"
			value = hFreqVal
			hint = "Low-pass filter cutoff frequency"
		case optPrepNotch:
			label = "Notch"
			value = notchVal
			hint = "Remove line noise at this frequency"
		case optPrepLineFreq:
			label = "Line Freq"
			value = lineFreqVal
			hint = "Power line frequency (50 or 60 Hz)"
		case optPrepZaplineFline:
			label = "Zapline"
			value = zaplineFlineVal
			hint = "Zapline: remove power line harmonics"
		case optPrepFindBreaks:
			label = "Find Breaks"
			value = m.boolToOnOff(m.prepFindBreaks)
			hint = "Identify gaps in continuous recording"

		// PyPREP settings (indented)
		case optPrepRansac:
			label = "RANSAC"
			value = m.boolToOnOff(m.prepRansac)
			hint = "RANSAC: robust bad channel detection"
		case optPrepRepeats:
			label = "Repeats"
			value = repeatsVal
			hint = "Number of detection iterations to run"
		case optPrepAverageReref:
			label = "Avg Reref"
			value = m.boolToOnOff(m.prepAverageReref)
			hint = "Average reference before detection"
		case optPrepFileExtension:
			label = "File Ext"
			value = fileExtVal
			hint = "Raw data file extension (e.g., .vhdr)"
		case optPrepConsiderPreviousBads:
			label = "Keep Bads"
			value = m.boolToOnOff(m.prepConsiderPreviousBads)
			hint = "Keep channels marked bad in previous runs"
		case optPrepOverwriteChansTsv:
			label = "Overwrite TSV"
			value = m.boolToOnOff(m.prepOverwriteChansTsv)
			hint = "Update channels.tsv with detected bads"
		case optPrepDeleteBreaks:
			label = "Del Breaks"
			value = m.boolToOnOff(m.prepDeleteBreaks)
			hint = "Remove break periods from data"
		case optPrepBreaksMinLength:
			label = "Break Len"
			value = breaksMinLenVal
			hint = "Minimum duration to qualify as break"
		case optPrepTStartAfterPrevious:
			label = "T Start"
			value = tStartPrevVal
			hint = "Seconds after previous event to include"
		case optPrepTStopBeforeNext:
			label = "T Stop"
			value = tStopNextVal
			hint = "Seconds before next event to include"
		case optPrepRenameAnotDict:
			label = "Rename Anot"
			value = renameAnotDictVal
			hint = "JSON: rename annotations (old:new)"
		case optPrepCustomBadDict:
			label = "Custom Bads"
			value = customBadDictVal
			hint = "JSON: custom bad channels per task/subject"

		// ICA settings (indented)
		case optPrepSpatialFilter:
			spatialFilterVal := []string{"ica", "ssp"}[m.prepSpatialFilter]
			label = "Spatial Filter"
			value = spatialFilterVal
			hint = "Spatial filter: ICA or SSP"
		case optPrepICAAlgorithm:
			icaAlgVal := []string{"extended_infomax", "fastica", "infomax", "picard"}[m.prepICAAlgorithm]
			label = "Algorithm"
			value = icaAlgVal
			hint = "ICA decomposition algorithm"
		case optPrepICAComp:
			label = "Components"
			value = icaCompVal
			hint = "Components: number (int) or variance (0-1)"
		case optPrepICALFreq:
			label = "ICA High-Pass"
			value = icaLFreqVal
			hint = "High-pass filter for ICA preprocessing"
		case optPrepICARejThresh:
			label = "ICA Reject"
			value = icaRejThreshVal
			hint = "Peak-to-peak threshold for ICA epochs (µV)"
		case optPrepProbThresh:
			label = "Prob Thresh"
			value = probThreshVal
			hint = "Minimum probability for IC label acceptance"
		case optPrepKeepMnebidsBads:
			label = "Keep BIDS"
			value = m.boolToOnOff(m.prepKeepMnebidsBads)
			hint = "Keep ICs flagged as bad in MNE-BIDS"
		case optIcaLabelsToKeep:
			label = "Labels Keep"
			value = icaLabelsVal
			hint = "Comma-separated IC labels to keep (e.g., brain,other)"

		// Epoching settings (indented)
		case optPrepConditions:
			condVal := conditionsVal
			if strings.TrimSpace(condVal) == "" {
				condVal = "(default)"
			}
			label = "Conditions"
			value = condVal
			hint = "Event types/triggers to epoch"
		case optPrepEpochsTmin:
			label = "Tmin"
			value = tminVal
			hint = "Epoch start time relative to event (s)"
		case optPrepEpochsTmax:
			label = "Tmax"
			value = tmaxVal
			hint = "Epoch end time relative to event (s)"
		case optPrepEpochsNoBaseline:
			label = "No Baseline"
			value = m.boolToOnOff(m.prepEpochsNoBaseline)
			hint = "Disable baseline correction"
		case optPrepEpochsBaseline:
			label = "Baseline"
			value = baselineVal
			hint = "Baseline correction window (start, end) seconds"
		case optPrepEpochsReject:
			label = "Reject (µV)"
			value = rejectVal
			hint = "Reject epochs exceeding this amplitude (µV)"
		case optPrepRejectMethod:
			rejectMethods := []string{"none", "autoreject_local", "autoreject_global"}
			label = "Reject Method"
			value = rejectMethods[m.prepRejectMethod]
			hint = "Algorithm: none, autoreject_local, autoreject_global"
		case optPrepRunSourceEstimation:
			label = "Source Est"
			value = m.boolToOnOff(m.prepRunSourceEstimation)
			hint = "Run source localization after preprocessing"
		case optPrepWriteCleanEvents:
			label = "Write Clean Events"
			value = m.boolToOnOff(m.prepWriteCleanEvents)
			hint = "Write clean events.tsv aligned to kept epochs"
		case optPrepOverwriteCleanEvents:
			label = "Overwrite Clean Events"
			value = m.boolToOnOff(m.prepOverwriteCleanEvents)
			hint = "Overwrite existing clean events.tsv"
		case optPrepCleanEventsStrict:
			label = "Clean Events Strict"
			value = m.boolToOnOff(m.prepCleanEventsStrict)
			hint = "Fail if clean events.tsv cannot be written"
		// ECG channels
		case optPrepEcgChannels:
			label = "ECG Channels"
			val := m.prepEcgChannels
			if strings.TrimSpace(val) == "" {
				val = "(none)"
			}
			if m.editingText && m.editingTextField == textFieldPrepEcgChannels {
				val = m.textBuffer + "█"
			}
			value = val
			hint = "space-separated ECG channel names"
		// Autoreject
		case optPrepAutorejectNInterpolate:
			label = "Autoreject N Interpolate"
			val := m.prepAutorejectNInterpolate
			if strings.TrimSpace(val) == "" {
				val = "(default)"
			}
			if m.editingText && m.editingTextField == textFieldPrepAutorejectNInterpolate {
				val = m.textBuffer + "█"
			}
			value = val
			hint = "comma-separated integers"
		// Alignment
		case optAlignAllowMisalignedTrim:
			label = "Allow Misaligned Trim"
			value = m.boolToOnOff(m.alignAllowMisalignedTrim)
			hint = "trim misaligned EEG/fMRI"
		case optAlignMinAlignmentSamples:
			label = "Min Alignment Samples"
			value = fmt.Sprintf("%d", m.alignMinAlignmentSamples)
			if m.editingNumber && m.isCurrentlyEditing(optAlignMinAlignmentSamples) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum overlap samples"
		case optAlignTrimToFirstVolume:
			label = "Trim to First Volume"
			value = m.boolToOnOff(m.alignTrimToFirstVolume)
			hint = "trim EEG to first fMRI volume"
		case optAlignFmriOnsetReference:
			refs := []string{"as_is", "first_volume", "scanner_trigger"}
			label = "fMRI Onset Reference"
			value = refs[m.alignFmriOnsetReference%len(refs)]
			hint = "alignment reference point"
		// Event Column Mapping
		case optEventColPredictor:
			label = "Predictor Columns"
			val := m.eventColPredictor
			if strings.TrimSpace(val) == "" {
				val = "(default)"
			}
			if m.editingText && m.editingTextField == textFieldEventColPredictor {
				val = m.textBuffer + "█"
			}
			value = val
			hint = "comma-separated column names"
		case optEventColRating:
			label = "Rating Columns"
			val := m.eventColOutcome
			if strings.TrimSpace(val) == "" {
				val = "(default)"
			}
			if m.editingText && m.editingTextField == textFieldEventColOutcome {
				val = m.textBuffer + "█"
			}
			value = val
			hint = "comma-separated column names"
		case optEventColBinaryOutcome:
			label = "Binary Outcome Columns"
			val := m.eventColBinaryOutcome
			if strings.TrimSpace(val) == "" {
				val = "(default)"
			}
			if m.editingText && m.editingTextField == textFieldEventColBinaryOutcome {
				val = m.textBuffer + "█"
			}
			value = val
			hint = "comma-separated column names"
		case optEventColCondition:
			label = "Condition Columns"
			val := m.eventColCondition
			if strings.TrimSpace(val) == "" {
				val = "(default)"
			}
			if m.editingText && m.editingTextField == textFieldEventColCondition {
				val = m.textBuffer + "█"
			}
			value = val
			hint = "comma-separated condition label columns"
		case optConditionPreferredPrefixes:
			label = "Condition Trigger Prefixes"
			val := m.conditionPreferredPrefixes
			if strings.TrimSpace(val) == "" {
				val = "(default)"
			}
			if m.editingText && m.editingTextField == textFieldConditionPreferredPrefixes {
				val = m.textBuffer + "█"
			}
			value = val
			hint = "comma-separated trigger prefixes"
		}

		styledHint := ""
		if hint != "" {
			styledHint = hintStyle.Render(hint)
		}
		b.WriteString(styles.RenderConfigLine(cursor, labelStyle.Render(label+":"), valueStyle.Render(value), styledHint, labelWidth, m.contentWidth) + "\n")
	}

	if showScrollIndicators && endLine < totalLines {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↓ %d more below", totalLines-endLine)) + "\n")
	}

	return b.String()
}
