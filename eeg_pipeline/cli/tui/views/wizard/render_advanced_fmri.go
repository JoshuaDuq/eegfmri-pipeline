// fMRI preprocessing pipeline advanced configuration.
package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

func (m Model) renderFmriAdvancedConfig() string {
	var b strings.Builder

	b.WriteString(styles.RenderStepHeader("Advanced", m.contentWidth) + "\n")
	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)

	if m.useDefaultAdvanced {
		return m.renderDefaultConfigView("fMRI preprocessing")
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
	engineVal := []string{"docker", "apptainer"}[m.fmriEngineIndex%2]

	imageVal := strings.TrimSpace(m.fmriFmriprepImage)
	if m.editingText && m.editingTextField == textFieldFmriFmriprepImage {
		imageVal = m.textBuffer + "█"
	}
	if imageVal == "" {
		imageVal = "(from config)"
	}

	spacesVal := strings.TrimSpace(m.fmriOutputSpacesSpec)
	if m.editingText && m.editingTextField == textFieldFmriOutputSpaces {
		spacesVal = m.textBuffer + "█"
	}
	if spacesVal == "" {
		spacesVal = "(default)"
	}

	ignoreVal := strings.TrimSpace(m.fmriIgnoreSpec)
	if m.editingText && m.editingTextField == textFieldFmriIgnore {
		ignoreVal = m.textBuffer + "█"
	}
	if ignoreVal == "" {
		ignoreVal = "(none)"
	}

	levelOptions := []string{"full", "resampling", "minimal"}
	levelVal := levelOptions[m.fmriLevelIndex%3]

	ciftiOptions := []string{"disabled", "91k", "170k"}
	ciftiVal := ciftiOptions[m.fmriCiftiOutputIndex%3]

	taskIdVal := strings.TrimSpace(m.fmriTaskId)
	if m.editingText && m.editingTextField == textFieldFmriTaskId {
		taskIdVal = m.textBuffer + "█"
	}
	if taskIdVal == "" {
		taskIdVal = "(all tasks)"
	}

	nthreadsVal := fmt.Sprintf("%d", m.fmriNThreads)
	if m.fmriNThreads == 0 {
		nthreadsVal = "(auto)"
	}
	ompNthreadsVal := fmt.Sprintf("%d", m.fmriOmpNThreads)
	if m.fmriOmpNThreads == 0 {
		ompNthreadsVal = "(auto)"
	}
	memVal := fmt.Sprintf("%d", m.fmriMemMb)
	if m.fmriMemMb == 0 {
		memVal = "(auto)"
	}

	skullStripTemplateVal := strings.TrimSpace(m.fmriSkullStripTemplate)
	if m.editingText && m.editingTextField == textFieldFmriSkullStripTemplate {
		skullStripTemplateVal = m.textBuffer + "█"
	}
	if skullStripTemplateVal == "" {
		skullStripTemplateVal = "OASIS30ANTs"
	}

	bold2t1wInitOptions := []string{"register", "header"}
	bold2t1wInitVal := bold2t1wInitOptions[m.fmriBold2T1wInitIndex%2]

	bold2t1wDofVal := fmt.Sprintf("%d", m.fmriBold2T1wDof)
	sliceTimeRefVal := fmt.Sprintf("%.2f", m.fmriSliceTimeRef)
	dummyScansVal := fmt.Sprintf("%d", m.fmriDummyScans)
	if m.fmriDummyScans == 0 {
		dummyScansVal = "(auto)"
	}

	fdSpikeVal := fmt.Sprintf("%.2f", m.fmriFdSpikeThreshold)
	dvarsSpikeVal := fmt.Sprintf("%.2f", m.fmriDvarsSpikeThreshold)

	randomSeedVal := fmt.Sprintf("%d", m.fmriRandomSeed)
	if m.fmriRandomSeed == 0 {
		randomSeedVal = "(random)"
	}

	extraArgsVal := strings.TrimSpace(m.fmriExtraArgs)
	if m.editingText && m.editingTextField == textFieldFmriExtraArgs {
		extraArgsVal = m.textBuffer + "█"
	}
	if extraArgsVal == "" {
		extraArgsVal = "(none)"
	}

	// Handle number editing
	if m.editingNumber {
		buffer := m.numberBuffer + "█"
		switch {
		case m.isCurrentlyEditing(optFmriNThreads):
			nthreadsVal = buffer
		case m.isCurrentlyEditing(optFmriOmpNThreads):
			ompNthreadsVal = buffer
		case m.isCurrentlyEditing(optFmriMemMb):
			memVal = buffer
		case m.isCurrentlyEditing(optFmriBold2T1wDof):
			bold2t1wDofVal = buffer
		case m.isCurrentlyEditing(optFmriSliceTimeRef):
			sliceTimeRefVal = buffer
		case m.isCurrentlyEditing(optFmriDummyScans):
			dummyScansVal = buffer
		case m.isCurrentlyEditing(optFmriFdSpikeThreshold):
			fdSpikeVal = buffer
		case m.isCurrentlyEditing(optFmriDvarsSpikeThreshold):
			dvarsSpikeVal = buffer
		case m.isCurrentlyEditing(optFmriRandomSeed):
			randomSeedVal = buffer
		}
	}

	options := m.getFmriPreprocessingOptions()

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
		case optConfigSetOverrides:
			label = "Config Overrides"
			value = strings.TrimSpace(m.configSetOverrides)
			if m.editingText && m.editingTextField == textFieldConfigSetOverrides {
				value = m.textBuffer + "█"
			} else if value == "" {
				value = "(none)"
			}
			hint = "Advanced/uncommon keys: key=value;key2=value2 (emits repeated --set)"

		// Group headers with chevron indicators
		case optFmriGroupRuntime:
			if m.fmriGroupRuntimeExpanded {
				label = "▾ Runtime"
			} else {
				label = "▸ Runtime"
			}
			hint = "Container settings"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		case optFmriGroupOutput:
			if m.fmriGroupOutputExpanded {
				label = "▾ Output"
			} else {
				label = "▸ Output"
			}
			hint = "Output spaces, formats"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		case optFmriGroupPerformance:
			if m.fmriGroupPerformanceExpanded {
				label = "▾ Performance"
			} else {
				label = "▸ Performance"
			}
			hint = "Threads, memory"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		case optFmriGroupAnatomical:
			if m.fmriGroupAnatomicalExpanded {
				label = "▾ Anatomical"
			} else {
				label = "▸ Anatomical"
			}
			hint = "FreeSurfer, skull-strip"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		case optFmriGroupBold:
			if m.fmriGroupBoldExpanded {
				label = "▾ BOLD Processing"
			} else {
				label = "▸ BOLD Processing"
			}
			hint = "Registration, timing"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		case optFmriGroupQc:
			if m.fmriGroupQcExpanded {
				label = "▾ Quality Control"
			} else {
				label = "▸ Quality Control"
			}
			hint = "Motion thresholds"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		case optFmriGroupDenoising:
			if m.fmriGroupDenoisingExpanded {
				label = "▾ Denoising"
			} else {
				label = "▸ Denoising"
			}
			hint = "ICA-AROMA"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		case optFmriGroupSurface:
			if m.fmriGroupSurfaceExpanded {
				label = "▾ Surface"
			} else {
				label = "▸ Surface"
			}
			hint = "Cortical surface options"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		case optFmriGroupMultiecho:
			if m.fmriGroupMultiechoExpanded {
				label = "▾ Multi-Echo"
			} else {
				label = "▸ Multi-Echo"
			}
			hint = "Multi-echo BOLD"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		case optFmriGroupRepro:
			if m.fmriGroupReproExpanded {
				label = "▾ Reproducibility"
			} else {
				label = "▸ Reproducibility"
			}
			hint = "Random seeds"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		case optFmriGroupValidation:
			if m.fmriGroupValidationExpanded {
				label = "▾ Validation"
			} else {
				label = "▸ Validation"
			}
			hint = "BIDS validation, errors"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		case optFmriGroupAdvanced:
			if m.fmriGroupAdvancedExpanded {
				label = "▾ Advanced"
			} else {
				label = "▸ Advanced"
			}
			hint = "Extra CLI arguments"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}

		// Runtime options (indented)
		case optFmriEngine:
			label = "Engine"
			value = engineVal
			hint = "docker/apptainer"
		case optFmriFmriprepImage:
			label = "Image"
			value = imageVal
			hint = "Container image"

		// Output options (indented)
		case optFmriOutputSpaces:
			label = "Output Spaces"
			value = spacesVal
			hint = "e.g., T1w MNI152..."
		case optFmriIgnore:
			label = "Ignore"
			value = ignoreVal
			hint = "fieldmaps, slicetiming"
		case optFmriLevel:
			label = "Level"
			value = levelVal
			hint = "full/resampling/minimal"
		case optFmriCiftiOutput:
			label = "CIFTI Output"
			value = ciftiVal
			hint = "91k/170k grayordinates"
		case optFmriTaskId:
			label = "Task ID"
			value = taskIdVal
			hint = "Specific task only"

		// Performance options (indented)
		case optFmriNThreads:
			label = "N Threads"
			value = nthreadsVal
			hint = "Max threads (0=auto)"
		case optFmriOmpNThreads:
			label = "OMP Threads"
			value = ompNthreadsVal
			hint = "Per process (0=auto)"
		case optFmriMemMb:
			label = "Mem (MB)"
			value = memVal
			hint = "Memory limit"
		case optFmriLowMem:
			label = "Low Memory"
			value = m.boolToOnOff(m.fmriLowMem)
			hint = "Reduce memory usage"

		// Anatomical options (indented)
		case optFmriSkipReconstruction:
			label = "Skip Recon-All"
			value = m.boolToOnOff(m.fmriSkipReconstruction)
			hint = "No FreeSurfer"
		case optFmriLongitudinal:
			label = "Longitudinal"
			value = m.boolToOnOff(m.fmriLongitudinal)
			hint = "Unbiased template"
		case optFmriSkullStripTemplate:
			label = "Skull Strip Tpl"
			value = skullStripTemplateVal
			hint = "Brain extraction"
		case optFmriSkullStripFixedSeed:
			label = "Fixed Seed"
			value = m.boolToOnOff(m.fmriSkullStripFixedSeed)
			hint = "Reproducible strip"

		// BOLD options (indented)
		case optFmriBold2T1wInit:
			label = "BOLD→T1w Init"
			value = bold2t1wInitVal
			hint = "register/header"
		case optFmriBold2T1wDof:
			label = "BOLD→T1w DOF"
			value = bold2t1wDofVal
			hint = "Degrees freedom"
		case optFmriSliceTimeRef:
			label = "Slice Time Ref"
			value = sliceTimeRefVal
			hint = "0=start, 0.5=mid, 1=end"
		case optFmriDummyScans:
			label = "Dummy Scans"
			value = dummyScansVal
			hint = "Non-steady vols"

		// QC options (indented)
		case optFmriFdSpikeThreshold:
			label = "FD Threshold"
			value = fdSpikeVal
			hint = "mm"
		case optFmriDvarsSpikeThreshold:
			label = "DVARS Threshold"
			value = dvarsSpikeVal
			hint = "Standardized"

		// Denoising options (indented)
		case optFmriUseAroma:
			label = "Use AROMA"
			value = m.boolToOnOff(m.fmriUseAroma)
			hint = "ICA-AROMA"

		// Surface options (indented)
		case optFmriMedialSurfaceNan:
			label = "Medial NaN"
			value = m.boolToOnOff(m.fmriMedialSurfaceNan)
			hint = "Fill medial wall"
		case optFmriNoMsm:
			label = "No MSM"
			value = m.boolToOnOff(m.fmriNoMsm)
			hint = "Disable MSM-Sulc"

		// Multi-echo options (indented)
		case optFmriMeOutputEchos:
			label = "Output Echos"
			value = m.boolToOnOff(m.fmriMeOutputEchos)
			hint = "Each echo separate"

		// Reproducibility options (indented)
		case optFmriRandomSeed:
			label = "Random Seed"
			value = randomSeedVal
			hint = "0=non-deterministic"

		// Validation options (indented)
		case optFmriSkipBidsValidation:
			label = "Skip Validation"
			value = m.boolToOnOff(m.fmriSkipBidsValidation)
			hint = "Skip bids-validator"
		case optFmriStopOnFirstCrash:
			label = "Stop on Crash"
			value = m.boolToOnOff(m.fmriStopOnFirstCrash)
			hint = "Abort on error"
		case optFmriCleanWorkdir:
			label = "Clean Workdir"
			value = m.boolToOnOff(m.fmriCleanWorkdir)
			hint = "Remove on success"

		// Advanced options (indented)
		case optFmriExtraArgs:
			label = "Extra Args"
			value = extraArgsVal
			hint = "Raw CLI args"
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
