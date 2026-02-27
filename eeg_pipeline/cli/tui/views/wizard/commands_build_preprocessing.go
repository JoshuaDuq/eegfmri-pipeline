package wizard

import (
	"fmt"
	"strings"
)

// Preprocessing pipeline advanced argument builder.

func (m Model) buildPreprocessingAdvancedArgs() []string {
	var args []string

	if !m.prepUsePyprep {
		args = append(args, "--no-pyprep")
	}
	if !m.prepUseIcalabel {
		args = append(args, "--no-icalabel")
	}
	if m.prepNJobs != 1 {
		args = append(args, "--n-jobs", fmt.Sprintf("%d", m.prepNJobs))
	}
	if strings.TrimSpace(m.prepMontage) != "" && m.prepMontage != "easycap-M1" {
		args = append(args, "--montage", m.prepMontage)
	}
	if strings.TrimSpace(m.prepChTypes) != "" && m.prepChTypes != "eeg" {
		args = append(args, "--ch-types", m.prepChTypes)
	}
	if strings.TrimSpace(m.prepEegReference) != "" && m.prepEegReference != "average" {
		args = append(args, "--eeg-reference", m.prepEegReference)
	}
	if strings.TrimSpace(m.prepEogChannels) != "" {
		args = append(args, "--eog-channels", m.prepEogChannels)
	}
	if m.prepRandomState != 42 {
		args = append(args, "--random-state", fmt.Sprintf("%d", m.prepRandomState))
	}
	if m.prepTaskIsRest {
		args = append(args, "--task-is-rest")
	}

	// Filtering
	if m.prepResample != 500 {
		args = append(args, "--resample", fmt.Sprintf("%d", m.prepResample))
	}
	if m.prepLFreq != 0.1 {
		args = append(args, "--l-freq", fmt.Sprintf("%.1f", m.prepLFreq))
	}
	if m.prepHFreq != 100.0 {
		args = append(args, "--h-freq", fmt.Sprintf("%.1f", m.prepHFreq))
	}
	if m.prepNotch != 60 {
		args = append(args, "--notch", fmt.Sprintf("%d", m.prepNotch))
	}
	if m.prepLineFreq != 0 && m.prepLineFreq != 60 {
		args = append(args, "--line-freq", fmt.Sprintf("%d", m.prepLineFreq))
	}
	if !m.prepFindBreaks {
		args = append(args, "--no-find-breaks")
	}

	// ICA
	if m.prepSpatialFilter != 0 {
		spatialFilterVal := []string{"ica", "ssp"}[m.prepSpatialFilter]
		args = append(args, "--spatial-filter", spatialFilterVal)
	}
	if m.prepICAAlgorithm != 0 {
		icaMethodVal := []string{"extended_infomax", "fastica", "infomax", "picard"}[m.prepICAAlgorithm]
		args = append(args, "--ica-method", icaMethodVal)
	}
	if m.prepICAComp != 0.99 {
		args = append(args, "--ica-components", fmt.Sprintf("%.2f", m.prepICAComp))
	}
	if m.prepICALFreq != 1.0 {
		args = append(args, "--ica-l-freq", fmt.Sprintf("%.1f", m.prepICALFreq))
	}
	if m.prepICARejThresh != 500.0 {
		args = append(args, "--ica-reject", fmt.Sprintf("%.0f", m.prepICARejThresh))
	}
	if m.prepProbThresh != 0.8 {
		args = append(args, "--prob-threshold", fmt.Sprintf("%.1f", m.prepProbThresh))
	}
	if strings.TrimSpace(m.icaLabelsToKeep) != "" && m.icaLabelsToKeep != "brain,other" {
		args = append(args, "--ica-labels-to-keep")
		args = append(args, splitCSVList(m.icaLabelsToKeep)...)
	}
	if m.prepKeepMnebidsBads {
		args = append(args, "--keep-mnebids-bads")
	}

	// PyPREP advanced options
	if !m.prepRansac {
		args = append(args, "--no-ransac")
	}
	if m.prepRepeats != 3 {
		args = append(args, "--repeats", fmt.Sprintf("%d", m.prepRepeats))
	}
	if m.prepAverageReref {
		args = append(args, "--average-reref")
	}
	if strings.TrimSpace(m.prepFileExtension) != "" && m.prepFileExtension != ".vhdr" {
		args = append(args, "--file-extension", m.prepFileExtension)
	}
	if m.prepConsiderPreviousBads {
		args = append(args, "--consider-previous-bads")
	} else {
		args = append(args, "--no-consider-previous-bads")
	}
	if !m.prepOverwriteChansTsv {
		args = append(args, "--no-overwrite-channels-tsv")
	}
	if m.prepDeleteBreaks {
		args = append(args, "--delete-breaks")
	}
	if m.prepBreaksMinLength != 20 {
		args = append(args, "--breaks-min-length", fmt.Sprintf("%d", m.prepBreaksMinLength))
	}
	if m.prepTStartAfterPrevious != 2 {
		args = append(args, "--t-start-after-previous", fmt.Sprintf("%d", m.prepTStartAfterPrevious))
	}
	if m.prepTStopBeforeNext != 2 {
		args = append(args, "--t-stop-before-next", fmt.Sprintf("%d", m.prepTStopBeforeNext))
	}
	if strings.TrimSpace(m.prepRenameAnotDict) != "" {
		args = append(args, "--rename-anot-dict", m.prepRenameAnotDict)
	}
	if strings.TrimSpace(m.prepCustomBadDict) != "" {
		args = append(args, "--custom-bad-dict", m.prepCustomBadDict)
	}

	// Epoching
	if strings.TrimSpace(m.prepConditions) != "" {
		args = append(args, "--conditions", m.prepConditions)
	}
	if m.prepEpochsTmin != -5.0 {
		args = append(args, "--tmin", fmt.Sprintf("%.1f", m.prepEpochsTmin))
	}
	if m.prepEpochsTmax != 15.0 {
		args = append(args, "--tmax", fmt.Sprintf("%.1f", m.prepEpochsTmax))
	}
	if m.prepEpochsNoBaseline {
		args = append(args, "--no-baseline")
	} else if m.prepEpochsBaselineStart != -2.0 || m.prepEpochsBaselineEnd != 0.0 {
		args = append(args, "--baseline", fmt.Sprintf("%.2f", m.prepEpochsBaselineStart), fmt.Sprintf("%.2f", m.prepEpochsBaselineEnd))
	}
	if m.prepEpochsReject > 0 {
		args = append(args, "--reject", fmt.Sprintf("%.0f", m.prepEpochsReject))
	}
	if m.prepRejectMethod != 1 {
		rejectMethodVal := []string{"none", "autoreject_local", "autoreject_global"}[m.prepRejectMethod]
		args = append(args, "--reject-method", rejectMethodVal)
	}

	// Clean events.tsv options
	if !m.prepWriteCleanEvents {
		args = append(args, "--no-write-clean-events")
	}
	if !m.prepOverwriteCleanEvents {
		args = append(args, "--no-overwrite-clean-events")
	}
	if !m.prepCleanEventsStrict {
		args = append(args, "--no-clean-events-strict")
	}

	// ECG channels
	if strings.TrimSpace(m.prepEcgChannels) != "" {
		args = append(args, "--ecg-channels", m.prepEcgChannels)
	}

	// Autoreject n_interpolate
	if strings.TrimSpace(m.prepAutorejectNInterpolate) != "" {
		args = append(args, "--autoreject-n-interpolate")
		args = append(args, splitCSVList(m.prepAutorejectNInterpolate)...)
	}

	// Alignment
	if m.alignAllowMisalignedTrim {
		args = append(args, "--allow-misaligned-trim")
	}
	if m.alignMinAlignmentSamples != 5 {
		args = append(args, "--min-alignment-samples", fmt.Sprintf("%d", m.alignMinAlignmentSamples))
	}
	if m.alignTrimToFirstVolume {
		args = append(args, "--trim-to-first-volume")
	}
	refs := []string{"as_is", "first_volume", "scanner_trigger"}
	if m.alignFmriOnsetReference != 0 {
		args = append(args, "--fmri-onset-reference", refs[m.alignFmriOnsetReference%len(refs)])
	}

	// Event Column Mapping
	if strings.TrimSpace(m.eventColPredictor) != "" {
		args = append(args, "--event-col-predictor")
		args = append(args, splitCSVList(m.eventColPredictor)...)
	}
	if strings.TrimSpace(m.eventColOutcome) != "" {
		args = append(args, "--event-col-outcome")
		args = append(args, splitCSVList(m.eventColOutcome)...)
	}
	if strings.TrimSpace(m.eventColBinaryOutcome) != "" {
		args = append(args, "--event-col-binary-outcome")
		args = append(args, splitCSVList(m.eventColBinaryOutcome)...)
	}
	if strings.TrimSpace(m.eventColCondition) != "" {
		args = append(args, "--event-col-condition")
		args = append(args, splitCSVList(m.eventColCondition)...)
	}
	if strings.TrimSpace(m.eventColRequired) != "" {
		args = append(args, "--event-col-required")
		args = append(args, splitCSVList(m.eventColRequired)...)
	}
	if strings.TrimSpace(m.conditionPreferredPrefixes) != "" {
		args = append(args, "--condition-preferred-prefixes")
		args = append(args, splitCSVList(m.conditionPreferredPrefixes)...)
	}

	return args
}
