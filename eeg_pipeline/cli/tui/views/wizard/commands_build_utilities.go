package wizard

import (
	"fmt"
	"strings"
)

func (m Model) buildRawToBidsAdvancedArgs() []string {
	var args []string

	if m.rawMontage != "" && m.rawMontage != "easycap-M1" {
		args = append(args, "--montage", m.rawMontage)
	}
	if m.rawLineFreq != 60 {
		args = append(args, "--line-freq", fmt.Sprintf("%d", m.rawLineFreq))
	}
	if m.rawOverwrite {
		args = append(args, "--overwrite")
	}
	if m.rawTrimToFirstVolume {
		args = append(args, "--trim-to-first-volume")
	}
	if m.rawEventPrefixes != "" {
		for _, prefix := range splitListInput(m.rawEventPrefixes) {
			args = append(args, "--event-prefix", prefix)
		}
	}
	if m.rawKeepAnnotations {
		args = append(args, "--keep-all-annotations")
	}

	return args
}

func (m Model) buildFmriRawToBidsAdvancedArgs() []string {
	var args []string

	if strings.TrimSpace(m.fmriRawSession) != "" {
		args = append(args, "--session", strings.TrimSpace(m.fmriRawSession))
	}
	if strings.TrimSpace(m.fmriRawRestTask) != "" && strings.TrimSpace(m.fmriRawRestTask) != "rest" {
		args = append(args, "--rest-task", strings.TrimSpace(m.fmriRawRestTask))
	}
	if !m.fmriRawIncludeRest {
		args = append(args, "--no-rest")
	}
	if !m.fmriRawIncludeFieldmaps {
		args = append(args, "--no-fieldmaps")
	}

	dicomMode := "symlink"
	switch m.fmriRawDicomModeIndex {
	case 1:
		dicomMode = "copy"
	case 2:
		dicomMode = "skip"
	}
	if dicomMode != "symlink" {
		args = append(args, "--dicom-mode", dicomMode)
	}

	if m.fmriRawOverwrite {
		args = append(args, "--overwrite")
	}
	if !m.fmriRawCreateEvents {
		args = append(args, "--no-events")
	}

	granularity := "phases"
	if m.fmriRawEventGranularity == 1 {
		granularity = "trial"
	}
	if granularity != "phases" {
		args = append(args, "--event-granularity", granularity)
	}

	onsetRef := "as_is"
	switch m.fmriRawOnsetRefIndex {
	case 1:
		onsetRef = "first_iti_start"
	case 2:
		onsetRef = "first_stim_start"
	}
	if onsetRef != "as_is" {
		args = append(args, "--onset-reference", onsetRef)
	}
	if m.fmriRawOnsetOffsetS != 0 {
		args = append(args, "--onset-offset-s", fmt.Sprintf("%.3f", m.fmriRawOnsetOffsetS))
	}
	if strings.TrimSpace(m.fmriRawDcm2niixPath) != "" {
		args = append(args, "--dcm2niix-path", strings.TrimSpace(m.fmriRawDcm2niixPath))
	}
	if strings.TrimSpace(m.fmriRawDcm2niixArgs) != "" {
		for _, tok := range splitListInput(m.fmriRawDcm2niixArgs) {
			args = append(args, "--dcm2niix-arg", tok)
		}
	}

	return args
}

// buildMergeBehaviorAdvancedArgs returns CLI args for merge-behavior advanced options
func (m Model) buildMergeBehaviorAdvancedArgs() []string {
	var args []string

	if m.mergeEventPrefixes != "" {
		for _, prefix := range splitListInput(m.mergeEventPrefixes) {
			args = append(args, "--event-prefix", prefix)
		}
	}
	if m.mergeEventTypes != "" {
		for _, eventType := range splitListInput(m.mergeEventTypes) {
			args = append(args, "--event-type", eventType)
		}
	}

	return args
}
