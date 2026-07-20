package wizard

// Preprocessing-stage advanced option builders.

func (m Model) getPreprocessingOptions() []optionType {
	mode := m.modeOptions[m.modeIndex]
	options := []optionType{optUseDefaults, optConfigSetOverrides}

	// General Settings group (montage, jobs, etc.)
	options = append(options, optPrepGroupGeneral)
	if m.prepGroupGeneralExpanded {
		options = append(options,
			optPrepMontage,
			optPrepChTypes,
			optPrepEegReference,
			optPrepEogChannels,
			optPrepEcgChannels,
			optPrepRandomState,
			optPrepTaskIsRest,
			optPrepNJobs,
			optPrepUsePyprep,
			optPrepUseIcalabel,
		)
	}

	// Filtering group
	if mode == "ica" || mode == "epochs" {
		options = append(options, optPrepGroupFiltering)
		if m.prepGroupFilteringExpanded {
			options = append(options,
				optPrepResample,
				optPrepLFreq,
				optPrepHFreq,
				optPrepNotch,
				optPrepLineFreq,
				optPrepFindBreaks,
			)
		}
	}

	// PyPREP Advanced group (part of bad channel detection if enabled)
	if mode == "bad-channels" && m.prepUsePyprep {
		options = append(options, optPrepGroupPyprep)
		if m.prepGroupPyprepExpanded {
			options = append(options,
				optPrepRansac,
				optPrepRepeats,
				optPrepAverageReref,
				optPrepFileExtension,
				optPrepBadChannelSyncPolicy,
				optPrepConsiderPreviousBads,
				optPrepOverwriteChansTsv,
				optPrepDeleteBreaks,
				optPrepBreaksMinLength,
				optPrepTStartAfterPrevious,
				optPrepTStopBeforeNext,
				optPrepRenameAnotDict,
				optPrepCustomBadDict,
			)
		}
	}

	// ICA group
	if mode == "ica" {
		options = append(options, optPrepGroupICA)
		if m.prepGroupICAExpanded {
			options = append(options,
				optPrepSpatialFilter,
				optPrepICAAlgorithm,
				optPrepICAComp,
				optPrepICALFreq,
				optPrepICARejThresh,
				optPrepProbThresh,
				optIcaLabelsToKeep,
			)
		}
	}

	// Epoching group
	if mode == "epochs" {
		options = append(options, optPrepGroupEpoching)
		if m.prepGroupEpochingExpanded {
			options = append(options,
				optPrepConditions,
				optPrepEpochsTmin,
				optPrepEpochsTmax,
				optPrepEpochsNoBaseline,
				optPrepEpochsBaseline,
				optPrepEpochsReject,
				optPrepRejectMethod,
				optPrepAutorejectNInterpolate,
				optPrepWriteCleanEvents,
				optPrepOverwriteCleanEvents,
				optPrepCleanEventsStrict,
				optPrepCleanEventsQCEnabled,
				optPrepCleanEventsQCEcgVarianceEnabled,
				optPrepCleanEventsQCEcgVarianceOutputColumn,
				optPrepCleanEventsQCEcgVarianceChannels,
				optPrepCleanEventsQCEcgVarianceWindow,
				optPrepCleanEventsQCPeripheralLowGammaEnabled,
				optPrepCleanEventsQCPeripheralLowGammaOutputColumn,
				optPrepCleanEventsQCPeripheralLowGammaChannels,
				optPrepCleanEventsQCPeripheralLowGammaBand,
				optPrepCleanEventsQCPeripheralLowGammaWindow,
			)
		}
	}

	return options
}
