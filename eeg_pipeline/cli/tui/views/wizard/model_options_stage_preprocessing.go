package wizard

// Preprocessing-stage advanced option builders.

func (m Model) getPreprocessingOptions() []optionType {
	isFull := m.modeIndex == 0 || m.modeOptions[m.modeIndex] == "full"
	options := []optionType{optUseDefaults}

	// Stage Selection group (only show if not in full mode)
	if !isFull {
		options = append(options, optPrepGroupStages)
		if m.prepGroupStagesExpanded {
			options = append(options,
				optPrepStageBadChannels,
				optPrepStageFiltering,
				optPrepStageICA,
				optPrepStageEpoching,
			)
		}
	}

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
	if isFull || m.prepStageSelected[1] {
		options = append(options, optPrepGroupFiltering)
		if m.prepGroupFilteringExpanded {
			options = append(options,
				optPrepResample,
				optPrepLFreq,
				optPrepHFreq,
				optPrepNotch,
				optPrepLineFreq,
				optPrepZaplineFline,
				optPrepFindBreaks,
			)
		}
	}

	// PyPREP Advanced group (part of bad channel detection if enabled)
	if (isFull || m.prepStageSelected[0]) && m.prepUsePyprep {
		options = append(options, optPrepGroupPyprep)
		if m.prepGroupPyprepExpanded {
			options = append(options,
				optPrepRansac,
				optPrepRepeats,
				optPrepAverageReref,
				optPrepFileExtension,
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
	if isFull || m.prepStageSelected[2] {
		options = append(options, optPrepGroupICA)
		if m.prepGroupICAExpanded {
			options = append(options,
				optPrepSpatialFilter,
				optPrepICAAlgorithm,
				optPrepICAComp,
				optPrepICALFreq,
				optPrepICARejThresh,
				optPrepProbThresh,
				optPrepKeepMnebidsBads,
				optIcaLabelsToKeep,
			)
		}
	}

	// Epoching group
	if isFull || m.prepStageSelected[3] {
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
				optPrepRunSourceEstimation,
				optPrepWriteCleanEvents,
				optPrepOverwriteCleanEvents,
				optPrepCleanEventsStrict,
			)
		}
	}

	// Alignment group
	options = append(options,
		optAlignAllowMisalignedTrim,
		optAlignMinAlignmentSamples,
		optAlignTrimToFirstVolume,
		optAlignFmriOnsetReference,
	)

	// Event Column Mapping
	options = append(options,
		optEventColPredictor,
		optEventColRating,
		optEventColBinaryOutcome,
		optConditionPreferredPrefixes,
	)

	return options
}
