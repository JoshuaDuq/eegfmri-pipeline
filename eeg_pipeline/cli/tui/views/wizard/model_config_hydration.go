package wizard

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

func (m *Model) ApplyConfigKeys(values map[string]interface{}) {

	if rawBands, ok := values["time_frequency_analysis.bands"]; ok {
		bands := parseConfigBands(rawBands)
		if len(bands) > 0 {
			m.bands = bands
			m.bandSelected = make(map[int]bool)
			for i := range m.bands {
				m.bandSelected[i] = true
			}
			m.bandCursor = 0
		}
	}

	type keyBinder struct {
		key   string
		apply func(v interface{})
	}
	binders := []keyBinder{
		{
			key: "paths.bids_fmri_root",
			apply: func(v interface{}) {
				if s, ok := asString(v); ok && s != "" {
					m.bidsFmriRoot = s
				}
			},
		},
		{
			key: "fmri_preprocessing.engine",
			apply: func(v interface{}) {
				if s, ok := asString(v); ok {
					switch s {
					case "apptainer":
						m.fmriEngineIndex = 1
					case "docker":
						m.fmriEngineIndex = 0
					}
				}
			},
		},
		{key: "fmri_preprocessing.fmriprep.image", apply: func(v interface{}) {
			if s, ok := asString(v); ok && s != "" {
				m.fmriFmriprepImage = s
			}
		}},
		{key: "fmri_preprocessing.fmriprep.output_dir", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.fmriFmriprepOutputDir = s
			}
		}},
		{key: "fmri_preprocessing.fmriprep.work_dir", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.fmriFmriprepWorkDir = s
			}
		}},
		{key: "fmri_preprocessing.fmriprep.fs_license_file", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.fmriFreesurferLicenseFile = s
			}
		}},
		{key: "paths.freesurfer_license", apply: func(v interface{}) {
			if strings.TrimSpace(m.fmriFreesurferLicenseFile) == "" {
				if s, ok := asString(v); ok {
					m.fmriFreesurferLicenseFile = s
				}
			}
		}},
		{key: "fmri_preprocessing.fmriprep.fs_subjects_dir", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.fmriFreesurferSubjectsDir = s
			}
		}},
		{key: "fmri_preprocessing.fmriprep.output_spaces", apply: func(v interface{}) {
			if list, ok := asStringList(v); ok && len(list) > 0 {
				m.fmriOutputSpacesSpec = strings.Join(list, " ")
			}
		}},
		{key: "fmri_preprocessing.fmriprep.ignore", apply: func(v interface{}) {
			if list, ok := asStringList(v); ok && len(list) > 0 {
				m.fmriIgnoreSpec = strings.Join(list, " ")
			}
		}},
		{key: "fmri_preprocessing.fmriprep.bids_filter_file", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.fmriBidsFilterFile = s
			}
		}},
		{key: "fmri_preprocessing.fmriprep.extra_args", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.fmriExtraArgs = s
			}
		}},
		{key: "fmri_preprocessing.fmriprep.use_aroma", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.fmriUseAroma = b
			}
		}},
		{key: "fmri_preprocessing.fmriprep.skip_bids_validation", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.fmriSkipBidsValidation = b
			}
		}},
		{key: "fmri_preprocessing.fmriprep.stop_on_first_crash", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.fmriStopOnFirstCrash = b
			}
		}},
		{key: "fmri_preprocessing.fmriprep.clean_workdir", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.fmriCleanWorkdir = b
			}
		}},
		{key: "fmri_preprocessing.fmriprep.fs_no_reconall", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.fmriSkipReconstruction = b
			}
		}},
		{key: "fmri_preprocessing.fmriprep.mem_mb", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.fmriMemMb = n
			}
		}},
		{key: "fmri_preprocessing.fmriprep.nthreads", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.fmriNThreads = n
			}
		}},
		{key: "fmri_preprocessing.fmriprep.omp_nthreads", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.fmriOmpNThreads = n
			}
		}},
		{key: "event_columns.predictor", apply: func(v interface{}) {
			if list, ok := asStringList(v); ok {
				m.eventColPredictor = strings.Join(list, ",")
			}
		}},
		{key: "event_columns.outcome", apply: func(v interface{}) {
			if list, ok := asStringList(v); ok {
				m.eventColOutcome = strings.Join(list, ",")
			}
		}},
		{key: "event_columns.binary_outcome", apply: func(v interface{}) {
			if list, ok := asStringList(v); ok {
				m.eventColBinaryOutcome = strings.Join(list, ",")
			}
		}},
		{key: "event_columns.condition", apply: func(v interface{}) {
			if list, ok := asStringList(v); ok {
				m.eventColCondition = strings.Join(list, ",")
			}
		}},
		{key: "event_columns.required", apply: func(v interface{}) {
			if list, ok := asStringList(v); ok {
				m.eventColRequired = strings.Join(list, ",")
			}
		}},
		{key: "preprocessing.condition_preferred_prefixes", apply: func(v interface{}) {
			if list, ok := asStringList(v); ok {
				m.conditionPreferredPrefixes = strings.Join(list, ",")
			}
		}},
		// Behavior pipeline hydration (YAML -> TUI model)
		{key: "behavior_analysis.statistics.correlation_method", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "pearson") {
					m.correlationMethod = "pearson"
				} else {
					m.correlationMethod = "spearman"
				}
			}
		}},
		{key: "behavior_analysis.predictor_type", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(s) {
				case "continuous", "":
					m.predictorType = 0
				case "binary":
					m.predictorType = 1
				case "categorical":
					m.predictorType = 2
				}
			}
		}},
		{key: "behavior_analysis.robust_correlation", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(s) {
				case "", "none":
					m.robustCorrelation = 0
				case "percentage_bend":
					m.robustCorrelation = 1
				case "winsorized":
					m.robustCorrelation = 2
				case "shepherd":
					m.robustCorrelation = 3
				}
			}
		}},
		{key: "behavior_analysis.bootstrap", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.bootstrapSamples = n
			}
		}},
		{key: "behavior_analysis.cluster.n_permutations", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.nPermutations = n
			}
		}},
		{key: "project.random_state", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.rngSeed = n
			}
		}},
		{key: "behavior_analysis.n_jobs", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.behaviorNJobs = n
			}
		}},
		{key: "behavior_analysis.min_samples.default", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.behaviorMinSamples = n
			}
		}},
		{key: "behavior_analysis.outcome_column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.behaviorOutcomeColumn = s
			}
		}},
		{key: "behavior_analysis.predictor_column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.behaviorPredictorColumn = s
			}
		}},
		{key: "behavior_analysis.predictor_control_enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.controlPredictor = b
			}
		}},
		{key: "behavior_analysis.control_trial_order", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.controlTrialOrder = b
			}
		}},
		{key: "behavior_analysis.run_adjustment.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.runAdjustmentEnabled = b
			}
		}},
		{key: "behavior_analysis.run_adjustment.column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.runAdjustmentColumn = s
			}
		}},
		{key: "behavior_analysis.run_adjustment.include_in_correlations", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.runAdjustmentIncludeInCorrelations = b
			}
		}},
		{key: "behavior_analysis.run_adjustment.max_dummies", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.runAdjustmentMaxDummies = n
			}
		}},
		{key: "behavior_analysis.statistics.fdr_alpha", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.fdrAlpha = f
			}
		}},
		{key: "behavior_analysis.correlations.compute_change_scores", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.behaviorComputeChangeScores = b
			}
		}},
		{key: "behavior_analysis.correlations.loso_stability", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.behaviorComputeLosoStability = b
			}
		}},
		{key: "behavior_analysis.correlations.compute_bayes_factors", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.behaviorComputeBayesFactors = b
			}
		}},
		{key: "behavior_analysis.correlations.primary_unit", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "run_mean") || strings.EqualFold(s, "run") {
					m.correlationsPrimaryUnit = 1
				} else {
					m.correlationsPrimaryUnit = 0
				}
			}
		}},
		{key: "behavior_analysis.correlations.min_runs", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.correlationsMinRuns = n
			}
		}},
		{key: "behavior_analysis.correlations.types", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.correlationsTypesSpec = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "behavior_analysis.correlations.target_column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.correlationsTargetColumn = s
			}
		}},
		{key: "behavior_analysis.correlations.power_segment_preference", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.correlationsPowerSegment = strings.TrimSpace(s)
			}
		}},
		{key: "behavior_analysis.correlations.use_crossfit_predictor_residual", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.correlationsUseCrossfitResidual = b
			}
		}},
		{key: "behavior_analysis.correlations.prefer_predictor_residual", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.correlationsPreferPredictorResidual = b
			}
		}},
		{key: "behavior_analysis.correlations.p_primary_mode", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.correlationsPermutationPrimary = !strings.EqualFold(strings.TrimSpace(s), "asymptotic")
			}
		}},
		{key: "behavior_analysis.correlations.permutation.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.correlationsPermutationPrimary = b
			}
		}},
		{key: "behavior_analysis.correlations.permutation.n_permutations", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.correlationsPermutations = n
			}
		}},
		{key: "behavior_analysis.group_level.block_permutation", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.groupLevelBlockPermutation = b
			}
		}},
		{key: "behavior_analysis.group_level.multilevel_correlations.block_permutation", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.groupLevelBlockPermutation = b
			}
		}},
		{key: "behavior_analysis.group_level.multilevel_correlations.target", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.groupLevelTarget = strings.TrimSpace(s)
			}
		}},
		{key: "behavior_analysis.group_level.multilevel_correlations.control_predictor", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.groupLevelControlPredictor = b
			}
		}},
		{key: "behavior_analysis.group_level.multilevel_correlations.control_trial_order", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.groupLevelControlTrialOrder = b
			}
		}},
		{key: "behavior_analysis.group_level.multilevel_correlations.control_run_effects", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.groupLevelControlRunEffects = b
			}
		}},
		{key: "behavior_analysis.group_level.multilevel_correlations.max_run_dummies", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.groupLevelMaxRunDummies = n
			}
		}},
		{key: "behavior_analysis.condition.compare_values", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.conditionCompareValues = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "behavior_analysis.condition.compare_labels", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.conditionCompareLabels = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "behavior_analysis.condition.compare_column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.conditionCompareColumn = strings.TrimSpace(s)
			}
		}},
		{key: "behavior_analysis.condition.fail_fast", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.conditionFailFast = b
			}
		}},
		{key: "behavior_analysis.condition.p_primary_mode", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.conditionPermutationPrimary = !strings.EqualFold(strings.TrimSpace(s), "asymptotic")
			}
		}},
		{key: "behavior_analysis.condition.permutation.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.conditionPermutationPrimary = b
			}
		}},
		{key: "behavior_analysis.condition.overwrite", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.conditionOverwrite = b
			}
		}},
		{key: "behavior_analysis.condition.primary_unit", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "run_mean") || strings.EqualFold(s, "run") {
					m.conditionPrimaryUnit = 1
				} else {
					m.conditionPrimaryUnit = 0
				}
			}
		}},
		{key: "behavior_analysis.regression.primary_unit", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "run_mean") || strings.EqualFold(s, "run") {
					m.regressionPrimaryUnit = 1
				} else {
					m.regressionPrimaryUnit = 0
				}
			}
		}},
		{key: "behavior_analysis.regression.n_permutations", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.regressionPermutations = n
			}
		}},
		{key: "behavior_analysis.temporal.target_column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.temporalTargetColumn = strings.TrimSpace(s)
			}
		}},
		{key: "behavior_analysis.temporal.correction_method", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "cluster") {
					m.temporalCorrectionMethod = 1
				} else {
					m.temporalCorrectionMethod = 0
				}
			}
		}},
		{key: "behavior_analysis.trial_table.format", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "tsv") {
					m.trialTableFormat = 1
				} else {
					m.trialTableFormat = 0
				}
			}
		}},
		{key: "behavior_analysis.trial_order.max_missing_fraction", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.trialOrderMaxMissingFraction = f
			}
		}},
		{key: "behavior_analysis.predictor_residual.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.predictorResidualEnabled = b
			}
		}},
		{key: "behavior_analysis.predictor_residual.method", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "poly") {
					m.predictorResidualMethod = 1
				} else {
					m.predictorResidualMethod = 0
				}
			}
		}},
		{key: "behavior_analysis.predictor_residual.min_samples", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.predictorResidualMinSamples = n
			}
		}},
		{key: "behavior_analysis.predictor_residual.spline_df_candidates", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.predictorResidualSplineDfCandidates = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "behavior_analysis.predictor_residual.poly_degree", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.predictorResidualPolyDegree = n
			}
		}},
		{key: "behavior_analysis.predictor_residual.crossfit.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.predictorResidualCrossfitEnabled = b
			}
		}},
		{key: "behavior_analysis.predictor_residual.crossfit.group_column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.predictorResidualCrossfitGroupColumn = s
			}
		}},
		{key: "behavior_analysis.predictor_residual.crossfit.n_splits", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.predictorResidualCrossfitNSplits = n
			}
		}},
		{key: "behavior_analysis.predictor_residual.crossfit.method", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "poly") {
					m.predictorResidualCrossfitMethod = 1
				} else {
					m.predictorResidualCrossfitMethod = 0
				}
			}
		}},
		{key: "behavior_analysis.predictor_residual.crossfit.spline_n_knots", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.predictorResidualCrossfitSplineKnots = n
			}
		}},
		{key: "behavior_analysis.statistics.predictor_control", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(s) {
				case "linear":
					m.behaviorStatsPredictorControl = 1
				case "none":
					m.behaviorStatsPredictorControl = 2
				default:
					m.behaviorStatsPredictorControl = 0
				}
			}
		}},
		{key: "behavior_analysis.statistics.allow_iid_trials", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.behaviorStatsAllowIIDTrials = b
			}
		}},
		{key: "behavior_analysis.statistics.hierarchical_fdr", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.behaviorStatsHierarchicalFDR = b
			}
		}},
		{key: "behavior_analysis.statistics.compute_reliability", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.behaviorStatsComputeReliability = b
			}
		}},
		{key: "behavior_analysis.permutation.scheme", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "circular_shift") {
					m.behaviorPermScheme = 1
				} else {
					m.behaviorPermScheme = 0
				}
			}
		}},
		{key: "behavior_analysis.permutation.group_column_preference", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.behaviorPermGroupColumnPreference = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "behavior_analysis.features.exclude_non_trialwise_features", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.behaviorExcludeNonTrialwiseFeatures = b
			}
		}},
		{key: "behavior_analysis.icc.unit_columns", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.iccUnitColumns = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "behavior_analysis.trial_table.disallow_positional_alignment", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.trialTableDisallowPositionalAlignment = b
			}
		}},
		{key: "statistics.alpha", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.statisticsAlpha = f
			}
		}},
		{key: "behavior_analysis.feature_registry.files", apply: func(v interface{}) {
			if s, ok := asCompactJSON(v); ok {
				m.behaviorFeatureRegistryFilesJSON = s
			}
		}},
		{key: "behavior_analysis.feature_registry.source_to_feature_type", apply: func(v interface{}) {
			if s, ok := asCompactJSON(v); ok {
				m.behaviorFeatureRegistrySourceJSON = s
			}
		}},
		{key: "behavior_analysis.feature_registry.feature_type_hierarchy", apply: func(v interface{}) {
			if s, ok := asCompactJSON(v); ok {
				m.behaviorFeatureRegistryHierarchyJSON = s
			}
		}},
		{key: "behavior_analysis.feature_registry.feature_patterns", apply: func(v interface{}) {
			if s, ok := asCompactJSON(v); ok {
				m.behaviorFeatureRegistryPatternsJSON = s
			}
		}},
		{key: "behavior_analysis.feature_registry.feature_classifiers", apply: func(v interface{}) {
			if s, ok := asCompactJSON(v); ok {
				m.behaviorFeatureRegistryClassifiersJSON = s
			}
		}},
		// RNG seed from project config
		{key: "project.random_state", apply: func(v interface{}) {
			if n, ok := asInt(v); ok && m.rngSeed == 0 {
				m.rngSeed = n
			}
		}},
		// Global permutations (statistics-level)
		{key: "behavior_analysis.statistics.n_permutations", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.nPermutations = n
			}
		}},
		// Output options
		{key: "behavior_analysis.output.also_save_csv", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.alsoSaveCsv = b
			}
		}},
		{key: "behavior_analysis.output.overwrite", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.behaviorOverwrite = b
			}
		}},
		// Regression — full set
		{key: "behavior_analysis.regression.outcome", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "predictor_residual":
					m.regressionOutcome = 1
				case "predictor":
					m.regressionOutcome = 2
				default:
					m.regressionOutcome = 0
				}
			}
		}},
		{key: "behavior_analysis.regression.include_predictor", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.regressionIncludePredictor = b
			}
		}},
		{key: "behavior_analysis.regression.predictor_control", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "outcome_hat":
					m.regressionPredictorControl = 1
				case "spline":
					m.regressionPredictorControl = 2
				default:
					m.regressionPredictorControl = 0
				}
			}
		}},
		{key: "behavior_analysis.regression.predictor_spline.n_knots", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.regressionPredictorSplineKnots = n
			}
		}},
		{key: "behavior_analysis.regression.predictor_spline.quantile_low", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.regressionPredictorSplineQlow = f
			}
		}},
		{key: "behavior_analysis.regression.predictor_spline.quantile_high", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.regressionPredictorSplineQhigh = f
			}
		}},
		{key: "behavior_analysis.regression.predictor_spline.min_samples", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.regressionPredictorSplineMinN = n
			}
		}},
		{key: "behavior_analysis.regression.include_trial_order", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.regressionIncludeTrialOrder = b
			}
		}},
		{key: "behavior_analysis.regression.include_prev_terms", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.regressionIncludePrev = b
			}
		}},
		{key: "behavior_analysis.regression.include_run_block", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.regressionIncludeRunBlock = b
			}
		}},
		{key: "behavior_analysis.regression.include_interaction", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.regressionIncludeInteraction = b
			}
		}},
		{key: "behavior_analysis.regression.standardize", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.regressionStandardize = b
			}
		}},
		{key: "behavior_analysis.regression.min_samples", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.regressionMinSamples = n
			}
		}},
		{key: "behavior_analysis.regression.max_features", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.regressionMaxFeatures = n
			}
		}},
		// Condition — missing fields
		{key: "behavior_analysis.condition.min_trials_per_condition", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.conditionMinTrials = n
			}
		}},
		{key: "behavior_analysis.condition.effect_size_threshold", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.conditionEffectThreshold = f
			}
		}},
		// Temporal — full set
		{key: "behavior_analysis.temporal.time_resolution_ms", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.temporalResolutionMs = n
			}
		}},
		{key: "behavior_analysis.temporal.smooth_window_ms", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.temporalSmoothMs = n
			}
		}},
		{key: "behavior_analysis.temporal.time_range_ms", apply: func(v interface{}) {
			raw, ok := v.([]interface{})
			if !ok || len(raw) != 2 {
				return
			}
			if lo, ok := asInt(raw[0]); ok {
				m.temporalTimeMinMs = lo
			}
			if hi, ok := asInt(raw[1]); ok {
				m.temporalTimeMaxMs = hi
			}
		}},
		{key: "behavior_analysis.temporal.split_by_condition", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.temporalSplitByCondition = b
			}
		}},
		{key: "behavior_analysis.temporal.condition_column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.temporalConditionColumn = strings.TrimSpace(s)
			}
		}},
		{key: "behavior_analysis.temporal.condition_values", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.temporalConditionValues = strings.Join(splitLooseList(spec), " ")
			}
		}},
		{key: "behavior_analysis.temporal.include_roi_averages", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.temporalIncludeROIAverages = b
			}
		}},
		{key: "behavior_analysis.temporal.include_tf_grid", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.temporalIncludeTFGrid = b
			}
		}},
		{key: "behavior_analysis.temporal.features.power", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.temporalFeaturePowerEnabled = b
			}
		}},
		{key: "behavior_analysis.temporal.features.itpc", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.temporalFeatureITPCEnabled = b
			}
		}},
		{key: "behavior_analysis.temporal.features.erds", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.temporalFeatureERDSEnabled = b
			}
		}},
		{key: "behavior_analysis.temporal.itpc.baseline_correction", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.temporalITPCBaselineCorrection = b
			}
		}},
		{key: "behavior_analysis.temporal.itpc.baseline_window", apply: func(v interface{}) {
			raw, ok := v.([]interface{})
			if !ok || len(raw) != 2 {
				return
			}
			if lo, ok := asFloat(raw[0]); ok {
				m.temporalITPCBaselineMin = lo
			}
			if hi, ok := asFloat(raw[1]); ok {
				m.temporalITPCBaselineMax = hi
			}
		}},
		{key: "behavior_analysis.temporal.erds.method", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(strings.TrimSpace(s), "zscore") {
					m.temporalERDSMethod = 1
				} else {
					m.temporalERDSMethod = 0
				}
			}
		}},
		{key: "behavior_analysis.temporal.erds.baseline_window", apply: func(v interface{}) {
			raw, ok := v.([]interface{})
			if !ok || len(raw) != 2 {
				return
			}
			if lo, ok := asFloat(raw[0]); ok {
				m.temporalERDSBaselineMin = lo
			}
			if hi, ok := asFloat(raw[1]); ok {
				m.temporalERDSBaselineMax = hi
			}
		}},
		// Cluster — full set
		{key: "behavior_analysis.cluster.forming_threshold", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.clusterThreshold = f
			}
		}},
		{key: "behavior_analysis.cluster.min_cluster_size", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.clusterMinSize = n
			}
		}},
		{key: "behavior_analysis.cluster.tail", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.clusterTail = n
			}
		}},
		{key: "behavior_analysis.cluster.condition_column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.clusterConditionColumn = strings.TrimSpace(s)
			}
		}},
		{key: "behavior_analysis.cluster.condition_values", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.clusterConditionValues = strings.Join(splitLooseList(spec), " ")
			}
		}},
		{key: "feature_engineering.sourcelocalization.subjects_dir", apply: func(v interface{}) {
			if s, ok := asString(v); ok && strings.TrimSpace(s) != "" {
				m.sourceLocSubjectsDir = s
				m.plotSourceSubjectsDir = s
			}
		}},
		{key: "plotting.plots.features.sourcelocalization.hemi", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.plotSourceHemi = s
			}
		}},
		{key: "plotting.plots.features.sourcelocalization.views", apply: func(v interface{}) {
			if list, ok := asStringList(v); ok && len(list) > 0 {
				m.plotSourceViews = strings.Join(list, " ")
			}
		}},
		{key: "plotting.plots.features.sourcelocalization.cortex", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.plotSourceCortex = s
			}
		}},
		{key: "plotting.plots.features.sourcelocalization.subjects_dir", apply: func(v interface{}) {
			if s, ok := asString(v); ok && strings.TrimSpace(s) != "" {
				m.plotSourceSubjectsDir = s
			}
		}},
		{key: "plotting.comparisons.compare_windows", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.plotCompareWindows = &b
			}
		}},
		{key: "plotting.comparisons.comparison_windows", apply: func(v interface{}) {
			if list, ok := asStringList(v); ok && len(list) > 0 {
				m.plotComparisonWindowsSpec = strings.Join(list, " ")
			}
		}},
		{key: "plotting.comparisons.compare_columns", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.plotCompareColumns = &b
			}
		}},
		{key: "plotting.comparisons.comparison_segment", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.plotComparisonSegment = s
			}
		}},
		{key: "plotting.comparisons.comparison_column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.plotComparisonColumn = s
			}
		}},
		{key: "plotting.comparisons.comparison_values", apply: func(v interface{}) {
			if list, ok := asStringList(v); ok && len(list) > 0 {
				m.plotComparisonValuesSpec = strings.Join(list, " ")
			}
		}},
		{key: "plotting.comparisons.comparison_labels", apply: func(v interface{}) {
			if list, ok := asStringList(v); ok && len(list) > 0 {
				m.plotComparisonLabelsSpec = strings.Join(list, " ")
			}
		}},
		{key: "plotting.comparisons.comparison_rois", apply: func(v interface{}) {
			if list, ok := asStringList(v); ok && len(list) > 0 {
				m.plotComparisonROIsSpec = strings.Join(list, " ")
			}
		}},
		{key: "plotting.overwrite", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.plotOverwrite = &b
			}
		}},
		{key: "feature_engineering.sourcelocalization.mode", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "fmri_informed":
					m.sourceLocMode = 1
				default:
					m.sourceLocMode = 0
				}
			}
		}},
		{key: "feature_engineering.sourcelocalization.method", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "eloreta":
					m.sourceLocMethod = 1
				default:
					m.sourceLocMethod = 0
				}
			}
		}},
		{key: "feature_engineering.sourcelocalization.spacing", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "oct5":
					m.sourceLocSpacing = 0
				case "oct6":
					m.sourceLocSpacing = 1
				case "ico4":
					m.sourceLocSpacing = 2
				case "ico5":
					m.sourceLocSpacing = 3
				}
			}
		}},
		{key: "feature_engineering.sourcelocalization.parcellation", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "aparc":
					m.sourceLocParc = 0
				case "aparc.a2009s":
					m.sourceLocParc = 1
				case "hcpmmp1":
					m.sourceLocParc = 2
				}
			}
		}},
		{key: "feature_engineering.sourcelocalization.connectivity_method", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "aec":
					m.sourceLocConnMethod = 0
				case "wpli":
					m.sourceLocConnMethod = 1
				case "plv":
					m.sourceLocConnMethod = 2
				}
			}
		}},
		{key: "feature_engineering.sourcelocalization.subject", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocSubject = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.trans", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocTrans = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.bem", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocBem = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.mindist_mm", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.sourceLocMindistMm = f
			}
		}},
		{key: "feature_engineering.sourcelocalization.contrast.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.sourceLocContrastEnabled = b
			}
		}},
		{key: "feature_engineering.sourcelocalization.contrast.condition_column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocContrastCondition = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.contrast.condition_a", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocContrastA = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.contrast.condition_b", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocContrastB = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.contrast.min_trials_per_condition", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.sourceLocContrastMinTrials = n
			}
		}},
		{key: "feature_engineering.sourcelocalization.contrast.emit_welch_stats", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.sourceLocContrastWelchStats = b
			}
		}},
		{key: "feature_engineering.sourcelocalization.reg", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.sourceLocReg = f
			}
		}},
		{key: "feature_engineering.sourcelocalization.snr", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.sourceLocSnr = f
			}
		}},
		{key: "feature_engineering.sourcelocalization.loose", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.sourceLocLoose = f
			}
		}},
		{key: "feature_engineering.sourcelocalization.depth", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.sourceLocDepth = f
			}
		}},
		{key: "feature_engineering.sourcelocalization.save_stc", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.sourceLocSaveStc = b
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.sourceLocFmriEnabled = b
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.stats_map_path", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocFmriStatsMap = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.provenance", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "same_dataset":
					m.sourceLocFmriProvenance = 1
				default:
					m.sourceLocFmriProvenance = 0
				}
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.require_provenance", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.sourceLocFmriRequireProv = b
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.threshold", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.sourceLocFmriThreshold = f
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.tail", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "abs":
					m.sourceLocFmriTail = 1
				default:
					m.sourceLocFmriTail = 0
				}
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.cluster_min_voxels", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.sourceLocFmriMinClusterVox = n
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.cluster_min_volume_mm3", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.sourceLocFmriMinClusterMM3 = f
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.max_clusters", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.sourceLocFmriMaxClusters = n
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.max_voxels_per_cluster", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.sourceLocFmriMaxVoxPerClus = n
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.max_total_voxels", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.sourceLocFmriMaxTotalVox = n
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.random_seed", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.sourceLocFmriRandomSeed = n
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.output_space", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "cluster":
					m.sourceLocFmriOutputSpace = 0
				case "atlas":
					m.sourceLocFmriOutputSpace = 1
				default:
					m.sourceLocFmriOutputSpace = 2 // dual
				}
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.sourceLocFmriContrastEnabled = b
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.type", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "paired-t-test":
					m.sourceLocFmriContrastType = 1
				case "f-test":
					m.sourceLocFmriContrastType = 2
				case "custom":
					m.sourceLocFmriContrastType = 3
				default:
					m.sourceLocFmriContrastType = 0
				}
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.cond_a.column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocFmriCondAColumn = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.cond_a.value", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocFmriCondAValue = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.cond_b.column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocFmriCondBColumn = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.cond_b.value", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocFmriCondBValue = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.condition_a.column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocFmriCondAColumn = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.condition_a.value", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocFmriCondAValue = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.condition_b.column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocFmriCondBColumn = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.condition_b.value", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocFmriCondBValue = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.formula", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocFmriContrastFormula = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.name", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocFmriContrastName = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.runs", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.sourceLocFmriRunsToInclude = strings.Join(splitCSVList(spec), ",")
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.hrf_model", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "spm":
					m.sourceLocFmriHrfModel = 0
				case "flobs":
					m.sourceLocFmriHrfModel = 1
				case "fir":
					m.sourceLocFmriHrfModel = 2
				}
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.drift_model", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "none":
					m.sourceLocFmriDriftModel = 0
				case "cosine":
					m.sourceLocFmriDriftModel = 1
				case "polynomial":
					m.sourceLocFmriDriftModel = 2
				}
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.high_pass_hz", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.sourceLocFmriHighPassHz = f
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.low_pass_hz", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.sourceLocFmriLowPassHz = f
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.condition_scope_column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocFmriConditionScopeColumn = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.condition_scope_trial_types", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.sourceLocFmriConditionScopeTrialTypes = strings.Join(splitLooseList(spec), " ")
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.phase_column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocFmriPhaseColumn = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.phase_scope_column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocFmriPhaseScopeColumn = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.phase_scope_value", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.sourceLocFmriPhaseScopeValue = s
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.stim_phases_to_model", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.sourceLocFmriStimPhasesToModel = strings.Join(splitLooseList(spec), " ")
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.cluster_correction", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.sourceLocFmriClusterCorrection = b
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.cluster_p_threshold", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.sourceLocFmriClusterPThreshold = f
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.output_type", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "z-score":
					m.sourceLocFmriOutputType = 0
				case "t-stat":
					m.sourceLocFmriOutputType = 1
				case "cope":
					m.sourceLocFmriOutputType = 2
				case "beta":
					m.sourceLocFmriOutputType = 3
				}
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.resample_to_freesurfer", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.sourceLocFmriResampleToFS = b
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.input_source", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "fmriprep":
					m.sourceLocFmriInputSource = 0
				case "bids_raw":
					m.sourceLocFmriInputSource = 1
				}
			}
		}},
		{key: "feature_engineering.sourcelocalization.fmri.contrast.require_fmriprep", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.sourceLocFmriRequireFmriprep = b
			}
		}},
		{key: "machine_learning.targets.regression", apply: func(v interface{}) {
			if s, ok := asString(v); ok && strings.TrimSpace(m.mlTarget) == "" {
				m.mlTarget = s
			}
		}},
		{key: "machine_learning.targets.classification", apply: func(v interface{}) {
			if s, ok := asString(v); ok && strings.TrimSpace(s) != "" {
				m.mlTarget = s
			}
		}},
		{key: "machine_learning.targets.binary_threshold", apply: func(v interface{}) {
			switch n := v.(type) {
			case float64:
				m.mlBinaryThresholdEnabled = true
				m.mlBinaryThreshold = n
			case int:
				m.mlBinaryThresholdEnabled = true
				m.mlBinaryThreshold = float64(n)
			}
		}},
		{key: "machine_learning.data.feature_families", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.mlFeatureFamiliesSpec = spec
			}
		}},
		{key: "machine_learning.data.feature_bands", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.mlFeatureBandsSpec = spec
			}
		}},
		{key: "machine_learning.data.feature_segments", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.mlFeatureSegmentsSpec = spec
			}
		}},
		{key: "machine_learning.data.feature_scopes", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.mlFeatureScopesSpec = spec
			}
		}},
		{key: "machine_learning.data.feature_stats", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.mlFeatureStatsSpec = spec
			}
		}},
		{key: "machine_learning.data.feature_harmonization", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "intersection":
					m.mlFeatureHarmonization = MLFeatureHarmonizationIntersection
				case "union_impute":
					m.mlFeatureHarmonization = MLFeatureHarmonizationUnionImpute
				default:
					m.mlFeatureHarmonization = MLFeatureHarmonizationDefault
				}
			}
		}},
		{key: "machine_learning.data.covariates", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.mlCovariatesSpec = spec
			}
		}},
		{key: "machine_learning.data.require_trial_ml_safe", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.mlRequireTrialMlSafe = b
			}
		}},
		{key: "machine_learning.plotting.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.mlPlotsEnabled = b
			}
		}},
		{key: "machine_learning.plotting.formats", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok && strings.TrimSpace(spec) != "" {
				m.mlPlotFormatsSpec = strings.Join(splitLooseList(spec), " ")
				return
			}
			if s, ok := asString(v); ok && s != "" {
				m.mlPlotFormatsSpec = strings.Join(splitLooseList(s), " ")
			}
		}},
		{key: "machine_learning.plotting.dpi", apply: func(v interface{}) {
			if n, ok := asInt(v); ok && n >= 72 {
				m.mlPlotDPI = n
			}
		}},
		{key: "machine_learning.plotting.top_n_features", apply: func(v interface{}) {
			if n, ok := asInt(v); ok && n >= 1 {
				m.mlPlotTopNFeatures = n
			}
		}},
		{key: "machine_learning.plotting.include_diagnostics", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.mlPlotDiagnostics = b
			}
		}},
		{key: "machine_learning.classification.model", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "svm":
					m.mlClassificationModel = MLClassificationSVM
				case "lr":
					m.mlClassificationModel = MLClassificationLR
				case "rf":
					m.mlClassificationModel = MLClassificationRF
				case "cnn":
					m.mlClassificationModel = MLClassificationCNN
				default:
					m.mlClassificationModel = MLClassificationDefault
				}
			}
		}},
		{key: "machine_learning.cv.inner_splits", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.innerSplits = n
			}
		}},
		{key: "machine_learning.models.elasticnet.alpha_grid", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok && strings.TrimSpace(spec) != "" {
				m.elasticNetAlphaGrid = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "machine_learning.models.elasticnet.l1_ratio_grid", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok && strings.TrimSpace(spec) != "" {
				m.elasticNetL1RatioGrid = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "machine_learning.models.ridge.alpha_grid", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok && strings.TrimSpace(spec) != "" {
				m.ridgeAlphaGrid = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "machine_learning.models.random_forest.n_estimators", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.rfNEstimators = n
			}
		}},
		{key: "machine_learning.models.random_forest.max_depth_grid", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok && strings.TrimSpace(spec) != "" {
				m.rfMaxDepthGrid = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "machine_learning.preprocessing.variance_threshold_grid", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok && strings.TrimSpace(spec) != "" {
				m.varianceThresholdGrid = strings.Join(splitLooseList(spec), ",")
			}
		}},
		// Additional ML hydration (YAML -> TUI)
		{key: "machine_learning.preprocessing.imputer_strategy", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "mean":
					m.mlImputer = 1
				case "most_frequent":
					m.mlImputer = 2
				default:
					m.mlImputer = 0
				}
			}
		}},
		{key: "machine_learning.preprocessing.power_transformer_method", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "box-cox":
					m.mlPowerTransformerMethod = 1
				default:
					m.mlPowerTransformerMethod = 0
				}
			}
		}},
		{key: "machine_learning.preprocessing.power_transformer_standardize", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.mlPowerTransformerStandardize = b
			}
		}},
		{key: "machine_learning.preprocessing.pca.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.mlPCAEnabled = b
			}
		}},
		{key: "machine_learning.preprocessing.pca.n_components", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok && f > 0 {
				m.mlPCANComponents = f
			}
		}},
		{key: "machine_learning.preprocessing.pca.whiten", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.mlPCAWhiten = b
			}
		}},
		{key: "machine_learning.preprocessing.pca.svd_solver", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "full":
					m.mlPCASvdSolver = 1
				case "randomized":
					m.mlPCASvdSolver = 2
				default:
					m.mlPCASvdSolver = 0
				}
			}
		}},
		{key: "machine_learning.preprocessing.pca.random_state", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlPCARngSeed = n
			}
		}},
		{key: "machine_learning.preprocessing.deconfound", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.mlDeconfound = b
			}
		}},
		{key: "machine_learning.preprocessing.feature_selection_percentile", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.mlFeatureSelectionPercentile = f
			}
		}},
		{key: "machine_learning.preprocessing.spatial_regions_allowed", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.mlSpatialRegionsAllowed = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "machine_learning.classification.calibrate_ensemble", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.mlEnsembleCalibrate = b
			}
		}},
		{key: "machine_learning.classification.resampler", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "undersample":
					m.mlClassificationResampler = 1
				case "smote":
					m.mlClassificationResampler = 2
				default:
					m.mlClassificationResampler = 0
				}
			}
		}},
		{key: "machine_learning.classification.resampler_seed", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlClassificationResamplerSeed = n
			}
		}},
		{key: "machine_learning.models.svm.kernel", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "linear":
					m.mlSvmKernel = 1
				case "poly":
					m.mlSvmKernel = 2
				default:
					m.mlSvmKernel = 0
				}
			}
		}},
		{key: "machine_learning.models.svm.C_grid", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.mlSvmCGrid = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "machine_learning.models.svm.gamma_grid", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.mlSvmGammaGrid = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "machine_learning.models.svm.class_weight", apply: func(v interface{}) {
			if v == nil {
				m.mlSvmClassWeight = 1
				return
			}
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "none") {
					m.mlSvmClassWeight = 1
				} else {
					m.mlSvmClassWeight = 0
				}
			}
		}},
		{key: "machine_learning.models.logistic_regression.penalty", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "l1":
					m.mlLrPenalty = 1
				case "elasticnet":
					m.mlLrPenalty = 2
				default:
					m.mlLrPenalty = 0
				}
			}
		}},
		{key: "machine_learning.models.logistic_regression.C_grid", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.mlLrCGrid = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "machine_learning.models.logistic_regression.l1_ratio_grid", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.mlLrL1RatioGrid = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "machine_learning.models.logistic_regression.max_iter", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlLrMaxIter = n
			}
		}},
		{key: "machine_learning.models.logistic_regression.class_weight", apply: func(v interface{}) {
			if v == nil {
				m.mlLrClassWeight = 1
				return
			}
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "none") {
					m.mlLrClassWeight = 1
				} else {
					m.mlLrClassWeight = 0
				}
			}
		}},
		{key: "machine_learning.models.random_forest.min_samples_split_grid", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.mlRfMinSamplesSplitGrid = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "machine_learning.models.random_forest.min_samples_leaf_grid", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.mlRfMinSamplesLeafGrid = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "machine_learning.models.random_forest.bootstrap", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.mlRfBootstrap = b
			}
		}},
		{key: "machine_learning.models.random_forest.class_weight", apply: func(v interface{}) {
			if v == nil {
				m.mlRfClassWeight = 2
				return
			}
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "balanced_subsample":
					m.mlRfClassWeight = 1
				case "none":
					m.mlRfClassWeight = 2
				default:
					m.mlRfClassWeight = 0
				}
			}
		}},
		{key: "machine_learning.models.cnn.temporal_filters", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlCnnFilters1 = n
			}
		}},
		{key: "machine_learning.models.cnn.pointwise_filters", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlCnnFilters2 = n
			}
		}},
		{key: "machine_learning.models.cnn.kernel_length", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlCnnKernelSize1 = n
			}
		}},
		{key: "machine_learning.models.cnn.separable_kernel_length", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlCnnKernelSize2 = n
			}
		}},
		{key: "machine_learning.models.cnn.pool_size", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlCnnPoolSize = n
			}
		}},
		{key: "machine_learning.models.cnn.dense_units", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlCnnDenseUnits = n
			}
		}},
		{key: "machine_learning.models.cnn.dropout_conv", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.mlCnnDropoutConv = f
			}
		}},
		{key: "machine_learning.models.cnn.dropout", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.mlCnnDropoutDense = f
			}
		}},
		{key: "machine_learning.models.cnn.batch_size", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlCnnBatchSize = n
			}
		}},
		{key: "machine_learning.models.cnn.max_epochs", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlCnnEpochs = n
			}
		}},
		{key: "machine_learning.models.cnn.learning_rate", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.mlCnnLearningRate = f
			}
		}},
		{key: "machine_learning.models.cnn.patience", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlCnnPatience = n
			}
		}},
		{key: "machine_learning.models.cnn.min_delta", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.mlCnnMinDelta = f
			}
		}},
		{key: "machine_learning.models.cnn.weight_decay", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.mlCnnL2Lambda = f
			}
		}},
		{key: "machine_learning.cv.hygiene_enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.mlCvHygieneEnabled = b
			}
		}},
		{key: "machine_learning.cv.permutation_scheme", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "within_subject_within_block") {
					m.mlCvPermutationScheme = 1
				} else {
					m.mlCvPermutationScheme = 0
				}
			}
		}},
		{key: "machine_learning.cv.min_valid_permutation_fraction", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.mlCvMinValidPermFraction = f
			}
		}},
		{key: "machine_learning.cv.default_n_bins", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlCvDefaultNBins = n
			}
		}},
		{key: "machine_learning.evaluation.ci_method", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "fixed_effects") {
					m.mlEvalCIMethod = 1
				} else {
					m.mlEvalCIMethod = 0
				}
			}
		}},
		{key: "machine_learning.evaluation.subject_weighting", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "trial_count") {
					m.mlEvalSubjectWeighting = 1
				} else {
					m.mlEvalSubjectWeighting = 0
				}
			}
		}},
		{key: "machine_learning.evaluation.bootstrap_iterations", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlEvalBootstrapIterations = n
			}
		}},
		{key: "machine_learning.data.covariates_strict", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.mlDataCovariatesStrict = b
			}
		}},
		{key: "machine_learning.data.max_excluded_subject_fraction", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.mlDataMaxExcludedSubjectFraction = f
			}
		}},
		{key: "machine_learning.incremental_validity.baseline_alpha", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.mlIncrementalBaselineAlpha = f
			}
		}},
		{key: "machine_learning.incremental_validity.require_baseline_predictors", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.mlIncrementalRequireBaselinePred = b
			}
		}},
		{key: "machine_learning.interpretability.grouped_outputs", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.mlInterpretabilityGroupedOutputs = b
			}
		}},
		{key: "machine_learning.analysis.time_generalization.min_subjects_per_cell", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlTimeGenMinSubjects = n
			}
		}},
		{key: "machine_learning.analysis.time_generalization.min_valid_permutation_fraction", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.mlTimeGenMinValidPermFraction = f
			}
		}},
		{key: "machine_learning.classification.min_subjects_with_auc_for_inference", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlClassMinSubjectsForAUC = n
			}
		}},
		{key: "machine_learning.classification.max_failed_fold_fraction", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.mlClassMaxFailedFoldFraction = f
			}
		}},
		{key: "machine_learning.targets.strict_regression_target_continuous", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.mlTargetsStrictRegressionCont = b
			}
		}},
		{key: "project.random_state", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.mlCnnRandomSeed = n
			}
		}},
		// Additional preprocessing/alignment hydration
		{key: "eeg.ecg_channels", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.prepEcgChannels = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "epochs.autoreject_n_interpolate", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.prepAutorejectNInterpolate = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "alignment.allow_misaligned_trim", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.alignAllowMisalignedTrim = b
			}
		}},
		{key: "alignment.min_alignment_samples", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.alignMinAlignmentSamples = n
			}
		}},
		{key: "alignment.trim_to_first_volume", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.alignTrimToFirstVolume = b
			}
		}},
		{key: "alignment.fmri_onset_reference", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "first_volume", "first_iti_start":
					m.alignFmriOnsetReference = 1
				case "scanner_trigger", "first_stim_start":
					m.alignFmriOnsetReference = 2
				default:
					m.alignFmriOnsetReference = 0
				}
			}
		}},
		// Additional feature-engineering hydration
		{key: "feature_engineering.spatial_transform_per_family.connectivity", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.spatialTransformPerFamilyConnectivity = spatialTransformOverrideIndex(s)
			}
		}},
		{key: "feature_engineering.spatial_transform_per_family.itpc", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.spatialTransformPerFamilyItpc = spatialTransformOverrideIndex(s)
			}
		}},
		{key: "feature_engineering.spatial_transform_per_family.pac", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.spatialTransformPerFamilyPac = spatialTransformOverrideIndex(s)
			}
		}},
		{key: "feature_engineering.spatial_transform_per_family.power", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.spatialTransformPerFamilyPower = spatialTransformOverrideIndex(s)
			}
		}},
		{key: "feature_engineering.spatial_transform_per_family.aperiodic", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.spatialTransformPerFamilyAperiodic = spatialTransformOverrideIndex(s)
			}
		}},
		{key: "feature_engineering.spatial_transform_per_family.bursts", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.spatialTransformPerFamilyBursts = spatialTransformOverrideIndex(s)
			}
		}},
		{key: "feature_engineering.spatial_transform_per_family.erds", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.spatialTransformPerFamilyErds = spatialTransformOverrideIndex(s)
			}
		}},
		{key: "feature_engineering.spatial_transform_per_family.complexity", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.spatialTransformPerFamilyComplexity = spatialTransformOverrideIndex(s)
			}
		}},
		{key: "feature_engineering.spatial_transform_per_family.ratios", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.spatialTransformPerFamilyRatios = spatialTransformOverrideIndex(s)
			}
		}},
		{key: "feature_engineering.spatial_transform_per_family.asymmetry", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.spatialTransformPerFamilyAsymmetry = spatialTransformOverrideIndex(s)
			}
		}},
		{key: "feature_engineering.spatial_transform_per_family.spectral", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.spatialTransformPerFamilySpectral = spatialTransformOverrideIndex(s)
			}
		}},
		{key: "feature_engineering.spatial_transform_per_family.erp", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.spatialTransformPerFamilyErp = spatialTransformOverrideIndex(s)
			}
		}},
		{key: "feature_engineering.spatial_transform_per_family.quality", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.spatialTransformPerFamilyQuality = spatialTransformOverrideIndex(s)
			}
		}},
		{key: "feature_engineering.spatial_transform_per_family.microstates", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.spatialTransformPerFamilyMicrostates = spatialTransformOverrideIndex(s)
			}
		}},
		{key: "feature_engineering.change_scores.transform", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "ratio":
					m.changeScoresTransform = 1
				case "log_ratio":
					m.changeScoresTransform = 2
				default:
					m.changeScoresTransform = 0
				}
			}
		}},
		{key: "feature_engineering.change_scores.window_pairs", apply: func(v interface{}) {
			if spec, ok := asWindowPairSpec(v); ok {
				m.changeScoresWindowPairs = spec
			}
		}},
		{key: "feature_engineering.itpc.min_segment_sec", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.itpcMinSegmentSec = f
			}
		}},
		{key: "feature_engineering.itpc.min_cycles_at_fmin", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.itpcMinCyclesAtFmin = f
			}
		}},
		{key: "feature_engineering.pac.min_segment_sec", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.pacMinSegmentSec = f
			}
		}},
		{key: "feature_engineering.pac.min_cycles_at_fmin", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.pacMinCyclesAtFmin = f
			}
		}},
		{key: "feature_engineering.pac.surrogate_method", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(strings.TrimSpace(s)) {
				case "circular_shift":
					m.pacSurrogateMethod = 1
				case "swap_phase_amp":
					m.pacSurrogateMethod = 2
				case "time_shift":
					m.pacSurrogateMethod = 3
				default:
					m.pacSurrogateMethod = 0
				}
			}
		}},
		{key: "feature_engineering.aperiodic.max_freq_resolution_hz", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.aperiodicMaxFreqResolutionHz = f
			}
		}},
		{key: "feature_engineering.aperiodic.multitaper_adaptive", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.aperiodicMultitaperAdaptive = b
			}
		}},
		{key: "feature_engineering.directedconnectivity.min_samples_per_mvar_parameter", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.directedConnMinSamplesPerMvarParam = n
			}
		}},
		{key: "feature_engineering.erds.laterality_marker_bands", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.erdsConditionMarkerBands = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "feature_engineering.erds.laterality_columns", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.erdsLateralityColumns = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "feature_engineering.erds.somatosensory_left_channels", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.erdsSomatosensoryLeftChannels = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "feature_engineering.erds.somatosensory_right_channels", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.erdsSomatosensoryRightChannels = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "feature_engineering.erds.onset_min_threshold_percent", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.erdsOnsetMinThresholdPercent = f
			}
		}},
		{key: "feature_engineering.erds.rebound_threshold_sigma", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.erdsReboundThresholdSigma = f
			}
		}},
		{key: "feature_engineering.erds.rebound_min_threshold_percent", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.erdsReboundMinThresholdPercent = f
			}
		}},
		{key: "feature_engineering.microstates.assign_from_gfp_peaks", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.microstatesAssignFromGfpPeaks = b
			}
		}},
		// Supplemental behavior hydration for configurable fields.
		{key: "behavior_analysis.feature_categories", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				joined := strings.Join(splitLooseList(spec), ",")
				if strings.TrimSpace(joined) != "" {
					m.correlationsFeaturesSpec = joined
					m.conditionFeaturesSpec = joined
					m.clusterFeaturesSpec = joined
				}
			}
		}},
		{key: "behavior_analysis.temporal.features", apply: func(v interface{}) {
			raw, ok := v.(map[string]interface{})
			if !ok {
				return
			}
			selected := make([]string, 0, 3)
			for _, name := range []string{"power", "itpc", "erds"} {
				if val, ok := raw[name]; ok {
					if b, ok := asBool(val); ok && b {
						selected = append(selected, name)
					}
				}
			}
			if len(selected) > 0 {
				m.temporalFeaturesSpec = strings.Join(selected, ",")
			}
		}},
	}

	for _, b := range binders {
		if v, ok := values[b.key]; ok {
			b.apply(v)
		}
	}
}

func asString(v interface{}) (string, bool) {
	s, ok := v.(string)
	return strings.TrimSpace(s), ok
}

func asBool(v interface{}) (bool, bool) {
	b, ok := v.(bool)
	return b, ok
}

func asInt(v interface{}) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	}
	return 0, false
}

func asFloat(v interface{}) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	}
	return 0, false
}

func asCompactJSON(v interface{}) (string, bool) {
	blob, err := json.Marshal(v)
	if err != nil {
		return "", false
	}
	s := strings.TrimSpace(string(blob))
	if s == "" || s == "null" {
		return "", false
	}
	return s, true
}

// asStringList extracts []string from a []interface{} (string items only).
func asStringList(v interface{}) ([]string, bool) {
	raw, ok := v.([]interface{})
	if !ok {
		return nil, false
	}
	var out []string
	for _, item := range raw {
		if s, ok := item.(string); ok {
			if s = strings.TrimSpace(s); s != "" {
				out = append(out, s)
			}
		}
	}
	return out, true
}

// asListSpec joins a []interface{} or []string into a space-separated spec string.
func asListSpec(v interface{}) (string, bool) {
	switch vals := v.(type) {
	case []interface{}:
		out := make([]string, 0, len(vals))
		for _, item := range vals {
			s := strings.TrimSpace(fmt.Sprintf("%v", item))
			if s != "" && s != "<nil>" {
				out = append(out, s)
			}
		}
		return strings.Join(out, " "), true
	case []string:
		out := make([]string, 0, len(vals))
		for _, item := range vals {
			if s := strings.TrimSpace(item); s != "" {
				out = append(out, s)
			}
		}
		return strings.Join(out, " "), true
	}
	return "", false
}

func spatialTransformOverrideIndex(value string) int {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "none":
		return 1
	case "csd":
		return 2
	case "laplacian":
		return 3
	default:
		return 0
	}
}

// asWindowPairSpec normalizes change-score window pairs to "a:b,c:d" format.
func asWindowPairSpec(v interface{}) (string, bool) {
	switch pairs := v.(type) {
	case []interface{}:
		out := make([]string, 0, len(pairs))
		for _, item := range pairs {
			switch pair := item.(type) {
			case []interface{}:
				if len(pair) < 2 {
					continue
				}
				left := strings.TrimSpace(fmt.Sprintf("%v", pair[0]))
				right := strings.TrimSpace(fmt.Sprintf("%v", pair[1]))
				if left != "" && left != "<nil>" && right != "" && right != "<nil>" {
					out = append(out, left+":"+right)
				}
			case []string:
				if len(pair) < 2 {
					continue
				}
				left := strings.TrimSpace(pair[0])
				right := strings.TrimSpace(pair[1])
				if left != "" && right != "" {
					out = append(out, left+":"+right)
				}
			case string:
				if s := strings.TrimSpace(pair); s != "" {
					out = append(out, s)
				}
			}
		}
		if len(out) == 0 {
			return "", false
		}
		return strings.Join(out, ","), true
	case []string:
		out := make([]string, 0, len(pairs))
		for _, pair := range pairs {
			if s := strings.TrimSpace(pair); s != "" {
				out = append(out, s)
			}
		}
		if len(out) == 0 {
			return "", false
		}
		return strings.Join(out, ","), true
	case string:
		if s := strings.TrimSpace(pairs); s != "" {
			return strings.Join(splitLooseList(s), ","), true
		}
	}
	return "", false
}

// sliceFromAny extracts []string from []interface{} or a comma-separated string.
func sliceFromAny(v interface{}) ([]string, bool) {
	switch val := v.(type) {
	case []interface{}:
		return asStringList(val)
	case string:
		var out []string
		for _, p := range strings.Split(val, ",") {
			if p = strings.TrimSpace(p); p != "" {
				out = append(out, p)
			}
		}
		return out, len(out) > 0
	}
	return nil, false
}

func parseConfigBands(value interface{}) []FrequencyBand {
	raw, ok := value.(map[string]interface{})
	if !ok {
		return nil
	}
	names := make([]string, 0, len(raw))
	for name := range raw {
		names = append(names, name)
	}
	sort.Strings(names)

	var bands []FrequencyBand
	for _, name := range names {
		entry, ok := raw[name].([]interface{})
		if !ok || len(entry) < 2 {
			continue
		}
		var low, high float64
		var okLow, okHigh bool

		if f, ok := entry[0].(float64); ok {
			low = f
			okLow = true
		} else if i, ok := entry[0].(int); ok {
			low = float64(i)
			okLow = true
		}

		if f, ok := entry[1].(float64); ok {
			high = f
			okHigh = true
		} else if i, ok := entry[1].(int); ok {
			high = float64(i)
			okHigh = true
		}

		if !okLow || !okHigh {
			continue
		}
		bands = append(bands, FrequencyBand{
			Key:         name,
			Name:        titleCase(name),
			Description: fmt.Sprintf("%.1f-%.1f Hz", low, high),
			LowHz:       low,
			HighHz:      high,
		})
	}
	return bands
}

func titleCase(value string) string {
	if value == "" {
		return value
	}
	return strings.ToUpper(value[:1]) + value[1:]
}

func (m Model) behaviorSections() []behaviorSection {
	return []behaviorSection{
		{Key: "general", Label: "General", Enabled: true},
		{Key: "trial_table", Label: "Trial Table", Enabled: m.isComputationSelected("trial_table")},
		{Key: "correlations", Label: "Correlations", Enabled: m.isComputationSelected("correlations")},
		{Key: "regression", Label: "Regression", Enabled: m.isComputationSelected("regression")},
		{Key: "condition", Label: "Condition", Enabled: m.isComputationSelected("condition")},
		{Key: "temporal", Label: "Temporal", Enabled: m.isComputationSelected("temporal")},
		{Key: "cluster", Label: "Cluster", Enabled: m.isComputationSelected("cluster")},
	}
}
