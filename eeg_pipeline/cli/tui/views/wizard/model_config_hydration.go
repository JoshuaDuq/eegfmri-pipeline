package wizard

import (
	"fmt"
	"sort"
	"strings"
)

func (m *Model) ApplyConfigKeys(values map[string]interface{}) {
	asString := func(v interface{}) (string, bool) {
		s, ok := v.(string)
		if !ok {
			return "", false
		}
		return strings.TrimSpace(s), true
	}
	asBool := func(v interface{}) (bool, bool) {
		b, ok := v.(bool)
		return b, ok
	}
	asInt := func(v interface{}) (int, bool) {
		switch n := v.(type) {
		case float64:
			return int(n), true
		case int:
			return n, true
		default:
			return 0, false
		}
	}
	asFloat := func(v interface{}) (float64, bool) {
		switch n := v.(type) {
		case float64:
			return n, true
		case int:
			return float64(n), true
		default:
			return 0, false
		}
	}
	asStringList := func(v interface{}) ([]string, bool) {
		raw, ok := v.([]interface{})
		if !ok {
			return nil, false
		}
		var out []string
		for _, item := range raw {
			s, ok := item.(string)
			if !ok {
				continue
			}
			s = strings.TrimSpace(s)
			if s != "" {
				out = append(out, s)
			}
		}
		return out, true
	}
	asListSpec := func(v interface{}) (string, bool) {
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
				s := strings.TrimSpace(item)
				if s != "" {
					out = append(out, s)
				}
			}
			return strings.Join(out, " "), true
		default:
			return "", false
		}
	}

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
		{key: "behavior_analysis.control_temperature", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.controlTemperature = b
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
		{key: "behavior_analysis.correlations.use_crossfit_pain_residual", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.correlationsUseCrossfitResidual = b
			}
		}},
		{key: "behavior_analysis.correlations.prefer_pain_residual", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.correlationsPreferPainResidual = b
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
		{key: "behavior_analysis.group_level.multilevel_correlations.control_temperature", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.groupLevelControlTemperature = b
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
		{key: "behavior_analysis.group_level.multilevel_correlations.allow_parametric_fallback", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.groupLevelAllowParametricFallback = b
			}
		}},
		{key: "behavior_analysis.pain_sensitivity.min_trials", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.painSensitivityMinTrials = n
			}
		}},
		{key: "behavior_analysis.pain_sensitivity.primary_unit", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "run_mean") || strings.EqualFold(s, "run") {
					m.painSensitivityPrimaryUnit = 1
				} else {
					m.painSensitivityPrimaryUnit = 0
				}
			}
		}},
		{key: "behavior_analysis.pain_sensitivity.n_permutations", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.painSensitivityPermutations = n
			}
		}},
		{key: "behavior_analysis.pain_sensitivity.p_primary_mode", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.painSensitivityPermutationPrimary = !strings.EqualFold(strings.TrimSpace(s), "asymptotic")
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
		{key: "behavior_analysis.condition.compare_windows", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.conditionCompareWindows = strings.Join(splitLooseList(spec), " ")
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
		{key: "behavior_analysis.condition.window_comparison.primary_unit", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "run_mean") || strings.EqualFold(s, "run") {
					m.conditionWindowPrimaryUnit = 1
				} else {
					m.conditionWindowPrimaryUnit = 0
				}
			}
		}},
		{key: "behavior_analysis.condition.window_comparison.min_samples", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.conditionWindowMinSamples = n
			}
		}},
		{key: "behavior_analysis.mixed_effects.include_temperature", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.mixedIncludeTemperature = b
			}
		}},
		{key: "behavior_analysis.mediation.p_primary_mode", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.mediationPermutationPrimary = !strings.EqualFold(strings.TrimSpace(s), "asymptotic")
			}
		}},
		{key: "behavior_analysis.moderation.p_primary_mode", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.moderationPermutationPrimary = !strings.EqualFold(strings.TrimSpace(s), "asymptotic")
			}
		}},
		{key: "behavior_analysis.report.top_n", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.reportTopN = n
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
		{key: "behavior_analysis.trial_table.add_lag_features", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.trialTableAddLagFeatures = b
			}
		}},
		{key: "behavior_analysis.trial_order.max_missing_fraction", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.trialOrderMaxMissingFraction = f
			}
		}},
		{key: "behavior_analysis.feature_summaries.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.featureSummariesEnabled = b
			}
		}},
		{key: "behavior_analysis.feature_qc.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.featureQCEnabled = b
			}
		}},
		{key: "behavior_analysis.feature_qc.max_missing_pct", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.featureQCMaxMissingPct = f
			}
		}},
		{key: "behavior_analysis.feature_qc.min_variance", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.featureQCMinVariance = f
			}
		}},
		{key: "behavior_analysis.feature_qc.check_within_run_variance", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.featureQCCheckWithinRunVariance = b
			}
		}},
		{key: "behavior_analysis.pain_residual.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.painResidualEnabled = b
			}
		}},
		{key: "behavior_analysis.pain_residual.method", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "poly") {
					m.painResidualMethod = 1
				} else {
					m.painResidualMethod = 0
				}
			}
		}},
		{key: "behavior_analysis.pain_residual.min_samples", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.painResidualMinSamples = n
			}
		}},
		{key: "behavior_analysis.pain_residual.spline_df_candidates", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.painResidualSplineDfCandidates = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "behavior_analysis.pain_residual.poly_degree", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.painResidualPolyDegree = n
			}
		}},
		{key: "behavior_analysis.temperature_models.model_comparison.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.painResidualModelCompareEnabled = b
			}
		}},
		{key: "behavior_analysis.temperature_models.model_comparison.min_samples", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.painResidualModelCompareMinSamples = n
			}
		}},
		{key: "behavior_analysis.temperature_models.model_comparison.poly_degrees", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.painResidualModelComparePolyDegrees = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "behavior_analysis.temperature_models.breakpoint_test.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.painResidualBreakpointEnabled = b
			}
		}},
		{key: "behavior_analysis.temperature_models.breakpoint_test.min_samples", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.painResidualBreakpointMinSamples = n
			}
		}},
		{key: "behavior_analysis.temperature_models.breakpoint_test.n_candidates", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.painResidualBreakpointCandidates = n
			}
		}},
		{key: "behavior_analysis.temperature_models.breakpoint_test.quantile_low", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.painResidualBreakpointQlow = f
			}
		}},
		{key: "behavior_analysis.temperature_models.breakpoint_test.quantile_high", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.painResidualBreakpointQhigh = f
			}
		}},
		{key: "behavior_analysis.pain_residual.crossfit.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.painResidualCrossfitEnabled = b
			}
		}},
		{key: "behavior_analysis.pain_residual.crossfit.group_column", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				m.painResidualCrossfitGroupColumn = s
			}
		}},
		{key: "behavior_analysis.pain_residual.crossfit.n_splits", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.painResidualCrossfitNSplits = n
			}
		}},
		{key: "behavior_analysis.pain_residual.crossfit.method", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				if strings.EqualFold(s, "poly") {
					m.painResidualCrossfitMethod = 1
				} else {
					m.painResidualCrossfitMethod = 0
				}
			}
		}},
		{key: "behavior_analysis.pain_residual.crossfit.spline_n_knots", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.painResidualCrossfitSplineKnots = n
			}
		}},
		{key: "behavior_analysis.statistics.temperature_control", apply: func(v interface{}) {
			if s, ok := asString(v); ok {
				switch strings.ToLower(s) {
				case "linear":
					m.behaviorStatsTempControl = 1
				case "none":
					m.behaviorStatsTempControl = 2
				default:
					m.behaviorStatsTempControl = 0
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
		{key: "behavior_analysis.statistics.default_n_bootstrap", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.globalNBootstrap = n
			}
		}},
		{key: "behavior_analysis.cluster_correction.enabled", apply: func(v interface{}) {
			if b, ok := asBool(v); ok {
				m.clusterCorrectionEnabled = b
			}
		}},
		{key: "behavior_analysis.cluster_correction.alpha", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.clusterCorrectionAlpha = f
			}
		}},
		{key: "behavior_analysis.cluster_correction.min_cluster_size", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.clusterCorrectionMinClusterSize = n
			}
		}},
		{key: "behavior_analysis.cluster_correction.tail", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				switch n {
				case 1:
					m.clusterCorrectionTailGlobal = 1
				case -1:
					m.clusterCorrectionTailGlobal = 2
				default:
					m.clusterCorrectionTailGlobal = 0
				}
			}
		}},
		{key: "validation.min_epochs", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.validationMinEpochs = n
			}
		}},
		{key: "validation.min_channels", apply: func(v interface{}) {
			if n, ok := asInt(v); ok {
				m.validationMinChannels = n
			}
		}},
		{key: "validation.max_amplitude_uv", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.validationMaxAmplitudeUv = f
			}
		}},
		{key: "io.constants.temperature_range", apply: func(v interface{}) {
			if spec, ok := asListSpec(v); ok {
				m.ioTemperatureRange = strings.Join(splitLooseList(spec), ",")
			}
		}},
		{key: "io.constants.max_missing_channels_fraction", apply: func(v interface{}) {
			if f, ok := asFloat(v); ok {
				m.ioMaxMissingChannelsFraction = f
			}
		}},
		{key: "feature_engineering.sourcelocalization.subjects_dir", apply: func(v interface{}) {
			if s, ok := asString(v); ok && strings.TrimSpace(s) != "" {
				m.sourceLocSubjectsDir = s
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
	}

	for _, binding := range binders {
		if v, ok := values[binding.key]; ok {
			binding.apply(v)
		}
	}
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
		{Key: "pain_sensitivity", Label: "Predictor Sensitivity", Enabled: m.isComputationSelected("pain_sensitivity")},
		{Key: "regression", Label: "Regression", Enabled: m.isComputationSelected("regression")},
		{Key: "stability", Label: "Stability", Enabled: m.isComputationSelected("stability")},
		{Key: "consistency", Label: "Consistency", Enabled: m.isComputationSelected("consistency")},
		{Key: "influence", Label: "Influence", Enabled: m.isComputationSelected("influence")},
		{Key: "condition", Label: "Condition", Enabled: m.isComputationSelected("condition")},
		{Key: "temporal", Label: "Temporal", Enabled: m.isComputationSelected("temporal")},
		{Key: "cluster", Label: "Cluster", Enabled: m.isComputationSelected("cluster")},
		{Key: "mediation", Label: "Mediation", Enabled: m.isComputationSelected("mediation")},
		{Key: "moderation", Label: "Moderation", Enabled: m.isComputationSelected("moderation")},
		{Key: "mixed_effects", Label: "Mixed Effects", Enabled: m.isComputationSelected("mixed_effects")},
	}
}
