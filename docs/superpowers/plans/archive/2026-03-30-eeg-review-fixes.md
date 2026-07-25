# EEG Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the EEG review findings around behavior progress finalization, unified feature-correlator loading, and PAC artifact consistency.

**Architecture:** Keep the fixes local to the current behavior pipeline, behavior context, feature discovery, and unified correlator. Add regression tests first, then make the minimal code changes needed to align all PAC loaders on one canonical artifact and route feature loading through the canonical path resolver.

**Tech Stack:** Python 3, `unittest`, pandas, existing EEG pipeline loaders/config.

---

### Task 1: Lock Down The Regressions

**Files:**
- Modify: `tests/pipelines/test_pipeline_behavior.py`
- Modify: `tests/behavior/test_behavior_validity_fixes.py`
- Modify: `tests/features/test_feature_io_pac_outputs.py`

- [ ] **Step 1: Write failing tests for setup-time progress failure, canonical correlator loading, and canonical PAC selection**
- [ ] **Step 2: Run targeted `unittest` commands to verify the new tests fail for the expected reasons**

### Task 2: Patch The Production Code

**Files:**
- Modify: `eeg_pipeline/pipelines/behavior.py`
- Modify: `eeg_pipeline/analysis/behavior/feature_correlator.py`
- Modify: `eeg_pipeline/context/behavior.py`
- Modify: `eeg_pipeline/utils/data/feature_discovery.py`
- Modify: `eeg_pipeline/utils/config/behavior_config.yaml`

- [ ] **Step 1: Finalize subject progress on all post-start failures in the behavior pipeline**
- [ ] **Step 2: Route unified correlator feature loads through canonical feature-path and table readers**
- [ ] **Step 3: Make `pac` consistently resolve to `features_pac.parquet` across bundle loading and discovery/config**

### Task 3: Verify

**Files:**
- Test: `tests/pipelines/test_pipeline_behavior.py`
- Test: `tests/behavior/test_behavior_validity_fixes.py`
- Test: `tests/features/test_feature_io_pac_outputs.py`

- [ ] **Step 1: Re-run the targeted `unittest` coverage and confirm all new regressions pass**
- [ ] **Step 2: Summarize any remaining verification gaps if environment tooling is unavailable**
