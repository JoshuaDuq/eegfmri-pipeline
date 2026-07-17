# Study 1 fMRI motion QC

## Scope

- Participants: the 13 IDs in `local_workflows/alliance_canada/study1_subjects.txt`
- Acquisition: `task-thermalactive`, runs 1–6
- Input: fMRIPrep `desc-confounds_timeseries.tsv` files
- Data analyzed: 78 runs, 44,460 volumes, and 44,382 defined framewise-displacement
  observations (the first volume of each run has undefined FD)
- Excluded from this summary: pilot participants and the resting-state acquisition

## Measures

Framewise displacement (FD) is the primary movement measure. The report gives mean, median,
95th percentile, and maximum FD, plus the fractions of volumes above 0.2 mm and 0.5 mm.
Translation and rotation ranges describe slow drift within each run but are not exclusion rules.

The 0.2 mm and 0.5 mm cutoffs are descriptive reference points. They are not treated as
preregistered exclusion criteria.

## Cohort result

- Mean FD across all defined observations: **0.084 mm**
- Median FD: **0.070 mm**
- 95th percentile FD: **0.178 mm**
- Volumes above 0.2 mm: **1,595 / 44,382 (3.59%)**
- Volumes above 0.5 mm: **204 / 44,382 (0.46%)**
- Runs with mean FD above 0.2 mm: **0 / 78**
- Participants with mean FD above 0.2 mm: **0 / 13**
- Runs with more than 20% of volumes above 0.5 mm: **0 / 78**

No participant shows sustained high movement by these descriptive benchmarks. The largest
participant-average FD values are sub-0009 (0.118 mm), sub-0012 (0.113 mm), and sub-0007
(0.107 mm). The highest-motion runs by mean FD are sub-0009 run 5 (0.164 mm), sub-0007 run 4
(0.147 mm), and sub-0009 run 6 (0.142 mm).

There are isolated large movements. The largest FD value is 2.585 mm at sub-0000 run 6,
volume 541. sub-0013 run 5 and several sub-0007 runs also contain isolated FD values above
1.5 mm. These spikes are sparse, so maximum FD alone should not be used to remove an entire
participant.

## Interpretation

The cohort does not support excluding an entire participant or run solely for pervasive head
motion under the descriptive thresholds above. For inferential analyses, retain motion
confounds and censor or model high-FD volumes according to a criterion chosen before testing
the scientific contrasts. Sensitivity analyses should pay particular attention to sub-0009
and sub-0007 because they have the greatest overall movement, and to the isolated spikes in
sub-0000 and sub-0013.

This assessment quantifies rigid-body motion estimates. It does not replace visual inspection
of fMRIPrep reports for registration, susceptibility distortion, dropout, or other image-quality
problems.
