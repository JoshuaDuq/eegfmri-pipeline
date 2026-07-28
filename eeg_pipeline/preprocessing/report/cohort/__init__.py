"""Cohort roll-up of the per-subject preprocessing evidence.

The subject report answers whether one recording is usable. It cannot answer whether a
study was preprocessed consistently, how much of it is affected by a given failure, or
whether the cohort kept any brain signal through cleaning -- and those are the questions
that decide whether a group analysis is viable.

This package reads the QC sidecars the subject report stage writes and assembles them into
one document. It measures nothing itself: aggregating recorded measurements rather than
recomputing them is what makes it impossible for a cohort figure and the subject figure
beneath it to disagree.
"""

from __future__ import annotations
