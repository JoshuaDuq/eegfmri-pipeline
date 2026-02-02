---
name: scientific-rigor-clean-code
description: Enforces a no-fallback, fail-fast, scientifically rigorous clean-code standard for this repository. Use whenever writing, refactoring, reviewing, or debugging code (especially scientific/EEG/fMRI pipelines), and whenever making architectural choices, validation, error handling, naming, formatting, or configuration decisions.
---

# Scientific Rigor + Clean Code (Fail Fast, No Fallbacks)

## Non-negotiables

- Do not add “fallbacks” that hide errors. Unexpected issues must surface.
- Prefer clarity and scientific rigor over cleverness.
- During refactors: preserve behavior (outputs + side effects) unless explicitly asked to change it.
- Less code is better if correctness and clarity are maintained.
- Seek root causes; do not patch symptoms.
- Keep code structured, tidy, highly organized, and non-redundant.

## When to apply

Apply this skill whenever:
- Editing or adding code, especially in scientific Python (EEG/fMRI) or CLI tooling.
- Refactoring, reorganizing modules, renaming, or “cleanup” changes.
- Adding validation or error handling.
- Designing interfaces, configs, pipelines, or data structures.

## Working protocol (how to behave)

### Before changing code

- Read the relevant code and understand current behavior.
- Identify the root cause and the smallest correct change.
- Prefer improving invariants at boundaries (entry-point validation) over scattered checks.
- Avoid broad “drive-by” changes that risk behavior drift.

### While changing code

- Keep functions small and single-purpose.
- Use descriptive, searchable names; avoid cryptic abbreviations.
- Prefer explicit data structures and stable interfaces over ad-hoc dicts.
- Keep code vertically dense and well-grouped; avoid alignment tricks and long lines.
- Validate assumptions at entry points (types, shapes, ranges). Fail fast with clear errors.
- Catch only specific expected exceptions; never blanket-catch to continue silently.

### After changing code

- Ensure outputs/side effects are preserved for refactors.
- Remove dead/commented-out code rather than disabling it.
- Keep changes minimal, consistent, and easy to review.

## Standards checklist (encompasses the rules)

### Philosophy

1. Prioritize clarity + scientific rigor over cleverness.
2. Do not change behavior when refactoring; preserve outputs and side effects.
3. Prefer less code if correctness and clarity are maintained.
4. Apply the boy scout rule: leave code cleaner than you found it.
5. Always seek the root cause; don’t patch symptoms.

### Scope and responsibility

6. Enforce single responsibility for each function/module.
7. Keep functions small and focused (“do one thing”).
8. Prefer fewer arguments; avoid long parameter lists.
9. Avoid flag arguments; split into separate functions instead.
10. Avoid side effects unless they are the point of the function.

### Readability and intent

11. Write code to be read and modified by someone else on the team.
12. Use explanatory variables to clarify complex expressions.
13. Encapsulate boundary conditions (edge cases handled in one place).
14. Avoid negative conditionals when a positive form is clearer.
15. Avoid logical dependency (a method shouldn’t rely on hidden state/order elsewhere).

### Naming rules (domain-aware)

16. Use descriptive, unambiguous, searchable names.
17. Use pronounceable names; avoid cryptic abbreviations.
18. Functions are verbs; variables are nouns.
19. Avoid type-encoding prefixes/suffixes in names.

### Style and formatting (consistency)

21. Follow PEP 8 consistently (scientific Python norms).
22. Keep lines short; avoid horizontal alignment tricks.
23. Use whitespace to group related code; don’t break indentation.
24. Keep related code vertically dense; separate distinct concepts vertically.
25. Declare variables close to usage.

### File and code organization

26. Organize imports and definitions in a clear dependency-to-implementation flow.
27. Keep dependent/similar functions near each other.
28. Place functions in downward direction (helpers above callers).
29. Use section dividers sparingly, only when they genuinely help:

```
###################################################################
# Section Name
###################################################################
```

### Error handling and validation

30. Validate assumptions at entry points: types, shapes, ranges.
31. Catch only specific expected exceptions; let unexpected errors surface.
32. Avoid silent failures and defensive over-engineering.
33. Prefer failing fast over hidden fallbacks.

### Scientific computing practices (EEG/fMRI)

34. Prefer established libraries: MNE-Python, Nilearn, NiBabel, NumPy/SciPy.
35. Follow domain conventions: BIDS, and standard formats (NIfTI/CIFTI/FIF).
36. Prefer vectorized NumPy operations over explicit loops when it improves clarity.
37. Ensure reproducibility (deterministic steps where applicable; controlled randomness).
38. Write code that is testable/validatable (clear inputs/outputs, minimal hidden state).

### Design rules (when architecture matters)

39. Keep configuration in dedicated .yaml (not scattered constants).
40. Prevent over-configurability (only expose what users must tune).
41. Prefer polymorphism over long if/elif chains when it reduces complexity.
42. Use dependency injection where it improves testability and decoupling.
43. Follow Law of Demeter: depend only on direct collaborators.
44. Keep multi-threading concerns separated from core logic.

### Objects and data structures

45. Hide internal structure; expose stable interfaces.
46. Prefer clean data structures over hybrids (half-object, half-dict).
47. Keep objects small: few instance variables; do one thing.
48. Base classes should know nothing about their derivatives.
49. Prefer many clear functions over “pass code/flags to select behavior”.
50. Prefer instance methods over static methods when behavior depends on object state.

### Comments policy (code explains itself)

51. Prefer expressing intent in code; don’t narrate obvious statements.
52. Comments are for intent, clarifications, warnings—not noise.
53. Do not comment out dead code; remove it.

### Code smells to actively prevent

54. Rigidity (small change causes many changes).
55. Fragility (one change breaks many places).
56. Immobility (hard to reuse parts elsewhere).
57. Needless complexity and needless repetition.
58. Opacity (hard to understand).

