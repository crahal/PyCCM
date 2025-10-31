# Known Issues and Limitations - PyCCM

**Last Updated:** October 31, 2025

This document tracks known limitations and issues in the PyCCM library that users should be aware of.

---

## 🔴 Critical Issues

### Issue #1: Population Age 0-4 Unabridging Produces Uniform Distribution

**Module:** `src/abridger.py`  
**Function:** `unabridge_df()`, `_apply_infant_adjustment()`  
**Severity:** Medium-High  
**Discovered:** October 31, 2025  
**Status:** Open

#### Description
When unabridging population data with only an aggregated `0-4` age group (no separate age 0), the function produces near-uniform weights (~20% per age) instead of using demographically realistic age patterns based on actual mortality.

#### Evidence
```python
# Observed for BOLIVAR 2018 Female Population
Age 0: 15,847.2  (20.10%)
Age 1: 15,773.6  (20.01%)
Age 2: 15,757.8  (19.99%)
Age 3: 15,742.1  (19.97%)
Age 4: 15,726.3  (19.95%)

# Compare to deaths (which work correctly):
Age 0: 90.59   (30.19%)  ← Correctly higher
Age 1: 74.91   (24.97%)
Age 2: 59.41   (19.80%)
Age 3: 44.46   (14.82%)
Age 4: 30.62   (10.21%)  ← Correctly lower
```

#### Root Cause
- `_apply_infant_adjustment()` uses hardcoded generic survivorship values (`l1 = 0.96`, `annual_q = 0.001`)
- These produce nearly equal person-years lived (nLx) across ages 0-4
- No mechanism exists to pass actual life table values to `unabridge_df()`
- Deaths work correctly because original data has separate `0-1` and `2-4` categories

#### Impact
- Population projections starting from unabridged data have slightly incorrect initial age structure
- Infant survival ratios computed from unabridged population are less accurate
- Birth estimates derived from age 0 population may be biased
- Effect compounds over multi-year projections

#### Workaround
See `docs/modules/abridger/DEMOGRAPHIC_IMPROVEMENTS_QUICKSTART.md` for function `post_correct_age_0_4_population()` that manually re-weights using actual life tables.

#### Proposed Solution
1. Add `lifetable: Optional[pd.DataFrame]` parameter to `unabridge_df()`
2. Extract `lx` values from provided life table
3. Pass actual `lx` to `_apply_infant_adjustment()`
4. Fallback to generic values only when no life table available

#### References
- Full analysis: `.github_issue_infant_adjustment.md`
- Documentation: `docs/modules/abridger/DEMOGRAPHIC_ASSESSMENT.md` (lines 110-180)
- Quickstart fix: `docs/modules/abridger/DEMOGRAPHIC_IMPROVEMENTS_QUICKSTART.md` (lines 7-95)

---

## 🟡 Medium Priority Issues

### Issue #2: Hardcoded Life Table Parameters

**Module:** `src/abridger.py`  
**Function:** `default_survivorship_0_to_5()`  
**Severity:** Medium  
**Status:** Open

#### Description
The default survivorship values assume 4% infant mortality (IMR ≈ 40 per 1000), which is too high for Colombia in 2020s (actual IMR ≈ 12-14 per 1000).

#### Impact
- Overestimates infant deaths
- Can produce incorrect age 0 population when actually implemented with proper life tables

#### Proposed Solution
Load region/period-specific life tables from UN World Population Prospects or DANE (Colombia statistics agency).

---

### Issue #3: Silent Fallback to Geometric Weights in Tail Harmonization

**Module:** `src/abridger.py`  
**Function:** `_weights_from_pop_or_geometric()`  
**Severity:** Low-Medium  
**Status:** Open

#### Description
When population data is missing or invalid for tail age groups (70+, 80+), function silently falls back to geometric weights (r=0.60) without warning user.

#### Impact
- User might expect population-weighted split but get geometric pattern instead
- No indication when/why fallback occurred

#### Proposed Solution
Add logging or return status indicator when fallback is triggered.

---

## 🟢 Low Priority / Enhancement Requests

### Variable-Specific Smoothness Parameters

**Module:** `src/abridger.py`  
**Current:** Fixed `ridge=1e-6` for all variables  
**Request:** Different smoothness penalties based on demographic process type
- Mortality: Very smooth (physiological) → lower ridge
- Migration: Can be rough (behavioral) → higher ridge
- Fertility: Moderately smooth → medium ridge

---

## Resolved Issues

*None yet*

---

## Contributing

If you discover a new issue or limitation:
1. Check this document to see if it's already known
2. Test with minimal reproducible example
3. Document the issue with evidence (code, output, expected behavior)
4. Open a GitHub issue or submit PR with fix

---

## Version History

- **v0.1.0** (October 2025): Initial known issues documentation
  - Added Issue #1: Uniform Age 0-4 distribution
  - Added Issue #2: Hardcoded life tables
  - Added Issue #3: Silent geometric fallback
