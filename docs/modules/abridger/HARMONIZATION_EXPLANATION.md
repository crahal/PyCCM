# Tail Harmonization Functions - Detailed Explanation

## Overview
These functions solve the "tail harmonization" problem: converting open-ended age groups like "70+" or "80+" into standard 5-year age bands (70-74, 75-79, 80-84, 85-89, 90+) to enable consistent demographic analysis.

---

## Function 1: `_weights_from_pop_or_geometric()`

### What It Does
**Purpose:** Calculate weights to split an open-ended age group into standard bands.

**Logic Flow:**
1. **Try population-based weights first** (lines 333-342):
   - Extract population counts for each target band from `pop_g` DataFrame
   - Calculate proportional weights: `w_i = pop_i / total_pop`
   - If successful, return normalized weights

2. **Fallback to geometric weights** (lines 343-345):
   - If population data is missing/invalid, use geometric decay
   - Formula: `w_j = r^j` where j=0,1,2,... (age index)
   - Default r=0.60 means each older band gets 60% of previous band's weight

**Example:**
```python
# Population-based: [1000, 800, 500, 300, 100] → weights [0.37, 0.30, 0.19, 0.11, 0.04]
# Geometric (r=0.6): [1.00, 0.60, 0.36, 0.22, 0.13] → weights [0.43, 0.26, 0.15, 0.09, 0.06]
```

### Potential Issues

#### Issue 1: **Silent Fallback to Geometric Weights**
**Location:** Lines 338-342
```python
if np.isfinite(S) and S > 0:
    w = np.nan_to_num(v / S, nan=0.0, posinf=0.0, neginf=0.0)
    s = w.sum()
    if s > 0:
        return w / s
# Falls through to geometric without warning
```
**Problem:** No indication when population data is ignored (e.g., all zeros, NaN values, or missing EDAD labels).

**Impact:** User might expect population-weighted split but get geometric instead.

**Fix:** Add logging or return indicator:
```python
import logging
logger = logging.getLogger(__name__)

if np.isfinite(S) and S > 0:
    # ... population weights ...
    return w / s
logger.warning(f"Population data unavailable/invalid, using geometric weights (r={r})")
# ... geometric fallback ...
```

#### Issue 2: **String Matching Fragility**
**Location:** Line 333
```python
v = np.array([float(pop_g.loc[pop_g["EDAD"].astype(str).str.strip().eq(b), pop_value_col].sum())
              for b in bins], dtype=float)
```
**Problem:** 
- Exact string match required: "70-74" won't match " 70-74 " or "70 - 74"
- Multiple rows with same EDAD label are summed (might be correct or bug depending on data)

**Test Cases Needed:**
- Whitespace variations in EDAD labels
- Multiple rows per age band (valid aggregation?)
- Missing age bands (returns 0, then geometric fallback)

#### Issue 3: **Zero-Division Edge Case**
**Location:** Lines 339-342
```python
w = np.nan_to_num(v / S, nan=0.0, posinf=0.0, neginf=0.0)
s = w.sum()
if s > 0:
    return w / s
```
**Problem:** If all `v[i] = 0` but `S > 0`, impossible. But if `np.nan_to_num` creates all zeros and `s=0`, falls through to geometric.

**Edge Case:** All population bands have zero population → geometric fallback (correct but silent).

---

## Function 2: `harmonize_migration_to_90plus()`

### What It Does
**Purpose:** Split migration data's "70+" and "80+" tails using population structure as a template.

**Logic Flow:**
1. **Validate inputs** (lines 364-377):
   - Check migration has required columns: series_keys + EDAD + value_col
   - Check population has value column (fallback VALOR → VALOR_corrected if needed)

2. **Process each group** (lines 381-420):
   - Group migration by series_keys (e.g., by department, year, sex)
   - Find matching population group with same series_keys values
   - Split "70+" into 5 bands using population weights
   - Split "80+" into 3 bands using population weights
   - Keep all other rows unchanged

3. **Collapse duplicates** (lines 423-427):
   - Aggregate by (series_keys, EDAD) to handle overlapping splits
   - Sum values for duplicate age bands

**Example:**
```
Input migration:
  DPTO  SEX  EDAD   VALOR
  001   M    65-69  100
  001   M    70+    500

Population template:
  DPTO  SEX  EDAD     VALOR_corrected
  001   M    70-74    1000
  001   M    75-79     800
  001   M    80-84     500
  001   M    85-89     300
  001   M    90+       100

Output migration:
  DPTO  SEX  EDAD     VALOR
  001   M    65-69    100
  001   M    70-74    185.2  (500 * 1000/2700)
  001   M    75-79    148.1  (500 * 800/2700)
  001   M    80-84     92.6  (500 * 500/2700)
  001   M    85-89     55.6  (500 * 300/2700)
  001   M    90+       18.5  (500 * 100/2700)
```

### Potential Issues

#### Issue 4: **Population Mismatch Silently Falls Back**
**Location:** Lines 386-392
```python
pop_g = pop.copy()
if not isinstance(keys, tuple):
    keys = (keys,)
for c, v in zip(series_keys, keys):
    pop_g = pop_g[pop_g[c].eq(v)]
```
**Problem:** If no population rows match the migration group's keys, `pop_g` becomes empty.
- `_weights_from_pop_or_geometric()` then uses geometric weights
- No warning that demographic-specific population structure was unavailable

**Test Case:** Migration for DPTO=999 but population only has DPTO=001,002.

#### Issue 5: **Duplicate EDAD After Split**
**Location:** Lines 395-417
**Problem:** If input has both "70+" AND "70-74", after splitting "70+" you get two "70-74" rows.

**Example:**
```
Input:
  EDAD   VALOR
  70+    500
  70-74  100

After split "70+":
  70-74  185  (from 70+ split)
  70-74  100  (original)

After groupby sum:
  70-74  285  ✓ (correct aggregation)
```
**Resolution:** Lines 423-427 correctly handle this with `.groupby().sum()`.

#### Issue 6: **Both 70+ and 80+ Present** ✅ RESOLVED
**Location:** Lines 430-461
**Status:** Code correctly handles this case via sequential processing.

**What the code does (lines 431-434):**
```python
mask_70p = g["EDAD"].astype(str).str.strip().eq("70+")
if mask_70p.any():
    total_70p = float(g.loc[mask_70p, value_col].sum())
    g = g.loc[~mask_70p].copy()  # ← KEY: Removes 70+ from g
```
Then checks for 80+ in the **modified** `g` (line 443).

**Example execution:**
```
Input (single group):
  EDAD   VALOR
  65-69    50
  70+     500
  80+     200

Step 1: Process 70+
  - Extract 500, remove 70+ row from g
  - Split: 70-74, 75-79, 80-84, 85-89, 90+ (totaling 500)
  - g now = [65-69: 50, 80+: 200]

Step 2: Process 80+ (from remaining g)
  - Extract 200, remove 80+ row from g
  - Split: 80-84, 85-89, 90+ (totaling 200)
  - g now = [65-69: 50]

Step 3: Final groupby aggregation (lines 458-461)
  - 65-69: 50
  - 70-74: (from 70+ split only)
  - 75-79: (from 70+ split only)
  - 80-84: (70+ split) + (80+ split) ← SUMMED
  - 85-89: (70+ split) + (80+ split) ← SUMMED
  - 90+:   (70+ split) + (80+ split) ← SUMMED
```

**Interpretation:** This treats 70+ and 80+ as **independent, additive records**.
- If they represent separate data sources (e.g., two reporting systems): **CORRECT**
- If 80+ is a refined subset of 70+ (double-counted): **PROBLEMATIC**

**Real-world context:** In Colombian migration data, departments report with different age categorizations. Having both 70+ and 80+ in the same (DPTO, YEAR, SEX) group would indicate **data quality issues** (mixed reporting conventions), but the code handles it gracefully by summing rather than crashing.

**Recommendation:** Add data validation warning (not error) if both present:
```python
if mask_70p.any() and mask_80p.any():
    import warnings
    warnings.warn(f"Group {keys} has both '70+' and '80+'; treating as independent records")
```

#### Issue 7: **Missing Columns in Output**
**Location:** Lines 398-404, 411-417
```python
row = {**{c: v for c, v in zip(series_keys, keys)}}
row["EDAD"] = band
row[value_col] = float(total_70p * w)
out_rows.append(row)
```
**Problem:** Only includes `series_keys + EDAD + value_col`. If input has extra columns (e.g., "FUENTE", "OMISION"), they're lost.

**Test Case:** Input has 10 columns, output has only 5.

---

## Function 3: `harmonize_conteos_to_90plus()`

### What It Does
**Purpose:** Split "70+" and "80+" in census counts (poblacion_total) and deaths (defunciones) using **geometric weights** tailored to each variable.

**Key Difference from Migration Function:**
- **No population template needed** (uses pure geometric weights)
- **Different geometric ratios** for population vs deaths:
  - Population: r=0.70 (r < 1 → decreasing weights, younger bands larger)
  - Deaths: r=1.45 (r > 1 → increasing weights, older bands larger)

**How Geometric Weights Work:**
The `_geom_weights(bands, r)` function computes weights using w_j ∝ r^j:
- **r < 1**: Weights decrease with age (e.g., 0.7^0=1.0, 0.7^1=0.7, 0.7^2=0.49)
- **r > 1**: Weights increase with age (e.g., 1.4^0=1.0, 1.4^1=1.4, 1.4^2=1.96)
- **r = 1**: All bands get equal weight

**Logic Flow:**
1. **Validate and group** (lines 469-489)
2. **Check VARIABLE type** (lines 491-493):
   - Only process "poblacion_total" and "defunciones"
   - Pass through all other variables unchanged
3. **Split tails with geometric weights** (lines 496-530)
4. **Aggregate duplicates** (lines 533-537)

**Example - Population (r=0.70, decreasing):**
```
Input: 70+ = 1000
Geometric sequence: [1.0, 0.7, 0.49, 0.34, 0.24] (r^j for j=0,1,2,3,4)
Weights (normalized): [0.36, 0.26, 0.18, 0.13, 0.09]
Output: 70-74=356, 75-79=256, 80-84=179, 85-89=125, 90+=88
```

**Example - Deaths (r=1.45, increasing):**
```
Input: 70+ = 100
Geometric sequence: [1.0, 1.45, 2.10, 3.05, 4.42] (r^j for j=0,1,2,3,4)
Weights (normalized): [0.08, 0.12, 0.18, 0.25, 0.37]
Output: 70-74=8, 75-79=12, 80-84=18, 85-89=25, 90+=37
```

### Potential Issues

#### Issue 8: **Hardcoded Geometric Ratios**
**Location:** Lines 461-462
```python
r_pop: float = 0.70,     # population tail: younger>older
r_deaths: float = 1.45   # deaths tail:    older>younger
```
**Problem:** These ratios might not match actual demographic patterns in the data.

**Test:** Compare geometric split vs actual single-year data (if available) to validate ratios.

**Improvement:** Make these configurable per department/year if mortality patterns vary significantly.

#### Issue 9: **Case-Sensitive Variable Matching**
**Location:** Line 491
```python
var_cmp = var_val.lower()
if var_cmp not in {"poblacion_total", "defunciones"}:
```
**Problem:** Converts to lowercase for comparison, but original spelling preserved. Could mismatch if data uses "Poblacion_Total" or "POBLACION_TOTAL".

**Test Cases:**
- "POBLACION_TOTAL" → should process (lowercase comparison works)
- "poblacion total" (with space) → won't match (should it?)
- "poblacion_total " (trailing space) → won't match after `.strip()` on line 490

#### Issue 10: **Deduplication Logic**
**Location:** Lines 477-478, 534-535
```python
group_top = list(dict.fromkeys(series_keys + ["VARIABLE"]))
# ...later...
group_final = list(dict.fromkeys(series_keys + ["VARIABLE", "EDAD"]))
```
**Purpose:** Remove duplicate columns from grouping list (Pandas raises error if duplicates in `.groupby()`).

**Problem:** If "VARIABLE" is already in `series_keys`, this handles it. But why would it be?

**Test Case:** Verify behavior when `series_keys = ["DPTO", "VARIABLE", "SEX"]`.

#### Issue 11: **Preserved vs Lost Rows**
**Location:** Line 531
```python
out_rows.extend(g.to_dict("records"))
```
**Problem:** After splitting 70+ and 80+, **all remaining rows** from group `g` are added, including any rows that might have been modified.

**Edge Case:** If group has "70-74" originally, and we split "70+", we add split results (lines 507-509) then add ALL remaining rows (line 531). This creates duplicates that are summed in line 537.

**Is this intentional?** Yes, for aggregation. But could be confusing if not expected.

---

## Summary of Critical Issues

### High Priority
1. **Issue 6:** Both 70+ and 80+ in same group → potential double-counting
2. **Issue 4:** Missing population data → silent geometric fallback
3. **Issue 7:** Extra columns dropped from output

### Medium Priority
4. **Issue 1:** Silent fallback to geometric weights (no logging)
5. **Issue 8:** Hardcoded geometric ratios may not fit data
6. **Issue 2:** String matching fragility with whitespace

### Low Priority (Edge Cases)
7. **Issue 9:** Case/whitespace variations in VARIABLE names
8. **Issue 10:** Deduplication logic (defensive, likely unnecessary)
9. **Issue 3:** Zero-division edge case (well-handled)

---

## Testing Strategy

### Test Categories
1. **Happy Path:** Standard inputs produce expected splits
2. **Edge Cases:** Empty groups, missing data, zero populations
3. **String Variations:** Whitespace, case sensitivity
4. **Duplicate Handling:** Multiple 70+/80+ rows, overlapping bands
5. **Fallback Behavior:** Population mismatch triggers geometric weights
6. **Variable Filtering:** Only process target variables
7. **Weight Validation:** Verify weights sum to 1.0, preserve totals

### Key Assertions
- Total population/migration/deaths preserved after split
- Weights always sum to 1.0 (±1e-10)
- Output has expected columns (no extras lost)
- Duplicates correctly aggregated
- Edge cases don't crash (empty DataFrames, NaN values)
