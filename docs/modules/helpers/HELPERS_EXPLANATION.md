# helpers.py: Utility Functions Module

## Overview

The `helpers.py` module provides a comprehensive suite of utility functions that support the cohort-component demographic projection pipeline. These functions are designed to be pure, side-effect-free (except for file I/O), and reusable across different modules.

**Module Purpose:** Centralize common operations to avoid circular dependencies and facilitate unit testing.

**Design Philosophy:**
- Pure functions with no side effects
- No imports from project-specific modules
- Liberal input handling (case-insensitive, flexible formats)
- Explicit demographic validation where appropriate

## Function Categories

### 1. Age Scaffolding Utilities

#### `_single_year_bins() -> list[str]`
Returns single-year age labels from 0 to 89, plus an open-ended "90+" category.

**Use Case:** Creating standard age structures for demographic data

**Output:** `['0', '1', '2', ..., '89', '90+']` (91 labels total)

#### `_bin_width(label: str) -> float`
Calculates the effective width of an age bin for weighting purposes.

**Convention:**
- `'lo-hi'` → `hi - lo + 1` (closed interval)
- `'X+'` → `5.0` (open-ended bins treated as width 5)
- `'k'` → `1.0` (single year)

**Examples:**
```python
_bin_width('0-4')   # 5.0
_bin_width('25-29') # 5.0
_bin_width('85+')   # 5.0
_bin_width('42')    # 1.0
```

#### `_widths_from_index(idx) -> np.ndarray`
Vectorized version of `_bin_width` that operates on an entire index.

**Use Case:** Computing weights for TFR calculations or rate aggregations

### 2. CSV Parsing Helpers

#### `_find_col(df: pd.DataFrame, must_include: list[str]) -> str | None`
Performs liberal column name matching to handle variations in input data.

**Matching Rules:**
- Case-insensitive
- Ignores whitespace and underscores
- All substrings in `must_include` must be present

**Use Case:** Robust data loading when column names vary across sources

**Example:**
```python
_find_col(df, ['dpto', 'nombre'])  # Finds 'DPTO_NOMBRE' or 'dpto nombre'
_find_col(df, ['age'])             # Finds 'Age', 'AGE_GROUP', etc.
```

**Returns:** Column name if found, `None` otherwise

#### `get_midpoint_weights(file_path: str) -> dict`
Reads departmental midpoint weights from a CSV file.

**Expected Format:**
- Column with department names (searches for 'DPTO_NOMBRE' or similar)
- Column with weights (searches for 'mid', 'weight', etc.)

**Returns:** `{department_name: weight}` mapping

**Use Case:** Loading spatial weights for interpolation or adjustment

### 3. Fertility/TFR Functions

#### `_tfr_from_asfr_df(asfr_df: pd.DataFrame) -> float`
Computes Total Fertility Rate as the width-weighted sum of Age-Specific Fertility Rates.

**Input Requirements:**
- DataFrame with column 'asfr'
- Index contains age labels (e.g., '15-19', '20-24', etc.)

**Formula:** 
$$\text{TFR} = \sum_{i} \text{ASFR}_i \times w_i$$

where $w_i$ is the width of age bin $i$

**Demographic Validation:**
- Result is implicitly validated through test suite (expected range: 0.3-10.0)
- Extreme values would indicate data quality issues

#### `_exp_tfr(TFR0: float, target: float, years: int, step: int, converge_frac: float = 0.99) -> float`
Exponential approach to target TFR.

**Mathematical Model:**
$$\text{gap}_t = \text{gap}_0 \times e^{-\kappa t}$$

where $\kappa = -\ln(1 - \text{converge\_frac}) / \text{years}$

**Parameters:**
- `TFR0`: Initial TFR value
- `target`: Target TFR to converge toward
- `years`: Convergence horizon
- `step`: Current time step (years elapsed)
- `converge_frac`: Fraction of gap to close by end of horizon (default 0.99)

**Properties:**
- Monotonic convergence (always moves toward target)
- Smooth trajectory (no sudden jumps)
- Asymptotic approach (never quite reaches target)

**Demographic Validation:**
- Tested with extreme initial values (0.5-9.0)
- All intermediate values stay within plausible range (0.3-10.0)
- No sudden jumps (validated in test suite)

#### `_logistic_tfr(TFR0: float, target: float, years: int, step: int, mid_frac: float = 0.5, steepness: float | None = None) -> float`
Logistic (S-curve) approach to target TFR.

**Mathematical Model:**
$$\text{TFR}(t) = \text{target} + \text{gap}_0 \times \frac{\text{scale}}{1 + e^{s(t - t_0)}}$$

where $t_0 = \text{mid\_frac} \times \text{years}$

**Parameters:**
- `TFR0`: Initial TFR value
- `target`: Target TFR
- `years`: Convergence horizon
- `step`: Current time step
- `mid_frac`: Fraction of horizon at inflection point (default 0.5)
- `steepness`: Curve steepness (auto-calibrated if None)

**Properties:**
- S-shaped transition
- Faster convergence in middle period
- Slower convergence at beginning and end
- More realistic for demographic transitions

**Use Case:** Modeling demographic transitions (fertility decline, post-conflict recovery)

**Demographic Validation:**
- Tested with extreme initial values
- Monotonic convergence guaranteed
- Stays within plausible TFR range

#### `_smooth_tfr(TFR0: float, target: float, years: int, step: int, kind: str = "exp", **kwargs) -> float`
Dispatcher function for TFR smoothing paths.

**Parameters:**
- `kind`: `"exp"` for exponential, `"logistic"` for S-curve
- Additional `kwargs` passed to chosen function

**Use Case:** Unified interface for projections module

**Error Handling:** Raises `ValueError` for invalid `kind`

### 4. Alignment and Reindexing

#### `fill_missing_age_bins(s: pd.Series, edad_order: list[str]) -> pd.Series`
Reindexes a Series to a standard age structure, filling missing bins with 0.

**Use Case:** Ensuring consistent age structure across datasets

**Properties:**
- Preserves existing values
- Adds zeros for missing age bins
- Returns float dtype

**Example:**
```python
s = pd.Series([100, 200], index=['0-4', '10-14'])
result = fill_missing_age_bins(s, ['0-4', '5-9', '10-14'])
# Result: [100.0, 0.0, 200.0]
```

#### `_ridx(s_like, edad_order: list[str]) -> pd.Series`
Advanced reindexing with duplicate handling.

**Features:**
- Accepts Series or array-like input
- Sums duplicate index entries
- Fills missing entries with 0
- Preserves data type

**Use Case:** Handling messy input data with potential duplicates

**Example:**
```python
s = pd.Series([50, 50, 100], index=['0-4', '0-4', '5-9'])
result = _ridx(s, ['0-4', '5-9', '10-14'])
# Result: [100.0, 100.0, 0.0]  # Duplicates summed
```

### 5. Specialized Preprocessing

#### `_collapse_defunciones_01_24_to_04(df_def: pd.DataFrame) -> pd.DataFrame`
Collapses '0-1' and '2-4' age bins into a single '0-4' bin for death data.

**Context:** Used in abridged population processing before omission corrections

**Grouping Keys:**
- DPTO_NOMBRE, DPTO_CODIGO (geography)
- ANO, SEXO (time and demographics)
- VARIABLE, FUENTE (data type and source)

**Aggregation Rules:**
- `VALOR`, `VALOR_withmissing`: Summed
- `OMISION`: Maximum level observed
- `VALOR_corrected`: Set to NaN (computed later)

**Returns:** DataFrame with '0-4' replacing any '0-1'/'2-4' pairs

**Use Case:** Harmonizing infant/child mortality data with different age breakdowns

## Demographic Validation

### Built-in Checks

The helpers module implements demographic validation through:

1. **TFR Bounds Testing** (in test suite)
   - All TFR values tested to stay within 0.3-10.0
   - Covers extreme initial conditions
   - Validates both exponential and logistic paths

2. **Monotonicity Guarantees**
   - Exponential and logistic functions mathematically guarantee monotonic convergence
   - Tested in multiple scenarios

3. **Smoothness Constraints**
   - No sudden jumps in TFR trajectories
   - Validated through difference testing

4. **Age Structure Consistency**
   - Age bin widths calculated consistently
   - Missing bin handling preserves totals
   - Duplicate handling prevents data loss

### Implicit Validation

Several functions provide implicit validation:

- `_bin_width()`: Always returns positive values (>= 1.0)
- `fill_missing_age_bins()`: Preserves sum of existing values
- `_ridx()`: Handles duplicates without data loss
- `_tfr_from_asfr_df()`: Mathematically correct integration

## Design Patterns

### 1. Pure Functions
All functions are stateless and side-effect-free (except file I/O in `get_midpoint_weights`).

**Benefits:**
- Easy to test
- Easy to reason about
- Safe for parallel execution
- No hidden dependencies

### 2. Liberal Input Handling
Functions accept flexible input formats and handle edge cases gracefully.

**Examples:**
- `_find_col()`: Case-insensitive, whitespace-tolerant
- `_bin_width()`: Handles single years, ranges, and open-ended bins
- `_ridx()`: Accepts Series, arrays, or list-like objects

### 3. Explicit Type Hints
All functions include type hints for better IDE support and documentation.

### 4. No Circular Dependencies
Helpers module imports only numpy and pandas, never project modules.

**Benefit:** Can be imported by any module without circular dependency issues

## Testing Coverage

**Test File:** `tests/test_helpers.py`

**Coverage:** 40 comprehensive tests across 6 test classes:

1. **TestAgeScaffolding** (6 tests)
   - Single-year bin generation
   - Bin width calculation (all formats)
   - Vectorized width calculation

2. **TestCSVHelpers** (6 tests)
   - Column matching (exact, case-insensitive, partial)
   - Midpoint weight loading
   - Not-found handling

3. **TestFertilityTFRFunctions** (13 tests)
   - TFR calculation from ASFR
   - Exponential convergence (decreasing and increasing)
   - Logistic convergence and S-curves
   - Method dispatcher
   - Demographic bounds validation

4. **TestAlignmentFunctions** (6 tests)
   - Missing bin filling
   - Advanced reindexing
   - Duplicate handling
   - Subset operations

5. **TestSpecializedFunctions** (4 tests)
   - Defunciones collapsing (all scenarios)
   - Column preservation
   - Edge cases (no data to collapse, partial data)

6. **TestDemographicValidation** (5 tests)
   - Extreme TFR values (exponential and logistic)
   - Rapid convergence smoothness
   - Realistic ASFR structures
   - Age bin consistency
   - Total preservation

**All 40 tests passing** ✓

## Integration Points

### Used By:
- `fertility.py`: TFR calculations and smoothing
- `mortality.py`: Age bin alignment
- `migration.py`: Age structure handling
- `abridger.py`: Defunciones preprocessing, age harmonization
- `data_loaders.py`: CSV column matching, midpoint weights
- `main_compute.py`: All utility functions

### Dependencies:
- numpy: Numerical operations
- pandas: DataFrame/Series operations
- os, re: File and string operations (minimal use)

## Recommendations

### Strengths
1. **Well-designed API**: Clear function names, consistent signatures
2. **Comprehensive Documentation**: Detailed docstrings for all functions
3. **No Side Effects**: Pure functions facilitate testing and reasoning
4. **Flexible Input Handling**: Robust to data variations
5. **Complete Test Coverage**: 40 tests cover all major functionality

### Demographic Quality
- TFR functions enforce plausible bounds through testing
- Age structure handling preserves data integrity
- Mathematical formulations guarantee monotonic convergence

### Potential Enhancements (Low Priority)
1. **Add Explicit TFR Validation**: Consider adding a `validate_tfr()` function similar to `validate_asfr()` in fertility module
2. **Age Bin Validation**: Could add explicit validation of age bin formats
3. **Enhanced Error Messages**: More descriptive errors for invalid inputs
4. **Performance Optimization**: Vectorize TFR smoothing for multiple steps at once

### Overall Assessment
**Rating: 5/5**

The helpers module is production-ready with:
- Clean, maintainable code
- Comprehensive test coverage
- Strong demographic foundations
- No critical issues identified

This module serves as an excellent foundation for the demographic projection pipeline.
