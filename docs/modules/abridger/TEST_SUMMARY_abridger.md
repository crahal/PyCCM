# abridger.py - Module Summary and Test Report

## Module Overview

The `abridger.py` module performs **age disaggregation (unabridging)** for demographic data, converting aggregated age groups (e.g., "0-4", "70+") into single-year age estimates. This is essential for cohort-component demographic projections.

## Key Components

### 1. Age Parsing Functions
- **`parse_edad(label)`**: Parses age labels into (min, max) tuples
  - Handles ranges: "0-4" → (0, 4)
  - Handles open-ended: "80+" → (80, None)
  - Handles single ages: "5" → (5, 5)
  
- **`collapse_convention2(df)`**: Applies Convention 2 where "a-(a+1)" denotes single year age a
  - Converts (a, a+1) → (a, a) for consistency

### 2. Life Table Weight Functions
Used for splitting the critical 0-4 age group with demographic realism:

- **`default_survivorship_0_to_5()`**: Returns illustrative survivorship l_x values
  - l₀ = 1.00 (birth)
  - l₁ = 0.96 (~4% infant mortality)
  - l₂-l₅ with low annual mortality

- **`nLx_1year(lx, a0=0.10)`**: Calculates person-years lived (nLx) for ages 0-4
  - Uses infant separation factor a₀ = 0.10 (deaths occur early in first year)
  - Formula: L₀ = l₁ + a₀(l₀ - l₁)
  - For ages 1-4: Lₓ = 0.5(lₓ + lₓ₊₁)

- **`weights_from_nLx(L, ages)`**: Converts nLx to distribution weights (sum to 1)
  - Accounts for differential survival by age
  - Falls back to uniform weights if data unavailable

### 3. Smoothing Solver
Implements constrained optimization for smooth age distributions:

- **`_second_diff_matrix(n)`**: Creates (n-2)×n second-difference matrix D
  - Row i: [0...0, 1, -2, 1, 0...0] at positions i, i+1, i+2
  - Measures curvature: D·x approximates second derivative

- **`_solve_smooth(A, b, n, ridge=1e-6)`**: Solves constrained smoothing problem
  - **Objective**: Minimize ||D·x||² + ridge·||x||² (smoothness + regularization)
  - **Constraints**: A·x = b (preserve aggregated totals)
  - **Method**: KKT system via least squares
  - **Output**: Non-negative smooth single-year estimates

### 4. Core Unabridging Functions

#### `_apply_infant_adjustment(constraints, variable, lx, a0)`
Special handling for population in 0-4 age group:

**Scenario 1**: Has 0-4 bin AND single age 0
- Extract total for age 0
- Split remainder (1-4) using nLx weights

**Scenario 2**: Has 0-4 bin only (no age 0)
- Split entire 0-4 using nLx weights

**Scenario 3**: Has 1-4 bin
- Split 1-4 using nLx weights

**Why?**: Infant mortality (age 0) is much higher than ages 1-4, so equal splitting would be demographically unrealistic.

#### `_unabridge_one_group(g, series_keys, variable, value_col, ridge)`
Disaggregates one demographic series:
1. Extracts constraints (lo, hi, total) from age bins
2. Applies infant adjustment if variable is "poblacion_total"
3. Builds linear system A·x = b where:
   - x = single-year estimates
   - A = aggregation matrix (maps x to bins)
   - b = observed bin totals
4. Solves for smooth x using `_solve_smooth()`

#### `unabridge_df(df, series_keys, value_col, ridge)`
Main unabridging function for DataFrames:
- Groups by series keys (department, sex, year, variable, etc.)
- Calls `_unabridge_one_group()` for each series
- Passes through open-ended ages (80+, 90+) and unparsable labels
- Returns DataFrame with single-year ages

### 5. Tail Harmonization Functions

#### `harmonize_migration_to_90plus(mig, pop, series_keys, ...)`
Splits migration data tails using population structure:

- **Splits "70+"** → [70-74, 75-79, 80-84, 85-89, 90+]
- **Splits "80+"** → [80-84, 85-89, 90+]
- **Weights**: Uses matching population shares when available
- **Fallback**: Geometric weights with r=0.60 (younger > older)
- **Preserves**: Total migration flows

**Why population weights?**: Migration patterns correlate with population age structure.

#### `harmonize_conteos_to_90plus(df, series_keys, value_col, r_pop, r_deaths)`
Splits conteos (counts) tails using geometric weights:

**For poblacion_total**:
- Uses r_pop=0.70 with decreasing pattern (younger bins > older bins)
- Reflects population age structure reality

**For defunciones (deaths)**:
- Uses r_deaths=1.45 with increasing pattern (older bins > younger bins)
- Reflects mortality concentration at oldest ages

#### `_geom_weights(bands, r)`
Generates geometric weight series using w_j ∝ r^j:
- **r < 1** (e.g., 0.7): Weights decrease with age → younger bins larger (for population)
- **r > 1** (e.g., 1.45): Weights increase with age → older bins larger (for deaths)
- **r = 1**: All bins equal weight
- Always normalized to sum to 1

### 6. Public API

#### `unabridge_all(df, emi, imi, series_keys, conteos_value_col, ridge)`
Unified interface for unabridging all datasets:
- **df**: Conteos (population, deaths) - uses `conteos_value_col`
- **emi**: Emigration - uses `VALOR`
- **imi**: Immigration - uses `VALOR`

Returns dictionary:
```python
{
    'conteos': DataFrame with single-year ages,
    'emi': DataFrame with single-year ages,
    'imi': DataFrame with single-year ages
}
```

#### `save_unabridged(objs, out_dir)`
Persists unabridged results to CSV files:
- Creates output directory if needed
- Saves as `{key}_unabridged_single_year.csv`

## Mathematical Details

### Constrained Smoothing Problem

Given age bins with totals, find single-year estimates x that:

1. **Satisfy constraints**: Each bin total equals sum of its single years
   - Example: x₅ + x₆ + x₇ + x₈ + x₉ = 1000 (for 5-9 bin)

2. **Minimize curvature**: Prefer smooth age patterns
   - Minimize Σ(xᵢ - 2xᵢ₊₁ + xᵢ₊₂)²
   - Penalizes jagged distributions

3. **Non-negative**: x ≥ 0 (counts can't be negative)

**Solution method**: Karush-Kuhn-Tucker (KKT) conditions
- Lagrangian: L(x,λ) = ||Dx||² + λᵀ(Ax - b)
- Solve block system: [2Q  Aᵀ] [x] = [0]
                       [A   0 ] [λ]   [b]
- Clip negatives for numerical stability

## Test Suite Results

### Test Coverage: 46 tests total
-  **36 tests passing** (78%)
-  **10 tests failing** (22%)

### Passing Test Categories

#### Age Parsing (8/8 tests) 
- Range parsing ("0-4", "15-19")
- Open-ended parsing ("80+")
- Single age parsing ("5")
- Edge cases (None, NaN, invalid input)
- Whitespace handling
- Reversed ranges

#### Convention 2 Collapsing (2/3 tests) 
- Adjacent age collapsing (a-(a+1) → a)
- Non-adjacent preservation

#### Life Table Functions (4/4 tests) 
- Default survivorship generation
- Person-years (nLx) calculation
- Weight normalization
- Missing data fallback

#### Second Difference Matrix (3/3 tests) 
- Matrix shape validation
- Coefficient correctness
- Edge cases (n < 3)

#### Smoothing Solver (3/3 tests) 
- Constraint satisfaction
- Multiple constraints
- Smoothness preference

#### Infant Adjustment (5/5 tests) 
- Non-population passthrough
- 0-4 splitting with age 0 present
- 0-4 splitting without age 0
- 1-4 splitting
- Total preservation

#### One-Group Unabridging (3/3 tests) 
- Simple unabridging
- Total preservation
- Empty group handling

#### Weight Functions (2/2 tests) 
- Population-based weights
- Geometric fallback

#### Tail Harmonization Conteos (2/3 tests) 
- Deaths 80+ splitting (increasing pattern)
- Non-target variable passthrough

#### Utility Functions (3/3 tests) 
- CSV saving
- Directory creation
- Default constants

### Failing Tests (10)

1. **TestCollapseConvention2::test_handles_nan**
   - Issue: NaN handling in EDAD_MAX column
   - Expected behavior differs from implementation

2. **TestUnabridgeDf** (2 tests)
   - Missing DPTO_CODIGO column in test data
   - Quick fix: Add all series_keys columns to test DataFrames

3. **TestGeomWeights::test_decreasing_weights**
   - Assertion logic error (decreasing vs increasing parameter)
   - Quick fix: Adjust test expectations or parameter

4. **TestHarmonizeMigrationTo90Plus** (3 tests)
   - DataFrame construction error (mismatched list lengths)
   - Quick fix: Ensure all DataFrame columns have same length

5. **TestHarmonizeConteosTo90Plus::test_splits_population_70_plus**
   - Similar DataFrame construction issue
   - Quick fix: Verify column alignment

6. **TestUnabridgeAll** (2 tests)
   - Missing series_keys columns in test data
   - Quick fix: Add all required columns

### Quick Fixes Needed

Most failures are test data issues, not code bugs:
- Add missing DPTO_CODIGO column to test DataFrames
- Fix DataFrame construction (ensure equal column lengths)
- Adjust geometric weight test expectations
- Handle NaN edge case in convention2 test

## Usage Examples

### Basic Unabridging
```python
import pandas as pd
from abridger import unabridge_df

# Load data with age groups
df = pd.read_csv('conteos.csv')

# Unabridge to single years
single_year = unabridge_df(
    df,
    value_col='VALOR_corrected',
    ridge=1e-6
)
```

### Full Pipeline
```python
from abridger import unabridge_all, save_unabridged

# Unabridge all datasets
results = unabridge_all(
    df=conteos,
    emi=emigration,
    imi=immigration,
    conteos_value_col='VALOR_corrected'
)

# Save results
save_unabridged(results, './output/unabridged')
```

### Tail Harmonization
```python
from abridger import harmonize_migration_to_90plus

# Split 70+ and 80+ tails
harmonized = harmonize_migration_to_90plus(
    mig=migration_data,
    pop=population_data,
    series_keys=['DPTO_NOMBRE', 'SEXO', 'ANO']
)
```

## Key Design Decisions

1. **Why smoothing?** 
   - Avoids unrealistic age patterns
   - Demographic data is naturally smooth
   - Regularization prevents overfitting

2. **Why life table weights for infants?**
   - Infant mortality >> mortality at ages 1-4
   - Equal splitting would underestimate age 0
   - Demographic realism essential for projections

3. **Why different tail patterns?**
   - Population: More young-old (70-74) than old-old (90+)
   - Deaths: Concentrated at oldest ages
   - Reflects demographic reality

4. **Why constrained optimization?**
   - Must preserve observed totals (constraint)
   - Want smooth distributions (objective)
   - Standard demographic methodology

## Dependencies
- numpy: Numerical computations
- pandas: Data manipulation
- Standard library: os, re, typing

## Recommendations

1. **For Production Use**: 
   - Provide actual life tables by department/sex/year
   - Tune ridge parameter for your data characteristics
   - Validate smoothing results visually

2. **For Testing**:
   - Fix DataFrame construction issues
   - Add missing series_keys columns
   - Consider parametrized tests for different scenarios

3. **For Enhancement**:
   - Add confidence intervals for estimates
   - Support custom smoothing penalties
   - Parallel processing for large datasets
   - Visualization tools for QA
