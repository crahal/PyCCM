# Fertility Module Explanation

## Overview
The `fertility.py` module handles **fertility rate calculations** and **target TFR (Total Fertility Rate) parameters** for demographic projections. It contains 2 main functions for computing Age-Specific Fertility Rates (ASFR) and loading projection targets.

---

## Function 1: `get_target_params(file_path)`

### Purpose
Reads a CSV file containing target TFR values by department and optional convergence years for demographic projections.

### Input
- `file_path`: Path to CSV with target fertility parameters

### Output
- `targets`: Dictionary mapping department name → target TFR value
- `conv_years`: Dictionary mapping department name → convergence year

### Example CSV Format
```csv
DPTO_NOMBRE,Target_TFR,convergence_year
AMAZONAS,1.0,2050
ANTIOQUIA,0.9,2045
BOGOTA,0.8,2040
```

### Logic Flow

1. **Read CSV file**
2. **Flexibly detect columns** using fuzzy matching:
   - Department: looks for "DPTO_NOMBRE", or columns containing "dpto" + "nombre"
   - TFR: looks for "Target_TFR", or columns containing "tfr" or "target"
   - Convergence: looks for columns containing "converge" + "year"
3. **Parse each row**:
   - Extract department name (stripped of whitespace)
   - Convert TFR to float (skip if invalid/NaN)
   - Convert convergence year to int (optional, skip if missing)
4. **Return dictionaries**

### Robustness Features
- **Flexible column detection**: Handles spelling variations ("convergeance_year", "convergence_years", etc.)
- **Graceful degradation**: Skips invalid values rather than crashing
- **Optional convergence**: Works even if convergence column missing

### Use Case in Demographic Projection
```
Current TFR (2020): 1.8 births per woman
Target TFR (2050): 1.0 births per woman
Convergence year: 2050

Projection interpolates from 1.8 → 1.0 over 30 years
After 2050, TFR stays at 1.0 (replacement level policy)
```

---

## Function 2: `compute_asfr(ages, population, births, *, min_exposure=1e-9, nonneg_asfr=True)`

### Purpose
Calculate **Age-Specific Fertility Rates (ASFR)** from population counts and birth counts.

### Formula
$$\text{ASFR}_a = \frac{\text{Births}_a}{\text{Population}_a}$$

Where:
- $\text{ASFR}_a$ = Fertility rate at age $a$ (births per woman per year)
- $\text{Births}_a$ = Number of births to mothers age $a$
- $\text{Population}_a$ = Number of women age $a$ (exposure)

### Parameters
- `ages`: List/Index of age labels (e.g., ["15", "16", ..., "49"])
- `population`: Series/array of female population by age
- `births`: Series/array of births by mother's age
- `min_exposure`: Minimum population threshold (default: 1e-9)
- `nonneg_asfr`: Clip negative rates to zero (default: True)

### Output
DataFrame with columns:
- `population`: Aligned female population
- `births`: Aligned birth counts
- `asfr`: Calculated fertility rates

### Logic Flow

1. **Convert inputs to Series** (preserves labels)
2. **Normalize age labels** (strip whitespace, convert to strings)
3. **Find common ages** present in all three: ages, population, births
4. **Reindex to common ages** (aligns data by labels)
5. **Apply guardrails**:
   - Clip births to ≥ 0 (no negative births)
   - Set population ≤ min_exposure to NaN (avoid division by tiny numbers)
6. **Calculate ASFR** = births / population
7. **Handle edge cases**:
   - Replace Inf/-Inf with NaN
   - Fill NaN with 0.0
   - Optionally clip to non-negative
8. **Return aligned DataFrame**

### Robustness Features

#### Problem: Misaligned Age Labels
```python
population.index = ["15", "16", "17", ...]
births.index = ["15 ", "16", " 17", ...]  # whitespace!
```
**Solution:** Strip and normalize all labels

#### Problem: Division by Zero
```python
population[age] = 0  # No women of this age
births[age] = 5      # But somehow 5 births?!
asfr = 5 / 0 = Inf   # Mathematical error
```
**Solution:** Set small populations to NaN, then fill with 0.0

#### Problem: Negative Values
```python
births[age] = -10   # Data error
asfr = -10 / 1000 = -0.01  # Impossible rate
```
**Solution:** Clip births to ≥ 0, optionally clip ASFR to ≥ 0

#### Problem: Missing Age Groups
```python
ages = ["15", "16", ..., "49"]  # Request all fertile ages
population has ages ["15", "20", "25", ...]  # 5-year groups
births has ages ["15", "16", ..., "49"]  # Single years
```
**Solution:** Only compute ASFR for intersection of available ages

---

## Demographic Context

### What is ASFR?
**Age-Specific Fertility Rate**: The number of live births per woman at a specific age during a year.

**Example:**
```
Age 25 women: 100,000
Births to age 25 mothers: 8,000
ASFR₂₅ = 8,000 / 100,000 = 0.08 = 80 per 1000 women
```

### What is TFR?
**Total Fertility Rate**: Sum of ASFRs across all reproductive ages (typically 15-49).

$$\text{TFR} = \sum_{a=15}^{49} \text{ASFR}_a$$

**Interpretation:**
- TFR = 2.1 → Replacement level (population stable)
- TFR = 1.0 → Below replacement (population declining)
- TFR = 3.0 → Above replacement (population growing)

**Colombia Context:**
- 1970s: TFR ≈ 5.0 (high fertility)
- 1990s: TFR ≈ 2.8 (declining)
- 2020s: TFR ≈ 1.8 (below replacement)
- Target: TFR ≈ 1.0-1.4 by 2050 (varies by department)

### Typical ASFR Pattern
```
Age    ASFR    Description
15     0.010   Very low (teenage)
20     0.080   Rising
25     0.120   Peak fertility
30     0.090   Declining
35     0.050   Lower
40     0.015   Rare
45     0.002   Very rare
49     0.000   End of fertility
```

The pattern is **bell-shaped** with peak around age 25-30.

---

## Integration with Cohort-Component Model

### How This Module Fits

```
Demographic Projection Pipeline:
1. [mortality.py] → Calculate death rates
2. [migration.py] → Calculate net migration
3. [fertility.py] → Calculate birth rates ← THIS MODULE
4. [projections.py] → Project population forward
```

### Usage in Projection

```python
# 1. Load target TFRs
targets, conv_years = get_target_params("data/target_tfrs.csv")
# targets = {"BOGOTA": 0.8, "AMAZONAS": 1.0, ...}

# 2. Calculate current ASFR from census data
current_asfr = compute_asfr(
    ages=["15", "16", ..., "49"],
    population=female_pop_by_age,
    births=births_by_mother_age
)

# 3. Calculate current TFR
current_tfr = current_asfr["asfr"].sum()  # e.g., 1.8

# 4. Project future ASFR
target_tfr = targets.get("BOGOTA", 1.0)  # 0.8
conv_year = conv_years.get("BOGOTA", 2050)  # 2050

# Linear interpolation from current to target
for year in range(2020, conv_year + 1):
    progress = (year - 2020) / (conv_year - 2020)
    projected_tfr = current_tfr + progress * (target_tfr - current_tfr)
    
    # Scale ASFR to match projected TFR
    scaling_factor = projected_tfr / current_tfr
    projected_asfr = current_asfr["asfr"] * scaling_factor
    
    # Use projected_asfr to calculate births in cohort-component model
    births_by_age = female_pop_by_age * projected_asfr

# 5. After convergence, hold constant
if year > conv_year:
    projected_tfr = target_tfr  # Fixed at target
```

---

## Key Design Decisions

### 1. Why Return Both Targets AND Conv_Years?
**Flexibility:** Some departments may converge faster/slower than others.

**Example:**
- Bogotá (urban, educated): converge to 0.8 by 2040 (fast)
- La Guajira (rural, lower education): converge to 1.9 by 2060 (slow)

### 2. Why min_exposure Parameter?
**Numerical stability:** Dividing by very small populations creates noise.

**Example:**
```python
population[age=48] = 0.001  # Tiny sample
births[age=48] = 1          # One birth recorded
asfr = 1 / 0.001 = 1000.0   # Absurdly high rate!
```

**Better:**
```python
if population < 1e-9:
    asfr = 0.0  # Treat as no data
```

### 3. Why Label-Based Alignment?
**Robustness:** Data may have different age groups or missing ages.

**Example:**
```python
# Population: 5-year groups
pop.index = ["15-19", "20-24", "25-29", ...]

# Births: single years (some missing)
births.index = ["15", "16", "18", "19", "20", ...]  # Missing age 17!

# compute_asfr only calculates where both exist
# Avoids creating fake ASFR for age 17 (no birth data)
```

### 4. Why Optional nonneg_asfr?
**Physical constraint:** Fertility rates cannot be negative.

**But:** Sometimes you want to detect data errors:
```python
# With nonneg_asfr=False
asfr[age=20] = -0.05  # ← Red flag! Data error!

# With nonneg_asfr=True (default)
asfr[age=20] = 0.0    # Silently corrected (may hide issues)
```

---

## Comparison to Standard Demographic Methods

### Method 1: This Module (Direct ASFR Calculation)
**Approach:** Births / Population (with guardrails)

**Pros:**
- Simple, transparent
- Handles missing data gracefully
- Flexible age groupings

**Cons:**
- No smoothing (noisy if small populations)
- Doesn't enforce biological constraints
- No model-based patterns

### Method 2: Brass Relational Model
**Approach:** Fit standard fertility schedule, adjust with relational parameters

**Pros:**
- Smooth results
- Based on empirical patterns
- Few parameters

**Cons:**
- Assumes standard pattern (may not fit all populations)
- More complex

### Method 3: Gompertz/Hadwiger Models
**Approach:** Fit parametric curve to ASFR

**Pros:**
- Very smooth
- Biologically motivated shape

**Cons:**
- Requires optimization
- Can't capture irregular patterns

**This module's approach is appropriate for:**
- High-quality data (census-based, large samples)
- Situations requiring transparency
- When preserving observed patterns matters

**Consider alternatives when:**
- Data is noisy (small populations)
- Need to forecast without targets
- Want to enforce biological realism

---

## Potential Use Cases

### 1. Historical ASFR Calculation
```python
# Calculate fertility rates from census
asfr_2018 = compute_asfr(
    ages=range(15, 50),
    population=census_2018_female_by_age,
    births=vital_stats_2018_births_by_mother_age
)
```

### 2. Subnational Comparisons
```python
for dept in departments:
    asfr = compute_asfr(
        ages=fertile_ages,
        population=pop_by_dept[dept],
        births=births_by_dept[dept]
    )
    tfr = asfr["asfr"].sum()
    print(f"{dept}: TFR = {tfr:.2f}")
```

### 3. Projection Targets
```python
targets, conv = get_target_params("policy_scenarios/low_fertility.csv")
# Use in long-term demographic projections
```

### 4. Validation & Quality Checks
```python
asfr_df = compute_asfr(...)

# Check for impossible values
assert (asfr_df["asfr"] >= 0).all(), "Negative ASFR detected!"
assert (asfr_df["asfr"] <= 0.5).all(), "ASFR > 0.5 (50% rate) unrealistic!"

# Check TFR in plausible range
tfr = asfr_df["asfr"].sum()
assert 0.5 < tfr < 8.0, f"TFR {tfr:.2f} outside realistic bounds"
```

---

## Summary

### Module Purpose
Supports **fertility component** of cohort-component population projection model.

### Core Functionality
1. **Load projection targets** (where fertility should go)
2. **Calculate current rates** (where fertility is now)

### Key Features
- Robust to data imperfections (missing ages, misaligned labels, edge cases)
- Flexible column detection
- Numerical stability guardrails

### Demographic Role
- Bridge between **observed fertility** (historical data) and **projected fertility** (policy targets)
- Enables department-specific, age-specific fertility modeling in population projections
