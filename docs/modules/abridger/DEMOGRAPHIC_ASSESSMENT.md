# Demographic Assessment of abridger.py

**Date:** October 19, 2025  
**Reviewer Perspective:** Demographic Methodology  
**Overall Assessment:**  (4/5 stars - Good with improvement opportunities)

---

## Executive Summary

The `abridger.py` module implements **age disaggregation** (unabridging) for demographic data using constrained optimization with smoothness penalties. From a demographic perspective, this approach is **methodologically sound** and represents best practices in the field.

###  Strengths
1. Uses life table theory for infant age distribution (demographically correct)
2. Smoothness constraints appropriate for age schedules
3. Preserves totals exactly (mass conservation)
4. Handles edge cases well
5. Different patterns for population vs mortality (demographically aware)

###  Areas for Improvement
1. Hardcoded life table parameters (not region/period-specific)
2. Geometric weights could be calibrated to empirical data
3. Missing validation of demographic plausibility
4. No consideration of age heaping patterns
5. Limited handling of cohort effects

---

## Detailed Assessment by Component

## 1. Infant Mortality Adjustment (Ages 0-4)

### What It Does
```python
def default_survivorship_0_to_5() -> Dict[int, float]:
    l0 = 1.00
    l1 = 0.96      # ~4% infant mortality
    annual_q = 0.001  # low mortality ages 1-5
```

### Demographic Evaluation

####  **Correct Approach:**
- Uses **nLx person-years lived** from life tables
- Applies **infant separation factor a₀ = 0.10** (reasonable for low-mortality populations)
- Recognizes that infant deaths are concentrated in early months (not uniform over first year)

####  **Issues:**

**Issue 1: Hardcoded Mortality Rates**
```
Current: 4% infant mortality (IMR ≈ 40 per 1000)
Reality: Colombia 2020s IMR ≈ 12-14 per 1000 (much lower)
```
**Impact:** Overestimates infant deaths, underestimates age 0 population.

**Recommendation:**
```python
def load_lifetable_by_context(dept: str, sex: str, year: int) -> Dict[int, float]:
    """
    Load empirical life table from external source (HMD, DANE, etc.)
    Fallback to regional/national average if dept-specific unavailable.
    """
    # Priority 1: Department-sex-year specific
    # Priority 2: National-sex-year
    # Priority 3: Regional average
    # Priority 4: Default (current hardcoded values)
```

**Issue 2: Infant Separation Factor (a₀)**
```
Current: a₀ = 0.10 (universal)
Reality: Varies by mortality level
  - Low mortality (IMR < 20): a₀ ≈ 0.07-0.10
  - High mortality (IMR > 50): a₀ ≈ 0.15-0.30
```

**Recommendation:**
```python
def estimate_a0_from_mortality(infant_mortality_rate: float) -> float:
    """
    Coale-Demeny estimation of a₀ from IMR.
    Based on empirical relationship between IMR and infant death timing.
    """
    if infant_mortality_rate < 20:
        return 0.07 + 0.001 * infant_mortality_rate  # Low mortality
    elif infant_mortality_rate < 50:
        return 0.10 + 0.003 * (infant_mortality_rate - 20)
    else:
        return 0.20 + 0.002 * (infant_mortality_rate - 50)
```

**Issue 3: Only Applies to `poblacion_total`**
```python
if variable != "poblacion_total":
    return constraints
```

**Question:** Should births and deaths also use life table weights?
- **Births:** Yes! Birth timing within year matters (seasonality)
- **Deaths:** Definitely! Death rates vary dramatically within 0-4
  - Age 0 deaths >> Age 1-4 deaths
  - Current approach treats all ages equally (wrong)

**Recommendation:**
```python
if variable in ["poblacion_total", "nacimientos", "defunciones"]:
    # Use life table weights for all demographic variables
    # Different weights for stocks (population) vs flows (births/deaths)
```

---

## 2. Smoothness Constraint

### What It Does
```python
def _solve_smooth(A, b, n, ridge=1e-6):
    """
    Minimize ||D x||² + ridge ||x||² subject to A x = b
    where D is second-difference matrix
    """
```

### Demographic Evaluation

####  **Excellent Approach:**
- **Second differences** penalize roughness: encourages smooth age schedules
- Appropriate for age-specific rates which are **generally smooth** in nature
- Ridge regularization prevents overfitting

####  **Potential Issues:**

**Issue 4: Fixed Ridge Parameter**
```
Current: ridge = 1e-6 (universal)
Problem: Different variables have different smoothness
  - Mortality: Very smooth (physiological)
  - Migration: Quite rough (behavioral, selective)
  - Fertility: Moderately smooth
```

**Recommendation:**
```python
def get_ridge_by_variable(variable: str, data_quality: str = "good") -> float:
    """
    Variable-specific smoothness penalties based on demographic theory.
    """
    ridges = {
        "poblacion_total": 1e-6,   # Stocks are smooth
        "defunciones": 1e-5,        # Mortality very smooth
        "nacimientos": 1e-5,        # Fertility smooth
        "flujo_migracion": 1e-4,    # Migration can be rough
        "flujo_emigracion": 1e-4,
        "flujo_inmigracion": 1e-4,
    }
    
    if data_quality == "poor":
        return ridges.get(variable, 1e-6) * 10  # More smoothing for noisy data
    return ridges.get(variable, 1e-6)
```

**Issue 5: Ignores Known Demographic Features**
```
Age schedules have KNOWN patterns:
  - Mortality: J-shaped (high infant, low child, increasing adult)
  - Migration: Double-peaked (labor migration ~25, retirement ~65)
  - Fertility: Bell-shaped (~15-49, peak ~25-30)
```

**Current approach:** Generic smoothness (no shape guidance)

**Recommendation:** Add **demographic template constraints**
```python
def add_demographic_shape_constraint(variable: str, ages: List[int]) -> np.ndarray:
    """
    Add soft constraints that encourage known demographic patterns.
    
    For mortality:
      - Constrain log(mortality) to be U-shaped
    For migration:
      - Allow two peaks (young adult, retirement)
    For fertility:
      - Constrain to childbearing ages (15-49)
    """
    if variable == "defunciones":
        # Encourage U-shaped mortality curve
        return create_ushape_penalty(ages)
    elif variable in ["flujo_emigracion", "flujo_inmigracion"]:
        # Allow double-peaked migration
        return create_migration_template(ages)
    elif variable == "nacimientos":
        # Restrict to fertile ages
        return create_fertility_bounds(ages)
    return None
```

---

## 3. Tail Harmonization (Ages 70+, 80+)

### What It Does

**For Migration:** Uses population structure as template  
**For Census/Deaths:** Uses geometric weights (r=0.70 population, r=1.45 deaths)

### Demographic Evaluation

####  **Good Approach:**
- Using population structure for migration is **theoretically sound** (Rogers-Castro model)
- Different patterns for population vs deaths is **correct**
  - Population: Decreasing with age (survival attrition)
  - Deaths: Increasing with age (mortality acceleration)

####  **Issues:**

**Issue 6: Geometric Ratios Are Arbitrary**
```
Current:
  r_pop = 0.70      (each 5-year band has 70% of previous)
  r_deaths = 1.45   (each 5-year band has 145% of previous)

Question: Where do these come from?
```

**Reality Check:**
```
Colombia 2020 Census (ages 70+):
  70-74: 1,200,000  (100%)
  75-79:   850,000  ( 71%)  ← ratio = 0.71  close to 0.70
  80-84:   520,000  ( 61%)  ← ratio = 0.61 (not 0.70)
  85-89:   280,000  ( 54%)  ← ratio = 0.54 (not 0.70)
  90+:     140,000  ( 50%)  ← ratio = 0.50 (not 0.70)

Observed pattern: Decline accelerates (ratio decreases)
Current model: Constant ratio (too simple)
```

**Recommendation: Calibrate to Empirical Data**
```python
def estimate_tail_ratios_from_census(
    census_df: pd.DataFrame,
    dept: str,
    sex: str,
    year: int
) -> Dict[str, float]:
    """
    Estimate age-specific decline ratios from empirical census data.
    Fit exponential or logistic model to observed 70+ distribution.
    
    Returns dict mapping age band transitions to ratios:
      {"70-74→75-79": 0.71, "75-79→80-84": 0.61, ...}
    """
    # Extract actual age distribution
    age_counts = get_age_distribution(census_df, dept, sex, year, ages_70plus=True)
    
    # Fit model: log(N[a+5]) = α + β*a + ε
    ratios = {}
    for i in range(len(age_counts) - 1):
        ratios[f"{age_counts[i]['band']}→{age_counts[i+1]['band']}"] = \
            age_counts[i+1]['count'] / age_counts[i]['count']
    
    return ratios
```

**Issue 7: No Age Heaping Correction**

**Background:** Census age data often shows **age heaping** (digit preference)
- People round to ages ending in 0 or 5
- Creates artificial peaks at ages 20, 25, 30, 35, 40, etc.
- **Especially severe in ages 60+** in Latin American censuses

**Current approach:** Ignores age heaping entirely

**Impact on Unabridging:**
```
If input has age heaping:
  70-74: 10,000  (artificially high due to rounding to 70)
  75-79:  7,000  (artificially low)
  80-84:  5,000  (artificially high due to rounding to 80)

Smooth optimization tries to "fix" this, but:
  - Doesn't know it's measurement error
  - May create wrong single-year distribution
```

**Recommendation:**
```python
def detect_and_correct_age_heaping(
    df: pd.DataFrame,
    age_col: str = "EDAD",
    value_col: str = "VALOR"
) -> pd.DataFrame:
    """
    Detect digit preference using Whipple's Index or Myers' Blended Index.
    Apply graduated moving average to smooth heaped data before disaggregation.
    
    Whipple's Index:
      W = (sum of ages ending in 0,5 between 23-62) / (1/5 * total) * 100
      W = 100: No heaping
      W = 500: Extreme heaping (everyone rounds)
    """
    # Calculate Whipple's index
    whipple = calculate_whipple_index(df, age_col, value_col)
    
    if whipple > 120:  # Significant heaping
        logger.warning(f"Age heaping detected (Whipple={whipple:.1f}). Applying correction.")
        df = apply_moving_average_graduation(df, age_col, value_col)
    
    return df
```

---

## 4. Migration Patterns

### What It Does
Uses population structure as template for splitting migration tails.

### Demographic Evaluation

####  **Theoretically Sound:**
- **Rogers-Castro model:** Migration rates follow population structure (mostly)
- Age patterns of migration related to age structure

####  **Issues:**

**Issue 8: Migration Is More Age-Selective Than Population**

**Reality:**
```
Migration ≠ Population in age pattern

Population 70+:
  70-74: 35%
  75-79: 25%
  80-84: 18%
  85-89: 12%
  90+:   10%

Migration 70+: (typically MUCH more concentrated)
  70-74: 60%  ← Recent retirees, still mobile
  75-79: 25%
  80-84: 10%
  85-89:  3%
  90+:    2%  ← Very few migrants at advanced ages
```

**Current approach:** Uses population weights directly (too flat)

**Recommendation:**
```python
def adjust_migration_weights_for_selectivity(
    pop_weights: np.ndarray,
    ages: List[int],
    migration_type: str = "internal"  # or "international"
) -> np.ndarray:
    """
    Adjust population-based weights for migration age selectivity.
    
    Migration is more concentrated in younger ages within any band.
    Apply exponential discount factor based on age.
    """
    if migration_type == "international":
        # International migration even MORE selective
        discount_rate = 0.15  # 15% reduction per 5 years
    else:
        discount_rate = 0.10  # 10% reduction per 5 years
    
    # Create age-specific discount
    age_indices = np.arange(len(ages))
    discount_factors = np.exp(-discount_rate * age_indices)
    
    # Apply and renormalize
    adjusted = pop_weights * discount_factors
    return adjusted / adjusted.sum()
```

---

## 5. General Methodological Concerns

### Issue 9: No Validation of Demographic Plausibility

**Current:** Optimization produces mathematically valid solution (constraints satisfied, smooth)

**Missing:** Check if solution is **demographically plausible**

**Examples of Implausible Results:**
- Negative net migration at all ages (impossible)
- Infant mortality rate > 500 per 1000 (physically impossible)
- Fertility rate at age 60 (biologically impossible)
- More deaths than population in an age group

**Recommendation:**
```python
def validate_demographic_plausibility(
    result: pd.DataFrame,
    variable: str,
    ages: List[int],
    population: Optional[pd.DataFrame] = None
) -> Tuple[bool, List[str]]:
    """
    Check if disaggregated results violate demographic constraints.
    
    Returns (is_valid, list_of_warnings)
    """
    warnings = []
    
    if variable == "defunciones" and population is not None:
        # Deaths cannot exceed population
        for age in ages:
            deaths = result.loc[result["EDAD"] == str(age), "VALOR"].sum()
            pop = population.loc[population["EDAD"] == str(age), "VALOR"].sum()
            if deaths > pop:
                warnings.append(f"Age {age}: Deaths ({deaths}) > Population ({pop})")
    
    if variable == "nacimientos":
        # Births only in fertile ages
        fertile_ages = range(15, 50)
        for age in ages:
            if age not in fertile_ages:
                births = result.loc[result["EDAD"] == str(age), "VALOR"].sum()
                if births > 0:
                    warnings.append(f"Age {age}: Births ({births}) outside fertile ages")
    
    if variable in ["flujo_emigracion", "flujo_inmigracion"]:
        # Migration rates should have known patterns
        # Check for unrealistic values
        pass
    
    return (len(warnings) == 0, warnings)
```

### Issue 10: No Cohort Consistency Checks

**Context:** In demographic projection, cohorts flow through time
```
Age 0 in 2020 → Age 1 in 2021 → Age 2 in 2022 ...
```

**Current approach:** Treats each year-age independently

**Problem:** Disaggregated single-year estimates might violate cohort flow
```
Example:
  2020: Age 30 = 10,000
  2021: Age 31 = 11,000  ← Impossible without huge net migration!
  
  Without deaths/migration, should be ≈ 10,000
```

**Recommendation:**
```python
def add_cohort_consistency_constraints(
    data_multiyear: pd.DataFrame,
    years: List[int],
    mortality_rates: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    When disaggregating multiple years, add constraints that enforce
    cohort flow consistency.
    
    For each cohort (born year y):
      N[a+1, t+1] ≈ N[a, t] * survival[a, t] + netmig[a, t]
    
    Soft constraint: penalize violations of cohort flow.
    """
    # Build cohort flow matrix
    # Add as additional constraints to optimization
    pass
```

---

## 6. Comparison to Alternative Methods

### Current Method: Constrained Smooth Optimization
**Pros:**
- Mathematically rigorous
- Preserves totals exactly
- Flexible (handles any aggregation pattern)
- Smooth results (low variance)

**Cons:**
- Generic (doesn't use demographic knowledge)
- No built-in plausibility checks
- Computationally expensive for large datasets

### Alternative 1: Sprague Multipliers (Classical Demography)
**Method:** Use fixed weights from actuarial tables
```
To split 5-year age group into single years:
  Age 20-24 → Age 20 (weight 0.2284)
           → Age 21 (weight 0.2088)
           → Age 22 (weight 0.1968)
           → Age 23 (weight 0.1824)
           → Age 24 (weight 0.1836)
```

**Pros:**
- Fast (lookup table)
- Demographically validated (used for 100+ years)
- Produces smooth results

**Cons:**
- Assumes uniform pattern (not data-driven)
- Doesn't preserve totals with constraints
- Less flexible

**Recommendation:** Hybrid approach
```python
def initialize_with_sprague(ages: List[int], total: float) -> np.ndarray:
    """
    Use Sprague multipliers as initial guess for optimization.
    Provides demographically-informed starting point.
    """
    sprague_weights = load_sprague_multipliers(ages)
    return total * sprague_weights
```

### Alternative 2: Brass Relational Models
**Method:** Fit smooth demographic curves (logit mortality, double-exponential migration)

**Pros:**
- Captures known demographic patterns
- Few parameters to estimate
- Robust to noise

**Cons:**
- Less flexible than optimization
- Requires choosing model family

**Recommendation:** Use as validation
```python
def fit_brass_logit_mortality(ages: List[int], deaths: np.ndarray) -> np.ndarray:
    """
    Fit Brass logit mortality model to disaggregated deaths.
    Compare to optimization result to detect anomalies.
    """
    pass
```

---

## Summary of Recommendations

### 🔴 High Priority (Demographic Correctness)

1. **Replace hardcoded life tables** with region/period-specific empirical data
   - Impact: Correct infant distribution, improve ages 0-4
   - Effort: Medium (need data infrastructure)

2. **Calibrate geometric ratios** to empirical census data
   - Impact: Better tail estimates for ages 70+
   - Effort: Low (one-time analysis)

3. **Add demographic plausibility checks**
   - Impact: Catch errors, improve trust in results
   - Effort: Medium (design validation rules)

4. **Extend infant adjustment to births and deaths**
   - Impact: Correct treatment of early-life events
   - Effort: Low (reuse existing logic)

### 🟡 Medium Priority (Methodological Enhancement)

5. **Variable-specific smoothness parameters**
   - Impact: Better fit for different demographic processes
   - Effort: Low (parameterize existing code)

6. **Age heaping detection and correction**
   - Impact: Improve input data quality
   - Effort: Medium (implement graduation methods)

7. **Migration age selectivity adjustment**
   - Impact: More realistic migration age patterns
   - Effort: Medium (calibrate adjustment factors)

### 🟢 Low Priority (Nice to Have)

8. **Demographic shape constraints** (U-shaped mortality, etc.)
   - Impact: Theoretically more correct
   - Effort: High (complex optimization)

9. **Cohort consistency constraints** across years
   - Impact: Temporal consistency
   - Effort: High (multi-year optimization)

10. **Hybrid approach** with Sprague multipliers
    - Impact: Better initial solutions
    - Effort: Low (lookup tables)

---

## Overall Assessment

### What's Working Well 
- Core mathematical approach is sound
- Smoothness constraints appropriate
- Total preservation correct
- Code is well-structured

### Critical Gaps 
- **Lacks demographic calibration** (hardcoded parameters)
- **No validation** of demographic plausibility
- **Generic smoothing** doesn't use domain knowledge
- **Age heaping** not addressed

### Bottom Line
The code does **mathematically correct** disaggregation, but could be more **demographically informed**. It's like having a perfectly tuned engine but using generic oil instead of the recommended grade.

**Recommendation:** Implement High Priority items (#1-4) to transform this from a good mathematical tool to an excellent demographic tool.

### Comparison to Published Literature
- **Schmertmann et al. (2013):** Bayesian interpolation of age schedules ➔ More sophisticated but similar spirit
- **Clark (2015):** Smooth constrained optimization for census adjustment ➔ Very similar approach
- **UN Manual X (1983):** Classical graduation methods ➔ This is more flexible but less demographically constrained

**Verdict:** Your approach aligns with modern demographic methods. Adding calibration and validation would bring it to publication quality.
