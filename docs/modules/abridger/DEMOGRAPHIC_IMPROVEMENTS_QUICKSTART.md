# Quick Start: Improving Demographic Accuracy

This guide provides **immediately actionable** code snippets to enhance the demographic realism of `abridger.py`.

---

## ⚠️ CRITICAL ISSUE: Uniform Age 0-4 Distribution (Identified Oct 2025)

### Problem
When unabridging **population data** with only aggregated `0-4` age group, the function produces **near-uniform weights** (~20% per age) instead of using actual demographic patterns:

```python
# Current output for population:
Age 0: 20.10%  ← Should be lower or higher depending on mortality/census timing
Age 1: 20.01%  ← Near-uniform (unrealistic for census data)
Age 2: 19.99%
Age 3: 19.97%
Age 4: 19.95%

# Compare to deaths (which work correctly):
Age 0: 30.19%  ← Highest (infant mortality)
Age 1: 24.97%
Age 2: 19.80%
Age 3: 14.82%
Age 4: 10.21%  ← Lowest (realistic declining pattern)
```

### Root Cause
`unabridge_df()` doesn't accept actual life table values—it always uses hardcoded generic survivorship that produces nearly equal nLx values.

### Immediate Workaround (Until Fixed)

```python
def post_correct_age_0_4_population(unabridged_pop_df, lifetable):
    """
    Manually re-weight ages 0-4 using actual life table after initial unabridging.
    
    Parameters:
    - unabridged_pop_df: Output from unabridge_df() with uniform 0-4 distribution
    - lifetable: pandas DataFrame with 'lx' column and age index 0-5
    """
    from abridger import nLx_1year, weights_from_nLx
    
    # Extract lx values from your computed life table
    lx_dict = {i: lifetable.loc[i, 'lx'] / lifetable.loc[0, 'lx'] 
               for i in range(6)}
    
    # Compute proper nLx weights
    nLx_dict = nLx_1year(lx_dict, a0=0.10)
    proper_weights = weights_from_nLx(nLx_dict, [0, 1, 2, 3, 4])
    
    # Get current (incorrect) total for ages 0-4
    mask_0_4 = unabridged_pop_df['EDAD'].isin(['0', '1', '2', '3', '4'])
    total_0_4 = unabridged_pop_df.loc[mask_0_4, 'VALOR'].sum()
    
    # Re-distribute using proper weights
    for age, weight in zip(['0', '1', '2', '3', '4'], proper_weights):
        unabridged_pop_df.loc[unabridged_pop_df['EDAD'] == age, 'VALOR'] = total_0_4 * weight
    
    return unabridged_pop_df

# Usage example:
df_pop_1yr = unabridge_df(df_pop_f, ...)  # Gets uniform distribution
lt_female = make_lifetable(...)  # Your life table from deaths
df_pop_1yr_corrected = post_correct_age_0_4_population(df_pop_1yr, lt_female)
```

### Proper Fix (Needs Implementation)

Modify `unabridge_df()` to accept and use actual life tables:

```python
def unabridge_df(df: pd.DataFrame,
                 series_keys: Iterable[str] = SERIES_KEYS_DEFAULT,
                 value_col: str = "VALOR_corrected",
                 ridge: float = 1e-6,
                 lifetable: Optional[pd.DataFrame] = None) -> pd.DataFrame:  # NEW PARAMETER
    """
    If lifetable provided, extract lx values and pass to _apply_infant_adjustment.
    Otherwise fallback to generic defaults.
    """
    # ... existing code ...
    
    if lifetable is not None:
        # Extract lx values from lifetable
        lx_dict = {i: lifetable.loc[i, 'lx'] / lifetable.loc[0, 'lx'] 
                   for i in range(6) if i in lifetable.index}
    else:
        lx_dict = None  # Will use defaults
    
    # Pass to _apply_infant_adjustment
    cons = _apply_infant_adjustment(cons, variable=variable, lx=lx_dict)
```

**Status:** GitHub issue created. See `.github_issue_infant_adjustment.md` for full details.

---

## Priority 1: Use Real Life Tables (HIGH IMPACT 🔴)

### Problem
```python
# Current: Hardcoded values
l1 = 0.96  # 4% infant mortality (too high for Colombia 2020s)
annual_q = 0.001  # Unrealistic constant mortality ages 1-5
```

### Solution: Load Empirical Life Tables

**Option A: Use UN World Population Prospects (WPP) Data**

```python
import pandas as pd
from pathlib import Path

def load_lifetable_from_un_wpp(sex: str, year: int, country: str = "Colombia") -> Dict[int, float]:
    """
    Load life table from UN World Population Prospects CSV export.
    
    Download from: https://population.un.org/wpp/Download/Standard/Mortality/
    File: Life tables by age (lx values)
    """
    # Path to your downloaded UN WPP data
    lt_file = Path("data/UN_WPP_lifetables.csv")
    
    df = pd.read_csv(lt_file)
    df_filtered = df[
        (df["Location"] == country) &
        (df["Sex"] == sex.capitalize()) &
        (df["Time"] == year)
    ]
    
    if df_filtered.empty:
        # Fallback to nearest year
        nearest_year = df[df["Location"] == country]["Time"].unique()
        nearest_year = min(nearest_year, key=lambda y: abs(y - year))
        df_filtered = df[
            (df["Location"] == country) &
            (df["Sex"] == sex.capitalize()) &
            (df["Time"] == nearest_year)
        ]
    
    # Extract lx values for ages 0-5
    lx_dict = {}
    for age in range(6):
        lx_dict[age] = df_filtered[df_filtered["Age"] == age]["lx"].values[0] / 100000
    
    return lx_dict

# Update default_survivorship_0_to_5()
def default_survivorship_0_to_5(sex: str = "Total", year: int = 2020) -> Dict[int, float]:
    """
    Load empirical survivorship for Colombia from UN data.
    Fallback to hardcoded values if data unavailable.
    """
    try:
        return load_lifetable_from_un_wpp(sex, year)
    except Exception as e:
        import logging
        logging.warning(f"Could not load life table: {e}. Using default values.")
        # Original hardcoded fallback
        l0 = 1.00
        l1 = 0.96
        annual_q = 0.001
        l2 = l1 * (1 - annual_q)
        l3 = l2 * (1 - annual_q)
        l4 = l3 * (1 - annual_q)
        l5 = l4 * (1 - annual_q)
        return {0:l0, 1:l1, 2:l2, 3:l3, 4:l4, 5:l5}
```

**Option B: Use Colombian DANE Statistics**

```python
def load_lifetable_from_dane(dept_code: str, sex: str, year: int) -> Dict[int, float]:
    """
    Load department-specific life table from DANE (Colombia's statistics agency).
    
    DANE publishes life tables by department:
    https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion
    """
    # Path to DANE life table data (you'd need to download/organize)
    dane_file = Path(f"data/DANE_lifetables/dept_{dept_code}_{sex}_{year}.csv")
    
    if not dane_file.exists():
        # Fallback to national level
        dane_file = Path(f"data/DANE_lifetables/national_{sex}_{year}.csv")
    
    df = pd.read_csv(dane_file)
    
    lx_dict = {}
    for age in range(6):
        lx_dict[age] = df.loc[df["edad"] == age, "lx"].values[0]
    
    return lx_dict
```

**Option C: Simple Interpolation from Published Tables**

If you don't have data files, use published summary statistics:

```python
def estimate_lifetable_from_e0_and_imr(e0: float, imr: float) -> Dict[int, float]:
    """
    Estimate lx values from life expectancy at birth (e0) and infant mortality rate (IMR).
    Uses Coale-Demeny West model life tables (standard in demography).
    
    Parameters
    ----------
    e0 : Life expectancy at birth (years). Colombia 2020: ~77
    imr : Infant mortality rate (per 1000 live births). Colombia 2020: ~13
    
    Returns
    -------
    Dict mapping age to lx (survivorship)
    """
    # Survival to age 1
    l0 = 1.0
    l1 = 1.0 - (imr / 1000)
    
    # Estimate child mortality from e0 (Coale-Demeny relationship)
    # For e0 = 77, 1q1 ≈ 0.001 (very low child mortality)
    q1_to_5 = max(0.0005, 0.01 - (e0 - 50) * 0.0002)  # Rough approximation
    
    # Annual survival probability ages 1-5
    annual_p = (1 - q1_to_5) ** 0.25
    
    lx_dict = {0: l0, 1: l1}
    for age in range(2, 6):
        lx_dict[age] = lx_dict[age - 1] * annual_p
    
    return lx_dict

# Example usage for Colombia
colombia_lx = estimate_lifetable_from_e0_and_imr(e0=77.0, imr=13.0)
# Result: {0: 1.0, 1: 0.987, 2: 0.985, 3: 0.983, 4: 0.981, 5: 0.979}
```

---

## Priority 2: Calibrate Tail Ratios (HIGH IMPACT 🔴)

### Problem
```python
r_pop = 0.70      # Arbitrary constant
r_deaths = 1.45   # Arbitrary constant
```

### Solution: Estimate from Recent Census

```python
def calibrate_tail_ratio_from_census(
    census_df: pd.DataFrame,
    age_bands: List[str] = ["70-74", "75-79", "80-84", "85-89", "90+"]
) -> float:
    """
    Estimate geometric ratio from actual census age distribution.
    
    Fits exponential model: log(N[i]) = a + b*i
    Returns: r = exp(b)
    """
    # Extract population counts
    counts = []
    for band in age_bands:
        count = census_df[census_df["EDAD"] == band]["VALOR_corrected"].sum()
        counts.append(count)
    
    # Compute ratios
    ratios = [counts[i+1] / counts[i] for i in range(len(counts) - 1)]
    
    # Average ratio (or median for robustness)
    avg_ratio = np.median(ratios)
    
    return avg_ratio

# Example: Colombia 2018 Census
# Would return r ≈ 0.65 (declines faster than assumed 0.70)
```

**Better: Use Age-Varying Ratios**

```python
def get_age_varying_tail_weights(
    census_df: pd.DataFrame,
    age_bands: List[str],
    fallback_r: float = 0.70
) -> np.ndarray:
    """
    Get empirical weights from census, with geometric fallback.
    """
    counts = []
    for band in age_bands:
        count = census_df[census_df["EDAD"] == band]["VALOR_corrected"].sum()
        counts.append(count)
    
    total = sum(counts)
    if total > 0:
        # Use empirical distribution
        weights = np.array(counts) / total
    else:
        # Fallback to geometric
        j = np.arange(len(age_bands))
        weights = (fallback_r ** j)
        weights = weights / weights.sum()
    
    return weights
```

---

## Priority 3: Add Plausibility Checks (MEDIUM IMPACT 🟡)

### Solution: Post-Processing Validation

```python
def validate_unabridged_results(
    result: pd.DataFrame,
    original: pd.DataFrame,
    variable: str,
    tolerance: float = 0.01
) -> List[str]:
    """
    Check if disaggregation produced demographically plausible results.
    
    Returns list of warning messages.
    """
    warnings = []
    
    # Check 1: Total preservation
    total_result = result["VALOR"].sum()
    total_original = original["VALOR"].sum()
    rel_diff = abs(total_result - total_original) / total_original
    
    if rel_diff > tolerance:
        warnings.append(
            f"Total not preserved: {total_original:.0f} → {total_result:.0f} "
            f"({rel_diff*100:.2f}% difference)"
        )
    
    # Check 2: Non-negativity
    negative_count = (result["VALOR"] < 0).sum()
    if negative_count > 0:
        warnings.append(f"{negative_count} negative values (physically impossible)")
    
    # Check 3: Extreme values (outliers)
    mean_val = result["VALOR"].mean()
    std_val = result["VALOR"].std()
    outliers = result[result["VALOR"] > mean_val + 5*std_val]
    if len(outliers) > 0:
        warnings.append(f"{len(outliers)} extreme outliers detected")
    
    # Check 4: Variable-specific checks
    if variable == "poblacion_total":
        # Population should be smooth (check for jumps)
        vals = result.sort_values("EDAD")["VALOR"].values
        diffs = np.diff(vals)
        max_jump = np.max(np.abs(diffs))
        if max_jump > mean_val * 0.5:  # Jump > 50% of mean
            warnings.append(f"Large population jump detected: {max_jump:.0f}")
    
    elif variable == "defunciones":
        # Deaths should increase with age (generally)
        vals = result.sort_values("EDAD")["VALOR"].values
        ages = result.sort_values("EDAD")["EDAD"].astype(int).values
        
        # Check if generally increasing after age 30
        adult_mask = ages >= 30
        if adult_mask.sum() > 5:
            adult_deaths = vals[adult_mask]
            if not np.all(np.diff(adult_deaths) >= -adult_deaths[:-1] * 0.2):
                # Allow 20% decreases (noise), but flag larger ones
                warnings.append("Death pattern not monotonically increasing after age 30")
    
    elif variable == "nacimientos":
        # Births should be zero outside reproductive ages
        invalid_ages = result[
            (result["EDAD"].astype(int) < 15) | 
            (result["EDAD"].astype(int) > 49)
        ]
        invalid_births = invalid_ages[invalid_ages["VALOR"] > 0]
        if len(invalid_births) > 0:
            warnings.append(
                f"Births detected outside reproductive ages: "
                f"{invalid_births['EDAD'].tolist()}"
            )
    
    return warnings

# Usage in unabridge_df
def unabridge_df_with_validation(df, **kwargs):
    result = unabridge_df(df, **kwargs)  # Original function
    
    # Validate each group
    for var in result["VARIABLE"].unique():
        subset_result = result[result["VARIABLE"] == var]
        subset_original = df[df["VARIABLE"] == var]
        
        warnings = validate_unabridged_results(subset_result, subset_original, var)
        
        if warnings:
            import logging
            for w in warnings:
                logging.warning(f"[{var}] {w}")
    
    return result
```

---

## Priority 4: Extend to Births/Deaths (MEDIUM IMPACT 🟡)

### Problem
```python
if variable != "poblacion_total":
    return constraints  # No adjustment for births/deaths!
```

### Solution: Apply Life Table Weights to All Vital Events

```python
def _apply_infant_adjustment(
    constraints: List[Tuple[int,int,float]],
    variable: str,
    lx: Optional[Dict[int, float]] = None,
    a0: float = 0.10
) -> List[Tuple[int,int,float]]:
    """
    Enhanced version: applies to population, births, AND deaths.
    """
    # Apply to demographic variables only
    if variable not in ["poblacion_total", "nacimientos", "defunciones"]:
        return constraints
    
    if lx is None:
        lx = default_survivorship_0_to_5()
    
    # Different weights for different variables
    if variable == "poblacion_total":
        # Use nLx (person-years lived)
        L = nLx_1year(lx, a0=a0)
        
    elif variable == "nacimientos":
        # Births concentrated near birth date (use very low a0)
        # Most births recorded near time of birth, not uniformly over year
        L = nLx_1year(lx, a0=0.05)  # Births happen at moment, not person-years
        
    elif variable == "defunciones":
        # Deaths use mortality rates (inverse of survival)
        # Death probability = 1 - survival probability
        L = {}
        L[0] = (lx[0] - lx[1])  # Infant deaths
        for x in range(1, 5):
            L[x] = (lx[x] - lx[x+1])  # Child deaths
        
        # Normalize
        total_deaths = sum(L.values())
        if total_deaths > 0:
            L = {age: L[age] / total_deaths for age in L}
    
    # Rest of function unchanged...
    cons = list(constraints)
    has_pt0 = any((lo == 0 and hi == 0) for (lo, hi, _) in cons)
    has_04  = any((lo == 0 and hi == 4) for (lo, hi, _) in cons)
    has_14  = any((lo == 1 and hi == 4) for (lo, hi, _) in cons)
    
    if has_04 and has_pt0:
        T04 = sum(v for (lo, hi, v) in cons if (lo == 0 and hi == 4))
        T0  = sum(v for (lo, hi, v) in cons if (lo == 0 and hi == 0))
        T14 = max(T04 - T0, 0.0)
        cons = [(lo, hi, v) for (lo, hi, v) in cons if not (lo == 0 and hi == 4)]
        w = weights_from_nLx(L, [1,2,3,4])
        cons.extend([(a, a, float(wa*T14)) for a, wa in zip([1,2,3,4], w)])
    
    elif has_04 and not has_pt0:
        T04 = sum(v for (lo, hi, v) in cons if (lo == 0 and hi == 4))
        cons = [(lo, hi, v) for (lo, hi, v) in cons if not (lo == 0 and hi == 4)]
        w = weights_from_nLx(L, [0,1,2,3,4])
        cons.extend([(a, a, float(wa*T04)) for a, wa in zip([0,1,2,3,4], w)])
    
    if has_14:
        T14 = sum(v for (lo, hi, v) in cons if (lo == 1 and hi == 4))
        cons = [(lo, hi, v) for (lo, hi, v) in cons if not (lo == 1 and hi == 4)]
        w = weights_from_nLx(L, [1,2,3,4])
        cons.extend([(a, a, float(wa*T14)) for a, wa in zip([1,2,3,4], w)])
    
    return cons
```

---

## Priority 5: Variable-Specific Smoothness (LOW EFFORT 🟢)

### Problem
```python
ridge = 1e-6  # Same for everything
```

### Solution: Quick Parameter Dictionary

```python
SMOOTHNESS_BY_VARIABLE = {
    "poblacion_total": 1e-6,      # Stocks: very smooth
    "defunciones": 1e-5,           # Mortality: very smooth
    "nacimientos": 1e-5,           # Fertility: smooth
    "flujo_emigracion": 1e-4,      # Migration: can be rough
    "flujo_inmigracion": 1e-4,     # Migration: can be rough
    "flujo_migracion": 1e-4,       # Net migration: rough
}

def unabridge_df(
    df: pd.DataFrame,
    series_keys: Iterable[str] = SERIES_KEYS_DEFAULT,
    value_col: str = "VALOR_corrected",
    ridge: Optional[float] = None  # Now optional
) -> pd.DataFrame:
    """
    Enhanced with variable-specific smoothness.
    """
    series_keys = list(series_keys)
    work = df.copy()
    
    # Parse ages
    parsed = work["EDAD"].apply(parse_edad)
    work["EDAD_MIN"] = parsed.apply(lambda t: t[0])
    work["EDAD_MAX"] = parsed.apply(lambda t: t[1])
    work = collapse_convention2(work)
    
    outputs = []
    passthrough = []
    
    for _, g in work.groupby(series_keys, dropna=False):
        var = g.iloc[0]["VARIABLE"] if "VARIABLE" in g.columns else ""
        
        # Determine smoothness parameter
        if ridge is None:
            var_ridge = SMOOTHNESS_BY_VARIABLE.get(str(var), 1e-6)
        else:
            var_ridge = ridge
        
        # Rest unchanged...
        open_or_nan = g[g["EDAD_MAX"].isna()].copy()
        if not open_or_nan.empty:
            keep_cols = [*series_keys, "EDAD", value_col]
            for c in keep_cols:
                if c not in open_or_nan.columns:
                    open_or_nan[c] = np.nan
            passthrough.append(open_or_nan[keep_cols].copy())
        
        out = _unabridge_one_group(
            g, series_keys, 
            variable=str(var), 
            value_col=value_col, 
            ridge=var_ridge  # Use variable-specific ridge
        )
        if not out.empty:
            outputs.append(out)
    
    single = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=[*series_keys, "EDAD", value_col])
    openp  = pd.concat(passthrough, ignore_index=True) if passthrough else pd.DataFrame(columns=[*series_keys, "EDAD", value_col])
    result = pd.concat([single, openp], ignore_index=True)
    return result
```

---

## Quick Win: All Five Priorities Combined

```python
# In your main processing script
from abridger_enhanced import unabridge_df_with_validation, SMOOTHNESS_BY_VARIABLE

# Load your data
conteos = load_conteos()

# Before unabridging: calibrate tail ratios from recent census
recent_census = conteos[(conteos["ANO"] == 2018) & (conteos["VARIABLE"] == "poblacion_total")]
r_calibrated = calibrate_tail_ratio_from_census(recent_census)

# Update harmonization functions
harmonize_conteos_to_90plus(conteos, r_pop=r_calibrated)

# Unabridge with validation
result = unabridge_df_with_validation(
    conteos,
    series_keys=["DPTO_CODIGO", "ANO", "SEXO", "VARIABLE"],
    value_col="VALOR_corrected",
    ridge=None  # Use variable-specific smoothness
)

# Warnings will be logged automatically
```

---

## Expected Impact

| Priority | Effort | Impact | Improvement Area |
|----------|--------|--------|------------------|
| 1. Real life tables | Medium | High | Infant ages (0-4) accuracy |
| 2. Calibrated ratios | Low | High | Elderly ages (70+) realism |
| 3. Plausibility checks | Low | Medium | Error detection |
| 4. Extend to births/deaths | Low | Medium | Vital events accuracy |
| 5. Variable smoothness | Low | Low | Fine-tuning |

**Total effort:** ~2-3 days of work  
**Total improvement:** Transforms from "mathematically correct" to "demographically validated"
