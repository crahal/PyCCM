# Comprehensive Explanation of mortality.py

**Date:** October 19, 2025  
**File:** `/src/mortality.py`  
**Purpose:** Period life table construction with advanced smoothing methods

---

## Overview

The `mortality.py` module implements **period life table** construction for demographic projections. This is a core component of cohort-component models, translating observed death data into age-specific mortality rates and life expectancy.

**Key Innovation:** Optional **P-spline (penalized spline) smoothing** on single-year ages, which is more sophisticated than traditional methods.

---

## Function Breakdown

### 1. `parse_age_labels(age_labels)`

**Purpose:** Extract numeric age bounds from string labels

**Input:** 
```python
age_labels = pd.Series(['0-4', '5-9', '10-14', '90+'])
```

**Output:**
```python
pd.Series([0, 5, 10, 90], dtype=int)
```

**Method:** Regex extraction `r'(\d+)'` → first number

**Demographic Context:**
- Life tables use **age intervals** (abridged tables): 0-4, 5-9, ..., 85+
- Single-year tables: 0, 1, 2, ..., 110
- This function handles the parsing step

---

### 2. `_expand_closed_intervals(ages, widths)`

**Purpose:** Expand age intervals into single-year ages for smoothing

**Input:**
```python
ages = [0, 5, 10, 15, 90]     # Starting ages
widths = [5, 5, 5, 5, np.inf] # Interval widths (last is open)
```

**Output:**
```python
single_ages = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, ...]
group_id = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, ...]
closed_idx = [0, 1, 2, 3]  # Indices of closed intervals (exclude 90+)
```

**Why This Matters:**
- P-spline smoothing works on **single-year ages**
- But data comes in **age groups** (5-year intervals)
- Must expand, smooth, then re-aggregate

**Algorithm:**
1. Exclude open interval (90+, infinity width)
2. For each closed interval `[a, a+n)`:
   - Create ages: `a, a+1, ..., a+n-1`
   - Track which interval each age belongs to

---

### 3. `_difference_matrix(n, order)`

**Purpose:** Create finite-difference operator for smoothness penalty

**Mathematical Background:**

Smoothness of a function $f$ can be measured by its **differences**:
- **1st difference:** $\Delta f_i = f_{i+1} - f_i$ (slope changes)
- **2nd difference:** $\Delta^2 f_i = f_{i+2} - 2f_{i+1} + f_i$ (curvature)
- **3rd difference:** $\Delta^3 f_i = f_{i+3} - 3f_{i+2} + 3f_{i+1} - f_i$ (jerk)

**Matrix Form:**

For $n=5$, order=2:
```
D = [ 1  -2   1   0   0 ]
    [ 0   1  -2   1   0 ]
    [ 0   0   1  -2   1 ]
```

Then $\|Df\|^2 = \sum_i (\Delta^2 f_i)^2$ measures total curvature.

**Why 3rd Order?**
- 1st order: Penalizes changes in level (too strong)
- 2nd order: Penalizes curvature (common choice)
- **3rd order: Penalizes changes in curvature** (allows smooth curves, penalizes wiggles)

For mortality data, 3rd order is standard because:
- Mortality curves should be smooth but not linear
- Allow natural curvature (infant mortality hump, old-age acceleration)
- Penalize artificial oscillations from noise

---

### 4. `_poisson_pspline_fit(E, D, lam=200.0, diff_order=3)`

**Purpose:** Fit smooth log-mortality rates via penalized Poisson likelihood

**Statistical Model:**

Deaths follow Poisson distribution:
$$D_i \sim \text{Poisson}(\mu_i), \quad \mu_i = E_i \cdot m_i$$

Where:
- $D_i$ = observed deaths at age $i$
- $E_i$ = exposure (population × time)
- $m_i$ = hazard rate (force of mortality)

**Log-Likelihood:**
$$\ell(f) = \sum_i [D_i \log(E_i e^{f_i}) - E_i e^{f_i}] = \sum_i [D_i f_i - E_i e^{f_i}] + C$$

Where $f_i = \log(m_i)$ (work on log-scale for stability).

**Penalized Objective:**
$$L(f) = -\ell(f) + \frac{\lambda}{2} \|\Delta^k f\|^2 = \sum_i [E_i e^{f_i} - D_i f_i] + \frac{\lambda}{2} \|D_k f\|^2$$

**Interpretation:**
- **First term:** Fit the data (Poisson deviance)
- **Second term:** Smoothness penalty (roughness)
- **λ (lambda):** Smoothing parameter
  - λ = 0: No smoothing (overfits noise)
  - λ = ∞: Completely smooth (underfits)
  - λ = 200: Empirical default (works well for mortality)

**Optimization:** Iteratively Reweighted Least Squares (IRLS)

**Algorithm:**
1. Initialize: $f = \log[(D + 0.5) / E]$ (stabilized log-rate)
2. Repeat until convergence:
   - Compute gradient: $g = E e^f - D$
   - Compute Hessian: $H = \text{diag}(E e^f) + \lambda P$ where $P = D_k^T D_k$
   - Solve Newton step: $H \delta = -(g + \lambda P f)$
   - Backtracking line search to ensure descent
   - Update: $f \leftarrow f + \alpha \delta$
3. Return: $f$ (smooth log-mortality)

**Why This is Sophisticated:**
- **Poisson likelihood:** Respects count nature of deaths (not just least squares)
- **Penalized:** Prevents overfitting from noise
- **IRLS with line search:** Numerically stable convergence
- **Log-scale:** Ensures positive mortality rates

---

### 5. `pspline_group_qx(ages, widths, population, deaths, lam=200.0)`

**Purpose:** Compute interval death probabilities using P-spline smoothing

**Life Table Quantities:**
- **$m_x$:** Hazard rate (instantaneous force of mortality)
- **$q_x$:** Death probability in age interval $[x, x+n)$
  $$q_x = \frac{\text{Deaths in }[x,x+n)}{\text{Alive at age }x}$$

**Relationship (constant hazard assumption):**
$$q_x = 1 - e^{-n \cdot m_x}$$

Where $n$ is interval width.

**Algorithm:**
1. **Expand intervals** into single-year ages (call `_expand_closed_intervals`)
2. **Split exposures uniformly:** 
   ```python
   E[age_i] = E[interval_j] / width[interval_j]
   D[age_i] = D[interval_j] / width[interval_j]
   ```
3. **Fit P-spline** on single-year log-rates (call `_poisson_pspline_fit`)
   - Returns: $f_i$ (smooth log-mortality)
4. **Convert to single-year probabilities:**
   ```python
   m_i = exp(f_i)
   q_i = 1 - exp(-m_i)  # Annual probability
   ```
5. **Aggregate to intervals:**
   For interval $[x, x+n)$ containing ages $i, i+1, ..., i+n-1$:
   $$q_x = 1 - \prod_{k=i}^{i+n-1} (1 - q_k) = 1 - \prod_{k=i}^{i+n-1} p_k$$
   
   (Probability of death = 1 - survival probability)
6. **Open interval:** $q_\omega = 1$ (everyone dies eventually)

**Return:**
- `q_int`: Array of interval death probabilities
- `meta`: Dictionary with smoothing parameters

---

### 6. `make_lifetable(ages, population, deaths, ...)`

**Purpose:** Construct complete period life table with multiple options

**Standard Life Table Columns:**

| Symbol | Name | Definition |
|--------|------|------------|
| $n$ | Interval width | Width of age interval |
| $m_x$ | Mortality rate | Deaths / Exposure |
| $a_x$ | Average years lived | Avg time lived in interval by those who die |
| $q_x$ | Death probability | Prob of death in interval |
| $p_x$ | Survival probability | $1 - q_x$ |
| $l_x$ | Survivors | Number alive at start of interval |
| $d_x$ | Deaths | Number of deaths in interval |
| $L_x$ | Person-years | Total person-years lived in interval |
| $T_x$ | Person-years remaining | Total person-years after age $x$ |
| $e_x$ | Life expectancy | Expectation of life at age $x$ |

**Key Formulas:**

**1. Mortality Rate:**
$$m_x = \frac{D_x}{E_x}$$

**2. Death Probability (from $m_x$ and $a_x$):**
$$q_x = \frac{n \cdot m_x}{1 + (n - a_x) \cdot m_x}$$

This is the **standard actuarial formula** assuming constant hazard within intervals.

**3. Average Years Lived ($a_x$):**

**General:** $a_x = 0.5 \cdot n$ (midpoint assumption)

**Special case for infant mortality (age 0-1):**
Uses **Coale-Demeny formula** based on infant mortality level:
```python
if m_0 < 0.01724:
    a_0 = 0.14903 - 2.05527 * m_0
elif m_0 < 0.06891:
    a_0 = 0.04667 + 3.88089 * m_0
else:
    a_0 = 0.31411
```

**Why?** Infant deaths concentrate in first days/weeks, not uniformly distributed.

**4. Survivorship:**
$$l_x = l_0 \cdot \prod_{y=0}^{x-1} p_y$$

Starting with radix $l_0 = 100{,}000$ (conventional).

**5. Deaths:**
$$d_x = l_x \cdot q_x$$

**6. Person-Years (closed intervals):**
$$L_x = n \cdot l_x - (n - a_x) \cdot d_x$$

**Open interval:**
$$L_\omega = \frac{l_\omega}{m_\omega}$$

**7. Total Person-Years:**
$$T_x = \sum_{y=x}^{\omega} L_y$$

**8. Life Expectancy:**
$$e_x = \frac{T_x}{l_x}$$

---

**Optional Features:**

**A) Moving Average Smoothing (`use_ma=True`)**

Smooths $\log(m_x)$ using centered moving average:
```python
log_mx_smooth[i] = mean(log_mx[i-2:i+3])  # 5-year window
```

**Purpose:** Reduce noise without full P-spline machinery (faster, simpler).

**B) P-Spline Smoothing (`use_pspline=True`)**

Replaces $q_x$ with P-spline smoothed values:
1. Expand to single-year ages
2. Fit penalized Poisson model
3. Re-aggregate to intervals

**Purpose:** Optimal smoothing respecting Poisson variance structure.

**C) Open Interval Repair**

**Problem:** Numerical issues can cause $L_\omega > L_{\omega-1}$ (impossible - person-years should decline).

**Solution:**
```python
if L[last] > L[prev]:
    # Increase m[last] to reduce L[last]
    new_m = l[last] / (L[prev] - epsilon)
    m[last] = max(m[last], new_m)
    L[last] = l[last] / m[last]
```

Forces monotonic decline in $L_x$.

---

## Demographic Context

### Life Tables in Population Projections

**Role in Cohort-Component Model:**
1. **Input:** Age-specific deaths and population (from census/vital stats)
2. **Output:** Survival probabilities ($p_x$) by age and sex
3. **Projection:** 
   ```python
   population[age+1, year+1] = population[age, year] * p[age, sex]
   ```

**Why Period Life Tables?**
- **Period:** Mortality rates in a calendar year (cross-sectional)
- **Cohort:** Actual mortality experienced by birth cohort (longitudinal)

For projections, period tables are standard because:
- Easier to construct (don't need to wait for cohort to die)
- Can apply improvements (mortality decline trends)
- Cohort tables mix historical and future conditions

### Smoothing Rationale

**Raw mortality rates are noisy** due to:
- Small population sizes (rural areas)
- Random year-to-year fluctuations
- Data errors (age misreporting, death registration gaps)

**Example:**
```
Age 37: 12 deaths / 1,200 population = 0.010 (10 per 1000)
Age 38: 8 deaths / 1,180 population = 0.0068 (6.8 per 1000)
Age 39: 16 deaths / 1,150 population = 0.014 (14 per 1000)

Pattern: 10 → 7 → 14 (zigzag)
Reality: Should be smooth curve
```

**Smoothing fixes this:**
```
Smoothed: 10 → 10.5 → 11.2 (monotonic increase)
```

---

## Design Decisions

###  Good Choices

**1. Penalized Spline Approach**
- State-of-the-art method (Currie et al., 2004)
- Optimal bias-variance trade-off
- Respects Poisson variance structure
- Used by Human Mortality Database

**2. IRLS with Line Search**
- Numerically stable
- Guaranteed descent
- Handles edge cases (zero deaths, small populations)

**3. Soft Failures**
- Warnings instead of crashes
- Automatic repairs (open interval monotonicity)
- Graceful degradation

**4. Flexible Options**
- Can choose: raw rates, moving average, or P-spline
- Adjustable smoothing parameter (λ)
- Metadata returned for auditing

**5. Standard Formulas**
- Coale-Demeny $a_x$ for infants
- Standard actuarial $q_x$ from $m_x$
- Radix = 100,000 (conventional)

###  Potential Issues

**1. Uniform Splitting of Deaths**
```python
D_split = repeat(D[interval] / width, width)
```

**Assumption:** Deaths uniform within age groups

**Reality:** Often concentrated (e.g., age 0 deaths in first month)

**Impact:** Minor for 5-year groups, more significant for 0-1 age split

**2. Default λ = 200**
- Works well for typical data
- But optimal λ varies by:
  - Population size (smaller → more smoothing needed)
  - Data quality (noisier → more smoothing)
  - Age range (older ages noisier)

**3. No Automatic λ Selection**
- Could use: Cross-validation, AIC, BIC
- Currently: User must tune manually

**4. Single λ for All Ages**
- Might want age-specific λ:
  - Less smoothing for infants (rapid changes)
  - More smoothing for old ages (sparse data)

---

## Comparison to Standard Methods

### Traditional Approaches:

**1. Greville (1943) - Moving Averages**
- Simple weighted averages
- No statistical model
- **Your code supports this:** `use_ma=True`

**2. Keyfitz-Frauenthal (1975) - Polynomial Fitting**
- Local polynomials
- Arbitrary choice of degree
- Sensitive to outliers

**3. Brass Logit System**
- Relational model: $\text{logit}(l_x) = \alpha + \beta \cdot \text{logit}(l_x^s)$
- Requires standard population
- Not implemented here

### Modern Approaches:

**4. Lee-Carter Model**
- Time-series forecasting
- Extrapolates trends
- **Different purpose:** Forecasting vs smoothing

**5. P-Splines (Eilers & Marx 1996; Currie et al. 2004)**
- **Your code implements this!**
- Penalized B-splines
- Optimal smoothing
- **Current best practice** for demographic rates

**Verdict:** Your P-spline implementation is **state-of-the-art**.

---

## Integration with Projections

**Typical Workflow:**

```python
# 1. Load mortality data
ages = ['0-4', '5-9', ..., '90+']
deaths = [120, 45, 38, ...]
population = [10000, 12000, 11500, ...]

# 2. Construct life table with smoothing
lt = make_lifetable(
    ages, population, deaths,
    use_pspline=True,
    pspline_kwargs={'lam': 200, 'diff_order': 3}
)

# 3. Extract survival probabilities
survival_probs = lt['px'].to_dict()  # {0: 0.998, 5: 0.999, ...}

# 4. Apply mortality improvements (if projecting)
# (handled elsewhere in projections module)

# 5. Use in cohort-component projection
pop_next[age+1] = pop_current[age] * survival_probs[age]
```

---

## Summary

**What This Module Does:**
1. Constructs period life tables from deaths and population
2. Offers multiple smoothing methods (raw, moving average, P-spline)
3. Handles edge cases gracefully (zero deaths, small populations)
4. Returns complete life table with all standard quantities

**Key Strengths:**
- State-of-the-art P-spline smoothing
- Numerically stable algorithms
- Flexible and well-documented
- Handles data imperfections

**Use Cases:**
- Demographic projections (cohort-component)
- Mortality analysis
- Life expectancy calculation
- Smoothing noisy mortality data

**Technical Level:** **Advanced** - Implements research-grade methods correctly.
