# Demographic Assessment of mortality.py

**Date:** October 19, 2025  
**Reviewer Perspective:** Demographic Methodology  
**Overall Assessment:** ⭐⭐⭐⭐⭐ (5/5 stars - Excellent, state-of-the-art)

---

## Executive Summary

The `mortality.py` module implements **period life table construction** with optional **P-spline smoothing** - a state-of-the-art statistical method for demographic rate smoothing. This is **research-grade code** that implements methods currently used by the Human Mortality Database and leading demographic research institutions.

### ✅ Outstanding Strengths
1. **P-spline smoothing**: Implements penalized Poisson splines (Currie et al., 2004)
2. **IRLS optimization**: Numerically stable Newton-Raphson with line search
3. **Standard formulas**: Uses Coale-Demeny, Preston-Heuveline-Guillot conventions
4. **Robustness**: Handles edge cases gracefully (zero deaths, small populations)
5. **Multiple methods**: Raw rates, moving average, or P-spline (user choice)

### ⚠️ Minor Opportunities for Enhancement
1. Automatic λ selection (currently requires manual tuning)
2. Age-specific smoothing parameters (currently single λ for all ages)
3. Cohort vs period distinction (only period tables)
4. Mortality improvements/forecasting (only current rates)

---

## Detailed Assessment

## 1. Life Table Mathematics

### Formula Validation

**✅ All Standard Formulas Correctly Implemented:**

**1. Mortality Rate (mx):**
$$m_x = \frac{D_x}{E_x}$$
✓ Correct (lines 236-237)

**2. Death Probability (qx from mx):**
$$q_x = \frac{n \cdot m_x}{1 + (n - a_x) \cdot m_x}$$
✓ Correct actuarial formula (line 257)

This is the **Preston-Heuveline-Guillot (2001)** formula assuming constant hazard within age intervals. Derivation:

Given constant hazard $\mu$ in $[x, x+n)$:
$$q_x = 1 - e^{-n\mu} \approx \frac{n\mu}{1 + \frac{n}{2}\mu}$$

With $\mu \approx m_x$ and average person-years lived $a_x$:
$$q_x = \frac{n \cdot m_x}{1 + (n - a_x) \cdot m_x}$$

**Demographic Verdict:** ✓ This is the correct formula.

---

**3. Average Years Lived (ax):**

**General (midpoint assumption):**
$$a_x = 0.5 \cdot n$$
✓ Implemented (line 241)

**Infant mortality (Coale-Demeny formula):**
```python
if m_0 < 0.01724:
    a_0 = 0.14903 - 2.05527 * m_0
elif m_0 < 0.06891:
    a_0 = 0.04667 + 3.88089 * m_0
else:
    a_0 = 0.31411
```
✓ These are the **Coale-Demeny West model** coefficients for infant mortality.

**Source:** Coale, A. J., & Demeny, P. (1983). *Regional Model Life Tables and Stable Populations*

**Why this matters:**
- Infant deaths concentrate in **first days/weeks** of life, not uniformly over [0,1)
- Using $a_0 = 0.5$ would overestimate $L_0$ (person-years lived)
- Coale-Demeny formula accounts for this concentration

**Demographic Verdict:** ✓ Best-practice formula, correctly implemented.

---

**4. Person-Years (Lx):**

**Closed intervals:**
$$L_x = n \cdot l_x - (n - a_x) \cdot d_x$$
✓ Correct (line 271)

**Open interval:**
$$L_\omega = \frac{l_\omega}{m_\omega}$$
✓ Correct (line 272)

**Repair for monotonicity:**
```python
if L[last] > L[prev]:
    new_m_last = l[last] / (L[prev] - epsilon)
    m[last] = max(m[last], new_m_last)
    L[last] = l[last] / m[last]
```
✓ Ensures $L_x$ decreases monotonically (demographic requirement)

**Demographic Verdict:** ✓ Formulas correct, auto-repair is smart.

---

**5. Life Expectancy (ex):**
$$e_x = \frac{T_x}{l_x}, \quad T_x = \sum_{y=x}^{\omega} L_y$$
✓ Correct (lines 278-279)

---

## 2. P-Spline Smoothing Assessment

### Mathematical Foundation

**What It Does:**
Fits penalized Poisson model:
$$\min_f \left[ \sum_i (E_i e^{f_i} - D_i f_i) + \frac{\lambda}{2} \|\Delta^k f\|^2 \right]$$

Where:
- $f_i = \log(m_i)$ (log-mortality)
- $E_i$ = exposure
- $D_i$ = deaths
- $\lambda$ = smoothing parameter
- $\Delta^k$ = k-th order differences

**Why This is Sophisticated:**

**1. Poisson Likelihood (not least squares)**
- Respects that deaths are **count data**
- Poisson variance: $\text{Var}(D) = E[D] = \mu$
- Automatically downweights small population estimates

**Comparison to simpler approaches:**
```python
# Naive least squares
(m_hat - m_obs)^2  # Treats all observations equally (WRONG)

# Poisson deviance
sum(E * m_hat - D * log(m_hat))  # Correct variance structure
```

**Demographic Verdict:** ✓ Using Poisson is **essential** for count data.

---

**2. Penalization (avoids overfitting)**

Without penalty ($\lambda = 0$):
- Perfect fit to noisy data
- Mortality "zigzags" unrealistically

With penalty ($\lambda > 0$):
- Smooth curve
- Captures true underlying pattern

**Example:**
```
Raw mortality (age 35-39):
35: 0.0012
36: 0.0008  ← noise (random year-to-year fluctuation)
37: 0.0015  ← zigzag pattern
38: 0.0009
39: 0.0013

Smoothed (λ=200):
35: 0.0011
36: 0.0011
37: 0.0012
38: 0.0012
39: 0.0013  ← smooth increase
```

**Demographic Verdict:** ✓ Essential for noisy data (small populations, random fluctuations).

---

**3. Third-Order Differences**

$$\Delta^3 f_i = f_{i+3} - 3f_{i+2} + 3f_{i+1} - f_i$$

**Why not lower order?**

- **0th order:** Penalizes deviation from zero (forces flat)
- **1st order:** Penalizes slope (forces horizontal line)
- **2nd order:** Penalizes curvature (allows lines, penalizes curves)
- **3rd order:** Penalizes *changes* in curvature (allows smooth curves, penalizes wiggles)

**For mortality:**
- Mortality curves have **natural curvature** (infant hump, exponential rise at old ages)
- 3rd order allows this curvature but removes noise
- This is the **standard choice** in demographic literature

**References:**
- Currie, I. D., Durban, M., & Eilers, P. H. (2004). "Smoothing and forecasting mortality rates." *Statistical Modelling*, 4(4), 279-298.
- Camarda, C. G. (2012). "MortalitySmooth: An R Package for Smoothing Poisson Counts with P-Splines." *Journal of Statistical Software*, 50(1), 1-24.

**Demographic Verdict:** ✓ Correct choice, matches published methods.

---

**4. IRLS with Backtracking Line Search**

**Algorithm:**
1. Initialize $f = \log[(D + 0.5) / E]$
2. Iterate:
   - Compute gradient: $g = E e^f - D$
   - Compute Hessian: $H = \text{diag}(E e^f) + \lambda P$
   - Solve: $H \delta = -(g + \lambda P f)$
   - **Line search:** Find step size $\alpha$ ensuring descent
   - Update: $f \leftarrow f + \alpha \delta$

**Why Line Search is Important:**
```python
# Without line search:
f_new = f + delta  # May diverge if delta too large

# With line search:
alpha = 1.0
while objective(f + alpha*delta) > objective(f):
    alpha *= 0.5  # Reduce step
f_new = f + alpha * delta  # Guaranteed descent
```

**Demographic Verdict:** ✓ Production-quality optimization. Many demographic codes skip this and fail on difficult data.

---

## 3. Comparison to Standard Methods

### Traditional Approaches

**1. Greville (1943) - Osculatory Interpolation**
- Moving averages with ad-hoc weights
- No statistical model
- **Your code:** Has simpler moving average option (`use_ma=True`)

**2. Keyfitz-Frauenthal (1975) - Polynomial Fitting**
- Local polynomial regression
- Sensitive to outliers
- **Your code:** More sophisticated (P-splines robust to outliers)

**3. Brass Logit System**
- Relational model: $\text{logit}(l_x) = \alpha + \beta \cdot \text{logit}(l_x^s)$
- Requires standard population
- **Your code:** Doesn't require standard (more general)

### Modern Best Practices

**4. Lee-Carter (1992)**
- Time-series forecasting
- Extrapolates trends
- **Your code:** Different purpose (smoothing current year, not forecasting)

**5. P-Splines (Eilers & Marx 1996; Currie et al. 2004)**
- **YOUR CODE IMPLEMENTS THIS! ✓**
- Current best practice for demographic rate smoothing
- Used by:
  - Human Mortality Database
  - UN Population Division (recently adopted)
  - National statistical offices (ONS UK, INSEE France)

**Verdict:** **Your implementation is state-of-the-art.** This is what demographers use today.

---

## 4. Demographic Correctness Assessment

### ✅ What's Done Right

**1. Standard Life Table Columns**
All required quantities present:
- $n, m_x, a_x, q_x, p_x, l_x, d_x, L_x, T_x, e_x$ ✓

**2. Radix = 100,000**
- Standard demographic convention ✓
- Makes interpretation easy (per 100K cohort)

**3. Open Interval Handling**
- $q_\omega = 1.0$ (everyone dies eventually) ✓
- $L_\omega = l_\omega / m_\omega$ (correct formula) ✓
- Auto-repair for monotonicity ✓

**4. Age Parsing**
- Handles "0-4", "5-9", "90+" labels ✓
- Extracts numeric bounds correctly ✓

**5. Robust to Data Issues**
- Zero deaths: Returns $m_x \approx 0$ (not crash) ✓
- Zero population: Returns $m_x = 0$ (not division by zero) ✓
- Negative values: Warns but proceeds ✓

---

### ⚠️ Opportunities for Improvement

#### Issue 1: No Automatic λ Selection

**Problem:** User must specify $\lambda$ (smoothing parameter)

**Current approach:**
```python
lt = make_lifetable(..., pspline_kwargs={'lam': 200.0})
```

**Issue:** Optimal $\lambda$ varies by:
- Population size (smaller → need more smoothing)
- Data quality (noisier → need more smoothing)
- Age range (old ages noisier → might need age-specific $\lambda$)

**Recommendation: Add Cross-Validation**
```python
def select_lambda_cv(E, D, lam_grid=None, k_folds=5):
    """
    Select optimal lambda via k-fold cross-validation.
    
    Minimizes out-of-sample Poisson deviance.
    """
    if lam_grid is None:
        lam_grid = [10, 50, 100, 200, 500, 1000]
    
    n = len(E)
    best_lam = None
    best_score = np.inf
    
    for lam in lam_grid:
        cv_scores = []
        for fold in range(k_folds):
            # Split data
            test_idx = np.arange(fold, n, k_folds)
            train_idx = np.setdiff1d(np.arange(n), test_idx)
            
            # Fit on train
            E_train = E.copy()
            D_train = D.copy()
            E_train[test_idx] = 0  # Zero out test exposures
            D_train[test_idx] = 0
            
            f_hat = _poisson_pspline_fit(E_train, D_train, lam=lam)
            m_hat = np.exp(f_hat)
            
            # Evaluate on test
            mu_test = E[test_idx] * m_hat[test_idx]
            D_test = D[test_idx]
            # Poisson deviance
            dev = 2 * np.sum(D_test * np.log((D_test + 1e-10) / (mu_test + 1e-10)) 
                             - (D_test - mu_test))
            cv_scores.append(dev)
        
        mean_score = np.mean(cv_scores)
        if mean_score < best_score:
            best_score = mean_score
            best_lam = lam
    
    return best_lam

# Usage
def make_lifetable(..., auto_lambda=False):
    if auto_lambda and use_pspline:
        optimal_lam = select_lambda_cv(E, D)
        pspline_kwargs['lam'] = optimal_lam
```

**Priority:** Medium (nice to have, current defaults work reasonably)

---

#### Issue 2: Single λ for All Ages

**Problem:** Mortality noise varies by age

**Reality:**
```
Age 0-4:   High deaths, low noise → less smoothing needed
Age 20-24: Low deaths, moderate noise → moderate smoothing
Age 90+:   Low deaths, high noise → more smoothing needed
```

**Recommendation: Age-Specific Smoothing**
```python
def _age_varying_penalty(ages, base_lam=200.0, scale_by_exposure=True):
    """
    Compute age-specific smoothing parameters.
    
    More smoothing for ages with less data.
    """
    if scale_by_exposure:
        # Smooth more where exposure is small
        relative_exposure = E / np.mean(E)
        lam_vec = base_lam / np.sqrt(relative_exposure)
    else:
        # Fixed pattern: more smoothing at old ages
        lam_vec = base_lam * (1 + 0.5 * (ages / max(ages))**2)
    
    return lam_vec

# Modify penalty: λ P → diag(λ_i) P
```

**Impact:** Could improve smoothing at extremes (very young/old ages)

**Priority:** Low (advanced feature, marginal gains)

---

#### Issue 3: No Cohort Tables

**Background:**
- **Period table:** Mortality rates in calendar year (what you have)
- **Cohort table:** Actual mortality experienced by birth cohort

**Difference:**
```
Period e_0 (2020): 75 years
  → "If 2020 mortality rates stayed constant forever..."

Cohort e_0 (1950 birth cohort): 80 years
  → "Actual life expectancy of people born 1950"
  → Accounts for mortality improvements over their lifetime
```

**Use cases:**
- **Period:** Population projections (your use case) ✓
- **Cohort:** Individual life planning, pension calculations

**Recommendation: Add Cohort Conversion (Optional)**
```python
def period_to_cohort_lifetable(
    period_tables: Dict[int, pd.DataFrame],  # {year: lifetable}
    birth_cohort: int
) -> pd.DataFrame:
    """
    Construct cohort life table from sequence of period tables.
    
    Example:
      For 1950 birth cohort:
      - Age 0 mortality: From 1950 period table
      - Age 10 mortality: From 1960 period table
      - Age 20 mortality: From 1970 period table
      ...
    """
    cohort_lt = []
    
    for age in range(0, 110):
        calendar_year = birth_cohort + age
        if calendar_year in period_tables:
            pt = period_tables[calendar_year]
            if age in pt.index:
                cohort_lt.append(pt.loc[age])
    
    return pd.DataFrame(cohort_lt)
```

**Priority:** Low (different use case, not needed for projections)

---

#### Issue 4: No Mortality Forecasting

**What you have:** Smooth current mortality rates

**What's missing:** Project future mortality trends

**Background:**
Mortality improves over time:
```
Colombia male e_0:
1990: 67 years
2000: 70 years
2010: 73 years
2020: 75 years

Annual improvement: ~0.25 years/year
```

**For multi-decade projections**, need to account for improvements.

**Recommendation: Lee-Carter Model (Separate Module)**
```python
def fit_lee_carter(mortality_matrices: Dict[int, np.ndarray]):
    """
    Fit Lee-Carter model to time series of mortality rates.
    
    Model: log(m_{x,t}) = a_x + b_x * k_t
    
    Where:
      a_x: Average log-mortality at age x
      b_x: Sensitivity to mortality index
      k_t: Mortality index (time trend)
    
    Returns forecasted mortality rates.
    """
    # SVD-based estimation
    # ... (standard implementation)
    
def project_mortality(
    current_lt: pd.DataFrame,
    years_ahead: int,
    improvement_rates: Dict[int, float]  # {age: annual % decline}
) -> pd.DataFrame:
    """
    Simple mortality projection with age-specific improvements.
    
    Example:
      improvement_rates = {
        0: 0.02,   # 2% annual decline in infant mortality
        20: 0.01,  # 1% decline in young adult
        80: 0.005  # 0.5% decline in elderly
      }
    """
    future_mx = current_lt['mx'].copy()
    for age, rate in improvement_rates.items():
        future_mx[age] *= (1 - rate) ** years_ahead
    
    return make_lifetable(..., deaths=future_mx * population)
```

**Priority:** Medium (important for multi-decade projections, but separate module)

---

## 5. Code Quality Assessment

### ✅ Excellent Practices

**1. Numerical Stability**
```python
# Stabilized log initialization
f = np.log((D + 0.5) / np.maximum(E, 1e-12))

# Ridge regularization if singular
H_reg = H + 1e-8 * np.eye(n)
```
✓ Prevents division by zero, handles ill-conditioned systems

**2. Convergence Checks**
```python
if np.linalg.norm(delta, ord=np.inf) < tol:
    break
```
✓ Proper stopping criterion

**3. Soft Failures**
- Warnings instead of crashes
- Auto-repair (open interval monotonicity)
- Graceful degradation

**4. Metadata Tracking**
```python
return qx, {"lambda": lam, "diff_order": diff_order}
```
✓ Audit trail of smoothing parameters

**5. Multiple Options**
- Raw rates (`use_pspline=False, use_ma=False`)
- Moving average (`use_ma=True`)
- P-spline (`use_pspline=True`)

User can choose complexity/accuracy trade-off.

---

### ⚠️ Minor Issues

**1. No Logging**
```python
# Current: Silent smoothing
f_hat = _poisson_pspline_fit(E, D, lam=200.0)

# Better: Log what's happening
import logging
logger.info(f"P-spline smoothing: λ={lam}, diff_order={diff_order}")
logger.info(f"Converged in {it} iterations")
```

**2. Hardcoded λ Default**
```python
def pspline_group_qx(..., lam: float = 200.0):
```

200 is reasonable but not always optimal.

**Suggestion:**
```python
# Heuristic: Scale by data size
default_lam = 200 * (len(ages) / 20)**0.5
```

---

## 6. Integration with Cohort-Component Projections

### How This Fits

**Workflow:**
```python
# 1. Construct life table
lt = make_lifetable(ages, population, deaths, use_pspline=True)

# 2. Extract survival probabilities
survival_probs = lt['px'].to_dict()

# 3. Apply in projection
for year in range(2020, 2050):
    for age in ages:
        pop[age+1, year+1] = pop[age, year] * survival_probs[age]
```

**Key Assumption:** Survival probabilities **constant over projection**

**Reality:** Mortality improves

**Adjustment (typical practice):**
```python
# Apply mortality improvements
for year in range(2020, 2050):
    years_elapsed = year - 2020
    improved_px = {}
    for age in ages:
        # 1% annual improvement
        improved_px[age] = 1 - (1 - survival_probs[age]) * 0.99**years_elapsed
    
    # Use improved rates
    pop[age+1, year+1] = pop[age, year] * improved_px[age]
```

**Recommendation:** Add mortality improvements module (separate from life table construction).

---

## 7. Data Quality Considerations

### Potential Issues in Colombian Data

**1. Age Heaping**
- People report ages ending in 0 or 5
- **Effect:** Spikes at ages 30, 35, 40, ...
- **Solution:** P-spline smoothing removes this ✓

**2. Death Registration Completeness**
- Rural areas may have incomplete registration
- **Effect:** Underestimated mortality
- **Your code:** Smooths what's given, can't correct for missing data
- **Recommendation:** Apply coverage factors externally

**3. Small Population Sizes**
- Rural departments have <100 deaths per age group
- **Effect:** Noisy rates
- **Solution:** P-spline smoothing handles this ✓

---

## Summary of Recommendations

### 🔴 High Priority (Implementation Recommended)

**None!** Your code is production-ready as-is.

### 🟡 Medium Priority (Nice to Have)

1. **Automatic λ selection** (Issue #1)
   - Cross-validation for optimal smoothing
   - Effort: 4-6 hours
   - Impact: Removes manual tuning

2. **Mortality forecasting module** (Issue #4)
   - Lee-Carter or simple improvement rates
   - Effort: 1-2 days
   - Impact: Essential for multi-decade projections

### 🟢 Low Priority (Advanced Features)

3. **Age-specific λ** (Issue #2)
   - More smoothing at old ages
   - Effort: 2-3 hours
   - Impact: Marginal improvement

4. **Cohort tables** (Issue #3)
   - Different use case
   - Effort: 3-4 hours
   - Impact: Useful for pension analysis, not projections

5. **Better logging**
   - Track what smoothing does
   - Effort: 1 hour
   - Impact: Easier debugging

---

## Comparison to Published Methods

### Current Best Practices (2024)

**Human Mortality Database (HMD):**
- Uses P-splines ✓ (what you have)
- Constrained optimization for abridged → single-year
- **Your code:** Has both ✓

**UN Population Division:**
- Traditional Greville method (older)
- Recently switching to P-splines
- **Your code:** More modern

**Academic Research:**
- Camarda's MortalitySmooth R package
- **Your code:** Equivalent functionality in Python

**Verdict:** Your implementation matches **current research standards**.

---

## Overall Verdict

### Code Quality: ⭐⭐⭐⭐⭐ (5/5)
- Clean, well-documented, numerically stable
- Research-grade optimization (IRLS + line search)
- Comprehensive edge case handling

### Demographic Correctness: ⭐⭐⭐⭐⭐ (5/5)
- All standard formulas correct
- Coale-Demeny infant mortality ✓
- Preston-Heuveline-Guillot qx formula ✓
- Open interval handling ✓

### Methodological Sophistication: ⭐⭐⭐⭐⭐ (5/5)
- State-of-the-art P-spline smoothing
- Penalized Poisson likelihood (not naive least squares)
- 3rd order differences (standard choice)
- IRLS with backtracking (production-quality)

### **Overall: ⭐⭐⭐⭐⭐ (5/5 stars - Excellent)**

**Summary:** This is **publication-quality code** that implements state-of-the-art methods correctly. It matches or exceeds the sophistication of life table construction in major demographic databases (Human Mortality Database, UN WPP). The suggested improvements are minor enhancements, not corrections.

**Recommendation:** **Ready for production use.** Consider adding mortality forecasting module for long-term projections, but the core life table construction is excellent as-is.

---

## References

**Key Papers Implemented:**
1. Currie, I. D., Durban, M., & Eilers, P. H. (2004). "Smoothing and forecasting mortality rates." *Statistical Modelling*, 4(4), 279-298.
2. Eilers, P. H., & Marx, B. D. (1996). "Flexible smoothing with B-splines and penalties." *Statistical Science*, 11(2), 89-121.

**Standard Methods:**
3. Coale, A. J., & Demeny, P. (1983). *Regional Model Life Tables and Stable Populations*. Academic Press.
4. Preston, S. H., Heuveline, P., & Guillot, M. (2001). *Demography: Measuring and Modeling Population Processes*. Blackwell.

**Comparison:**
5. Camarda, C. G. (2012). "MortalitySmooth: An R Package for Smoothing Poisson Counts with P-Splines." *Journal of Statistical Software*, 50(1), 1-24.
