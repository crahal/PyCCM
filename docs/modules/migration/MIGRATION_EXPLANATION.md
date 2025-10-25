# Comprehensive Explanation of migration.py

**Date:** October 20, 2025  
**File:** `/src/migration.py`  
**Purpose:** Migration data processing for cohort-component projections

---

## Overview

The `migration.py` module processes **internal and international migration data** for demographic projections. It aggregates department-level migration flows to national totals by age and sex, computing net migration rates that feed into cohort-component population projections.

**Key Function:** `create_migration_frame()` - Single function that does all the work

**Input:** `conteos` DataFrame with department-level population and migration flows  
**Output:** National-level panel with net migration rates by (year, age, sex)

---

## Function Breakdown

### `create_migration_frame(conteos, year=2018)`

**Purpose:** Build national migration panel from department-level data

**Inputs:**
- `conteos`: DataFrame containing:
  - `VARIABLE`: {"poblacion_total", "flujo_inmigracion", "flujo_emigracion"}
  - `ANO`: Year (e.g., 2018)
  - `EDAD`: Age group ("0-4", "5-9", ..., "80+")
  - `SEXO`: Sex (1=Male, 2=Female)
  - `VALOR`: Value (population count or migration flow)
  - `DPTO`: Department code (aggregated out)

- `year`: Optional year filter (default 2018)

**Output:** DataFrame with columns:
- `ANO`: Year
- `EDAD`: Age group (ordered categorical)
- `SEXO`: Sex code
- `inmigracion_F`: Immigration flows (in-migrants)
- `emigracion_F`: Emigration flows (out-migrants)
- `net_migration`: Net migration (in - out)
- `poblacion_total`: Total population (denominator)
- `net_mig_rate`: Net migration rate (net/population)

---

## Algorithm Walkthrough

### Step 1: Data Preparation

```python
df = conteos.copy()

# Coerce numerics & normalize labels
df["VALOR"] = pd.to_numeric(df["VALOR"], errors="coerce").fillna(0.0)
df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").astype("Int64")
df["SEXO"] = pd.to_numeric(df["SEXO"], errors="coerce").astype("Int64")
df["EDAD"] = df["EDAD"].astype(str).str.strip()
```

**What this does:**
- Converts all numeric columns to proper types
- Fills invalid values with 0
- Strips whitespace from age labels
- Uses nullable Int64 (handles missing years)

**Why this matters:**
Real-world data has issues:
- String numbers: "1234" → 1234
- Invalid entries: "N/A" → 0
- Whitespace: " 0-4 " → "0-4"

---

### Step 2: Aggregate Population to National Level

```python
pop_nat = (
    df.loc[df["VARIABLE"] == "poblacion_total", ["ANO", "EDAD", "SEXO", "VALOR"]]
      .groupby(["ANO","EDAD","SEXO"], as_index=False)["VALOR"].sum()
      .rename(columns={"VALOR": "poblacion_total"})
)
```

**What this does:**
Sum population across all departments for each (year, age, sex) combination.

**Example:**
```
Input (department level):
DPTO  ANO  EDAD  SEXO  VALOR (population)
05    2018  0-4   1     50,000  (Antioquia males 0-4)
11    2018  0-4   1     80,000  (Bogotá males 0-4)
...

Output (national level):
ANO  EDAD  SEXO  poblacion_total
2018  0-4   1     2,500,000  (All Colombia males 0-4)
```

**Demographic Context:**
- Cohort-component projections work at **national** or **regional** level
- Department details matter for subnational projections but not here
- Aggregation preserves total population (conservation of mass)

---

### Step 3: Aggregate Migration Flows

```python
mig = df.loc[df["VARIABLE"].isin([IN, OUT]), ["ANO","EDAD","SEXO","VARIABLE","VALOR"]]

mig_nat = (
    pd.pivot_table(
        mig,
        index=["ANO","EDAD","SEXO"],
        columns="VARIABLE",
        values="VALOR",
        aggfunc="sum",
        fill_value=0.0,
    )
    .reset_index()
    .rename(columns={IN: "inmigracion_F", OUT: "emigracion_F"})
)
```

**What this does:**
1. Filter to migration variables only
2. Pivot from long to wide format
3. Sum flows across departments
4. Rename columns

**Example:**
```
Input (long format):
ANO  EDAD  SEXO  VARIABLE              VALOR
2018  20-24  1    flujo_inmigracion      1,200
2018  20-24  1    flujo_emigracion         800

Output (wide format):
ANO  EDAD  SEXO  inmigracion_F  emigracion_F
2018  20-24  1    1,200          800
```

**Demographic Interpretation:**

**Immigration (inmigracion_F):**
- People entering from other countries
- Or entering from other regions (if subnational)
- Positive contribution to population

**Emigration (emigracion_F):**
- People leaving to other countries
- Or leaving to other regions
- Negative contribution to population

---

### Step 4: Handle Missing Categories

```python
for col in ["inmigracion_F", "emigracion_F"]:
    if col not in mig_nat.columns:
        mig_nat[col] = 0.0
```

**Why this is needed:**
If a dataset has **only** immigration or **only** emigration data, the pivot will create only one column. This ensures both columns always exist.

**Example scenario:**
```
# Historical data: Only immigration recorded
→ emigracion_F would be missing
→ Code adds it as 0.0
```

---

### Step 5: Merge Population and Migration

```python
nat_age_sex = pop_nat.merge(mig_nat, on=["ANO","EDAD","SEXO"], how="left")
nat_age_sex[["inmigracion_F","emigracion_F"]] = nat_age_sex[["inmigracion_F","emigracion_F"]].fillna(0.0)
```

**Merge logic:**
- `how="left"`: Keep all population records
- Even if migration data missing for some (age, sex, year)
- Fill missing migration with 0

**Why left join?**
Population grid is **complete** (all age-sex combinations exist), but migration might have gaps:
```
Population: All 17 age groups × 2 sexes = 34 rows per year
Migration: Might only have data for working ages (15-64)
```

Left join ensures we don't lose age groups.

---

### Step 6: Calculate Derived Quantities

```python
nat_age_sex["net_migration"] = nat_age_sex["inmigracion_F"] - nat_age_sex["emigracion_F"]

denom = pd.to_numeric(nat_age_sex["poblacion_total"], errors="coerce")
nat_age_sex["net_mig_rate"] = np.where(denom > 0, nat_age_sex["net_migration"] / denom, np.nan)
```

**Net Migration:**
$$\text{Net Migration} = \text{Immigration} - \text{Emigration}$$

**Net Migration Rate:**
$$\text{Net Migration Rate} = \frac{\text{Net Migration}}{\text{Population}}$$

**Example:**
```
Age 25-29, Males, 2018:
  Immigration:    5,000
  Emigration:     3,000
  Net migration:  2,000
  Population:     500,000
  Net mig rate:   2,000 / 500,000 = 0.004 (4 per 1000)
```

**Demographic Interpretation:**
- **Positive rate:** More in-migrants than out-migrants (net gain)
- **Negative rate:** More out-migrants (net loss)
- **Zero rate:** Balanced flows or closed population

**Units:** Rate is typically expressed **per 1,000** population:
```python
rate_per_1000 = net_mig_rate * 1000
```

---

### Step 7: Order and Format

```python
nat_age_sex["EDAD"] = pd.Categorical(nat_age_sex["EDAD"], categories=EDAD_ORDER, ordered=True)
nat_age_sex["SEXO"] = pd.to_numeric(nat_age_sex["SEXO"], errors="coerce")

panel = (
    nat_age_sex[out_cols]
    .sort_values(["ANO","EDAD","SEXO"], kind="mergesort")
    .reset_index(drop=True)
)
```

**Why ordered categorical for EDAD?**
Ensures age groups sort correctly:
```
Wrong: "0-4", "10-14", "20-24", "5-9" (alphabetical)
Right: "0-4", "5-9", "10-14", "20-24" (age order)
```

**Why mergesort?**
- Stable sort (preserves relative order of equal elements)
- Important for reproducibility

---

### Step 8: Optional Year Filter

```python
if year is not None:
    panel = panel[panel["ANO"] == year].reset_index(drop=True)
```

**Use cases:**
- `year=2018`: Single year (for projections starting in 2018)
- `year=None`: All years (for analyzing trends)

---

## Demographic Context

### Migration in Cohort-Component Models

**Basic equation:**
$$P(t+1) = P(t) + B(t) - D(t) + M(t)$$

Where:
- $P(t)$ = Population at time $t$
- $B(t)$ = Births
- $D(t)$ = Deaths
- $M(t)$ = **Net migration**

**By age:**
$$P_{x+1}(t+1) = P_x(t) \cdot S_x + M_x(t)$$

Where:
- $S_x$ = Survival rate (from mortality module)
- $M_x(t)$ = Net migrants at age $x$

**This module provides:** $M_x(t)$ by age and sex

---

### Types of Migration

**1. International Migration**
- Between countries
- Colombia ↔ Venezuela, USA, Spain, etc.
- **What this module handles:** 

**2. Internal Migration**
- Between departments within Colombia
- Bogotá ↔ Antioquia, etc.
- **Depends on aggregation level:**
  - National projections: Internal migration cancels out (one person's emigration = another's immigration)
  - Department projections: Internal migration matters

**3. Net vs Gross Migration**
- **Gross:** Separate in-flows and out-flows
- **Net:** Difference (in - out)
- **This module:** Computes **both** (stores gross, uses net)

---

### Age Pattern of Migration

**Typical Pattern:**
```
Age     Net Migration Rate  Interpretation
0-4     Low                 Young children (move with parents)
5-14    Low                 School-age (stable)
15-19   Moderate            Some education/work migration
20-24   HIGH                University, first jobs (peak mobility)
25-29   HIGH                Career establishment
30-34   Moderate            Family formation
35-44   Low-Moderate        Established careers
45-54   Low                 Pre-retirement stability
55-64   Very low            Near retirement
65+     Very low            Retirees (some return migration)
```

**Key Features:**
1. **Peak at young adult ages (20-34):** Labor mobility
2. **Low at young/old ages:** Dependents, retirees
3. **Sex differences:** 
   - Males often higher (labor migration)
   - Females may peak earlier (marriage migration)

**Rogers-Castro Model (standard in demography):**
$$m(x) = a_1 e^{-\alpha_1 x} + a_2 e^{-\alpha_2(x-\mu_2)^2} + a_3 e^{\lambda_3 x} + c$$

Parameters:
- $a_1, \alpha_1$: Childhood component
- $a_2, \mu_2, \alpha_2$: Labor force peak (age $\mu_2 \approx 25$)
- $a_3, \lambda_3$: Elderly component
- $c$: Constant

**This module:** Uses empirical rates (doesn't fit Rogers-Castro, but could)

---

## Data Structure

### Input: `conteos` DataFrame

**Expected structure:**
```
DPTO  ANO  EDAD   SEXO  VARIABLE                VALOR
05    2018  0-4    1     poblacion_total         50000
05    2018  0-4    1     flujo_inmigracion       120
05    2018  0-4    1     flujo_emigracion        80
05    2018  0-4    2     poblacion_total         48000
...
```

**Key features:**
- Long format (one row per observation)
- Multi-level: Department × Year × Age × Sex × Variable
- Flexible: Can have multiple years, departments

---

### Output: National Panel

**Structure:**
```
ANO  EDAD   SEXO  inmigracion_F  emigracion_F  net_migration  poblacion_total  net_mig_rate
2018  0-4    1     5,200          3,800         1,400          2,500,000        0.00056
2018  0-4    2     5,000          3,600         1,400          2,450,000        0.00057
2018  5-9    1     3,800          2,900         900            2,400,000        0.00038
...
```

**Key features:**
- Wide format (one row per age-sex-year)
- National aggregates (departments summed)
- Both gross and net flows
- Ready for projections

---

## Design Decisions

###  Good Choices

**1. Robust Data Cleaning**
```python
pd.to_numeric(..., errors="coerce").fillna(0.0)
```
- Handles invalid data gracefully
- Fills missing with sensible defaults (0)

**2. Flexible Year Filtering**
```python
if year is not None: ...
```
- Single year: For projections
- All years: For trend analysis

**3. Ordered Age Categories**
```python
pd.Categorical(EDAD, categories=EDAD_ORDER, ordered=True)
```
- Ensures correct sorting
- Prevents alphabetical chaos

**4. Conservative Merging**
```python
.merge(..., how="left")
```
- Keeps all population records
- Doesn't lose age groups without migration data

**5. Safe Division**
```python
np.where(denom > 0, net_migration / denom, np.nan)
```
- Avoids division by zero
- Returns NaN for invalid rates

---

###  Potential Issues

**1. Hardcoded Age Groups**
```python
EDAD_ORDER = ['0-4','5-9',...,'80+']
```

**Problem:** What if data has different age groups?
```
Input: ['0-1', '1-4', '5-9', ...]
Your code: Fails to recognize '0-1'
```

**Impact:** Age sorting breaks, might lose data

**Solution:** Make age groups flexible or validate input

---

**2. No Validation of Input Format**

**Problem:** Assumes specific column names:
- `VARIABLE`, `ANO`, `EDAD`, `SEXO`, `VALOR`

**What if:**
- Column named `Year` instead of `ANO`?
- `Age` instead of `EDAD`?

**Impact:** KeyError crash

**Solution:** Add input validation

---

**3. Silent Coercion to Zero**
```python
pd.to_numeric(..., errors="coerce").fillna(0.0)
```

**Problem:** Invalid data becomes 0 **silently**

**Example:**
```
Input: VALOR = "N/A" (data quality issue)
→ Coerced to 0
→ No warning!
```

**Impact:** Underestimates migration (zeros where should be NaN or error)

**Solution:** Log warnings for coerced values

---

**4. Aggregation Might Hide Department Patterns**

**Example:**
```
Department A: +1,000 migrants (gain)
Department B: -1,000 migrants (loss)
National total: 0 migrants

Projection assumes: No migration
Reality: Significant spatial redistribution
```

**When this matters:**
- Subnational projections
- Regional inequality analysis

**When it's OK:**
- National-only projections (your use case)

---

**5. No Temporal Smoothing**

**Problem:** Year-to-year fluctuations

**Example:**
```
2016: +5,000 migrants
2017: +15,000 migrants (Venezuela crisis)
2018: +8,000 migrants
```

**For projections:** Should you use 2018 value, or smooth across years?

**Current code:** Uses single year

**Better:** Could average 2016-2018, or fit trend

---

**6. Assumes Rates Constant Over Projection**

**Typical usage:**
```python
# Get 2018 rates
rates_2018 = create_migration_frame(conteos, year=2018)

# Project to 2050 using 2018 rates (assumed constant)
for year in range(2019, 2051):
    population[year] += population[year] * rates_2018['net_mig_rate']
```

**Problem:** Migration rates **change over time**

**Example (Colombia):**
```
2015: Small immigration (pre-crisis)
2018: Large immigration (Venezuela crisis)
2025: ??? (crisis may end, reverse flows)
```

**Solution:** Module is fine (processes single year correctly), but **downstream** projections need migration scenarios

---

## Use in Cohort-Component Projections

### Typical Workflow

```python
# 1. Load census/vital stats data
conteos = pd.read_rds("data/conteos.rds")

# 2. Create migration frame for base year
mig_2018 = create_migration_frame(conteos, year=2018)

# 3. Extract rates by age-sex
rates = {}
for _, row in mig_2018.iterrows():
    age = row['EDAD']
    sex = row['SEXO']
    rates[(age, sex)] = row['net_mig_rate']

# 4. Apply in projection loop
for year in range(2019, 2051):
    for age in age_groups:
        for sex in [1, 2]:
            # Survivors from previous age/year
            survivors = pop[age-1, sex, year-1] * survival_rate[age-1, sex]
            
            # Add net migrants
            migrants = pop[age, sex, year-1] * rates[(age, sex)]
            
            # Update population
            pop[age, sex, year] = survivors + migrants
```

---

## Comparison to Standard Practice

### Common Approaches

**1. Net Migration Numbers (not rates)**
```python
net_mig[age] = 5000  # Fixed number
```
- **Your code:** Uses rates (better - scales with population)

**2. Single National Rate (not age-specific)**
```python
net_mig_rate = 0.002  # Same for all ages
```
- **Your code:** Age-specific (better - captures age selectivity)

**3. No Sex Disaggregation**
```python
net_mig_rate[age] = 0.003  # Same for males and females
```
- **Your code:** Sex-specific (better - males ≠ females)

**Verdict:** Your approach is **best practice** (age- and sex-specific rates)

---

### Alternative Methods (Not Implemented)

**1. Rogers-Castro Model**
- Smooth age pattern
- Fewer parameters
- **Your code:** Empirical rates (more flexible but noisier)

**2. Cohort-Based Migration**
- Track specific cohorts over time
- Accounts for selection (healthier migrants)
- **Your code:** Period rates (standard approach)

**3. Origin-Destination Matrices**
- Full spatial detail
- Tracks flows between departments
- **Your code:** Net flows only (simpler, adequate for national projections)

---

## Summary

### What This Module Does

**Input:** Department-level population and migration data  
**Output:** National net migration rates by age and sex  
**Purpose:** Feed into cohort-component population projections

### Strengths

1.  Robust data cleaning (handles invalid values)
2.  Age- and sex-specific rates (best practice)
3.  Flexible year filtering (single or multiple)
4.  Safe calculations (no division by zero)
5.  Preserves gross flows (for analysis)

### Limitations

1.  Hardcoded age groups (not flexible)
2.  No input validation (assumes format)
3.  Silent coercion (missing data → 0)
4.  Single year (no temporal smoothing)
5.  Assumes constant rates (projection limitation)

### Overall Assessment

**For its purpose (processing migration data for projections):**  **Well done**

**Code quality:** 4/5 stars (could use more validation, flexibility)  
**Demographic correctness:** 5/5 stars (uses rates, age-sex-specific, correct formulas)

---

## References

**Migration in Demographic Projections:**
1. Rogers, A., & Castro, L. J. (1981). *Model Migration Schedules*. Research Report RR-81-030, IIASA.
2. Preston, S. H., Heuveline, P., & Guillot, M. (2001). *Demography: Measuring and Modeling Population Processes*. Blackwell, Chapter 7 (Migration).
3. UN (2018). *Manual on the Estimation and Projection of International Migration*. UN Department of Economic and Social Affairs.

**Cohort-Component Method:**
4. Smith, S. K., Tayman, J., & Swanson, D. A. (2013). *A Practitioner's Guide to State and Local Population Projections*. Springer, Chapter 5.
