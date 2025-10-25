# Visual Guide: Tail Harmonization Logic

## Overview Flowchart

```
Input Data with Open-Ended Tails (70+, 80+)
                |
                v
        ┌───────┴───────┐
        |               |
    Migration       Census/Deaths
    (use pop)       (use geometric)
        |               |
        v               v
┌───────────────┐  ┌────────────────┐
│ Match by      │  │ Check VARIABLE │
│ series_keys   │  │ type           │
└───────┬───────┘  └────────┬───────┘
        |                   |
        v                   v
┌───────────────┐  ┌────────────────┐
│ Pop available?│  │ poblacion or   │
│ Yes → weights │  │ defunciones?   │
│ No → geometric│  │ Yes → split    │
└───────┬───────┘  │ No → pass thru │
        |          └────────┬───────┘
        v                   v
┌───────────────────────────────────┐
│    Split 70+ → 5 bands            │
│    Split 80+ → 3 bands            │
│    Keep other ages unchanged      │
└───────────────┬───────────────────┘
                v
        ┌───────────────┐
        │ Aggregate     │
        │ duplicates    │
        └───────┬───────┘
                v
        Output (standard 5-year bands)
```

---

## Example 1: Migration with Population Template

### Input Migration Data
```
DPTO  SEX  EDAD   VALOR
001   M    60-64  100
001   M    65-69  150
001   M    70+    500  ← needs splitting
```

### Population Template (DPTO=001, SEX=M)
```
EDAD     VALOR_corrected  Weight
70-74    1000            37.0%
75-79     800            29.6%
80-84     500            18.5%
85-89     300            11.1%
90+       100             3.7%
Total:   2700           100.0%
```

### Calculation
```
Split 70+ total (500) by population weights:
  70-74: 500 × 0.370 = 185.2
  75-79: 500 × 0.296 = 148.1
  80-84: 500 × 0.185 =  92.6
  85-89: 500 × 0.111 =  55.6
  90+:   500 × 0.037 =  18.5
        Sum = 500.0 
```

### Output Migration Data
```
DPTO  SEX  EDAD   VALOR
001   M    60-64  100.0
001   M    65-69  150.0
001   M    70-74  185.2  ← from split
001   M    75-79  148.1  ← from split
001   M    80-84   92.6  ← from split
001   M    85-89   55.6  ← from split
001   M    90+     18.5  ← from split
```

---

## Example 2: Census Population with Geometric Weights

### Input Census Data
```
DPTO  VARIABLE          EDAD   VALOR_corrected
001   poblacion_total   60-64  50000
001   poblacion_total   65-69  40000
001   poblacion_total   70+    25000  ← needs splitting
```

### Geometric Weights (r=0.70, decreasing with age)
```
Formula: w[i] = 0.70^(n-1-i) / sum

j   Age      Calculation      Raw      Normalized
0   70-74    0.70^(5-1-0)    2.401     35.5%
1   75-79    0.70^(5-1-1)    1.681     24.8%
2   80-84    0.70^(5-1-2)    1.176     17.4%
3   85-89    0.70^(5-1-3)    0.823     12.2%
4   90+      0.70^(5-1-4)    0.576     10.1%
                Sum:         6.757    100.0%
```

### Calculation
```
Split 70+ total (25000) by geometric weights:
  70-74: 25000 × 0.355 = 8875
  75-79: 25000 × 0.248 = 6200
  80-84: 25000 × 0.174 = 4350
  85-89: 25000 × 0.122 = 3050
  90+:   25000 × 0.101 = 2525
         Sum = 25000 
```

### Output Census Data
```
DPTO  VARIABLE          EDAD   VALOR_corrected
001   poblacion_total   60-64  50000
001   poblacion_total   65-69  40000
001   poblacion_total   70-74   8875  ← from split
001   poblacion_total   75-79   6200  ← from split
001   poblacion_total   80-84   4350  ← from split
001   poblacion_total   85-89   3050  ← from split
001   poblacion_total   90+     2525  ← from split
```

---

## Example 3: Deaths with Geometric Weights (Increasing)

### Input Death Data
```
DPTO  VARIABLE      EDAD   VALOR_corrected
001   defunciones   60-64  100
001   defunciones   65-69  150
001   defunciones   70+    500  ← needs splitting
```

### Geometric Weights (r=1.45, increasing with age)
```
Formula: w[i] = 1.45^i / sum

j   Age      Calculation      Raw      Normalized
0   70-74    1.45^0          1.000      9.8%
1   75-79    1.45^1          1.450     14.2%
2   80-84    1.45^2          2.103     20.6%
3   85-89    1.45^3          3.049     29.9%
4   90+      1.45^4          4.421     25.5%
                Sum:        10.203    100.0%
```

### Calculation
```
Split 70+ total (500) by geometric weights:
  70-74: 500 × 0.098 =  49
  75-79: 500 × 0.142 =  71
  80-84: 500 × 0.206 = 103
  85-89: 500 × 0.299 = 150
  90+:   500 × 0.255 = 127
         Sum = 500 
```

### Output Death Data
```
DPTO  VARIABLE      EDAD   VALOR_corrected
001   defunciones   60-64  100
001   defunciones   65-69  150
001   defunciones   70-74   49  ← from split (smaller)
001   defunciones   75-79   71  ← from split
001   defunciones   80-84  103  ← from split
001   defunciones   85-89  150  ← from split
001   defunciones   90+    127  ← from split (larger)
```

**Note:** Deaths increase with age (more deaths in older bands), opposite to population.

---

## Example 4: Duplicate Handling (70+ and 70-74)

### Input (Conflicting)
```
DPTO  SEX  EDAD   VALOR
001   M    70+    500    ← to be split
001   M    70-74  100    ← already present
```

### After Split (Before Aggregation)
```
DPTO  SEX  EDAD   VALOR
001   M    70-74  185.2  ← from 70+ split
001   M    75-79  148.1  ← from 70+ split
001   M    80-84   92.6  ← from 70+ split
001   M    85-89   55.6  ← from 70+ split
001   M    90+     18.5  ← from 70+ split
001   M    70-74  100.0  ← original row
```

### After Aggregation (groupby + sum)
```
DPTO  SEX  EDAD   VALOR
001   M    70-74  285.2  ← 185.2 + 100.0
001   M    75-79  148.1
001   M    80-84   92.6
001   M    85-89   55.6
001   M    90+     18.5
Total:           600.0  (500 + 100)
```

---

## Example 5: Both 70+ and 80+ (Potential Double-Count?)

### Input (Ambiguous)
```
DPTO  SEX  EDAD   VALOR
001   M    70+    500
001   M    80+    200
```

### Question: Is 80+ a subset of 70+, or separate?

### Scenario A: Separate Records (Correct Aggregation)
```
70+ split:
  70-74: 185.2
  75-79: 148.1
  80-84:  92.6  ← part of 70+
  85-89:  55.6  ← part of 70+
  90+:    18.5  ← part of 70+

80+ split:
  80-84:  74.1  ← separate from 70+
  85-89:  55.6  ← separate from 70+
  90+:    70.4  ← separate from 70+

Aggregated:
  70-74: 185.2  (only from 70+)
  75-79: 148.1  (only from 70+)
  80-84: 166.7  (92.6 + 74.1) 
  85-89: 111.2  (55.6 + 55.6) 
  90+:    88.9  (18.5 + 70.4) 
  Total: 700.0  (500 + 200)
```

### Scenario B: 80+ is Subset of 70+ (Double-Count!)
```
If 80+ is already included in 70+, then we're counting it twice!

Correct total should be: 500 (not 700)
But output gives: 700

This is WRONG if 80+ ⊂ 70+
```

### Recommendation
Add validation:
```python
if has_70_plus and has_80_plus:
    raise ValueError(
        "Both '70+' and '80+' present in same group. "
        "Cannot determine if '80+' is subset or separate. "
        "Please clarify data semantics."
    )
```

---

## Weight Comparison: Population vs Geometric

### Population-Based (Demographic Reality)
```
Based on actual age distribution
Example (DPTO=001, SEX=M):
  70-74: 37.0%  (largest, recent retirees)
  75-79: 29.6%
  80-84: 18.5%
  85-89: 11.1%
  90+:    3.7%  (smallest, few survivors)
```

### Geometric Decreasing (r=0.70, Population Proxy)
```
Approximates population distribution
Smooth exponential decay:
  70-74: 35.5%  (close to 37.0%)
  75-79: 24.8%  (close to 29.6%)
  80-84: 17.4%  (close to 18.5%)
  85-89: 12.2%  (close to 11.1%)
  90+:   10.1%  (overestimates vs 3.7%)
```

### Geometric Increasing (r=1.45, Death Concentration)
```
Captures mortality concentration at old ages
Weights increase with age:
  70-74:  9.8%  (few deaths, healthier cohort)
  75-79: 14.2%
  80-84: 20.6%
  85-89: 29.9%  (peak mortality risk)
  90+:   25.5%  (many deaths, though small cohort)
```

---

## Key Insights

### When to Use Population-Based
 **Migration data:** Migrant age structure mirrors population  
 **High-quality data:** Population counts available per group  
 **Regional variation:** Different regions have different age structures

### When to Use Geometric
 **Census/deaths:** No template needed  
 **Missing data:** Population unavailable for some groups  
 **Consistency:** Same pattern applied across all groups

### Geometric Ratio Selection
- **r < 1.0:** Decreasing pattern (younger > older)
  - Use for: Population, labor force, economic activity
- **r = 1.0:** Uniform weights (all equal)
  - Use for: Maximum entropy, no prior information
- **r > 1.0:** Increasing pattern (older > younger)
  - Use for: Deaths, disabilities, care needs

---

## Mathematical Properties

### Weight Normalization
All methods ensure: $\sum_{i} w_i = 1.0$

### Total Preservation
If input has $N$ individuals in "70+":
$$\text{Total Output} = \sum_{i \in \{70-74, ..., 90+\}} N \cdot w_i = N \cdot \sum_i w_i = N$$

Always exact to floating-point precision (tested < 1e-10).

### Aggregation Commutes
Order of operations doesn't matter:
```
(Split then Aggregate) = (Aggregate then Split)
```

Tested with duplicate EDAD labels and overlapping tails.
