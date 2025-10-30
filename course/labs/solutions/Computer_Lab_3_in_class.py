#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Computer Lab 3: Fertility Analysis
# In-Class Version - Streamlined for teaching

# # Demographic Research Methods and the PyCCM Library
# ## Computer Lab 3: Fertility Analysis
# ## Instructor: Jiani Yan
# ## Date: October 29th, 2025
# ---




# ### 1.1 Load in the conteos.rds file, using the data_loaders module or otherwise.

# ---- code cell ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os

os.chdir('/Users/valler/Python/PyCCM/course/labs/solutions')

# Import PyCCM functions
from data_loaders import load_all_data, get_fertility
from fertility import compute_asfr, get_target_params
from helpers import _tfr_from_asfr_df, _smooth_tfr

# Path to the data file
data_path = Path.cwd().parents[2]/'data'


# Load the data using PyCCM data_loaders
data_dict = load_all_data(data_path)
df = pd.DataFrame(data_dict['conteos'])

print(f"✓ Loaded conteos data: {len(df):,} rows")
print(f"  Years: {df['ANO'].min():.0f} - {df['ANO'].max():.0f}")
print(f"  Variables: {', '.join(df['VARIABLE'].unique())}")

# ### 1.2 Filter for EEVV fertility and censos exposures for births and deaths respectively

# ---- code cell ----
year = 2018

# Get births from EEVV (vital statistics - more reliable for births)
df_births = df[(df['ANO'] == year) & 
               (df['FUENTE'] == 'EEVV') & 
               (df['VARIABLE'] == 'nacimientos')].copy()

# Get population from censo (census - better coverage for exposures)
df_pop = df[(df['ANO'] == year) & 
            (df['FUENTE'] == f'censo_{year}') & 
            (df['VARIABLE'] == 'poblacion_total')].copy()

print(f"\n✓ Filtered data for {year}:")
print(f"  Births (EEVV): {df_births['VALOR'].sum():,.0f} total births")
print(f"  Population (censo_{year}): {df_pop['VALOR'].sum():,.0f} total population")

# ### 1.3 Filter for your favourite DPTO

# ---- code cell ----
favorite_dpto = 'BOLIVAR'

df_births_dpto = df_births[df_births['DPTO_NOMBRE'] == favorite_dpto].copy()
df_pop_dpto = df_pop[df_pop['DPTO_NOMBRE'] == favorite_dpto].copy()

print(f"\n✓ Filtered for {favorite_dpto}:")
print(f"  Births: {df_births_dpto['VALOR'].sum():,.0f}")
print(f"  Population: {df_pop_dpto['VALOR'].sum():,.0f}")

# ### 1.4 (Optional) Account for Omissions by increasing the exposures and births accordingly

# ---- code cell ----
print("\n" + "="*60)
print("OPTIONAL: Adjusting for Census Omissions")
print("="*60)

# The OMISION column contains omission rates (e.g., 3.0 means 3% undercount)
# Adjust births and population upward to account for undercount
if 'OMISION' in df_births_dpto.columns:
    # births_adjusted = births / (1 - omission_rate/100)
    df_births_dpto['VALOR_adjusted'] = df_births_dpto['VALOR'] / (1 - df_births_dpto['OMISION']/100)
    df_pop_dpto['VALOR_adjusted'] = df_pop_dpto['VALOR']
    
    births_original = df_births_dpto['VALOR'].sum()
    births_adjusted = df_births_dpto['VALOR_adjusted'].sum()
    
    print(f"\nOmission adjustment for {favorite_dpto}:")
    print(f"  Original births:  {births_original:,.0f}")
    print(f"  Adjusted births:  {births_adjusted:,.0f}")
    print(f"  Increase:         {births_adjusted - births_original:,.0f} ({(births_adjusted/births_original - 1)*100:.1f}%)")
    print("\n✓ Using adjusted values for analysis")
    
    # Use adjusted values
    value_col = 'VALOR_adjusted'
else:
    print("\nNo OMISION column found - using unadjusted values")
    value_col = 'VALOR'

# ### 1.5 Aggregate births from males and females (focus on female fertility)

# ---- code cell ----
# For fertility analysis, we typically focus on FEMALE population (SEXO=2)
# and total births (both sexes), or births to mothers

# Filter for females only (reproductive ages)
reproductive_ages = ['10-14','15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']

# Female population (SEXO=2)
df_pop_female = df_pop_dpto[df_pop_dpto['SEXO'] == 2.0].copy()

# Births - sum across both sexes (total births to mothers)
df_births_agg = df_births_dpto.groupby('EDAD')[value_col].sum().reset_index()
df_births_agg.columns = ['EDAD', 'births']

# Female population by age
df_pop_agg = df_pop_female.groupby('EDAD')[value_col].sum().reset_index()
df_pop_agg.columns = ['EDAD', 'population']

print(f"\n✓ Aggregated data:")
print(f"  Total births (all ages): {df_births_agg['births'].sum():,.0f}")
print(f"  Female population (all ages): {df_pop_agg['population'].sum():,.0f}")
print(f"  Reproductive age groups ({len(reproductive_ages)}): {', '.join(reproductive_ages)}")

# ### 2. Calculate ASFRs.

# ---- code cell ----
print("\n" + "="*60)
print("Section 2: Calculate Age-Specific Fertility Rates (ASFR)")
print("="*60)

# Merge births and population data
df_fertility = df_births_agg.merge(df_pop_agg, on='EDAD', how='inner')

# Filter for reproductive ages only
df_fertility = df_fertility[df_fertility['EDAD'].isin(reproductive_ages)].copy()

# Use PyCCM's compute_asfr function
asfr_result = compute_asfr(
    ages=reproductive_ages,
    population=df_fertility.set_index('EDAD')['population'],
    births=df_fertility.set_index('EDAD')['births'],
    validate=True  # Check biological plausibility
)



""" 
births = df_fertility.set_index('EDAD')['births']
births['45-49'] = 50000
# Use PyCCM's compute_asfr function
asfr_result = compute_asfr(
    ages=reproductive_ages,
    population=df_fertility.set_index('EDAD')['population'],
    births=births,
    validate=True  # Check biological plausibility
)
"""


print("\nAge-Specific Fertility Rates (ASFR):")
print(asfr_result)

# ### 3. From these ASFRs, calculate the TFR.

# ---- code cell ----
print("\n" + "="*60)
print("Section 3: Calculate Total Fertility Rate (TFR)")
print("="*60)

# Use PyCCM's helper function to calculate TFR
baseline_tfr = _tfr_from_asfr_df(asfr_result)

print(f"\nTotal Fertility Rate (TFR) for {favorite_dpto} in {year}:")
print(f"  TFR = {baseline_tfr:.3f} children per woman")
print(f"\nInterpretation:")
print(f"  - TFR represents the average number of children a woman would have")
print(f"    over her lifetime if current age-specific rates remain constant")
if baseline_tfr < 2.1:
    print(f"  - Below replacement level (2.1): population will decline without migration")
elif baseline_tfr < 3.0:
    print(f"  - Moderate fertility: near replacement or modest growth")
else:
    print(f"  - High fertility: population will grow rapidly")

# ### 4. Using functionality from PyCCM, adjust the TFR towards a target variable at _one step_

# ---- code cell ----
print("\n" + "="*60)
print("Section 4: Adjust TFR Towards Target (Single Step)")
print("="*60)

# Load target TFRs from the example file
target_file = data_path / 'target_tfrs_example.csv'

if target_file.exists():
    targets_dict, conv_years_dict = get_target_params(str(target_file))
    
    # Get target for our department (or use default)
    target_tfr = targets_dict.get(favorite_dpto, 1.5)
    convergence_years = conv_years_dict.get(favorite_dpto, 50)
    
    print(f"\nTarget TFR parameters:")
    print(f"  Current TFR:       {baseline_tfr:.3f}")
    print(f"  Target TFR:        {target_tfr:.3f}")
    print(f"  Convergence years: {convergence_years}")
else:
    # Use default values
    target_tfr = 1.5
    convergence_years = 50
    print(f"\nUsing default target TFR: {target_tfr:.3f} over {convergence_years} years")

# Calculate TFR at step=1 (after 1 year) using exponential smoothing
tfr_step1 = _smooth_tfr(
    TFR0=baseline_tfr,
    target=target_tfr,
    years=convergence_years,
    step=1,
    kind='logistic'  # "exp", "logistic"
)

print(f"\nTFR after 1 year of adjustment:")
print(f"  Year 0 (baseline): {baseline_tfr:.3f}")
print(f"  Year 1 (adjusted): {tfr_step1:.3f}")
print(f"  Change:            {tfr_step1 - baseline_tfr:.3f} ({(tfr_step1/baseline_tfr - 1)*100:.2f}%)")

# ### 5. With this adjusted TFR, re-calculate the ASFR.

# ---- code cell ----
print("\n" + "="*60)
print("Section 5: Re-calculate ASFR with Adjusted TFR")
print("="*60)

# Proportionally adjust all ASFRs to match the new TFR
# New ASFR = Old ASFR × (New TFR / Old TFR)
scaling_factor = tfr_step1 / baseline_tfr

asfr_adjusted = asfr_result.copy()
asfr_adjusted['asfr'] = asfr_result['asfr'] * scaling_factor

# Verify the new TFR
new_tfr = _tfr_from_asfr_df(asfr_adjusted)

print(f"\nAdjusted ASFRs (Year 1):")
print(asfr_adjusted[['asfr']])
print(f"\nVerification:")
print(f"  Target TFR for step 1: {tfr_step1:.3f}")
print(f"  Actual TFR from ASFRs: {new_tfr:.3f}")
print(f"  Match: {'✓' if abs(new_tfr - tfr_step1) < 0.001 else '✗'}")

# ### 6. Do this in a loop over an appropriate horizon, saving your incremental TFRs and ASFRs

# ---- code cell ----
print("\n" + "="*60)
print("Section 6: Project TFR Over Time Horizon")
print("="*60)

# Project over the convergence horizon
projection_years = min(convergence_years, 100)  # Cap at 100 years for visualization

tfr_trajectory = []
years_list = []

for step in range(projection_years + 1):
    tfr_at_step = _smooth_tfr(
        TFR0=baseline_tfr,
        target=target_tfr,
        years=convergence_years,
        step=step,
        kind='exp'
    )
    tfr_trajectory.append(tfr_at_step)
    years_list.append(year + step)

# Create trajectory dataframe
df_trajectory = pd.DataFrame({
    'year': years_list,
    'tfr': tfr_trajectory
})

print(f"\nProjected TFR trajectory ({projection_years} years):")
print(f"  Starting year: {year} (TFR={baseline_tfr:.3f})")
print(f"  Final year:    {year + projection_years} (TFR={tfr_trajectory[-1]:.3f})")
print(f"\nSample milestones:")
for milestone in [0, 10, 25, 50]:
    if milestone <= projection_years:
        idx = milestone
        print(f"  Year {years_list[idx]}: TFR = {tfr_trajectory[idx]:.3f}")

# ### 7. Visualise the TFR pathway.

# ---- code cell ----
print("\n" + "="*60)
print("Section 7: Visualize TFR Trajectory")
print("="*60)

plt.figure(figsize=(12, 6))
plt.plot(df_trajectory['year'], df_trajectory['tfr'], 
         linewidth=2.5, color='darkblue', label='Projected TFR')
plt.axhline(y=target_tfr, color='red', linestyle='--', linewidth=2, 
            label=f'Target TFR = {target_tfr:.2f}')
plt.axhline(y=2.1, color='gray', linestyle=':', linewidth=1.5, 
            label='Replacement level (2.1)')
plt.axhline(y=baseline_tfr, color='green', linestyle='--', linewidth=1.5, 
            alpha=0.7, label=f'Baseline TFR = {baseline_tfr:.2f}')

plt.xlabel('Year', fontsize=12)
plt.ylabel('Total Fertility Rate (TFR)', fontsize=12)
plt.title(f'TFR Projection for {favorite_dpto}: {year}-{year + projection_years}\n'
          f'Exponential convergence to target over {convergence_years} years',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.show()

print("\n✓ TFR trajectory visualization complete")

# ### 8. Repeat this, changing the target improvement parameters. Visualise your results.

# ---- code cell ----
print("\n" + "="*60)
print("Section 8: Compare Different Convergence Scenarios")
print("="*60)

# Define multiple scenarios
scenarios = [
    {'name': 'Fast (25 years)', 'years': 25, 'color': 'red'},
    {'name': 'Medium (50 years)', 'years': 50, 'color': 'blue'},
    {'name': 'Slow (75 years)', 'years': 75, 'color': 'green'},
    {'name': 'Logistic (50 years)', 'years': 50, 'kind': 'logistic', 'color': 'purple'}
]

plt.figure(figsize=(14, 7))

for scenario in scenarios:
    conv_years = scenario['years']
    kind = scenario.get('kind', 'exp')
    
    tfr_scenario = []
    for step in range(projection_years + 1):
        tfr_at_step = _smooth_tfr(
            TFR0=baseline_tfr,
            target=target_tfr,
            years=conv_years,
            step=step,
            kind=kind
        )
        tfr_scenario.append(tfr_at_step)
    
    plt.plot(years_list, tfr_scenario, linewidth=2.5, 
             color=scenario['color'], label=scenario['name'])

# Add reference lines
plt.axhline(y=target_tfr, color='black', linestyle='--', linewidth=2, 
            alpha=0.7, label=f'Target TFR = {target_tfr:.2f}')
plt.axhline(y=2.1, color='gray', linestyle=':', linewidth=1.5, 
            label='Replacement level (2.1)')

plt.xlabel('Year', fontsize=12)
plt.ylabel('Total Fertility Rate (TFR)', fontsize=12)
plt.title(f'TFR Projection Scenarios for {favorite_dpto}\n'
          f'Comparing different convergence speeds and methods',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.show()

print("\n✓ Scenario comparison complete")
print(f"\nKey Insights:")
print(f"  - Faster convergence (25 years) reaches target quickly but may be less realistic")
print(f"  - Slower convergence (75 years) is more gradual and potentially more realistic")
print(f"  - Logistic method has an S-shaped curve with slower start and faster middle transition")
print(f"  - Exponential methods have constant proportional change (smooth decay)")

# ### BONUS: Visualize ASFR patterns

# ---- code cell ----
print("\n" + "="*60)
print("BONUS: Visualize Age-Specific Fertility Patterns")
print("="*60)

# Extract age midpoints for plotting
age_midpoints = [12, 17, 22, 27, 32, 37, 42, 47]

plt.figure(figsize=(12, 6))
plt.bar(age_midpoints, asfr_result['asfr'], width=4.5, 
        alpha=0.7, color='steelblue', edgecolor='navy', label=f'Baseline ({year})')

# Add adjusted ASFR for comparison
plt.bar([x + 0.3 for x in age_midpoints], asfr_adjusted['asfr'], width=4.5, 
        alpha=0.5, color='coral', edgecolor='darkred', label='After 1 year adjustment')

plt.xlabel('Age Group (midpoint)', fontsize=12)
plt.ylabel('Age-Specific Fertility Rate (births per woman)', fontsize=12)
plt.title(f'Age-Specific Fertility Rates - {favorite_dpto}, {year}\n'
          f'Baseline vs. 1-Year Adjusted',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Find peak fertility age
peak_age = asfr_result['asfr'].idxmax()
peak_rate = asfr_result['asfr'].max()

print(f"\nFertility Pattern Analysis:")
print(f"  Peak fertility age group: {peak_age}")
print(f"  Peak fertility rate:      {peak_rate:.4f} births per woman")
print(f"  Total TFR:                {baseline_tfr:.3f} children per woman")

print("\n" + "="*60)
print("LAB 3 COMPLETE! 🎉")
print("="*60)
print(f"\nKey Results for {favorite_dpto} ({year}):")
print(f"  Baseline TFR:     {baseline_tfr:.3f}")
print(f"  Target TFR:       {target_tfr:.3f}")
print(f"  Convergence:      {convergence_years} years")
print(f"  Peak fertility:   Age {peak_age}")
print(f"\nYou've successfully:")
print(f"  ✓ Calculated ASFRs from births and population data")
print(f"  ✓ Computed TFR from ASFRs")
print(f"  ✓ Applied TFR adjustment towards a target")
print(f"  ✓ Projected fertility over time using PyCCM smoothing functions")
print(f"  ✓ Compared different convergence scenarios")
print(f"  ✓ Visualized fertility patterns and trajectories")

