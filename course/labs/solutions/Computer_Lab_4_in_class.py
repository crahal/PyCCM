#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Computer Lab 4: Combining previous labs into a projection
# In-Class Version - Streamlined for teaching

# # Demographic Research Methods and the PyCCM Library
# ## Computer Lab 4: Building Leslie Matrices and Population Projections
# ## Instructor: Jiani Yan
# ## Date: October 30th, 2025
# ---

# ## Section 1: Loading data and unabridging

# ### 1.1 Load in the conteos.rds file, using the data_loaders module or otherwise.

# ---- code cell ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os

os.chdir('/Users/valler/Python/PyCCM/src') # change to your own path


# Import PyCCM functions
from data_loaders import load_all_data
from abridger import unabridge_df
from fertility import compute_asfr, get_target_params
from mortality import make_lifetable
from migration import create_migration_frame
from helpers import _collapse_defunciones_01_24_to_04, _tfr_from_asfr_df, _smooth_tfr


# Path to the data file
data_path = Path.cwd().parent/'data'



# Load the data using PyCCM data_loaders
data_dict = load_all_data(data_path)
df = pd.DataFrame(data_dict['conteos'])

print("="*60)
print("LAB 4: POPULATION PROJECTIONS WITH LESLIE MATRICES")
print("="*60)
print(f"\n✓ Loaded conteos data: {len(df):,} rows")

# ### 1.2 Filter for your favorite DPTO. Focus on FEMALES first. 
# Obtain one array/series for population counts, deaths, and fertility.

# ---- code cell ----
year = 2018
dpto = 'BOLIVAR'

# Filter for department and year
df_dpto = df[(df['ANO'] == year) & (df['DPTO_NOMBRE'] == dpto)].copy()

# Get FEMALE data (SEXO = 2)
df_female = df_dpto[df_dpto['SEXO'] == 2.0].copy()

# Extract population, deaths, and births for females
df_pop_f = df_female[(df_female['VARIABLE'] == 'poblacion_total') & 
                     (df_female['FUENTE'] == f'censo_{year}')].copy()

df_deaths_f = df_female[(df_female['VARIABLE'] == 'defunciones') & 
                        (df_female['FUENTE'] == f'censo_{year}')].copy()

# Births use EEVV (vital statistics) - note births to mothers, so use female data
df_births_f = df_female[(df_female['VARIABLE'] == 'nacimientos') & 
                        (df_female['FUENTE'] == 'EEVV')].copy()

print(f"\n✓ Filtered FEMALE data for {dpto} ({year}):")
print(f"  Population: {df_pop_f['VALOR'].sum():,.0f}")
print(f"  Deaths:     {df_deaths_f['VALOR'].sum():,.0f}")
print(f"  Births:     {df_births_f['VALOR'].sum():,.0f}")

# ### 1.3 Unabridge each of these series using tools from PyCCM.

# ---- code cell ----
print("\n" + "="*60)
print("Section 1.3: Unabridging to Single-Year Age Groups")
print("="*60)

# Fix infant deaths (collapse 0-1 and 2-4 into 0-4)
df_deaths_f_fixed = _collapse_defunciones_01_24_to_04(df_deaths_f)

# Unabridge population (5-year → 1-year)
df_pop_f_1yr = unabridge_df(
    df_pop_f,
    series_keys=['DPTO_NOMBRE', 'ANO', 'SEXO', 'VARIABLE', 'FUENTE'],
    value_col='VALOR'
)

# Unabridge deaths (5-year → 1-year)
df_deaths_f_1yr = unabridge_df(
    df_deaths_f_fixed,
    series_keys=['DPTO_NOMBRE', 'ANO', 'SEXO', 'VARIABLE', 'FUENTE'],
    value_col='VALOR'
)

# Unabridge births (5-year → 1-year)
df_births_f_1yr = unabridge_df(
    df_births_f,
    series_keys=['DPTO_NOMBRE', 'ANO', 'SEXO', 'VARIABLE', 'FUENTE'],
    value_col='VALOR')


print(f"\n✓ Unabridged to single-year ages:")
print(f"  Ages: 0, 1, 2, ..., 89, 90+")
print(f"  Population ages: {len(df_pop_f_1yr['EDAD'].unique())} age groups")
print(f"  Deaths ages:     {len(df_deaths_f_1yr['EDAD'].unique())} age groups")
print(f"  Births ages:     {len(df_births_f_1yr['EDAD'].unique())} age groups")

# ### 1.4 Calculate survivorship ratios and fertility rates on unabridged data

# ---- code cell ----
print("\n" + "="*60)
print("Section 1.4: Calculate Survivorship & Fertility")
print("="*60)

# Prepare population and deaths for life table
ages_1yr = [str(i) for i in range(0, 90)] + ['90+']

# remove any rows with NaN ages

# Filter out NaN ages and align to expected age range
df_pop_f_1yr = df_pop_f_1yr[df_pop_f_1yr['EDAD'].notna()]
df_deaths_f_1yr = df_deaths_f_1yr[df_deaths_f_1yr['EDAD'].notna()]
# df_deaths_f_1yr = df_deaths_f_1yr[df_deaths_f_1yr.notna().all(axis=1)]  # ensure no NaN in any column

# Reindex to ensure we have all ages
df_pop_f_1yr.set_index('EDAD', inplace=True)
df_pop_f_1yr = df_pop_f_1yr.reindex(ages_1yr, fill_value=0.0)

df_deaths_f_1yr.set_index('EDAD', inplace=True)
df_deaths_f_1yr = df_deaths_f_1yr.reindex(ages_1yr, fill_value=0.0)

# Build life table using PyCCM - pass the Series with age index
lt_female = make_lifetable(
    ages=df_pop_f_1yr.index,  # Pass the age index
    population=df_pop_f_1yr['VALOR'].values,
    deaths=df_deaths_f_1yr['VALOR'].values,
    radix=100000,
    use_ma=True,  # Moving average smoothing
    ma_window=5
)

print(f"\n✓ Life table calculated:")
print(f"  Female life expectancy (e_0): {lt_female.iloc[0]['ex']:.2f} years")

# Calculate survivorship ratios (S_x = l_{x+1} / l_x)
# This tells us what fraction survives from age x to age x+1
survivorship_f = np.zeros(len(ages_1yr))
lx_values = lt_female['lx'].values

for i in range(len(ages_1yr) - 1):
    if lx_values[i] > 0:
        survivorship_f[i] = lx_values[i + 1] / lx_values[i]
    else:
        survivorship_f[i] = 0.0

# Last age group (open interval) - use life expectancy
ex_last = lt_female.iloc[-1]['ex']
if ex_last > 0:
    survivorship_f[-1] = np.exp(-1.0 / ex_last)  # Approximate survival, e^(-1/ex)
else:
    survivorship_f[-1] = 0.0

print(f"  Survivorship ratios calculated: {len(survivorship_f)} values")
print(f"  Infant survival (age 0→1): {survivorship_f[0]:.4f}")
print(f"  Survival at age 50→51:     {survivorship_f[50]:.4f}")

# Calculate ASFRs (reproductive ages 10-49)
reproductive_ages = [str(i) for i in range(10, 50)]

births_f_by_age = df_births_f_1yr.groupby('EDAD')['VALOR'].sum()

# Use PyCCM's compute_asfr
asfr_result = compute_asfr(
    ages=reproductive_ages,
    population=df_pop_f_1yr['VALOR'],
    births=births_f_by_age,
    validate=True
)

asfr_values = asfr_result['asfr'].values
tfr_female = _tfr_from_asfr_df(asfr_result)

print(f"\n✓ Fertility calculated:")
print(f"  TFR (Total Fertility Rate): {tfr_female:.3f}")
print(f"  Reproductive ages: {reproductive_ages[0]} to {reproductive_ages[-1]}")
print(f"  Peak fertility age: {asfr_result['asfr'].idxmax()}")

# ### 1.5 Summary: We now have the three key inputs for Leslie Matrix

# ---- code cell ----
print("\n" + "="*60)
print("Section 1.5: Summary of Key Inputs")
print("="*60)

# Create population vector (age 0 to 90+)
population_f = df_pop_f_1yr['VALOR'].values  # check the order matches ages_1yr

print(f"\n✓ Three key components for Leslie Matrix:")
print(f"  1. Population counts:     {len(population_f)} ages (0 to 90+)")
print(f"  2. Survivorship ratios:   {len(survivorship_f)} values")
print(f"  3. ASFRs:                 {len(asfr_values)} values (ages {reproductive_ages[0]}-{reproductive_ages[-1]})")
print(f"\nTotal female population: {population_f.sum():,.0f}")

# ## Section 2: Building our first Leslie Matrix

# ### 2.1 Define an empty numpy array of the appropriate size

# ---- code cell ----
print("\n" + "="*60)
print("Section 2: Building Leslie Matrix for FEMALES")
print("="*60)

n_ages = len(ages_1yr)  # 91 ages (0, 1, ..., 89, 90+)

# Create empty Leslie matrix
leslie_matrix_F = np.zeros((n_ages, n_ages))

print(f"\n✓ Created empty Leslie matrix: {n_ages} × {n_ages}")
print(f"  Shape: {leslie_matrix_F.shape}")

# ### 2.2 Survive your ASFRs as defined in the lecture

# ---- code cell ----
print("\n--- Step 2.2: Survive ASFRs ---")

# The first row of the Leslie matrix contains "survived fertility"
# Formula: ASFR_x * S_x * (1 + ASFR_{x+1} * S_{x+1}) / 2
# This accounts for births happening throughout the year and survival

# We need to adjust ASFR by sex ratio at birth (SRB)
# Get male births for SRB calculation
df_male = df_dpto[df_dpto['SEXO'] == 1.0].copy()
df_births_m = df_male[(df_male['VARIABLE'] == 'nacimientos') & 
                      (df_male['FUENTE'] == 'EEVV')].copy()

total_births_f = df_births_f['VALOR'].sum()
total_births_m = df_births_m['VALOR'].sum()

# SRB = male births / female births
srb = total_births_m / total_births_f if total_births_f > 0 else 1.05
# Proportion female = 1 / (1 + SRB)
prop_female = 1.0 / (1.0 + srb)

print(f"\nSex Ratio at Birth (SRB): {srb:.3f}")
print(f"Proportion female births: {prop_female:.3f}")

# Create survived ASFR array (full age range 0-90)
survived_asfr = np.zeros(n_ages)

# Place ASFRs in appropriate positions (ages 10-49)
for i, age_str in enumerate(reproductive_ages):
    age_idx = int(age_str)
    
    # Basic survived ASFR with sex ratio adjustment
    # ASFR * proportion_female * survival to next year
    if age_idx < n_ages - 1:
        survived_asfr[age_idx] = (
            asfr_values[i] * prop_female * 
            (survivorship_f[age_idx] + survivorship_f[age_idx + 1]) / 2 )
    else:
        survived_asfr[age_idx] = asfr_values[i] * prop_female * survivorship_f[age_idx]

print(f"\n✓ Survived ASFRs calculated")
print(f"  Non-zero fertility ages: {np.sum(survived_asfr > 0)}")
print(f"  Max survived ASFR: {survived_asfr.max():.4f} at age {survived_asfr.argmax()}")

# ### 2.3 Position survived ASFR into the first row of the Leslie Matrix

# ---- code cell ----
print("\n--- Step 2.3: Place Survived ASFR in First Row ---")

# The first row represents births from mothers of all ages
leslie_matrix_F[0, :] = survived_asfr

print(f"\n✓ First row populated with survived ASFRs")
print(f"  Non-zero entries in first row: {np.sum(leslie_matrix_F[0, :] > 0)}")
print(f"  Sample first row [ages 20-25]: {leslie_matrix_F[0, 20:26]}")

# ### 2.4 (This section duplicates 2.3, so we'll skip to 2.5)

# ### 2.5 Position survivorship ratios on the lower off-diagonal

# ---- code cell ----
print("\n--- Step 2.5: Place Survivorship on Sub-diagonal ---")

# The sub-diagonal (A[i+1, i]) represents aging/survival
# People at age x survive to age x+1 with probability S_x

for i in range(n_ages - 1):
    leslie_matrix_F[i + 1, i] = survivorship_f[i]

# Last age group stays in last age group (90+ → 90+)
leslie_matrix_F[-1, -1] = survivorship_f[-1]

print(f"\n✓ Sub-diagonal populated with survivorship ratios")
print(f"  Sample survivorship [ages 0-5]: {survivorship_f[0:6]}")
print(f"  Last age group survival (90+): {survivorship_f[-1]:.4f}")

print("\n✓ Leslie Matrix COMPLETE!")
print(f"  Total non-zero entries: {np.sum(leslie_matrix_F > 0)}")

# ### 2.6 Multiply this matrix by your population count array

# ---- code cell ----
print("\n" + "="*60)
print("Section 2.6: Project Population One Step Forward")
print("="*60)

# Project one year forward: pop_{t+1} = L * pop_t
population_f_next = leslie_matrix_F @ population_f # Matrix multiplication

print(f"\nPopulation projection (1 year forward):")
print(f"  Current (year {year}):      {population_f.sum():,.0f}")
print(f"  Projected (year {year+1}):  {population_f_next.sum():,.0f}")
print(f"  Change:                     {population_f_next.sum() - population_f.sum():,.0f}")
print(f"  Growth rate:                {(population_f_next.sum() / population_f.sum() - 1) * 100:.2f}%")

# Check births
births_projected = population_f_next[0]
print(f"\nBirths projected (age 0):   {births_projected:,.0f}")

# ### 2.7 Create multi-year projections

# ---- code cell ----
print("\n" + "="*60)
print("Section 2.7: Multi-Year Population Projections")
print("="*60)

# Project over multiple years
projection_years = 50
years_list = [year + i for i in range(projection_years + 1)]

# Store projections
projections_f = np.zeros((projection_years + 1, n_ages))
projections_f[0, :] = population_f  # Initial year

# Project forward
for t in range(projection_years):
    projections_f[t + 1, :] = leslie_matrix_F @ projections_f[t, :]

# Calculate total population by year
total_pop_f = projections_f.sum(axis=1)

print(f"\n✓ Projected {projection_years} years forward:")
print(f"  Start year {year}:        {total_pop_f[0]:,.0f}")
print(f"  End year {year+projection_years}:          {total_pop_f[-1]:,.0f}")
print(f"  Total change:             {total_pop_f[-1] - total_pop_f[0]:,.0f}")
print(f"  Average annual growth:    {((total_pop_f[-1]/total_pop_f[0])**(1/projection_years) - 1)*100:.2f}%")

# Visualize projection
plt.figure(figsize=(12, 6))
plt.plot(years_list, total_pop_f / 1000, linewidth=2.5, color='darkblue', marker='o', markersize=3)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Female Population (thousands)', fontsize=12)
plt.title(f'Female Population Projection - {dpto}\n'
          f'Leslie Matrix Cohort-Component Method',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✓ Projection visualization complete")

# ## Section 3: Repeat for Males

# ---- code cell ----
print("\n" + "="*60)
print("Section 3: Building Leslie Matrix for MALES")
print("="*60)

# Get MALE data (SEXO = 1)
df_male = df_dpto[df_dpto['SEXO'] == 1.0].copy()

df_pop_m = df_male[(df_male['VARIABLE'] == 'poblacion_total') & 
                   (df_male['FUENTE'] == f'censo_{year}')].copy()

df_deaths_m = df_male[(df_male['VARIABLE'] == 'defunciones') & 
                      (df_male['FUENTE'] == f'censo_{year}')].copy()

# Unabridge male data
df_deaths_m_fixed = _collapse_defunciones_01_24_to_04(df_deaths_m)

df_pop_m_1yr = unabridge_df(
    df_pop_m,
    series_keys=['DPTO_NOMBRE', 'ANO', 'SEXO', 'VARIABLE', 'FUENTE'],
    value_col='VALOR',
    ridge=1e-6
)

df_deaths_m_1yr = unabridge_df(
    df_deaths_m_fixed,
    series_keys=['DPTO_NOMBRE', 'ANO', 'SEXO', 'VARIABLE', 'FUENTE'],
    value_col='VALOR',
    ridge=1e-6
)

# Build male life table
pop_m_by_age = df_pop_m_1yr.groupby('EDAD')['VALOR'].sum()
deaths_m_by_age = df_deaths_m_1yr.groupby('EDAD')['VALOR'].sum()

# Filter out NaN ages and align to expected age range
pop_m_by_age = pop_m_by_age[pop_m_by_age.index.notna()]
deaths_m_by_age = deaths_m_by_age[deaths_m_by_age.index.notna()]

# Reindex to ensure we have all ages
pop_m_by_age = pop_m_by_age.reindex(ages_1yr, fill_value=0.0)
deaths_m_by_age = deaths_m_by_age.reindex(ages_1yr, fill_value=0.0)

lt_male = make_lifetable(
    ages=pop_m_by_age.index,  # Pass the age index
    population=pop_m_by_age.values,
    deaths=deaths_m_by_age.values,
    radix=100000,
    use_ma=True,
    ma_window=5
)

print(f"\n✓ Male life table calculated:")
print(f"  Male life expectancy (e_0): {lt_male.iloc[0]['ex']:.2f} years")

# Calculate male survivorship ratios
survivorship_m = np.zeros(len(ages_1yr))
lx_values_m = lt_male['lx'].values

for i in range(len(ages_1yr) - 1):
    if lx_values_m[i] > 0:
        survivorship_m[i] = lx_values_m[i + 1] / lx_values_m[i]
    else:
        survivorship_m[i] = 0.0

ex_last_m = lt_male.iloc[-1]['ex']
if ex_last_m > 0:
    survivorship_m[-1] = np.exp(-1.0 / ex_last_m)
else:
    survivorship_m[-1] = 0.0

# Create male population vector
population_m = pop_m_by_age.reindex(ages_1yr, fill_value=0.0).values

print(f"  Total male population: {population_m.sum():,.0f}")
print(f"  Infant survival (age 0→1): {survivorship_m[0]:.4f}")

# Build male Leslie matrix
# Note: Males are born from FEMALE fertility, so first row uses female ASFR
leslie_matrix_M = np.zeros((n_ages, n_ages))

# Male births from female fertility (proportion male = SRB / (1 + SRB))
prop_male = srb / (1.0 + srb)

# Survived ASFR for male births
survived_asfr_m = np.zeros(n_ages)
for i, age_str in enumerate(reproductive_ages):
    age_idx = int(age_str)
    if age_idx < n_ages - 1:
        survived_asfr_m[age_idx] = (
            asfr_values[i] * prop_male * 
            (survivorship_m[age_idx] + survivorship_m[age_idx + 1]) / 2
        )
    else:
        survived_asfr_m[age_idx] = asfr_values[i] * prop_male * survivorship_m[age_idx]

# Place in first row
leslie_matrix_M[0, :] = survived_asfr_m

# Male survivorship on sub-diagonal
for i in range(n_ages - 1):
    leslie_matrix_M[i + 1, i] = survivorship_m[i]

leslie_matrix_M[-1, -1] = survivorship_m[-1]

print(f"\n✓ Male Leslie Matrix complete")
print(f"  Proportion male births: {prop_male:.3f}")

# Project males
projections_m = np.zeros((projection_years + 1, n_ages))
projections_m[0, :] = population_m

for t in range(projection_years):
    projections_m[t + 1, :] = leslie_matrix_M @ projections_m[t, :]

total_pop_m = projections_m.sum(axis=1)

print(f"\n✓ Male projections complete:")
print(f"  Start year {year}:        {total_pop_m[0]:,.0f}")
print(f"  End year {year+projection_years}:          {total_pop_m[-1]:,.0f}")

# ## Section 4: Combine Male and Female Projections

# ---- code cell ----
print("\n" + "="*60)
print("Section 4: Combined Population Projections")
print("="*60)

# Total population
total_pop_combined = total_pop_f + total_pop_m

print(f"\nCombined population projections:")
print(f"  Year {year} - Female:     {total_pop_f[0]:,.0f}")
print(f"  Year {year} - Male:       {total_pop_m[0]:,.0f}")
print(f"  Year {year} - TOTAL:      {total_pop_combined[0]:,.0f}")
print(f"\n  Year {year+projection_years} - Female:     {total_pop_f[-1]:,.0f}")
print(f"  Year {year+projection_years} - Male:       {total_pop_m[-1]:,.0f}")
print(f"  Year {year+projection_years} - TOTAL:      {total_pop_combined[-1]:,.0f}")

# Visualize combined projections
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Population Projections - {dpto} ({year}-{year+projection_years})',
             fontsize=14, fontweight='bold')

# Plot 1: Total population
axes[0].plot(years_list, total_pop_combined / 1000, linewidth=2.5, 
             color='purple', marker='o', markersize=3, label='Total')
axes[0].plot(years_list, total_pop_f / 1000, linewidth=2, 
             color='red', alpha=0.7, label='Female')
axes[0].plot(years_list, total_pop_m / 1000, linewidth=2, 
             color='blue', alpha=0.7, label='Male')
axes[0].set_xlabel('Year', fontsize=12)
axes[0].set_ylabel('Population (thousands)', fontsize=12)
axes[0].set_title('Total Population Over Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Sex ratio over time
sex_ratio = total_pop_m / total_pop_f
axes[1].plot(years_list, sex_ratio, linewidth=2.5, color='green', marker='o', markersize=3)
axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal ratio')
axes[1].set_xlabel('Year', fontsize=12)
axes[1].set_ylabel('Sex Ratio (Males/Females)', fontsize=12)
axes[1].set_title('Sex Ratio Over Time')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ Combined projection visualizations complete")

# ## Section 5: Add Migration

# ---- code cell ----
print("\n" + "="*60)
print("Section 5: Adding Migration to Projections")
print("="*60)

# Use PyCCM's migration function to create migration frame
df_migration = create_migration_frame(df, year=year)

print(f"\n✓ Migration data loaded:")
print(f"  Total immigration: {df_migration['inmigracion_F'].sum():,.0f}")
print(f"  Total emigration:  {df_migration['emigracion_F'].sum():,.0f}")
print(f"  Net migration:     {df_migration['net_migration'].sum():,.0f}")

# Separate by sex and unabridge migration (immigration & emigration separately)
df_mig_f = df_migration[df_migration['SEXO'] == 2].copy().reset_index(drop=True)
df_mig_m = df_migration[df_migration['SEXO'] == 1].copy().reset_index(drop=True)

# Ensure EDAD column is string type
df_mig_f['EDAD'] = df_mig_f['EDAD'].astype(str)
df_mig_m['EDAD'] = df_mig_m['EDAD'].astype(str)

# Unabridge IMMIGRATION for females
df_mig_f_in = unabridge_df(
    df_mig_f,
    series_keys=['ANO', 'SEXO'],
    value_col='inmigracion_F'
)
df_mig_f_in = df_mig_f_in.groupby('EDAD')['inmigracion_F'].sum().reindex(ages_1yr, fill_value=0.0)
inmig_f = df_mig_f_in.values

# Unabridge EMIGRATION for females
df_mig_f_out = unabridge_df(
    df_mig_f,
    series_keys=['ANO', 'SEXO'],
    value_col='emigracion_F'
)
df_mig_f_out = df_mig_f_out.groupby('EDAD')['emigracion_F'].sum().reindex(ages_1yr, fill_value=0.0)
emig_f = df_mig_f_out.values

# Unabridge IMMIGRATION for males
df_mig_m_in = unabridge_df(
    df_mig_m,
    series_keys=['ANO', 'SEXO'],
    value_col='inmigracion_F'
)
df_mig_m_in = df_mig_m_in.groupby('EDAD')['inmigracion_F'].sum().reindex(ages_1yr, fill_value=0.0)
inmig_m = df_mig_m_in.values

# Unabridge EMIGRATION for males
df_mig_m_out = unabridge_df(
    df_mig_m,
    series_keys=['ANO', 'SEXO'],
    value_col='emigracion_F'
)
df_mig_m_out = df_mig_m_out.groupby('EDAD')['emigracion_F'].sum().reindex(ages_1yr, fill_value=0.0)
emig_m = df_mig_m_out.values

print(f"\n✓ Migration unabridged to single-year ages:")
print(f"  Female immigration by age: {inmig_f.sum():,.0f}")
print(f"  Female emigration by age:  {emig_f.sum():,.0f}")
print(f"  Female net migration:      {(inmig_f - emig_f).sum():,.0f}")
print(f"  Male immigration by age:   {inmig_m.sum():,.0f}")
print(f"  Male emigration by age:    {emig_m.sum():,.0f}")
print(f"  Male net migration:        {(inmig_m - emig_m).sum():,.0f}")

# Project with migration using half-before/half-after timing
# This better reflects exposure: emigrants leave mid-year, immigrants arrive mid-year
projections_f_mig = np.zeros((projection_years + 1, n_ages))
projections_m_mig = np.zeros((projection_years + 1, n_ages))

projections_f_mig[0, :] = population_f
projections_m_mig[0, :] = population_m

for t in range(projection_years):
    # Start-of-year population
    prev_f = projections_f_mig[t, :].copy()
    prev_m = projections_m_mig[t, :].copy()
    
    # Remove half of emigrants before projection (reduced exposure to mortality/fertility)
    prev_f_adj = prev_f - 0.5 * emig_f
    prev_m_adj = prev_m - 0.5 * emig_m
    
    # Ensure no negative population by age
    prev_f_adj = np.maximum(prev_f_adj, 0.0)
    prev_m_adj = np.maximum(prev_m_adj, 0.0)
    
    # Project with Leslie matrices
    proj_f = leslie_matrix_F @ prev_f_adj
    proj_m = leslie_matrix_M @ prev_m_adj
    
    # Add remaining half of emigrants plus full immigration (end-of-year inflows)
    projections_f_mig[t + 1, :] = proj_f - 0.5 * emig_f + inmig_f
    projections_m_mig[t + 1, :] = proj_m - 0.5 * emig_m + inmig_m

total_pop_f_mig = projections_f_mig.sum(axis=1)
total_pop_m_mig = projections_m_mig.sum(axis=1)
total_pop_mig = total_pop_f_mig + total_pop_m_mig

print(f"\n✓ Projections with migration complete:")
print(f"  Start year {year}:  {total_pop_mig[0]:,.0f}")
print(f"  End year {year+projection_years}:    {total_pop_mig[-1]:,.0f}")
print(f"  Difference from baseline: {total_pop_mig[-1] - total_pop_combined[-1]:,.0f}")

# Compare baseline vs migration scenario
print(f"\n✓ Migration impact over {projection_years} years:")
print(f"  Without migration: {total_pop_combined[-1]:,.0f}")
print(f"  With migration:    {total_pop_mig[-1]:,.0f}")
print(f"  Net effect:        {total_pop_mig[-1] - total_pop_combined[-1]:+,.0f} ({((total_pop_mig[-1]/total_pop_combined[-1] - 1)*100):+.2f}%)")

# ## Section 6: Time-Varying Fertility and Mortality Improvements

# ---- code cell ----
print("\n" + "="*60)
print("Section 6: Time-Varying Fertility & Mortality Improvements")
print("="*60)

# Load target TFR parameters using PyCCM function
target_file = data_path / "target_tfrs_example.csv"
target_dict, conv_years_dict = get_target_params(target_file)

target_tfr = target_dict.get(dpto, 1.5)
conv_years = conv_years_dict.get(dpto, 50)
baseline_tfr = tfr_female

print(f"\n✓ Fertility convergence parameters:")
print(f"  Baseline TFR: {baseline_tfr:.3f}")
print(f"  Target TFR:   {target_tfr:.3f}")
print(f"  Convergence:  {conv_years} years")

# Mortality improvement rate (1% annual reduction in hazard)
mort_improvement_rate = 0.01

print(f"\n✓ Mortality improvement parameters:")
print(f"  Annual hazard reduction: {mort_improvement_rate*100:.1f}%")

# Project with time-varying fertility and mortality improvements
projections_f_dynamic = np.zeros((projection_years + 1, n_ages))
projections_m_dynamic = np.zeros((projection_years + 1, n_ages))

projections_f_dynamic[0, :] = population_f
projections_m_dynamic[0, :] = population_m

# Store survivorship for updates
survivorship_f_t = survivorship_f.copy()
survivorship_m_t = survivorship_m.copy()

for t in range(projection_years):
    # 1. Calculate time-varying TFR using PyCCM's _smooth_tfr function
    tfr_at_step = _smooth_tfr(
        TFR0=baseline_tfr,
        target=target_tfr,
        years=conv_years,
        step=t,
        kind='exp'  # exponential convergence
    )
    
    # Scale ASFRs proportionally to TFR
    scaling_factor = tfr_at_step / baseline_tfr
    asfr_scaled = asfr_values * scaling_factor
    
    # 2. Apply mortality improvements (reduce hazard rates)
    for age_idx in range(n_ages):
        # For females
        if 0 < survivorship_f_t[age_idx] < 1:
            hazard_f = -np.log(survivorship_f_t[age_idx])
            improved_hazard_f = hazard_f * (1 - mort_improvement_rate)
            survivorship_f_t[age_idx] = np.exp(-improved_hazard_f)
        
        # For males
        if 0 < survivorship_m_t[age_idx] < 1:
            hazard_m = -np.log(survivorship_m_t[age_idx])
            improved_hazard_m = hazard_m * (1 - mort_improvement_rate)
            survivorship_m_t[age_idx] = np.exp(-improved_hazard_m)
    
    # 3. Rebuild Leslie matrices with updated rates
    # Female matrix
    leslie_F_t = np.zeros((n_ages, n_ages))
    survived_asfr_f = np.zeros(n_ages)
    for i, age_str in enumerate(reproductive_ages):
        age_idx = int(age_str)
        if age_idx < n_ages - 1:
            survived_asfr_f[age_idx] = (
                asfr_scaled[i] * prop_female * 
                (survivorship_f_t[age_idx] + survivorship_f_t[age_idx + 1]) / 2
            )
        else:
            survived_asfr_f[age_idx] = asfr_scaled[i] * prop_female * survivorship_f_t[age_idx]
    
    leslie_F_t[0, :] = survived_asfr_f
    for i in range(n_ages - 1):
        leslie_F_t[i + 1, i] = survivorship_f_t[i]
    leslie_F_t[-1, -1] = survivorship_f_t[-1]
    
    # Male matrix
    leslie_M_t = np.zeros((n_ages, n_ages))
    survived_asfr_m = np.zeros(n_ages)
    for i, age_str in enumerate(reproductive_ages):
        age_idx = int(age_str)
        if age_idx < n_ages - 1:
            survived_asfr_m[age_idx] = (
                asfr_scaled[i] * prop_male * 
                (survivorship_m_t[age_idx] + survivorship_m_t[age_idx + 1]) / 2
            )
        else:
            survived_asfr_m[age_idx] = asfr_scaled[i] * prop_male * survivorship_m_t[age_idx]
    
    leslie_M_t[0, :] = survived_asfr_m
    for i in range(n_ages - 1):
        leslie_M_t[i + 1, i] = survivorship_m_t[i]
    leslie_M_t[-1, -1] = survivorship_m_t[-1]
    
    # 4. Project with updated matrices using half-before/half-after migration timing
    prev_f_dyn = projections_f_dynamic[t, :].copy()
    prev_m_dyn = projections_m_dynamic[t, :].copy()
    
    # Remove half of emigrants before projection
    prev_f_dyn_adj = np.maximum(prev_f_dyn - 0.5 * emig_f, 0.0)
    prev_m_dyn_adj = np.maximum(prev_m_dyn - 0.5 * emig_m, 0.0)
    
    # Project with updated Leslie matrices
    proj_f_dyn = leslie_F_t @ prev_f_dyn_adj
    proj_m_dyn = leslie_M_t @ prev_m_dyn_adj
    
    # 5. Add remaining emigrants and immigration
    projections_f_dynamic[t + 1, :] = proj_f_dyn - 0.5 * emig_f + inmig_f
    projections_m_dynamic[t + 1, :] = proj_m_dyn - 0.5 * emig_m + inmig_m

total_pop_f_dynamic = projections_f_dynamic.sum(axis=1)
total_pop_m_dynamic = projections_m_dynamic.sum(axis=1)
total_pop_dynamic = total_pop_f_dynamic + total_pop_m_dynamic

# Calculate final TFR and life expectancy
final_tfr = _smooth_tfr(TFR0=baseline_tfr, target=target_tfr, years=conv_years, step=projection_years-1, kind='exp')
final_le_f = lt_female.iloc[0]['ex'] * (1 + mort_improvement_rate * projection_years)
final_le_m = lt_male.iloc[0]['ex'] * (1 + mort_improvement_rate * projection_years)

print(f"\n✓ Dynamic projections complete:")
print(f"  Start year {year}:  {total_pop_dynamic[0]:,.0f}")
print(f"  End year {year+projection_years}:    {total_pop_dynamic[-1]:,.0f}")
print(f"\n  TFR trajectory: {baseline_tfr:.3f} → {final_tfr:.3f}")
print(f"  Life expectancy (F): {lt_female.iloc[0]['ex']:.2f} → ~{final_le_f:.2f} years")
print(f"  Life expectancy (M): {lt_male.iloc[0]['ex']:.2f} → ~{final_le_m:.2f} years")
print(f"\n  Difference from migration-only: {total_pop_dynamic[-1] - total_pop_mig[-1]:+,.0f}")

# ## Section 7: Account for Census Omissions

# ---- code cell ----
print("\n" + "="*60)
print("Section 7: Census Omissions Adjustment")
print("="*60)

# Check if OMISION column exists and extract adjustment factors
if 'OMISION' in df.columns:
    df_omision = df[(df['ANO'] == year) & 
                    (df['DPTO_NOMBRE'] == dpto) & 
                    (df['VARIABLE'] == 'poblacion_total')].copy()
    
    if not df_omision.empty and 'OMISION' in df_omision.columns:
        # Get omission rates by sex (in percentage)
        omision_rate_f = df_omision[df_omision['SEXO'] == 2.0]['OMISION'].mean()
        omision_rate_m = df_omision[df_omision['SEXO'] == 1.0]['OMISION'].mean()
        
        # Calculate adjustment factors: adjusted = observed / (1 - omission_rate/100)
        if pd.notna(omision_rate_f) and omision_rate_f > 0:
            adjustment_factor_f = 1 / (1 - omision_rate_f / 100)
        else:
            adjustment_factor_f = 1.0
            omision_rate_f = 0.0
        
        if pd.notna(omision_rate_m) and omision_rate_m > 0:
            adjustment_factor_m = 1 / (1 - omision_rate_m / 100)
        else:
            adjustment_factor_m = 1.0
            omision_rate_m = 0.0
        
        print(f"\n✓ Census omission rates found:")
        print(f"  Female omission rate: {omision_rate_f:.2f}%")
        print(f"  Male omission rate:   {omision_rate_m:.2f}%")
        print(f"  Female adjustment factor: {adjustment_factor_f:.4f}")
        print(f"  Male adjustment factor:   {adjustment_factor_m:.4f}")
        
        # Apply adjustment to the dynamic projection (which includes migration and improvements)
        projections_f_final = projections_f_dynamic * adjustment_factor_f
        projections_m_final = projections_m_dynamic * adjustment_factor_m
        
        print(f"\n✓ Omission adjustment applied to projections")
    else:
        print("\n✗ No omission data available in dataset for this department/year")
        print("  Using unadjusted projections")
        projections_f_final = projections_f_dynamic
        projections_m_final = projections_m_dynamic
        adjustment_factor_f = 1.0
        adjustment_factor_m = 1.0
        omision_rate_f = 0.0
        omision_rate_m = 0.0
else:
    print("\n✗ OMISION column not found in dataset")
    print("  Using unadjusted projections")
    projections_f_final = projections_f_dynamic
    projections_m_final = projections_m_dynamic
    adjustment_factor_f = 1.0
    adjustment_factor_m = 1.0
    omision_rate_f = 0.0
    omision_rate_m = 0.0

total_pop_f_final = projections_f_final.sum(axis=1)
total_pop_m_final = projections_m_final.sum(axis=1)
total_pop_final = total_pop_f_final + total_pop_m_final

print(f"\n✓ Final adjusted projections:")
print(f"  Start year {year}:  {total_pop_final[0]:,.0f}")
print(f"  End year {year+projection_years}:    {total_pop_final[-1]:,.0f}")
if adjustment_factor_f > 1.0 or adjustment_factor_m > 1.0:
    print(f"  Difference from unadjusted: {total_pop_final[-1] - total_pop_dynamic[-1]:+,.0f}")

# ## Section 8: Comparison of All Scenarios

# ---- code cell ----
print("\n" + "="*60)
print("Section 8: Comparison of Projection Scenarios")
print("="*60)

# Create comprehensive comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Population Projection Scenarios - {dpto} ({year}-{year+projection_years})',
             fontsize=16, fontweight='bold')

# Plot 1: All scenarios comparison
ax1 = axes[0, 0]
ax1.plot(years_list, total_pop_combined / 1000, linewidth=2.5, 
         label='Baseline (constant)', color='gray', alpha=0.7)
ax1.plot(years_list, total_pop_mig / 1000, linewidth=2.5, 
         label='+ Migration', color='blue')
ax1.plot(years_list, total_pop_dynamic / 1000, linewidth=2.5, 
         label='+ TFR/Mortality changes', color='green')
ax1.plot(years_list, total_pop_final / 1000, linewidth=3, 
         label='Final (omission-adjusted)', color='red', linestyle='--')
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Population (thousands)', fontsize=11)
ax1.set_title('Total Population by Scenario')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Female vs Male (final scenario)
ax2 = axes[0, 1]
ax2.plot(years_list, total_pop_f_final / 1000, linewidth=2.5, label='Female', color='red')
ax2.plot(years_list, total_pop_m_final / 1000, linewidth=2.5, label='Male', color='blue')
ax2.plot(years_list, total_pop_final / 1000, linewidth=2.5, label='Total', color='purple', alpha=0.7)
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('Population (thousands)', fontsize=11)
ax2.set_title('Final Projection by Sex')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Growth rates comparison
ax3 = axes[1, 0]
growth_baseline = np.diff(total_pop_combined) / total_pop_combined[:-1] * 100
growth_final = np.diff(total_pop_final) / total_pop_final[:-1] * 100
ax3.plot(years_list[1:], growth_baseline, linewidth=2, label='Baseline', color='gray', alpha=0.7)
ax3.plot(years_list[1:], growth_final, linewidth=2, label='Final', color='red')
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax3.set_xlabel('Year', fontsize=11)
ax3.set_ylabel('Annual Growth Rate (%)', fontsize=11)
ax3.set_title('Population Growth Rate Over Time')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Cumulative effects
ax4 = axes[1, 1]
diff_mig = (total_pop_mig - total_pop_combined) / 1000
diff_dynamic = (total_pop_dynamic - total_pop_mig) / 1000
diff_omis = (total_pop_final - total_pop_dynamic) / 1000

ax4.fill_between(years_list, 0, diff_mig, alpha=0.4, label='Migration effect', color='blue')
ax4.fill_between(years_list, diff_mig, diff_mig + diff_dynamic, alpha=0.4, 
                 label='TFR/Mortality effect', color='green')
ax4.fill_between(years_list, diff_mig + diff_dynamic, diff_mig + diff_dynamic + diff_omis, 
                 alpha=0.4, label='Omission effect', color='purple')
ax4.set_xlabel('Year', fontsize=11)
ax4.set_ylabel('Population Difference (thousands)', fontsize=11)
ax4.set_title('Cumulative Effects vs Baseline')
ax4.legend(fontsize=9, loc='best')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ Comparison visualizations complete")

# Print summary statistics
print("\n" + "="*60)
print("PROJECTION SUMMARY TABLE")
print("="*60)

print(f"\nScenario Comparison (Year {year+projection_years}):")
print(f"{'Scenario':<35} {'Population':>15} {'Difference':>15}")
print("-" * 66)
print(f"{'1. Baseline (constant rates)':<35} {total_pop_combined[-1]:>15,.0f} {'-':>15}")
print(f"{'2. + Migration':<35} {total_pop_mig[-1]:>15,.0f} {total_pop_mig[-1] - total_pop_combined[-1]:>+15,.0f}")
print(f"{'3. + TFR/Mortality changes':<35} {total_pop_dynamic[-1]:>15,.0f} {total_pop_dynamic[-1] - total_pop_mig[-1]:>+15,.0f}")
print(f"{'4. + Omission adjustment (FINAL)':<35} {total_pop_final[-1]:>15,.0f} {total_pop_final[-1] - total_pop_dynamic[-1]:>+15,.0f}")

print(f"\nFinal Projection Details (Year {year+projection_years}):")
print(f"  Female population: {total_pop_f_final[-1]:>12,.0f}")
print(f"  Male population:   {total_pop_m_final[-1]:>12,.0f}")
print(f"  Total population:  {total_pop_final[-1]:>12,.0f}")
print(f"\n  Total change from {year}: {total_pop_final[-1] - total_pop_final[0]:>+10,.0f}  ({((total_pop_final[-1]/total_pop_final[0] - 1)*100):>+6.2f}%)")
print(f"  Avg annual growth rate: {((total_pop_final[-1]/total_pop_final[0])**(1/projection_years) - 1)*100:>6.2f}%")

# ## FINAL SUMMARY

print("\n" + "="*60)
print("LAB 4 COMPLETE! 🎉")
print("="*60)

print(f"\nWhat you've learned:")
print(f"  ✓ Unabridging age data from 5-year to 1-year groups using PyCCM")
print(f"  ✓ Building life tables and calculating survivorship ratios")
print(f"  ✓ Computing age-specific fertility rates (ASFR)")
print(f"  ✓ Constructing Leslie matrices for females and males")
print(f"  ✓ Projecting populations forward using cohort-component method")
print(f"  ✓ Combining male and female projections")
print(f"  ✓ Adding migration flows using create_migration_frame()")
print(f"  ✓ Implementing time-varying fertility with _smooth_tfr()")
print(f"  ✓ Applying mortality improvements over time")
print(f"  ✓ Adjusting for census omissions")
print(f"  ✓ Comparing multiple projection scenarios")
print(f"  ✓ Visualizing population trajectories and demographic dynamics")

print(f"\nKey Results for {dpto}:")
print(f"  Female life expectancy:   {lt_female.iloc[0]['ex']:.2f} years → ~{final_le_f:.2f} years")
print(f"  Male life expectancy:     {lt_male.iloc[0]['ex']:.2f} years → ~{final_le_m:.2f} years")
print(f"  Total Fertility Rate:     {tfr_female:.3f} → {final_tfr:.3f} (target: {target_tfr:.3f})")
print(f"  Sex Ratio at Birth:       {srb:.3f}")
print(f"  Omission adjustment (F):  {adjustment_factor_f:.4f}  ({omision_rate_f:.2f}%)")
print(f"  Omission adjustment (M):  {adjustment_factor_m:.4f}  ({omision_rate_m:.2f}%)")
print(f"  Immigration (annual):     {(inmig_f.sum() + inmig_m.sum()):+,.0f}")
print(f"  Emigration (annual):      {(emig_f.sum() + emig_m.sum()):+,.0f}")
print(f"  Net migration (annual):   {(inmig_f.sum() - emig_f.sum() + inmig_m.sum() - emig_m.sum()):+,.0f}")
print(f"\n  Starting population ({year}):      {total_pop_final[0]:,.0f}")
print(f"  Final projection ({year+projection_years}):       {total_pop_final[-1]:,.0f}")
print(f"  Total change:                      {total_pop_final[-1] - total_pop_final[0]:+,.0f}")
print(f"  Average annual growth:             {((total_pop_final[-1]/total_pop_final[0])**(1/projection_years) - 1)*100:+.2f}%")

print("\n🎓 Congratulations! You've completed a comprehensive demographic projection")
print("   using all major components of the PyCCM library!")
print("\n💡 Key insight: The final projection differs from baseline by:")
print(f"   {total_pop_final[-1] - total_pop_combined[-1]:+,.0f} people ({((total_pop_final[-1]/total_pop_combined[-1] - 1)*100):+.2f}%)")
print("   showing the importance of including migration, fertility/mortality trends,")
print("   and census omission adjustments in demographic projections.")
