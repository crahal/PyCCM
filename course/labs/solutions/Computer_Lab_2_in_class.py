#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Computer Lab 2: Building lifetables in Python
# In-Class Version - Streamlined for teaching

# # Demographic Research Methods and the PyCCM Library
# ## Computer Lab 2: Building lifetables in Python
# ## Instructor: Jiani Yan
# ## Date: October 28th, 2025
# ---

# #### 1.1.1 Load in the conteos.rds file, using the data_loaders module or otherwise.

# ---- code cell ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os

os.chdir('/Users/valler/Python/PyCCM/course/labs/solutions')

# Add src directory to path
src_path = Path.cwd().parents[2]/'src'
sys.path.insert(0, str(src_path))

# Import PyCCM functions
from data_loaders import load_all_data
from helpers import _collapse_defunciones_01_24_to_04
from mortality import make_lifetable
from abridger import unabridge_df

# Path to the data file
data_path = Path.cwd().parents[2]/'data'

# Load the data using PyCCM data_loaders
data_dict = load_all_data(data_path)
df = pd.DataFrame(data_dict['conteos'])

df.head(10)

# #### 1.1.2 Filter for censos_2018 and the year 2018 for exposures and deaths

# ---- code cell ----
year = 2018
df_2018 = df[df['ANO'] == year].copy()

# #### 1.1.3 Filter for your favourite DPTO

# ---- code cell ----
favorite_dpto = 'BOLIVAR'
df_dpto = df_2018[df_2018['DPTO_NOMBRE'] == favorite_dpto].copy()
df_dpto.head()

# ### 1.1.4 Optional: Blend EEVV and censo_2018 data sources (midpoint approach)

# ---- code cell ----
print("\n" + "="*60)
print("OPTIONAL: Data Source Blending")
print("="*60)
print("\nThe PyCCM library supports blending two death data sources:")
print("  - EEVV: Vital statistics (may have underreporting)")
print("  - censo_2018: Census-based (may have coverage issues)")
print("  - midpoint: Weighted blend of both sources")

# Example: Compare the two sources for defunciones
df_eevv = df[(df['ANO'] == 2018) & (df['DPTO_NOMBRE'] == favorite_dpto) & 
             (df['FUENTE'] == 'EEVV') & (df['VARIABLE'] == 'defunciones')].copy()
df_censo = df[(df['ANO'] == 2018) & (df['DPTO_NOMBRE'] == favorite_dpto) & 
              (df['FUENTE'] == 'censo_2018') & (df['VARIABLE'] == 'defunciones')].copy()

if not df_eevv.empty and not df_censo.empty:
    total_eevv = df_eevv['VALOR'].sum()
    total_censo = df_censo['VALOR'].sum()
    print(f"\nTotal deaths in {favorite_dpto} (2018):")
    print(f"  EEVV:       {total_eevv:,.0f}")
    print(f"  Censo_2018: {total_censo:,.0f}")
    print(f"  Difference: {abs(total_eevv - total_censo):,.0f} ({abs(total_eevv - total_censo)/total_censo*100:.1f}%)")
    
    # You could blend them with a weight (e.g., 0.5 = equal blend)
    # This is what main_compute.py does for death_choice='midpoint'
    weight_eevv = 0.5
    print(f"\nWith 50/50 blend: {weight_eevv * total_eevv + (1 - weight_eevv) * total_censo:,.0f}")
    print("\nNote: For this lab, we'll use censo_2018 (default for teaching)")
else:
    print("\nNote: Both sources not available for comparison")

print("\n✓ Continuing with censo_2018 data source...")

# ### 1.2 Create a new dataframe which has three columns: age, exposures, deaths

# ---- code cell ----
# Fix overlapping infant age groups using PyCCM helper
df_censo = df_dpto.loc[df_dpto['FUENTE'] == 'censo_2018'].copy()
df_deaths = df_censo[df_censo['VARIABLE'] == 'defunciones'].copy()

# Get the sum of deaths for 0-1 and 2-4 age groups (across both sexes)
deaths_0_1 = df_deaths[df_deaths['EDAD'] == '0-1']['VALOR'].sum()
deaths_2_4 = df_deaths[df_deaths['EDAD'] == '2-4']['VALOR'].sum()
total_deaths_0_4 = deaths_0_1 + deaths_2_4

# Get a template row to use for the new 0-4 entry
template_row = df_censo[df_censo['VARIABLE'] == 'defunciones'].iloc[0].copy()

# Create the new 0-4 row by appending directly
new_index = df_censo.index.max() + 1
df_censo.loc[new_index] = template_row
df_censo.loc[new_index, 'EDAD'] = '0-4'
df_censo.loc[new_index, 'VARIABLE'] = 'defunciones'
df_censo.loc[new_index, 'VALOR'] = total_deaths_0_4
df_censo.loc[new_index, 'SEXO'] = 3.0  # Combined sexes

# Remove the original 0-1 and 2-4 rows since we now have 0-4
df_censo = df_censo.loc[~df_censo['EDAD'].isin(['0-1', '2-4'])]

# Sum sexes first, then pivot (simpler!)
df_summed = df_censo.groupby(['EDAD', 'VARIABLE'])['VALOR'].sum().reset_index()


df_pivot = df_summed.pivot(
    index='EDAD',
    columns='VARIABLE',
    values='VALOR'
).reset_index()

# melt function to reverse the pivot
# df_melted_pivot = pd.melt(df_pivot, id_vars=['EDAD'], value_vars=['defunciones', 'poblacion_total'], var_name='VARIABLE', value_name='VALOR')

# Create dataframe with age, exposures, deaths
lt_data = pd.DataFrame({
    'age': df_pivot['EDAD'],
    'exposures': df_pivot['poblacion_total'].values,
    'deaths': df_pivot['defunciones'].values
})

# Order ages properly
age_order = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79",'80-84', '85-89', '90+']
#age_order = [str(x) for x in range(0,90)]+["90+"]
lt_data['age'] = pd.Categorical(lt_data['age'], categories=age_order, ordered=True)
lt_data = lt_data.sort_values('age').reset_index(drop=True)

print("✓ Created age, exposures, deaths dataframe")

# ### 1.3 Drop any deaths which are not attributable to ages.

# ---- code cell ----
lt_data = lt_data.dropna(subset=['age']).copy()
lt_data = lt_data[lt_data['exposures'] > 0].copy()
lt_data = lt_data.reset_index(drop=True)

print(f" Cleaned data: {len(lt_data)} age groups")

# ### 1.4 Create a basic m_x column

# ---- code cell ----
lt_data['m_x'] = lt_data['deaths'] / lt_data['exposures']

print(f"✓ Calculated m_x (central death rate)")

# ### 1.5 Convert this to q_x using the Greville equation (use a reasonable a_x)

# ---- code cell ----
# Apply Coale-Demeny formula for infant a_x
infant_mx = lt_data.iloc[0]['m_x']
if infant_mx < 0.01724:
    infant_ax = 0.14903 - 2.05527 * infant_mx
elif infant_mx < 0.06891:
    infant_ax = 0.04667 + 3.88089 * infant_mx
else:
    infant_ax = 0.31411


# Set age interval widths and a_x values
if len(lt_data)==19:
    lt_data['n'] = 5
    lt_data['a_x'] = 2.5
else:
    lt_data['n'] = 1
    lt_data['a_x'] = 0.5
lt_data.loc[0, 'a_x'] = infant_ax

# Convert m_x to q_x using Greville formula
lt_data['q_x'] = (lt_data['n'] * lt_data['m_x']) / (1 + (lt_data['n'] - lt_data['a_x']) * lt_data['m_x'])

# Last age group has q_x = 1.0 (open-ended interval)
lt_data.loc[lt_data.index[-1], 'q_x'] = 1.0
lt_data['q_x'] = lt_data['q_x'].clip(0, 1)

print(f"✓ Calculated q_x (probability of death)")
print(f"  Note: Last age group ({lt_data.iloc[-1]['age']}) has q_x = 1.0 (open interval)")

# ### 1.6 Convert this to q_x using the Greville equation (use a reasonable a_x)

# ---- code cell ----
# (Duplicate question - already completed above)

# ### 1.6 Set a radix, and calculate a l_x column.

# ---- code cell ----
radix = 100000
lt_data['l_x'] = 0.0
lt_data.loc[0, 'l_x'] = radix

for i in range(len(lt_data) - 1):
    lt_data.loc[i + 1, 'l_x'] = lt_data.loc[i, 'l_x'] * (1 - lt_data.loc[i, 'q_x'])

print(f"✓ Calculated l_x (survivorship) with radix={radix:,}")

# ### 1.7 Calculate a d_x from this.

# ---- code cell ----
lt_data['d_x'] = lt_data['l_x'] * lt_data['q_x']
print(f"✓ Calculated d_x (deaths in interval)")

# ### 1.8 Build a L_x column

# ---- code cell ----
lt_data['L_x'] = 0.0

# All ages except the last - IMPORTANT: Must multiply by interval width n!
for i in range(len(lt_data) - 1):
    n_i = lt_data.loc[i, 'n']
    lt_data.loc[i, 'L_x'] = n_i * (lt_data.loc[i, 'l_x'] + lt_data.loc[i + 1, 'l_x']) / 2

# Last age group: L_x = l_x / m_x (all remaining person-years)
last_idx = len(lt_data) - 1
if lt_data.loc[last_idx, 'm_x'] > 0:
    lt_data.loc[last_idx, 'L_x'] = lt_data.loc[last_idx, 'l_x'] / lt_data.loc[last_idx, 'm_x']
else:
    lt_data.loc[last_idx, 'L_x'] = lt_data.loc[last_idx, 'l_x'] * 10

print(f"✓ Calculated L_x (person-years lived)")
print(f"  Note: Open interval uses L_x = l_x / m_x (causes spike in plots)")

# ### 1.9 From the L_x column, calculate the T_x

# ---- code cell ----
lt_data['T_x'] = 0.0

# Start from the last age group and work backwards
last_idx = len(lt_data) - 1
lt_data.loc[last_idx, 'T_x'] = lt_data.loc[last_idx, 'L_x']

for i in range(len(lt_data) - 2, -1, -1):
    lt_data.loc[i, 'T_x'] = lt_data.loc[i, 'L_x'] + lt_data.loc[i + 1, 'T_x']

print(f"✓ Calculated T_x (total person-years remaining)")

# ### 1.10 Finally, build a life expectancy column.

# ---- code cell ----
lt_data['e_x'] = lt_data['T_x'] / lt_data['l_x']

print("\n" + "="*60)
print("MANUAL LIFE TABLE COMPLETE")
print("="*60)
print(f" Life expectancy at birth (e_0): {lt_data.iloc[0]['e_x']:.2f} years")
print(f"Infant mortality rate: {lt_data.iloc[0]['m_x']*lt_data.iloc[0]['n']*1000:.2f} per 1,000")
print(f" Infant death probability (q_x): {lt_data.iloc[0]['q_x']*100:.2f}%")

# ### 2. Visualise these columns (and their logged values). What can you see?

# ---- code cell ----
last_age_group = lt_data.iloc[-1]['age']

# Plot 1: m_x (log scale)
plt.figure(figsize=(10, 6))
plt.plot(lt_data['age'], lt_data['m_x'], 'o-', color='red', linewidth=2)
plt.xlabel('Age')
plt.ylabel('m_x (death rate)')
plt.title(f'Central Death Rate (log scale) - {favorite_dpto}, {year}')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: q_x
plt.figure(figsize=(10, 6))
plt.plot(lt_data['age'], lt_data['q_x'], 'o-', color='orange', linewidth=2)
plt.xlabel('Age')
plt.ylabel('q_x (probability)')
plt.title(f'Probability of Death - {favorite_dpto}, {year}')
plt.grid(True, alpha=0.3)
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
plt.text(0.98, 0.95, f'q_x = 1.0 for open interval ({last_age_group})', 
         transform=plt.gca().transAxes, ha='right', va='top',
         fontsize=9, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 3: l_x (survivorship)
plt.figure(figsize=(10, 6))
plt.plot(lt_data['age'], lt_data['l_x'], 'o-', color='blue', linewidth=2)
plt.xlabel('Age')
plt.ylabel('l_x (survivors)')
plt.title(f'Survivorship Curve (radix={radix:,}) - {favorite_dpto}, {year}')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 4: d_x
plt.figure(figsize=(10, 6))
plt.plot(lt_data['age'], lt_data['d_x'], 'o-', color='purple', linewidth=2)
plt.xlabel('Age')
plt.ylabel('d_x (deaths)')
plt.title(f'Deaths by Age - {favorite_dpto}, {year}')
plt.grid(True, alpha=0.3)
modal_age_idx = lt_data['d_x'].idxmax()
modal_age = lt_data.loc[modal_age_idx, 'age']
plt.axvline(x=modal_age, color='red', linestyle='--', label=f'Modal age: {modal_age}')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 5: L_x (log scale)
plt.figure(figsize=(10, 6))
plt.plot(lt_data['age'], lt_data['L_x'], 'o-', color='green', linewidth=2)
plt.xlabel('Age')
plt.ylabel('L_x (person-years)')
plt.title(f'Person-Years Lived (log scale) - {favorite_dpto}, {year}')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 6: e_x
plt.figure(figsize=(10, 6))
plt.plot(lt_data['age'], lt_data['e_x'], 'o-', color='brown', linewidth=2)
plt.xlabel('Age')
plt.ylabel('e_x (years)')
plt.title(f'Life Expectancy - {favorite_dpto}, {year}')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("✓ Visualizations complete")

# ### 3. Can you re-run this by constructing a midpoint between the two death counts?

# ---- code cell ----
print("\n" + "="*60)
print("Question 3: Comparison with PyCCM make_lifetable()")
print("="*60)



# ---- code cell ----
# Fix overlapping infant age groups using PyCCM helper
df_censo = df_dpto.loc[df_dpto['FUENTE'] == 'censo_2018']
df_eevv = df_dpto.loc[df_dpto['FUENTE'] == 'EEVV'] # only have deaths and births defunciones

df_deaths = df_dpto[df_dpto['VARIABLE'] == 'defunciones'].copy()
df_deaths = df_deaths.groupby(['EDAD','SEXO']).mean(numeric_only=True).reset_index()
df_deaths['DPTO_NOMBRE'] = 'BOLIVAR'
df_deaths['VARIABLE'] = 'defunciones'
df_deaths['FUENTE'] = 'combined'

df_deaths_fixed = _collapse_defunciones_01_24_to_04(df_deaths)  # deaths has 0-1, 2-4
df_pop = df_dpto[df_dpto['VARIABLE'] == 'poblacion_total'].copy()
df_combined = pd.concat([df_deaths_fixed, df_pop], ignore_index=True)

# Unabridge deaths data (before aggregation to preserve metadata)
# df_combined_unabridged = unabridge_df(
#    df_combined,
#    series_keys=['DPTO_NOMBRE', 'ANO', 'SEXO', 'VARIABLE', 'FUENTE'],
#    value_col='VALOR',
#    ridge=1e-6
#)

# Sum sexes first, then pivot (simpler!)
df_summed = df_combined.groupby(['EDAD', 'VARIABLE'])['VALOR'].sum().reset_index()
df_pivot = df_summed.pivot(
    index='EDAD',
    columns='VARIABLE',
    values='VALOR'
).reset_index()

# Create dataframe with age, exposures, deaths
lt_data_new = pd.DataFrame({
    'age': df_pivot['EDAD'],
    'exposures': df_pivot['poblacion_total'].values,
    'deaths': df_pivot['defunciones'].values
})

age_order = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79",'80-84', '85-89', '90+']
#age_order = [str(x) for x in range(0,90)]+["90+"]

lt_data_new['age'] = pd.Categorical(lt_data_new['age'], categories=age_order, ordered=True)
lt_data_new = lt_data_new.sort_values('age').reset_index(drop=True)

lt_pyccm = make_lifetable(
    ages=lt_data_new['age'],
    population=lt_data_new['exposures'],
    deaths=lt_data_new['deaths'],
    radix=100000,
    use_ma=False
)

print(f"\nWithout combining calculation e_0: {lt_data.iloc[0]['e_x']:.2f} years")
print(f"PyCCM make_lifetable e_0: {lt_pyccm.iloc[0]['ex']:.2f} years")
print(f"Difference: {abs(lt_data.iloc[0]['e_x'] - lt_pyccm.iloc[0]['ex']):.2f} years")

# ### 4. How does PyCCM use moving averages to smooth lifetable values? How can you incorporate this into your code?

# ---- code cell ----
print("\n" + "="*60)
print("Question 4: Moving Average Smoothing")
print("="*60)

lt_no_smooth = make_lifetable(
    ages=lt_data['age'],
    population=lt_data['exposures'],
    deaths=lt_data['deaths'],
    radix=100000,
    use_ma=False
)

lt_smooth = make_lifetable(
    ages=lt_data['age'],
    population=lt_data['exposures'],
    deaths=lt_data['deaths'],
    radix=100000,
    use_ma=True,
    ma_window=5
)

print(f"\nWithout smoothing: {lt_no_smooth.iloc[0]['ex']:.2f} years")
print(f"With MA smoothing (window=5): {lt_smooth.iloc[0]['ex']:.2f} years")
print(f"Difference: {abs(lt_smooth.iloc[0]['ex'] - lt_no_smooth.iloc[0]['ex']):.2f} years")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Effect of Moving Average Smoothing', fontsize=14, fontweight='bold')

lt_no_smooth_plot = lt_no_smooth.reset_index()
lt_smooth_plot = lt_smooth.reset_index()

# Death rates
axes[0].plot(lt_no_smooth_plot['age'], lt_no_smooth_plot['mx'], 'o--', 
             color='red', alpha=0.5, label='Unsmoothed')
axes[0].plot(lt_smooth_plot['age'], lt_smooth_plot['mx'], 'o-', 
             color='blue', linewidth=2, label='MA smoothed (window=5)')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('m_x (death rate)')
axes[0].set_title('Death Rates: Smoothed vs Unsmoothed')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Life expectancy
axes[1].plot(lt_no_smooth_plot['age'], lt_no_smooth_plot['ex'], 'o--', 
             color='red', alpha=0.5, label='Unsmoothed')
axes[1].plot(lt_smooth_plot['age'], lt_smooth_plot['ex'], 'o-', 
             color='blue', linewidth=2, label='MA smoothed (window=5)')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('e_x (years)')
axes[1].set_title('Life Expectancy: Smoothed vs Unsmoothed')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Smoothing reduces noise from small sample sizes")

# ### 5. How does PyCCM use mortality improvements? How can you incorporate this into your code?

# ---- code cell ----
print("\n" + "="*60)
print("Question 5: Mortality Improvements Over Time")
print("="*60)

# Simplified approach - define helper function first
def quick_life_table(df_input, dpto, year, data_source=None):
    """Quick helper to build life table for a department and year."""
    
    # Auto-detect censo source based on year if not specified
    if data_source is None:
        data_source = f'censo_{year}'
    
    df_filtered = df_input[(df_input['ANO'] == year) & 
                           (df_input['DPTO_NOMBRE'] == dpto) &
                           (df_input['FUENTE'] == data_source)].copy()
    
    # Get deaths and population
    df_deaths = df_filtered[df_filtered['VARIABLE'] == 'defunciones'].copy()
    df_deaths_fixed = _collapse_defunciones_01_24_to_04(df_deaths)
    df_pop = df_filtered[df_filtered['VARIABLE'] == 'poblacion_total'].copy()
    df_combined = pd.concat([df_deaths_fixed, df_pop], ignore_index=True)
    
    # Sum sexes first, then pivot (simpler!)
    df_summed = df_combined.groupby(['EDAD', 'VARIABLE'])['VALOR'].sum().reset_index()
    df_pivot = df_summed.pivot(
        index='EDAD', columns='VARIABLE', values='VALOR'
    ).reset_index()
    
    # Build life table using PyCCM
    return make_lifetable(
        ages=df_pivot['EDAD'], population=df_pivot['poblacion_total'],
        deaths=df_pivot['defunciones'], radix=100000, use_ma=True, ma_window=5
    )

# Build life tables for each year - much simpler!
years_to_compare = [2005, 2018]
life_expectancies = []

for yr in years_to_compare:
    lt_yr = quick_life_table(df, favorite_dpto, yr)
    life_expectancies.append(lt_yr.iloc[0]['ex'])

# Plot improvements
plt.figure(figsize=(10, 6))
plt.plot(years_to_compare, life_expectancies, 'o-', linewidth=2, markersize=10, color='green')
plt.xlabel('Year')
plt.ylabel('Life Expectancy at Birth (e_0)')
plt.title(f'Mortality Trends in {favorite_dpto}, Colombia')
plt.grid(True, alpha=0.3)

z = np.polyfit(years_to_compare, life_expectancies, 1)
p = np.poly1d(z)
trend_color = 'red' if z[0] < 0 else 'blue'
trend_direction = 'deterioration' if z[0] < 0 else 'improvement'
plt.plot(years_to_compare, p(years_to_compare), "--", color=trend_color, alpha=0.7, 
         label=f'Trend: {z[0]:.3f} years/year ({trend_direction})')
plt.legend()
plt.show()

change = life_expectancies[-1] - life_expectancies[0]
if change > 0:
    print(f"\n✓ Mortality IMPROVEMENT: {change:.2f} years (2005-2018)")
else:
    print(f"\n⚠ Mortality DETERIORATION: {abs(change):.2f} years (2005-2018)")
    print(f"   (Likely due to data quality differences between census years)")

# ### 6. Can you modularise your work into a function? This will be useful for the final lab!

# ---- code cell ----
print("\n" + "="*60)
print("Question 6: Modularized Function")
print("="*60)

def build_life_table_for_dpto(df_input, dpto_name, year_val, use_smoothing=True):
    """
    Build a life table for a specific department and year using PyCCM functions.
    
    Parameters:
    -----------
    df_input : pd.DataFrame
        The full conteos dataframe
    dpto_name : str
        Department name (e.g., 'BOLIVAR')
    year_val : int
        Year to analyze
    use_smoothing : bool
        Whether to apply moving average smoothing
    
    Returns:
    --------
    pd.DataFrame
        Complete life table with all demographic columns
    """
    df_filtered = df_input[(df_input['ANO'] == year_val) & 
                           (df_input['DPTO_NOMBRE'] == dpto_name)].copy()
    
    df_deaths = df_filtered[df_filtered['VARIABLE'] == 'defunciones'].copy()
    df_deaths_fixed = _collapse_defunciones_01_24_to_04(df_deaths)
    df_pop = df_filtered[df_filtered['VARIABLE'] == 'poblacion_total'].copy()
    df_combined = pd.concat([df_deaths_fixed, df_pop], ignore_index=True)
    
    # Sum sexes first, then pivot (simpler!)
    df_summed = df_combined.groupby(['EDAD', 'VARIABLE'])['VALOR'].sum().reset_index()
    df_pivot = df_summed.pivot(
        index='EDAD', columns='VARIABLE', values='VALOR'
    ).reset_index()
    
    lt = make_lifetable(
        ages=df_pivot['EDAD'],
        population=df_pivot['poblacion_total'],
        deaths=df_pivot['defunciones'],
        radix=100000,
        use_ma=use_smoothing,
        ma_window=5 if use_smoothing else None
    )
    
    return lt
lt_test = build_life_table_for_dpto(df, 'BOLIVAR', 2018, use_smoothing=True)
print(f"\n✓ Function created and tested: e_0 = {lt_test.iloc[0]['ex']:.2f} years")

# ### 7. Loop over multiple DPTOs. Store your m_x and e_0 values. Plot them in a 1x2 figure.

# ---- code cell ----
print("\n" + "="*60)
print("Question 7: Multi-Department Comparison")
print("="*60)

dptos_to_compare = ['BOLIVAR', 'ANTIOQUIA', 'VALLE', 'CUNDINAMARCA', 'ATLANTICO', 
                    'SANTANDER', 'TOLIMA', 'BOYACA', 'CALDAS', 'NORTE_SANTANDER']

results = []

for dpto in dptos_to_compare:
    try:
        lt = build_life_table_for_dpto(df, dpto, 2018, use_smoothing=True)
        infant_n = lt.iloc[0]['n']
        infant_mx_5yr = lt.iloc[0]['mx'] * infant_n
        results.append({
            'dpto': dpto,
            'e_0': lt.iloc[0]['ex'],
            'infant_mx': infant_mx_5yr
        })
    except Exception as e:
        pass

results_df = pd.DataFrame(results)

# Create 1x2 comparison figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Mortality Comparison Across Colombian Departments ({year})', 
             fontsize=14, fontweight='bold')

# Infant mortality
axes[0].barh(results_df['dpto'], results_df['infant_mx']*1000, color='red', alpha=0.7)
axes[0].set_xlabel('Infant Mortality Rate (per 1,000, 5-year period)')
axes[0].set_ylabel('Department')
axes[0].set_title('Infant Mortality Rate (age 0-4, 5-year period)')
axes[0].grid(True, alpha=0.3, axis='x')
axes[0].invert_yaxis()

fav_idx = results_df[results_df['dpto'] == favorite_dpto].index
if len(fav_idx) > 0:
    axes[0].barh(results_df.loc[fav_idx[0], 'dpto'], 
                results_df.loc[fav_idx[0], 'infant_mx']*1000, 
                color='darkred', alpha=1.0, label='Your department')
    axes[0].legend()

# Life expectancy
axes[1].barh(results_df['dpto'], results_df['e_0'], color='blue', alpha=0.7)
axes[1].set_xlabel('Life Expectancy at Birth (years)')
axes[1].set_ylabel('Department')
axes[1].set_title('Life Expectancy at Birth (e_0)')
axes[1].grid(True, alpha=0.3, axis='x')
axes[1].invert_yaxis()

if len(fav_idx) > 0:
    axes[1].barh(results_df.loc[fav_idx[0], 'dpto'], 
                results_df.loc[fav_idx[0], 'e_0'], 
                color='darkblue', alpha=1.0, label='Your department')
    axes[1].legend()

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("LAB 2 COMPLETE! 🎉")
print("="*60)
print(f"\nAnalyzed {len(results_df)} departments successfully")
print(f"Life expectancy range: {results_df['e_0'].min():.1f} - {results_df['e_0'].max():.1f} years")
print(f"Infant mortality range: {results_df['infant_mx'].min()*1000:.1f} - {results_df['infant_mx'].max()*1000:.1f} per 1,000")
