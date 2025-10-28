# Solution script generated from Computer_Lab_1.ipynb
# Review the notebook version for cell outputs and richer formatting.

# # Demographic Research Methods and the PyCCM Library
# ## Computer Lab 1: An Introduction to Python for PyCCM
# ## Instructor: Jiani Yan
# ## Date: October 27th, 2025
# ---
# 
# In this class, we are going to learn about the basic building blocks of Python. We will try to finish both sections in the class, but it might be that Section 2 should be done as a 'class homework'. Section 1 will be live coded. Solutions for Section 2 will either be presented at the start of the second day, or at the end of the class, time permitting.


# ## 1. Section 1: To be taught by JY
# ---


# #### 1.1 Whats a float? Whats an int?

# An int (integer) is a whole number
population = 1000000  # int - no decimal
print(type(population))  # <class 'int'>

# A float is a number with decimals
birth_rate = 12.5  # float - has decimal
print(type(birth_rate))  # <class 'float'>

# Quick check
age = 25        # int
tfr = 1.8       # float (Total Fertility Rate)


# #### 1.1.1 What common arithmetic operators can we use on them?

# Basic math operations
print(10 + 5)   # Addition: 15
print(10 - 5)   # Subtraction: 5
print(10 * 5)   # Multiplication: 50
print(10 / 5)   # Division: 2.0 (always float!)
print(10 ** 2)  # Exponentiation: 100

# Demographic example
initial_pop = 100000
growth_rate = 0.02
final_pop = initial_pop * (1 + growth_rate)
print(f"After 1 year: {final_pop}")


# #### 1.1.2. Is 2/1 an int or a float? Why?

result = 2 / 1
print(result)        # 2.0
print(type(result))  # <class 'float'>

# Division (/) ALWAYS returns a float in Python 3
# Even when the result is a whole number!


# ### 1.2 How do we print things?

# Basic printing
print("Hola, Columbia!")

# Printing variables
city = "New York"
print(city)

# Using f-strings (recommended!)
population = 8336000
print(f"{city} has {population:,} people")


# ### 1.3 How do we assign things?

# Create variables
country = "Mexico"
population = 128000000
birth_rate = 17.6

# Multiple assignment
a, b = 10, 20
print(a, b)  # 10 20

# Shorthand operators
count = 5
count += 1  # Same as: count = count + 1
print(count)  # 6


# ### 1.4 What's a character set?

# Character sets define which characters we can use
# Python 3 uses UTF-8 (supports all languages!)

text_english = "Hello"
text_spanish = "Hola"
text_chinese = "你好"
emoji = "😊"

# All work perfectly in Python 3!


# ### 1.5 What's a string?

# A string is text in quotes
city = "Columbia"
greeting = 'Hola'  # Single or double quotes work

# Combine strings
message = greeting + ", " + city + "!"
print(message)  # Hola, Columbia!

# String length
print(len(city))  # 8


# #### 1.5.1 How do we slice strings?

age_range = "25-29"

# Get parts of the string
first_part = age_range[0:2]   # "25"
second_part = age_range[3:5]  # "29"

print(f"Ages {first_part} to {second_part}")

# Negative indexing
text = "Columbia"
print(text[-1])   # "a" (last character)
print(text[:4])   # "Colu" (first 4 characters)


# #### 1.5.2 What are some common string methods?

country = "  Mexico  "

# Useful string methods
print(country.strip())     # "Mexico" (remove spaces)
print(country.upper())     # "  MEXICO  "
print(country.lower())     # "  mexico  "
print(country.replace("Mexico", "Colombia"))  # "  Colombia  "

# Check contents
print("123".isdigit())     # True
print("abc".isalpha())     # True


# #### 1.5.3 How do we join and split strings?

# Split: break string into list
ages = "0-14,15-64,65+"
age_list = ages.split(',')
print(age_list)  # ['0-14', '15-64', '65+']

# Join: combine list into string
countries = ['USA', 'Canada', 'Mexico']
result = ', '.join(countries)
print(result)  # 'USA, Canada, Mexico'


# ### 1.6. What's a collection?

# Collections hold multiple items together
# Main types: lists, dictionaries, tuples

# List example
populations = [1000000, 1500000, 2000000]

# Dictionary example  
rates = {'birth': 12.5, 'death': 8.3}

# Tuple example
coordinates = (40.7128, -74.0060)  # NYC


# #### 1.6.1 What's a list?

# Lists are ordered collections (can change)
death_counts = [150, 200, 180, 220]

# Access by index (starts at 0!)
print(death_counts[0])   # 150 (first item)
print(death_counts[-1])  # 220 (last item)

# Modify
death_counts[0] = 155
death_counts.append(210)  # Add to end

# Useful operations
print(len(death_counts))  # Length
print(sum(death_counts))  # Sum
print(max(death_counts))  # Maximum


# #### 1.6.2 What's a dictionary?

# Dictionaries store key-value pairs
vital_rates = {
    'birth_rate': 12.5,
    'death_rate': 8.3,
    'tfr': 1.8
}

# Access by key
print(vital_rates['birth_rate'])  # 12.5

# Add new key-value
vital_rates['migration_rate'] = 2.1

# Check keys and values
print(vital_rates.keys())    # All keys
print(vital_rates.values())  # All values


# #### 1.6.3 What's a tuple?

# Tuples are like lists but CANNOT be changed
age_range = (0, 100)

# Access like lists
print(age_range[0])  # 0

# But cannot modify!
# age_range[0] = 5  # ERROR!

# Useful for fixed data
nyc_coords = (40.7128, -74.0060)
lat, lon = nyc_coords  # Unpack values


# #### 1.6.4 How do these things differ?

# Lists: Can change, use [ ]
my_list = [1, 2, 3]
my_list[0] = 10  # ✓ OK

# Tuples: Cannot change, use ( )
my_tuple = (1, 2, 3)
# my_tuple[0] = 10  # ✗ ERROR

# Dictionaries: Key-value pairs, use { }
my_dict = {'name': 'Alice', 'age': 30}
print(my_dict['name'])  # Access by key


# ### 1.7. What's an iteration?

# Iteration = repeating code for each item

# Example: print each population
populations = [1000000, 1500000, 2000000]
for pop in populations:
    print(f"Population: {pop:,}")

# Example: sum death counts
deaths = [150, 200, 180]
total = 0
for count in deaths:
    total += count
print(f"Total deaths: {total}")


# #### 1.7.1. How does indentation in Python work?

# Python uses INDENTATION (spaces) to show code blocks
# Use 4 spaces for each level

# Example: for loop
for i in range(3):
    print(f"Number: {i}")    # Indented = inside loop
    print("Still in loop")   # Same indent = still inside
print("Loop finished")       # Not indented = outside loop

# IMPORTANT: All code in a block must have SAME indentation!


# #### 1.7.2 How can we iterate over a collection?

# Method 1: Direct iteration (easiest!)
countries = ['USA', 'Canada', 'Mexico']
for country in countries:
    print(country)

# Method 2: With index using enumerate()
for i, country in enumerate(countries):
    print(f"{i}: {country}")

# Dictionary iteration
rates = {'birth': 12.5, 'death': 8.3}
for key, value in rates.items():
    print(f"{key}: {value}")


# #### 1.7.3 How can we iterate over multiple collections simultaneously?

# Use zip() to combine multiple lists
countries = ['USA', 'Canada', 'Mexico']
populations = [331000000, 38000000, 128000000]

for country, pop in zip(countries, populations):
    print(f"{country}: {pop:,}")

# Demographic example: calculate death rates
deaths = [830, 120, 256]
pops = [100000, 15000, 32000]

for d, p in zip(deaths, pops):
    rate = (d / p) * 1000
    print(f"Death rate: {rate:.2f} per 1,000")


# ### 1.8. What's Boolean Logic?

# Boolean = True or False

# Comparisons return True/False
print(5 > 3)     # True
print(5 == 5)    # True (== means "equals")
print(5 != 3)    # True (!= means "not equal")

# Logical operators
print(True and True)   # True (both must be True)
print(True or False)   # True (at least one is True)
print(not True)        # False (inverts)

# Demographic example
birth_rate = 12.5
death_rate = 8.3
is_growing = birth_rate > death_rate
print(f"Population growing: {is_growing}")  # True


# #### 1.8.1 How can we control flow in Python?

# if-else: choose between options
age = 25
if age >= 18:
    print("Adult")
else:
    print("Minor")

# if-elif-else: multiple conditions
population = 1500000
if population > 1000000:
    print("Large city")
elif population > 100000:
    print("Medium city")
else:
    print("Small city")

# One-line version
status = "Adult" if age >= 18 else "Minor"


# #### 1.8.2 How can we combine iteration and control in Python?

# Combine for loops with if statements
populations = [50000, 150000, 800000, 1500000]

for pop in populations:
    if pop > 1000000:
        print(f"{pop:,} - Large city")
    elif pop > 100000:
        print(f"{pop:,} - Medium city")
    else:
        print(f"{pop:,} - Small city")

# Process only certain items
ages = [5, 25, 35, 75]
for age in ages:
    if 15 <= age < 65:  # Working age only
        print(f"Age {age}: working age")


# #### 1.8.3 How do while loops work?

# while loop: repeat while condition is True
count = 0
while count < 3:
    print(f"Count: {count}")
    count += 1  # IMPORTANT: must update or infinite loop!

# Demographic example: population doubling
population = 100000
year = 0
while population < 200000:
    population = population * 1.02  # 2% growth
    year += 1
print(f"Population doubled in {year} years")


# #### 1.8.4 How can we continue if a condition is met?

# 'continue' skips to next iteration
for i in range(5):
    if i == 2:
        continue  # Skip when i is 2
    print(i)
# Prints: 0, 1, 3, 4 (skips 2)

# Example: skip missing data
values = [10, None, 20, None, 30]
total = 0
for val in values:
    if val is None:
        continue  # Skip None values
    total += val
print(f"Sum: {total}")  # 60


# #### 1.8.5 How can we break out of a loop at a specific point in time?

# 'break' exits the loop completely
numbers = [1, 3, 5, 7, 9, 11]
for num in numbers:
    if num == 7:
        print("Found 7!")
        break  # Stop searching
    print(f"Checking {num}")
# Output: Checking 1, Checking 3, Checking 5, Found 7!

# Example: stop when threshold exceeded
cumulative = 0
for i in range(1, 100):
    cumulative += i
    if cumulative > 100:
        print(f"Exceeded 100 at i={i}")
        break


# ### 1.9. How can we handle errors with 'try' and 'except'?

# Prevent crashes when errors might occur
try:
    result = 10 / 0  # This causes an error!
except ZeroDivisionError:
    print("Cannot divide by zero!")
    result = None

# Demographic example: handle missing data
populations = [100000, 0, 250000]
deaths = [830, 100, 2075]

for pop, death in zip(populations, deaths):
    try:
        rate = (death / pop) * 1000
        print(f"Rate: {rate:.2f}")
    except ZeroDivisionError:
        print("Skipping: zero population")


# ### 1.10. How can we define a function?

# Functions are reusable code blocks
def greet():
    print("Hola from Columbia!")

greet()  # Call the function

# Function with parameters
def greet_person(name):
    print(f"Hola, {name}!")

greet_person("Maria")

# Function with default values
def calculate_rate(deaths, population, per=1000):
    return (deaths / population) * per

rate = calculate_rate(830, 100000)
print(f"Death rate: {rate:.2f} per 1,000")


# #### 1.10.1 How can we return something from it?

# Return a single value
def add(a, b):
    return a + b

result = add(5, 3)
print(result)  # 8

# Return multiple values (as tuple)
def calculate_vital_rates(births, deaths, population):
    birth_rate = (births / population) * 1000
    death_rate = (deaths / population) * 1000
    return birth_rate, death_rate

br, dr = calculate_vital_rates(12500, 8300, 1000000)
print(f"Birth rate: {br:.2f}, Death rate: {dr:.2f}")


# #### 1.10.2 What's the difference between a local and a global variable?

# Global variable (defined outside functions)
population = 1000000

def print_population():
    print(f"Global population: {population}")  # Can read global

print_population()

# Local variable (defined inside function)
def calculate_rate():
    deaths = 8300  # Local - only exists in this function
    return deaths / population

print(calculate_rate())
# print(deaths)  # ERROR! deaths doesn't exist outside function

# Best practice: pass variables as parameters, don't use globals


# #### 1.11.1 How do we read files in Python
import os
# Best practice: use 'with' (auto-closes file)
with open('./course/labs/files/example_text.txt', 'r') as file:
    content = file.read()
    print(content)

# Read line by linef
with open('./course/labs/files/example_text.txt', 'r') as file:
    for line in file:
        print(line.strip())  # strip() removes newline

# Read all lines into a list
with open('./course/labs/files/example_text.txt', 'r') as file:
    lines = file.readlines()
    
# Note: 'r' = read mode


# #### 1.11.2 How do we write files in Python

# Write mode ('w') - creates new file or overwrites existing
with open('course/labs/files/output.txt', 'w') as file:
    file.write('Hola, Columbia!\n')
    file.write('This is line 2.\n')

# Append mode ('a') - adds to end of file
with open('course/labs/files/output.txt', 'a') as file:
    file.write('This line is appended.\n')

# Write multiple lines
lines = ['Line 1\n', 'Line 2\n', 'Line 3\n']
with open('course/labs/files/output.txt', 'w') as file:
    file.writelines(lines)


# ### 1.12. How do we import libraries, like numpy?

# Standard import with alias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Now we can use:
# np.array()
# pd.DataFrame()
# plt.plot()

# Example
data = np.array([1, 2, 3, 4, 5])
print(data)


# ### 1.12.1 What's a numpy array? why's it good?

import numpy as np

# Create numpy array
populations = np.array([100000, 150000, 200000])

# Why numpy is great:
# 1. Fast math operations (no loops needed!)
doubled = populations * 2
print(doubled)  # [200000 300000 400000]

# 2. Built-in statistics
print(populations.mean())  # Average
print(populations.sum())   # Total
print(populations.max())   # Maximum

# 3. Works with formulas
growth_rates = populations * 1.02
print(growth_rates)


# ### 1.12.2 How can we do some simple arithmetic operation with numpy?

import numpy as np

# Basic operations (element-wise)
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(a + b)    # [ 11  22  33  44]
print(a * b)    # [ 10  40  90 160]
print(a ** 2)   # [ 1  4  9 16]

# Works with single numbers too
print(a * 10)   # [10 20 30 40]

# Demographic example
deaths = np.array([830, 1200, 1650])
populations = np.array([100000, 150000, 200000])
death_rates = (deaths / populations) * 1000
print(f"Death rates: {death_rates}")


# ### 1.13. What's a dataframe?

import pandas as pd

# DataFrame = table with rows and columns (like Excel!)
data = {
    'country': ['USA', 'Canada', 'Mexico'],
    'population': [331000000, 38000000, 128000000],
    'birth_rate': [12.5, 10.9, 17.6]
}

df = pd.DataFrame(data)
print(df)

# Output:
#   country  population  birth_rate
# 0     USA   331000000        12.5
# 1  Canada    38000000        10.9
# 2  Mexico   128000000        17.6


# #### 1.13.1 How can we create an arbitrary dataframe?

import pandas as pd

# Method 1: From dictionary (easiest!)
data = {
    'age': [25, 30, 35, 40],
    'population': [150000, 180000, 165000, 142000]
}
df = pd.DataFrame(data)
print(df)

# Method 2: From list of lists
data = [
    ['USA', 331000000],
    ['Canada', 38000000],
    ['Mexico', 128000000]
]
df = pd.DataFrame(data, columns=['Country', 'Population'])
print(df)


# #### 1.13.2 How can we populate a dataframe from a numpy array?

import pandas as pd
import numpy as np

# Create numpy array
array = np.array([
    [100000, 12.5],
    [150000, 11.8],
    [200000, 10.5]
])

# Convert to DataFrame with column names
df = pd.DataFrame(array, columns=['Population', 'Birth Rate'])
print(df)

# From 1D arrays
ages = np.arange(0, 5)  # [0, 1, 2, 3, 4]
pops = np.array([50000, 48000, 46000, 44000, 42000])

df = pd.DataFrame({'age': ages, 'population': pops})
print(df)


# #### 1.13.3 How can we filter dataframes?

import pandas as pd

data = {
    'country': ['USA', 'Canada', 'Mexico', 'Brazil'],
    'population': [331000000, 38000000, 128000000, 212000000],
    'birth_rate': [12.5, 10.9, 17.6, 14.6]
}

df = pd.DataFrame(data)
# Filter for large countries
large = df[df['population'] > 100000000]
print(large)

# Filter for high birth rate
high_birth = df[df['birth_rate'] > 15]
print(high_birth)

# Multiple conditions (use & for AND, | for OR)
result = df[(df['population'] > 50000000) & (df['birth_rate'] > 12)]
print(result)


# #### 1.13.4 How can we sort them?

import pandas as pd

data = {
    'country': ['Brazil', 'USA', 'Mexico', 'Canada'],
    'population': [212000000, 331000000, 128000000, 38000000]
}
df = pd.DataFrame(data)

# Sort by population (ascending)
sorted_df = df.sort_values('population')
print(sorted_df)

# Sort descending
sorted_df = df.sort_values('population', ascending=False)
print(sorted_df)

# Sort by multiple columns
sorted_df = df.sort_values(['population', 'country'])
print(sorted_df)


# #### 1.13.5 How can we read files from disk with pandas?

import pandas as pd

# Read CSV file (most common)
df = pd.read_csv('course/labs/files/data.csv')

# With specific columns
df = pd.read_csv('course/labs/files/data.csv', usecols=['birth_rate', 'population'])

# For PyCCM: Read R data files (.rds)
# We'll use the data_loaders module
from src.data_loaders import load_all_data
df = load_all_data('./data')
df = pd.DataFrame(df['conteos'])

# #### 1.13.6 How can we write files to disk with pandas?

import pandas as pd

data = {'country': ['USA', 'Canada'], 'population': [331000000, 38000000]}
df = pd.DataFrame(data)

# Write to CSV (without row numbers)
df.to_csv('course/labs/files/output.csv', index=False)

# Write to Excel
df.to_excel('course/labs/files/output.xlsx', index=False)

# Append to existing CSV
df.to_csv('course/labs/files/output.csv', mode='a', header=False, index=False)


# ### 1.14. How can we import matplotlib?

import matplotlib.pyplot as plt
import numpy as np

# Standard import
# plt = plotting library

# Quick example
years = np.array([2015, 2016, 2017, 2018, 2019, 2020])
population = np.array([1000000, 1020000, 1041000, 1062000, 1084000, 1106000])

plt.plot(years, population)
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population Growth')
plt.show()


# #### 1.14.1 How can we plot a numpy array with matplotlib?

import matplotlib.pyplot as plt
import numpy as np

# Line plot
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

plt.plot(x, y, marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Line Plot')
plt.show()

# Demographic example
ages = np.arange(0, 101, 10)  # 0, 10, 20, ..., 100
death_rates = 0.001 + 0.0001 * np.exp(0.08 * ages)

plt.plot(ages, death_rates, linewidth=2, color='red')
plt.xlabel('Age')
plt.ylabel('Death Rate')
plt.title('Age-Specific Mortality')
plt.show()


# #### 1.14.2 How can we plot a pandas dataframe?

import matplotlib.pyplot as plt
import pandas as pd

data = {
    'year': [2015, 2016, 2017, 2018, 2019, 2020],
    'births': [130000, 132000, 128000, 126000, 124000, 122000],
    'deaths': [82000, 84000, 86000, 88000, 90000, 92000]
}
df = pd.DataFrame(data)

# Easy way: use DataFrame.plot()
df.plot(x='year', y=['births', 'deaths'], kind='line')
plt.title('Vital Statistics')
plt.ylabel('Count')
plt.show()

# Alternative: use matplotlib directly
plt.plot(df['year'], df['births'], label='Births', marker='o')
plt.plot(df['year'], df['deaths'], label='Deaths', marker='s')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()
plt.show()


# #### 1.15. How can we import functions from local modules?

# For PyCCM, we import from the src directory
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Now we can import PyCCM modules
from src.data_loaders import load_all_data
from src.mortality import make_lifetable
from src.fertility import make_fertility

# Example usage
# df = load_conteos('data/conteos.rds')
# lt = make_life_table(df)

# This allows us to use all PyCCM functions!


# ---
# ## 2. Section 2: Class questions -- either to be done in class, or as optional homework
# ---


# ### Q1. Replicate the exponential growth module seen in the lectures.
# 
# i. Create three numpy arrays with a starting value.
# 
# ii. Sequentially multiple them by the exponent.
# 
# iii. Plot the three different values.

import numpy as np
import matplotlib.pyplot as plt

# Starting population
initial_pop = 100000

# Years to project
years = np.arange(0, 31)  # 0 to 30 years

# Three growth scenarios
low_growth = initial_pop * (1.01) ** years     # 1% growth
med_growth = initial_pop * (1.02) ** years     # 2% growth
high_growth = initial_pop * (1.03) ** years    # 3% growth

# Plot all three
plt.figure(figsize=(10, 6))
plt.plot(years, low_growth, label='Low (1%)', marker='o', markersize=3)
plt.plot(years, med_growth, label='Medium (2%)', marker='s', markersize=3)
plt.plot(years, high_growth, label='High (3%)', marker='^', markersize=3)

plt.xlabel('Years')
plt.ylabel('Population')
plt.title('Exponential Population Growth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"After 30 years:")
print(f"  Low growth:  {low_growth[-1]:,.0f}")
print(f"  Med growth:  {med_growth[-1]:,.0f}")
print(f"  High growth: {high_growth[-1]:,.0f}")


# ### Question 2.
# 
# i. Load in the conteos.rds file hosted in this repository (`./data`) as a pandas dataframe. Ideally, use the data_loaders module.
# 
# ii. Filter the dataframe for total national data, EEVV, and the year 2018.
# 
# iii. Drop values where age is NaN.
# 
# iv. Add the population count for males to females.
# 
# v. Add the death counts for males to females.
# 
# vi. Divide death counts by the population values.
# 
# vii. PLot this -- death rates -- ('m_x') value. 
# 
# viiii. Repeat step vii., but instead with logged death rates.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# i. Load data
from src.data_loaders import load_all_data
df = load_all_data('../../data/')

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ii. Filter for 2018 and EEVV (adjust column names as needed)
df_filtered = df[(df['year'] == 2018) & (df['state'] == 'EEVV')].copy()

# iii. Drop NaN ages
df_filtered = df_filtered.dropna(subset=['age'])

# iv. Add male + female population (adjust column names!)
df_filtered['total_pop'] = df_filtered['pop_male'] + df_filtered['pop_female']

# v. Add male + female deaths
df_filtered['total_deaths'] = df_filtered['deaths_male'] + df_filtered['deaths_female']

# vi. Calculate death rate
df_filtered['m_x'] = df_filtered['total_deaths'] / df_filtered['total_pop']

# Sort by age
df_filtered = df_filtered.sort_values('age')

# vii. Plot death rates
plt.figure(figsize=(10, 6))
plt.plot(df_filtered['age'], df_filtered['m_x'], linewidth=2, marker='o')
plt.xlabel('Age')
plt.ylabel('Death Rate (m_x)')
plt.title('Age-Specific Death Rates - Mexico 2018')
plt.grid(True, alpha=0.3)
plt.show()

# viii. Plot logged death rates
plt.figure(figsize=(10, 6))
plt.semilogy(df_filtered['age'], df_filtered['m_x'], linewidth=2, marker='o')
plt.xlabel('Age')
plt.ylabel('Death Rate (log scale)')
plt.title('Age-Specific Death Rates (Log Scale) - Mexico 2018')
plt.grid(True, alpha=0.3)
plt.show()

print("\nDone! Check the plots.")
