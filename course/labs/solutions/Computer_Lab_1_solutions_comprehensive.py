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

# An int (integer) is a whole number without decimals
x = 5
print(type(x))  # <class 'int'>

# A float is a number with a decimal point
y = 5.0
print(type(y))  # <class 'float'>

# Examples
age = 25  # int
temperature = 36.6  # float
population = 1000000  # int
growth_rate = 0.015  # float


# #### 1.1.1 What common arithmetic operators can we use on them?

# Addition
result = 10 + 5  # 15

# Subtraction
result = 10 - 5  # 5

# Multiplication
result = 10 * 5  # 50

# Division (always returns float)
result = 10 / 5  # 2.0

# Floor division (returns int if both operands are int)
result = 10 // 3  # 3

# Modulus (remainder)
result = 10 % 3  # 1

# Exponentiation
result = 2 ** 3  # 8

# Example with demographic calculations
initial_population = 1000000
growth_rate = 0.02
years = 5
final_population = initial_population * (1 + growth_rate) ** years
print(f"Population after {years} years: {final_population:.0f}")


# #### 1.1.2. Is 2/1 an int or a float? Why?

result = 2 / 1
print(type(result))  # <class 'float'>
print(result)  # 2.0

# The division operator (/) ALWAYS returns a float in Python 3,
# even when dividing two integers that result in a whole number.
# This is to ensure consistency and avoid unexpected behavior.
# If you want integer division, use the // operator instead:

result_int = 2 // 1
print(type(result_int))  # <class 'int'>
print(result_int)  # 2


# ### 1.2 How do we print things?

# Basic printing
print("Hello, World!")

# Printing variables
name = "Jiani"
print(name)

# Printing multiple items (separated by spaces)
age = 30
print("My name is", name, "and I am", age, "years old")

# Using f-strings (recommended for Python 3.6+)
print(f"My name is {name} and I am {age} years old")

# Using format method
print("My name is {} and I am {} years old".format(name, age))

# Printing with custom separators
print("A", "B", "C", sep="-")  # A-B-C

# Printing without newline
print("Hello", end=" ")
print("World")  # Hello World


# ### 1.3 How do we assign things?

# Basic assignment
x = 5
name = "Alice"
is_student = True

# Multiple assignment
a, b, c = 1, 2, 3
print(a, b, c)  # 1 2 3

# Assigning the same value to multiple variables
x = y = z = 0

# Swapping values
a, b = 10, 20
a, b = b, a  # Now a=20, b=10

# Augmented assignment operators
count = 10
count += 5  # count = count + 5 (now 15)
count -= 3  # count = count - 3 (now 12)
count *= 2  # count = count * 2 (now 24)
count /= 4  # count = count / 4 (now 6.0)


# ### 1.4 What's a character set?

# A character set is a collection of characters that can be used in text.
# Common character sets:
# - ASCII: 128 characters (English letters, digits, symbols)
# - UTF-8: Universal character set (includes all languages, emojis, etc.)

# Python 3 uses Unicode (UTF-8) by default
text = "Hello"
chinese = "你好"
emoji = "😊"
mathematical = "α β γ"

# Each character has a numeric code
print(ord('A'))  # 65 (ASCII/Unicode code for 'A')
print(chr(65))   # 'A' (character for code 65)

# Encoding and decoding
text = "Hello"
encoded = text.encode('utf-8')  # Convert to bytes
print(encoded)  # b'Hello'
decoded = encoded.decode('utf-8')  # Convert back to string
print(decoded)  # 'Hello'


# ### 1.5 What's a string?

# A string is a sequence of characters enclosed in quotes
single_quoted = 'Hello'
double_quoted = "World"
triple_quoted = """This is a
multi-line string"""

# Strings are immutable (cannot be changed after creation)
text = "Hello"
# text[0] = 'h'  # This would cause an error!

# String concatenation
greeting = "Hello" + " " + "World"
print(greeting)  # Hello World

# String repetition
print("Ha" * 3)  # HaHaHa

# String length
print(len("Hello"))  # 5

# Checking membership
print("ell" in "Hello")  # True
print("xyz" in "Hello")  # False


# #### 1.5.1 How do we slice strings?

text = "Hello, World!"

# Basic indexing (0-based)
print(text[0])    # 'H' (first character)
print(text[-1])   # '!' (last character)
print(text[-2])   # 'd' (second to last)

# Slicing: [start:stop:step]
print(text[0:5])   # 'Hello' (characters 0-4)
print(text[7:12])  # 'World' (characters 7-11)
print(text[:5])    # 'Hello' (from start to 5)
print(text[7:])    # 'World!' (from 7 to end)
print(text[:])     # 'Hello, World!' (entire string)

# Step parameter
print(text[::2])   # 'Hlo ol!' (every 2nd character)
print(text[::-1])  # '!dlroW ,olleH' (reverse string)

# Practical example with age groups
age_label = "25-29"
start_age = age_label[:2]   # '25'
end_age = age_label[3:]     # '29'
print(f"Start: {start_age}, End: {end_age}")


# #### 1.5.2 What are some common string methods?

text = "  Hello, World!  "

# Case conversion
print(text.upper())       # '  HELLO, WORLD!  '
print(text.lower())       # '  hello, world!  '
print(text.capitalize())  # '  hello, world!  '
print(text.title())       # '  Hello, World!  '

# Whitespace removal
print(text.strip())       # 'Hello, World!' (both sides)
print(text.lstrip())      # 'Hello, World!  ' (left side)
print(text.rstrip())      # '  Hello, World!' (right side)

# Searching
print(text.find('World'))    # 9 (index of 'World')
print(text.find('xyz'))      # -1 (not found)
print(text.count('l'))       # 3 (occurrences of 'l')
print(text.startswith('  H')) # True
print(text.endswith('!  '))   # True

# Replacement
print(text.replace('World', 'Python'))  # '  Hello, Python!  '

# Checking content
print('123'.isdigit())    # True
print('abc'.isalpha())    # True
print('abc123'.isalnum()) # True


# #### 1.5.3 How do we join and split strings?

# Splitting strings
text = "apple,banana,orange"
fruits = text.split(',')
print(fruits)  # ['apple', 'banana', 'orange']

sentence = "Hello World from Python"
words = sentence.split()  # Split by whitespace (default)
print(words)  # ['Hello', 'World', 'from', 'Python']

# Split with limit
data = "1,2,3,4,5"
parts = data.split(',', 2)  # Split at most 2 times
print(parts)  # ['1', '2', '3,4,5']

# Joining strings
fruits = ['apple', 'banana', 'orange']
result = ','.join(fruits)
print(result)  # 'apple,banana,orange'

words = ['Hello', 'World']
sentence = ' '.join(words)
print(sentence)  # 'Hello World'

# Practical demographic example
age_groups = ['0-4', '5-9', '10-14', '15-19']
age_string = ', '.join(age_groups)
print(f"Age groups: {age_string}")
# Output: Age groups: 0-4, 5-9, 10-14, 15-19


# ### 1.6. What's a collection?

# A collection is a container that holds multiple items together.
# Python has several built-in collection types:
# - List: Ordered, mutable sequence (can be changed)
# - Tuple: Ordered, immutable sequence (cannot be changed)
# - Dictionary: Unordered collection of key-value pairs
# - Set: Unordered collection of unique elements

# Collections allow you to:
# 1. Store multiple values in a single variable
# 2. Organize related data together
# 3. Iterate over multiple items
# 4. Perform operations on groups of data

# Example: storing demographic data
populations = [1000000, 1500000, 2000000]  # list
coordinates = (40.7128, -74.0060)          # tuple (latitude, longitude)
age_structure = {'0-14': 25.5, '15-64': 65.0, '65+': 9.5}  # dictionary
unique_countries = {'USA', 'Canada', 'Mexico'}  # set


# #### 1.6.1 What's a list?

# A list is an ordered, mutable collection of items
# Items can be of any type and can be duplicated

# Creating lists
empty_list = []
numbers = [1, 2, 3, 4, 5]
mixed = [1, 'hello', 3.14, True]
nested = [[1, 2], [3, 4], [5, 6]]

# Accessing elements (0-indexed)
print(numbers[0])   # 1 (first element)
print(numbers[-1])  # 5 (last element)

# Slicing
print(numbers[1:4])  # [2, 3, 4]

# Modifying lists
numbers[0] = 10
print(numbers)  # [10, 2, 3, 4, 5]

# Adding elements
numbers.append(6)        # Add to end: [10, 2, 3, 4, 5, 6]
numbers.insert(1, 99)    # Insert at index 1: [10, 99, 2, 3, 4, 5, 6]
numbers.extend([7, 8])   # Add multiple: [10, 99, 2, 3, 4, 5, 6, 7, 8]

# Removing elements
numbers.remove(99)   # Remove first occurrence of 99
popped = numbers.pop()   # Remove and return last element
del numbers[0]       # Delete element at index 0

# Other useful operations
print(len(numbers))      # Length
print(sum(numbers))      # Sum (for numeric lists)
print(max(numbers))      # Maximum
print(min(numbers))      # Minimum
print(3 in numbers)      # Check membership

# Demographic example
death_counts = [150, 200, 180, 220, 195]
death_counts.append(210)  # Add new year
average_deaths = sum(death_counts) / len(death_counts)
print(f"Average annual deaths: {average_deaths:.1f}")


# #### 1.6.2 What's a dictionary?

# A dictionary is an unordered collection of key-value pairs
# Keys must be unique and immutable (strings, numbers, tuples)
# Values can be any type

# Creating dictionaries
empty_dict = {}
person = {
    'name': 'Alice',
    'age': 30,
    'city': 'New York'
}

# Accessing values
print(person['name'])        # 'Alice'
print(person.get('age'))     # 30
print(person.get('country', 'USA'))  # 'USA' (default if key doesn't exist)

# Modifying dictionaries
person['age'] = 31           # Update existing key
person['country'] = 'USA'    # Add new key-value pair

# Removing items
del person['city']           # Remove key-value pair
age = person.pop('age')      # Remove and return value

# Dictionary methods
print(person.keys())         # dict_keys(['name', 'country'])
print(person.values())       # dict_values(['Alice', 'USA'])
print(person.items())        # dict_items([('name', 'Alice'), ('country', 'USA')])

# Checking membership
print('name' in person)      # True (checks keys, not values)

# Demographic example
vital_rates = {
    'birth_rate': 12.5,
    'death_rate': 8.3,
    'tfr': 1.8,
    'life_expectancy': 78.5
}
print(f"Total Fertility Rate: {vital_rates['tfr']}")
vital_rates['migration_rate'] = 2.1  # Add new rate
print(f"Natural increase: {vital_rates['birth_rate'] - vital_rates['death_rate']:.1f} per 1000")


# #### 1.6.3 What's a tuple?

# A tuple is an ordered, immutable collection of items
# Once created, items cannot be added, removed, or changed
# Tuples are faster than lists and can be used as dictionary keys

# Creating tuples
empty_tuple = ()
single_item = (1,)  # Note: comma is required for single-item tuple
coordinates = (40.7128, -74.0060)
mixed = (1, 'hello', 3.14, True)
nested = ((1, 2), (3, 4))

# Accessing elements (same as lists)
print(coordinates[0])   # 40.7128
print(coordinates[-1])  # -74.0060

# Slicing works too
numbers = (1, 2, 3, 4, 5)
print(numbers[1:4])  # (2, 3, 4)

# Tuples are immutable - this would cause an error:
# coordinates[0] = 50.0  # TypeError!

# Unpacking tuples
x, y = coordinates
print(f"Latitude: {x}, Longitude: {y}")

# Tuple methods (limited due to immutability)
print(numbers.count(3))  # 1 (occurrences of 3)
print(numbers.index(4))  # 3 (index of first 4)

# When to use tuples vs lists:
# - Use tuples for data that shouldn't change (coordinates, RGB colors, etc.)
# - Use lists for data that needs to be modified

# Demographic example
age_range = (0, 100)  # Min and max age
start, end = age_range
print(f"Age range: {start} to {end} years")


# #### 1.6.4 How do these things differ?

# LISTS vs TUPLES vs DICTIONARIES

# 1. MUTABILITY
my_list = [1, 2, 3]
my_list[0] = 10      # ✓ OK - lists are mutable
my_tuple = (1, 2, 3)
# my_tuple[0] = 10   # ✗ ERROR - tuples are immutable

# 2. SYNTAX
list_example = [1, 2, 3]              # Square brackets
tuple_example = (1, 2, 3)              # Parentheses
dict_example = {'a': 1, 'b': 2}        # Curly braces with key:value

# 3. ORDERING
# Lists and tuples: ordered (maintain insertion order)
# Dictionaries: ordered as of Python 3.7+ (maintains insertion order)

# 4. ACCESS METHOD
my_list = ['a', 'b', 'c']
print(my_list[0])                      # Access by index: 'a'

my_dict = {'name': 'Alice', 'age': 30}
print(my_dict['name'])                 # Access by key: 'Alice'

# 5. USE CASES
# Lists: Ordered collection of items that may change
ages = [25, 30, 35, 40]
ages.append(45)  # Can add more

# Tuples: Ordered collection that shouldn't change
coordinates = (40.7128, -74.0060)  # NYC coordinates (fixed)

# Dictionaries: Labeled data with key-value relationships
person = {
    'name': 'Bob',
    'age': 35,
    'occupation': 'Demographer'
}

# 6. PERFORMANCE
# Tuples: Faster than lists (due to immutability)
# Dictionaries: Very fast lookups by key (O(1) average)
# Lists: Fast for sequential access, slower for searches


# ### 1.7. What's an iteration?

# Iteration is the process of repeating a block of code multiple times.
# It allows you to process each item in a collection or perform an action
# a specific number of times.

# The most common iteration tool in Python is the 'for' loop

# Example 1: Iterating over a list
populations = [1000000, 1500000, 2000000, 2500000]
for pop in populations:
    print(f"Population: {pop:,}")

# Example 2: Iterating over a range of numbers
for i in range(5):  # 0, 1, 2, 3, 4
    print(f"Year {i}")

# Example 3: Iterating with index
age_groups = ['0-14', '15-64', '65+']
for i, group in enumerate(age_groups):
    print(f"Index {i}: {group}")

# Example 4: Iterating over dictionary
vital_rates = {'births': 12.5, 'deaths': 8.3, 'tfr': 1.8}
for key, value in vital_rates.items():
    print(f"{key}: {value}")

# Iteration is fundamental for:
# - Processing data in collections
# - Performing calculations on multiple items
# - Automating repetitive tasks
# - Building up results incrementally


# #### 1.7.1. How does indentation in Python work?

# Python uses indentation (whitespace) to define code blocks.
# Other languages use curly braces {}, but Python uses indentation.
# Standard practice: use 4 spaces per indentation level

# Example 1: Indentation in loops
for i in range(3):
    print(f"Outer loop: {i}")    # Indented - part of loop
    for j in range(2):
        print(f"  Inner loop: {j}")  # Double indented - part of inner loop
    print("Back to outer loop")  # Indented - still part of outer loop
print("Loop finished")           # Not indented - outside loop

# Example 2: Indentation in conditionals
age = 25
if age >= 18:
    print("Adult")               # Indented - part of if block
    print("Can vote")            # Still indented - still part of if block
print("Check complete")          # Not indented - outside if block

# IMPORTANT RULES:
# 1. Consistent indentation is REQUIRED (mixing tabs and spaces causes errors)
# 2. All statements in a block must have the same indentation level
# 3. Use 4 spaces (not tabs) as the standard
# 4. Colons (:) indicate the start of an indented block

# Common errors:
# IndentationError: unexpected indent
# IndentationError: expected an indented block


# #### 1.7.2 How can we iterate over a collection?

# Method 1: Direct iteration (most Pythonic)
populations = [1000000, 1500000, 2000000]
for pop in populations:
    print(f"Population: {pop:,}")

# Method 2: Iteration with index using enumerate()
for i, pop in enumerate(populations):
    print(f"City {i+1}: {pop:,}")

# Method 3: Using range and length (less Pythonic)
for i in range(len(populations)):
    print(f"Index {i}: {populations[i]:,}")

# Iterating over strings
name = "Python"
for char in name:
    print(char)  # P, y, t, h, o, n

# Iterating over dictionaries
vital_rates = {'births': 12.5, 'deaths': 8.3, 'tfr': 1.8}

# Just keys
for key in vital_rates:
    print(key)

# Keys explicitly
for key in vital_rates.keys():
    print(key)

# Values
for value in vital_rates.values():
    print(value)

# Key-value pairs
for key, value in vital_rates.items():
    print(f"{key}: {value}")

# Iterating over tuples
coordinates = [(0, 0), (1, 2), (3, 4)]
for x, y in coordinates:
    print(f"Point: ({x}, {y})")


# #### 1.7.3 How can we iterate over multiple collections simultaneously?

# Method 1: Using zip() - most common and Pythonic
countries = ['USA', 'Canada', 'Mexico']
populations = [331000000, 38000000, 128000000]
gdp = [21.43, 1.74, 1.29]  # in trillions

for country, pop, gdp_val in zip(countries, populations, gdp):
    print(f"{country}: {pop:,} people, ${gdp_val}T GDP")

# zip() stops at the shortest list
list1 = [1, 2, 3, 4, 5]
list2 = ['a', 'b', 'c']
for num, letter in zip(list1, list2):
    print(num, letter)  # Only prints 3 pairs

# Method 2: Using enumerate() with multiple indexing
for i, country in enumerate(countries):
    print(f"{i}: {country} has {populations[i]:,} people")

# Method 3: Creating pairs with zip
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
pairs = list(zip(names, ages))
print(pairs)  # [('Alice', 25), ('Bob', 30), ('Charlie', 35)]

# Method 4: Parallel iteration with indices
for i in range(len(countries)):
    print(f"{countries[i]}: {populations[i]:,}")

# Demographic example: calculating death rates
age_groups = ['0-14', '15-64', '65+']
deaths = [5000, 15000, 80000]
population = [250000, 650000, 100000]

for age, d, p in zip(age_groups, deaths, population):
    death_rate = (d / p) * 1000
    print(f"Age {age}: {death_rate:.2f} per 1,000")


# ### 1.8. What's Boolean Logic?

# Boolean logic deals with True and False values.
# It's fundamental for making decisions in code.

# Boolean values
is_alive = True
is_deceased = False

# Comparison operators (return Boolean values)
print(5 > 3)    # True
print(5 < 3)    # False
print(5 == 5)   # True (equality)
print(5 != 3)   # True (not equal)
print(5 >= 5)   # True (greater than or equal)
print(5 <= 4)   # False (less than or equal)

# Logical operators
# AND - both conditions must be True
print(True and True)    # True
print(True and False)   # False

# OR - at least one condition must be True
print(True or False)    # True
print(False or False)   # False

# NOT - inverts the Boolean value
print(not True)         # False
print(not False)        # True

# Combining conditions
age = 25
income = 50000
print(age > 18 and income > 30000)  # True

# Demographic example
birth_rate = 12.5
death_rate = 8.3
is_growing = birth_rate > death_rate
print(f"Population growing: {is_growing}")  # True

# Truthiness in Python
# False values: False, None, 0, 0.0, '', [], {}, ()
# Everything else is True
print(bool(0))      # False
print(bool(1))      # True
print(bool([]))     # False (empty list)
print(bool([1]))    # True (non-empty list)


# #### 1.8.1 How can we control flow in Python?

# IF statement - execute code only if condition is True
age = 25
if age >= 18:
    print("Adult")

# IF-ELSE - choose between two options
age = 15
if age >= 18:
    print("Adult")
else:
    print("Minor")

# IF-ELIF-ELSE - multiple conditions
age = 35
if age < 18:
    print("Minor")
elif age < 65:
    print("Working age")
else:
    print("Senior")

# Nested conditions
population = 100000
growth_rate = 0.02

if population > 50000:
    print("Large population")
    if growth_rate > 0:
        print("And it's growing!")
    else:
        print("But it's declining")

# One-line conditional (ternary operator)
age = 20
status = "Adult" if age >= 18 else "Minor"
print(status)

# Multiple conditions with and/or
birth_rate = 12.5
death_rate = 8.3
migration_rate = 2.0

if birth_rate > death_rate and migration_rate > 0:
    print("Population is growing from both natural increase and migration")

# Demographic example: classify population growth
natural_increase = birth_rate - death_rate
total_growth = natural_increase + migration_rate

if total_growth > 2.0:
    print("Rapid growth")
elif total_growth > 0.5:
    print("Moderate growth")
elif total_growth > 0:
    print("Slow growth")
else:
    print("Declining population")


# #### 1.8.2 How can we combine iteration and control in Python?

# Combining for loops with if statements
populations = [50000, 150000, 300000, 800000, 1500000]

for pop in populations:
    if pop > 1000000:
        print(f"{pop:,} - Large city")
    elif pop > 100000:
        print(f"{pop:,} - Medium city")
    else:
        print(f"{pop:,} - Small city")

# List comprehension with condition (advanced but very Pythonic)
large_cities = [pop for pop in populations if pop > 100000]
print(f"Large cities: {large_cities}")

# Processing demographic data with conditions
ages = [5, 15, 25, 35, 45, 55, 65, 75, 85]
deaths = [10, 5, 8, 12, 20, 35, 60, 90, 120]

# Calculate death rates only for working age
for age, death_count in zip(ages, deaths):
    if 15 <= age < 65:
        print(f"Age {age}: {death_count} deaths (working age)")

# Count items meeting a condition
high_mortality_ages = 0
for age, death_count in zip(ages, deaths):
    if death_count > 50:
        high_mortality_ages += 1
print(f"Ages with high mortality: {high_mortality_ages}")

# Filter and transform data
vital_statistics = {
    'births': 125000,
    'deaths': 83000,
    'marriages': 45000,
    'divorces': 22000
}

# Print only statistics above threshold
threshold = 50000
print("Major vital events:")
for event, count in vital_statistics.items():
    if count > threshold:
        print(f"  {event}: {count:,}")


# #### 1.8.3 How do while loops work?

# While loops continue executing as long as a condition is True
# Be careful: infinite loops can occur if condition never becomes False!

# Basic while loop
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1  # IMPORTANT: update the condition variable!

# While loop with user input (simulation)
population = 100000
year = 0
growth_rate = 0.03

print("Population projection:")
while population < 200000:
    print(f"Year {year}: {population:,.0f}")
    population = population * (1 + growth_rate)
    year += 1
print(f"Doubled in {year} years!")

# While True with break (infinite loop with exit condition)
total = 0
i = 0
while True:
    total += i
    i += 1
    if total > 100:
        break  # Exit the loop
print(f"Total: {total}")

# While with else (executes if loop completes normally)
n = 0
while n < 3:
    print(n)
    n += 1
else:
    print("Loop completed normally")

# Demographic example: find year when TFR drops below replacement
tfr = 2.5
year = 2020
decline_rate = 0.02  # 2% annual decline

while tfr > 2.1:  # Replacement level
    print(f"{year}: TFR = {tfr:.2f}")
    tfr = tfr * (1 - decline_rate)
    year += 1
print(f"Below replacement in {year}")


# #### 1.8.4 How can we continue if a condition is met?

# The 'continue' statement skips the rest of the current iteration
# and moves to the next iteration of the loop

# Example 1: Skip even numbers
for i in range(10):
    if i % 2 == 0:  # If even
        continue     # Skip to next iteration
    print(i)  # Only prints odd numbers: 1, 3, 5, 7, 9

# Example 2: Skip negative values
values = [10, -5, 20, -3, 30, 0, 15]
positive_sum = 0
for value in values:
    if value <= 0:
        continue  # Skip non-positive values
    positive_sum += value
print(f"Sum of positive values: {positive_sum}")  # 75

# Example 3: Skip missing data
death_counts = [120, None, 135, 142, None, 150, 148]
total_deaths = 0
valid_count = 0

for count in death_counts:
    if count is None:
        continue  # Skip missing values
    total_deaths += count
    valid_count += 1

average = total_deaths / valid_count
print(f"Average deaths (excluding None): {average:.1f}")

# Demographic example: process only specific age groups
age_groups = ['0-14', '15-24', '25-64', '65-74', '75+']
populations = [250000, 180000, 650000, 120000, 80000]

print("Working age populations:")
for age, pop in zip(age_groups, populations):
    if age == '0-14' or age == '75+':
        continue  # Skip children and elderly
    print(f"  {age}: {pop:,}")


# #### 1.8.5 How can we break out of a loop at a specific point in time?

# The 'break' statement immediately exits the loop entirely
# (unlike 'continue' which only skips the current iteration)

# Example 1: Stop when target is found
numbers = [1, 3, 5, 7, 9, 11, 13]
target = 9

for num in numbers:
    if num == target:
        print(f"Found {target}!")
        break  # Exit loop immediately
    print(f"Checking {num}")
# Output: Checking 1, Checking 3, Checking 5, Checking 7, Found 9!

# Example 2: Stop when threshold is exceeded
cumulative_sum = 0
for i in range(1, 100):
    cumulative_sum += i
    if cumulative_sum > 100:
        print(f"Sum exceeded 100 at i={i}, sum={cumulative_sum}")
        break

# Example 3: Search with early exit
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston']
populations = [8336000, 3979000, 2693000, 2320000]

threshold = 3000000
for city, pop in zip(cities, populations):
    if pop > threshold:
        print(f"First city over {threshold:,}: {city} ({pop:,})")
        break

# Demographic example: Find year when population exceeds target
population = 1000000
target_pop = 2000000
year = 2020
growth_rate = 0.02

for i in range(100):  # Maximum 100 years
    if population >= target_pop:
        print(f"Target reached in {year} (population: {population:,.0f})")
        break
    population *= (1 + growth_rate)
    year += 1
else:
    print("Target not reached in 100 years")

# break vs continue vs pass
for i in range(5):
    if i == 2:
        break  # Stops loop completely (only prints 0, 1)
    print(i)

for i in range(5):
    if i == 2:
        continue  # Skips only i=2 (prints 0, 1, 3, 4)
    print(i)

for i in range(5):
    if i == 2:
        pass  # Does nothing (prints 0, 1, 2, 3, 4)
    print(i)


# ### 1.9. How can we handle errors with 'try' and 'except'?

# Try-except blocks allow you to handle errors gracefully
# instead of crashing your program

# Basic try-except
try:
    result = 10 / 0  # This will cause a ZeroDivisionError
except ZeroDivisionError:
    print("Cannot divide by zero!")
    result = None

# Multiple exception types
try:
    numbers = [1, 2, 3]
    print(numbers[10])  # IndexError
except IndexError:
    print("Index out of range")
except ValueError:
    print("Invalid value")

# Catch any exception
try:
    risky_operation = int("not a number")
except Exception as e:
    print(f"An error occurred: {e}")

# Try-except-else-finally
try:
    death_rate = 8300 / 100000
except ZeroDivisionError:
    print("Cannot calculate rate: zero population")
else:
    print(f"Death rate: {death_rate}")  # Runs if no error
finally:
    print("Calculation attempt complete")  # Always runs

# Practical demographic example
populations = [100000, 0, 250000, None, 500000]
deaths = [830, 100, 2075, 1500, 4150]

death_rates = []
for pop, death in zip(populations, deaths):
    try:
        rate = (death / pop) * 1000
        death_rates.append(rate)
    except ZeroDivisionError:
        print(f"Warning: Zero population, skipping")
        death_rates.append(None)
    except TypeError:
        print(f"Warning: Missing data, skipping")
        death_rates.append(None)

print(f"Death rates: {death_rates}")

# File handling with error handling
try:
    with open('data.csv', 'r') as file:
        data = file.read()
except FileNotFoundError:
    print("File not found")
except PermissionError:
    print("No permission to read file")
except Exception as e:
    print(f"Unexpected error: {e}")


# ### 1.10. How can we define a function?

# Functions are reusable blocks of code that perform a specific task
# Defined using the 'def' keyword

# Basic function with no parameters
def greet():
    print("Hello, World!")

greet()  # Call the function

# Function with parameters
def greet_person(name):
    print(f"Hello, {name}!")

greet_person("Alice")

# Function with multiple parameters
def calculate_growth_rate(births, deaths, population):
    natural_increase = births - deaths
    growth_rate = (natural_increase / population) * 100
    return growth_rate

# Function with default parameter values
def calculate_population(initial, rate=0.02, years=10):
    final = initial * (1 + rate) ** years
    return final

# Call with defaults
pop1 = calculate_population(100000)
# Call with custom values
pop2 = calculate_population(100000, rate=0.03, years=5)

# Function with keyword arguments
def create_person(name, age, city="Unknown"):
    return {
        'name': name,
        'age': age,
        'city': city
    }

person1 = create_person("Alice", 30, city="New York")
person2 = create_person(age=25, name="Bob")  # Order doesn't matter with keywords

# Demographic example
def calculate_death_rate(deaths, population, per=1000):
    """
    Calculate death rate per specified population size.
    
    Parameters:
    - deaths: number of deaths
    - population: total population
    - per: rate per this many people (default: 1000)
    
    Returns:
    - death rate as float
    """
    return (deaths / population) * per

rate = calculate_death_rate(8300, 100000)
print(f"Death rate: {rate:.2f} per 1,000")


# #### 1.10.1 How can we return something from it?

# Functions can return values using the 'return' statement

# Simple return
def add(a, b):
    return a + b

result = add(5, 3)
print(result)  # 8

# Return without value returns None
def print_greeting(name):
    print(f"Hello, {name}")
    return  # Optional, returns None

# Multiple return values (as a tuple)
def calculate_vital_rates(births, deaths, population):
    birth_rate = (births / population) * 1000
    death_rate = (deaths / population) * 1000
    natural_increase = birth_rate - death_rate
    return birth_rate, death_rate, natural_increase

br, dr, ni = calculate_vital_rates(12500, 8300, 1000000)
print(f"Birth rate: {br:.2f}, Death rate: {dr:.2f}, Natural increase: {ni:.2f}")

# Return different types based on condition
def classify_growth(rate):
    if rate > 2.0:
        return "rapid"
    elif rate > 0:
        return "slow"
    else:
        return "declining"

status = classify_growth(1.5)
print(status)  # "slow"

# Early return (exit function before end)
def calculate_rate(numerator, denominator):
    if denominator == 0:
        return None  # Early exit if invalid
    return (numerator / denominator) * 1000

# Return dictionary
def get_population_stats(pop_data):
    return {
        'total': sum(pop_data),
        'average': sum(pop_data) / len(pop_data),
        'max': max(pop_data),
        'min': min(pop_data)
    }

populations = [100000, 150000, 200000]
stats = get_population_stats(populations)
print(stats)  # {'total': 450000, 'average': 150000.0, 'max': 200000, 'min': 100000}


# #### 1.10.2 What's the difference between a local and a global variable?

# GLOBAL VARIABLES: Defined outside functions, accessible everywhere
population = 1000000  # Global variable

def print_population():
    print(f"Population: {population}")  # Can read global variable

print_population()

# LOCAL VARIABLES: Defined inside functions, only accessible within that function
def calculate_rate():
    deaths = 8300  # Local variable
    rate = (deaths / population) * 1000
    return rate

print(calculate_rate())
# print(deaths)  # ERROR! 'deaths' is not accessible outside the function

# Modifying global variables requires 'global' keyword
counter = 0

def increment_counter():
    global counter  # Tell Python we want to modify the global variable
    counter += 1

increment_counter()
print(counter)  # 1

# Without 'global', Python creates a new local variable
value = 10

def try_to_modify():
    value = 20  # Creates a NEW local variable, doesn't modify global
    print(f"Inside function: {value}")

try_to_modify()  # Inside function: 20
print(f"Outside function: {value}")  # Outside function: 10

# Best practice: avoid global variables, pass parameters instead
total_pop = 1000000

def calculate_death_rate(deaths, population):  # Parameters, not globals
    return (deaths / population) * 1000

rate = calculate_death_rate(8300, total_pop)

# Variable shadowing example
x = 100  # Global

def test_scope():
    x = 50  # Local variable, "shadows" global x
    print(f"Local x: {x}")

test_scope()  # Local x: 50
print(f"Global x: {x}")  # Global x: 100

# Using nonlocal for nested functions
def outer():
    count = 0
    
    def inner():
        nonlocal count  # Modify outer function's variable
        count += 1
    
    inner()
    print(f"Count: {count}")  # 1

outer()


# #### 1.11.1 How do we read files in Python

# Method 1: Using open() with manual close
file = open('example.txt', 'r')  # 'r' = read mode
content = file.read()
file.close()  # IMPORTANT: always close files!

# Method 2: Using 'with' statement (recommended - auto-closes)
with open('example.txt', 'r') as file:
    content = file.read()
# File is automatically closed after the with block

# Read entire file as string
with open('data.txt', 'r') as file:
    content = file.read()
    print(content)

# Read line by line
with open('data.txt', 'r') as file:
    for line in file:
        print(line.strip())  # strip() removes newline characters

# Read all lines into a list
with open('data.txt', 'r') as file:
    lines = file.readlines()  # Returns list of lines
    print(lines)

# Read specific number of characters
with open('data.txt', 'r') as file:
    first_100_chars = file.read(100)

# Different file modes:
# 'r'  - Read (default)
# 'w'  - Write (overwrites file)
# 'a'  - Append (adds to end)
# 'r+' - Read and write
# 'rb' - Read binary
# 'wb' - Write binary

# Demographic example: reading CSV manually
with open('population_data.csv', 'r') as file:
    header = file.readline().strip()
    print(f"Columns: {header}")
    
    for line in file:
        values = line.strip().split(',')
        print(f"Country: {values[0]}, Population: {values[1]}")

# Error handling when reading files
try:
    with open('might_not_exist.txt', 'r') as file:
        content = file.read()
except FileNotFoundError:
    print("File not found!")
except PermissionError:
    print("No permission to read file!")


# #### 1.11.2 How do we write files in Python

# Method 1: Write mode ('w') - overwrites existing file
with open('output.txt', 'w') as file:
    file.write('Hello, World!\n')
    file.write('This is a new line.\n')

# Method 2: Append mode ('a') - adds to end of file
with open('output.txt', 'a') as file:
    file.write('This line is appended.\n')

# Write multiple lines at once
lines = ['Line 1\n', 'Line 2\n', 'Line 3\n']
with open('output.txt', 'w') as file:
    file.writelines(lines)

# Writing formatted data
populations = [1000000, 1500000, 2000000]
with open('populations.txt', 'w') as file:
    for i, pop in enumerate(populations, start=1):
        file.write(f"City {i}: {pop:,}\n")

# Writing CSV data
data = [
    ['Country', 'Population', 'Birth Rate'],
    ['USA', '331000000', '12.5'],
    ['Canada', '38000000', '10.9'],
    ['Mexico', '128000000', '17.6']
]

with open('countries.csv', 'w') as file:
    for row in data:
        line = ','.join(row) + '\n'
        file.write(line)

# Demographic example: save death rates
age_groups = ['0-14', '15-64', '65+']
death_rates = [1.2, 5.5, 45.8]

with open('death_rates.csv', 'w') as file:
    file.write('Age Group,Death Rate per 1000\n')  # Header
    for age, rate in zip(age_groups, death_rates):
        file.write(f'{age},{rate}\n')

# Write with error handling
try:
    with open('protected_file.txt', 'w') as file:
        file.write('Important data')
except PermissionError:
    print("Cannot write to file: permission denied")

# Note: Using 'w' mode will CREATE the file if it doesn't exist,
# and OVERWRITE it if it does exist. Use 'a' to preserve existing content.


# ### 1.12. How do we import libraries, like numpy?

# Method 1: Import entire module
import numpy
array1 = numpy.array([1, 2, 3])

# Method 2: Import with alias (most common for numpy)
import numpy as np
array2 = np.array([1, 2, 3])

# Method 3: Import specific functions
from numpy import array, mean, std
array3 = array([1, 2, 3])
average = mean(array3)

# Method 4: Import everything (not recommended - clutters namespace)
from numpy import *
# Now you can use any numpy function without prefix, but this can cause conflicts

# Standard library imports (no installation needed)
import math
import random
import datetime
import os

# Third-party imports (need installation via pip)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Multiple imports on one line
import os, sys, math

# Better practice: one per line
import os
import sys
import math

# Importing from submodules
from numpy import random as np_random
from matplotlib import pyplot as plt

# Checking what's available in a module
import numpy as np
# print(dir(np))  # Lists all functions and attributes

# Demographic example
import numpy as np
import pandas as pd

# Create population data
ages = np.arange(0, 101)  # 0 to 100
populations = np.random.randint(5000, 15000, size=101)

# Create DataFrame
df = pd.DataFrame({
    'age': ages,
    'population': populations
})

print(df.head())


# ### 1.12.1 What's a numpy array? why's it good?

import numpy as np

# A numpy array is like a Python list, but much faster and more powerful
# for numerical operations

# Creating arrays
list_data = [1, 2, 3, 4, 5]
array_data = np.array([1, 2, 3, 4, 5])

print(type(list_data))   # <class 'list'>
print(type(array_data))  # <class 'numpy.ndarray'>

# Why numpy arrays are better than lists:

# 1. SPEED: Much faster for numerical operations (10-100x)
# Lists require loops, arrays use optimized C code

# 2. MEMORY EFFICIENCY: Arrays use less memory
# Lists store pointers to objects, arrays store values directly

# 3. VECTORIZED OPERATIONS: No loops needed
list_result = [x * 2 for x in list_data]  # List: need loop
array_result = array_data * 2              # Array: direct operation

# 4. MATHEMATICAL OPERATIONS: Built-in functions
populations = np.array([100000, 150000, 200000, 250000])
print(f"Mean: {populations.mean()}")
print(f"Std: {populations.std()}")
print(f"Sum: {populations.sum()}")
print(f"Max: {populations.max()}")
print(f"Min: {populations.min()}")

# 5. MULTI-DIMENSIONAL: Easy to work with matrices
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(matrix.shape)  # (3, 3)

# Creating special arrays
zeros = np.zeros(5)           # [0. 0. 0. 0. 0.]
ones = np.ones(5)             # [1. 1. 1. 1. 1.]
range_array = np.arange(0, 10, 2)  # [0 2 4 6 8]
linspace = np.linspace(0, 1, 5)    # [0.   0.25 0.5  0.75 1.  ]

# Array indexing and slicing (like lists)
ages = np.array([25, 30, 35, 40, 45])
print(ages[0])      # 25
print(ages[-1])     # 45
print(ages[1:4])    # [30 35 40]

# Boolean indexing (very powerful!)
print(ages > 35)         # [False False False  True  True]
print(ages[ages > 35])   # [40 45]

# Demographic example
death_counts = np.array([120, 135, 142, 150, 148, 155])
populations = np.array([100000, 102000, 104000, 106000, 108000, 110000])
death_rates = (death_counts / populations) * 1000  # Vectorized!
print(f"Death rates: {death_rates}")


# ### 1.12.2 How can we do some simple arithmetic operation with numpy?

import numpy as np

# Basic arithmetic (element-wise)
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

# Addition
print(a + b)      # [11 22 33 44 55]

# Subtraction
print(b - a)      # [ 9 18 27 36 45]

# Multiplication
print(a * b)      # [ 10  40  90 160 250]

# Division
print(b / a)      # [10. 10. 10. 10. 10.]

# Exponentiation
print(a ** 2)     # [ 1  4  9 16 25]

# Scalar operations (broadcasting)
print(a * 10)     # [10 20 30 40 50]
print(a + 5)      # [ 6  7  8  9 10]

# Mathematical functions
populations = np.array([100000, 150000, 200000, 250000])

# Square root
print(np.sqrt(populations))

# Logarithm
print(np.log(populations))
print(np.log10(populations))

# Exponential
growth_rates = np.array([0.01, 0.02, 0.03])
print(np.exp(growth_rates))

# Trigonometric
angles = np.array([0, np.pi/2, np.pi])
print(np.sin(angles))
print(np.cos(angles))

# Statistical functions
data = np.array([12.5, 8.3, 15.2, 10.8, 9.4])
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Std: {np.std(data)}")
print(f"Variance: {np.var(data)}")
print(f"Sum: {np.sum(data)}")
print(f"Min: {np.min(data)}")
print(f"Max: {np.max(data)}")

# Cumulative operations
print(np.cumsum(a))   # Cumulative sum: [ 1  3  6 10 15]
print(np.cumprod(a))  # Cumulative product: [  1   2   6  24 120]

# Demographic example: population projection
initial_pop = np.array([100000, 150000, 200000])
growth_rate = 0.02
years = 5

# Calculate final population after 5 years
final_pop = initial_pop * (1 + growth_rate) ** years
print(f"Initial: {initial_pop}")
print(f"Final (after {years} years): {final_pop.astype(int)}")

# Calculate death rates
deaths = np.array([830, 1200, 1650])
populations = np.array([100000, 150000, 200000])
death_rates = (deaths / populations) * 1000
print(f"Death rates per 1,000: {death_rates}")


# ### 1.13. What's a dataframe?

import pandas as pd
import numpy as np

# A DataFrame is a 2-dimensional labeled data structure (like a spreadsheet or SQL table)
# It's the primary data structure in pandas and essential for data analysis

# Think of it as:
# - A table with rows and columns
# - Each column is a pandas Series
# - Columns can have different data types
# - Rows and columns have labels (indices)

# Key features:
# 1. Handles heterogeneous data (different types in different columns)
# 2. Built-in operations for cleaning, transforming, and analyzing data
# 3. Easy filtering, grouping, and aggregation
# 4. Can read/write many file formats (CSV, Excel, SQL, etc.)

# Simple example
data = {
    'country': ['USA', 'Canada', 'Mexico'],
    'population': [331000000, 38000000, 128000000],
    'birth_rate': [12.5, 10.9, 17.6],
    'death_rate': [8.3, 7.8, 6.2]
}

df = pd.DataFrame(data)
print(df)

# Output:
#   country  population  birth_rate  death_rate
# 0     USA   331000000        12.5         8.3
# 1  Canada    38000000        10.9         7.8
# 2  Mexico   128000000        17.6         6.2

# Accessing columns
print(df['country'])         # Single column (Series)
print(df[['country', 'population']])  # Multiple columns (DataFrame)

# Accessing rows
print(df.loc[0])           # Row by label
print(df.iloc[0])          # Row by position

# Basic info
print(df.shape)            # (3, 4) - 3 rows, 4 columns
print(df.columns)          # Column names
print(df.dtypes)           # Data types of each column
print(df.head())           # First 5 rows
print(df.tail())           # Last 5 rows
print(df.info())           # Summary information
print(df.describe())       # Statistical summary

# Why use DataFrames?
# - Easy data manipulation
# - Powerful filtering and grouping
# - Integration with visualization libraries
# - Industry standard for data science in Python


# #### 1.13.1 How can we create an arbitrary dataframe?

import pandas as pd
import numpy as np

# Method 1: From a dictionary (most common)
data = {
    'age': [25, 30, 35, 40],
    'population': [150000, 180000, 165000, 142000],
    'deaths': [75, 108, 132, 170]
}
df1 = pd.DataFrame(data)
print(df1)

# Method 2: From a list of lists
data = [
    ['USA', 331000000, 12.5],
    ['Canada', 38000000, 10.9],
    ['Mexico', 128000000, 17.6]
]
df2 = pd.DataFrame(data, columns=['Country', 'Population', 'Birth Rate'])
print(df2)

# Method 3: From a list of dictionaries
data = [
    {'name': 'Alice', 'age': 25, 'city': 'New York'},
    {'name': 'Bob', 'age': 30, 'city': 'Boston'},
    {'name': 'Charlie', 'age': 35, 'city': 'Chicago'}
]
df3 = pd.DataFrame(data)
print(df3)

# Method 4: From a numpy array
array = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
df4 = pd.DataFrame(array, columns=['A', 'B', 'C'])
print(df4)

# Method 5: Empty DataFrame with columns
df5 = pd.DataFrame(columns=['Year', 'Population', 'Growth Rate'])
print(df5)

# Method 6: From Series
ages = pd.Series([25, 30, 35, 40], name='age')
pops = pd.Series([150000, 180000, 165000, 142000], name='population')
df6 = pd.DataFrame({'age': ages, 'population': pops})
print(df6)

# Demographic example: create vital statistics table
years = list(range(2015, 2021))
births = [125000, 127000, 129000, 131000, 128000, 126000]
deaths = [83000, 84000, 85000, 87000, 88000, 90000]

vital_stats = pd.DataFrame({
    'year': years,
    'births': births,
    'deaths': deaths
})

# Add calculated column
vital_stats['natural_increase'] = vital_stats['births'] - vital_stats['deaths']
print(vital_stats)


# #### 1.13.2 How can we populate a dataframe from a numpy array?

import pandas as pd
import numpy as np

# Method 1: Basic conversion
array = np.array([
    [100000, 12.5, 8.3],
    [150000, 11.8, 7.9],
    [200000, 10.5, 7.2]
])

# Without column names
df1 = pd.DataFrame(array)
print(df1)

# With column names
df2 = pd.DataFrame(array, columns=['Population', 'Birth Rate', 'Death Rate'])
print(df2)

# Method 2: With custom index
df3 = pd.DataFrame(
    array,
    columns=['Population', 'Birth Rate', 'Death Rate'],
    index=['City A', 'City B', 'City C']
)
print(df3)

# Method 3: From 1D arrays (columns)
ages = np.arange(0, 101)  # 0 to 100
males = np.random.randint(5000, 15000, size=101)
females = np.random.randint(5000, 15000, size=101)

population_df = pd.DataFrame({
    'age': ages,
    'male': males,
    'female': females
})
population_df['total'] = population_df['male'] + population_df['female']
print(population_df.head())

# Method 4: Reshape and convert
# Create a 2D array
data = np.random.randn(5, 3)  # 5 rows, 3 columns of random numbers
df4 = pd.DataFrame(
    data,
    columns=['Measure 1', 'Measure 2', 'Measure 3'],
    index=[f'Year {i}' for i in range(2020, 2025)]
)
print(df4)

# Demographic example: Life table from arrays
ages = np.arange(0, 101, 5)  # 0, 5, 10, ..., 100
lx = 100000 * np.exp(-0.01 * ages)  # Survivors
dx = np.diff(lx, prepend=lx[0]) * -1  # Deaths (simplified)
qx = dx / lx  # Mortality rate

life_table = pd.DataFrame({
    'age': ages,
    'lx': lx.astype(int),
    'dx': dx.astype(int),
    'qx': qx
})
print(life_table.head(10))

# Converting back to numpy
array_from_df = life_table.to_numpy()
print(type(array_from_df))  # <class 'numpy.ndarray'>


# #### 1.13.3 How can we filter dataframes?

import pandas as pd
import numpy as np

# Create sample data
data = {
    'country': ['USA', 'Canada', 'Mexico', 'Brazil', 'Argentina', 'Chile'],
    'population': [331000000, 38000000, 128000000, 212000000, 45000000, 19000000],
    'birth_rate': [12.5, 10.9, 17.6, 14.6, 16.5, 13.2],
    'death_rate': [8.3, 7.8, 6.2, 6.7, 7.5, 6.1],
    'continent': ['NA', 'NA', 'NA', 'SA', 'SA', 'SA']
}
df = pd.DataFrame(data)

# Method 1: Boolean indexing (most common)
# Filter for countries with population > 50 million
large_countries = df[df['population'] > 50000000]
print(large_countries)

# Filter for high birth rate
high_birth = df[df['birth_rate'] > 15]
print(high_birth)

# Method 2: Multiple conditions with & (and) and | (or)
# Must use parentheses around each condition!
result = df[(df['population'] > 50000000) & (df['birth_rate'] > 12)]
print(result)

result = df[(df['continent'] == 'NA') | (df['population'] > 100000000)]
print(result)

# Method 3: NOT condition with ~
not_north_america = df[~(df['continent'] == 'NA')]
print(not_north_america)

# Method 4: Using .isin() for multiple values
countries_of_interest = df[df['country'].isin(['USA', 'Brazil', 'Argentina'])]
print(countries_of_interest)

# Method 5: String methods
countries_with_a = df[df['country'].str.contains('a', case=False)]
print(countries_with_a)

# Method 6: Using .query() method (alternative syntax)
result = df.query('population > 50000000 and birth_rate > 12')
print(result)

# Method 7: Using .loc[] for label-based filtering
result = df.loc[df['birth_rate'] > 15, ['country', 'birth_rate']]
print(result)

# Demographic example
vital_data = pd.DataFrame({
    'age': [0, 1, 5, 15, 25, 45, 65, 85],
    'deaths': [500, 50, 20, 35, 45, 120, 450, 850],
    'population': [50000, 48000, 45000, 52000, 58000, 48000, 35000, 12000]
})

# Filter for working age (15-64)
working_age = vital_data[(vital_data['age'] >= 15) & (vital_data['age'] < 65)]
print(working_age)

# Filter for high mortality (>10 per 1000)
vital_data['death_rate'] = (vital_data['deaths'] / vital_data['population']) * 1000
high_mortality = vital_data[vital_data['death_rate'] > 10]
print(high_mortality)


# #### 1.13.4 How can we sort them?

import pandas as pd

# Create sample data
data = {
    'country': ['Brazil', 'USA', 'Mexico', 'Canada', 'Argentina'],
    'population': [212000000, 331000000, 128000000, 38000000, 45000000],
    'birth_rate': [14.6, 12.5, 17.6, 10.9, 16.5],
    'death_rate': [6.7, 8.3, 6.2, 7.8, 7.5]
}
df = pd.DataFrame(data)

# Method 1: Sort by single column (ascending)
sorted_pop = df.sort_values('population')
print(sorted_pop)

# Method 2: Sort descending
sorted_pop_desc = df.sort_values('population', ascending=False)
print(sorted_pop_desc)

# Method 3: Sort by multiple columns
# First by birth_rate (descending), then by population (ascending)
sorted_multi = df.sort_values(['birth_rate', 'population'], ascending=[False, True])
print(sorted_multi)

# Method 4: Sort by index
df_indexed = df.set_index('country')
sorted_index = df_indexed.sort_index()
print(sorted_index)

# Method 5: Sort and reset index
sorted_reset = df.sort_values('population', ascending=False).reset_index(drop=True)
print(sorted_reset)

# Method 6: In-place sorting (modifies original DataFrame)
df_copy = df.copy()
df_copy.sort_values('birth_rate', inplace=True)
print(df_copy)

# Method 7: Sort with NaN values
data_with_nan = {
    'age': [25, 30, 35, np.nan, 45],
    'population': [150000, np.nan, 165000, 142000, 138000]
}
df_nan = pd.DataFrame(data_with_nan)

# NaN values go to end by default
sorted_nan = df_nan.sort_values('population')
print(sorted_nan)

# Put NaN first
sorted_nan_first = df_nan.sort_values('population', na_position='first')
print(sorted_nan_first)

# Demographic example
vital_stats = pd.DataFrame({
    'year': [2020, 2018, 2019, 2017, 2021],
    'births': [126000, 131000, 129000, 127000, 124000],
    'deaths': [90000, 85000, 87000, 84000, 92000]
})

# Calculate natural increase and sort by year
vital_stats['natural_increase'] = vital_stats['births'] - vital_stats['deaths']
vital_stats_sorted = vital_stats.sort_values('year')
print(vital_stats_sorted)

# Find years with highest natural increase
top_years = vital_stats.sort_values('natural_increase', ascending=False)
print(top_years)


# #### 1.13.5 How can we read files from disk with pandas?

import pandas as pd

# Method 1: Read CSV file (most common)
df = pd.read_csv('data.csv')

# With specific parameters
df = pd.read_csv(
    'data.csv',
    sep=',',           # Delimiter (default is comma)
    header=0,          # Row number for column names (0 = first row)
    index_col=0,       # Column to use as row labels
    encoding='utf-8',  # File encoding
    na_values=['NA', 'N/A', '']  # Values to treat as NaN
)

# Method 2: Read Excel file
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Method 3: Read specific columns
df = pd.read_csv('data.csv', usecols=['age', 'population', 'deaths'])

# Method 4: Read with custom data types
df = pd.read_csv('data.csv', dtype={'age': int, 'population': int})

# Method 5: Read file without header
df = pd.read_csv('data.csv', header=None, names=['col1', 'col2', 'col3'])

# Method 6: Skip rows
df = pd.read_csv('data.csv', skiprows=2)  # Skip first 2 rows
df = pd.read_csv('data.csv', skiprows=[0, 2, 5])  # Skip specific rows

# Method 7: Read limited number of rows
df = pd.read_csv('data.csv', nrows=100)  # Only first 100 rows

# Method 8: Read JSON
df = pd.read_json('data.json')

# Method 9: Read from URL
url = 'https://example.com/data.csv'
df = pd.read_csv(url)

# Method 10: Read R data files (.rds)
# Note: requires pyreadr package
# import pyreadr
# result = pyreadr.read_r('data.rds')
# df = result[None]  # None gets the first (and usually only) dataframe

# Demographic example: read population data
# Assuming file structure: year, age, male, female, total
population = pd.read_csv(
    'population_data.csv',
    usecols=['year', 'age', 'male', 'female'],
    dtype={'year': int, 'age': int}
)

# Error handling when reading files
try:
    df = pd.read_csv('might_not_exist.csv')
except FileNotFoundError:
    print("File not found!")
except pd.errors.EmptyDataError:
    print("File is empty!")
except pd.errors.ParserError:
    print("Error parsing file!")


# #### 1.13.6 How can we write files to disk with pandas?

import pandas as pd
import numpy as np

# Create sample DataFrame
data = {
    'country': ['USA', 'Canada', 'Mexico'],
    'population': [331000000, 38000000, 128000000],
    'birth_rate': [12.5, 10.9, 17.6]
}
df = pd.DataFrame(data)

# Method 1: Write to CSV (most common)
df.to_csv('output.csv')

# Without index column
df.to_csv('output.csv', index=False)

# With specific parameters
df.to_csv(
    'output.csv',
    index=False,
    sep=',',           # Delimiter
    encoding='utf-8',  # File encoding
    header=True,       # Include column names
    na_rep='NA'        # How to represent missing values
)

# Method 2: Write to Excel
df.to_excel('output.xlsx', sheet_name='Population Data', index=False)

# Write multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df.to_excel(writer, sheet_name='Countries', index=False)
    df.head(2).to_excel(writer, sheet_name='Top 2', index=False)

# Method 3: Write to JSON
df.to_json('output.json', orient='records', indent=2)
# orient options: 'records', 'index', 'columns', 'values', 'table'

# Method 4: Write specific columns only
df[['country', 'population']].to_csv('population_only.csv', index=False)

# Method 5: Append to existing CSV
df.to_csv('output.csv', mode='a', header=False, index=False)

# Method 6: Write with compression
df.to_csv('output.csv.gz', compression='gzip', index=False)

# Method 7: Write to clipboard (for pasting)
df.to_clipboard(index=False)

# Method 8: Write to HTML
df.to_html('output.html', index=False)

# Demographic example: save life table
ages = np.arange(0, 101, 5)
lx = 100000 * np.exp(-0.01 * ages)
life_table = pd.DataFrame({
    'age': ages,
    'lx': lx.astype(int)
})

# Save with formatted output
life_table.to_csv(
    'life_table.csv',
    index=False,
    float_format='%.2f'  # 2 decimal places for floats
)

# Save multiple DataFrames to Excel
vital_stats = pd.DataFrame({
    'year': [2018, 2019, 2020],
    'births': [130000, 128000, 126000],
    'deaths': [85000, 87000, 90000]
})

with pd.ExcelWriter('demographic_data.xlsx') as writer:
    life_table.to_excel(writer, sheet_name='Life Table', index=False)
    vital_stats.to_excel(writer, sheet_name='Vital Statistics', index=False)

print("Files saved successfully!")


# ### 1.14. How can we import matplotlib?

import matplotlib.pyplot as plt
import numpy as np

# Standard import (most common)
import matplotlib.pyplot as plt

# The 'pyplot' module provides MATLAB-like plotting interface
# 'plt' is the conventional alias

# For Jupyter notebooks, use this magic command to display plots inline:
# %matplotlib inline

# Alternative imports (less common)
from matplotlib import pyplot as plt
import matplotlib as mpl

# Basic plot to verify import
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

plt.plot(x, y)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Simple Line Plot')
plt.show()

# Demographic example
years = np.array([2015, 2016, 2017, 2018, 2019, 2020])
population = np.array([1000000, 1020000, 1041000, 1062000, 1084000, 1106000])

plt.figure(figsize=(10, 6))
plt.plot(years, population, marker='o', linewidth=2, color='blue')
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population Growth 2015-2020')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# #### 1.14.1 How can we plot a numpy array with matplotlib?

import matplotlib.pyplot as plt
import numpy as np

# Simple line plot
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

plt.plot(x, y)
plt.show()

# Plot with customization
plt.figure(figsize=(10, 6))
plt.plot(x, y, color='blue', linewidth=2, marker='o', markersize=8, label='Data')
plt.xlabel('X values', fontsize=12)
plt.ylabel('Y values', fontsize=12)
plt.title('Line Plot Example', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Multiple lines
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='sin(x)', color='blue')
plt.plot(x, y2, label='cos(x)', color='red')
plt.legend()
plt.title('Trigonometric Functions')
plt.show()

# Scatter plot
x = np.random.randn(50)
y = np.random.randn(50)

plt.scatter(x, y, alpha=0.6, s=100, c='purple')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.show()

# Bar plot
categories = np.array(['A', 'B', 'C', 'D'])
values = np.array([23, 45, 56, 78])

plt.bar(categories, values, color='skyblue')
plt.ylabel('Values')
plt.title('Bar Chart')
plt.show()

# Histogram
data = np.random.normal(100, 15, 1000)

plt.hist(data, bins=30, color='green', alpha=0.7, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

# Demographic example: Population projection
years = np.arange(2020, 2051)
initial_pop = 1000000
growth_rate = 0.015
population = initial_pop * (1 + growth_rate) ** (years - 2020)

plt.figure(figsize=(12, 6))
plt.plot(years, population, linewidth=2, color='darkblue', marker='o', markersize=3)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Population', fontsize=12)
plt.title('Population Projection (1.5% annual growth)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Age-specific mortality rates
ages = np.arange(0, 101, 5)
death_rates = 0.001 + 0.0001 * np.exp(0.08 * ages)  # Exponential mortality

plt.figure(figsize=(10, 6))
plt.semilogy(ages, death_rates, linewidth=2, color='red', marker='s')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Death Rate (log scale)', fontsize=12)
plt.title('Age-Specific Mortality Rates', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# #### 1.14.2 How can we plot a pandas dataframe?

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create sample DataFrame
data = {
    'year': [2015, 2016, 2017, 2018, 2019, 2020],
    'births': [130000, 132000, 128000, 126000, 124000, 122000],
    'deaths': [82000, 84000, 86000, 88000, 90000, 92000]
}
df = pd.DataFrame(data)

# Method 1: Using DataFrame.plot() - easiest method
df.plot(x='year', y='births', kind='line', title='Births over Time')
plt.show()

# Method 2: Plot multiple columns
df.plot(x='year', y=['births', 'deaths'], kind='line', 
        title='Births and Deaths', figsize=(10, 6))
plt.ylabel('Count')
plt.show()

# Method 3: Different plot types
# Line plot
df.plot(x='year', y='births', kind='line', marker='o')
plt.show()

# Bar plot
df.plot(x='year', y='births', kind='bar', color='skyblue')
plt.show()

# Scatter plot
df.plot(x='births', y='deaths', kind='scatter', alpha=0.7, s=100)
plt.show()

# Method 4: Using matplotlib directly with DataFrame columns
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['births'], marker='o', label='Births', linewidth=2)
plt.plot(df['year'], df['deaths'], marker='s', label='Deaths', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Vital Statistics')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Method 5: Subplots with DataFrame
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df.plot(x='year', y='births', ax=axes[0], color='blue', marker='o')
axes[0].set_title('Births')
axes[0].set_ylabel('Count')

df.plot(x='year', y='deaths', ax=axes[1], color='red', marker='s')
axes[1].set_title('Deaths')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()

# Demographic example: Age-sex pyramid
age_groups = ['0-14', '15-29', '30-44', '45-59', '60-74', '75+']
male = np.array([250000, 280000, 320000, 290000, 180000, 80000])
female = np.array([240000, 270000, 310000, 295000, 195000, 105000])

pyramid_df = pd.DataFrame({
    'age_group': age_groups,
    'male': -male,  # Negative for left side
    'female': female
})

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(pyramid_df['age_group'], pyramid_df['male'], color='steelblue', label='Male')
ax.barh(pyramid_df['age_group'], pyramid_df['female'], color='coral', label='Female')
ax.set_xlabel('Population')
ax.set_ylabel('Age Group')
ax.set_title('Population Pyramid')
ax.legend()
plt.axvline(x=0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()

# Multiple time series
years = list(range(2010, 2021))
vital_df = pd.DataFrame({
    'year': years,
    'births': 130000 - 600 * np.arange(11),
    'deaths': 80000 + 1000 * np.arange(11),
    'marriages': 50000 - 300 * np.arange(11)
})

vital_df.plot(x='year', y=['births', 'deaths', 'marriages'], 
              kind='line', marker='o', figsize=(12, 6))
plt.title('Vital Events Over Time')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# #### 1.15. How can we import functions from local modules?

# Assume you have a file called 'my_functions.py' with these functions:
# def calculate_death_rate(deaths, population):
#     return (deaths / population) * 1000
# 
# def calculate_growth_rate(births, deaths, population):
#     return ((births - deaths) / population) * 100

# Method 1: Import entire module
import my_functions
rate = my_functions.calculate_death_rate(8300, 1000000)

# Method 2: Import specific functions
from my_functions import calculate_death_rate, calculate_growth_rate
rate = calculate_death_rate(8300, 1000000)

# Method 3: Import with alias
import my_functions as mf
rate = mf.calculate_death_rate(8300, 1000000)

# Method 4: Import everything (not recommended)
from my_functions import *
rate = calculate_death_rate(8300, 1000000)

# Importing from subdirectories
# If you have: src/utilities/helpers.py
from src.utilities.helpers import some_function

# Or with package structure:
from src.utilities import helpers
helpers.some_function()

# Example with PyCCM modules (from the parent src directory)
# Assuming we're in a script in the same project
import sys
sys.path.append('../src')  # Add src directory to path

from src.data_loaders import load_all_data
from src.mortality import make_life_table
from src.fertility import make_fertility

# Or relative imports (if in a package)
# from .helpers import calculate_rate
# from ..data import load_data

# Practical example structure:
# project/
#   ├── src/
#   │   ├── __init__.py
#   │   ├── data_loaders.py
#   │   ├── mortality.py
#   │   └── helpers.py
#   └── analysis/
#       └── my_analysis.py

# From my_analysis.py:
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loaders import load_conteos
from mortality import make_life_table
from helpers import calculate_death_rate

# Load data
df = load_conteos('data/conteos.rds')

# Calculate life table
lt = make_life_table(df)

# Calculate custom rate
rate = calculate_death_rate(8300, 1000000)
print(f"Death rate: {rate:.2f} per 1,000")


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

# i. Create three numpy arrays with different starting populations
years = np.arange(0, 51)  # 0 to 50 years

# Three scenarios with different growth rates
population_low = np.zeros(len(years))
population_medium = np.zeros(len(years))
population_high = np.zeros(len(years))

# Starting values
initial_population = 100000
population_low[0] = initial_population
population_medium[0] = initial_population
population_high[0] = initial_population

# Growth rates
growth_rate_low = 0.01    # 1% annual growth
growth_rate_medium = 0.02  # 2% annual growth
growth_rate_high = 0.03    # 3% annual growth

# ii. Sequentially multiply by the exponent (exponential growth)
for t in range(1, len(years)):
    population_low[t] = population_low[0] * (1 + growth_rate_low) ** t
    population_medium[t] = population_medium[0] * (1 + growth_rate_medium) ** t
    population_high[t] = population_high[0] * (1 + growth_rate_high) ** t

# Alternative vectorized approach (more Pythonic)
population_low_v2 = initial_population * (1 + growth_rate_low) ** years
population_medium_v2 = initial_population * (1 + growth_rate_medium) ** years
population_high_v2 = initial_population * (1 + growth_rate_high) ** years

# iii. Plot the three different values
plt.figure(figsize=(12, 7))

plt.plot(years, population_low, linewidth=2, marker='o', markersize=4, 
         label=f'Low growth (r={growth_rate_low*100:.0f}%)', color='green')
plt.plot(years, population_medium, linewidth=2, marker='s', markersize=4,
         label=f'Medium growth (r={growth_rate_medium*100:.0f}%)', color='blue')
plt.plot(years, population_high, linewidth=2, marker='^', markersize=4,
         label=f'High growth (r={growth_rate_high*100:.0f}%)', color='red')

plt.xlabel('Years', fontsize=12)
plt.ylabel('Population', fontsize=12)
plt.title(f'Exponential Population Growth from {initial_population:,}', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print final populations
print(f"After {years[-1]} years:")
print(f"  Low growth:    {population_low[-1]:,.0f}")
print(f"  Medium growth: {population_medium[-1]:,.0f}")
print(f"  High growth:   {population_high[-1]:,.0f}")


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

# Add the src directory to path to import PyCCM modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# i. Load the conteos.rds file using the data_loaders module
from src.data_loaders import load_conteos

# Load the data (assuming the file is in the data directory)
data_path = os.path.join(os.path.dirname(__file__), '../../data/conteos.rds')
df = load_conteos(data_path)

print(f"Data loaded. Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ii. Filter for total national data, EEVV, and year 2018
# Assuming 'EEVV' refers to a specific region/state code
# and there's a column indicating national/total data
df_filtered = df[
    (df['year'] == 2018) & 
    (df['state'] == 'EEVV')  # Adjust column name as needed
].copy()

print(f"After filtering for 2018 and EEVV: {df_filtered.shape}")

# iii. Drop values where age is NaN
df_filtered = df_filtered.dropna(subset=['age'])
print(f"After dropping NaN ages: {df_filtered.shape}")

# iv. Add the population count for males to females
# Assuming columns are named 'pop_male' and 'pop_female' or similar
# Adjust column names based on actual data structure
df_filtered['total_population'] = df_filtered['pop_male'] + df_filtered['pop_female']

# v. Add the death counts for males to females
df_filtered['total_deaths'] = df_filtered['deaths_male'] + df_filtered['deaths_female']

# vi. Divide death counts by population values (death rate)
df_filtered['m_x'] = df_filtered['total_deaths'] / df_filtered['total_population']

# Sort by age for plotting
df_filtered = df_filtered.sort_values('age')

# vii. Plot death rates (m_x)
plt.figure(figsize=(12, 6))
plt.plot(df_filtered['age'], df_filtered['m_x'], 
         linewidth=2, marker='o', markersize=4, color='darkred')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Death Rate (m_x)', fontsize=12)
plt.title('Age-Specific Death Rates - Mexico 2018 (EEVV)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# viii. Plot logged death rates
plt.figure(figsize=(12, 6))
plt.semilogy(df_filtered['age'], df_filtered['m_x'], 
             linewidth=2, marker='o', markersize=4, color='navy')
# Alternative: plt.plot(df_filtered['age'], np.log(df_filtered['m_x']))
plt.xlabel('Age', fontsize=12)
plt.ylabel('Death Rate (m_x) - Log Scale', fontsize=12)
plt.title('Age-Specific Death Rates (Log Scale) - Mexico 2018 (EEVV)', fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nDeath Rate Statistics:")
print(f"  Mean: {df_filtered['m_x'].mean():.6f}")
print(f"  Median: {df_filtered['m_x'].median():.6f}")
print(f"  Min: {df_filtered['m_x'].min():.6f} (age {df_filtered.loc[df_filtered['m_x'].idxmin(), 'age']:.0f})")
print(f"  Max: {df_filtered['m_x'].max():.6f} (age {df_filtered.loc[df_filtered['m_x'].idxmax(), 'age']:.0f})")

# Display first few rows
print("\nFirst few rows of processed data:")
print(df_filtered[['age', 'total_population', 'total_deaths', 'm_x']].head(10))
