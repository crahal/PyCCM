# data_loaders.py - Test Summary

## File Overview

The `data_loaders.py` file is a comprehensive data loading and configuration utility module for PyCCM (Python Cohort Component Model), a demographic projection system. It handles:

1. **Configuration Management**: Loading and merging YAML config files with default settings
2. **Data Loading**: Reading R data files (.rds) and CSV files
3. **Data Processing**: Handling missing age data, census omission corrections
4. **Results Collection**: Gathering fertility and mortality projection results

## Functions Explained

### Configuration Functions

- **`_get_base_dir()`**: Returns the directory containing this script
- **`return_default_config()`**: Returns comprehensive default configuration with paths, projections, fertility, mortality, and other settings
- **`_resolve(ROOT_DIR, p)`**: Converts relative paths to absolute paths
- **`_deep_merge(dst, src)`**: Recursively merges two dictionaries (for combining user config with defaults)
- **`_load_config(ROOT_DIR, path)`**: Loads YAML configuration file and merges with defaults

### Data Processing Functions

- **`allocate_and_drop_missing_age(df)`**: Redistributes rows with missing age data proportionally across known ages, then removes missing rows
- **`read_rds_file(file_path)`**: Reads R data files using pyreadr library
- **`load_all_data(data_dir)`**: Loads the main conteos.rds data file
- **`correct_valor_for_omission(df, sample_type, distribution, ...)`**: Adjusts census values based on omission levels using various statistical distributions (uniform, pert, beta, normal)

### Results Collection Functions

- **`get_lifetables_ex(DPTO)`**: Collects life expectancy data from multiple distribution result directories
- **`get_fertility()`**: Collects age-specific fertility rates and calculates total fertility rates from result files

## Test Suite

Created comprehensive unit tests with **38 test cases** covering:

### Test Classes:
1. **TestGetBaseDir** (3 tests) - Directory path functions
2. **TestReturnDefaultConfig** (5 tests) - Default configuration structure
3. **TestResolve** (2 tests) - Path resolution
4. **TestDeepMerge** (4 tests) - Dictionary merging logic
5. **TestLoadConfig** (4 tests) - YAML config loading
6. **TestAllocateAndDropMissingAge** (3 tests) - Missing age data handling
7. **TestReadRdsFile** (2 tests) - R data file reading
8. **TestLoadAllData** (2 tests) - Main data loading
9. **TestCorrectValorForOmission** (11 tests) - Census omission corrections
10. **TestGetLifetablesEx** (1 test) - Life table collection
11. **TestGetFertility** (1 test) - Fertility data collection

### Key Test Features:
- Uses mocking to avoid dependencies on actual data files
- Tests edge cases (missing data, invalid inputs)
- Tests all statistical distributions (uniform, pert, beta, normal)
- Tests proportional allocation logic
- Tests configuration merging and path resolution
- All 38 tests passing ✅

## Running Tests

```bash
# Run all tests
python3 -m pytest tests/test_data_loaders.py -v

# Run with coverage
python3 -m pytest tests/test_data_loaders.py --cov=src/data_loaders --cov-report=html
```

## Dependencies
- pandas
- numpy
- pyyaml
- pyreadr (for R data files)
- pytest (for testing)
