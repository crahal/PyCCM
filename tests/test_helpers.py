"""
Comprehensive test suite for helpers.py

Tests cover:
- Age scaffolding utilities
- CSV parsing and column matching
- Fertility/TFR computation and smoothing
- Alignment and reindexing functions
- Specialized data preprocessing
- Demographic plausibility checks
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.helpers import (
    _single_year_bins,
    _bin_width,
    _widths_from_index,
    _find_col,
    get_midpoint_weights,
    _tfr_from_asfr_df,
    _exp_tfr,
    _logistic_tfr,
    _smooth_tfr,
    fill_missing_age_bins,
    _ridx,
    _collapse_defunciones_01_24_to_04
)


class TestAgeScaffolding:
    """Test age bin utilities"""
    
    def test_single_year_bins(self):
        """Test generation of single-year age bins"""
        bins = _single_year_bins()
        
        assert len(bins) == 91  # 0-89 plus 90+
        assert bins[0] == '0'
        assert bins[50] == '50'
        assert bins[89] == '89'
        assert bins[90] == '90+'
        
    def test_bin_width_single_year(self):
        """Test bin width calculation for single-year bins"""
        assert _bin_width('25') == 1
        assert _bin_width('0') == 1
        assert _bin_width('100') == 1
        
    def test_bin_width_age_range(self):
        """Test bin width calculation for age ranges (closed intervals)"""
        # For "lo-hi", width = hi - lo + 1
        assert _bin_width('0-4') == 5.0  # 4 - 0 + 1
        assert _bin_width('5-9') == 5.0  # 9 - 5 + 1
        assert _bin_width('15-19') == 5.0  # 19 - 15 + 1
        assert _bin_width('80-84') == 5.0  # 84 - 80 + 1
        
    def test_bin_width_open_ended(self):
        """Test bin width for open-ended bins"""
        # Open-ended bins return 5 by default
        assert _bin_width('85+') == 5
        assert _bin_width('100+') == 5
        
    def test_widths_from_index(self):
        """Test width calculation from full index"""
        index = pd.Index(['0-4', '5-9', '10-14', '15-19', '20+'])
        widths = _widths_from_index(index)
        
        assert len(widths) == 5
        assert widths[0] == 5
        assert widths[1] == 5
        assert widths[2] == 5
        assert widths[3] == 5
        assert widths[4] == 5  # Open-ended
        
    def test_widths_from_index_mixed(self):
        """Test width calculation with mixed bin sizes"""
        index = pd.Index(['0', '1', '2-4', '5-9', '10-14'])
        widths = _widths_from_index(index)
        
        assert widths[0] == 1.0  # Single year
        assert widths[1] == 1.0  # Single year
        assert widths[2] == 3.0  # 2-4: 4-2+1
        assert widths[3] == 5.0  # 5-9: 9-5+1
        assert widths[4] == 5.0  # 10-14: 14-10+1


class TestCSVHelpers:
    """Test CSV parsing utilities"""
    
    def test_find_col_exact_match(self):
        """Test finding column with exact match"""
        df = pd.DataFrame(columns=['age', 'sex', 'population'])
        
        # _find_col takes a list of substrings
        assert _find_col(df, ['age']) == 'age'
        assert _find_col(df, ['sex']) == 'sex'
        assert _find_col(df, ['population']) == 'population'
        
    def test_find_col_case_insensitive(self):
        """Test case-insensitive column matching"""
        df = pd.DataFrame(columns=['Age', 'SEX', 'Population'])
        
        assert _find_col(df, ['age']) == 'Age'
        assert _find_col(df, ['sex']) == 'SEX'
        assert _find_col(df, ['population']) == 'Population'
        
    def test_find_col_with_underscores(self):
        """Test column matching with multiple required substrings"""
        df = pd.DataFrame(columns=['age_group', 'sex_id', 'pop_count'])
        
        # All substrings must be present
        assert _find_col(df, ['age']) == 'age_group'
        assert _find_col(df, ['sex']) == 'sex_id'
        
    def test_find_col_partial_match(self):
        """Test partial column name matching with multiple substrings"""
        df = pd.DataFrame(columns=['total_population', 'male_pop', 'female_pop'])
        
        # Should find columns containing all search terms
        assert _find_col(df, ['total']) == 'total_population'
        assert _find_col(df, ['male']) == 'male_pop'
        
    def test_find_col_not_found(self):
        """Test behavior when column not found"""
        df = pd.DataFrame(columns=['age', 'sex'])
        
        # Returns None when not found
        result = _find_col(df, ['nonexistent'])
        assert result is None
            
    def test_get_midpoint_weights_basic(self):
        """Test basic midpoint weight loading from file"""
        # get_midpoint_weights expects DPTO_NOMBRE column and a value column
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("DPTO_NOMBRE,midpoint_weight\nDpto1,0.5\nDpto2,0.7\nDpto3,0.9\n")
            temp_path = f.name
        
        try:
            weights = get_midpoint_weights(temp_path)
            
            assert isinstance(weights, dict)
            assert len(weights) == 3
            assert 'Dpto1' in weights
            assert 'Dpto2' in weights
            assert 'Dpto3' in weights
            assert weights['Dpto1'] == 0.5
            assert weights['Dpto2'] == 0.7
            assert weights['Dpto3'] == 0.9
        finally:
            os.unlink(temp_path)


class TestFertilityTFRFunctions:
    """Test TFR computation and smoothing"""
    
    def test_tfr_from_asfr_df_basic(self):
        """Test TFR calculation from ASFR DataFrame"""
        # Function expects a DataFrame with column 'asfr' and age labels as index
        df = pd.DataFrame({
            'asfr': [0.050, 0.120, 0.150, 0.100, 0.050, 0.010, 0.001]
        }, index=['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49'])
        
        tfr = _tfr_from_asfr_df(df)
        
        # TFR = sum(ASFR * width) where width=5 for all groups (closed intervals)
        expected_tfr = (0.050 + 0.120 + 0.150 + 0.100 + 0.050 + 0.010 + 0.001) * 5
        
        assert isinstance(tfr, float)
        assert tfr == pytest.approx(expected_tfr, rel=1e-6)
        
    def test_tfr_from_asfr_df_single_year(self):
        """Test TFR calculation with single-year age groups"""
        # Create single-year ASFR data
        df = pd.DataFrame({
            'asfr': [0.05] * 35
        }, index=[str(age) for age in range(15, 50)])
        
        tfr = _tfr_from_asfr_df(df)
        
        # TFR = sum(ASFR * 1) for single-year data
        expected_tfr = 0.05 * 35  # 35 age groups * width 1
        
        assert isinstance(tfr, float)
        assert tfr == pytest.approx(expected_tfr, rel=1e-6)
        
    def test_exp_tfr_convergence(self):
        """Test exponential TFR convergence"""
        initial = 2.5
        target = 2.1
        years = 50
        
        # Test at different steps
        tfr_start = _exp_tfr(initial, target, years, step=0)
        tfr_mid = _exp_tfr(initial, target, years, step=25)
        tfr_end = _exp_tfr(initial, target, years, step=50)
        
        assert tfr_start == pytest.approx(initial, rel=1e-6)
        assert tfr_end == pytest.approx(target, rel=0.01)  # Should be close to target
        assert initial > tfr_mid > target  # Monotonic decrease
        
    def test_exp_tfr_increasing(self):
        """Test exponential TFR with increasing trend"""
        initial = 1.5
        target = 2.1
        years = 30
        
        tfr_start = _exp_tfr(initial, target, years, step=0)
        tfr_mid = _exp_tfr(initial, target, years, step=15)
        tfr_end = _exp_tfr(initial, target, years, step=30)
        
        assert tfr_start == pytest.approx(initial, rel=1e-6)
        assert tfr_end == pytest.approx(target, rel=0.01)
        assert initial < tfr_mid < target  # Monotonic increase
        
    def test_exp_tfr_demographic_bounds(self):
        """Test that exponential TFR stays within demographic bounds"""
        initial = 2.5
        target = 2.1
        years = 50
        
        # Sample several points
        tfr_values = [_exp_tfr(initial, target, years, step=t) for t in range(0, 51, 10)]
        
        # All values should be within plausible TFR range (0.3 to 10.0)
        assert all(0.3 <= tfr <= 10.0 for tfr in tfr_values)
        
    def test_logistic_tfr_convergence(self):
        """Test logistic TFR convergence"""
        initial = 3.0
        target = 2.1
        years = 50
        
        tfr_start = _logistic_tfr(initial, target, years, step=0)
        tfr_end = _logistic_tfr(initial, target, years, step=50)
        
        assert tfr_start == pytest.approx(initial, rel=1e-6)
        assert tfr_end == pytest.approx(target, rel=0.05)  # Close to target
        
    def test_logistic_tfr_s_curve(self):
        """Test logistic TFR produces S-curve shape"""
        initial = 4.0
        target = 2.1
        years = 50
        
        # Sample points throughout the curve
        tfr_values = [_logistic_tfr(initial, target, years, step=t) for t in range(0, 51, 5)]
        
        # Should be monotonically decreasing
        for i in range(len(tfr_values) - 1):
            assert tfr_values[i] >= tfr_values[i+1]
        
    def test_logistic_tfr_demographic_bounds(self):
        """Test that logistic TFR stays within demographic bounds"""
        initial = 5.0
        target = 1.8
        years = 60
        
        # Sample several points
        tfr_values = [_logistic_tfr(initial, target, years, step=t) for t in range(0, 61, 10)]
        
        # All values should be within plausible TFR range
        assert all(0.3 <= tfr <= 10.0 for tfr in tfr_values)
        
    def test_smooth_tfr_exponential(self):
        """Test smooth_tfr with exponential method"""
        initial = 2.5
        target = 2.1
        years = 50
        
        tfr_start = _smooth_tfr(initial, target, years, step=0, kind='exp')
        tfr_end = _smooth_tfr(initial, target, years, step=50, kind='exp')
        
        assert tfr_start == pytest.approx(initial, rel=1e-6)
        assert tfr_end == pytest.approx(target, rel=0.01)
        
    def test_smooth_tfr_logistic(self):
        """Test smooth_tfr with logistic method"""
        initial = 3.5
        target = 2.1
        years = 40
        
        tfr_start = _smooth_tfr(initial, target, years, step=0, kind='logistic')
        tfr_end = _smooth_tfr(initial, target, years, step=40, kind='logistic')
        
        assert tfr_start == pytest.approx(initial, rel=1e-6)
        assert tfr_end == pytest.approx(target, rel=0.05)
        
    def test_smooth_tfr_invalid_method(self):
        """Test smooth_tfr raises error for invalid method"""
        with pytest.raises(ValueError, match="Unknown TFR path kind"):
            _smooth_tfr(2.5, 2.1, years=50, step=25, kind='invalid')
            
    def test_smooth_tfr_demographic_validation(self):
        """Test that smooth_tfr validates demographic plausibility"""
        # Test with extreme values
        initial = 8.0
        target = 1.5
        years = 50
        
        for kind in ['exp', 'logistic']:
            # Sample several points
            tfr_values = [_smooth_tfr(initial, target, years, step=t, kind=kind) 
                         for t in range(0, 51, 10)]
            
            # Should stay within demographic bounds
            assert all(0.3 <= tfr <= 10.0 for tfr in tfr_values), \
                f"{kind}: TFR outside bounds"


class TestAlignmentFunctions:
    """Test alignment and reindexing utilities"""
    
    def test_fill_missing_age_bins_basic(self):
        """Test filling missing age bins with zeros"""
        # fill_missing_age_bins expects a Series, not DataFrame
        s = pd.Series([100, 150, 200], index=['0-4', '10-14', '20-24'])
        
        expected_ages = ['0-4', '5-9', '10-14', '15-19', '20-24']
        result = fill_missing_age_bins(s, expected_ages)
        
        assert len(result) == 5
        assert result.loc['0-4'] == 100
        assert result.loc['5-9'] == 0.0  # Filled
        assert result.loc['10-14'] == 150
        assert result.loc['15-19'] == 0.0  # Filled
        assert result.loc['20-24'] == 200
        
    def test_fill_missing_age_bins_complete(self):
        """Test with already complete age bins"""
        s = pd.Series([100, 150, 200], index=['0-4', '5-9', '10-14'])
        
        expected_ages = ['0-4', '5-9', '10-14']
        result = fill_missing_age_bins(s, expected_ages)
        
        # Should return with same values
        assert len(result) == 3
        assert all(result == s)
        
    def test_fill_missing_age_bins_subset(self):
        """Test filling when some bins are missing"""
        s = pd.Series([100, 200], index=['0-4', '10-14'])
        
        expected_ages = ['0-4', '5-9', '10-14']
        result = fill_missing_age_bins(s, expected_ages)
        
        assert len(result) == 3
        assert result.loc['0-4'] == 100
        assert result.loc['5-9'] == 0.0
        assert result.loc['10-14'] == 200
        
    def test_ridx_basic(self):
        """Test basic reindexing with _ridx"""
        # _ridx takes a Series-like object or list
        s = pd.Series([100, 150, 200], index=['0-4', '5-9', '10-14'])
        
        new_index = ['0-4', '5-9', '10-14', '15-19']
        result = _ridx(s, new_index)
        
        assert len(result) == 4
        assert result.loc['0-4'] == 100
        assert result.loc['5-9'] == 150
        assert result.loc['10-14'] == 200
        assert result.loc['15-19'] == 0.0
        
    def test_ridx_with_duplicates(self):
        """Test _ridx handles duplicates by summing"""
        s = pd.Series([50, 50, 150], index=['0-4', '0-4', '5-9'])
        
        new_index = ['0-4', '5-9', '10-14']
        result = _ridx(s, new_index)
        
        # Duplicates should be summed
        assert result.loc['0-4'] == 100.0  # 50 + 50
        assert result.loc['5-9'] == 150.0
        assert result.loc['10-14'] == 0.0
        
    def test_ridx_subset_index(self):
        """Test _ridx with subset of original index"""
        s = pd.Series([100, 150, 200, 250], index=['0-4', '5-9', '10-14', '15-19'])
        
        new_index = ['5-9', '10-14']
        result = _ridx(s, new_index)
        
        assert len(result) == 2
        assert result.loc['5-9'] == 150.0
        assert result.loc['10-14'] == 200.0


class TestSpecializedFunctions:
    """Test specialized preprocessing functions"""
    
    def test_collapse_defunciones_basic(self):
        """Test collapsing death data from 0-1, 2-4 to 0-4"""
        df = pd.DataFrame({
            'EDAD': ['0-1', '2-4', '5-9', '10-14'],
            'VALOR': [10, 15, 20, 25],
            'DPTO_NOMBRE': ['Dept1']*4,
            'DPTO_CODIGO': ['01']*4,
            'ANO': [2020]*4,
            'SEXO': ['M']*4,
            'VARIABLE': ['defunciones']*4,
            'FUENTE': ['src1']*4
        })
        
        result = _collapse_defunciones_01_24_to_04(df)
        
        # Should have one less row (0-1 and 2-4 combined)
        assert len(result) == 3
        
        # 0-4 should be sum of 0-1 and 2-4
        collapsed_row = result[result['EDAD'] == '0-4']
        assert len(collapsed_row) == 1
        assert collapsed_row['VALOR'].iloc[0] == 25  # 10 + 15
        
        # Other rows unchanged
        assert len(result[result['EDAD'] == '5-9']) == 1
        assert result[result['EDAD'] == '5-9']['VALOR'].iloc[0] == 20
        
    def test_collapse_defunciones_no_01_24(self):
        """Test collapse when 0-1 and 2-4 don't exist"""
        df = pd.DataFrame({
            'EDAD': ['0-4', '5-9', '10-14'],
            'VALOR': [30, 20, 25],
            'DPTO_NOMBRE': ['Dept1']*3,
            'DPTO_CODIGO': ['01']*3,
            'ANO': [2020]*3,
            'SEXO': ['M']*3,
            'VARIABLE': ['defunciones']*3,
            'FUENTE': ['src1']*3
        })
        
        result = _collapse_defunciones_01_24_to_04(df)
        
        # Should return unchanged
        assert len(result) == 3
        assert result[result['EDAD'] == '0-4']['VALOR'].iloc[0] == 30
        
    def test_collapse_defunciones_only_01(self):
        """Test collapse with only 0-1 (no 2-4)"""
        df = pd.DataFrame({
            'EDAD': ['0-1', '5-9', '10-14'],
            'VALOR': [10, 20, 25],
            'DPTO_NOMBRE': ['Dept1']*3,
            'DPTO_CODIGO': ['01']*3,
            'ANO': [2020]*3,
            'SEXO': ['M']*3,
            'VARIABLE': ['defunciones']*3,
            'FUENTE': ['src1']*3
        })
        
        result = _collapse_defunciones_01_24_to_04(df)
        
        # 0-1 should become 0-4 with same value
        assert len(result) == 3
        assert result[result['EDAD'] == '0-4']['VALOR'].iloc[0] == 10
        
    def test_collapse_defunciones_preserves_columns(self):
        """Test that collapse preserves all other columns"""
        df = pd.DataFrame({
            'EDAD': ['0-1', '2-4', '5-9'],
            'VALOR': [10, 15, 20],
            'DPTO_NOMBRE': ['Dept1']*3,
            'DPTO_CODIGO': ['01']*3,
            'ANO': [2020, 2020, 2020],
            'SEXO': ['M', 'M', 'M'],
            'VARIABLE': ['defunciones']*3,
            'FUENTE': ['src1']*3
        })
        
        result = _collapse_defunciones_01_24_to_04(df)
        
        assert 'ANO' in result.columns
        assert 'SEXO' in result.columns
        collapsed_row = result[result['EDAD'] == '0-4']
        assert collapsed_row['ANO'].iloc[0] == 2020
        assert collapsed_row['SEXO'].iloc[0] == 'M'


class TestDemographicValidation:
    """Test demographic plausibility checks in helper functions"""
    
    def test_tfr_extreme_values_exponential(self):
        """Test exponential TFR handles extreme starting values"""
        # Very high initial TFR
        tfr_values_high = [_exp_tfr(9.0, 2.1, years=50, step=t) for t in range(0, 51, 10)]
        assert all(0.3 <= tfr <= 10.0 for tfr in tfr_values_high)
        
        # Very low initial TFR
        tfr_values_low = [_exp_tfr(0.5, 2.1, years=50, step=t) for t in range(0, 51, 10)]
        assert all(0.3 <= tfr <= 10.0 for tfr in tfr_values_low)
        
    def test_tfr_extreme_values_logistic(self):
        """Test logistic TFR handles extreme starting values"""
        # Very high initial TFR
        tfr_values_high = [_logistic_tfr(9.5, 2.1, years=60, step=t) for t in range(0, 61, 10)]
        assert all(0.3 <= tfr <= 10.0 for tfr in tfr_values_high)
        
        # Very low initial TFR  
        tfr_values_low = [_logistic_tfr(0.4, 2.1, years=60, step=t) for t in range(0, 61, 10)]
        assert all(0.3 <= tfr <= 10.0 for tfr in tfr_values_low)
        
    def test_tfr_rapid_convergence(self):
        """Test TFR convergence with rapid parameters"""
        # Sample several points with convergence
        tfr_values = [_smooth_tfr(4.0, 2.1, years=30, step=t, kind='exp') 
                     for t in range(0, 31, 5)]
        
        # Should be monotonically decreasing
        for i in range(len(tfr_values) - 1):
            assert tfr_values[i] >= tfr_values[i+1], "TFR should decrease monotonically"
        
        # Values should be in reasonable range
        assert all(0.3 <= tfr <= 10.0 for tfr in tfr_values)
        
    def test_asfr_demographic_structure(self):
        """Test that TFR calculation handles realistic ASFR age structure"""
        # Peak fertility at 25-29 (realistic)
        df = pd.DataFrame({
            'asfr': [0.030, 0.110, 0.160, 0.120, 0.060, 0.015, 0.002]
        }, index=['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49'])
        
        tfr = _tfr_from_asfr_df(df)
        
        # TFR should be in plausible range
        assert 0.3 <= tfr <= 10.0
        # For this data, should be around 2.4-2.5
        assert 2.0 <= tfr <= 3.0
        
    def test_age_bin_consistency(self):
        """Test that age bin widths are calculated consistently"""
        # Standard 5-year bins
        standard_bins = ['0-4', '5-9', '10-14', '15-19', '20-24']
        widths = [_bin_width(b) for b in standard_bins]
        
        assert all(w == 5.0 for w in widths)
        
    def test_fill_missing_preserves_totals(self):
        """Test that filling missing bins doesn't alter existing totals"""
        s = pd.Series([100, 150, 200], index=['0-4', '10-14', '20-24'])
        
        original_sum = s.sum()
        
        expected_ages = ['0-4', '5-9', '10-14', '15-19', '20-24']
        result = fill_missing_age_bins(s, expected_ages)
        
        # Total should remain same (zeros added for missing bins)
        assert result.sum() == original_sum
