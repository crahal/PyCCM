# tests/test_migration.py
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from migration import create_migration_frame, EDAD_ORDER


# ============================================================================
# Test Fixtures
# ============================================================================
@pytest.fixture
def sample_conteos():
    """Create sample conteos DataFrame for testing."""
    data = []
    
    # Two departments, one year, two age groups, two sexes
    dptos = ['05', '11']  # Antioquia, Bogotá
    ages = ['0-4', '5-9']
    sexes = [1, 2]
    
    for dpto in dptos:
        for age in ages:
            for sex in sexes:
                # Population
                pop_value = 50000 if dpto == '05' else 80000
                data.append({
                    'DPTO': dpto,
                    'ANO': 2018,
                    'EDAD': age,
                    'SEXO': sex,
                    'VARIABLE': 'poblacion_total',
                    'VALOR': pop_value
                })
                
                # Immigration
                inmig_value = 100 if age == '0-4' else 200
                data.append({
                    'DPTO': dpto,
                    'ANO': 2018,
                    'EDAD': age,
                    'SEXO': sex,
                    'VARIABLE': 'flujo_inmigracion',
                    'VALOR': inmig_value
                })
                
                # Emigration
                emig_value = 50 if age == '0-4' else 150
                data.append({
                    'DPTO': dpto,
                    'ANO': 2018,
                    'EDAD': age,
                    'SEXO': sex,
                    'VARIABLE': 'flujo_emigracion',
                    'VALOR': emig_value
                })
    
    return pd.DataFrame(data)


@pytest.fixture
def multi_year_conteos():
    """Create multi-year conteos for testing."""
    data = []
    
    for year in [2016, 2017, 2018]:
        for age in ['0-4', '5-9']:
            for sex in [1, 2]:
                data.append({
                    'DPTO': '05',
                    'ANO': year,
                    'EDAD': age,
                    'SEXO': sex,
                    'VARIABLE': 'poblacion_total',
                    'VALOR': 100000
                })
                data.append({
                    'DPTO': '05',
                    'ANO': year,
                    'EDAD': age,
                    'SEXO': sex,
                    'VARIABLE': 'flujo_inmigracion',
                    'VALOR': 500
                })
                data.append({
                    'DPTO': '05',
                    'ANO': year,
                    'EDAD': age,
                    'SEXO': sex,
                    'VARIABLE': 'flujo_emigracion',
                    'VALOR': 300
                })
    
    return pd.DataFrame(data)


# ============================================================================
# Test Basic Functionality
# ============================================================================
class TestBasicFunctionality:
    """Test core migration frame creation."""
    
    def test_basic_structure(self, sample_conteos):
        """Test that basic structure is created correctly."""
        result = create_migration_frame(sample_conteos, year=2018)
        
        # Check required columns
        required_cols = ['ANO', 'EDAD', 'SEXO', 'inmigracion_F', 'emigracion_F',
                        'net_migration', 'poblacion_total', 'net_mig_rate']
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"
        
        # Check shape: 2 ages × 2 sexes = 4 rows
        assert len(result) == 4
    
    def test_aggregation_to_national(self, sample_conteos):
        """Test that departments are aggregated to national level."""
        result = create_migration_frame(sample_conteos, year=2018)
        
        # Population should be sum across departments
        # Dept 05: 50,000, Dept 11: 80,000 → Total: 130,000
        pop_0_4_sex1 = result[(result['EDAD'] == '0-4') & (result['SEXO'] == 1)]['poblacion_total'].values[0]
        assert pop_0_4_sex1 == 130000
    
    def test_immigration_aggregation(self, sample_conteos):
        """Test immigration flows are summed correctly."""
        result = create_migration_frame(sample_conteos, year=2018)
        
        # Each dept contributes 100 for age 0-4 → Total: 200
        inmig_0_4_sex1 = result[(result['EDAD'] == '0-4') & (result['SEXO'] == 1)]['inmigracion_F'].values[0]
        assert inmig_0_4_sex1 == 200
    
    def test_emigration_aggregation(self, sample_conteos):
        """Test emigration flows are summed correctly."""
        result = create_migration_frame(sample_conteos, year=2018)
        
        # Each dept contributes 50 for age 0-4 → Total: 100
        emig_0_4_sex1 = result[(result['EDAD'] == '0-4') & (result['SEXO'] == 1)]['emigracion_F'].values[0]
        assert emig_0_4_sex1 == 100
    
    def test_net_migration_calculation(self, sample_conteos):
        """Test net migration is correctly calculated."""
        result = create_migration_frame(sample_conteos, year=2018)
        
        # Immigration 200 - Emigration 100 = Net 100
        net_0_4_sex1 = result[(result['EDAD'] == '0-4') & (result['SEXO'] == 1)]['net_migration'].values[0]
        assert net_0_4_sex1 == 100
    
    def test_net_mig_rate_calculation(self, sample_conteos):
        """Test net migration rate is correctly calculated."""
        result = create_migration_frame(sample_conteos, year=2018)
        
        # Net 100 / Population 130,000 = 0.000769...
        rate_0_4_sex1 = result[(result['EDAD'] == '0-4') & (result['SEXO'] == 1)]['net_mig_rate'].values[0]
        expected_rate = 100 / 130000
        assert abs(rate_0_4_sex1 - expected_rate) < 1e-6


# ============================================================================
# Test Year Filtering
# ============================================================================
class TestYearFiltering:
    """Test year filtering functionality."""
    
    def test_single_year_filter(self, multi_year_conteos):
        """Test filtering to single year."""
        result = create_migration_frame(multi_year_conteos, year=2018)
        
        # Should only have 2018 data
        assert result['ANO'].nunique() == 1
        assert result['ANO'].iloc[0] == 2018
    
    def test_all_years(self, multi_year_conteos):
        """Test returning all years when year=None."""
        result = create_migration_frame(multi_year_conteos, year=None)
        
        # Should have 3 years
        assert result['ANO'].nunique() == 3
        assert set(result['ANO'].unique()) == {2016, 2017, 2018}
    
    def test_nonexistent_year(self, multi_year_conteos):
        """Test filtering to year that doesn't exist."""
        result = create_migration_frame(multi_year_conteos, year=2020)
        
        # Should return empty DataFrame
        assert len(result) == 0


# ============================================================================
# Test Data Cleaning
# ============================================================================
class TestDataCleaning:
    """Test data cleaning and coercion."""
    
    def test_numeric_coercion(self):
        """Test that string numbers are converted."""
        data = pd.DataFrame([
            {'DPTO': '05', 'ANO': '2018', 'EDAD': '0-4', 'SEXO': '1',
             'VARIABLE': 'poblacion_total', 'VALOR': '100000'},
            {'DPTO': '05', 'ANO': '2018', 'EDAD': '0-4', 'SEXO': '1',
             'VARIABLE': 'flujo_inmigracion', 'VALOR': '500'},
            {'DPTO': '05', 'ANO': '2018', 'EDAD': '0-4', 'SEXO': '1',
             'VARIABLE': 'flujo_emigracion', 'VALOR': '300'},
        ])
        
        result = create_migration_frame(data, year=2018)
        
        # Should convert successfully
        assert result.iloc[0]['poblacion_total'] == 100000
        assert result.iloc[0]['inmigracion_F'] == 500
    
    def test_invalid_valor_handling(self):
        """Test that invalid VALOR becomes 0."""
        data = pd.DataFrame([
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'poblacion_total', 'VALOR': 'invalid'},
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'flujo_inmigracion', 'VALOR': 'N/A'},
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'flujo_emigracion', 'VALOR': 100},
        ])
        
        result = create_migration_frame(data, year=2018)
        
        # Invalid values should become 0
        assert result.iloc[0]['poblacion_total'] == 0
        assert result.iloc[0]['inmigracion_F'] == 0
        assert result.iloc[0]['emigracion_F'] == 100
    
    def test_whitespace_stripping(self):
        """Test that whitespace in age labels is stripped."""
        data = pd.DataFrame([
            {'DPTO': '05', 'ANO': 2018, 'EDAD': ' 0-4 ', 'SEXO': 1,
             'VARIABLE': 'poblacion_total', 'VALOR': 100000},
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '  0-4', 'SEXO': 1,
             'VARIABLE': 'flujo_inmigracion', 'VALOR': 500},
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4  ', 'SEXO': 1,
             'VARIABLE': 'flujo_emigracion', 'VALOR': 300},
        ])
        
        result = create_migration_frame(data, year=2018)
        
        # All should be recognized as same age group
        assert result.iloc[0]['EDAD'] == '0-4'


# ============================================================================
# Test Edge Cases
# ============================================================================
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_migration_variable(self):
        """Test when only immigration or only emigration present."""
        data = pd.DataFrame([
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'poblacion_total', 'VALOR': 100000},
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'flujo_inmigracion', 'VALOR': 500},
            # No emigration data
        ])
        
        result = create_migration_frame(data, year=2018)
        
        # Should have immigration, emigration should be 0
        assert result.iloc[0]['inmigracion_F'] == 500
        assert result.iloc[0]['emigracion_F'] == 0
        assert result.iloc[0]['net_migration'] == 500
    
    def test_zero_population(self):
        """Test division by zero handling."""
        data = pd.DataFrame([
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'poblacion_total', 'VALOR': 0},
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'flujo_inmigracion', 'VALOR': 100},
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'flujo_emigracion', 'VALOR': 50},
        ])
        
        result = create_migration_frame(data, year=2018)
        
        # Rate should be NaN, not error
        assert pd.isna(result.iloc[0]['net_mig_rate'])
    
    def test_negative_migration(self):
        """Test that negative migration is preserved."""
        data = pd.DataFrame([
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'poblacion_total', 'VALOR': 100000},
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'flujo_inmigracion', 'VALOR': 200},
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'flujo_emigracion', 'VALOR': 500},
        ])
        
        result = create_migration_frame(data, year=2018)
        
        # Net should be negative (more out than in)
        assert result.iloc[0]['net_migration'] == -300
        assert result.iloc[0]['net_mig_rate'] < 0
    
    def test_empty_input(self):
        """Test with empty DataFrame."""
        data = pd.DataFrame(columns=['DPTO', 'ANO', 'EDAD', 'SEXO', 'VARIABLE', 'VALOR'])
        
        result = create_migration_frame(data, year=2018)
        
        # Should return empty but with correct columns
        assert len(result) == 0
        assert 'net_mig_rate' in result.columns


# ============================================================================
# Test Age Ordering
# ============================================================================
class TestAgeOrdering:
    """Test age group ordering."""
    
    def test_age_sorting(self):
        """Test that ages are sorted correctly."""
        data = []
        # Add in random order
        for age in ['20-24', '0-4', '10-14', '5-9']:
            data.append({
                'DPTO': '05', 'ANO': 2018, 'EDAD': age, 'SEXO': 1,
                'VARIABLE': 'poblacion_total', 'VALOR': 100000
            })
            data.append({
                'DPTO': '05', 'ANO': 2018, 'EDAD': age, 'SEXO': 1,
                'VARIABLE': 'flujo_inmigracion', 'VALOR': 100
            })
            data.append({
                'DPTO': '05', 'ANO': 2018, 'EDAD': age, 'SEXO': 1,
                'VARIABLE': 'flujo_emigracion', 'VALOR': 50
            })
        
        df = pd.DataFrame(data)
        result = create_migration_frame(df, year=2018)
        
        # Check ages are sorted correctly
        ages = result['EDAD'].tolist()
        assert ages == ['0-4', '5-9', '10-14', '20-24']
    
    def test_edad_is_categorical(self):
        """Test that EDAD is ordered categorical."""
        data = pd.DataFrame([
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'poblacion_total', 'VALOR': 100000},
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'flujo_inmigracion', 'VALOR': 100},
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'flujo_emigracion', 'VALOR': 50},
        ])
        
        result = create_migration_frame(data, year=2018)
        
        assert pd.api.types.is_categorical_dtype(result['EDAD'])
        assert result['EDAD'].cat.ordered


# ============================================================================
# Test Multi-Sex Handling
# ============================================================================
class TestMultiSex:
    """Test sex disaggregation."""
    
    def test_sex_separation(self, sample_conteos):
        """Test that males and females are kept separate."""
        result = create_migration_frame(sample_conteos, year=2018)
        
        # Should have separate rows for sex 1 and 2
        sex1_rows = result[result['SEXO'] == 1]
        sex2_rows = result[result['SEXO'] == 2]
        
        assert len(sex1_rows) == 2  # 2 age groups
        assert len(sex2_rows) == 2  # 2 age groups
    
    def test_sex_specific_rates(self, sample_conteos):
        """Test that rates can differ by sex."""
        # Modify sample data to have different migration by sex
        sample_conteos.loc[
            (sample_conteos['SEXO'] == 2) & 
            (sample_conteos['VARIABLE'] == 'flujo_inmigracion'),
            'VALOR'
        ] = 300  # Higher for females
        
        result = create_migration_frame(sample_conteos, year=2018)
        
        rate_sex1 = result[(result['EDAD'] == '0-4') & (result['SEXO'] == 1)]['net_mig_rate'].values[0]
        rate_sex2 = result[(result['EDAD'] == '0-4') & (result['SEXO'] == 2)]['net_mig_rate'].values[0]
        
        # Rates should differ
        assert rate_sex1 != rate_sex2


# ============================================================================
# Test Realistic Scenarios
# ============================================================================
class TestRealisticScenarios:
    """Test with realistic migration patterns."""
    
    def test_young_adult_peak(self):
        """Test typical age pattern with peak at young adult ages."""
        data = []
        
        # Age-specific migration (peak at 20-24)
        age_mig = {
            '0-4': 50,
            '5-9': 40,
            '10-14': 60,
            '15-19': 150,
            '20-24': 500,  # Peak
            '25-29': 400,
            '30-34': 200,
            '35-39': 100,
        }
        
        for age, mig_val in age_mig.items():
            data.append({
                'DPTO': '05', 'ANO': 2018, 'EDAD': age, 'SEXO': 1,
                'VARIABLE': 'poblacion_total', 'VALOR': 100000
            })
            data.append({
                'DPTO': '05', 'ANO': 2018, 'EDAD': age, 'SEXO': 1,
                'VARIABLE': 'flujo_inmigracion', 'VALOR': mig_val
            })
            data.append({
                'DPTO': '05', 'ANO': 2018, 'EDAD': age, 'SEXO': 1,
                'VARIABLE': 'flujo_emigracion', 'VALOR': mig_val // 2
            })
        
        df = pd.DataFrame(data)
        result = create_migration_frame(df, year=2018)
        
        # Rate at 20-24 should be highest
        rates = result.set_index('EDAD')['net_mig_rate']
        max_age = rates.idxmax()
        assert max_age == '20-24'
    
    def test_net_zero_migration(self):
        """Test balanced migration (in = out)."""
        data = pd.DataFrame([
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'poblacion_total', 'VALOR': 100000},
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'flujo_inmigracion', 'VALOR': 500},
            {'DPTO': '05', 'ANO': 2018, 'EDAD': '0-4', 'SEXO': 1,
             'VARIABLE': 'flujo_emigracion', 'VALOR': 500},
        ])
        
        result = create_migration_frame(data, year=2018)
        
        # Net should be zero
        assert result.iloc[0]['net_migration'] == 0
        assert result.iloc[0]['net_mig_rate'] == 0


# ============================================================================
# Integration Tests
# ============================================================================
class TestIntegration:
    """Test complete workflows."""
    
    def test_full_workflow(self):
        """Test complete workflow from raw data to rates."""
        # Create realistic multi-department, multi-year data
        data = []
        
        for year in [2017, 2018]:
            for dpto in ['05', '11', '13']:  # 3 departments
                for age in ['0-4', '5-9', '10-14', '15-19', '20-24']:
                    for sex in [1, 2]:
                        # Population varies by department
                        pop = {'05': 80000, '11': 120000, '13': 40000}[dpto]
                        
                        data.append({
                            'DPTO': dpto, 'ANO': year, 'EDAD': age, 'SEXO': sex,
                            'VARIABLE': 'poblacion_total', 'VALOR': pop
                        })
                        
                        # Migration varies by age (higher for young adults)
                        base_mig = 50 if age in ['20-24', '25-29'] else 20
                        
                        data.append({
                            'DPTO': dpto, 'ANO': year, 'EDAD': age, 'SEXO': sex,
                            'VARIABLE': 'flujo_inmigracion', 'VALOR': base_mig * 1.5
                        })
                        data.append({
                            'DPTO': dpto, 'ANO': year, 'EDAD': age, 'SEXO': sex,
                            'VARIABLE': 'flujo_emigracion', 'VALOR': base_mig
                        })
        
        df = pd.DataFrame(data)
        
        # Test single year
        result_2018 = create_migration_frame(df, year=2018)
        assert len(result_2018) == 10  # 5 ages × 2 sexes
        assert result_2018['ANO'].unique()[0] == 2018
        
        # Test all years
        result_all = create_migration_frame(df, year=None)
        assert len(result_all) == 20  # 2 years × 5 ages × 2 sexes
        assert set(result_all['ANO'].unique()) == {2017, 2018}
        
        # Check aggregation: sum of 3 departments
        pop_total = result_2018.iloc[0]['poblacion_total']
        assert pop_total == 80000 + 120000 + 40000
    
    def test_use_in_projection(self, sample_conteos):
        """Test typical usage in population projection."""
        # Get migration rates
        mig_rates = create_migration_frame(sample_conteos, year=2018)
        
        # Simulate using rates in projection
        for _, row in mig_rates.iterrows():
            age = row['EDAD']
            sex = row['SEXO']
            rate = row['net_mig_rate']
            pop = row['poblacion_total']
            
            # Apply rate to get net migrants
            net_migrants = pop * rate
            
            # Should match original net migration
            assert abs(net_migrants - row['net_migration']) < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
