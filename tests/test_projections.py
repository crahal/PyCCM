# tests/test_projections.py
"""
Test suite for projections.py module.

Tests cover:
- Leslie matrix construction
- Population projection mechanics
- Survival probability calculations
- Age label formatting
- File I/O operations
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from projections import (
    _s_open_from_ex,
    _hazard_from_survival,
    _format_age_labels_from_lifetable_index,
    make_projections,
    save_projections,
    save_LL,
)


# ============================================================================
# Test Helper Functions
# ============================================================================

class TestSOpenFromEx:
    """Test open interval survival probability from life expectancy."""
    
    def test_normal_life_expectancy(self):
        """Standard case: e_x = 10 years, step = 5 years."""
        result = _s_open_from_ex(step=5.0, e=10.0)
        expected = np.exp(-5.0 / 10.0)  # exp(-0.5) ≈ 0.6065
        assert np.isclose(result, expected, rtol=1e-6)
        assert 0.0 <= result <= 1.0
    
    def test_very_high_life_expectancy(self):
        """High life expectancy should give survival close to 1."""
        result = _s_open_from_ex(step=1.0, e=100.0)
        assert result > 0.99
        assert result <= 1.0
    
    def test_very_low_life_expectancy(self):
        """Low life expectancy should give lower survival."""
        result = _s_open_from_ex(step=5.0, e=2.0)
        expected = np.exp(-5.0 / 2.0)  # exp(-2.5) ≈ 0.082
        assert np.isclose(result, expected, rtol=1e-6)
        assert result < 0.1
    
    def test_zero_life_expectancy(self):
        """Zero life expectancy should return 0."""
        assert _s_open_from_ex(step=5.0, e=0.0) == 0.0
    
    def test_negative_life_expectancy(self):
        """Negative life expectancy should return 0."""
        assert _s_open_from_ex(step=5.0, e=-10.0) == 0.0
    
    def test_infinite_life_expectancy(self):
        """Infinite life expectancy should return 0."""
        assert _s_open_from_ex(step=5.0, e=np.inf) == 0.0
    
    def test_nan_life_expectancy(self):
        """NaN life expectancy should return 0."""
        assert _s_open_from_ex(step=5.0, e=np.nan) == 0.0
    
    def test_clipping_bounds(self):
        """Result should always be in [0, 1]."""
        # Edge cases that might produce out-of-bounds without clipping
        assert 0.0 <= _s_open_from_ex(step=0.0, e=10.0) <= 1.0
        assert 0.0 <= _s_open_from_ex(step=100.0, e=0.1) <= 1.0


class TestHazardFromSurvival:
    """Test hazard rate calculation from survival probability."""
    
    def test_perfect_survival(self):
        """Survival = 1.0 should give hazard ≈ 0."""
        result = _hazard_from_survival(s=1.0, step=5.0)
        assert result < 1e-10  # Essentially zero
    
    def test_50_percent_survival(self):
        """50% survival over 5 years."""
        result = _hazard_from_survival(s=0.5, step=5.0)
        expected = -np.log(0.5) / 5.0  # ln(2)/5 ≈ 0.1386
        assert np.isclose(result, expected, rtol=1e-6)
    
    def test_low_survival(self):
        """10% survival should give high hazard."""
        result = _hazard_from_survival(s=0.1, step=5.0)
        expected = -np.log(0.1) / 5.0  # ln(10)/5 ≈ 0.4605
        assert np.isclose(result, expected, rtol=1e-6)
        assert result > 0.4
    
    def test_zero_survival_clamped(self):
        """Zero survival should be clamped to avoid log(0)."""
        # Should not raise error, returns finite value
        result = _hazard_from_survival(s=0.0, step=5.0)
        assert np.isfinite(result)
        assert result > 0  # Very high hazard
    
    def test_survival_above_one_clamped(self):
        """Survival > 1 should be clamped to 1."""
        result = _hazard_from_survival(s=1.5, step=5.0)
        # Should clamp to 1.0, giving hazard ≈ 0
        assert result < 1e-10
    
    def test_different_time_steps(self):
        """Hazard should scale inversely with time step."""
        s = 0.5
        h1 = _hazard_from_survival(s, step=1.0)
        h5 = _hazard_from_survival(s, step=5.0)
        # h1 should be 5× h5
        assert np.isclose(h1, 5.0 * h5, rtol=1e-6)


class TestFormatAgeLabelsFromLifetableIndex:
    """Test age label formatting for single-year and abridged life tables."""
    
    def test_single_year_ages(self):
        """Single-year ages (step=1)."""
        ages = np.array([0, 1, 2, 3, 4, 5])
        result = _format_age_labels_from_lifetable_index(ages, step=1)
        expected = ['0', '1', '2', '3', '4', '5+']
        assert result == expected
    
    def test_five_year_ages(self):
        """5-year age groups."""
        ages = np.array([0, 5, 10, 15, 20, 80])
        result = _format_age_labels_from_lifetable_index(ages, step=5)
        expected = ['0-4', '5-9', '10-14', '15-19', '20-24', '80+']
        assert result == expected
    
    def test_ten_year_ages(self):
        """10-year age groups."""
        ages = np.array([0, 10, 20, 30])
        result = _format_age_labels_from_lifetable_index(ages, step=10)
        expected = ['0-9', '10-19', '20-29', '30+']
        assert result == expected
    
    def test_empty_ages(self):
        """Empty age array."""
        ages = np.array([])
        result = _format_age_labels_from_lifetable_index(ages, step=5)
        assert result == []
    
    def test_single_age(self):
        """Single age (open interval only)."""
        ages = np.array([0])
        result = _format_age_labels_from_lifetable_index(ages, step=1)
        assert result == ['0+']
    
    def test_single_age_abridged(self):
        """Single age with 5-year step."""
        ages = np.array([80])
        result = _format_age_labels_from_lifetable_index(ages, step=5)
        assert result == ['80+']


# ============================================================================
# Test Main Projection Function
# ============================================================================

class TestMakeProjections:
    """Test Leslie matrix construction and population projection."""
    
    @pytest.fixture
    def minimal_lifetable(self):
        """Create minimal life table for testing."""
        ages = np.array([0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        n_ages = len(ages)
        
        # Realistic survivorship (l_x starts at 100,000, declines)
        lx = 100000 * np.array([1.0, 0.98, 0.97, 0.96, 0.95, 0.94, 
                                0.92, 0.90, 0.87, 0.83, 0.78, 0.72])
        
        # Life expectancy at each age
        ex = np.array([75, 74, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25])
        
        df = pd.DataFrame({
            'lx': lx,
            'ex': ex,
            'n': [1] * n_ages  # 1-year age groups
        }, index=ages)
        
        return df
    
    @pytest.fixture
    def minimal_asfr(self):
        """Create minimal ASFR data."""
        ages = ['15', '20', '25', '30', '35', '40', '45']
        asfr_vals = [0.02, 0.08, 0.12, 0.10, 0.06, 0.02, 0.01]
        
        df = pd.DataFrame({
            'asfr': asfr_vals
        }, index=ages)
        
        return df
    
    def test_basic_projection_runs(self, minimal_lifetable, minimal_asfr):
        """Test that basic projection executes without errors."""
        n = len(minimal_lifetable) - 1  # n age groups (k = n+1 ages)
        X = 1  # Project 1 step
        
        # Initial population (arbitrary)
        pop_F = np.ones(n + 1) * 1000
        pop_M = np.ones(n + 1) * 1000
        
        # Net migration (zero for simplicity)
        net_F = np.zeros(n + 1)
        net_M = np.zeros(n + 1)
        
        # Birth counts (for ratio calculation)
        births_M = 500
        births_F = 500
        
        result = make_projections(
            net_F=net_F,
            net_M=net_M,
            n=n,
            X=X,
            fert_start_idx=3,  # Fertility starts at index 3 (age 15)
            conteos_all_2018_M_n_t=births_M,
            conteos_all_2018_F_n_t=births_F,
            lt_2018_F_t=minimal_lifetable,
            lt_2018_M_t=minimal_lifetable,
            conteos_all_2018_F_p_t=pop_F,
            conteos_all_2018_M_p_t=pop_M,
            asfr_2018=minimal_asfr,
            l0=100000,
            year=2020,
            DPTO="Test",
            death_choice="EEVV",
            mort_improv_F=0.01,
            mort_improv_M=0.01
        )
        
        # Unpack results
        L_MM, L_MF, L_FF, df_M, df_F, df_T = result
        
        # Check matrix dimensions
        assert L_MM.shape == (n + 1, n + 1)
        assert L_MF.shape == (n + 1, n + 1)
        assert L_FF.shape == (n + 1, n + 1)
        
        # Check DataFrame structure
        assert 'EDAD' in df_F.columns
        assert 'VALOR_corrected' in df_F.columns
        assert 'year' in df_F.columns
        assert len(df_F) == n + 1
    
    def test_leslie_matrix_structure(self, minimal_lifetable, minimal_asfr):
        """Test Leslie matrix has correct structure."""
        n = len(minimal_lifetable) - 1
        pop_F = np.ones(n + 1) * 1000
        pop_M = np.ones(n + 1) * 1000
        net_F = np.zeros(n + 1)
        net_M = np.zeros(n + 1)
        
        L_MM, L_MF, L_FF, _, _, _ = make_projections(
            net_F=net_F, net_M=net_M, n=n, X=1, fert_start_idx=3,
            conteos_all_2018_M_n_t=500, conteos_all_2018_F_n_t=500,
            lt_2018_F_t=minimal_lifetable, lt_2018_M_t=minimal_lifetable,
            conteos_all_2018_F_p_t=pop_F, conteos_all_2018_M_p_t=pop_M,
            asfr_2018=minimal_asfr, l0=100000,
            year=2020, DPTO="Test", death_choice="EEVV"
        )
        
        # Check survival probabilities on subdiagonal
        for i in range(1, n + 1):
            s_FF = L_FF[i, i-1]
            s_MM = L_MM[i, i-1]
            # Survival probabilities should be in [0, 1]
            assert 0.0 <= s_FF <= 1.0, f"FF survival at {i} out of bounds: {s_FF}"
            assert 0.0 <= s_MM <= 1.0, f"MM survival at {i} out of bounds: {s_MM}"
        
        # Check open interval has survival on diagonal
        assert 0.0 <= L_FF[-1, -1] <= 1.0
        assert 0.0 <= L_MM[-1, -1] <= 1.0
        
        # Check fertility row (first row) has non-zero values
        assert L_FF[0, :].sum() > 0, "Female fertility row is all zeros"
        assert L_MF[0, :].sum() > 0, "Male from female fertility row is all zeros"
    
    def test_projection_conservation(self, minimal_lifetable, minimal_asfr):
        """Test population is conserved (approximately) without migration."""
        n = len(minimal_lifetable) - 1
        initial_pop = 10000
        pop_F = np.ones(n + 1) * (initial_pop / (n + 1))
        pop_M = np.ones(n + 1) * (initial_pop / (n + 1))
        net_F = np.zeros(n + 1)  # No migration
        net_M = np.zeros(n + 1)
        
        _, _, _, df_M, df_F, df_T = make_projections(
            net_F=net_F, net_M=net_M, n=n, X=1, fert_start_idx=3,
            conteos_all_2018_M_n_t=500, conteos_all_2018_F_n_t=500,
            lt_2018_F_t=minimal_lifetable, lt_2018_M_t=minimal_lifetable,
            conteos_all_2018_F_p_t=pop_F, conteos_all_2018_M_p_t=pop_M,
            asfr_2018=minimal_asfr, l0=100000,
            year=2020, DPTO="Test", death_choice="EEVV",
            mort_improv_F=0.0,  # No mortality improvement
            mort_improv_M=0.0
        )
        
        # Projected population
        proj_total = df_T['VALOR_corrected'].sum()
        
        # Should be close to initial (allowing for births/deaths)
        # With low mortality and fertility, shouldn't differ by >50%
        assert 0.5 * (2 * initial_pop) < proj_total < 1.5 * (2 * initial_pop)
    
    def test_zero_population(self, minimal_lifetable, minimal_asfr):
        """Test projection with zero initial population."""
        n = len(minimal_lifetable) - 1
        pop_F = np.zeros(n + 1)
        pop_M = np.zeros(n + 1)
        net_F = np.zeros(n + 1)
        net_M = np.zeros(n + 1)
        
        _, _, _, df_M, df_F, df_T = make_projections(
            net_F=net_F, net_M=net_M, n=n, X=1, fert_start_idx=3,
            conteos_all_2018_M_n_t=1, conteos_all_2018_F_n_t=1,
            lt_2018_F_t=minimal_lifetable, lt_2018_M_t=minimal_lifetable,
            conteos_all_2018_F_p_t=pop_F, conteos_all_2018_M_p_t=pop_M,
            asfr_2018=minimal_asfr, l0=100000,
            year=2020, DPTO="Test", death_choice="EEVV"
        )
        
        # Projected population should be zero (or very small from numerical noise)
        assert df_F['VALOR_corrected'].sum() < 1e-6
        assert df_M['VALOR_corrected'].sum() < 1e-6
        assert df_T['VALOR_corrected'].sum() < 1e-6
    
    def test_missing_lx_column_raises(self, minimal_asfr):
        """Test that missing 'lx' column raises ValueError."""
        bad_lt = pd.DataFrame({
            'ex': [70, 65, 60],
            'n': [5, 5, 5]
        }, index=[0, 5, 10])
        
        n = 2
        pop_F = np.ones(n + 1) * 1000
        
        with pytest.raises(ValueError, match="Life tables must include an 'lx' column"):
            make_projections(
                net_F=np.zeros(n + 1), net_M=np.zeros(n + 1),
                n=n, X=1, fert_start_idx=0,
                conteos_all_2018_M_n_t=500, conteos_all_2018_F_n_t=500,
                lt_2018_F_t=bad_lt, lt_2018_M_t=bad_lt,
                conteos_all_2018_F_p_t=pop_F, conteos_all_2018_M_p_t=pop_F,
                asfr_2018=minimal_asfr, l0=100000,
                year=2020, DPTO="Test", death_choice="EEVV"
            )
    
    def test_mortality_improvement_effect(self, minimal_lifetable, minimal_asfr):
        """Test that mortality improvement increases survival."""
        n = len(minimal_lifetable) - 1
        pop_F = np.ones(n + 1) * 1000
        net_F = np.zeros(n + 1)
        net_M = np.zeros(n + 1)
        
        # No improvement
        L_MM_0, _, _, _, _, _ = make_projections(
            net_F=net_F, net_M=net_F, n=n, X=1, fert_start_idx=3,
            conteos_all_2018_M_n_t=500, conteos_all_2018_F_n_t=500,
            lt_2018_F_t=minimal_lifetable, lt_2018_M_t=minimal_lifetable,
            conteos_all_2018_F_p_t=pop_F, conteos_all_2018_M_p_t=pop_F,
            asfr_2018=minimal_asfr, l0=100000,
            year=2020, DPTO="Test", death_choice="EEVV",
            mort_improv_F=0.0, mort_improv_M=0.0
        )
        
        # With improvement
        L_MM_1, _, _, _, _, _ = make_projections(
            net_F=net_F, net_M=net_F, n=n, X=1, fert_start_idx=3,
            conteos_all_2018_M_n_t=500, conteos_all_2018_F_n_t=500,
            lt_2018_F_t=minimal_lifetable, lt_2018_M_t=minimal_lifetable,
            conteos_all_2018_F_p_t=pop_F, conteos_all_2018_M_p_t=pop_F,
            asfr_2018=minimal_asfr, l0=100000,
            year=2020, DPTO="Test", death_choice="EEVV",
            mort_improv_F=0.02, mort_improv_M=0.02  # 2% improvement
        )
        
        # Survival probabilities should be higher with improvement
        # Check a middle age group
        assert L_MM_1[5, 4] > L_MM_0[5, 4], "Mortality improvement should increase survival"


# ============================================================================
# Test File I/O Functions
# ============================================================================

class TestSaveProjections:
    """Test saving projection DataFrames to disk."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    @pytest.fixture
    def sample_projection_df(self):
        """Create sample projection DataFrame."""
        return pd.DataFrame({
            'EDAD': ['0-4', '5-9', '10-14'],
            'DPTO_NOMBRE': ['Test'] * 3,
            'year': [2025] * 3,
            'VALOR_corrected': [1000, 1100, 1200],
            'death_choice': ['EEVV'] * 3
        })
    
    def test_save_projections_creates_files(self, temp_dir, sample_projection_df):
        """Test that files are created in correct directory structure."""
        save_projections(
            proj_F=sample_projection_df,
            proj_M=sample_projection_df,
            proj_T=sample_projection_df,
            sample_type="mid_omissions",
            distribution=None,
            suffix="_test",
            death_choice="_EEVV",
            year=2020,
            results_dir=temp_dir
        )
        
        # Check that files exist
        expected_files = [
            "projections/age_structures_df_F/mid_omissions/age_structures_df_F_test_EEVV.csv",
            "projections/age_structures_df_M/mid_omissions/age_structures_df_M_test_EEVV.csv",
            "projections/age_structures_df_T/mid_omissions/age_structures_df_T_test_EEVV.csv",
        ]
        
        for file_path in expected_files:
            full_path = os.path.join(temp_dir, file_path)
            assert os.path.exists(full_path), f"File not found: {full_path}"
            
            # Check file is readable
            df = pd.read_csv(full_path)
            assert len(df) == 3
    
    def test_save_projections_with_distribution(self, temp_dir, sample_projection_df):
        """Test saving with distribution subdirectory."""
        save_projections(
            proj_F=sample_projection_df,
            proj_M=sample_projection_df,
            proj_T=sample_projection_df,
            sample_type="draw",
            distribution="uniform",
            suffix="_001",
            death_choice="_EEVV",
            year=2020,
            results_dir=temp_dir
        )
        
        # Check that files exist in distribution subdirectory
        expected_path = os.path.join(
            temp_dir, "projections", "age_structures_df_F", "draw", "uniform",
            "age_structures_df_F_001_EEVV.csv"
        )
        assert os.path.exists(expected_path)


class TestSaveLL:
    """Test saving Leslie matrices to disk."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    @pytest.fixture
    def sample_leslie_matrix(self):
        """Create sample Leslie matrix."""
        return np.random.rand(10, 10)
    
    def test_save_LL_creates_files(self, temp_dir, sample_leslie_matrix):
        """Test that Leslie matrices are saved correctly."""
        save_LL(
            L_MM=sample_leslie_matrix,
            L_MF=sample_leslie_matrix,
            L_FF=sample_leslie_matrix,
            death_choice="EEVV",
            DPTO="Bogota",
            sample_type="mid_omissions",
            distribution=None,
            suffix="_test",
            year=2020,
            results_dir=temp_dir
        )
        
        # Check that files exist
        expected_files = [
            "projections/EEVV/Bogota/mid_omissions/L_MM_test2020.csv",
            "projections/EEVV/Bogota/mid_omissions/L_MF_test2020.csv",
            "projections/EEVV/Bogota/mid_omissions/L_FF_test2020.csv",
        ]
        
        for file_path in expected_files:
            full_path = os.path.join(temp_dir, file_path)
            assert os.path.exists(full_path), f"File not found: {full_path}"
            
            # Check matrix is readable
            df = pd.read_csv(full_path)
            assert df.shape == (10, 10)
    
    def test_save_LL_with_distribution(self, temp_dir, sample_leslie_matrix):
        """Test saving with distribution subdirectory."""
        save_LL(
            L_MM=sample_leslie_matrix,
            L_MF=sample_leslie_matrix,
            L_FF=sample_leslie_matrix,
            death_choice="EEVV",
            DPTO="Antioquia",
            sample_type="draw",
            distribution="beta",
            suffix="_draw_042",
            year=2025,
            results_dir=temp_dir
        )
        
        # Check files exist in distribution subdirectory
        expected_path = os.path.join(
            temp_dir, "projections", "EEVV", "Antioquia", "draw", "beta",
            "L_MM_draw_0422025.csv"
        )
        assert os.path.exists(expected_path)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
