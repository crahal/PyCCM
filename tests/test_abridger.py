# tests/test_abridger.py
import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from abridger import (
    parse_edad,
    collapse_convention2,
    default_survivorship_0_to_5,
    nLx_1year,
    weights_from_nLx,
    _second_diff_matrix,
    _solve_smooth,
    _apply_infant_adjustment,
    _unabridge_one_group,
    unabridge_df,
    harmonize_migration_to_90plus,
    harmonize_conteos_to_90plus,
    unabridge_all,
    save_unabridged,
    SERIES_KEYS_DEFAULT,
    _weights_from_pop_or_geometric,
    _geom_weights,
)


class TestParseEdad:
    """Test age label parsing function"""
    
    def test_parse_range(self):
        """Test parsing age ranges like '0-4'"""
        assert parse_edad("0-4") == (0, 4)
        assert parse_edad("15-19") == (15, 19)
        assert parse_edad("70-74") == (70, 74)
    
    def test_parse_with_spaces(self):
        """Test parsing with extra whitespace"""
        assert parse_edad("  0 - 4  ") == (0, 4)
        assert parse_edad("15 -19") == (15, 19)
    
    def test_parse_reversed_range(self):
        """Test that reversed ranges are corrected"""
        assert parse_edad("4-0") == (0, 4)
        assert parse_edad("19-15") == (15, 19)
    
    def test_parse_open_ended(self):
        """Test parsing open-ended ages like '80+'"""
        assert parse_edad("80+") == (80, None)
        assert parse_edad("90+") == (90, None)
        assert parse_edad("  70 + ") == (70, None)
    
    def test_parse_single_age(self):
        """Test parsing single ages like '5'"""
        assert parse_edad("0") == (0, 0)
        assert parse_edad("15") == (15, 15)
        assert parse_edad("  25  ") == (25, 25)
    
    def test_parse_none(self):
        """Test handling None input"""
        assert parse_edad(None) == (None, None)
    
    def test_parse_nan(self):
        """Test handling NaN input"""
        assert parse_edad(np.nan) == (None, None)
    
    def test_parse_invalid(self):
        """Test handling invalid formats"""
        assert parse_edad("invalid") == (None, None)
        assert parse_edad("abc-def") == (None, None)
        assert parse_edad("") == (None, None)


class TestCollapseConvention2:
    """Test convention 2 collapsing (a-(a+1) means single year a)"""
    
    def test_collapses_adjacent_ages(self):
        """Test that (a, a+1) becomes (a, a)"""
        df = pd.DataFrame({
            "EDAD_MIN": [0, 5, 10],
            "EDAD_MAX": [1, 6, 11]
        })
        result = collapse_convention2(df)
        
        assert result.loc[0, "EDAD_MAX"] == 0
        assert result.loc[1, "EDAD_MAX"] == 5
        assert result.loc[2, "EDAD_MAX"] == 10
    
    def test_preserves_non_adjacent(self):
        """Test that non-adjacent ranges are preserved"""
        df = pd.DataFrame({
            "EDAD_MIN": [0, 5, 10],
            "EDAD_MAX": [4, 9, 14]
        })
        result = collapse_convention2(df)
        
        assert result.loc[0, "EDAD_MAX"] == 4
        assert result.loc[1, "EDAD_MAX"] == 9
        assert result.loc[2, "EDAD_MAX"] == 14
    
    def test_handles_nan(self):
        """Test handling of NaN values"""
        df = pd.DataFrame({
            "EDAD_MIN": [0, np.nan, 10],
            "EDAD_MAX": [1, 5, np.nan]
        })
        result = collapse_convention2(df)
        
        # Row 0: 0-1 should collapse to 0-0
        assert result.loc[0, "EDAD_MAX"] == 0
        # Row 1: NaN-5 should pass through unchanged (NaN in MIN prevents collapse)
        assert result.loc[1, "EDAD_MAX"] == 5
        # Row 2: 10-NaN should pass through unchanged
        assert pd.isna(result.loc[2, "EDAD_MAX"])


class TestLifeTableFunctions:
    """Test life table weight calculation functions"""
    
    def test_default_survivorship(self):
        """Test default survivorship values"""
        lx = default_survivorship_0_to_5()
        
        assert lx[0] == 1.0
        assert 0 < lx[1] < lx[0]  # Some infant mortality
        assert all(lx[i] > lx[i+1] for i in range(4))  # Monotonic decrease
    
    def test_nLx_1year(self):
        """Test person-years calculation"""
        lx = {0: 1.0, 1: 0.96, 2: 0.959, 3: 0.958, 4: 0.957, 5: 0.956}
        L = nLx_1year(lx, a0=0.10)
        
        assert 0 in L
        assert L[0] > 0  # Infant person-years
        assert all(L[i] > 0 for i in range(5))
    
    def test_weights_from_nLx(self):
        """Test weight calculation from nLx"""
        L = {0: 0.98, 1: 0.96, 2: 0.96, 3: 0.96, 4: 0.96}
        ages = [0, 1, 2, 3, 4]
        
        weights = weights_from_nLx(L, ages)
        
        assert len(weights) == 5
        assert np.isclose(weights.sum(), 1.0)
        assert all(w >= 0 for w in weights)
    
    def test_weights_handles_empty_L(self):
        """Test weights with missing L values"""
        L = {}
        ages = [0, 1, 2]
        
        weights = weights_from_nLx(L, ages)
        
        assert len(weights) == 3
        assert np.isclose(weights.sum(), 1.0)
        # Should use uniform weights as fallback
        assert np.allclose(weights, 1/3)


class TestSecondDiffMatrix:
    """Test second difference matrix creation"""
    
    def test_matrix_shape(self):
        """Test matrix dimensions"""
        D = _second_diff_matrix(5)
        assert D.shape == (3, 5)
        
        D = _second_diff_matrix(10)
        assert D.shape == (8, 10)
    
    def test_matrix_values(self):
        """Test matrix contains correct differences"""
        D = _second_diff_matrix(5)
        
        # First row should be [1, -2, 1, 0, 0]
        assert np.allclose(D[0], [1, -2, 1, 0, 0])
        # Second row should be [0, 1, -2, 1, 0]
        assert np.allclose(D[1], [0, 1, -2, 1, 0])
    
    def test_small_n(self):
        """Test with n < 3"""
        D = _second_diff_matrix(2)
        assert D.shape == (0, 2)
        
        D = _second_diff_matrix(1)
        assert D.shape == (0, 1)


class TestSolveSmooth:
    """Test constrained smoothing solver"""
    
    def test_satisfies_constraints(self):
        """Test that solution satisfies Ax = b"""
        # Simple constraint: x[0] + x[1] + x[2] = 10
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([10.0])
        n = 3
        
        x = _solve_smooth(A, b, n)
        
        assert len(x) == 3
        assert np.isclose(A @ x, b).all()
        assert all(x >= 0)  # Non-negative
    
    def test_multiple_constraints(self):
        """Test with multiple constraints"""
        # x[0] + x[1] = 5, x[2] + x[3] = 7
        A = np.array([[1.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 1.0]])
        b = np.array([5.0, 7.0])
        n = 4
        
        x = _solve_smooth(A, b, n)
        
        assert len(x) == 4
        assert np.allclose(A @ x, b, atol=1e-5)
        assert all(x >= 0)
    
    def test_smoothness_preference(self):
        """Test that solution prefers smooth values"""
        # Single constraint summing to 100 over 5 values
        A = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
        b = np.array([100.0])
        n = 5
        
        x = _solve_smooth(A, b, n, ridge=1e-6)
        
        # Solution should be relatively smooth (not highly variable)
        diffs = np.diff(x)
        assert np.std(diffs) < 10  # Low variability in differences


class TestApplyInfantAdjustment:
    """Test infant age adjustment for 0-4 group"""
    
    def test_no_adjustment_for_non_population(self):
        """Test that non-population variables pass through unchanged"""
        cons = [(0, 4, 100.0)]
        result = _apply_infant_adjustment(cons, variable="defunciones")
        
        assert result == cons
    
    def test_splits_0_4_with_age_0_present(self):
        """Test splitting 0-4 when age 0 is already present"""
        cons = [(0, 0, 20.0), (0, 4, 100.0)]
        result = _apply_infant_adjustment(cons, variable="poblacion_total")
        
        # Should have age 0 and split 1-4
        ages = {(lo, hi) for lo, hi, _ in result}
        assert (0, 0) in ages
        assert (1, 1) in ages
        assert (2, 2) in ages
        assert (3, 3) in ages
        assert (4, 4) in ages
        assert (0, 4) not in ages  # Original 0-4 removed
    
    def test_splits_0_4_without_age_0(self):
        """Test splitting 0-4 when age 0 is not present"""
        cons = [(0, 4, 100.0)]
        result = _apply_infant_adjustment(cons, variable="poblacion_total")
        
        # Should split into 0,1,2,3,4
        ages = {(lo, hi) for lo, hi, _ in result}
        assert (0, 0) in ages
        assert (1, 1) in ages
        assert (4, 4) in ages
        assert (0, 4) not in ages
    
    def test_splits_1_4(self):
        """Test splitting 1-4 group"""
        cons = [(1, 4, 80.0)]
        result = _apply_infant_adjustment(cons, variable="poblacion_total")
        
        # Should split into 1,2,3,4
        ages = {(lo, hi) for lo, hi, _ in result}
        assert (1, 1) in ages
        assert (2, 2) in ages
        assert (3, 3) in ages
        assert (4, 4) in ages
        assert (1, 4) not in ages
    
    def test_preserves_total(self):
        """Test that total population is preserved"""
        cons = [(0, 4, 100.0)]
        result = _apply_infant_adjustment(cons, variable="poblacion_total")
        
        total = sum(v for _, _, v in result)
        assert np.isclose(total, 100.0)


class TestUnabridgeOneGroup:
    """Test single-group unabridging"""
    
    def test_simple_unabridging(self):
        """Test basic unabridging of age groups"""
        df = pd.DataFrame({
            "DPTO_NOMBRE": ["TestDpto", "TestDpto"],
            "SEXO": ["M", "M"],
            "FUENTE": ["X", "X"],
            "ANO": [2020, 2020],
            "VARIABLE": ["poblacion_total", "poblacion_total"],
            "OMISION": [1, 1],
            "EDAD": ["5-9", "10-14"],
            "EDAD_MIN": [5, 10],
            "EDAD_MAX": [9, 14],
            "VALOR": [1000, 1200]
        })
        
        series_keys = ["DPTO_NOMBRE", "SEXO", "FUENTE", "ANO", "VARIABLE", "OMISION"]
        result = _unabridge_one_group(df, series_keys, "poblacion_total", "VALOR")
        
        assert not result.empty
        assert "EDAD" in result.columns
        assert "VALOR" in result.columns
        # Should have single-year ages from 5 to 14
        ages = result["EDAD"].astype(int).tolist()
        assert min(ages) == 5
        assert max(ages) == 14
    
    def test_preserves_total(self):
        """Test that total is preserved after unabridging"""
        df = pd.DataFrame({
            "DPTO_NOMBRE": ["TestDpto"],
            "SEXO": ["M"],
            "FUENTE": ["X"],
            "ANO": [2020],
            "VARIABLE": ["poblacion_total"],
            "OMISION": [1],
            "EDAD": ["5-9"],
            "EDAD_MIN": [5],
            "EDAD_MAX": [9],
            "VALOR": [1000.0]
        })
        
        series_keys = ["DPTO_NOMBRE", "SEXO", "FUENTE", "ANO", "VARIABLE", "OMISION"]
        result = _unabridge_one_group(df, series_keys, "poblacion_total", "VALOR")
        
        total = result["VALOR"].sum()
        assert np.isclose(total, 1000.0, rtol=0.01)
    
    def test_handles_empty_group(self):
        """Test handling of empty groups"""
        df = pd.DataFrame({
            "DPTO_NOMBRE": [],
            "SEXO": [],
            "EDAD_MIN": [],
            "EDAD_MAX": [],
            "VALOR": []
        })
        
        series_keys = ["DPTO_NOMBRE", "SEXO"]
        result = _unabridge_one_group(df, series_keys, "poblacion_total", "VALOR")
        
        assert result.empty


class TestUnabridgeDf:
    """Test full DataFrame unabridging"""
    
    def test_unabridge_multiple_groups(self):
        """Test unabridging with multiple series"""
        df = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1", "Dept1", "Dept2", "Dept2"],
            "DPTO_CODIGO": ["01", "01", "02", "02"],
            "SEXO": ["M", "M", "F", "F"],
            "FUENTE": ["X", "X", "Y", "Y"],
            "ANO": [2020, 2020, 2020, 2020],
            "VARIABLE": ["poblacion_total"] * 4,
            "OMISION": [1, 1, 1, 1],
            "EDAD": ["5-9", "10-14", "5-9", "10-14"],
            "VALOR_corrected": [1000, 1200, 900, 1100]
        })
        
        result = unabridge_df(df, value_col="VALOR_corrected")
        
        assert not result.empty
        # Should have multiple groups
        groups = result.groupby(["DPTO_NOMBRE", "SEXO"]).ngroups
        assert groups == 2
    
    def test_passthrough_open_ended(self):
        """Test that open-ended ages pass through"""
        df = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1", "Dept1"],
            "DPTO_CODIGO": ["01", "01"],
            "SEXO": ["M", "M"],
            "FUENTE": ["X", "X"],
            "ANO": [2020, 2020],
            "VARIABLE": ["poblacion_total"] * 2,
            "OMISION": [1, 1],
            "EDAD": ["70-74", "80+"],
            "VALOR_corrected": [500, 300]
        })
        
        result = unabridge_df(df, value_col="VALOR_corrected")
        
        # Should include the 80+ row
        assert "80+" in result["EDAD"].values


class TestWeightsFromPopOrGeometric:
    """Test weight calculation for tail harmonization"""
    
    def test_uses_population_when_available(self):
        """Test using actual population for weights"""
        pop_df = pd.DataFrame({
            "EDAD": ["70-74", "75-79", "80-84", "85-89", "90+"],
            "VALOR_corrected": [1000, 800, 600, 400, 200]
        })
        
        bins = ["70-74", "75-79", "80-84", "85-89", "90+"]
        weights = _weights_from_pop_or_geometric(pop_df, bins)
        
        assert len(weights) == 5
        assert np.isclose(weights.sum(), 1.0)
        # Should be proportional to population
        assert weights[0] > weights[-1]  # More in 70-74 than 90+
    
    def test_uses_geometric_fallback(self):
        """Test geometric fallback when population unavailable"""
        pop_df = pd.DataFrame({
            "EDAD": [],
            "VALOR_corrected": []
        })
        
        bins = ["80-84", "85-89", "90+"]
        weights = _weights_from_pop_or_geometric(pop_df, bins, r=0.6)
        
        assert len(weights) == 3
        assert np.isclose(weights.sum(), 1.0)
        # Geometric with r<1 should decrease
        assert weights[0] > weights[1] > weights[2]


class TestGeomWeights:
    """Test geometric weight generation"""
    
    def test_decreasing_weights(self):
        """Test decreasing geometric progression (r < 1)"""
        bands = ["70-74", "75-79", "80-84"]
        weights = _geom_weights(bands, r=0.7)
        
        assert len(weights) == 3
        assert np.isclose(weights.sum(), 1.0)
        assert weights[0] > weights[1] > weights[2]
    
    def test_increasing_weights(self):
        """Test increasing geometric progression (r > 1)"""
        bands = ["80-84", "85-89", "90+"]
        weights = _geom_weights(bands, r=1.4)
        
        assert len(weights) == 3
        assert np.isclose(weights.sum(), 1.0)
        assert weights[0] < weights[1] < weights[2]


class TestHarmonizeMigrationTo90Plus:
    """Test migration tail harmonization"""
    
    def test_splits_70_plus(self):
        """Test splitting 70+ into detailed bands"""
        mig = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1"],
            "SEXO": ["M"],
            "ANO": [2020],
            "EDAD": ["70+"],
            "VALOR": [100.0]
        })
        
        pop = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1", "Dept1", "Dept1", "Dept1", "Dept1"],
            "SEXO": ["M", "M", "M", "M", "M"],
            "ANO": [2020, 2020, 2020, 2020, 2020],
            "EDAD": ["70-74", "75-79", "80-84", "85-89", "90+"],
            "VALOR_corrected": [50, 30, 15, 4, 1]
        })
        
        series_keys = ["DPTO_NOMBRE", "SEXO", "ANO"]
        result = harmonize_migration_to_90plus(mig, pop, series_keys)
        
        # Should have 5 bands instead of 70+
        assert "70+" not in result["EDAD"].values
        assert "70-74" in result["EDAD"].values
        assert "90+" in result["EDAD"].values
        # Total should be preserved
        assert np.isclose(result["VALOR"].sum(), 100.0)
    
    def test_splits_80_plus(self):
        """Test splitting 80+ into detailed bands"""
        mig = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1"],
            "SEXO": ["F"],
            "ANO": [2020],
            "EDAD": ["80+"],
            "VALOR": [50.0]
        })
        
        pop = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1", "Dept1", "Dept1"],
            "SEXO": ["F", "F", "F"],
            "ANO": [2020, 2020, 2020],
            "EDAD": ["80-84", "85-89", "90+"],
            "VALOR_corrected": [20, 8, 2]
        })
        
        series_keys = ["DPTO_NOMBRE", "SEXO", "ANO"]
        result = harmonize_migration_to_90plus(mig, pop, series_keys)
        
        assert "80+" not in result["EDAD"].values
        assert "80-84" in result["EDAD"].values
        assert np.isclose(result["VALOR"].sum(), 50.0)
    
    def test_preserves_other_ages(self):
        """Test that non-tail ages pass through"""
        mig = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1", "Dept1"],
            "SEXO": ["M", "M"],
            "ANO": [2020, 2020],
            "EDAD": ["20-24", "70+"],
            "VALOR": [200.0, 100.0]
        })
        
        pop = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1", "Dept1", "Dept1", "Dept1", "Dept1"],
            "SEXO": ["M", "M", "M", "M", "M"],
            "ANO": [2020, 2020, 2020, 2020, 2020],
            "EDAD": ["70-74", "75-79", "80-84", "85-89", "90+"],
            "VALOR_corrected": [50, 30, 15, 4, 1]
        })
        
        series_keys = ["DPTO_NOMBRE", "SEXO", "ANO"]
        result = harmonize_migration_to_90plus(mig, pop, series_keys)
        
        assert "20-24" in result["EDAD"].values


class TestHarmonizeConteosTo90Plus:
    """Test conteos tail harmonization"""
    
    def test_splits_population_70_plus(self):
        """Test splitting population 70+ with decreasing weights"""
        df = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1"],
            "SEXO": ["M"],
            "ANO": [2020],
            "FUENTE": ["X"],
            "OMISION": [1],
            "VARIABLE": ["poblacion_total"],
            "EDAD": ["70+"],
            "VALOR_corrected": [1000.0]
        })
        
        series_keys = ["DPTO_NOMBRE", "SEXO", "ANO", "FUENTE", "OMISION"]
        result = harmonize_conteos_to_90plus(df, series_keys)
        
        assert "70+" not in result["EDAD"].values
        assert "70-74" in result["EDAD"].values
        assert "90+" in result["EDAD"].values
        
        # Total preserved
        assert np.isclose(result["VALOR_corrected"].sum(), 1000.0)
        
        # Should have decreasing pattern (younger > older)
        val_70_74 = result[result["EDAD"] == "70-74"]["VALOR_corrected"].iloc[0]
        val_90 = result[result["EDAD"] == "90+"]["VALOR_corrected"].iloc[0]
        assert val_70_74 > val_90
    
    def test_splits_deaths_80_plus(self):
        """Test splitting deaths 80+ with increasing weights"""
        df = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1"],
            "SEXO": ["F"],
            "ANO": [2020],
            "FUENTE": ["Y"],
            "OMISION": [2],
            "VARIABLE": ["defunciones"],
            "EDAD": ["80+"],
            "VALOR_corrected": [500.0]
        })
        
        series_keys = ["DPTO_NOMBRE", "SEXO", "ANO", "FUENTE", "OMISION"]
        result = harmonize_conteos_to_90plus(df, series_keys, r_deaths=1.45)
        
        assert "80+" not in result["EDAD"].values
        
        # Should have increasing pattern (older > younger) for deaths
        val_80_84 = result[result["EDAD"] == "80-84"]["VALOR_corrected"].iloc[0]
        val_90 = result[result["EDAD"] == "90+"]["VALOR_corrected"].iloc[0]
        assert val_90 > val_80_84
    
    def test_passthrough_non_target_variables(self):
        """Test that non-target variables pass through unchanged"""
        df = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1"],
            "SEXO": ["M"],
            "ANO": [2020],
            "FUENTE": ["X"],
            "OMISION": [1],
            "VARIABLE": ["other_variable"],
            "EDAD": ["70+"],
            "VALOR_corrected": [100.0]
        })
        
        series_keys = ["DPTO_NOMBRE", "SEXO", "ANO", "FUENTE", "OMISION"]
        result = harmonize_conteos_to_90plus(df, series_keys)
        
        # Should pass through unchanged
        assert "70+" in result["EDAD"].values


class TestUnabridgeAll:
    """Test unified unabridging API"""
    
    def test_unabridges_all_datasets(self):
        """Test that all three datasets are unabridged"""
        df = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1"],
            "DPTO_CODIGO": ["01"],
            "SEXO": ["M"],
            "ANO": [2020],
            "FUENTE": ["X"],
            "VARIABLE": ["poblacion_total"],
            "OMISION": [1],
            "EDAD": ["5-9"],
            "VALOR_corrected": [1000.0]
        })
        
        emi = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1"],
            "DPTO_CODIGO": ["01"],
            "SEXO": ["M"],
            "ANO": [2020],
            "FUENTE": ["X"],
            "OMISION": [1],
            "EDAD": ["20-24"],
            "VALOR": [100.0]
        })
        
        imi = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1"],
            "DPTO_CODIGO": ["01"],
            "SEXO": ["F"],
            "ANO": [2020],
            "FUENTE": ["X"],
            "OMISION": [1],
            "EDAD": ["25-29"],
            "VALOR": [80.0]
        })
        
        result = unabridge_all(df=df, emi=emi, imi=imi)
        
        assert "conteos" in result
        assert "emi" in result
        assert "imi" in result
        assert not result["conteos"].empty
        assert not result["emi"].empty
        assert not result["imi"].empty
    
    def test_uses_correct_value_columns(self):
        """Test that correct value columns are used"""
        df = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1"],
            "DPTO_CODIGO": ["01"],
            "SEXO": ["M"],
            "ANO": [2020],
            "FUENTE": ["X"],
            "VARIABLE": ["poblacion_total"],
            "OMISION": [1],
            "EDAD": ["10-14"],
            "VALOR_corrected": [1000.0]
        })
        
        emi = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1"],
            "DPTO_CODIGO": ["01"],
            "SEXO": ["M"],
            "ANO": [2020],
            "FUENTE": ["X"],
            "OMISION": [1],
            "EDAD": ["30"],
            "VALOR": [50.0]
        })
        
        imi = pd.DataFrame({
            "DPTO_NOMBRE": ["Dept1"],
            "DPTO_CODIGO": ["01"],
            "SEXO": ["F"],
            "ANO": [2020],
            "FUENTE": ["X"],
            "OMISION": [1],
            "EDAD": ["35"],
            "VALOR": [40.0]
        })
        
        result = unabridge_all(
            df=df,
            emi=emi,
            imi=imi,
            conteos_value_col="VALOR_corrected"
        )
        
        # Should successfully process all
        assert all(key in result for key in ["conteos", "emi", "imi"])


class TestSaveUnabridged:
    """Test saving unabridged results"""
    
    def test_saves_to_csv(self):
        """Test that files are created"""
        objs = {
            "conteos": pd.DataFrame({"EDAD": ["0", "1"], "VALOR": [100, 95]}),
            "emi": pd.DataFrame({"EDAD": ["20"], "VALOR": [10]}),
            "imi": pd.DataFrame({"EDAD": ["25"], "VALOR": [15]})
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_unabridged(objs, tmpdir)
            
            # Check files exist
            assert os.path.exists(os.path.join(tmpdir, "conteos_unabridged_single_year.csv"))
            assert os.path.exists(os.path.join(tmpdir, "emi_unabridged_single_year.csv"))
            assert os.path.exists(os.path.join(tmpdir, "imi_unabridged_single_year.csv"))
    
    def test_creates_directory(self):
        """Test that output directory is created if it doesn't exist"""
        objs = {
            "conteos": pd.DataFrame({"EDAD": ["0"], "VALOR": [100]})
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "new_subdir")
            save_unabridged(objs, out_dir)
            
            assert os.path.exists(out_dir)


class TestSeriesKeysDefault:
    """Test default series keys constant"""
    
    def test_has_expected_keys(self):
        """Test that default keys are present"""
        expected = ["DPTO_NOMBRE", "DPTO_CODIGO", "ANO", "SEXO", "VARIABLE", "FUENTE", "OMISION"]
        
        assert SERIES_KEYS_DEFAULT == expected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
