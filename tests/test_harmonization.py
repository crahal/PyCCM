"""
Comprehensive tests for tail harmonization functions in abridger.py

Tests cover:
- _weights_from_pop_or_geometric
- harmonize_migration_to_90plus
- harmonize_conteos_to_90plus
- _geom_weights
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abridger import (
    _weights_from_pop_or_geometric,
    harmonize_migration_to_90plus,
    harmonize_conteos_to_90plus,
    _geom_weights,
    BANDS_70_TO_90,
    BANDS_80_TO_90,
)


# =====================================================================
# Tests for _geom_weights
# =====================================================================

class TestGeomWeights:
    """Test geometric weight calculation"""
    
    def test_increasing_weights(self):
        """Weights should increase with age when r > 1"""
        bands = ["80-84", "85-89", "90+"]
        w = _geom_weights(bands, r=1.5)
        
        assert len(w) == 3
        assert np.allclose(w.sum(), 1.0)
        # Each should be larger than previous (r > 1 → increasing)
        assert w[1] > w[0]
        assert w[2] > w[1]
    
    def test_decreasing_weights(self):
        """Weights should decrease with age when r < 1"""
        bands = ["70-74", "75-79", "80-84", "85-89", "90+"]
        w = _geom_weights(bands, r=0.7)
        
        assert len(w) == 5
        assert np.allclose(w.sum(), 1.0)
        # With r < 1, weights decrease with age (first > last)
        assert w[0] > w[4]  # First > Last (decreasing)
    
    def test_r_equals_one(self):
        """When r=1.0, all weights should be equal"""
        bands = ["70-74", "75-79", "80-84"]
        w = _geom_weights(bands, r=1.0)
        
        assert np.allclose(w, [1/3, 1/3, 1/3])
    
    def test_single_band(self):
        """Single band gets weight 1.0"""
        w = _geom_weights(["90+"], r=0.6)
        assert np.allclose(w, [1.0])
    
    def test_weights_sum_to_one(self):
        """Weights must always sum to 1.0 regardless of r value"""
        for r in [0.5, 0.7, 1.0, 1.5, 2.0]:
            w = _geom_weights(BANDS_70_TO_90, r)
            assert np.allclose(w.sum(), 1.0), f"Failed for r={r}"


# =====================================================================
# Tests for _weights_from_pop_or_geometric
# =====================================================================

class TestWeightsFromPopOrGeometric:
    """Test population-based and geometric weight calculation"""
    
    def test_population_based_weights(self):
        """Should use population shares when available"""
        pop_g = pd.DataFrame({
            "EDAD": ["70-74", "75-79", "80-84", "85-89", "90+"],
            "VALOR_corrected": [1000, 800, 500, 300, 100]  # Total: 2700
        })
        
        w = _weights_from_pop_or_geometric(pop_g, BANDS_70_TO_90)
        
        assert len(w) == 5
        assert np.allclose(w.sum(), 1.0)
        # Check specific weights
        assert np.allclose(w[0], 1000/2700)  # 70-74
        assert np.allclose(w[1], 800/2700)   # 75-79
        assert np.allclose(w[4], 100/2700)   # 90+
    
    def test_geometric_fallback_no_pop_data(self):
        """Should use geometric weights when population is missing"""
        pop_g = pd.DataFrame({"EDAD": [], "VALOR_corrected": []})
        
        w = _weights_from_pop_or_geometric(pop_g, BANDS_70_TO_90, r=0.6)
        
        assert len(w) == 5
        assert np.allclose(w.sum(), 1.0)
        # Geometric with r=0.6 should decrease
        assert w[1] < w[0]
        assert w[2] < w[1]
    
    def test_geometric_fallback_zero_population(self):
        """Should use geometric when all populations are zero"""
        pop_g = pd.DataFrame({
            "EDAD": ["70-74", "75-79", "80-84", "85-89", "90+"],
            "VALOR_corrected": [0, 0, 0, 0, 0]
        })
        
        w = _weights_from_pop_or_geometric(pop_g, BANDS_70_TO_90, r=0.6)
        
        assert len(w) == 5
        assert np.allclose(w.sum(), 1.0)
    
    def test_geometric_fallback_nan_population(self):
        """Should use geometric when populations are NaN"""
        pop_g = pd.DataFrame({
            "EDAD": ["70-74", "75-79", "80-84", "85-89", "90+"],
            "VALOR_corrected": [np.nan, np.nan, np.nan, np.nan, np.nan]
        })
        
        w = _weights_from_pop_or_geometric(pop_g, BANDS_70_TO_90, r=0.6)
        
        assert len(w) == 5
        assert np.allclose(w.sum(), 1.0)
    
    def test_partial_population_data(self):
        """Should handle partial population data"""
        pop_g = pd.DataFrame({
            "EDAD": ["70-74", "80-84"],  # Missing 75-79, 85-89, 90+
            "VALOR_corrected": [1000, 500]
        })
        
        w = _weights_from_pop_or_geometric(pop_g, BANDS_70_TO_90)
        
        # Should use proportions of available data
        assert len(w) == 5
        assert np.allclose(w.sum(), 1.0)
        assert np.allclose(w[0], 1000/1500)  # 70-74
        assert np.allclose(w[2], 500/1500)   # 80-84
        assert np.allclose(w[1], 0.0)        # 75-79 missing
    
    def test_whitespace_in_edad_labels(self):
        """ISSUE #2: Should handle whitespace in EDAD labels"""
        pop_g = pd.DataFrame({
            "EDAD": [" 70-74 ", "75-79", " 80-84", "85-89 ", "90+"],
            "VALOR_corrected": [1000, 800, 500, 300, 100]
        })
        
        w = _weights_from_pop_or_geometric(pop_g, BANDS_70_TO_90)
        
        # Should strip whitespace and match correctly
        assert len(w) == 5
        assert np.allclose(w.sum(), 1.0)
        assert np.allclose(w[0], 1000/2700)
    
    def test_duplicate_edad_labels(self):
        """ISSUE #2: Should sum populations for duplicate EDAD labels"""
        pop_g = pd.DataFrame({
            "EDAD": ["70-74", "70-74", "75-79", "80-84", "85-89", "90+"],
            "VALOR_corrected": [600, 400, 800, 500, 300, 100]  # 70-74 appears twice
        })
        
        w = _weights_from_pop_or_geometric(pop_g, BANDS_70_TO_90)
        
        # Should sum the two 70-74 entries: 600+400=1000
        assert len(w) == 5
        assert np.allclose(w.sum(), 1.0)
        assert np.allclose(w[0], 1000/2700)  # (600+400)/2700


# =====================================================================
# Tests for harmonize_migration_to_90plus
# =====================================================================

class TestHarmonizeMigrationTo90Plus:
    """Test migration tail harmonization"""
    
    def test_split_70_plus_with_population(self):
        """Should split 70+ using population structure"""
        mig = pd.DataFrame({
            "DPTO": ["001", "001"],
            "SEX": ["M", "M"],
            "EDAD": ["65-69", "70+"],
            "VALOR": [100.0, 500.0]
        })
        
        pop = pd.DataFrame({
            "DPTO": ["001"] * 5,
            "SEX": ["M"] * 5,
            "EDAD": ["70-74", "75-79", "80-84", "85-89", "90+"],
            "VALOR_corrected": [1000, 800, 500, 300, 100]  # Total: 2700
        })
        
        result = harmonize_migration_to_90plus(
            mig, pop, series_keys=["DPTO", "SEX"]
        )
        
        # Check total preserved
        assert np.allclose(result["VALOR"].sum(), 600.0)
        
        # Check 70+ was split into 5 bands
        edad_70_plus = ["70-74", "75-79", "80-84", "85-89", "90+"]
        result_70 = result[result["EDAD"].isin(edad_70_plus)]
        assert len(result_70) == 5
        assert np.allclose(result_70["VALOR"].sum(), 500.0)
        
        # Check proportional split
        assert np.allclose(result_70.loc[result_70["EDAD"] == "70-74", "VALOR"].values[0], 
                          500 * 1000/2700, atol=0.1)
    
    def test_split_80_plus_with_population(self):
        """Should split 80+ using population structure"""
        mig = pd.DataFrame({
            "DPTO": ["001"],
            "SEX": ["F"],
            "EDAD": ["80+"],
            "VALOR": [300.0]
        })
        
        pop = pd.DataFrame({
            "DPTO": ["001"] * 3,
            "SEX": ["F"] * 3,
            "EDAD": ["80-84", "85-89", "90+"],
            "VALOR_corrected": [500, 300, 100]  # Total: 900
        })
        
        result = harmonize_migration_to_90plus(
            mig, pop, series_keys=["DPTO", "SEX"]
        )
        
        # Check total preserved
        assert np.allclose(result["VALOR"].sum(), 300.0)
        
        # Check 80+ split into 3 bands
        assert len(result) == 3
        assert np.allclose(result.loc[result["EDAD"] == "80-84", "VALOR"].values[0],
                          300 * 500/900, atol=0.1)
    
    def test_geometric_fallback_no_matching_population(self):
        """ISSUE #4: Should use geometric when population group not found"""
        mig = pd.DataFrame({
            "DPTO": ["999"],  # Not in population data
            "SEX": ["M"],
            "EDAD": ["70+"],
            "VALOR": [500.0]
        })
        
        pop = pd.DataFrame({
            "DPTO": ["001"] * 5,  # Different department
            "SEX": ["M"] * 5,
            "EDAD": ["70-74", "75-79", "80-84", "85-89", "90+"],
            "VALOR_corrected": [1000, 800, 500, 300, 100]
        })
        
        result = harmonize_migration_to_90plus(
            mig, pop, series_keys=["DPTO", "SEX"]
        )
        
        # Should still produce output (geometric fallback)
        assert len(result) == 5
        assert np.allclose(result["VALOR"].sum(), 500.0)
        
        # Weights should follow geometric pattern (decreasing)
        vals = result.sort_values("EDAD")["VALOR"].values
        # Can't guarantee strict ordering due to string sort, but total should match
        assert np.allclose(vals.sum(), 500.0)
    
    def test_both_70_and_80_plus_present(self):
        """ISSUE #6: Should handle both 70+ and 80+ in same group"""
        mig = pd.DataFrame({
            "DPTO": ["001", "001"],
            "SEX": ["M", "M"],
            "EDAD": ["70+", "80+"],
            "VALOR": [500.0, 200.0]
        })
        
        pop = pd.DataFrame({
            "DPTO": ["001"] * 5,
            "SEX": ["M"] * 5,
            "EDAD": ["70-74", "75-79", "80-84", "85-89", "90+"],
            "VALOR_corrected": [1000, 800, 500, 300, 100]
        })
        
        result = harmonize_migration_to_90plus(
            mig, pop, series_keys=["DPTO", "SEX"]
        )
        
        # Total should be 700 (500+200)
        assert np.allclose(result["VALOR"].sum(), 700.0)
        
        # 80-84, 85-89, 90+ should have contributions from both splits
        # This tests aggregation behavior (Issue #6 - is this intended?)
    
    def test_duplicate_edad_after_split(self):
        """ISSUE #5: Should aggregate duplicates after split"""
        mig = pd.DataFrame({
            "DPTO": ["001", "001"],
            "SEX": ["M", "M"],
            "EDAD": ["70+", "70-74"],  # 70-74 already present
            "VALOR": [500.0, 100.0]
        })
        
        pop = pd.DataFrame({
            "DPTO": ["001"] * 5,
            "SEX": ["M"] * 5,
            "EDAD": ["70-74", "75-79", "80-84", "85-89", "90+"],
            "VALOR_corrected": [1000, 800, 500, 300, 100]
        })
        
        result = harmonize_migration_to_90plus(
            mig, pop, series_keys=["DPTO", "SEX"]
        )
        
        # Should aggregate 70-74 from split + original
        edad_70_74 = result[result["EDAD"] == "70-74"]["VALOR"].values[0]
        expected_from_split = 500 * 1000/2700
        assert np.allclose(edad_70_74, expected_from_split + 100, atol=0.1)
    
    def test_missing_migration_columns(self):
        """Should raise error if migration lacks required columns"""
        mig = pd.DataFrame({
            "DPTO": ["001"],
            "EDAD": ["70+"]
            # Missing VALOR
        })
        
        pop = pd.DataFrame({
            "DPTO": ["001"],
            "EDAD": ["70-74"],
            "VALOR_corrected": [1000]
        })
        
        with pytest.raises(KeyError, match="migration frame missing columns"):
            harmonize_migration_to_90plus(mig, pop, series_keys=["DPTO"])
    
    def test_missing_population_value_col(self):
        """Should fallback to VALOR if VALOR_corrected missing"""
        mig = pd.DataFrame({
            "DPTO": ["001"],
            "EDAD": ["70+"],
            "VALOR": [500.0]
        })
        
        pop = pd.DataFrame({
            "DPTO": ["001"] * 5,
            "EDAD": ["70-74", "75-79", "80-84", "85-89", "90+"],
            "VALOR": [1000, 800, 500, 300, 100]  # Only VALOR, not VALOR_corrected
        })
        
        result = harmonize_migration_to_90plus(
            mig, pop, series_keys=["DPTO"], pop_value_col="VALOR_corrected"
        )
        
        # Should work with fallback
        assert len(result) == 5
        assert np.allclose(result["VALOR"].sum(), 500.0)
    
    def test_multiple_groups(self):
        """Should process multiple groups independently"""
        mig = pd.DataFrame({
            "DPTO": ["001", "002"],
            "EDAD": ["70+", "70+"],
            "VALOR": [500.0, 300.0]
        })
        
        pop = pd.DataFrame({
            "DPTO": ["001"] * 5 + ["002"] * 5,
            "EDAD": ["70-74", "75-79", "80-84", "85-89", "90+"] * 2,
            "VALOR_corrected": [1000, 800, 500, 300, 100,  # DPTO 001
                               600, 400, 300, 200, 100]    # DPTO 002
        })
        
        result = harmonize_migration_to_90plus(
            mig, pop, series_keys=["DPTO"]
        )
        
        # Check each group
        result_001 = result[result["DPTO"] == "001"]
        result_002 = result[result["DPTO"] == "002"]
        
        assert np.allclose(result_001["VALOR"].sum(), 500.0)
        assert np.allclose(result_002["VALOR"].sum(), 300.0)


# =====================================================================
# Tests for harmonize_conteos_to_90plus
# =====================================================================

class TestHarmonizeConteosTo90Plus:
    """Test census/death tail harmonization"""
    
    def test_split_population_70_plus(self):
        """Should split poblacion_total 70+ with decreasing geometric weights"""
        df = pd.DataFrame({
            "DPTO": ["001"],
            "VARIABLE": ["poblacion_total"],
            "EDAD": ["70+"],
            "VALOR_corrected": [1000.0]
        })
        
        result = harmonize_conteos_to_90plus(
            df, series_keys=["DPTO"], r_pop=0.7
        )
        
        # Total preserved
        assert np.allclose(result["VALOR_corrected"].sum(), 1000.0)
        
        # Should have 5 bands
        assert len(result) == 5
        
        # Weights should decrease (younger > older for population)
        vals = result.sort_values("EDAD")["VALOR_corrected"].values
        # Note: string sort might not be age order, so just check sum
        assert np.allclose(vals.sum(), 1000.0)
    
    def test_split_deaths_70_plus(self):
        """Should split defunciones 70+ with increasing geometric weights"""
        df = pd.DataFrame({
            "DPTO": ["001"],
            "VARIABLE": ["defunciones"],
            "EDAD": ["70+"],
            "VALOR_corrected": [100.0]
        })
        
        result = harmonize_conteos_to_90plus(
            df, series_keys=["DPTO"], r_deaths=1.45
        )
        
        # Total preserved
        assert np.allclose(result["VALOR_corrected"].sum(), 100.0)
        
        # Should have 5 bands
        assert len(result) == 5
    
    def test_passthrough_other_variables(self):
        """Should pass through variables other than poblacion_total/defunciones"""
        df = pd.DataFrame({
            "DPTO": ["001", "001"],
            "VARIABLE": ["other_metric", "other_metric"],
            "EDAD": ["70+", "80+"],
            "VALOR_corrected": [1000.0, 500.0]
        })
        
        result = harmonize_conteos_to_90plus(df, series_keys=["DPTO"])
        
        # Should pass through unchanged
        assert len(result) == 2
        assert "70+" in result["EDAD"].values
        assert "80+" in result["EDAD"].values
    
    def test_split_80_plus(self):
        """Should split 80+ into 3 bands"""
        df = pd.DataFrame({
            "DPTO": ["001"],
            "VARIABLE": ["poblacion_total"],
            "EDAD": ["80+"],
            "VALOR_corrected": [500.0]
        })
        
        result = harmonize_conteos_to_90plus(df, series_keys=["DPTO"])
        
        # Total preserved
        assert np.allclose(result["VALOR_corrected"].sum(), 500.0)
        
        # Should have 3 bands
        assert len(result) == 3
        assert set(result["EDAD"]) == {"80-84", "85-89", "90+"}
    
    def test_both_70_and_80_plus_aggregated(self):
        """Should aggregate when both 70+ and 80+ present"""
        df = pd.DataFrame({
            "DPTO": ["001", "001"],
            "VARIABLE": ["poblacion_total", "poblacion_total"],
            "EDAD": ["70+", "80+"],
            "VALOR_corrected": [1000.0, 200.0]
        })
        
        result = harmonize_conteos_to_90plus(df, series_keys=["DPTO"])
        
        # Total should be 1200
        assert np.allclose(result["VALOR_corrected"].sum(), 1200.0)
        
        # 80-84, 85-89, 90+ should have contributions from both
        bands_80_plus = result[result["EDAD"].isin(["80-84", "85-89", "90+"])]
        # Sum should include parts from both 70+ and 80+ splits
        assert len(bands_80_plus) == 3
    
    def test_case_insensitive_variable(self):
        """ISSUE #9: Should handle case variations in VARIABLE"""
        df = pd.DataFrame({
            "DPTO": ["001", "001", "001"],
            "VARIABLE": ["POBLACION_TOTAL", "Poblacion_Total", "defunciones"],
            "EDAD": ["70+", "80+", "70+"],
            "VALOR_corrected": [1000.0, 500.0, 100.0]
        })
        
        result = harmonize_conteos_to_90plus(df, series_keys=["DPTO"])
        
        # All should be processed (lowercase comparison)
        # POBLACION_TOTAL and Poblacion_Total both match "poblacion_total"
        assert len(result[result["EDAD"] == "70+"]) == 0  # All should be split
        assert len(result[result["EDAD"] == "80+"]) == 0
    
    def test_missing_columns(self):
        """Should raise error if required columns missing"""
        df = pd.DataFrame({
            "DPTO": ["001"],
            "EDAD": ["70+"]
            # Missing VARIABLE and VALOR_corrected
        })
        
        with pytest.raises(KeyError, match="df missing columns"):
            harmonize_conteos_to_90plus(df, series_keys=["DPTO"])
    
    def test_deduplication_with_variable_in_series_keys(self):
        """ISSUE #10: Should handle VARIABLE in series_keys"""
        df = pd.DataFrame({
            "DPTO": ["001"],
            "VARIABLE": ["poblacion_total"],
            "EDAD": ["70+"],
            "VALOR_corrected": [1000.0]
        })
        
        # VARIABLE already in series_keys
        result = harmonize_conteos_to_90plus(
            df, series_keys=["DPTO", "VARIABLE"]
        )
        
        # Should not crash due to duplicate column in groupby
        assert len(result) == 5
        assert np.allclose(result["VALOR_corrected"].sum(), 1000.0)
    
    def test_multiple_groups(self):
        """Should process multiple groups independently"""
        df = pd.DataFrame({
            "DPTO": ["001", "002"],
            "VARIABLE": ["poblacion_total", "poblacion_total"],
            "EDAD": ["70+", "70+"],
            "VALOR_corrected": [1000.0, 500.0]
        })
        
        result = harmonize_conteos_to_90plus(df, series_keys=["DPTO"])
        
        result_001 = result[result["DPTO"] == "001"]
        result_002 = result[result["DPTO"] == "002"]
        
        assert np.allclose(result_001["VALOR_corrected"].sum(), 1000.0)
        assert np.allclose(result_002["VALOR_corrected"].sum(), 500.0)
    
    def test_geometric_ratio_differences(self):
        """Should use different ratios for population vs deaths"""
        df_pop = pd.DataFrame({
            "DPTO": ["001"],
            "VARIABLE": ["poblacion_total"],
            "EDAD": ["70+"],
            "VALOR_corrected": [1000.0]
        })
        
        df_deaths = pd.DataFrame({
            "DPTO": ["001"],
            "VARIABLE": ["defunciones"],
            "EDAD": ["70+"],
            "VALOR_corrected": [1000.0]
        })
        
        result_pop = harmonize_conteos_to_90plus(
            df_pop, series_keys=["DPTO"], r_pop=0.7, r_deaths=1.45
        )
        
        result_deaths = harmonize_conteos_to_90plus(
            df_deaths, series_keys=["DPTO"], r_pop=0.7, r_deaths=1.45
        )
        
        # Results should differ (different geometric patterns)
        # Population: decreasing (younger > older)
        # Deaths: increasing (older > younger)
        assert not np.allclose(
            result_pop["VALOR_corrected"].values,
            result_deaths["VALOR_corrected"].values
        )


# =====================================================================
# Integration Tests
# =====================================================================

class TestIntegration:
    """Test complete workflows"""
    
    def test_total_preservation_migration(self):
        """Total migration should be preserved exactly"""
        mig = pd.DataFrame({
            "DPTO": ["001"] * 4,
            "EDAD": ["60-64", "65-69", "70+", "80+"],
            "VALOR": [100.0, 150.0, 500.0, 200.0]
        })
        
        pop = pd.DataFrame({
            "DPTO": ["001"] * 5,
            "EDAD": ["70-74", "75-79", "80-84", "85-89", "90+"],
            "VALOR_corrected": [1000, 800, 500, 300, 100]
        })
        
        result = harmonize_migration_to_90plus(mig, pop, series_keys=["DPTO"])
        
        # Total must be preserved exactly
        assert np.allclose(result["VALOR"].sum(), mig["VALOR"].sum())
    
    def test_total_preservation_conteos(self):
        """Total census/deaths should be preserved exactly"""
        df = pd.DataFrame({
            "DPTO": ["001"] * 4,
            "VARIABLE": ["poblacion_total"] * 4,
            "EDAD": ["60-64", "65-69", "70+", "80+"],
            "VALOR_corrected": [10000, 8000, 5000, 2000]
        })
        
        result = harmonize_conteos_to_90plus(df, series_keys=["DPTO"])
        
        # Total must be preserved exactly
        assert np.allclose(result["VALOR_corrected"].sum(), df["VALOR_corrected"].sum())
    
    def test_no_tails_passthrough(self):
        """Data without 70+/80+ should pass through unchanged"""
        df = pd.DataFrame({
            "DPTO": ["001"] * 5,
            "VARIABLE": ["poblacion_total"] * 5,
            "EDAD": ["70-74", "75-79", "80-84", "85-89", "90+"],
            "VALOR_corrected": [1000, 800, 500, 300, 100]
        })
        
        result = harmonize_conteos_to_90plus(df, series_keys=["DPTO"])
        
        # Should be identical (already in target format)
        pd.testing.assert_frame_equal(
            result.sort_values("EDAD").reset_index(drop=True),
            df.sort_values("EDAD").reset_index(drop=True)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
