"""
Comprehensive tests for fertility.py

Tests cover:
- get_target_params: CSV loading, column detection, error handling
- compute_asfr: Rate calculation, alignment, edge cases
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from io import StringIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fertility import get_target_params, compute_asfr


# =====================================================================
# Tests for get_target_params
# =====================================================================

class TestGetTargetParams:
    """Test target TFR parameter loading"""
    
    def test_standard_format(self, tmp_path):
        """Should load standard CSV format"""
        csv_content = """DPTO_NOMBRE,Target_TFR
BOGOTA,0.8
ANTIOQUIA,0.9
AMAZONAS,1.0
"""
        csv_file = tmp_path / "targets.csv"
        csv_file.write_text(csv_content)
        
        targets, conv_years = get_target_params(str(csv_file))
        
        assert len(targets) == 3
        assert targets["BOGOTA"] == 0.8
        assert targets["ANTIOQUIA"] == 0.9
        assert targets["AMAZONAS"] == 1.0
        assert len(conv_years) == 0  # No convergence column
    
    def test_with_convergence_years(self, tmp_path):
        """Should load convergence years when present"""
        csv_content = """DPTO_NOMBRE,Target_TFR,convergence_year
BOGOTA,0.8,2040
ANTIOQUIA,0.9,2045
AMAZONAS,1.0,2050
"""
        csv_file = tmp_path / "targets.csv"
        csv_file.write_text(csv_content)
        
        targets, conv_years = get_target_params(str(csv_file))
        
        assert len(targets) == 3
        assert len(conv_years) == 3
        assert conv_years["BOGOTA"] == 2040
        assert conv_years["ANTIOQUIA"] == 2045
        assert conv_years["AMAZONAS"] == 2050
    
    def test_flexible_column_names(self, tmp_path):
        """Should detect columns with flexible naming"""
        # Misspelled "convergence"
        csv_content = """dpto_nombre,target_tfr,convergeance_year
BOGOTA,0.8,2040
ANTIOQUIA,0.9,2045
"""
        csv_file = tmp_path / "targets.csv"
        csv_file.write_text(csv_content)
        
        targets, conv_years = get_target_params(str(csv_file))
        
        assert len(targets) == 2
        assert targets["BOGOTA"] == 0.8
        assert len(conv_years) == 2
        assert conv_years["BOGOTA"] == 2040
    
    def test_whitespace_in_department_names(self, tmp_path):
        """Should strip whitespace from department names"""
        csv_content = """DPTO_NOMBRE,Target_TFR
  BOGOTA  ,0.8
ANTIOQUIA   ,0.9
   AMAZONAS,1.0
"""
        csv_file = tmp_path / "targets.csv"
        csv_file.write_text(csv_content)
        
        targets, conv_years = get_target_params(str(csv_file))
        
        assert "BOGOTA" in targets  # Stripped
        assert "  BOGOTA  " not in targets
        assert targets["BOGOTA"] == 0.8
    
    def test_invalid_tfr_values(self, tmp_path):
        """Should skip invalid TFR values"""
        csv_content = """DPTO_NOMBRE,Target_TFR
BOGOTA,0.8
ANTIOQUIA,invalid
AMAZONAS,
CAUCA,1.2
"""
        csv_file = tmp_path / "targets.csv"
        csv_file.write_text(csv_content)
        
        targets, conv_years = get_target_params(str(csv_file))
        
        assert len(targets) == 2  # Only valid ones
        assert "BOGOTA" in targets
        assert "CAUCA" in targets
        assert "ANTIOQUIA" not in targets  # Skipped invalid
        assert "AMAZONAS" not in targets  # Skipped empty
    
    def test_nan_and_inf_tfr_values(self, tmp_path):
        """Should skip NaN and Inf TFR values"""
        csv_content = """DPTO_NOMBRE,Target_TFR
BOGOTA,0.8
ANTIOQUIA,NaN
AMAZONAS,inf
CAUCA,1.2
"""
        csv_file = tmp_path / "targets.csv"
        csv_file.write_text(csv_content)
        
        targets, conv_years = get_target_params(str(csv_file))
        
        assert len(targets) == 2
        assert "BOGOTA" in targets
        assert "CAUCA" in targets
        assert "ANTIOQUIA" not in targets
        assert "AMAZONAS" not in targets
    
    def test_invalid_convergence_years(self, tmp_path):
        """Should skip invalid convergence years"""
        csv_content = """DPTO_NOMBRE,Target_TFR,convergence_year
BOGOTA,0.8,2040
ANTIOQUIA,0.9,invalid
AMAZONAS,1.0,
CAUCA,1.2,-1
CESAR,1.3,0
CHOCO,1.4,2050
"""
        csv_file = tmp_path / "targets.csv"
        csv_file.write_text(csv_content)
        
        targets, conv_years = get_target_params(str(csv_file))
        
        # All TFRs should load
        assert len(targets) == 6
        
        # Only valid convergence years (> 0)
        assert len(conv_years) == 2
        assert "BOGOTA" in conv_years
        assert "CHOCO" in conv_years
        assert "ANTIOQUIA" not in conv_years
        assert "CAUCA" not in conv_years  # -1 rejected
        assert "CESAR" not in conv_years  # 0 rejected
    
    def test_missing_dpto_column(self, tmp_path):
        """Should raise error if department column missing"""
        csv_content = """Target_TFR
0.8
0.9
"""
        csv_file = tmp_path / "targets.csv"
        csv_file.write_text(csv_content)
        
        with pytest.raises(KeyError, match="Could not find 'DPTO_NOMBRE'"):
            get_target_params(str(csv_file))
    
    def test_missing_tfr_column(self, tmp_path):
        """Should raise error if TFR column missing"""
        csv_content = """DPTO_NOMBRE
BOGOTA
ANTIOQUIA
"""
        csv_file = tmp_path / "targets.csv"
        csv_file.write_text(csv_content)
        
        with pytest.raises(KeyError, match="Could not find 'Target_TFR'"):
            get_target_params(str(csv_file))
    
    def test_alternative_column_detection(self, tmp_path):
        """Should detect alternative column names"""
        csv_content = """departamento_dpto_nombre,tfr_target,years_for_convergence
BOGOTA,0.8,2040
"""
        csv_file = tmp_path / "targets.csv"
        csv_file.write_text(csv_content)
        
        targets, conv_years = get_target_params(str(csv_file))
        
        assert len(targets) == 1
        assert targets["BOGOTA"] == 0.8
        assert len(conv_years) == 1
        assert conv_years["BOGOTA"] == 2040
    
    def test_empty_csv(self, tmp_path):
        """Should handle empty CSV gracefully"""
        csv_content = """DPTO_NOMBRE,Target_TFR
"""
        csv_file = tmp_path / "targets.csv"
        csv_file.write_text(csv_content)
        
        targets, conv_years = get_target_params(str(csv_file))
        
        assert len(targets) == 0
        assert len(conv_years) == 0
    
    def test_real_example_file(self):
        """Should load the actual example file"""
        example_file = Path(__file__).parent.parent / "data" / "target_tfrs_example.csv"
        
        if example_file.exists():
            targets, conv_years = get_target_params(str(example_file))
            
            assert len(targets) > 0
            # Check some expected departments
            assert all(isinstance(v, float) for v in targets.values())
            assert all(v > 0 for v in targets.values())  # TFRs should be positive


# =====================================================================
# Tests for compute_asfr
# =====================================================================

class TestComputeASFR:
    """Test ASFR calculation"""
    
    def test_basic_calculation(self):
        """Should compute ASFR = births / population"""
        ages = ["20", "25", "30"]
        population = pd.Series([1000, 1200, 1100], index=ages)
        births = pd.Series([80, 120, 88], index=ages)
        
        result = compute_asfr(ages, population, births)
        
        assert len(result) == 3
        assert np.allclose(result.loc["20", "asfr"], 80/1000)
        assert np.allclose(result.loc["25", "asfr"], 120/1200)
        assert np.allclose(result.loc["30", "asfr"], 88/1100)
    
    def test_alignment_by_labels(self):
        """Should align population and births by age labels"""
        ages = ["20", "25", "30", "35"]
        # Population has different order
        population = pd.Series([1100, 1000, 1300, 1200], index=["30", "20", "35", "25"])
        births = pd.Series([88, 80, 104, 120], index=["30", "20", "35", "25"])
        
        result = compute_asfr(ages, population, births)
        
        assert len(result) == 4
        # Should compute correctly despite reordering
        assert np.allclose(result.loc["20", "asfr"], 80/1000)
        assert np.allclose(result.loc["25", "asfr"], 120/1200)
        assert np.allclose(result.loc["30", "asfr"], 88/1100)
        assert np.allclose(result.loc["35", "asfr"], 104/1300)
    
    def test_whitespace_in_labels(self):
        """Should handle whitespace in age labels"""
        ages = ["20", "25", "30"]
        population = pd.Series([1000, 1200, 1100], index=["20 ", " 25", "30  "])
        births = pd.Series([80, 120, 88], index=[" 20", "25 ", "  30"])
        
        result = compute_asfr(ages, population, births)
        
        # Should strip and match correctly
        assert len(result) == 3
        assert np.allclose(result.loc["20", "asfr"], 80/1000)
        assert np.allclose(result.loc["25", "asfr"], 120/1200)
        assert np.allclose(result.loc["30", "asfr"], 88/1100)
    
    def test_missing_ages_in_population(self):
        """Should only compute ASFR for ages present in both"""
        ages = ["20", "25", "30", "35"]
        population = pd.Series([1000, 1200], index=["20", "25"])  # Missing 30, 35
        births = pd.Series([80, 120, 88, 104], index=["20", "25", "30", "35"])
        
        result = compute_asfr(ages, population, births)
        
        # Should only have ages 20, 25 (intersection)
        assert len(result) == 2
        assert "20" in result.index
        assert "25" in result.index
        assert "30" not in result.index
        assert "35" not in result.index
    
    def test_missing_ages_in_births(self):
        """Should only compute ASFR for ages present in both"""
        ages = ["20", "25", "30", "35"]
        population = pd.Series([1000, 1200, 1100, 1300], index=["20", "25", "30", "35"])
        births = pd.Series([80, 120], index=["20", "25"])  # Missing 30, 35
        
        result = compute_asfr(ages, population, births)
        
        # Should only have ages 20, 25
        assert len(result) == 2
        assert "20" in result.index
        assert "25" in result.index
    
    def test_zero_population(self):
        """Should handle zero population (no division by zero)"""
        ages = ["20", "25", "30"]
        population = pd.Series([1000, 0, 1100], index=ages)
        births = pd.Series([80, 120, 88], index=ages)
        
        result = compute_asfr(ages, population, births)
        
        # Age 25 has zero population -> ASFR = 0 (not Inf)
        assert len(result) == 3
        assert np.allclose(result.loc["20", "asfr"], 80/1000)
        assert result.loc["25", "asfr"] == 0.0
        assert np.allclose(result.loc["30", "asfr"], 88/1100)
    
    def test_very_small_population(self):
        """Should treat very small populations as zero exposure"""
        ages = ["20", "25", "30"]
        population = pd.Series([1000, 1e-10, 1100], index=ages)
        births = pd.Series([80, 1, 88], index=ages)
        
        result = compute_asfr(ages, population, births, min_exposure=1e-9)
        
        # Age 25: population < min_exposure -> treated as zero
        assert result.loc["25", "asfr"] == 0.0
        assert np.isnan(result.loc["25", "population"])
    
    def test_negative_births(self):
        """Should clip negative births to zero"""
        ages = ["20", "25", "30"]
        population = pd.Series([1000, 1200, 1100], index=ages)
        births = pd.Series([80, -10, 88], index=ages)  # Negative births (data error)
        
        result = compute_asfr(ages, population, births)
        
        # Negative births clipped to zero
        assert result.loc["25", "births"] == 0.0
        assert result.loc["25", "asfr"] == 0.0
    
    def test_negative_population(self):
        """Should treat negative population as zero exposure"""
        ages = ["20", "25", "30"]
        population = pd.Series([1000, -100, 1100], index=ages)  # Negative (error)
        births = pd.Series([80, 120, 88], index=ages)
        
        result = compute_asfr(ages, population, births)
        
        # Negative population -> treated as missing
        assert result.loc["25", "asfr"] == 0.0
        assert np.isnan(result.loc["25", "population"])
    
    def test_nan_values(self):
        """Should handle NaN values gracefully"""
        ages = ["20", "25", "30"]
        population = pd.Series([1000, np.nan, 1100], index=ages)
        births = pd.Series([80, 120, np.nan], index=ages)
        
        result = compute_asfr(ages, population, births)
        
        # NaN propagates to ASFR = 0
        assert result.loc["25", "asfr"] == 0.0  # NaN population
        assert result.loc["30", "asfr"] == 0.0  # NaN births
    
    def test_nonneg_asfr_true(self):
        """Should clip negative ASFR to zero when nonneg_asfr=True"""
        ages = ["20"]
        population = pd.Series([1000], index=ages)
        births = pd.Series([-50], index=ages)  # Would give negative ASFR
        
        result = compute_asfr(ages, population, births, nonneg_asfr=True)
        
        assert result.loc["20", "asfr"] >= 0.0
    
    def test_nonneg_asfr_false(self):
        """Should allow negative ASFR when nonneg_asfr=False for debugging"""
        # This is theoretical - in practice births are already clipped
        # But tests the parameter exists and works
        ages = ["20"]
        population = pd.Series([1000], index=ages)
        births = pd.Series([50], index=ages)
        
        # Even with nonneg_asfr=False, births are clipped, so ASFR still positive
        result = compute_asfr(ages, population, births, nonneg_asfr=False)
        
        assert result.loc["20", "asfr"] >= 0.0  # Births clipped first
    
    def test_output_structure(self):
        """Should return DataFrame with correct columns"""
        ages = ["20", "25", "30"]
        population = pd.Series([1000, 1200, 1100], index=ages)
        births = pd.Series([80, 120, 88], index=ages)
        
        result = compute_asfr(ages, population, births)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["population", "births", "asfr"]
        assert list(result.index) == ages
    
    def test_realistic_fertility_pattern(self):
        """Should compute realistic fertility schedule"""
        ages = [str(a) for a in range(15, 50)]
        
        # Create realistic population (roughly constant by age)
        population = pd.Series([10000] * 35, index=ages)
        
        # Create realistic birth pattern (bell-shaped, peak at 25-30)
        births_vals = []
        for age in range(15, 50):
            if age < 20:
                rate = 0.02 * (age - 15) / 5  # Rising from 15-20
            elif age < 30:
                rate = 0.02 + 0.10 * (age - 20) / 10  # Peak 20-30
            elif age < 40:
                rate = 0.12 - 0.10 * (age - 30) / 10  # Declining 30-40
            else:
                rate = 0.02 - 0.02 * (age - 40) / 10  # Low 40-50
            births_vals.append(rate * 10000)
        
        births = pd.Series(births_vals, index=ages)
        
        result = compute_asfr(ages, population, births)
        
        # Check TFR (sum of ASFRs) is in realistic range
        tfr = result["asfr"].sum()
        assert 0.5 < tfr < 5.0, f"TFR {tfr:.2f} outside realistic bounds"
        
        # Check peak is in fertile ages (20-35)
        peak_age = result["asfr"].idxmax()
        assert 20 <= int(peak_age) <= 35, f"Peak fertility at age {peak_age} unusual"
    
    def test_empty_inputs(self):
        """Should handle empty inputs"""
        ages = []
        population = pd.Series([], dtype=float)
        births = pd.Series([], dtype=float)
        
        result = compute_asfr(ages, population, births)
        
        assert len(result) == 0
        assert list(result.columns) == ["population", "births", "asfr"]
    
    def test_single_age(self):
        """Should work with single age"""
        ages = ["25"]
        population = pd.Series([1200], index=ages)
        births = pd.Series([120], index=ages)
        
        result = compute_asfr(ages, population, births)
        
        assert len(result) == 1
        assert np.allclose(result.loc["25", "asfr"], 0.1)
    
    def test_preserve_age_order(self):
        """Should preserve age order from ages parameter"""
        ages = ["30", "20", "25"]  # Out of order
        population = pd.Series([1100, 1000, 1200], index=["30", "20", "25"])
        births = pd.Series([88, 80, 120], index=["30", "20", "25"])
        
        result = compute_asfr(ages, population, births)
        
        # Should preserve order from ages parameter
        assert list(result.index) == ["30", "20", "25"]


# =====================================================================
# Integration Tests
# =====================================================================

class TestIntegration:
    """Test complete workflows"""
    
    def test_full_workflow_with_target_params(self, tmp_path):
        """Test complete fertility projection workflow"""
        # 1. Create target TFR file
        csv_content = """DPTO_NOMBRE,Target_TFR,convergence_year
BOGOTA,0.8,2050
"""
        csv_file = tmp_path / "targets.csv"
        csv_file.write_text(csv_content)
        
        # 2. Load targets
        targets, conv_years = get_target_params(str(csv_file))
        
        assert targets["BOGOTA"] == 0.8
        assert conv_years["BOGOTA"] == 2050
        
        # 3. Compute current ASFR
        ages = [str(a) for a in range(15, 50)]
        population = pd.Series([10000] * 35, index=ages)
        births = pd.Series([100] * 35, index=ages)  # Simplified
        
        current_asfr = compute_asfr(ages, population, births)
        
        # 4. Calculate current TFR
        current_tfr = current_asfr["asfr"].sum()
        
        # 5. Project to target
        target_tfr = targets["BOGOTA"]
        scaling_factor = target_tfr / current_tfr
        projected_asfr = current_asfr["asfr"] * scaling_factor
        
        # 6. Verify projection
        projected_tfr = projected_asfr.sum()
        assert np.allclose(projected_tfr, target_tfr)
    
    def test_tfr_calculation_from_asfr(self):
        """Test TFR calculation matches demographic definition"""
        ages = [str(a) for a in range(15, 50)]
        population = pd.Series([10000] * 35, index=ages)
        
        # Known ASFR pattern with TFR = 2.0
        # Bell-shaped with peak at 25-30
        births_vals = []
        target_tfr = 2.0
        for i, age in enumerate(range(15, 50)):
            # Gaussian-like pattern
            peak_age = 27
            sigma = 7
            rate = (target_tfr / (sigma * np.sqrt(2 * np.pi))) * \
                   np.exp(-0.5 * ((age - peak_age) / sigma) ** 2)
            births_vals.append(rate * 10000)
        
        births = pd.Series(births_vals, index=ages)
        
        result = compute_asfr(ages, population, births)
        calculated_tfr = result["asfr"].sum()
        
        # Should be close to target (within numerical error)
        # Relaxed tolerance due to discretization
        assert np.allclose(calculated_tfr, target_tfr, rtol=0.05)


# =====================================================================
# Tests for validate_asfr
# =====================================================================

class TestValidateASFR:
    """Test biological plausibility validation"""
    
    def test_valid_asfr_no_warnings(self):
        """Should return no warnings for realistic ASFR"""
        from fertility import validate_asfr
        
        ages = [str(a) for a in range(15, 50)]
        # Realistic bell-shaped pattern
        asfr_vals = []
        for age in range(15, 50):
            if age < 20:
                rate = 0.02 * (age - 15) / 5  # Low at young ages
            elif age <= 30:
                rate = 0.02 + 0.10 * (age - 20) / 10  # Peak 20-30
            else:
                rate = 0.12 * np.exp(-(age - 30) / 10)  # Decline after 30
            asfr_vals.append(rate)
        
        asfr = pd.Series(asfr_vals, index=ages)
        warnings = validate_asfr(asfr, ages)
        
        assert len(warnings) == 0, f"Expected no warnings, got: {warnings}"
    
    def test_warns_on_age_under_10(self):
        """Should warn about fertility under age 10"""
        from fertility import validate_asfr
        
        ages = ['5', '10', '15', '20']
        asfr = pd.Series([0.05, 0.10, 0.15, 0.20], index=ages)
        
        warnings = validate_asfr(asfr, ages)
        
        assert len(warnings) >= 1
        assert any("Age 5" in w for w in warnings)
        assert any("outside reproductive ages" in w for w in warnings)
    
    def test_warns_on_age_over_55(self):
        """Should warn about fertility over age 55 (post-menopause)"""
        from fertility import validate_asfr
        
        ages = ['50', '55', '60']
        asfr = pd.Series([0.05, 0.03, 0.02], index=ages)
        
        warnings = validate_asfr(asfr, ages)
        
        assert len(warnings) >= 1
        assert any("Age 60" in w for w in warnings)
        assert any("Biologically implausible" in w for w in warnings)
    
    def test_warns_on_excessive_asfr(self):
        """Should warn when ASFR exceeds biological maximum (~0.40)"""
        from fertility import validate_asfr
        
        ages = [str(a) for a in range(20, 35)]
        asfr = pd.Series([0.50] * 15, index=ages)  # Impossible 50% rate
        
        warnings = validate_asfr(asfr, ages)
        
        assert len(warnings) >= 1
        assert any("exceeds biological maximum" in w for w in warnings)
        assert any("0.40" in w for w in warnings)
    
    def test_warns_on_excessive_tfr(self):
        """Should warn when TFR exceeds historical maximum (~10)"""
        from fertility import validate_asfr
        
        ages = [str(a) for a in range(15, 50)]
        asfr = pd.Series([0.35] * 35, index=ages)  # Would give TFR > 10
        
        warnings = validate_asfr(asfr, ages)
        
        assert len(warnings) >= 1
        assert any("exceeds historical maximum" in w for w in warnings)
    
    def test_warns_on_very_low_tfr(self):
        """Should warn when TFR is extremely low (<0.3)"""
        from fertility import validate_asfr
        
        ages = [str(a) for a in range(15, 50)]
        asfr = pd.Series([0.005] * 35, index=ages)  # TFR = 0.175
        
        warnings = validate_asfr(asfr, ages)
        
        assert len(warnings) >= 1
        assert any("below minimum observed" in w for w in warnings)
    
    def test_warns_on_unusual_peak_age(self):
        """Should warn when peak fertility is at unusual age"""
        from fertility import validate_asfr
        
        ages = [str(a) for a in range(15, 50)]
        asfr_vals = [0.01] * 35
        asfr_vals[0] = 0.20   # Peak at age 15 (unusual)
        asfr = pd.Series(asfr_vals, index=ages)
        
        warnings = validate_asfr(asfr, ages)
        
        assert len(warnings) >= 1
        assert any("Peak fertility at age 15" in w for w in warnings)
        assert any("unusual" in w for w in warnings)
    
    def test_compute_asfr_with_validation(self):
        """Should log warnings when validate=True and return result"""
        
        ages = [str(a) for a in range(15, 65)]  # Includes age 60-64 (too old)
        population = pd.Series([1000] * len(ages), index=ages)
        births = pd.Series([10] * len(ages), index=ages)  # Uniform (unrealistic)
        
        # Should still compute and return result even with validation warnings
        result = compute_asfr(ages, population, births, validate=True)
        
        # Should return result regardless of warnings
        assert not result.empty
        assert "asfr" in result.columns
        assert len(result) == len(ages)
        
        # Verify that validation would detect issues
        from fertility import validate_asfr
        warnings = validate_asfr(result['asfr'], ages)
        assert len(warnings) > 0, "Should have warnings for ages 60-64"
    
    def test_validation_error_mode(self):
        """Should raise ValueError when warnings_only=False"""
        from fertility import validate_asfr
        
        ages = ['5', '10', '60']
        asfr = pd.Series([0.1, 0.2, 0.1], index=ages)
        
        with pytest.raises(ValueError) as exc_info:
            validate_asfr(asfr, ages, warnings_only=False)
        
        assert "ASFR validation failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
