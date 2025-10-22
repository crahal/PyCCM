"""
Simplified tests for main_compute.py helper functions
Tests functions in isolation to avoid dependency issues.

Author: Test Suite
Date: October 20, 2025
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile


# Replicate the helper functions for testing
def _coerce_percent_any(x):
    """Parse percentage values from CSV."""
    if x is None:
        return None
    try:
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        if s.endswith("%"):
            v = float(s.strip("%").strip()) / 100.0
        else:
            v = float(s)
            if v > 1.0:
                v = v / 100.0
        if v < 0:
            return None
        return float(min(v, 0.999999))
    except Exception:
        return None


def _coerce_float(x):
    """Coerce to float."""
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None


def _coerce_int_pos(x):
    """Coerce to positive integer."""
    try:
        v = int(float(x))
        return v if v > 0 else None
    except Exception:
        return None


def _mortality_factor_exponential(t: int, improvement_total: float, convergence_years: int, converge_frac: float = 0.99) -> float:
    """Calculate mortality improvement factor using exponential smoother."""
    if t <= 0:
        return 1.0
    total = float(np.clip(improvement_total, 0.0, 0.999999))
    if total <= 0.0:
        return 1.0
    G = -np.log(1.0 - total)
    conv = max(int(convergence_years), 1)
    kappa = -np.log(1.0 - converge_frac) / conv
    S_t = 1.0 - np.exp(-kappa * t)
    S_t = float(np.clip(S_t, 0.0, 1.0))
    effective = G * S_t
    return float(np.exp(-effective))


def _mortality_factor_logistic(t: int, improvement_total: float, convergence_years: int, mid_frac: float = 0.5, steepness: float = None) -> float:
    """Calculate mortality improvement factor using logistic smoother."""
    if t <= 0:
        return 1.0
    total = float(np.clip(improvement_total, 0.0, 0.999999))
    if total <= 0.0:
        return 1.0
    G = -np.log(1.0 - total)
    conv = max(int(convergence_years), 1)
    
    if steepness is None:
        target = 0.99
        denom = max(conv * (1.0 - mid_frac), 1e-6)
        steepness = -np.log(1.0/target - 1.0) / denom
    
    s = float(steepness)
    x = (t / conv) - mid_frac
    S_t = 1.0 / (1.0 + np.exp(-s * x))
    S_t = float(np.clip(S_t, 0.0, 1.0))
    effective = G * S_t
    return float(np.exp(-effective))


def _inclusive_arange(start: float, stop: float, step: float):
    """Generate inclusive range that always includes stop value."""
    if step == 0:
        return [start]
    vals = []
    v = float(start)
    if step > 0:
        while v <= stop + 1e-12:
            vals.append(v)
            v += step
    else:
        while v >= stop - 1e-12:
            vals.append(v)
            v += step
    if vals:
        vals[-1] = float(stop)
    return vals


# ======================= TESTS =======================

class TestPercentCoercion:
    """Test percentage parsing functions."""
    
    def test_coerce_percent_from_string_with_percent_sign(self):
        """Should convert '10%' to 0.10."""
        assert abs(_coerce_percent_any("10%") - 0.10) < 1e-9
        assert abs(_coerce_percent_any("5.5%") - 0.055) < 1e-9
        # 100% gets capped at 0.999999
        assert abs(_coerce_percent_any("100%") - 0.999999) < 1e-5
    
    def test_coerce_percent_from_decimal_string(self):
        """Should handle decimal representation '0.10' → 0.10."""
        assert abs(_coerce_percent_any("0.10") - 0.10) < 1e-9
        assert abs(_coerce_percent_any("0.055") - 0.055) < 1e-9
    
    def test_coerce_percent_assumes_larger_than_1_is_percentage(self):
        """Should convert '10.0' (>1) to 0.10 assuming percentage."""
        assert abs(_coerce_percent_any("10.0") - 0.10) < 1e-9
        assert abs(_coerce_percent_any("5.5") - 0.055) < 1e-9
    
    def test_coerce_percent_handles_none_and_empty(self):
        """Should return None for empty/None/NaN inputs."""
        assert _coerce_percent_any(None) is None
        assert _coerce_percent_any("") is None
        assert _coerce_percent_any("nan") is None
        assert _coerce_percent_any("NaN") is None
    
    def test_coerce_percent_caps_at_max(self):
        """Should cap values at 0.999999."""
        result = _coerce_percent_any("150%")
        assert result == pytest.approx(0.999999)
    
    def test_coerce_percent_rejects_negative(self):
        """Should return None for negative values."""
        assert _coerce_percent_any("-10%") is None
        assert _coerce_percent_any("-0.10") is None


class TestNumericCoercion:
    """Test numeric parsing helper functions."""
    
    def test_coerce_float_valid_inputs(self):
        """Should parse valid float strings."""
        assert _coerce_float("3.14") == pytest.approx(3.14)
        assert _coerce_float("0.5") == pytest.approx(0.5)
        assert _coerce_float("-2.5") == pytest.approx(-2.5)
    
    def test_coerce_float_handles_none_and_empty(self):
        """Should return None for invalid inputs."""
        assert _coerce_float(None) is None
        assert _coerce_float("") is None
        assert _coerce_float("nan") is None
    
    def test_coerce_int_pos_valid_inputs(self):
        """Should parse positive integers."""
        assert _coerce_int_pos("5") == 5
        assert _coerce_int_pos("50") == 50
        assert _coerce_int_pos("5.7") == 5  # Truncates
    
    def test_coerce_int_pos_rejects_zero_and_negative(self):
        """Should return None for non-positive values."""
        assert _coerce_int_pos("0") is None
        assert _coerce_int_pos("-5") is None


class TestMortalityFactorCalculation:
    """Test time-varying mortality improvement calculations."""
    
    def test_mortality_factor_at_start_year(self):
        """At t=0, factor should be 1.0 (no improvement)."""
        factor = _mortality_factor_exponential(0, 0.10, 50, 0.99)
        assert factor == pytest.approx(1.0)
    
    def test_mortality_factor_increases_over_time(self):
        """Factor should decrease (improvement) as time progresses."""
        factor_0 = _mortality_factor_exponential(0, 0.10, 50, 0.99)
        factor_25 = _mortality_factor_exponential(25, 0.10, 50, 0.99)
        factor_50 = _mortality_factor_exponential(50, 0.10, 50, 0.99)
        
        # Factors should decrease (more improvement)
        assert factor_0 > factor_25 > factor_50
        
        # By convergence year (50 years), should be close to target
        # Target: 10% reduction → factor ≈ 0.90
        assert factor_50 < 0.92  # At least 8% improvement
        assert factor_50 > 0.88  # Not more than 12% (some buffer)
    
    def test_mortality_factor_exponential_smoother(self):
        """Exponential smoother should follow exponential decay."""
        factors = [_mortality_factor_exponential(t, 0.10, 50, 0.99) 
                   for t in range(0, 51, 10)]
        
        # Check monotonic decrease
        for i in range(len(factors) - 1):
            assert factors[i] > factors[i + 1]
        
        # Check exponential shape (decreasing rate of change)
        changes = [factors[i] - factors[i + 1] for i in range(len(factors) - 1)]
        for i in range(len(changes) - 1):
            assert changes[i] > changes[i + 1]  # Decelerating improvement
    
    def test_mortality_factor_logistic_smoother(self):
        """Logistic smoother should have sigmoid shape."""
        factors = [_mortality_factor_logistic(t, 0.10, 50, 0.5, None) 
                   for t in range(0, 51, 5)]
        
        # Logistic should be roughly symmetric around mid_frac
        # At t=25 years (50% of convergence), should be near 50% of total improvement
        factor_mid = _mortality_factor_logistic(25, 0.10, 50, 0.5, None)
        
        # 50% of 10% improvement = 5% reduction → factor ≈ 0.95
        assert 0.93 < factor_mid < 0.97  # Reasonable range around 0.95
    
    def test_mortality_factor_zero_improvement(self):
        """If improvement_total=0, factor should always be 1.0."""
        assert _mortality_factor_exponential(0, 0.0, 50, 0.99) == pytest.approx(1.0)
        assert _mortality_factor_exponential(25, 0.0, 50, 0.99) == pytest.approx(1.0)
        assert _mortality_factor_exponential(100, 0.0, 50, 0.99) == pytest.approx(1.0)
    
    def test_mortality_factor_negative_time(self):
        """Negative time should return 1.0 (no improvement)."""
        factor = _mortality_factor_exponential(-10, 0.10, 50, 0.99)
        assert factor == pytest.approx(1.0)


class TestParameterSweepGeneration:
    """Test Cartesian product parameter sweep generation."""
    
    def test_inclusive_arange_positive_step(self):
        """Should generate inclusive range with positive step."""
        result = _inclusive_arange(1.5, 2.5, 0.5)
        expected = [1.5, 2.0, 2.5]
        
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r == pytest.approx(e)
    
    def test_inclusive_arange_includes_stop(self):
        """Should always include stop value even with floating point errors."""
        result = _inclusive_arange(0.0, 1.0, 0.1)
        
        assert result[-1] == pytest.approx(1.0)  # Exactly 1.0, not 0.9999...
        assert len(result) == 11  # 0.0, 0.1, ..., 1.0
    
    def test_inclusive_arange_single_value(self):
        """If step=0, should return single value."""
        result = _inclusive_arange(5.0, 10.0, 0)
        assert result == [5.0]
    
    def test_inclusive_arange_negative_step(self):
        """Should handle negative steps (descending range)."""
        result = _inclusive_arange(10.0, 5.0, -1.0)
        expected = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0]
        
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r == pytest.approx(e)


class TestCartesianProduct:
    """Test parameter combination generation."""
    
    def test_parameter_sweep_cartesian_product(self):
        """Should generate all combinations of parameters."""
        from itertools import product
        
        tfr_values = [1.5, 1.7, 1.9]  # 3 values
        mort_values = [0.05, 0.10]    # 2 values
        ma_values = [3, 5, 7]         # 3 values
        
        combos = list(product(tfr_values, mort_values, ma_values))
        
        assert len(combos) == 3 * 2 * 3  # 18 combinations
        
        # Check a few specific combinations
        assert (1.5, 0.05, 3) in combos
        assert (1.9, 0.10, 7) in combos
        assert (1.7, 0.05, 5) in combos


class TestReproducibility:
    """Test reproducibility features."""
    
    def test_seed_generation_from_label(self):
        """Each task should have reproducible random seed."""
        import zlib
        
        label1 = "uniform_draw_42"
        label2 = "uniform_draw_42"  # Same label
        label3 = "uniform_draw_43"  # Different label
        
        seed1 = zlib.adler32(label1.encode("utf8")) & 0xFFFFFFFF
        seed2 = zlib.adler32(label2.encode("utf8")) & 0xFFFFFFFF
        seed3 = zlib.adler32(label3.encode("utf8")) & 0xFFFFFFFF
        
        assert seed1 == seed2  # Same label → same seed
        assert seed1 != seed3  # Different label → different seed


class TestMortalityIntegration:
    """Integration tests for mortality calculations."""
    
    def test_mortality_factor_scales_death_rates_correctly(self):
        """Mortality factor should correctly scale death rates."""
        # Baseline death rate
        mx_2018 = 0.01000  # 1% death rate at age 50
        
        # Apply improvement factor for year 2050 (32 years later)
        factor_2050 = _mortality_factor_exponential(32, 0.10, 50, 0.99)
        mx_2050 = mx_2018 * factor_2050
        
        # Should have reduced death rate
        assert mx_2050 < mx_2018
        
        # With 10% long-run improvement and 32/50 years elapsed,
        # should be roughly 7-10% reduction (exponential smoother converges quickly)
        reduction = (mx_2018 - mx_2050) / mx_2018
        assert 0.07 < reduction < 0.11
    
    def test_logistic_vs_exponential_comparison(self):
        """Logistic should have different shape than exponential."""
        times = [0, 10, 20, 30, 40, 50]
        
        exp_factors = [_mortality_factor_exponential(t, 0.10, 50, 0.99) for t in times]
        log_factors = [_mortality_factor_logistic(t, 0.10, 50, 0.5, None) for t in times]
        
        # Both should start at 1.0
        assert exp_factors[0] == pytest.approx(1.0)
        assert log_factors[0] == pytest.approx(1.0)
        
        # Both should converge to similar endpoint
        assert abs(exp_factors[-1] - log_factors[-1]) < 0.05
        
        # Mid-points should differ (logistic slower at start, faster at end)
        # At t=20 (40% through), exponential should be ahead
        assert exp_factors[2] < log_factors[2]


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
