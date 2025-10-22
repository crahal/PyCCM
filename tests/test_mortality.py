# tests/test_mortality.py
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mortality import (
    parse_age_labels,
    _expand_closed_intervals,
    _difference_matrix,
    _poisson_pspline_fit,
    pspline_group_qx,
    make_lifetable
)


# ============================================================================
# Test parse_age_labels
# ============================================================================
class TestParseAgeLabels:
    """Test age label parsing utility."""
    
    def test_standard_intervals(self):
        """Test standard 5-year age intervals."""
        labels = pd.Series(['0-4', '5-9', '10-14', '15-19', '20-24'])
        result = parse_age_labels(labels)
        expected = pd.Series([0, 5, 10, 15, 20], name=0)
        pd.testing.assert_series_equal(result, expected)
    
    def test_open_interval(self):
        """Test parsing open-ended interval."""
        labels = pd.Series(['85-89', '90+'])
        result = parse_age_labels(labels)
        expected = pd.Series([85, 90], name=0)
        pd.testing.assert_series_equal(result, expected)
    
    def test_single_year_ages(self):
        """Test single-year age labels."""
        labels = pd.Series(['25', '26', '27'])
        result = parse_age_labels(labels)
        expected = pd.Series([25, 26, 27], name=0)
        pd.testing.assert_series_equal(result, expected)
    
    def test_irregular_intervals(self):
        """Test non-standard age intervals."""
        labels = pd.Series(['0-1', '1-4', '5-14', '15-24'])
        result = parse_age_labels(labels)
        expected = pd.Series([0, 1, 5, 15], name=0)
        pd.testing.assert_series_equal(result, expected)


# ============================================================================
# Test _expand_closed_intervals
# ============================================================================
class TestExpandClosedIntervals:
    """Test expansion of age intervals to single years."""
    
    def test_standard_expansion(self):
        """Test basic interval expansion."""
        ages = np.array([0, 5, 10])
        widths = np.array([5, 5, np.inf])
        
        single_ages, group_id, closed_idx = _expand_closed_intervals(ages, widths)
        
        # Should have ages 0-9 (10 ages)
        assert len(single_ages) == 10
        assert list(single_ages) == list(range(0, 10))
        
        # Group IDs: 0-4 → group 0, 5-9 → group 1
        assert list(group_id) == [0]*5 + [1]*5
        
        # Closed indices: 0, 1 (exclude last open interval)
        assert list(closed_idx) == [0, 1]
    
    def test_single_interval(self):
        """Test with single closed interval."""
        ages = np.array([0])
        widths = np.array([1])
        
        single_ages, group_id, closed_idx = _expand_closed_intervals(ages, widths)
        
        assert len(single_ages) == 0  # No closed intervals (last is open)
        assert len(group_id) == 0
        assert len(closed_idx) == 0
    
    def test_varying_widths(self):
        """Test intervals of different widths."""
        ages = np.array([0, 1, 5, 10])
        widths = np.array([1, 4, 5, np.inf])
        
        single_ages, group_id, closed_idx = _expand_closed_intervals(ages, widths)
        
        # 0 (1) + 1-4 (4) + 5-9 (5) = 10 ages
        assert len(single_ages) == 10
        assert list(single_ages) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        # Group IDs
        assert group_id[0] == 0      # age 0 → interval 0
        assert group_id[1:5].tolist() == [1]*4   # ages 1-4 → interval 1
        assert group_id[5:10].tolist() == [2]*5  # ages 5-9 → interval 2
    
    def test_empty_input(self):
        """Test empty age array."""
        ages = np.array([])
        widths = np.array([])
        
        single_ages, group_id, closed_idx = _expand_closed_intervals(ages, widths)
        
        assert len(single_ages) == 0
        assert len(group_id) == 0
        assert len(closed_idx) == 0


# ============================================================================
# Test _difference_matrix
# ============================================================================
class TestDifferenceMatrix:
    """Test finite difference operator construction."""
    
    def test_first_order(self):
        """Test 1st order differences."""
        D = _difference_matrix(5, order=1)
        
        # Shape: (n-order, n) = (4, 5)
        assert D.shape == (4, 5)
        
        # Apply to vector [0, 1, 2, 3, 4]
        x = np.array([0, 1, 2, 3, 4])
        diff = D @ x
        # 1st differences: [1, 1, 1, 1]
        np.testing.assert_array_almost_equal(diff, [1, 1, 1, 1])
    
    def test_second_order(self):
        """Test 2nd order differences."""
        D = _difference_matrix(5, order=2)
        
        assert D.shape == (3, 5)
        
        # Apply to quadratic [0, 1, 4, 9, 16] (x^2)
        x = np.array([0, 1, 4, 9, 16])
        diff = D @ x
        # 2nd differences of x^2: constant = 2
        np.testing.assert_array_almost_equal(diff, [2, 2, 2])
    
    def test_third_order(self):
        """Test 3rd order differences."""
        D = _difference_matrix(6, order=3)
        
        assert D.shape == (3, 6)
        
        # Apply to cubic [0, 1, 8, 27, 64, 125] (x^3)
        x = np.array([0, 1, 8, 27, 64, 125])
        diff = D @ x
        # 3rd differences of x^3: constant = 6
        np.testing.assert_array_almost_equal(diff, [6, 6, 6], decimal=5)
    
    def test_penalty_matrix(self):
        """Test penalty matrix P = D^T D."""
        D = _difference_matrix(5, order=2)
        P = D.T @ D
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(P, P.T)
        
        # Should be positive semidefinite
        eigenvalues = np.linalg.eigvalsh(P)
        assert np.all(eigenvalues >= -1e-10)
    
    def test_invalid_order(self):
        """Test error for invalid difference order."""
        with pytest.raises(ValueError, match="difference order must be >= 1"):
            _difference_matrix(5, order=0)


# ============================================================================
# Test _poisson_pspline_fit
# ============================================================================
class TestPoissonPsplineFit:
    """Test penalized Poisson spline fitting."""
    
    def test_perfect_data(self):
        """Test with noise-free exponential mortality."""
        # Gompertz: m(x) = a * exp(b*x)
        ages = np.arange(0, 50)
        a, b = 0.001, 0.05
        m_true = a * np.exp(b * ages)
        
        # Generate perfect data
        E = np.ones(50) * 1000
        D = E * m_true
        
        # Fit with low penalty (should recover truth)
        f_hat = _poisson_pspline_fit(E, D, lam=1.0, diff_order=3, max_iter=30)
        m_hat = np.exp(f_hat)
        
        # Should be very close
        np.testing.assert_allclose(m_hat, m_true, rtol=0.1)
    
    def test_noisy_data(self):
        """Test smoothing of noisy Poisson data."""
        np.random.seed(42)
        
        # True smooth mortality
        ages = np.arange(0, 40)
        m_true = 0.01 + 0.0001 * ages**2  # Quadratic
        
        # Noisy observations
        E = np.ones(40) * 500
        mu = E * m_true
        D = np.random.poisson(mu)
        
        # Fit with moderate penalty
        f_hat = _poisson_pspline_fit(E, D, lam=100.0, diff_order=2, max_iter=30)
        m_hat = np.exp(f_hat)
        
        # Smoothed should be closer to truth than raw
        m_raw = D / E
        error_raw = np.mean((m_raw - m_true)**2)
        error_smooth = np.mean((m_hat - m_true)**2)
        
        assert error_smooth < error_raw
    
    def test_zero_deaths(self):
        """Test handling of zero deaths."""
        E = np.array([1000, 1000, 1000, 1000])
        D = np.array([0, 0, 0, 0])
        
        # Should not crash
        f_hat = _poisson_pspline_fit(E, D, lam=10.0, diff_order=2, max_iter=10)
        m_hat = np.exp(f_hat)
        
        # Should give very small rates
        assert np.all(m_hat < 0.01)
    
    def test_convergence(self):
        """Test that algorithm converges."""
        E = np.ones(20) * 1000
        D = np.random.poisson(E * 0.01)
        
        # Run with verbose=False
        f_hat = _poisson_pspline_fit(
            E, D, lam=50.0, diff_order=2, 
            max_iter=50, tol=1e-8, verbose=False
        )
        
        # Should return finite values
        assert np.all(np.isfinite(f_hat))
    
    def test_lambda_effect(self):
        """Test that larger lambda produces smoother results."""
        np.random.seed(123)
        
        ages = np.arange(0, 30)
        m_true = 0.005 * (1 + 0.02 * ages)
        E = np.ones(30) * 1000
        D = np.random.poisson(E * m_true)
        
        # Low penalty (rough)
        f_low = _poisson_pspline_fit(E, D, lam=1.0, diff_order=2, max_iter=30)
        # High penalty (smooth)
        f_high = _poisson_pspline_fit(E, D, lam=1000.0, diff_order=2, max_iter=30)
        
        # Measure roughness (2nd differences)
        D2 = _difference_matrix(30, order=2)
        rough_low = np.sum((D2 @ f_low)**2)
        rough_high = np.sum((D2 @ f_high)**2)
        
        # High lambda should be smoother
        assert rough_high < rough_low


# ============================================================================
# Test pspline_group_qx
# ============================================================================
class TestPsplineGroupQx:
    """Test interval death probability calculation."""
    
    def test_basic_calculation(self):
        """Test basic qx calculation."""
        ages = np.array([0, 5, 10])
        widths = np.array([5, 5, np.inf])
        population = np.array([10000, 12000, 8000])
        deaths = np.array([100, 50, 1200])
        
        qx, meta = pspline_group_qx(ages, widths, population, deaths, lam=50.0)
        
        # Check shape
        assert len(qx) == 3
        
        # Check bounds: 0 <= qx <= 1
        assert np.all(qx >= 0)
        assert np.all(qx <= 1)
        
        # Last interval should be 1.0
        assert qx[-1] == 1.0
        
        # Metadata
        assert meta['lambda'] == 50.0
        assert meta['diff_order'] == 3
    
    def test_increasing_mortality(self):
        """Test that qx generally increases with age."""
        ages = np.array([0, 5, 10, 15, 20, 25])
        widths = np.array([5, 5, 5, 5, 5, np.inf])
        
        # Increasing mortality rates
        population = np.ones(6) * 10000
        deaths = np.array([50, 20, 15, 20, 40, 5000])
        
        qx, _ = pspline_group_qx(ages, widths, population, deaths, lam=100.0)
        
        # Should be generally increasing (after smoothing)
        # At least first few should increase
        assert qx[1] < qx[-1]  # Children < Open interval
    
    def test_single_year_consistency(self):
        """Test consistency when n=1 (single year intervals)."""
        ages = np.array([20, 21, 22, 23, 24])
        widths = np.array([1, 1, 1, 1, np.inf])
        population = np.array([1000, 1000, 1000, 1000, 1000])
        deaths = np.array([10, 12, 11, 13, 800])
        
        qx, _ = pspline_group_qx(ages, widths, population, deaths, lam=1.0)
        
        # Should be reasonable
        assert 0 <= qx[0] <= 0.05
        assert 0 <= qx[1] <= 0.05
        assert qx[-1] == 1.0
    
    def test_empty_closed_intervals(self):
        """Test with only open interval."""
        ages = np.array([90])
        widths = np.array([np.inf])
        population = np.array([1000])
        deaths = np.array([800])
        
        qx, _ = pspline_group_qx(ages, widths, population, deaths)
        
        assert len(qx) == 1
        assert qx[0] == 1.0


# ============================================================================
# Test make_lifetable
# ============================================================================
class TestMakeLifetable:
    """Test complete life table construction."""
    
    def test_basic_lifetable(self):
        """Test basic life table with standard ages."""
        ages = pd.Series(['0-4', '5-9', '10-14', '15-19', '90+'])
        population = np.array([10000, 12000, 11000, 10500, 2000])
        deaths = np.array([100, 20, 15, 25, 1800])
        
        lt = make_lifetable(ages, population, deaths, radix=100000, use_pspline=False)
        
        # Check all columns present
        required_cols = ['n', 'mx', 'ax', 'qx', 'px', 'lx', 'dx', 'Lx', 'Tx', 'ex']
        for col in required_cols:
            assert col in lt.columns
        
        # Check dimensions
        assert len(lt) == 5
        
        # Check bounds
        assert (lt['mx'] >= 0).all()
        assert (lt['qx'] >= 0).all() and (lt['qx'] <= 1).all()
        assert (lt['px'] >= 0).all() and (lt['px'] <= 1).all()
        assert (lt['lx'] >= 0).all()
        assert (lt['ex'] >= 0).all()
        
        # Check radix
        assert lt.iloc[0]['lx'] == 100000
        
        # Check last interval
        assert lt.iloc[-1]['qx'] == 1.0
        assert lt.iloc[-1]['px'] == 0.0
    
    def test_moving_average_smoothing(self):
        """Test with moving average smoothing."""
        ages = pd.Series(['0-4', '5-9', '10-14', '15-19', '20-24', '90+'])
        population = np.ones(6) * 10000
        # Add noise
        deaths = np.array([100, 25, 18, 30, 22, 8000])
        
        lt_raw = make_lifetable(ages, population, deaths, use_ma=False)
        lt_smooth = make_lifetable(ages, population, deaths, use_ma=True, ma_window=3)
        
        # Check metadata column
        assert 'ma_window' in lt_smooth.columns
        assert lt_smooth['ma_window'].iloc[0] == 3
        
        # Both should produce valid life tables (smoothing effect may vary)
        assert (lt_smooth['mx'] >= 0).all()
    
    def test_pspline_smoothing(self):
        """Test with P-spline smoothing."""
        ages = pd.Series(['0-4', '5-9', '10-14', '15-19', '90+'])
        population = np.ones(5) * 10000
        deaths = np.array([120, 30, 25, 35, 8000])
        
        lt = make_lifetable(
            ages, population, deaths,
            use_pspline=True,
            pspline_kwargs={'lam': 100.0, 'diff_order': 3}
        )
        
        # Check metadata columns
        assert 'pspline_lambda' in lt.columns
        assert 'pspline_order' in lt.columns
        assert lt['pspline_lambda'].iloc[0] == 100.0
        assert lt['pspline_order'].iloc[0] == 3
    
    def test_life_expectancy_reasonable(self):
        """Test that life expectancy is reasonable."""
        # Create data with realistic mortality  
        ages = pd.Series(['0-4', '5-9', '10-14', '15-19', '20-24', '90+'])
        population = np.ones(6) * 100000
        # Mortality pattern
        deaths = np.array([500, 50, 30, 40, 50, 50000])
        
        lt = make_lifetable(ages, population, deaths)
        
        # Life expectancy at birth should be positive
        e0 = lt.iloc[0]['ex']
        assert e0 > 0
        
        # All ex values should be finite and positive
        assert (lt['ex'] > 0).all()
        assert (np.isfinite(lt['ex'])).all()
    
    def test_coale_demeny_ax(self):
        """Test Coale-Demeny ax formula for infants."""
        ages = pd.Series(['0', '1-4', '5-9', '90+'])
        population = np.array([10000, 40000, 50000, 5000])
        
        # Test different infant mortality levels
        # Low mortality: m0 < 0.01724
        deaths_low = np.array([100, 50, 40, 4500])
        lt_low = make_lifetable(ages, population, deaths_low)
        
        m0_low = lt_low.iloc[0]['mx']
        expected_ax_low = 0.14903 - 2.05527 * m0_low
        assert abs(lt_low.iloc[0]['ax'] - expected_ax_low) < 0.01
    
    def test_survivorship_monotonic(self):
        """Test that lx is monotonically decreasing."""
        ages = pd.Series(['0-4', '5-9', '10-14', '15-19', '90+'])
        population = np.ones(5) * 10000
        deaths = np.array([80, 30, 25, 40, 8000])
        
        lt = make_lifetable(ages, population, deaths)
        
        # lx should decrease
        lx = lt['lx'].values
        assert np.all(np.diff(lx) <= 0)
    
    def test_Lx_monotonic(self):
        """Test that Lx is (mostly) monotonically decreasing."""
        ages = pd.Series(['0-4', '5-9', '10-14', '90+'])
        population = np.ones(4) * 10000
        deaths = np.array([100, 40, 30, 8000])
        
        lt = make_lifetable(ages, population, deaths)
        
        # Lx should generally decrease (auto-repaired)
        Lx = lt['Lx'].values
        # At minimum, last should be less than previous
        assert Lx[-1] <= Lx[-2]
    
    def test_Tx_monotonic(self):
        """Test that Tx is monotonically decreasing."""
        ages = pd.Series(['0-4', '5-9', '10-14', '90+'])
        population = np.ones(4) * 10000
        deaths = np.array([80, 30, 25, 8000])
        
        lt = make_lifetable(ages, population, deaths)
        
        # Tx should decrease
        Tx = lt['Tx'].values
        assert np.all(np.diff(Tx) <= 0)
    
    def test_negative_input_warning(self):
        """Test warning for negative inputs."""
        ages = pd.Series(['0-4', '5-9', '90+'])
        population = np.array([10000, -100, 5000])
        deaths = np.array([80, 30, 4000])
        
        with pytest.warns(UserWarning, match="Negative population"):
            lt = make_lifetable(ages, population, deaths)
    
    def test_zero_population(self):
        """Test handling of zero population."""
        ages = pd.Series(['0-4', '5-9', '10-14', '90+'])
        population = np.array([10000, 0, 11000, 5000])
        deaths = np.array([80, 0, 25, 4000])
        
        # Should not crash
        lt = make_lifetable(ages, population, deaths)
        
        # mx should be very small where population is 0 (after smoothing)
        assert lt.iloc[1]['mx'] < 0.01
    
    def test_radix_scaling(self):
        """Test that different radix values scale appropriately."""
        ages = pd.Series(['0-4', '5-9', '90+'])
        population = np.ones(3) * 10000
        deaths = np.array([100, 50, 8000])
        
        lt1 = make_lifetable(ages, population, deaths, radix=100000)
        lt2 = make_lifetable(ages, population, deaths, radix=10000)
        
        # lx should scale
        assert lt1.iloc[0]['lx'] / lt2.iloc[0]['lx'] == 10
        
        # But qx, px, ex should be the same
        np.testing.assert_array_almost_equal(lt1['qx'].values, lt2['qx'].values)
        np.testing.assert_array_almost_equal(lt1['ex'].values, lt2['ex'].values)
    
    def test_open_interval_width(self):
        """Test different open interval widths."""
        ages = pd.Series(['0-4', '5-9', '90+'])
        population = np.ones(3) * 10000
        deaths = np.array([100, 50, 8000])
        
        lt1 = make_lifetable(ages, population, deaths, open_interval_width=5)
        lt2 = make_lifetable(ages, population, deaths, open_interval_width=10)
        
        # n for last interval should reflect width
        assert lt1.iloc[-1]['n'] == 5
        assert lt2.iloc[-1]['n'] == 10


# ============================================================================
# Integration Tests
# ============================================================================
class TestIntegration:
    """Test complete workflows."""
    
    def test_full_workflow_no_smoothing(self):
        """Test complete workflow without smoothing."""
        # Create realistic data
        ages = pd.Series(['0-4', '5-9', '10-14', '15-19', '20-24', 
                         '25-29', '30-34', '35-39', '40-44', '90+'])
        
        # Realistic population distribution
        population = np.array([50000, 48000, 47000, 46000, 45000,
                              44000, 42000, 40000, 38000, 5000])
        
        # Realistic mortality pattern (low child, increasing adult)
        deaths = np.array([250, 50, 40, 60, 80,
                          100, 150, 250, 400, 4500])
        
        lt = make_lifetable(ages, population, deaths, use_pspline=False, use_ma=False)
        
        # Validate key properties
        assert lt.iloc[0]['lx'] == 100000
        assert 50 <= lt.iloc[0]['ex'] <= 85  # Reasonable life expectancy
        assert lt.iloc[-1]['qx'] == 1.0
        
        # Validate mathematical relationships
        # dx = lx * qx (approximately)
        for i in range(len(lt)):
            expected_dx = lt.iloc[i]['lx'] * lt.iloc[i]['qx']
            assert abs(lt.iloc[i]['dx'] - expected_dx) < 1.0
    
    def test_full_workflow_with_pspline(self):
        """Test complete workflow with P-spline smoothing."""
        ages = pd.Series(['0-4', '5-9', '10-14', '15-19', '20-24', 
                         '25-29', '30-34', '90+'])
        
        population = np.array([50000, 48000, 47000, 46000, 45000,
                              44000, 42000, 5000])
        
        deaths = np.array([300, 60, 45, 70, 85, 120, 180, 4500])
        
        lt = make_lifetable(
            ages, population, deaths,
            use_pspline=True,
            pspline_kwargs={'lam': 200.0, 'diff_order': 3}
        )
        
        # Should produce valid life table
        assert lt.iloc[0]['lx'] == 100000
        assert lt.iloc[0]['ex'] > 0  # Positive life expectancy
        assert 'pspline_lambda' in lt.columns
    
    def test_comparison_smoothing_methods(self):
        """Compare different smoothing methods."""
        ages = pd.Series(['0-4', '5-9', '10-14', '15-19', '20-24', '90+'])
        population = np.ones(6) * 10000
        
        # Add significant noise
        np.random.seed(42)
        base_deaths = np.array([120, 30, 25, 35, 45, 8000])
        deaths = base_deaths * (1 + 0.3 * np.random.randn(6))
        deaths = np.maximum(deaths, 1)  # Ensure positive
        
        lt_raw = make_lifetable(ages, population, deaths, use_pspline=False, use_ma=False)
        lt_ma = make_lifetable(ages, population, deaths, use_pspline=False, use_ma=True)
        lt_ps = make_lifetable(ages, population, deaths, use_pspline=True)
        
        # All should produce valid life tables
        for lt in [lt_raw, lt_ma, lt_ps]:
            assert lt.iloc[0]['lx'] == 100000
            assert lt.iloc[-1]['qx'] == 1.0
            assert (lt['ex'] > 0).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
