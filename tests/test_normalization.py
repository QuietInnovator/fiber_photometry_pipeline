"""
Unit tests for the normalization module.
"""

import pytest
import numpy as np
import warnings
from scipy import signal

from fiber_photometry.normalization import Normalizer


class TestNormalizer:
    """Test cases for Normalizer."""
    
    @pytest.fixture
    def normalizer(self):
        """Create normalizer instance for testing."""
        return Normalizer(fps=100.0)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample signal and control data."""
        np.random.seed(42)  # For reproducible tests
        n_samples = 1000
        time_vec = np.arange(n_samples) / 100.0  # 100 Hz
        
        # Create synthetic signal with trend and noise
        baseline = 100 + 5 * np.exp(-time_vec / 3)  # Exponential decay
        signal_470 = baseline + 2 * np.sin(0.1 * time_vec) + np.random.normal(0, 0.5, n_samples)
        
        # Control signal correlated with main signal
        signal_410 = 0.8 * signal_470 + 10 + np.random.normal(0, 0.3, n_samples)
        
        return time_vec, signal_470, signal_410
    
    def test_linear_control_fit(self, normalizer, sample_data):
        """Test linear fitting of control to signal."""
        time_vec, signal_470, signal_410 = sample_data
        
        fitted_control, fit_info = normalizer.fit_control_to_signal(
            signal_470, signal_410, method='linear'
        )
        
        # Check fit results
        assert fit_info['method'] == 'linear'
        assert 'slope' in fit_info
        assert 'intercept' in fit_info
        assert 'r_squared' in fit_info
        assert len(fitted_control) == len(signal_470)
        assert fit_info['r_squared'] > 0.5  # Should have reasonable correlation
        
        # Check that fitted control follows expected relationship
        expected_control = fit_info['slope'] * signal_410 + fit_info['intercept']
        np.testing.assert_array_almost_equal(fitted_control, expected_control)
    
    def test_polynomial_control_fit(self, normalizer, sample_data):
        """Test polynomial fitting of control to signal."""
        time_vec, signal_470, signal_410 = sample_data
        
        fitted_control, fit_info = normalizer.fit_control_to_signal(
            signal_470, signal_410, method='polynomial'
        )
        
        # Check fit results
        assert fit_info['method'] == 'polynomial'
        assert 'degree' in fit_info
        assert 'coefficients' in fit_info
        assert 'r_squared' in fit_info
        assert len(fitted_control) == len(signal_470)
        assert fit_info['degree'] == 2  # Default degree
    
    def test_dff_control_fit(self, normalizer, sample_data):
        """Test ΔF/F calculation using control fitting."""
        time_vec, signal_470, signal_410 = sample_data
        
        dff, calc_info = normalizer.calculate_dff(
            signal_470, signal_410, baseline_method='control_fit'
        )
        
        # Check results
        assert calc_info['method'] == 'control_fit'
        assert 'fit_info' in calc_info
        assert 'baseline' in calc_info
        assert len(dff) == len(signal_470)
        
        # ΔF/F should be dimensionless and centered around 0
        assert np.abs(np.nanmean(dff)) < 0.1  # Should be roughly centered
        assert not np.any(np.isinf(dff))  # No infinite values
        
        # Check ΔF/F calculation manually for a few points
        baseline = calc_info['baseline']
        expected_dff = (signal_470 - baseline) / baseline
        np.testing.assert_array_almost_equal(dff, expected_dff, decimal=10)
    
    def test_dff_baseline_fit(self, normalizer, sample_data):
        """Test ΔF/F calculation using baseline fitting."""
        time_vec, signal_470, signal_410 = sample_data
        
        dff, calc_info = normalizer.calculate_dff(
            signal_470, baseline_method='baseline_fit', percentile=10
        )
        
        # Check results
        assert calc_info['method'] == 'baseline_fit'
        assert calc_info['percentile'] == 10
        assert 'baseline' in calc_info
        assert len(dff) == len(signal_470)
        assert not np.any(np.isinf(dff))
    
    def test_dff_detrend(self, normalizer, sample_data):
        """Test ΔF/F calculation using detrending."""
        time_vec, signal_470, signal_410 = sample_data
        
        dff, calc_info = normalizer.calculate_dff(
            signal_470, baseline_method='detrend', method='exponential'
        )
        
        # Check results
        assert calc_info['method'] == 'detrend'
        assert calc_info['detrend_method'] == 'exponential'
        assert 'baseline' in calc_info
        assert len(dff) == len(signal_470)
    
    def test_dff_percentile(self, normalizer, sample_data):
        """Test ΔF/F calculation using percentile baseline."""
        time_vec, signal_470, signal_410 = sample_data
        
        dff, calc_info = normalizer.calculate_dff(
            signal_470, baseline_method='percentile', percentile=5
        )
        
        # Check results
        assert calc_info['method'] == 'percentile'
        assert calc_info['percentile'] == 5
        assert 'baseline_value' in calc_info
        assert len(dff) == len(signal_470)
        
        # Baseline should be constant
        baseline = calc_info['baseline']
        assert np.all(baseline == baseline[0])
    
    def test_zscore_standard(self, normalizer):
        """Test standard z-score calculation."""
        # Create known data
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        expected_mean = np.mean(data)
        expected_std = np.std(data)
        
        zscore, calc_info = normalizer.calculate_zscore(data, method='standard')
        
        # Check results
        assert calc_info['method'] == 'standard'
        assert calc_info['mean'] == expected_mean
        assert calc_info['std'] == expected_std
        
        # Check z-score calculation
        expected_zscore = (data - expected_mean) / expected_std
        np.testing.assert_array_almost_equal(zscore, expected_zscore)
        
        # Z-score should have mean ~0 and std ~1
        assert np.abs(np.mean(zscore)) < 1e-10
        assert np.abs(np.std(zscore) - 1) < 1e-10
    
    def test_zscore_baseline(self, normalizer):
        """Test baseline z-score calculation."""
        # Create data with baseline period
        data = np.concatenate([
            np.random.normal(0, 1, 100),  # Baseline period
            np.random.normal(2, 1, 100)   # Response period
        ])
        time_vec = np.arange(len(data)) / 100.0
        baseline_window = (0.0, 1.0)  # First 1 second
        
        zscore, calc_info = normalizer.calculate_zscore(
            data, method='baseline', baseline_window=baseline_window, time_vec=time_vec
        )
        
        # Check results
        assert calc_info['method'] == 'baseline'
        assert calc_info['baseline_window'] == baseline_window
        assert 'n_baseline_points' in calc_info
        assert len(zscore) == len(data)
        
        # Baseline period should have mean ~0
        baseline_mask = (time_vec >= baseline_window[0]) & (time_vec <= baseline_window[1])
        baseline_zscore = zscore[baseline_mask]
        assert np.abs(np.mean(baseline_zscore)) < 0.1
    
    def test_zscore_modified(self, normalizer):
        """Test modified z-score (MAD-based) calculation."""
        # Create data with outliers
        data = np.concatenate([
            np.random.normal(0, 1, 100),
            [10, -10]  # Outliers
        ])
        
        zscore, calc_info = normalizer.calculate_zscore(data, method='modified')
        
        # Check results
        assert calc_info['method'] == 'modified'
        assert 'median' in calc_info
        assert 'mad' in calc_info
        assert len(zscore) == len(data)
        
        # Modified z-score should be more robust to outliers
        assert np.abs(calc_info['median']) < 0.5  # Should be close to 0
    
    def test_invalid_baseline_values(self, normalizer):
        """Test handling of invalid baseline values."""
        signal = np.array([1, 2, 3, 4, 5])
        control = np.array([0, -1, 0.1, 2, 3])  # Contains zero and negative values
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dff, calc_info = normalizer.calculate_dff(signal, control, baseline_method='control_fit')
            
            # Should produce warning about invalid baseline values
            assert len(w) > 0
            assert "invalid baseline values" in str(w[0].message).lower()
        
        # ΔF/F should contain NaN where baseline was invalid
        assert np.any(np.isnan(dff))
    
    def test_exponential_fit_fallback(self, normalizer):
        """Test exponential fitting with fallback to linear."""
        # Create data that's difficult to fit with exponential
        signal = np.array([1, 1, 1, 1, 1])  # Constant signal
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dff, calc_info = normalizer.calculate_dff(signal, baseline_method='detrend', method='exponential')
            
            # Should fall back to linear trend if exponential fit fails
            # May or may not produce warning depending on optimization behavior
        
        assert len(dff) == len(signal)
        assert not np.any(np.isinf(dff))
    
    def test_nan_handling(self, normalizer):
        """Test handling of NaN values in input data."""
        signal = np.array([1, 2, np.nan, 4, 5])
        control = np.array([0.5, 1, 1.5, np.nan, 2.5])
        
        dff, calc_info = normalizer.calculate_dff(signal, control, baseline_method='control_fit')
        
        # Should handle NaN values gracefully
        assert len(dff) == len(signal)
        # NaN positions may vary depending on fitting algorithm
    
    def test_zero_variance_zscore(self, normalizer):
        """Test z-score calculation with zero variance data."""
        data = np.array([5, 5, 5, 5, 5])  # Constant data
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            zscore, calc_info = normalizer.calculate_zscore(data, method='standard')
            
            # Should produce warning about zero standard deviation
            assert len(w) > 0
            assert "standard deviation is zero" in str(w[0].message).lower()
        
        # Z-score should be NaN
        assert np.all(np.isnan(zscore))
    
    def test_empty_baseline_window(self, normalizer):
        """Test z-score calculation with empty baseline window."""
        data = np.random.normal(0, 1, 100)
        time_vec = np.arange(len(data)) / 100.0
        baseline_window = (5.0, 6.0)  # Window outside data range
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            zscore, calc_info = normalizer.calculate_zscore(
                data, method='baseline', baseline_window=baseline_window, time_vec=time_vec
            )
            
            # Should fall back to standard z-score
            assert len(w) > 0
        
        assert len(zscore) == len(data)


if __name__ == '__main__':
    pytest.main([__file__])