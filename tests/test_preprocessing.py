"""
Unit tests for the preprocessing module.
"""

import pytest
import numpy as np
import warnings

from fiber_photometry.preprocessing import Preprocessor


class TestPreprocessor:
    """Test cases for Preprocessor."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance for testing."""
        return Preprocessor(fps=100.0)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        fps = 100.0
        duration = 10.0  # 10 seconds
        time_vec = np.arange(0, duration, 1/fps)
        
        # Create signal with trend, noise, and artifacts
        signal = 100 + 2 * np.sin(0.5 * time_vec) + np.random.normal(0, 0.5, len(time_vec))
        
        # Add some outliers
        signal[50] += 20  # Positive outlier
        signal[150] -= 15  # Negative outlier
        
        return time_vec, signal
    
    def test_trim_led_artifact(self, preprocessor, sample_data):
        """Test LED artifact trimming."""
        time_vec, signal = sample_data
        
        # Test trimming 1 second
        trimmed_time, trimmed_signal = preprocessor.trim_led_artifact(
            time_vec, signal, trim_seconds=1.0
        )
        
        # Check that data was trimmed
        assert len(trimmed_time) < len(time_vec)
        assert len(trimmed_signal) < len(signal)
        
        # Check that time starts from 0 after trimming
        assert trimmed_time[0] == 0.0
        
        # Check that the right amount was trimmed (approximately)
        expected_samples_trimmed = int(1.0 * 100)  # 1 second at 100 Hz
        assert len(time_vec) - len(trimmed_time) == expected_samples_trimmed
    
    def test_trim_zero_seconds(self, preprocessor, sample_data):
        """Test trimming with zero seconds (no trimming)."""
        time_vec, signal = sample_data
        
        trimmed_time, trimmed_signal = preprocessor.trim_led_artifact(
            time_vec, signal, trim_seconds=0.0
        )
        
        # Should return original data
        np.testing.assert_array_equal(trimmed_time, time_vec)
        np.testing.assert_array_equal(trimmed_signal, signal)
    
    def test_trim_excessive_duration(self, preprocessor, sample_data):
        """Test trimming with duration exceeding recording length."""
        time_vec, signal = sample_data
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trimmed_time, trimmed_signal = preprocessor.trim_led_artifact(
                time_vec, signal, trim_seconds=20.0  # More than 10s recording
            )
            
            # Should produce warning
            assert len(w) > 0
            assert "exceeds recording length" in str(w[0].message).lower()
        
        # Should return original data (no trimming)
        np.testing.assert_array_equal(trimmed_time, time_vec)
        np.testing.assert_array_equal(trimmed_signal, signal)
    
    def test_moving_average_smooth(self, preprocessor, sample_data):
        """Test moving average smoothing."""
        time_vec, signal = sample_data
        
        # Apply smoothing
        smoothed = preprocessor.moving_average_smooth(signal, window_seconds=0.1)
        
        # Check that signal is smoothed (variance should be reduced)
        assert len(smoothed) == len(signal)
        assert np.var(smoothed) < np.var(signal)
        
        # Check that mean is approximately preserved
        assert np.abs(np.mean(smoothed) - np.mean(signal)) < 0.1
    
    def test_moving_average_zero_window(self, preprocessor, sample_data):
        """Test moving average with zero window (no smoothing)."""
        time_vec, signal = sample_data
        
        smoothed = preprocessor.moving_average_smooth(signal, window_seconds=0.0)
        
        # Should return original signal
        np.testing.assert_array_equal(smoothed, signal)
    
    def test_butterworth_filter(self, preprocessor, sample_data):
        """Test Butterworth filtering."""
        time_vec, signal = sample_data
        
        # Apply low-pass filter
        filtered = preprocessor.butterworth_filter(signal, cutoff_hz=10.0, order=4, filter_type='low')
        
        # Check that signal is filtered
        assert len(filtered) == len(signal)
        # High-frequency noise should be reduced
        assert np.var(filtered) <= np.var(signal)
    
    def test_butterworth_invalid_cutoff(self, preprocessor, sample_data):
        """Test Butterworth filter with invalid cutoff frequency."""
        time_vec, signal = sample_data
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Cutoff above Nyquist frequency
            filtered = preprocessor.butterworth_filter(signal, cutoff_hz=60.0, order=4, filter_type='low')
            
            # Should produce warning
            assert len(w) > 0
            assert "invalid cutoff frequency" in str(w[0].message).lower()
        
        # Should return original signal
        np.testing.assert_array_equal(filtered, signal)
    
    def test_remove_outliers_zscore(self, preprocessor):
        """Test outlier removal using z-score method."""
        # Create signal with known outliers
        signal = np.concatenate([
            np.random.normal(0, 1, 100),  # Normal data
            [10, -8]  # Clear outliers
        ])
        
        cleaned, outlier_mask = preprocessor.remove_outliers(signal, method='zscore', threshold=3.0)
        
        # Check that outliers were detected
        assert len(cleaned) == len(signal)
        assert len(outlier_mask) == len(signal)
        assert np.any(outlier_mask)  # Some outliers should be detected
        
        # Check that outliers were interpolated
        outlier_indices = np.where(outlier_mask)[0]
        assert np.abs(cleaned[outlier_indices[0]]) < np.abs(signal[outlier_indices[0]])
    
    def test_remove_outliers_mad(self, preprocessor):
        """Test outlier removal using MAD method."""
        signal = np.concatenate([
            np.random.normal(0, 1, 100),
            [15, -12]  # Outliers
        ])
        
        cleaned, outlier_mask = preprocessor.remove_outliers(signal, method='mad', threshold=3.5)
        
        # Check results
        assert len(cleaned) == len(signal)
        assert len(outlier_mask) == len(signal)
        assert np.any(outlier_mask)
    
    def test_remove_outliers_iqr(self, preprocessor):
        """Test outlier removal using IQR method."""
        signal = np.concatenate([
            np.random.normal(0, 1, 100),
            [20, -18]  # Extreme outliers
        ])
        
        cleaned, outlier_mask = preprocessor.remove_outliers(signal, method='iqr', threshold=1.5)
        
        # Check results
        assert len(cleaned) == len(signal)
        assert len(outlier_mask) == len(signal)
        assert np.any(outlier_mask)
    
    def test_remove_outliers_invalid_method(self, preprocessor, sample_data):
        """Test outlier removal with invalid method."""
        time_vec, signal = sample_data
        
        with pytest.raises(ValueError, match="Unknown outlier detection method"):
            preprocessor.remove_outliers(signal, method='invalid_method')
    
    def test_detect_bad_segments(self, preprocessor):
        """Test bad segment detection."""
        # Create signal with bad segments
        fps = 100.0
        time_vec = np.arange(0, 10, 1/fps)
        
        # Normal signal
        signal = np.random.normal(0, 1, len(time_vec))
        
        # Add bad segment (high variance region)
        bad_start = int(3 * fps)
        bad_end = int(5 * fps)
        signal[bad_start:bad_end] += np.random.normal(0, 10, bad_end - bad_start)
        
        bad_segments = preprocessor.detect_bad_segments(
            signal, time_vec, mad_threshold=3.0, min_segment_duration=0.5
        )
        
        # Should detect at least one bad segment
        assert len(bad_segments) > 0
        
        # Bad segment should be in the expected time range
        detected_start, detected_end = bad_segments[0]
        assert 2.0 < detected_start < 4.0  # Around 3s
        assert 4.0 < detected_end < 6.0    # Around 5s
    
    def test_remove_bad_segments(self, preprocessor):
        """Test bad segment removal."""
        time_vec = np.arange(0, 10, 0.01)  # 10 seconds
        signal = np.random.normal(0, 1, len(time_vec))
        
        bad_segments = [(2.0, 3.0), (7.0, 8.0)]  # Remove 2 segments
        
        clean_time, clean_signal, removed_indices = preprocessor.remove_bad_segments(
            time_vec, signal, bad_segments
        )
        
        # Check that data was removed
        assert len(clean_time) < len(time_vec)
        assert len(clean_signal) < len(signal)
        assert len(removed_indices) == 2
        
        # Check that the right segments were removed
        assert not np.any((clean_time >= 2.0) & (clean_time <= 3.0))
        assert not np.any((clean_time >= 7.0) & (clean_time <= 8.0))
    
    def test_remove_bad_segments_empty(self, preprocessor, sample_data):
        """Test bad segment removal with no bad segments."""
        time_vec, signal = sample_data
        
        clean_time, clean_signal, removed_indices = preprocessor.remove_bad_segments(
            time_vec, signal, []
        )
        
        # Should return original data
        np.testing.assert_array_equal(clean_time, time_vec)
        np.testing.assert_array_equal(clean_signal, signal)
        assert len(removed_indices) == 0
    
    def test_preprocess_pipeline(self, preprocessor, sample_data):
        """Test complete preprocessing pipeline."""
        time_vec, signal = sample_data
        
        config = {
            'trim_led_artifact': 0.5,
            'filter_window': 0.1,
            'remove_outliers': True,
            'outlier_method': 'mad',
            'outlier_threshold': 3.5,
            'remove_bad_segments': False
        }
        
        processed_time, processed_signal, log = preprocessor.preprocess_pipeline(
            time_vec, signal, config
        )
        
        # Check results
        assert len(processed_time) < len(time_vec)  # Should be trimmed
        assert len(processed_signal) == len(processed_time)
        
        # Check processing log
        assert 'trimmed_seconds' in log
        assert 'smoothing_window' in log
        assert 'outliers_removed' in log
        assert 'final_duration' in log
        assert 'final_samples' in log
        
        assert log['trimmed_seconds'] == 0.5
        assert log['smoothing_window'] == 0.1
        assert log['outliers_removed'] >= 0  # Should detect outliers we added
    
    def test_preprocess_pipeline_2d_signal(self, preprocessor, sample_data):
        """Test preprocessing pipeline with 2D signal (multiple channels)."""
        time_vec, signal = sample_data
        
        # Create 2-channel signal
        signal_2d = np.column_stack([signal, signal * 0.8 + np.random.normal(0, 0.3, len(signal))])
        
        config = {
            'trim_led_artifact': 0.2,
            'filter_window': 0.05,
            'remove_outliers': True,
            'outlier_method': 'zscore',
            'outlier_threshold': 4.0
        }
        
        processed_time, processed_signal, log = preprocessor.preprocess_pipeline(
            time_vec, signal_2d, config
        )
        
        # Check results
        assert processed_signal.ndim == 2
        assert processed_signal.shape[1] == 2  # Two channels
        assert len(processed_time) == processed_signal.shape[0]
    
    def test_preprocess_pipeline_minimal_config(self, preprocessor, sample_data):
        """Test preprocessing pipeline with minimal configuration."""
        time_vec, signal = sample_data
        
        config = {}  # Empty config - should use defaults/skip processing
        
        processed_time, processed_signal, log = preprocessor.preprocess_pipeline(
            time_vec, signal, config
        )
        
        # Should return similar to original data (minimal processing)
        assert len(processed_time) == len(time_vec)
        assert len(processed_signal) == len(signal)
        assert 'final_duration' in log
        assert 'final_samples' in log


if __name__ == '__main__':
    pytest.main([__file__])