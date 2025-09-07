"""
Preprocessing module for fiber-photometry signals.

Handles smoothing, artifact removal, denoising, and data trimming
with zero-phase filtering to avoid temporal shifts.
"""

import warnings
from typing import Optional, Tuple, Union, List

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt


class Preprocessor:
    """Signal preprocessing for fiber-photometry data."""
    
    def __init__(self, fps: float):
        self.fps = fps
        self.nyquist = fps / 2
        
    def trim_led_artifact(
        self, 
        time_vec: np.ndarray, 
        signal_data: np.ndarray,
        trim_seconds: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trim initial LED stabilization artifact.
        
        Args:
            time_vec: Time vector in seconds
            signal_data: Signal array (can be 1D or 2D)
            trim_seconds: Seconds to trim from beginning
            
        Returns:
            Tuple of (trimmed_time, trimmed_signal)
        """
        if trim_seconds <= 0:
            return time_vec, signal_data
            
        trim_idx = np.searchsorted(time_vec, trim_seconds)
        
        if trim_idx >= len(time_vec):
            warnings.warn(f"Trim duration ({trim_seconds}s) exceeds recording length")
            trim_idx = 0
            
        trimmed_time = time_vec[trim_idx:] - time_vec[trim_idx]  # Reset to start at 0
        
        if signal_data.ndim == 1:
            trimmed_signal = signal_data[trim_idx:]
        else:
            trimmed_signal = signal_data[trim_idx:, :]
            
        return trimmed_time, trimmed_signal
    
    def moving_average_smooth(
        self, 
        signal_data: np.ndarray, 
        window_seconds: float
    ) -> np.ndarray:
        """
        Apply moving average smoothing (zero-phase).
        
        Args:
            signal_data: Input signal (1D or 2D array)
            window_seconds: Window size in seconds
            
        Returns:
            Smoothed signal array
        """
        if window_seconds <= 0:
            return signal_data
            
        window_samples = int(window_seconds * self.fps)
        if window_samples < 3:
            window_samples = 3
            
        # Ensure odd window size for symmetry
        if window_samples % 2 == 0:
            window_samples += 1
            
        # Create moving average kernel
        kernel = np.ones(window_samples) / window_samples
        
        if signal_data.ndim == 1:
            # Use same padding to handle edges
            smoothed = signal.filtfilt(kernel, [1], signal_data)
        else:
            smoothed = np.zeros_like(signal_data)
            for i in range(signal_data.shape[1]):
                smoothed[:, i] = signal.filtfilt(kernel, [1], signal_data[:, i])
                
        return smoothed
    
    def butterworth_filter(
        self,
        signal_data: np.ndarray,
        cutoff_hz: float,
        order: int = 4,
        filter_type: str = 'low'
    ) -> np.ndarray:
        """
        Apply Butterworth filter (zero-phase).
        
        Args:
            signal_data: Input signal (1D or 2D array)
            cutoff_hz: Cutoff frequency in Hz
            order: Filter order
            filter_type: 'low', 'high', 'band', or 'bandstop'
            
        Returns:
            Filtered signal array
        """
        if cutoff_hz <= 0 or cutoff_hz >= self.nyquist:
            warnings.warn(f"Invalid cutoff frequency {cutoff_hz} Hz for Nyquist {self.nyquist} Hz")
            return signal_data
            
        # Design filter
        sos = butter(order, cutoff_hz, btype=filter_type, fs=self.fps, output='sos')
        
        if signal_data.ndim == 1:
            filtered = signal.sosfiltfilt(sos, signal_data)
        else:
            filtered = np.zeros_like(signal_data)
            for i in range(signal_data.shape[1]):
                filtered[:, i] = signal.sosfiltfilt(sos, signal_data[:, i])
                
        return filtered
    
    def remove_outliers(
        self,
        signal_data: np.ndarray,
        method: str = 'zscore',
        threshold: float = 4.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and interpolate outliers.
        
        Args:
            signal_data: Input signal (1D or 2D array)
            method: 'zscore', 'mad', or 'iqr'
            threshold: Outlier detection threshold
            
        Returns:
            Tuple of (cleaned_signal, outlier_mask)
        """
        if signal_data.ndim == 1:
            return self._remove_outliers_1d(signal_data, method, threshold)
        else:
            cleaned = np.zeros_like(signal_data)
            outlier_masks = []
            
            for i in range(signal_data.shape[1]):
                cleaned[:, i], mask = self._remove_outliers_1d(
                    signal_data[:, i], method, threshold
                )
                outlier_masks.append(mask)
                
            return cleaned, np.column_stack(outlier_masks)
    
    def _remove_outliers_1d(
        self,
        signal_data: np.ndarray,
        method: str,
        threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers from 1D signal."""
        if method == 'zscore':
            z_scores = np.abs((signal_data - np.mean(signal_data)) / np.std(signal_data))
            outliers = z_scores > threshold
        elif method == 'mad':
            median = np.median(signal_data)
            mad = np.median(np.abs(signal_data - median))
            modified_z = 0.6745 * (signal_data - median) / mad
            outliers = np.abs(modified_z) > threshold
        elif method == 'iqr':
            q75, q25 = np.percentile(signal_data, [75, 25])
            iqr = q75 - q25
            outliers = ((signal_data < (q25 - threshold * iqr)) | 
                       (signal_data > (q75 + threshold * iqr)))
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        # Interpolate outliers
        cleaned_signal = signal_data.copy()
        if np.any(outliers):
            # Linear interpolation
            valid_indices = np.where(~outliers)[0]
            outlier_indices = np.where(outliers)[0]
            
            if len(valid_indices) > 1:
                cleaned_signal[outliers] = np.interp(
                    outlier_indices, valid_indices, signal_data[valid_indices]
                )
        
        return cleaned_signal, outliers
    
    def detect_bad_segments(
        self,
        signal_data: np.ndarray,
        time_vec: np.ndarray,
        mad_threshold: float = 5.0,
        min_segment_duration: float = 0.5
    ) -> List[Tuple[float, float]]:
        """
        Automatically detect potentially bad data segments.
        
        Args:
            signal_data: Input signal (1D array)
            time_vec: Time vector
            mad_threshold: MAD threshold for segment detection
            min_segment_duration: Minimum duration for bad segments
            
        Returns:
            List of (start_time, end_time) tuples for bad segments
        """
        # Calculate moving statistics
        window_size = int(self.fps * 1.0)  # 1 second window
        
        moving_mad = []
        for i in range(len(signal_data) - window_size + 1):
            segment = signal_data[i:i + window_size]
            mad = np.median(np.abs(segment - np.median(segment)))
            moving_mad.append(mad)
        
        moving_mad = np.array(moving_mad)
        overall_mad = np.median(moving_mad)
        
        # Detect segments with excessive MAD
        bad_mask = moving_mad > (mad_threshold * overall_mad)
        
        # Find contiguous bad segments
        bad_segments = []
        in_bad_segment = False
        segment_start = None
        
        for i, is_bad in enumerate(bad_mask):
            if is_bad and not in_bad_segment:
                # Start of bad segment
                in_bad_segment = True
                segment_start = time_vec[i]
            elif not is_bad and in_bad_segment:
                # End of bad segment
                in_bad_segment = False
                segment_end = time_vec[i]
                
                if segment_end - segment_start >= min_segment_duration:
                    bad_segments.append((segment_start, segment_end))
        
        # Handle case where recording ends in bad segment
        if in_bad_segment and segment_start is not None:
            segment_end = time_vec[-1]
            if segment_end - segment_start >= min_segment_duration:
                bad_segments.append((segment_start, segment_end))
        
        return bad_segments
    
    def remove_bad_segments(
        self,
        time_vec: np.ndarray,
        signal_data: np.ndarray,
        bad_segments: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Remove bad segments from data.
        
        Args:
            time_vec: Time vector
            signal_data: Signal array (1D or 2D)
            bad_segments: List of (start_time, end_time) tuples
            
        Returns:
            Tuple of (clean_time, clean_signal, removed_indices)
        """
        if not bad_segments:
            return time_vec, signal_data, []
        
        # Create mask for good data
        good_mask = np.ones(len(time_vec), dtype=bool)
        removed_indices = []
        
        for start_time, end_time in bad_segments:
            start_idx = np.searchsorted(time_vec, start_time)
            end_idx = np.searchsorted(time_vec, end_time)
            
            good_mask[start_idx:end_idx] = False
            removed_indices.append((start_idx, end_idx))
        
        clean_time = time_vec[good_mask]
        
        if signal_data.ndim == 1:
            clean_signal = signal_data[good_mask]
        else:
            clean_signal = signal_data[good_mask, :]
        
        return clean_time, clean_signal, removed_indices
    
    def preprocess_pipeline(
        self,
        time_vec: np.ndarray,
        signal_data: np.ndarray,
        config: dict
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            time_vec: Time vector in seconds
            signal_data: Signal array (1D or 2D)
            config: Preprocessing configuration dictionary
            
        Returns:
            Tuple of (processed_time, processed_signal, processing_log)
        """
        processing_log = {}
        processed_time = time_vec.copy()
        processed_signal = signal_data.copy()
        
        # 1. Trim LED artifact
        if config.get('trim_led_artifact', 0) > 0:
            processed_time, processed_signal = self.trim_led_artifact(
                processed_time, processed_signal, config['trim_led_artifact']
            )
            processing_log['trimmed_seconds'] = config['trim_led_artifact']
        
        # 2. Remove outliers
        if config.get('remove_outliers', False):
            processed_signal, outlier_mask = self.remove_outliers(
                processed_signal,
                method=config.get('outlier_method', 'mad'),
                threshold=config.get('outlier_threshold', 4.0)
            )
            processing_log['outliers_removed'] = int(np.sum(outlier_mask))
        
        # 3. Apply smoothing
        if config.get('filter_window', 0) > 0:
            processed_signal = self.moving_average_smooth(
                processed_signal, config['filter_window']
            )
            processing_log['smoothing_window'] = config['filter_window']
        
        # 4. Butterworth filtering
        if 'butterworth_cutoff' in config:
            processed_signal = self.butterworth_filter(
                processed_signal,
                cutoff_hz=config['butterworth_cutoff'],
                order=config.get('butterworth_order', 4),
                filter_type=config.get('butterworth_type', 'low')
            )
            processing_log['butterworth_filter'] = config['butterworth_cutoff']
        
        # 5. Detect and optionally remove bad segments
        if config.get('remove_bad_segments', False):
            if processed_signal.ndim == 1:
                check_signal = processed_signal
            else:
                # For multi-channel, check the first channel
                check_signal = processed_signal[:, 0]
                
            bad_segments = self.detect_bad_segments(
                check_signal, processed_time,
                mad_threshold=config.get('bad_segment_threshold', 5.0)
            )
            
            if bad_segments:
                processed_time, processed_signal, removed_idx = self.remove_bad_segments(
                    processed_time, processed_signal, bad_segments
                )
                processing_log['bad_segments_removed'] = len(bad_segments)
                processing_log['bad_segment_times'] = bad_segments
            else:
                processing_log['bad_segments_removed'] = 0
        
        processing_log['final_duration'] = processed_time[-1] - processed_time[0]
        processing_log['final_samples'] = len(processed_time)
        
        return processed_time, processed_signal, processing_log