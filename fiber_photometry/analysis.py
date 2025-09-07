"""
Analysis module for fiber-photometry data.

Handles event-related analysis (PSTHs, AUC, peaks) and transient detection.
"""

import warnings
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import signal, integrate, stats
from scipy.signal import find_peaks


class EventAnalyzer:
    """Event-related analysis for fiber-photometry data."""
    
    def __init__(self, fps: float):
        self.fps = fps
        
    def build_psth(
        self,
        dff_signal: np.ndarray,
        time_vec: np.ndarray,
        events_df: pd.DataFrame,
        event_label: str,
        time_window: Tuple[float, float] = (-3.0, 5.0),
        baseline_window: Tuple[float, float] = (-1.0, 0.0)
    ) -> Dict[str, Any]:
        """
        Build peri-stimulus time histogram (PSTH) for specific event type.
        
        Args:
            dff_signal: ΔF/F signal data
            time_vec: Time vector in seconds
            events_df: DataFrame with columns ['timestamp', 'event_label', 'trial_index']
            event_label: Event type to analyze
            time_window: (pre, post) time window in seconds around event
            baseline_window: (pre, post) baseline window for correction
            
        Returns:
            Dictionary with PSTH results
        """
        # Filter events for this label
        event_times = events_df[events_df['event_label'] == event_label]['timestamp'].values
        
        if len(event_times) == 0:
            warnings.warn(f"No events found for label '{event_label}'")
            return self._empty_psth_result(event_label, time_window)
        
        # Convert time windows to sample indices
        pre_samples = int(abs(time_window[0]) * self.fps)
        post_samples = int(time_window[1] * self.fps)
        total_samples = pre_samples + post_samples
        
        # Collect trials
        trials = []
        valid_event_times = []
        
        for event_time in event_times:
            event_idx = np.searchsorted(time_vec, event_time)
            
            # Check if we have enough data around this event
            if (event_idx - pre_samples >= 0 and 
                event_idx + post_samples < len(dff_signal)):
                
                trial_data = dff_signal[event_idx - pre_samples:event_idx + post_samples]
                
                # Baseline correction
                if baseline_window is not None:
                    baseline_start_idx = pre_samples + int(baseline_window[0] * self.fps)
                    baseline_end_idx = pre_samples + int(baseline_window[1] * self.fps)
                    
                    if 0 <= baseline_start_idx < baseline_end_idx <= len(trial_data):
                        baseline_mean = np.nanmean(trial_data[baseline_start_idx:baseline_end_idx])
                        trial_data = trial_data - baseline_mean
                
                trials.append(trial_data)
                valid_event_times.append(event_time)
        
        if len(trials) == 0:
            warnings.warn(f"No valid trials found for event '{event_label}'")
            return self._empty_psth_result(event_label, time_window)
        
        # Stack trials and compute statistics
        trials_array = np.array(trials)
        psth_time = np.linspace(time_window[0], time_window[1], total_samples)
        
        psth_mean = np.nanmean(trials_array, axis=0)
        psth_sem = stats.sem(trials_array, axis=0, nan_policy='omit')
        psth_std = np.nanstd(trials_array, axis=0)
        
        # Calculate trial metrics
        trial_metrics = []
        for i, trial in enumerate(trials_array):
            metrics = self._calculate_trial_metrics(
                trial, psth_time, valid_event_times[i], baseline_window
            )
            trial_metrics.append(metrics)
        
        # Summary statistics
        auc_values = [m['auc'] for m in trial_metrics if not np.isnan(m['auc'])]
        peak_amplitudes = [m['peak_amplitude'] for m in trial_metrics 
                          if not np.isnan(m['peak_amplitude'])]
        peak_latencies = [m['peak_latency'] for m in trial_metrics 
                         if not np.isnan(m['peak_latency'])]
        
        result = {
            'event_label': event_label,
            'n_trials': len(trials),
            'time_window': time_window,
            'baseline_window': baseline_window,
            'psth_time': psth_time,
            'psth_mean': psth_mean,
            'psth_sem': psth_sem,
            'psth_std': psth_std,
            'trials': trials_array,
            'event_times': valid_event_times,
            'trial_metrics': trial_metrics,
            'summary_stats': {
                'mean_auc': np.mean(auc_values) if auc_values else np.nan,
                'sem_auc': stats.sem(auc_values) if auc_values else np.nan,
                'mean_peak_amplitude': np.mean(peak_amplitudes) if peak_amplitudes else np.nan,
                'sem_peak_amplitude': stats.sem(peak_amplitudes) if peak_amplitudes else np.nan,
                'mean_peak_latency': np.mean(peak_latencies) if peak_latencies else np.nan,
                'sem_peak_latency': stats.sem(peak_latencies) if peak_latencies else np.nan,
            }
        }
        
        return result
    
    def _empty_psth_result(self, event_label: str, time_window: Tuple[float, float]) -> Dict[str, Any]:
        """Return empty PSTH result structure."""
        return {
            'event_label': event_label,
            'n_trials': 0,
            'time_window': time_window,
            'baseline_window': None,
            'psth_time': np.array([]),
            'psth_mean': np.array([]),
            'psth_sem': np.array([]),
            'psth_std': np.array([]),
            'trials': np.array([]),
            'event_times': np.array([]),
            'trial_metrics': [],
            'summary_stats': {
                'mean_auc': np.nan,
                'sem_auc': np.nan,
                'mean_peak_amplitude': np.nan,
                'sem_peak_amplitude': np.nan,
                'mean_peak_latency': np.nan,
                'sem_peak_latency': np.nan,
            }
        }
    
    def _calculate_trial_metrics(
        self,
        trial_data: np.ndarray,
        time_vec: np.ndarray,
        event_time: float,
        baseline_window: Optional[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """Calculate metrics for a single trial."""
        metrics = {
            'event_time': event_time,
            'auc': np.nan,
            'peak_amplitude': np.nan,
            'peak_latency': np.nan,
            'time_to_peak': np.nan,
            'baseline_mean': np.nan,
            'response_duration': np.nan
        }
        
        if len(trial_data) == 0 or np.all(np.isnan(trial_data)):
            return metrics
        
        # Area under curve (trapezoidal integration)
        valid_mask = ~np.isnan(trial_data)
        if np.sum(valid_mask) > 1:
            # Use trapezoid for newer scipy versions, fallback to trapz for older versions
            try:
                metrics['auc'] = integrate.trapezoid(
                    trial_data[valid_mask], 
                    time_vec[valid_mask]
                )
            except AttributeError:
                metrics['auc'] = integrate.trapz(
                    trial_data[valid_mask], 
                    time_vec[valid_mask]
                )
        
        # Peak detection
        if np.any(~np.isnan(trial_data)):
            clean_data = trial_data.copy()
            clean_data[np.isnan(clean_data)] = np.nanmean(trial_data)
            
            # Find peaks and troughs
            peaks, _ = find_peaks(clean_data, height=None, distance=int(0.1 * self.fps))
            troughs, _ = find_peaks(-clean_data, height=None, distance=int(0.1 * self.fps))
            
            # Get the most prominent peak
            if len(peaks) > 0:
                peak_idx = peaks[np.argmax(clean_data[peaks])]
                metrics['peak_amplitude'] = trial_data[peak_idx]
                metrics['peak_latency'] = time_vec[peak_idx]
                metrics['time_to_peak'] = time_vec[peak_idx] - 0  # Relative to event (t=0)
            
            # Check for most prominent trough if no positive peak
            elif len(troughs) > 0:
                trough_idx = troughs[np.argmin(clean_data[troughs])]
                metrics['peak_amplitude'] = trial_data[trough_idx]  # Negative value
                metrics['peak_latency'] = time_vec[trough_idx]
                metrics['time_to_peak'] = time_vec[trough_idx] - 0
        
        # Baseline statistics
        if baseline_window is not None:
            baseline_mask = (time_vec >= baseline_window[0]) & (time_vec <= baseline_window[1])
            if np.any(baseline_mask):
                baseline_data = trial_data[baseline_mask]
                metrics['baseline_mean'] = np.nanmean(baseline_data)
        
        # Response duration (time above/below threshold)
        threshold = 2 * np.nanstd(trial_data)  # 2 SD threshold
        if not np.isnan(threshold) and threshold > 0:
            above_threshold = np.abs(trial_data) > threshold
            if np.any(above_threshold):
                # Find contiguous segments above threshold
                diff = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                
                if len(starts) > 0 and len(ends) > 0:
                    durations = (ends - starts) / self.fps
                    metrics['response_duration'] = np.sum(durations)
        
        return metrics
    
    def analyze_all_events(
        self,
        dff_signal: np.ndarray,
        time_vec: np.ndarray,
        events_df: pd.DataFrame,
        event_labels: List[str],
        time_window: Tuple[float, float] = (-3.0, 5.0),
        baseline_window: Tuple[float, float] = (-1.0, 0.0)
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple event types.
        
        Returns:
            Dictionary mapping event labels to PSTH results
        """
        results = {}
        
        for event_label in event_labels:
            try:
                result = self.build_psth(
                    dff_signal, time_vec, events_df, event_label,
                    time_window, baseline_window
                )
                results[event_label] = result
            except Exception as e:
                warnings.warn(f"Failed to analyze event '{event_label}': {e}")
                results[event_label] = self._empty_psth_result(event_label, time_window)
        
        return results
    
    def create_summary_dataframe(self, psth_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Create summary DataFrame from PSTH results."""
        summary_rows = []
        
        for event_label, result in psth_results.items():
            if result['n_trials'] > 0:
                stats = result['summary_stats']
                row = {
                    'event_label': event_label,
                    'n_trials': result['n_trials'],
                    'mean_auc': stats['mean_auc'],
                    'sem_auc': stats['sem_auc'],
                    'mean_peak_amplitude': stats['mean_peak_amplitude'],
                    'sem_peak_amplitude': stats['sem_peak_amplitude'],
                    'mean_peak_latency': stats['mean_peak_latency'],
                    'sem_peak_latency': stats['sem_peak_latency'],
                }
                summary_rows.append(row)
        
        return pd.DataFrame(summary_rows)


class TransientAnalyzer:
    """Transient detection and analysis for fiber-photometry data."""
    
    def __init__(self, fps: float):
        self.fps = fps
    
    def detect_transients(
        self,
        dff_signal: np.ndarray,
        time_vec: np.ndarray,
        method: str = 'mad',
        threshold: float = 3.5,
        min_peak_distance: float = 1.0,
        min_peak_width: float = 0.1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Detect transients in ΔF/F signal.
        
        Args:
            dff_signal: ΔF/F signal data
            time_vec: Time vector in seconds
            method: Detection method ('mad', 'std', 'threshold')
            threshold: Detection threshold
            min_peak_distance: Minimum distance between peaks (seconds)
            min_peak_width: Minimum peak width (seconds)
            
        Returns:
            Dictionary with transient detection results
        """
        if method == 'mad':
            peaks = self._detect_mad_peaks(dff_signal, threshold, min_peak_distance, min_peak_width)
        elif method == 'std':
            peaks = self._detect_std_peaks(dff_signal, threshold, min_peak_distance, min_peak_width)
        elif method == 'threshold':
            peaks = self._detect_threshold_peaks(dff_signal, threshold, min_peak_distance, min_peak_width)
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        # Calculate peak properties
        transients = []
        for peak_idx in peaks:
            transient = self._analyze_single_transient(
                dff_signal, time_vec, peak_idx, min_peak_width
            )
            transients.append(transient)
        
        # Session statistics
        duration_hours = (time_vec[-1] - time_vec[0]) / 3600
        frequency = len(transients) / duration_hours if duration_hours > 0 else 0
        
        amplitudes = [t['amplitude'] for t in transients if not np.isnan(t['amplitude'])]
        durations = [t['duration'] for t in transients if not np.isnan(t['duration'])]
        
        result = {
            'method': method,
            'threshold': threshold,
            'n_transients': len(transients),
            'frequency_per_hour': frequency,
            'transients': transients,
            'summary_stats': {
                'mean_amplitude': np.mean(amplitudes) if amplitudes else np.nan,
                'std_amplitude': np.std(amplitudes) if amplitudes else np.nan,
                'mean_duration': np.mean(durations) if durations else np.nan,
                'std_duration': np.std(durations) if durations else np.nan,
                'amplitude_range': (np.min(amplitudes), np.max(amplitudes)) if amplitudes else (np.nan, np.nan),
                'duration_range': (np.min(durations), np.max(durations)) if durations else (np.nan, np.nan),
            }
        }
        
        return result
    
    def _detect_mad_peaks(
        self,
        signal: np.ndarray,
        threshold: float,
        min_distance: float,
        min_width: float
    ) -> np.ndarray:
        """Detect peaks using MAD-based threshold."""
        # Remove NaN values for median calculation
        valid_signal = signal[~np.isnan(signal)]
        if len(valid_signal) == 0:
            return np.array([])
        
        median_val = np.median(valid_signal)
        mad_val = np.median(np.abs(valid_signal - median_val))
        
        if mad_val == 0:
            warnings.warn("MAD is zero, cannot detect transients")
            return np.array([])
        
        # Threshold based on MAD
        peak_threshold = median_val + threshold * mad_val
        
        # Find peaks
        min_distance_samples = int(min_distance * self.fps)
        min_width_samples = int(min_width * self.fps)
        
        peaks, properties = find_peaks(
            signal,
            height=peak_threshold,
            distance=min_distance_samples,
            width=min_width_samples
        )
        
        return peaks
    
    def _detect_std_peaks(
        self,
        signal: np.ndarray,
        threshold: float,
        min_distance: float,
        min_width: float
    ) -> np.ndarray:
        """Detect peaks using standard deviation threshold."""
        valid_signal = signal[~np.isnan(signal)]
        if len(valid_signal) == 0:
            return np.array([])
        
        mean_val = np.mean(valid_signal)
        std_val = np.std(valid_signal)
        
        if std_val == 0:
            warnings.warn("Standard deviation is zero, cannot detect transients")
            return np.array([])
        
        peak_threshold = mean_val + threshold * std_val
        
        min_distance_samples = int(min_distance * self.fps)
        min_width_samples = int(min_width * self.fps)
        
        peaks, properties = find_peaks(
            signal,
            height=peak_threshold,
            distance=min_distance_samples,
            width=min_width_samples
        )
        
        return peaks
    
    def _detect_threshold_peaks(
        self,
        signal: np.ndarray,
        threshold: float,
        min_distance: float,
        min_width: float
    ) -> np.ndarray:
        """Detect peaks using absolute threshold."""
        min_distance_samples = int(min_distance * self.fps)
        min_width_samples = int(min_width * self.fps)
        
        peaks, properties = find_peaks(
            signal,
            height=threshold,
            distance=min_distance_samples,
            width=min_width_samples
        )
        
        return peaks
    
    def _analyze_single_transient(
        self,
        signal: np.ndarray,
        time_vec: np.ndarray,
        peak_idx: int,
        min_width: float
    ) -> Dict[str, Any]:
        """Analyze properties of a single transient."""
        transient = {
            'peak_time': time_vec[peak_idx],
            'peak_index': peak_idx,
            'amplitude': signal[peak_idx],
            'duration': np.nan,
            'width_at_half_max': np.nan,
            'prominence': np.nan,
            'area_under_curve': np.nan,
            'rise_time': np.nan,
            'decay_time': np.nan
        }
        
        # Calculate prominence
        try:
            prominences = signal.signal.peak_prominences(signal, [peak_idx])
            transient['prominence'] = prominences[0][0]
        except:
            pass
        
        # Find transient boundaries
        peak_amplitude = signal[peak_idx]
        half_max = peak_amplitude / 2
        
        # Search for half-maximum points
        left_bound = peak_idx
        right_bound = peak_idx
        
        # Search left
        for i in range(peak_idx - 1, max(0, peak_idx - int(5 * self.fps)), -1):
            if signal[i] <= half_max:
                left_bound = i
                break
        
        # Search right
        for i in range(peak_idx + 1, min(len(signal), peak_idx + int(5 * self.fps))):
            if signal[i] <= half_max:
                right_bound = i
                break
        
        # Width at half maximum
        if right_bound > left_bound:
            transient['width_at_half_max'] = (right_bound - left_bound) / self.fps
        
        # Find base of transient (return to baseline)
        baseline_threshold = 0.1 * peak_amplitude  # 10% of peak
        
        # Search for baseline return
        left_base = peak_idx
        right_base = peak_idx
        
        for i in range(peak_idx - 1, max(0, peak_idx - int(10 * self.fps)), -1):
            if signal[i] <= baseline_threshold:
                left_base = i
                break
        
        for i in range(peak_idx + 1, min(len(signal), peak_idx + int(10 * self.fps))):
            if signal[i] <= baseline_threshold:
                right_base = i
                break
        
        # Total duration
        if right_base > left_base:
            transient['duration'] = (right_base - left_base) / self.fps
            
            # Area under curve
            transient_data = signal[left_base:right_base + 1]
            transient_time = time_vec[left_base:right_base + 1]
            
            if len(transient_data) > 1:
                # Use trapezoid for newer scipy versions, fallback to trapz for older versions
                try:
                    transient['area_under_curve'] = integrate.trapezoid(
                        transient_data, transient_time
                    )
                except AttributeError:
                    transient['area_under_curve'] = integrate.trapz(
                        transient_data, transient_time
                    )
            
            # Rise and decay times
            transient['rise_time'] = (peak_idx - left_base) / self.fps
            transient['decay_time'] = (right_base - peak_idx) / self.fps
        
        return transient
    
    def create_transients_dataframe(self, transient_results: Dict[str, Any]) -> pd.DataFrame:
        """Create DataFrame from transient detection results."""
        transients = transient_results.get('transients', [])
        
        if not transients:
            return pd.DataFrame(columns=[
                'peak_time', 'peak_index', 'amplitude', 'duration',
                'width_at_half_max', 'prominence', 'area_under_curve',
                'rise_time', 'decay_time'
            ])
        
        return pd.DataFrame(transients)