"""
Normalization module for fiber-photometry signals.

Handles ΔF/F calculations, z-score normalization, and baseline correction
using various methods including control channel fitting and debleaching.
"""

import warnings
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
from scipy import optimize
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class Normalizer:
    """Signal normalization for fiber-photometry data."""
    
    def __init__(self, fps: float):
        self.fps = fps
        
    def fit_control_to_signal(
        self,
        signal: np.ndarray,
        control: np.ndarray,
        method: str = 'linear'
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fit control channel to signal channel.
        
        Args:
            signal: Signal channel data (e.g., 470nm)
            control: Control channel data (e.g., 410nm)
            method: Fitting method ('linear', 'polynomial', 'robust')
            
        Returns:
            Tuple of (fitted_control, fit_info)
        """
        if len(signal) != len(control):
            raise ValueError("Signal and control must have same length")
            
        # Remove NaN values
        valid_mask = ~(np.isnan(signal) | np.isnan(control))
        if not np.any(valid_mask):
            raise ValueError("No valid data points found")
            
        signal_clean = signal[valid_mask]
        control_clean = control[valid_mask]
        
        if method == 'linear':
            fitted_control, fit_info = self._linear_fit(
                signal_clean, control_clean, valid_mask
            )
        elif method == 'polynomial':
            fitted_control, fit_info = self._polynomial_fit(
                signal_clean, control_clean, valid_mask, degree=2
            )
        elif method == 'robust':
            fitted_control, fit_info = self._robust_fit(
                signal_clean, control_clean, valid_mask
            )
        else:
            raise ValueError(f"Unknown fitting method: {method}")
            
        return fitted_control, fit_info
    
    def _linear_fit(
        self,
        signal: np.ndarray,
        control: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Linear least squares fit."""
        # Fit control to signal: signal = a * control + b
        A = np.vstack([control, np.ones(len(control))]).T
        coeffs, residuals, rank, s = np.linalg.lstsq(A, signal, rcond=None)
        
        slope, intercept = coeffs
        fitted_control = np.full_like(valid_mask, np.nan, dtype=float)
        fitted_control[valid_mask] = slope * control + intercept
        
        # Calculate R²
        ss_res = np.sum((signal - (slope * control + intercept)) ** 2)
        ss_tot = np.sum((signal - np.mean(signal)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        fit_info = {
            'method': 'linear',
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'residuals': residuals[0] if len(residuals) > 0 else np.nan
        }
        
        return fitted_control, fit_info
    
    def _polynomial_fit(
        self,
        signal: np.ndarray,
        control: np.ndarray,
        valid_mask: np.ndarray,
        degree: int = 2
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Polynomial fit."""
        poly_features = PolynomialFeatures(degree=degree, include_bias=True)
        control_poly = poly_features.fit_transform(control.reshape(-1, 1))
        
        reg = LinearRegression()
        reg.fit(control_poly, signal)
        
        # Predict on full control data
        full_control = np.full(len(valid_mask), np.nan)
        full_control[valid_mask] = control
        
        fitted_control = np.full_like(valid_mask, np.nan, dtype=float)
        
        # Only predict for valid points
        if np.any(valid_mask):
            valid_control_poly = poly_features.transform(
                full_control[valid_mask].reshape(-1, 1)
            )
            fitted_control[valid_mask] = reg.predict(valid_control_poly)
        
        r_squared = reg.score(control_poly, signal)
        
        fit_info = {
            'method': 'polynomial',
            'degree': degree,
            'coefficients': reg.coef_,
            'intercept': reg.intercept_,
            'r_squared': r_squared
        }
        
        return fitted_control, fit_info
    
    def _robust_fit(
        self,
        signal: np.ndarray,
        control: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Robust linear fit using RANSAC-like approach."""
        from sklearn.linear_model import RANSACRegressor
        from sklearn.linear_model import LinearRegression
        
        reg = RANSACRegressor(
            LinearRegression(),
            min_samples=int(0.8 * len(control)),
            residual_threshold=np.std(signal) * 2,
            random_state=42
        )
        
        reg.fit(control.reshape(-1, 1), signal)
        
        fitted_control = np.full_like(valid_mask, np.nan, dtype=float)
        fitted_control[valid_mask] = reg.predict(control.reshape(-1, 1))
        
        r_squared = reg.score(control.reshape(-1, 1), signal)
        
        fit_info = {
            'method': 'robust',
            'slope': reg.estimator_.coef_[0],
            'intercept': reg.estimator_.intercept_,
            'r_squared': r_squared,
            'inlier_mask': reg.inlier_mask_
        }
        
        return fitted_control, fit_info
    
    def calculate_dff(
        self,
        signal: np.ndarray,
        control: Optional[np.ndarray] = None,
        baseline_method: str = 'control_fit',
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calculate ΔF/F using various baseline methods.
        
        Args:
            signal: Raw signal channel
            control: Control channel (optional, needed for control_fit)
            baseline_method: Method for baseline calculation
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Tuple of (dff, calculation_info)
        """
        if baseline_method == 'control_fit' and control is not None:
            return self._dff_control_fit(signal, control, **kwargs)
        elif baseline_method == 'baseline_fit':
            return self._dff_baseline_fit(signal, **kwargs)
        elif baseline_method == 'detrend':
            return self._dff_detrend(signal, **kwargs)
        elif baseline_method == 'percentile':
            return self._dff_percentile(signal, **kwargs)
        else:
            raise ValueError(f"Unknown baseline method: {baseline_method}")
    
    def _dff_control_fit(
        self,
        signal: np.ndarray,
        control: np.ndarray,
        fit_method: str = 'linear'
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calculate ΔF/F using control channel fitting."""
        fitted_control, fit_info = self.fit_control_to_signal(
            signal, control, method=fit_method
        )
        
        # Calculate ΔF/F = (Signal - FittedControl) / FittedControl
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dff = (signal - fitted_control) / fitted_control
            
        # Handle division by zero/negative values
        invalid_mask = (fitted_control <= 0) | np.isnan(fitted_control)
        if np.any(invalid_mask):
            warnings.warn(f"Found {np.sum(invalid_mask)} invalid baseline values")
            dff[invalid_mask] = np.nan
        
        calc_info = {
            'method': 'control_fit',
            'fit_info': fit_info,
            'baseline': fitted_control
        }
        
        return dff, calc_info
    
    def _dff_baseline_fit(
        self,
        signal: np.ndarray,
        percentile: float = 10.0,
        window_size: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calculate ΔF/F using percentile-based baseline fitting."""
        if window_size is None:
            window_size = int(30 * self.fps)  # 30 second window
        
        # Calculate rolling percentile baseline
        baseline = self._rolling_percentile(signal, percentile, window_size)
        
        # Calculate ΔF/F
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dff = (signal - baseline) / baseline
        
        # Handle invalid values
        invalid_mask = (baseline <= 0) | np.isnan(baseline)
        if np.any(invalid_mask):
            dff[invalid_mask] = np.nan
        
        calc_info = {
            'method': 'baseline_fit',
            'percentile': percentile,
            'window_size': window_size,
            'baseline': baseline
        }
        
        return dff, calc_info
    
    def _dff_detrend(
        self,
        signal: np.ndarray,
        method: str = 'exponential'
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calculate ΔF/F using detrending methods."""
        if method == 'exponential':
            baseline = self._fit_exponential_decay(signal)
        elif method == 'linear':
            baseline = self._fit_linear_trend(signal)
        elif method == 'savgol':
            window_length = min(int(30 * self.fps), len(signal) // 4)
            if window_length % 2 == 0:
                window_length += 1
            if window_length < 3:
                window_length = 3
            baseline = savgol_filter(signal, window_length, polyorder=2)
        else:
            raise ValueError(f"Unknown detrend method: {method}")
        
        # Calculate ΔF/F
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dff = (signal - baseline) / baseline
        
        invalid_mask = (baseline <= 0) | np.isnan(baseline)
        if np.any(invalid_mask):
            dff[invalid_mask] = np.nan
        
        calc_info = {
            'method': 'detrend',
            'detrend_method': method,
            'baseline': baseline
        }
        
        return dff, calc_info
    
    def _dff_percentile(
        self,
        signal: np.ndarray,
        percentile: float = 10.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calculate ΔF/F using global percentile baseline."""
        baseline_value = np.percentile(signal[~np.isnan(signal)], percentile)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dff = (signal - baseline_value) / baseline_value
        
        calc_info = {
            'method': 'percentile',
            'percentile': percentile,
            'baseline_value': baseline_value,
            'baseline': np.full_like(signal, baseline_value)
        }
        
        return dff, calc_info
    
    def _rolling_percentile(
        self,
        signal: np.ndarray,
        percentile: float,
        window_size: int
    ) -> np.ndarray:
        """Calculate rolling percentile baseline."""
        baseline = np.full_like(signal, np.nan)
        half_window = window_size // 2
        
        for i in range(len(signal)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(signal), i + half_window + 1)
            window_data = signal[start_idx:end_idx]
            
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) > 0:
                baseline[i] = np.percentile(valid_data, percentile)
        
        return baseline
    
    def _fit_exponential_decay(self, signal: np.ndarray) -> np.ndarray:
        """Fit exponential decay to signal for bleaching correction."""
        time_vec = np.arange(len(signal)) / self.fps
        
        # Remove NaN values for fitting
        valid_mask = ~np.isnan(signal)
        if not np.any(valid_mask):
            return np.full_like(signal, np.nan)
        
        time_valid = time_vec[valid_mask]
        signal_valid = signal[valid_mask]
        
        def exp_func(t, a, b, c):
            return a * np.exp(-b * t) + c
        
        # Initial parameter guess
        a_guess = signal_valid.max() - signal_valid.min()
        b_guess = 0.001  # Small decay rate
        c_guess = signal_valid.min()
        
        try:
            params, _ = optimize.curve_fit(
                exp_func, time_valid, signal_valid,
                p0=[a_guess, b_guess, c_guess],
                bounds=([0, 0, 0], [np.inf, 1, np.inf]),
                maxfev=1000
            )
            baseline = exp_func(time_vec, *params)
        except (RuntimeError, ValueError):
            # Fallback to linear trend if exponential fit fails
            warnings.warn("Exponential fit failed, using linear trend")
            baseline = self._fit_linear_trend(signal)
        
        return baseline
    
    def _fit_linear_trend(self, signal: np.ndarray) -> np.ndarray:
        """Fit linear trend to signal."""
        time_vec = np.arange(len(signal))
        valid_mask = ~np.isnan(signal)
        
        if not np.any(valid_mask):
            return np.full_like(signal, np.nan)
        
        coeffs = np.polyfit(time_vec[valid_mask], signal[valid_mask], deg=1)
        baseline = np.polyval(coeffs, time_vec)
        
        return baseline
    
    def calculate_zscore(
        self,
        dff: np.ndarray,
        method: str = 'standard',
        baseline_window: Optional[Tuple[float, float]] = None,
        time_vec: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calculate z-score normalization of ΔF/F.
        
        Args:
            dff: ΔF/F values
            method: 'standard', 'baseline', or 'modified'
            baseline_window: Time window for baseline calculation (start, end) in seconds
            time_vec: Time vector for baseline window calculation
            
        Returns:
            Tuple of (zscore, calculation_info)
        """
        if method == 'standard':
            return self._zscore_standard(dff)
        elif method == 'baseline':
            if baseline_window is None or time_vec is None:
                raise ValueError("baseline_window and time_vec required for baseline z-score")
            return self._zscore_baseline(dff, baseline_window, time_vec)
        elif method == 'modified':
            return self._zscore_modified(dff)
        else:
            raise ValueError(f"Unknown z-score method: {method}")
    
    def _zscore_standard(self, dff: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Standard z-score: (x - mean) / std"""
        valid_data = dff[~np.isnan(dff)]
        if len(valid_data) == 0:
            return np.full_like(dff, np.nan), {'method': 'standard', 'mean': np.nan, 'std': np.nan}
        
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
        
        if std_val == 0:
            warnings.warn("Standard deviation is zero, cannot calculate z-score")
            zscore = np.full_like(dff, np.nan)
        else:
            zscore = (dff - mean_val) / std_val
        
        calc_info = {
            'method': 'standard',
            'mean': mean_val,
            'std': std_val
        }
        
        return zscore, calc_info
    
    def _zscore_baseline(
        self,
        dff: np.ndarray,
        baseline_window: Tuple[float, float],
        time_vec: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Baseline z-score using specified time window."""
        start_time, end_time = baseline_window
        baseline_mask = (time_vec >= start_time) & (time_vec <= end_time)
        
        if not np.any(baseline_mask):
            warnings.warn(f"No data found in baseline window {baseline_window}")
            return self._zscore_standard(dff)
        
        baseline_data = dff[baseline_mask]
        valid_baseline = baseline_data[~np.isnan(baseline_data)]
        
        if len(valid_baseline) == 0:
            warnings.warn("No valid data in baseline window")
            return self._zscore_standard(dff)
        
        mean_val = np.mean(valid_baseline)
        std_val = np.std(valid_baseline)
        
        if std_val == 0:
            warnings.warn("Baseline standard deviation is zero")
            zscore = np.full_like(dff, np.nan)
        else:
            zscore = (dff - mean_val) / std_val
        
        calc_info = {
            'method': 'baseline',
            'baseline_window': baseline_window,
            'mean': mean_val,
            'std': std_val,
            'n_baseline_points': len(valid_baseline)
        }
        
        return zscore, calc_info
    
    def _zscore_modified(self, dff: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Modified z-score using median absolute deviation (MAD)."""
        valid_data = dff[~np.isnan(dff)]
        if len(valid_data) == 0:
            return np.full_like(dff, np.nan), {'method': 'modified', 'median': np.nan, 'mad': np.nan}
        
        median_val = np.median(valid_data)
        mad_val = np.median(np.abs(valid_data - median_val))
        
        if mad_val == 0:
            warnings.warn("MAD is zero, cannot calculate modified z-score")
            zscore = np.full_like(dff, np.nan)
        else:
            # Modified z-score = 0.6745 * (x - median) / MAD
            zscore = 0.6745 * (dff - median_val) / mad_val
        
        calc_info = {
            'method': 'modified',
            'median': median_val,
            'mad': mad_val
        }
        
        return zscore, calc_info