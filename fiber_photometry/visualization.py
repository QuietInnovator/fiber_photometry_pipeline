"""
Visualization module for fiber-photometry analysis results.

Creates publication-quality figures for raw signals, ΔF/F, PSTHs,
heatmaps, and transient overlays.
"""

import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


class Visualizer:
    """Publication-quality visualization for fiber-photometry analysis."""
    
    def __init__(self, style: str = "seaborn-v0_8", dpi: int = 300):
        self.style = style
        self.dpi = dpi
    
    def _safe_legend(self, ax):
        """Add legend only if there are labeled artists."""
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend()
    
    def _setup_style(self):
        """Setup matplotlib style."""
        try:
            plt.style.use(self.style)
        except OSError:
            warnings.warn(f"Style '{self.style}' not found, using default")
            plt.style.use('default')
        
        # Set global parameters
        plt.rcParams.update({
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'lines.linewidth': 1.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
    
    def plot_raw_signals(
        self,
        time_vec: np.ndarray,
        signal_470: np.ndarray,
        signal_410: Optional[np.ndarray] = None,
        fitted_control: Optional[np.ndarray] = None,
        title: str = "Raw Fluorescence Signals",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot raw fluorescence signals with optional control fitting.
        
        Args:
            time_vec: Time vector in seconds
            signal_470: 470nm signal data
            signal_410: 410nm control data (optional)
            fitted_control: Fitted control signal (optional)
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot signals
        ax.plot(time_vec, signal_470, label='470nm Signal', color='green', alpha=0.8)
        
        if signal_410 is not None:
            ax.plot(time_vec, signal_410, label='410nm Control', color='purple', alpha=0.6)
        
        if fitted_control is not None:
            ax.plot(time_vec, fitted_control, label='Fitted Control', 
                   color='red', linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fluorescence (AU)')
        ax.set_title(title)
        self._safe_legend(ax)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_dff_and_zscores(
        self,
        time_vec: np.ndarray,
        dff: np.ndarray,
        zscore_std: Optional[np.ndarray] = None,
        zscore_baseline: Optional[np.ndarray] = None,
        zscore_modified: Optional[np.ndarray] = None,
        events_df: Optional[pd.DataFrame] = None,
        title: str = "ΔF/F and Z-scores",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ΔF/F and various z-score normalizations with event markers.
        """
        n_plots = 1  # ΔF/F is always shown
        z_scores = [zscore_std, zscore_baseline, zscore_modified]
        z_labels = ['Standard Z-score', 'Baseline Z-score', 'Modified Z-score']
        z_scores_to_plot = [(z, label) for z, label in zip(z_scores, z_labels) if z is not None]
        n_plots += len(z_scores_to_plot)
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]
        
        # Plot ΔF/F
        axes[0].plot(time_vec, dff, color='green', alpha=0.8, linewidth=1)
        axes[0].set_ylabel('ΔF/F')
        axes[0].set_title(title)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add event markers to ΔF/F plot
        if events_df is not None and len(events_df) > 0:
            self._add_event_markers(axes[0], events_df, time_vec)
        
        # Plot z-scores
        colors = ['blue', 'orange', 'red']
        for i, ((zscore, label), color) in enumerate(zip(z_scores_to_plot, colors)):
            ax = axes[i + 1]
            ax.plot(time_vec, zscore, color=color, alpha=0.8, linewidth=1)
            ax.set_ylabel('Z-score')
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add significance thresholds
            ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.axhline(y=-2, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            
            # Add event markers
            if events_df is not None and len(events_df) > 0:
                self._add_event_markers(ax, events_df, time_vec)
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_psth(
        self,
        psth_result: Dict[str, Any],
        show_individual_trials: bool = False,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot peri-stimulus time histogram (PSTH).
        
        Args:
            psth_result: PSTH analysis result dictionary
            show_individual_trials: Whether to show individual trial traces
            title: Custom title (if None, uses event label)
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        if psth_result['n_trials'] == 0:
            warnings.warn(f"No trials to plot for event '{psth_result['event_label']}'")
            return plt.figure(figsize=(8, 6))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        time_vec = psth_result['psth_time']
        mean_response = psth_result['psth_mean']
        sem_response = psth_result['psth_sem']
        
        # Plot individual trials if requested
        if show_individual_trials and len(psth_result['trials']) > 0:
            for i, trial in enumerate(psth_result['trials']):
                ax.plot(time_vec, trial, color='gray', alpha=0.3, linewidth=0.5)
        
        # Plot mean response with SEM
        ax.plot(time_vec, mean_response, color='green', linewidth=2, label='Mean')
        ax.fill_between(time_vec, 
                       mean_response - sem_response,
                       mean_response + sem_response,
                       color='green', alpha=0.3, label='±SEM')
        
        # Add event marker at t=0
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                  label=f"Event: {psth_result['event_label']}")
        
        # Add baseline window shading
        if psth_result['baseline_window'] is not None:
            baseline_start, baseline_end = psth_result['baseline_window']
            ax.axvspan(baseline_start, baseline_end, alpha=0.2, color='blue', 
                      label='Baseline Window')
        
        # Formatting
        ax.set_xlabel('Time relative to event (s)')
        ax.set_ylabel('ΔF/F (baseline corrected)')
        ax.grid(True, alpha=0.3)
        self._safe_legend(ax)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        if title is None:
            title = f"PSTH: {psth_result['event_label']} (n={psth_result['n_trials']} trials)"
        ax.set_title(title)
        
        # Add summary statistics as text
        stats = psth_result['summary_stats']
        stats_text = f"AUC: {stats['mean_auc']:.3f} ± {stats['sem_auc']:.3f}\n"
        stats_text += f"Peak: {stats['mean_peak_amplitude']:.3f} ± {stats['sem_peak_amplitude']:.3f}\n"
        stats_text += f"Latency: {stats['mean_peak_latency']:.2f} ± {stats['sem_peak_latency']:.2f} s"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_psth_comparison(
        self,
        psth_results: Dict[str, Dict[str, Any]],
        title: str = "PSTH Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare multiple PSTHs on the same plot.
        
        Args:
            psth_results: Dictionary of PSTH results for different events
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(psth_results)))
        
        for i, (event_label, result) in enumerate(psth_results.items()):
            if result['n_trials'] == 0:
                continue
                
            time_vec = result['psth_time']
            mean_response = result['psth_mean']
            sem_response = result['psth_sem']
            
            color = colors[i]
            
            # Plot mean with SEM
            ax.plot(time_vec, mean_response, color=color, linewidth=2, 
                   label=f"{event_label} (n={result['n_trials']})")
            ax.fill_between(time_vec, 
                           mean_response - sem_response,
                           mean_response + sem_response,
                           color=color, alpha=0.2)
        
        # Add event marker at t=0
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.8, linewidth=2, 
                  label='Event Onset')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax.set_xlabel('Time relative to event (s)')
        ax.set_ylabel('ΔF/F (baseline corrected)')
        ax.set_title(title)
        self._safe_legend(ax)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_psth_heatmap(
        self,
        psth_result: Dict[str, Any],
        sort_by: str = 'peak_amplitude',
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot PSTH as a heatmap with individual trials.
        
        Args:
            psth_result: PSTH analysis result dictionary
            sort_by: How to sort trials ('peak_amplitude', 'auc', 'peak_latency', or None)
            title: Custom title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        if psth_result['n_trials'] == 0:
            warnings.warn(f"No trials to plot for event '{psth_result['event_label']}'")
            return plt.figure(figsize=(10, 6))
        
        trials = psth_result['trials']
        time_vec = psth_result['psth_time']
        
        # Sort trials if requested
        if sort_by is not None and sort_by in ['peak_amplitude', 'auc', 'peak_latency']:
            sort_values = [metric[sort_by] for metric in psth_result['trial_metrics']]
            sort_indices = np.argsort(sort_values)[::-1]  # Descending order
            trials_sorted = trials[sort_indices]
        else:
            trials_sorted = trials
            sort_indices = np.arange(len(trials))
        
        # Create figure with heatmap and mean trace
        fig = plt.figure(figsize=(12, 8))
        gs = plt.GridSpec(3, 1, height_ratios=[2, 0.1, 1], hspace=0.3)
        
        # Heatmap
        ax_heat = fig.add_subplot(gs[0])
        
        # Determine colormap limits
        vmin = np.nanpercentile(trials_sorted, 5)
        vmax = np.nanpercentile(trials_sorted, 95)
        
        im = ax_heat.imshow(trials_sorted, aspect='auto', cmap='RdBu_r',
                           extent=[time_vec[0], time_vec[-1], len(trials_sorted), 0],
                           vmin=vmin, vmax=vmax)
        
        ax_heat.axvline(x=0, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax_heat.set_ylabel('Trial (sorted by ' + (sort_by or 'original order') + ')')
        ax_heat.set_title(title or f"PSTH Heatmap: {psth_result['event_label']}")
        
        # Colorbar
        ax_cbar = fig.add_subplot(gs[1])
        plt.colorbar(im, cax=ax_cbar, orientation='horizontal', 
                    label='ΔF/F (baseline corrected)')
        
        # Mean trace below
        ax_mean = fig.add_subplot(gs[2])
        mean_response = psth_result['psth_mean']
        sem_response = psth_result['psth_sem']
        
        ax_mean.plot(time_vec, mean_response, color='green', linewidth=2)
        ax_mean.fill_between(time_vec, 
                           mean_response - sem_response,
                           mean_response + sem_response,
                           color='green', alpha=0.3)
        
        ax_mean.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax_mean.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax_mean.set_xlabel('Time relative to event (s)')
        ax_mean.set_ylabel('Mean ΔF/F')
        ax_mean.grid(True, alpha=0.3)
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_transients(
        self,
        time_vec: np.ndarray,
        dff_signal: np.ndarray,
        transient_result: Dict[str, Any],
        show_overlay: bool = True,
        title: str = "Transient Detection",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot transient detection results.
        
        Args:
            time_vec: Time vector in seconds
            dff_signal: ΔF/F signal data
            transient_result: Transient detection result dictionary
            show_overlay: Whether to show overlaid transients
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        if show_overlay:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        else:
            fig, ax1 = plt.subplots(figsize=(14, 6))
            ax2 = None
        
        # Main signal plot
        ax1.plot(time_vec, dff_signal, color='blue', alpha=0.7, linewidth=1, label='ΔF/F')
        
        # Mark detected transients
        transients = transient_result['transients']
        if transients:
            for transient in transients:
                peak_time = transient['peak_time']
                amplitude = transient['amplitude']
                ax1.scatter(peak_time, amplitude, color='red', s=30, zorder=5)
                
                # Add duration bars if available
                if not np.isnan(transient['duration']):
                    start_time = peak_time - transient['rise_time']
                    end_time = peak_time + transient['decay_time']
                    ax1.plot([start_time, end_time], [amplitude*0.9, amplitude*0.9], 
                           color='orange', linewidth=2, alpha=0.7)
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('ΔF/F')
        ax1.set_title(f"{title} - {len(transients)} transients detected")
        self._safe_legend(ax1)
        ax1.grid(True, alpha=0.3)
        
        # Add summary statistics
        stats = transient_result['summary_stats']
        stats_text = f"Frequency: {transient_result['frequency_per_hour']:.2f} events/hr\n"
        stats_text += f"Mean amplitude: {stats['mean_amplitude']:.3f} ± {stats['std_amplitude']:.3f}\n"
        stats_text += f"Mean duration: {stats['mean_duration']:.2f} ± {stats['std_duration']:.2f} s"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=9)
        
        # Overlay plot
        if show_overlay and ax2 is not None and transients:
            self._plot_transient_overlay(ax2, transients, transient_result)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _plot_transient_overlay(self, ax, transients, transient_result):
        """Plot overlaid transients aligned to peak."""
        # This would require extracting segments around each transient
        # For now, plot a histogram of amplitudes and durations
        amplitudes = [t['amplitude'] for t in transients if not np.isnan(t['amplitude'])]
        durations = [t['duration'] for t in transients if not np.isnan(t['duration'])]
        
        if amplitudes:
            ax.hist(amplitudes, bins=20, alpha=0.7, color='green', label='Amplitude')
            ax.set_xlabel('Amplitude (ΔF/F)')
            ax.set_ylabel('Count')
            ax.set_title('Transient Amplitude Distribution')
            self._safe_legend(ax)
            ax.grid(True, alpha=0.3)
    
    def _add_event_markers(self, ax, events_df: pd.DataFrame, time_vec: np.ndarray):
        """Add vertical lines for event markers."""
        if 'timestamp' not in events_df.columns:
            return
        
        # Get unique event types for color coding
        unique_events = events_df['event_label'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_events)))
        event_colors = dict(zip(unique_events, colors))
        
        for _, event in events_df.iterrows():
            event_time = event['timestamp']
            event_label = event['event_label']
            
            # Only plot if event is within time range
            if time_vec[0] <= event_time <= time_vec[-1]:
                ax.axvline(x=event_time, color=event_colors[event_label], 
                          linestyle=':', alpha=0.7, linewidth=1.5,
                          label=event_label if event_label not in ax.get_legend_handles_labels()[1] else "")
    
    def _save_figure(self, fig: plt.Figure, save_path: str, formats: List[str] = ['png', 'svg']):
        """Save figure in multiple formats."""
        save_path = Path(save_path)
        
        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        for fmt in formats:
            output_path = save_path.with_suffix(f'.{fmt}')
            fig.savefig(output_path, format=fmt, dpi=self.dpi, bbox_inches='tight')
    
    def create_summary_report_figure(
        self,
        psth_results: Dict[str, Dict[str, Any]],
        transient_result: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive summary figure with multiple panels.
        
        Args:
            psth_results: Dictionary of PSTH results for different events
            transient_result: Transient detection results (optional)
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        # Determine layout based on available data
        n_events = len([r for r in psth_results.values() if r['n_trials'] > 0])
        has_transients = transient_result is not None and len(transient_result['transients']) > 0
        
        if has_transients:
            fig = plt.figure(figsize=(16, 12))
            gs = plt.GridSpec(3, 2, height_ratios=[2, 2, 1], width_ratios=[2, 1])
        else:
            fig = plt.figure(figsize=(14, 8))
            gs = plt.GridSpec(2, 2, height_ratios=[2, 1])
        
        # PSTH comparison (top left)
        ax_psth = fig.add_subplot(gs[0, 0])
        colors = plt.cm.Set1(np.linspace(0, 1, n_events))
        
        i = 0
        for event_label, result in psth_results.items():
            if result['n_trials'] == 0:
                continue
                
            time_vec = result['psth_time']
            mean_response = result['psth_mean']
            sem_response = result['psth_sem']
            
            color = colors[i]
            ax_psth.plot(time_vec, mean_response, color=color, linewidth=2,
                        label=f"{event_label} (n={result['n_trials']})")
            ax_psth.fill_between(time_vec, 
                               mean_response - sem_response,
                               mean_response + sem_response,
                               color=color, alpha=0.2)
            i += 1
        
        ax_psth.axvline(x=0, color='black', linestyle='--', alpha=0.8, linewidth=1)
        ax_psth.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax_psth.set_xlabel('Time relative to event (s)')
        ax_psth.set_ylabel('ΔF/F (baseline corrected)')
        ax_psth.set_title('Event-Related Responses (PSTH)')
        self._safe_legend(ax_psth)
        ax_psth.grid(True, alpha=0.3)
        
        # Summary statistics table (top right)
        ax_table = fig.add_subplot(gs[0, 1])
        ax_table.axis('off')
        
        # Create summary statistics table
        summary_data = []
        for event_label, result in psth_results.items():
            if result['n_trials'] > 0:
                stats = result['summary_stats']
                summary_data.append([
                    event_label,
                    result['n_trials'],
                    f"{stats['mean_auc']:.3f} ± {stats['sem_auc']:.3f}",
                    f"{stats['mean_peak_amplitude']:.3f} ± {stats['sem_peak_amplitude']:.3f}",
                    f"{stats['mean_peak_latency']:.2f} ± {stats['sem_peak_latency']:.2f}"
                ])
        
        if summary_data:
            table = ax_table.table(
                cellText=summary_data,
                colLabels=['Event', 'N Trials', 'AUC (Mean±SEM)', 'Peak Amp', 'Peak Lat (s)'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax_table.set_title('Summary Statistics')
        
        # Bar plot of key metrics (bottom panels)
        if summary_data:
            ax_auc = fig.add_subplot(gs[1, 0])
            ax_amp = fig.add_subplot(gs[1, 1])
            
            event_names = [row[0] for row in summary_data]
            auc_values = [float(row[2].split(' ±')[0]) for row in summary_data]
            auc_errors = [float(row[2].split('± ')[1]) for row in summary_data]
            amp_values = [float(row[3].split(' ±')[0]) for row in summary_data]
            amp_errors = [float(row[3].split('± ')[1]) for row in summary_data]
            
            x_pos = np.arange(len(event_names))
            
            ax_auc.bar(x_pos, auc_values, yerr=auc_errors, capsize=5,
                      color=colors[:len(event_names)], alpha=0.7)
            ax_auc.set_xlabel('Event Type')
            ax_auc.set_ylabel('Area Under Curve')
            ax_auc.set_title('AUC by Event Type')
            ax_auc.set_xticks(x_pos)
            ax_auc.set_xticklabels(event_names, rotation=45)
            ax_auc.grid(True, alpha=0.3)
            
            ax_amp.bar(x_pos, amp_values, yerr=amp_errors, capsize=5,
                      color=colors[:len(event_names)], alpha=0.7)
            ax_amp.set_xlabel('Event Type')
            ax_amp.set_ylabel('Peak Amplitude (ΔF/F)')
            ax_amp.set_title('Peak Amplitude by Event Type')
            ax_amp.set_xticks(x_pos)
            ax_amp.set_xticklabels(event_names, rotation=45)
            ax_amp.grid(True, alpha=0.3)
        
        # Transient summary (bottom row if present)
        if has_transients:
            ax_trans = fig.add_subplot(gs[2, :])
            
            transients = transient_result['transients']
            amplitudes = [t['amplitude'] for t in transients if not np.isnan(t['amplitude'])]
            durations = [t['duration'] for t in transients if not np.isnan(t['duration'])]
            
            # Histogram of transient properties
            if amplitudes and durations:
                ax_trans.hist2d(durations, amplitudes, bins=20, cmap='Blues', alpha=0.7)
                ax_trans.set_xlabel('Duration (s)')
                ax_trans.set_ylabel('Amplitude (ΔF/F)')
                ax_trans.set_title(f'Transient Properties (n={len(transients)}, '
                                 f'{transient_result["frequency_per_hour"]:.1f} events/hr)')
        
        plt.suptitle('Fiber Photometry Analysis Summary', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig