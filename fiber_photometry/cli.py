"""
Command-line interface for fiber-photometry analysis pipeline.

Provides a comprehensive CLI with configuration file support and 
progress tracking for batch analysis.
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional

import click
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .parser import FluorescenceParser
from .preprocessing import Preprocessor
from .normalization import Normalizer
from .analysis import EventAnalyzer, TransientAnalyzer
from .visualization import Visualizer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_results(
    output_dir: Path,
    metadata: Dict[str, Any],
    timeseries_df: pd.DataFrame,
    events_df: pd.DataFrame,
    psth_results: Dict[str, Dict[str, Any]],
    transient_results: Optional[Dict[str, Any]],
    processing_log: Dict[str, Any]
):
    """Save all analysis results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Save timeseries data
    timeseries_df.to_csv(output_dir / 'timeseries.csv', index=False)
    
    # Save events
    if not events_df.empty:
        events_df.to_csv(output_dir / 'events.csv', index=False)
    
    # Save PSTH results
    for event_label, result in psth_results.items():
        if result['n_trials'] > 0:
            # Save individual PSTH
            psth_df = pd.DataFrame({
                'time': result['psth_time'],
                'mean': result['psth_mean'],
                'sem': result['psth_sem'],
                'std': result['psth_std']
            })
            psth_df.to_csv(output_dir / f'psth_{event_label.replace(" ", "_")}.csv', index=False)
            
            # Save trial data
            if len(result['trials']) > 0:
                trials_df = pd.DataFrame(result['trials'].T, columns=[f'trial_{i}' for i in range(len(result['trials']))])
                trials_df['time'] = result['psth_time']
                trials_df.to_csv(output_dir / f'trials_{event_label.replace(" ", "_")}.csv', index=False)
    
    # Save transient results
    if transient_results and len(transient_results['transients']) > 0:
        transients_df = pd.DataFrame(transient_results['transients'])
        transients_df.to_csv(output_dir / 'transients.csv', index=False)
    
    # Save processing log
    with open(output_dir / 'processing_log.json', 'w') as f:
        json.dump(processing_log, f, indent=2, default=str)


def create_report(
    output_dir: Path,
    metadata: Dict[str, Any],
    processing_log: Dict[str, Any],
    psth_results: Dict[str, Dict[str, Any]],
    transient_results: Optional[Dict[str, Any]],
    config: Dict[str, Any]
) -> str:
    """Create markdown analysis report."""
    report_lines = [
        "# Fiber Photometry Analysis Report",
        "",
        f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Input File:** {config['input_file']}",
        f"**Output Directory:** {output_dir}",
        "",
        "## Data Summary",
        "",
        f"- **Recording Duration:** {processing_log.get('final_duration', 'N/A'):.2f} seconds",
        f"- **Sample Count:** {processing_log.get('final_samples', 'N/A')}",
        f"- **Sampling Rate:** {metadata.get('Fps', 'N/A')} Hz",
        f"- **Channels:** {', '.join(metadata.get('detected_channels', []))}",
        f"- **Dual Channel:** {'Yes' if metadata.get('dual_channel', False) else 'No'}",
        "",
        "## Preprocessing",
        ""
    ]
    
    if 'trimmed_seconds' in processing_log:
        report_lines.append(f"- **LED Artifact Trimmed:** {processing_log['trimmed_seconds']} seconds")
    
    if 'smoothing_window' in processing_log:
        report_lines.append(f"- **Smoothing Window:** {processing_log['smoothing_window']} seconds")
    
    if 'outliers_removed' in processing_log:
        report_lines.append(f"- **Outliers Removed:** {processing_log['outliers_removed']}")
    
    if 'bad_segments_removed' in processing_log:
        report_lines.extend([
            f"- **Bad Segments Removed:** {processing_log['bad_segments_removed']}",
        ])
        if 'bad_segment_times' in processing_log:
            for i, (start, end) in enumerate(processing_log['bad_segment_times']):
                report_lines.append(f"  - Segment {i+1}: {start:.1f} - {end:.1f} s")
    
    # Event analysis results
    report_lines.extend([
        "",
        "## Event-Related Analysis",
        "",
        "| Event Type | N Trials | Mean AUC | Peak Amplitude | Peak Latency (s) |",
        "|------------|----------|----------|----------------|------------------|"
    ])
    
    for event_label, result in psth_results.items():
        if result['n_trials'] > 0:
            stats = result['summary_stats']
            report_lines.append(
                f"| {event_label} | {result['n_trials']} | "
                f"{stats['mean_auc']:.3f} ± {stats['sem_auc']:.3f} | "
                f"{stats['mean_peak_amplitude']:.3f} ± {stats['sem_peak_amplitude']:.3f} | "
                f"{stats['mean_peak_latency']:.2f} ± {stats['sem_peak_latency']:.2f} |"
            )
    
    # Transient analysis results
    if transient_results and len(transient_results['transients']) > 0:
        stats = transient_results['summary_stats']
        report_lines.extend([
            "",
            "## Transient Analysis",
            "",
            f"- **Total Transients:** {transient_results['n_transients']}",
            f"- **Frequency:** {transient_results['frequency_per_hour']:.2f} events/hour",
            f"- **Mean Amplitude:** {stats['mean_amplitude']:.3f} ± {stats['std_amplitude']:.3f} ΔF/F",
            f"- **Mean Duration:** {stats['mean_duration']:.2f} ± {stats['std_duration']:.2f} seconds",
            f"- **Amplitude Range:** {stats['amplitude_range'][0]:.3f} - {stats['amplitude_range'][1]:.3f} ΔF/F",
            f"- **Duration Range:** {stats['duration_range'][0]:.2f} - {stats['duration_range'][1]:.2f} seconds",
        ])
    
    # Analysis parameters
    report_lines.extend([
        "",
        "## Analysis Parameters",
        "",
        "### Preprocessing",
        f"- Trim LED artifact: {config['preprocessing']['trim_led_artifact']} s",
        f"- Filter window: {config['preprocessing']['filter_window']} s",
        f"- Remove bad segments: {config['preprocessing']['remove_bad_segments']}",
        "",
        "### Normalization",
        f"- Method: {config['normalization']['method']}",
        f"- Baseline percentile: {config['normalization']['baseline_percentile']}%",
        "",
        "### Event Analysis",
        f"- PSTH window: {config['event_analysis']['psth_window']} s",
        f"- Baseline window: {config['event_analysis']['baseline_window']} s",
        f"- Events analyzed: {', '.join(config['event_analysis']['events'])}",
        "",
        "### Transient Detection",
        f"- Method: {config['transient_detection']['method']}",
        f"- Threshold: {config['transient_detection']['mad_threshold']}",
        f"- Min peak distance: {config['transient_detection']['min_peak_distance']} s",
        f"- Min peak width: {config['transient_detection']['min_peak_width']} s",
        "",
        "## Files Generated",
        "",
        "- `metadata.json` - Recording metadata and parameters",
        "- `timeseries.csv` - Processed time-series data",
        "- `events.csv` - Event markers and timestamps",
        "- `psth_*.csv` - PSTH data for each event type",
        "- `trials_*.csv` - Individual trial data for each event type",
        "- `transients.csv` - Detected transients with properties",
        "- `processing_log.json` - Detailed processing log",
        "- Figures: Raw signals, ΔF/F, PSTHs, heatmaps, transients",
        "",
        "---",
        "",
        "*Generated with fiber-photometry analysis pipeline*"
    ])
    
    report_content = "\n".join(report_lines)
    
    # Save report
    with open(output_dir / 'REPORT.md', 'w') as f:
        f.write(report_content)
    
    return report_content


@click.command()
@click.option('--input', '-i', 'input_file', required=True, 
              help='Path to input CSV file')
@click.option('--output', '-o', 'output_dir', default='output',
              help='Output directory for results')
@click.option('--config', '-c', 'config_file', default='config.yaml',
              help='Path to configuration file')
@click.option('--fps', type=float, help='Override sampling rate (Hz)')
@click.option('--trim-led', type=float, help='LED artifact trim time (seconds)')
@click.option('--filter-window', type=float, help='Smoothing window (seconds)')
@click.option('--psth-window', nargs=2, type=float, help='PSTH window (pre post)')
@click.option('--baseline-window', nargs=2, type=float, help='Baseline window (pre post)')
@click.option('--channels', multiple=True, help='Channel wavelengths to analyze')
@click.option('--events', multiple=True, help='Event types to analyze')
@click.option('--no-figures', is_flag=True, help='Skip figure generation')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def main(
    input_file: str,
    output_dir: str,
    config_file: str,
    fps: Optional[float],
    trim_led: Optional[float],
    filter_window: Optional[float],
    psth_window: Optional[List[float]],
    baseline_window: Optional[List[float]],
    channels: List[str],
    events: List[str],
    no_figures: bool,
    verbose: bool
):
    """
    Analyze fiber-photometry data with comprehensive preprocessing,
    normalization, event analysis, and visualization.
    
    Examples:
    
        # Basic analysis with default config
        analyze-photometry -i Fluorescence.csv
        
        # Custom parameters
        analyze-photometry -i data.csv -o results --fps 100 --trim-led 1.0
        
        # Specify events and channels
        analyze-photometry -i data.csv --events cocaine "foot shock" --channels 470 410
        
        # Use custom config file
        analyze-photometry -i data.csv -c my_config.yaml
    """
    
    # Load configuration
    config_path = Path(config_file)
    if not config_path.exists():
        click.echo(f"Warning: Config file {config_file} not found, using defaults")
        config = {
            'preprocessing': {'trim_led_artifact': 1.0, 'filter_window': 0.5},
            'event_analysis': {'psth_window': [-3.0, 5.0], 'baseline_window': [-1.0, 0.0], 'events': ['cocaine', 'foot shock', 'lick']},
            'transient_detection': {'method': 'mad', 'mad_threshold': 3.5, 'min_peak_distance': 1.0, 'min_peak_width': 0.1},
            'normalization': {'method': 'control_fit'},
            'visualization': {'figure_format': ['png', 'svg']}
        }
    else:
        config = load_config(config_file)
    
    # Override config with command line arguments
    if fps is not None:
        config['fps'] = fps
    if trim_led is not None:
        config['preprocessing']['trim_led_artifact'] = trim_led
    if filter_window is not None:
        config['preprocessing']['filter_window'] = filter_window
    if psth_window is not None:
        config['event_analysis']['psth_window'] = list(psth_window)
    if baseline_window is not None:
        config['event_analysis']['baseline_window'] = list(baseline_window)
    if channels:
        config['channels'] = list(channels)
    if events:
        config['event_analysis']['events'] = list(events)
    
    # Set up output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        click.echo(f"Starting fiber-photometry analysis...")
        click.echo(f"Input: {input_file}")
        click.echo(f"Output: {output_path}")
    
    try:
        # 1. Parse data
        if verbose:
            click.echo("1. Parsing CSV data...")
        
        parser = FluorescenceParser(input_file)
        metadata, timeseries_df, events_df = parser.parse()
        
        if verbose:
            summary = parser.summary()
            click.echo(f"   Duration: {summary['duration_seconds']:.1f} s")
            click.echo(f"   Samples: {summary['n_samples']}")
            click.echo(f"   Channels: {summary['channels']}")
            click.echo(f"   Events: {summary['n_events']}")
        
        # 2. Preprocessing
        if verbose:
            click.echo("2. Preprocessing signals...")
        
        fps_rate = config.get('fps', metadata.get('Fps', 100.0))
        preprocessor = Preprocessor(fps_rate)
        
        time_vec = parser.get_time_vector()
        signal_470 = parser.get_channel_data('470')
        signal_410 = parser.get_channel_data('410')
        
        if signal_470 is None:
            raise ValueError("No 470nm signal channel found")
        
        # Stack signals for processing
        if signal_410 is not None:
            signals = np.column_stack([signal_470, signal_410])
        else:
            signals = signal_470
        
        processed_time, processed_signals, processing_log = preprocessor.preprocess_pipeline(
            time_vec, signals, config['preprocessing']
        )
        
        if processed_signals.ndim == 1:
            processed_470 = processed_signals
            processed_410 = None
        else:
            processed_470 = processed_signals[:, 0]
            processed_410 = processed_signals[:, 1] if processed_signals.shape[1] > 1 else None
        
        if verbose:
            click.echo(f"   Final duration: {processing_log['final_duration']:.1f} s")
            click.echo(f"   Final samples: {processing_log['final_samples']}")
        
        # 3. Normalization
        if verbose:
            click.echo("3. Calculating ΔF/F and z-scores...")
        
        normalizer = Normalizer(fps_rate)
        
        # Calculate ΔF/F
        if processed_410 is not None:
            dff, dff_info = normalizer.calculate_dff(
                processed_470, processed_410, 
                baseline_method=config['normalization']['method']
            )
        else:
            dff, dff_info = normalizer.calculate_dff(
                processed_470, baseline_method='baseline_fit',
                percentile=config['normalization'].get('baseline_percentile', 10)
            )
        
        # Calculate z-scores
        zscore_std, _ = normalizer.calculate_zscore(dff, method='standard')
        zscore_baseline, _ = normalizer.calculate_zscore(
            dff, method='baseline', 
            baseline_window=tuple(config['event_analysis']['baseline_window']),
            time_vec=processed_time
        )
        zscore_modified, _ = normalizer.calculate_zscore(dff, method='modified')
        
        # 4. Create comprehensive timeseries DataFrame
        timeseries_data = {
            'time': processed_time,
            'signal_470': processed_470,
            'dff': dff,
            'zscore_std': zscore_std,
            'zscore_baseline': zscore_baseline,
            'zscore_modified': zscore_modified
        }
        
        if processed_410 is not None:
            timeseries_data['signal_410'] = processed_410
            if 'baseline' in dff_info:
                timeseries_data['fitted_control'] = dff_info['baseline']
        
        final_timeseries_df = pd.DataFrame(timeseries_data)
        
        # 5. Event analysis
        if verbose:
            click.echo("4. Analyzing events...")
        
        event_analyzer = EventAnalyzer(fps_rate)
        
        psth_results = event_analyzer.analyze_all_events(
            dff, processed_time, events_df,
            event_labels=config['event_analysis']['events'],
            time_window=tuple(config['event_analysis']['psth_window']),
            baseline_window=tuple(config['event_analysis']['baseline_window'])
        )
        
        if verbose:
            for event_label, result in psth_results.items():
                click.echo(f"   {event_label}: {result['n_trials']} trials")
        
        # 6. Transient detection
        if verbose:
            click.echo("5. Detecting transients...")
        
        transient_analyzer = TransientAnalyzer(fps_rate)
        transient_results = transient_analyzer.detect_transients(
            dff, processed_time,
            method=config['transient_detection']['method'],
            threshold=config['transient_detection']['mad_threshold'],
            min_peak_distance=config['transient_detection']['min_peak_distance'],
            min_peak_width=config['transient_detection']['min_peak_width']
        )
        
        if verbose:
            click.echo(f"   Detected {transient_results['n_transients']} transients")
            click.echo(f"   Frequency: {transient_results['frequency_per_hour']:.1f} events/hr")
        
        # 7. Save results
        if verbose:
            click.echo("6. Saving results...")
        
        save_results(
            output_path, metadata, final_timeseries_df, events_df,
            psth_results, transient_results, processing_log
        )
        
        # 8. Generate figures
        if not no_figures:
            if verbose:
                click.echo("7. Generating figures...")
            
            visualizer = Visualizer(
                style=config.get('visualization', {}).get('style', 'seaborn-v0_8'),
                dpi=config.get('visualization', {}).get('dpi', 300)
            )
            
            figure_formats = config.get('visualization', {}).get('figure_format', ['png'])
            
            with tqdm(total=6, desc="Creating figures", disable=not verbose) as pbar:
                # Raw signals
                fig_raw = visualizer.plot_raw_signals(
                    processed_time, processed_470, processed_410,
                    fitted_control=dff_info.get('baseline'),
                    save_path=str(output_path / 'raw_signals')
                )
                plt.close(fig_raw)
                pbar.update(1)
                
                # ΔF/F and z-scores
                fig_dff = visualizer.plot_dff_and_zscores(
                    processed_time, dff, zscore_std, zscore_baseline, zscore_modified,
                    events_df=events_df,
                    save_path=str(output_path / 'dff_and_zscores')
                )
                plt.close(fig_dff)
                pbar.update(1)
                
                # Individual PSTHs
                for event_label, result in psth_results.items():
                    if result['n_trials'] > 0:
                        fig_psth = visualizer.plot_psth(
                            result,
                            save_path=str(output_path / f'psth_{event_label.replace(" ", "_")}')
                        )
                        plt.close(fig_psth)
                pbar.update(1)
                
                # PSTH comparison
                fig_comp = visualizer.plot_psth_comparison(
                    psth_results,
                    save_path=str(output_path / 'psth_comparison')
                )
                plt.close(fig_comp)
                pbar.update(1)
                
                # PSTH heatmaps
                for event_label, result in psth_results.items():
                    if result['n_trials'] > 0:
                        fig_heat = visualizer.plot_psth_heatmap(
                            result,
                            save_path=str(output_path / f'psth_heatmap_{event_label.replace(" ", "_")}')
                        )
                        plt.close(fig_heat)
                pbar.update(1)
                
                # Transients
                fig_trans = visualizer.plot_transients(
                    processed_time, dff, transient_results,
                    save_path=str(output_path / 'transients')
                )
                plt.close(fig_trans)
                pbar.update(1)
                
                # Summary report figure
                fig_summary = visualizer.create_summary_report_figure(
                    psth_results, transient_results,
                    save_path=str(output_path / 'summary_report')
                )
                plt.close(fig_summary)
        
        # 9. Create analysis report
        if verbose:
            click.echo("8. Creating analysis report...")
        
        report_content = create_report(
            output_path, metadata, processing_log,
            psth_results, transient_results, config
        )
        
        if verbose:
            click.echo(f"Analysis completed successfully!")
            click.echo(f"Results saved to: {output_path}")
            click.echo(f"Report available at: {output_path / 'REPORT.md'}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()