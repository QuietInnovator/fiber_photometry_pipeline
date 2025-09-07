#!/usr/bin/env python3
"""
Fiber Photometry Analysis Script

Example script demonstrating usage of the fiber-photometry analysis pipeline.
This script provides a simplified interface for analyzing your Fluorescence.csv file.

Usage:
    python analyze.py --input Fluorescence.csv --output results
    python analyze.py --help
"""

import sys
import argparse
from pathlib import Path

# Add the current directory to Python path to import our modules
sys.path.insert(0, str(Path(__file__).parent))

from fiber_photometry.cli import main as cli_main

def create_example_config():
    """Create an example configuration file."""
    example_config = """# Fiber Photometry Analysis Configuration

# Input/Output settings
input_file: "Fluorescence.csv"
output_dir: "output"

# Data acquisition parameters
fps: 100.0  # Frames per second (will be validated against metadata)
channels:
  - "470"  # Signal channel (GCaMP, etc.)
  - "410"  # Isosbestic control channel

# Preprocessing parameters
preprocessing:
  trim_led_artifact: 1.0      # Seconds to trim from start (LED stabilization)
  filter_window: 0.5          # Moving average window in seconds (0 = disable)
  butterworth_order: 4        # Butterworth filter order (if used)
  remove_bad_segments: false  # Manual bad segment removal (interactive)
  remove_outliers: true       # Remove outliers
  outlier_method: "mad"       # mad, zscore, or iqr
  outlier_threshold: 4.0      # Outlier detection threshold

# Normalization parameters
normalization:
  method: "control_fit"       # "control_fit", "baseline_fit", or "detrend"
  baseline_percentile: 10     # For baseline fitting methods
  
# Event analysis parameters
event_analysis:
  psth_window: [-3.0, 5.0]     # Pre/post event window in seconds
  baseline_window: [-1.0, 0.0] # Baseline window for correction
  events:
    - "cocaine"
    - "foot shock" 
    - "lick"
    - "start"

# Transient detection parameters
transient_detection:
  method: "mad"                # "mad", "std", or "threshold"
  mad_threshold: 3.5           # MAD-based detection threshold
  min_peak_distance: 1.0       # Minimum distance between peaks (seconds)
  min_peak_width: 0.1          # Minimum peak width (seconds)
  
# Visualization parameters
visualization:
  figure_format: ["png", "svg"]  # Output formats
  dpi: 300                       # Figure resolution
  style: "seaborn-v0_8"         # Matplotlib style
  
# Quality control parameters
quality_control:
  validate_fps: true      # Validate FPS against timestamp intervals
  check_data_gaps: true   # Check for missing data points
  report_outliers: true   # Report potential outlier detection
"""
    
    with open('example_config.yaml', 'w') as f:
        f.write(example_config)
    
    print("Created example_config.yaml - customize this file for your analysis!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze fiber-photometry data with comprehensive preprocessing and analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic analysis with your Fluorescence.csv file
  python analyze.py --input Fluorescence.csv
  
  # Custom output directory and parameters
  python analyze.py --input Fluorescence.csv --output my_results --fps 100 --trim-led 2.0
  
  # Use custom configuration file
  python analyze.py --input Fluorescence.csv --config my_config.yaml --verbose
  
  # Analyze specific events
  python analyze.py --input Fluorescence.csv --events cocaine "foot shock" lick
  
  # Create example configuration file
  python analyze.py --create-config
        '''
    )
    
    parser.add_argument('--input', '-i', help='Path to input CSV file (e.g., Fluorescence.csv)')
    parser.add_argument('--output', '-o', default='output', help='Output directory (default: output)')
    parser.add_argument('--config', '-c', default='config.yaml', help='Configuration file (default: config.yaml)')
    parser.add_argument('--fps', type=float, help='Override sampling rate (Hz)')
    parser.add_argument('--trim-led', type=float, help='LED artifact trim time (seconds)')
    parser.add_argument('--filter-window', type=float, help='Smoothing window (seconds)')
    parser.add_argument('--psth-window', nargs=2, type=float, metavar=('PRE', 'POST'), 
                       help='PSTH window (pre post) in seconds')
    parser.add_argument('--baseline-window', nargs=2, type=float, metavar=('PRE', 'POST'),
                       help='Baseline window (pre post) in seconds')
    parser.add_argument('--channels', nargs='+', help='Channel wavelengths to analyze (e.g., 470 410)')
    parser.add_argument('--events', nargs='+', help='Event types to analyze (e.g., cocaine "foot shock" lick)')
    parser.add_argument('--no-figures', action='store_true', help='Skip figure generation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--create-config', action='store_true', help='Create example configuration file and exit')
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.create_config:
        create_example_config()
        sys.exit(0)
    
    if not args.input:
        print("Error: Input file is required (use --input or -i)")
        print("Run 'python analyze.py --help' for usage information")
        print("Run 'python analyze.py --create-config' to create example configuration")
        sys.exit(1)
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    # Build command line arguments for the CLI
    cli_args = [
        '--input', args.input,
        '--output', args.output,
        '--config', args.config,
    ]
    
    if args.fps:
        cli_args.extend(['--fps', str(args.fps)])
    if args.trim_led:
        cli_args.extend(['--trim-led', str(args.trim_led)])
    if args.filter_window:
        cli_args.extend(['--filter-window', str(args.filter_window)])
    if args.psth_window:
        cli_args.extend(['--psth-window'] + [str(x) for x in args.psth_window])
    if args.baseline_window:
        cli_args.extend(['--baseline-window'] + [str(x) for x in args.baseline_window])
    if args.channels:
        for channel in args.channels:
            cli_args.extend(['--channels', channel])
    if args.events:
        for event in args.events:
            cli_args.extend(['--events', event])
    if args.no_figures:
        cli_args.append('--no-figures')
    if args.verbose:
        cli_args.append('--verbose')
    
    # Print welcome message
    if args.verbose:
        print("=" * 60)
        print("   FIBER PHOTOMETRY ANALYSIS PIPELINE")
        print("=" * 60)
        print(f"Input file: {args.input}")
        print(f"Output directory: {args.output}")
        print(f"Config file: {args.config}")
        print("=" * 60)
    
    # Call the main CLI function with our arguments
    import sys
    sys.argv = ['analyze-photometry'] + cli_args
    
    try:
        cli_main()
        if args.verbose:
            print("=" * 60)
            print("   ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Results saved to: {args.output}/")
            print(f"View report: {args.output}/REPORT.md")
            print("=" * 60)
    except SystemExit as e:
        if e.code != 0:
            print("Analysis failed. Run with --verbose for more details.")
            sys.exit(e.code)