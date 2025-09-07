# Fiber Photometry Analysis Pipeline

A robust, reproducible analysis pipeline for fiber-photometry recordings, inspired by GuPPy (Scientific Reports, 2021). This package provides comprehensive tools for parsing custom CSV exports, signal preprocessing, normalization, event-related analysis, and visualization—without manual Excel manipulation.

## Features

### 🔧 Robust Data Parsing
- **Custom CSV Format Support**: Handles semicolon-separated JSON metadata headers
- **Flexible Channel Detection**: Automatically detects 410nm (isosbestic) and 470nm (signal) channels
- **Event Marker Extraction**: Parses event timestamps and labels from metadata
- **Error-Resistant**: Graceful handling of malformed metadata with fallback parsing

### 🧮 Advanced Preprocessing
- **Zero-Phase Filtering**: Moving average and Butterworth filters without temporal shifts
- **Artifact Removal**: LED stabilization trimming and automated bad segment detection
- **Outlier Detection**: Multiple methods (Z-score, MAD, IQR) with interpolation
- **Quality Control**: Data validation, FPS verification, and missing data reporting

### 📊 Comprehensive Normalization
- **ΔF/F Calculation**: Multiple baseline methods including control channel fitting
- **Z-Score Variants**: Standard, baseline-corrected, and MAD-based z-scores
- **Bleaching Correction**: Exponential decay fitting for photobleaching
- **Robust Statistics**: Handles edge cases and invalid baseline values

### 📈 Event-Related Analysis
- **PSTHs**: Peri-stimulus time histograms with customizable time windows
- **Trial Metrics**: AUC, peak amplitude, latency, and duration analysis
- **Multiple Events**: Simultaneous analysis of different behavioral markers
- **Statistical Summaries**: Mean, SEM, and per-trial statistics

### 🔍 Transient Detection
- **Multiple Methods**: MAD-based, standard deviation, and absolute threshold detection
- **Peak Properties**: Amplitude, duration, rise/decay times, and prominence
- **Session Statistics**: Frequency analysis and amplitude distributions
- **Quality Metrics**: Width at half-maximum and area under curve

### 📊 Publication-Quality Visualization
- **Raw Signal Plots**: Multi-channel traces with control fitting
- **ΔF/F Time Series**: Z-scores with event markers and significance thresholds
- **PSTH Plots**: Individual and comparison plots with error bars
- **Heatmaps**: Trial-by-trial response visualization with sorting options
- **Summary Figures**: Multi-panel analysis reports
- **Export Formats**: PNG and SVG for publications

## Installation

### From Source
```bash
git clone https://github.com/your-repo/fiber-photometry.git
cd fiber-photometry
pip install -e .
```

### Dependencies
- Python 3.10+
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn
- Scikit-learn
- Click (CLI), PyYAML (config), tqdm (progress)

## Quick Start

### Command Line Interface

```bash
# Basic analysis with default settings
analyze-photometry -i Fluorescence.csv

# Custom parameters
analyze-photometry -i data.csv -o results \\
    --fps 100 --trim-led 1.0 \\
    --events cocaine "foot shock" lick

# Use custom configuration file
analyze-photometry -i data.csv -c my_config.yaml --verbose
```

### Python API

```python
from fiber_photometry import FluorescenceParser, Preprocessor, Normalizer

# Parse data
parser = FluorescenceParser('Fluorescence.csv')
metadata, timeseries, events = parser.parse()

# Preprocess signals
preprocessor = Preprocessor(fps=100.0)
time_vec = parser.get_time_vector()
signal_470 = parser.get_channel_data('470')
signal_410 = parser.get_channel_data('410')

processed_time, processed_signals, log = preprocessor.preprocess_pipeline(
    time_vec, np.column_stack([signal_470, signal_410]),
    {'trim_led_artifact': 1.0, 'filter_window': 0.5}
)

# Calculate ΔF/F
normalizer = Normalizer(fps=100.0)
dff, dff_info = normalizer.calculate_dff(
    processed_signals[:, 0], processed_signals[:, 1], 
    method='control_fit'
)

# Event analysis
from fiber_photometry.analysis import EventAnalyzer
analyzer = EventAnalyzer(fps=100.0)
psth_results = analyzer.analyze_all_events(
    dff, processed_time, events,
    event_labels=['cocaine', 'foot shock'],
    time_window=(-3, 5)
)
```

## Configuration

The pipeline uses YAML configuration files for reproducible analysis:

```yaml
# Data parameters
fps: 100.0
channels: ["470", "410"]

# Preprocessing
preprocessing:
  trim_led_artifact: 1.0      # seconds
  filter_window: 0.5          # moving average window
  remove_bad_segments: true   # automated artifact detection

# Normalization  
normalization:
  method: "control_fit"       # control_fit, baseline_fit, detrend
  baseline_percentile: 10     # for baseline methods

# Event analysis
event_analysis:
  psth_window: [-3.0, 5.0]    # pre/post event window
  baseline_window: [-1.0, 0.0] # baseline correction window
  events: ["cocaine", "foot shock", "lick"]

# Transient detection
transient_detection:
  method: "mad"               # mad, std, threshold
  mad_threshold: 3.5          # detection threshold
  min_peak_distance: 1.0      # seconds between peaks
```

## Output Files

The analysis generates a comprehensive set of outputs:

### Data Files
- `metadata.json` - Recording parameters and analysis settings
- `timeseries.csv` - Processed signals, ΔF/F, and z-scores
- `events.csv` - Event markers with timestamps
- `psth_*.csv` - PSTH data for each event type
- `trials_*.csv` - Individual trial responses
- `transients.csv` - Detected transient properties

### Figures (PNG + SVG)
- `raw_signals` - Multi-channel fluorescence traces
- `dff_and_zscores` - Normalized signals with event markers
- `psth_*` - Individual event response plots
- `psth_comparison` - Multi-event comparison
- `psth_heatmap_*` - Trial-by-trial heatmaps
- `transients` - Transient detection results
- `summary_report` - Multi-panel analysis overview

### Reports
- `REPORT.md` - Comprehensive markdown summary
- `processing_log.json` - Detailed processing parameters

## Advanced Usage

### Custom Event Analysis

```python
# Analyze specific events with custom parameters
psth_result = analyzer.build_psth(
    dff_signal=dff,
    time_vec=processed_time,
    events_df=events,
    event_label='cocaine',
    time_window=(-5, 10),      # Longer window
    baseline_window=(-2, 0)    # Custom baseline
)

# Extract trial metrics
for trial_metric in psth_result['trial_metrics']:
    print(f"Trial AUC: {trial_metric['auc']:.3f}")
    print(f"Peak amplitude: {trial_metric['peak_amplitude']:.3f}")
    print(f"Peak latency: {trial_metric['peak_latency']:.2f} s")
```

### Custom Visualization

```python
from fiber_photometry.visualization import Visualizer

visualizer = Visualizer(style='seaborn-v0_8', dpi=300)

# Create custom PSTH plot
fig = visualizer.plot_psth(
    psth_result,
    show_individual_trials=True,
    save_path='custom_psth'
)

# Multi-event comparison
fig = visualizer.plot_psth_comparison(
    psth_results,
    title='Behavioral Response Comparison'
)
```

### Transient Analysis

```python
from fiber_photometry.analysis import TransientAnalyzer

transient_analyzer = TransientAnalyzer(fps=100.0)
transients = transient_analyzer.detect_transients(
    dff_signal=dff,
    time_vec=processed_time,
    method='mad',
    threshold=4.0,
    min_peak_distance=2.0
)

print(f"Detected {transients['n_transients']} transients")
print(f"Frequency: {transients['frequency_per_hour']:.1f} events/hour")
```

## Data Format Requirements

The pipeline expects CSV files with:

1. **Header Line**: JSON-like metadata with semicolon separators:
```
{"Light":{"Led410Enable":true;"Led470Enable":true;"Led410Value":25.0;"Led470Value":60.0};"Fps":100.0;"AllMarking":[{"name":"cocaine";"type":1}]}
```

2. **Data Columns**: TimeStamp, Events, CH1-410, CH1-470
```
TimeStamp,Events,CH1-410,CH1-470,
0.000,,18.324,28.240,
20.004,,18.320,28.234,
```

3. **Event Markers**: Either in Events column or metadata AllMarking field

## Troubleshooting

### Common Issues

**"No signal channels found"**
- Check that column names contain wavelength numbers (410, 470, etc.)
- Verify CSV format and delimiter consistency

**"FPS mismatch warning"**
- Metadata FPS doesn't match timestamp intervals
- Use `--fps` parameter to override if needed

**"No events found for label"**
- Check event names match between config and data
- Verify Events column contains timestamps, not just metadata

**"Invalid baseline values"**
- Control channel contains zeros or negative values
- Consider using baseline_fit method instead of control_fit

### Debug Mode

```bash
# Run with verbose output for debugging
analyze-photometry -i data.csv --verbose

# Check intermediate outputs
ls output/
cat output/processing_log.json
```

## Testing

Run the test suite to verify installation:

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_parser.py -v

# Run with coverage
pytest --cov=fiber_photometry tests/
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{fiber_photometry_pipeline,
  title={Fiber Photometry Analysis Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/fiber-photometry}
}
```

And the original GuPPy paper that inspired this work:
> Sych, Y., Chernysheva, M., Sumanovski, L.T. et al. High-density multi-fiber photometry for studying large-scale brain circuit dynamics. Nat Methods 16, 553–560 (2019).

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, bug reports, or feature requests:
- 📧 Email: [your.email@domain.com]
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/fiber-photometry/issues)
- 📖 Documentation: [ReadTheDocs](https://fiber-photometry.readthedocs.io/)