# 🧠 FLAIR - Fiber-photometry Live Analysis and Intuitive Results

**F**iber-photometry **L**ive **A**nalysis and **I**ntuitive **R**esults

*Product Owner: Elsa Karam*  
*Developer: Chadi Abi Fadel*

---

## Overview

FLAIR is a user-friendly web interface for fiber photometry analysis, built specifically for neuroscientists who need powerful analysis capabilities without command-line complexity. It provides an intuitive GUI on top of the robust fiber photometry analysis pipeline.

## Features

### 🚀 **One-Click Analysis**
- Upload your CSV file and run complete analysis with a single click
- Real-time progress tracking with detailed status updates
- No command-line knowledge required

### 🔧 **Interactive Configuration**
- **Data Parameters**: Sampling rate, channel selection
- **Preprocessing**: LED artifact trimming, smoothing, outlier detection
- **Normalization**: Multiple ΔF/F calculation methods
- **Event Analysis**: Customizable PSTH windows and baseline correction
- **Transient Detection**: Automated peak detection with adjustable thresholds

### 📊 **Rich Visualizations**
- **Raw & Processed Signals**: Multi-channel fluorescence traces
- **ΔF/F Time Series**: Z-scores with event markers
- **PSTH Analysis**: Individual and comparison plots with error bars
- **Trial Heatmaps**: Visual trial-by-trial response patterns
- **Transient Detection**: Automated peak identification and statistics

### 📈 **Interactive Plots**
- Zoom, pan, and explore your data interactively
- Hover for detailed information
- Export plots in multiple formats

### 💾 **Comprehensive Results**
- Download processed timeseries data
- Export event analysis results
- Save configuration files for reproducible analysis
- Generate analysis summary reports

## Quick Start

### Installation

1. **Install dependencies:**
   ```bash
   pip install streamlit plotly
   # Or install the full package:
   pip install -e .
   ```

2. **Run FLAIR:**
   ```bash
   streamlit run FLAIR.py
   ```
   
   Or if installed:
   ```bash
   flair
   ```

3. **Open in browser:**
   - FLAIR will automatically open in your default browser
   - If not, navigate to `http://localhost:8501`

### Usage

1. **Configure Analysis Parameters**
   - Use the sidebar to adjust preprocessing and analysis settings
   - All parameters have helpful tooltips explaining their purpose

2. **Upload Data**
   - Click "Choose your Fluorescence CSV file"
   - Upload the CSV file from your fiber photometry system

3. **Run Analysis**
   - Click "🚀 Start Analysis"
   - Monitor progress in real-time
   - Analysis typically takes 30 seconds to 2 minutes

4. **Explore Results**
   - **Raw & Processed Signals**: View your fluorescence data at different processing stages
   - **Event Analysis**: Examine PSTH responses for different behavioral events
   - **Transient Detection**: See automatically detected calcium transients
   - **Data Tables**: Browse numerical results in tabular format
   - **Download Results**: Export data and reports

## Data Format

FLAIR expects CSV files in the same format as the command-line tool:

- **Header**: JSON-like metadata with semicolon separators
- **Columns**: TimeStamp, Events, CH1-410, CH1-470
- **Events**: Either in Events column or metadata AllMarking field

Example header:
```
{"Light":{"Led410Enable":true;"Led470Enable":true;"Led410Value":25.0;"Led470Value":60.0};"Fps":100.0;"AllMarking":[{"name":"cocaine";"type":1}]}
```

## Advantages Over CLI

- **User-Friendly**: No command-line knowledge required
- **Interactive**: Real-time parameter adjustment and visualization
- **Visual**: Rich plots and interactive exploration
- **Accessible**: Web-based interface works on any device
- **Guided**: Helpful tooltips and clear workflow
- **Integrated**: Complete analysis workflow in one interface

## Technical Architecture

FLAIR is built using:
- **Streamlit**: Modern web app framework for data science
- **Plotly**: Interactive visualization library
- **Original Pipeline**: All the robust analysis capabilities of the CLI tool

The GUI maintains full compatibility with the original fiber photometry analysis pipeline while providing an intuitive interface layer.

## Configuration Options

### Data Parameters
- **Sampling Rate**: Override the FPS from metadata if needed
- **Channels**: Specify which wavelengths to analyze

### Preprocessing
- **LED Artifact Trim**: Remove initial LED stabilization period
- **Smoothing Window**: Apply moving average filter
- **Outlier Detection**: Remove artifacts using MAD, Z-score, or IQR methods

### Normalization
- **Method**: Choose between control fitting, baseline fitting, or detrending
- **Baseline Percentile**: For baseline-based methods

### Event Analysis
- **PSTH Window**: Pre/post event time windows
- **Baseline Window**: Time window for baseline correction
- **Events**: List of behavioral events to analyze

### Transient Detection
- **Method**: MAD, standard deviation, or absolute threshold
- **Detection Threshold**: Sensitivity of peak detection
- **Peak Constraints**: Minimum distance and width requirements

## Output Files

When you download results from FLAIR, you get:

### Data Files
- `flair_timeseries.csv` - Complete processed timeseries
- `flair_events.csv` - Event markers and timestamps
- `flair_transients.csv` - Detected transient properties

### Configuration & Reports
- `flair_config.yaml` - Analysis parameters for reproducibility
- `flair_analysis_summary.json` - Comprehensive analysis report

## Troubleshooting

### Common Issues

**"No 470nm signal channel found"**
- Check that your CSV has columns with wavelength numbers
- Verify the file format matches the expected structure

**Analysis fails during processing**
- Try adjusting preprocessing parameters
- Check that events are properly formatted in your data

**Slow performance**
- Large files (>1M samples) may take longer to process
- Consider trimming your data if it's excessively long

### Getting Help

1. **Tooltips**: Hover over any parameter for explanation
2. **Error Messages**: FLAIR provides detailed error information
3. **Default Settings**: Start with default parameters if unsure

## Development

FLAIR is designed to be:
- **Maintainable**: Clean, well-documented code
- **Extensible**: Easy to add new features
- **Compatible**: Full compatibility with existing pipeline

### Adding New Features

The modular design makes it easy to add:
- New visualization types
- Additional analysis methods
- Export formats
- Configuration options

---

**Built for neuroscientists, by developers who care about user experience.**

*For questions or support, please contact the development team.*