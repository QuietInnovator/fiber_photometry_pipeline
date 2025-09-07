# Fiber Photometry Analysis Report

**Analysis Date:** 2025-09-07 14:13:30
**Input File:** Fluorescence.csv
**Output Directory:** output

## Data Summary

- **Recording Duration:** 649.39 seconds
- **Sample Count:** 32471
- **Sampling Rate:** 100.0 Hz
- **Channels:** CH1-410, CH1-470
- **Dual Channel:** Yes

## Preprocessing

- **LED Artifact Trimmed:** 1.0 seconds
- **Smoothing Window:** 0.5 seconds

## Event-Related Analysis

| Event Type | N Trials | Mean AUC | Peak Amplitude | Peak Latency (s) |
|------------|----------|----------|----------------|------------------|

## Transient Analysis

- **Total Transients:** 19
- **Frequency:** 105.33 events/hour
- **Mean Amplitude:** 0.023 ± 0.005 ΔF/F
- **Mean Duration:** 2.63 ± 1.81 seconds
- **Amplitude Range:** 0.018 - 0.041 ΔF/F
- **Duration Range:** 0.78 - 6.25 seconds

## Analysis Parameters

### Preprocessing
- Trim LED artifact: 1.0 s
- Filter window: 0.5 s
- Remove bad segments: False

### Normalization
- Method: control_fit
- Baseline percentile: 10%

### Event Analysis
- PSTH window: [-3.0, 5.0] s
- Baseline window: [-1.0, 0.0] s
- Events analyzed: cocaine, foot shock, lick, start

### Transient Detection
- Method: mad
- Threshold: 3.5
- Min peak distance: 1.0 s
- Min peak width: 0.1 s

## Files Generated

- `metadata.json` - Recording metadata and parameters
- `timeseries.csv` - Processed time-series data
- `events.csv` - Event markers and timestamps
- `psth_*.csv` - PSTH data for each event type
- `trials_*.csv` - Individual trial data for each event type
- `transients.csv` - Detected transients with properties
- `processing_log.json` - Detailed processing log
- Figures: Raw signals, ΔF/F, PSTHs, heatmaps, transients

---

*Generated with fiber-photometry analysis pipeline*