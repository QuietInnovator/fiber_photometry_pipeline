#!/usr/bin/env python3
"""
FLAIR - Fiber-photometry Live Analysis and Intuitive Results
A user-friendly Streamlit GUI for fiber photometry analysis

Product Owner: Elsa Karam
Developer: Chadi Abi Fadel

This application provides an intuitive web interface for neuroscientists
to analyze fiber photometry data without command-line complexity.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import json
import io
import zipfile
from pathlib import Path
import tempfile
import os
import sys
from typing import Dict, Any, Optional, List

# Add the current directory to Python path to import our modules
sys.path.insert(0, str(Path(__file__).parent))

from fiber_photometry.parser import FluorescenceParser
from fiber_photometry.preprocessing import Preprocessor
from fiber_photometry.normalization import Normalizer
from fiber_photometry.analysis import EventAnalyzer, TransientAnalyzer
from fiber_photometry.visualization import Visualizer

# Set page configuration
st.set_page_config(
    page_title="FLAIR - Fiber-photometry Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #e0e0e0;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'config' not in st.session_state:
        st.session_state.config = load_default_config()

def load_default_config() -> Dict[str, Any]:
    """Load default configuration"""
    return {
        'fps': 100.0,
        'channels': ['470', '410'],
        'preprocessing': {
            'trim_led_artifact': 1.0,
            'filter_window': 0.5,
            'butterworth_order': 4,
            'remove_bad_segments': False,
            'remove_outliers': True,
            'outlier_method': 'mad',
            'outlier_threshold': 4.0
        },
        'normalization': {
            'method': 'control_fit',
            'baseline_percentile': 10
        },
        'event_analysis': {
            'psth_window': [-3.0, 5.0],
            'baseline_window': [-1.0, 0.0],
            'events': ['cocaine', 'foot shock', 'lick', 'start']
        },
        'transient_detection': {
            'method': 'mad',
            'mad_threshold': 3.5,
            'min_peak_distance': 1.0,
            'min_peak_width': 0.1
        },
        'visualization': {
            'figure_format': ['png', 'svg'],
            'dpi': 300,
            'style': 'seaborn-v0_8'
        }
    }

def render_header():
    """Render the main header"""
    st.markdown('<h1 class="main-header">🧠 FLAIR</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">'
        '<strong>F</strong>iber-photometry <strong>L</strong>ive <strong>A</strong>nalysis and <strong>I</strong>ntuitive <strong>R</strong>esults</p>',
        unsafe_allow_html=True
    )

def render_sidebar():
    """Render the sidebar with configuration options"""
    st.sidebar.markdown("## 🔧 Analysis Configuration")
    
    # Data Parameters
    st.sidebar.markdown("### 📊 Data Parameters")
    st.session_state.config['fps'] = st.sidebar.number_input(
        "Sampling Rate (Hz)", 
        min_value=1.0, 
        max_value=1000.0, 
        value=st.session_state.config['fps'],
        help="Frames per second of the recording"
    )
    
    # Preprocessing Parameters
    st.sidebar.markdown("### 🔧 Preprocessing")
    st.session_state.config['preprocessing']['trim_led_artifact'] = st.sidebar.number_input(
        "LED Artifact Trim (s)", 
        min_value=0.0, 
        max_value=10.0, 
        value=st.session_state.config['preprocessing']['trim_led_artifact'],
        help="Seconds to trim from start for LED stabilization"
    )
    
    st.session_state.config['preprocessing']['filter_window'] = st.sidebar.number_input(
        "Smoothing Window (s)", 
        min_value=0.0, 
        max_value=5.0, 
        value=st.session_state.config['preprocessing']['filter_window'],
        help="Moving average window size (0 to disable)"
    )
    
    st.session_state.config['preprocessing']['outlier_method'] = st.sidebar.selectbox(
        "Outlier Detection Method",
        options=['mad', 'zscore', 'iqr'],
        index=['mad', 'zscore', 'iqr'].index(st.session_state.config['preprocessing']['outlier_method']),
        help="Method for detecting outliers"
    )
    
    st.session_state.config['preprocessing']['outlier_threshold'] = st.sidebar.number_input(
        "Outlier Threshold", 
        min_value=1.0, 
        max_value=10.0, 
        value=st.session_state.config['preprocessing']['outlier_threshold'],
        help="Threshold for outlier detection"
    )
    
    # Normalization Parameters
    st.sidebar.markdown("### 📐 Normalization")
    st.session_state.config['normalization']['method'] = st.sidebar.selectbox(
        "Normalization Method",
        options=['control_fit', 'baseline_fit', 'detrend'],
        index=['control_fit', 'baseline_fit', 'detrend'].index(st.session_state.config['normalization']['method']),
        help="Method for calculating ΔF/F"
    )
    
    if st.session_state.config['normalization']['method'] in ['baseline_fit']:
        st.session_state.config['normalization']['baseline_percentile'] = st.sidebar.slider(
            "Baseline Percentile (%)", 
            min_value=1, 
            max_value=50, 
            value=st.session_state.config['normalization']['baseline_percentile'],
            help="Percentile for baseline calculation"
        )
    
    # Event Analysis Parameters
    st.sidebar.markdown("### 🎯 Event Analysis")
    psth_pre = st.sidebar.number_input("PSTH Pre-event (s)", value=3.0, min_value=0.1, max_value=30.0)
    psth_post = st.sidebar.number_input("PSTH Post-event (s)", value=5.0, min_value=0.1, max_value=30.0)
    st.session_state.config['event_analysis']['psth_window'] = [-psth_pre, psth_post]
    
    baseline_pre = st.sidebar.number_input("Baseline Pre-event (s)", value=1.0, min_value=0.1, max_value=10.0)
    baseline_post = st.sidebar.number_input("Baseline Post-event (s)", value=0.0, min_value=0.0, max_value=1.0)
    st.session_state.config['event_analysis']['baseline_window'] = [-baseline_pre, baseline_post]
    
    # Events to analyze
    events_text = st.sidebar.text_area(
        "Events to Analyze (one per line)",
        value="\n".join(st.session_state.config['event_analysis']['events']),
        help="Enter event labels, one per line"
    )
    st.session_state.config['event_analysis']['events'] = [e.strip() for e in events_text.split('\n') if e.strip()]
    
    # Transient Detection Parameters
    st.sidebar.markdown("### 🔍 Transient Detection")
    st.session_state.config['transient_detection']['method'] = st.sidebar.selectbox(
        "Detection Method",
        options=['mad', 'std', 'threshold'],
        index=['mad', 'std', 'threshold'].index(st.session_state.config['transient_detection']['method']),
        help="Method for detecting transients"
    )
    
    st.session_state.config['transient_detection']['mad_threshold'] = st.sidebar.number_input(
        "Detection Threshold", 
        min_value=1.0, 
        max_value=10.0, 
        value=st.session_state.config['transient_detection']['mad_threshold'],
        help="Threshold for transient detection"
    )
    
    st.session_state.config['transient_detection']['min_peak_distance'] = st.sidebar.number_input(
        "Min Peak Distance (s)", 
        min_value=0.1, 
        max_value=10.0, 
        value=st.session_state.config['transient_detection']['min_peak_distance'],
        help="Minimum time between detected peaks"
    )

def upload_file():
    """Handle file upload"""
    st.markdown('<div class="section-header">📁 Data Upload</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your Fluorescence CSV file",
        type=['csv'],
        help="Upload the CSV file exported from your fiber photometry system"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.info(f"File size: {uploaded_file.size:,} bytes")
        
        return uploaded_file
    else:
        st.info("👆 Please upload a CSV file to begin analysis")
        return None

def run_analysis(uploaded_file):
    """Run the complete analysis pipeline"""
    
    if st.button("🚀 Start Analysis", type="primary"):
        
        with st.spinner("🔄 Running fiber photometry analysis..."):
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Parse data
                status_text.text("📖 Parsing CSV data...")
                progress_bar.progress(10)
                
                parser = FluorescenceParser(tmp_file_path)
                metadata, timeseries_df, events_df = parser.parse()
                
                # Step 2: Preprocessing
                status_text.text("🔧 Preprocessing signals...")
                progress_bar.progress(25)
                
                fps_rate = st.session_state.config['fps']
                preprocessor = Preprocessor(fps_rate)
                
                time_vec = parser.get_time_vector()
                signal_470 = parser.get_channel_data('470')
                signal_410 = parser.get_channel_data('410')
                
                if signal_470 is None:
                    st.error("❌ No 470nm signal channel found in the data")
                    return
                
                # Stack signals for processing
                if signal_410 is not None:
                    signals = np.column_stack([signal_470, signal_410])
                else:
                    signals = signal_470
                
                processed_time, processed_signals, processing_log = preprocessor.preprocess_pipeline(
                    time_vec, signals, st.session_state.config['preprocessing']
                )
                
                if processed_signals.ndim == 1:
                    processed_470 = processed_signals
                    processed_410 = None
                else:
                    processed_470 = processed_signals[:, 0]
                    processed_410 = processed_signals[:, 1] if processed_signals.shape[1] > 1 else None
                
                # Step 3: Normalization
                status_text.text("📐 Calculating ΔF/F and z-scores...")
                progress_bar.progress(50)
                
                normalizer = Normalizer(fps_rate)
                
                # Calculate ΔF/F
                if processed_410 is not None:
                    dff, dff_info = normalizer.calculate_dff(
                        processed_470, processed_410, 
                        baseline_method=st.session_state.config['normalization']['method']
                    )
                else:
                    dff, dff_info = normalizer.calculate_dff(
                        processed_470, baseline_method='baseline_fit',
                        percentile=st.session_state.config['normalization'].get('baseline_percentile', 10)
                    )
                
                # Calculate z-scores
                zscore_std, _ = normalizer.calculate_zscore(dff, method='standard')
                zscore_baseline, _ = normalizer.calculate_zscore(
                    dff, method='baseline', 
                    baseline_window=tuple(st.session_state.config['event_analysis']['baseline_window']),
                    time_vec=processed_time
                )
                zscore_modified, _ = normalizer.calculate_zscore(dff, method='modified')
                
                # Step 4: Event analysis
                status_text.text("🎯 Analyzing events...")
                progress_bar.progress(70)
                
                event_analyzer = EventAnalyzer(fps_rate)
                
                psth_results = event_analyzer.analyze_all_events(
                    dff, processed_time, events_df,
                    event_labels=st.session_state.config['event_analysis']['events'],
                    time_window=tuple(st.session_state.config['event_analysis']['psth_window']),
                    baseline_window=tuple(st.session_state.config['event_analysis']['baseline_window'])
                )
                
                # Step 5: Transient detection
                status_text.text("🔍 Detecting transients...")
                progress_bar.progress(85)
                
                transient_analyzer = TransientAnalyzer(fps_rate)
                transient_results = transient_analyzer.detect_transients(
                    dff, processed_time,
                    method=st.session_state.config['transient_detection']['method'],
                    threshold=st.session_state.config['transient_detection']['mad_threshold'],
                    min_peak_distance=st.session_state.config['transient_detection']['min_peak_distance'],
                    min_peak_width=st.session_state.config['transient_detection']['min_peak_width']
                )
                
                # Store results
                status_text.text("💾 Finalizing results...")
                progress_bar.progress(100)
                
                st.session_state.results = {
                    'metadata': metadata,
                    'processed_time': processed_time,
                    'processed_470': processed_470,
                    'processed_410': processed_410,
                    'dff': dff,
                    'zscore_std': zscore_std,
                    'zscore_baseline': zscore_baseline,
                    'zscore_modified': zscore_modified,
                    'events_df': events_df,
                    'psth_results': psth_results,
                    'transient_results': transient_results,
                    'processing_log': processing_log,
                    'dff_info': dff_info
                }
                
                st.session_state.analysis_complete = True
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
                progress_bar.empty()
                status_text.empty()
                
                st.success("✅ Analysis completed successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)

def display_results():
    """Display analysis results"""
    if not st.session_state.analysis_complete:
        return
    
    results = st.session_state.results
    
    st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        duration = results['processing_log'].get('final_duration', 0)
        st.metric("Recording Duration", f"{duration:.1f} s")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        samples = results['processing_log'].get('final_samples', 0)
        st.metric("Sample Count", f"{samples:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        fps = st.session_state.config['fps']
        st.metric("Sampling Rate", f"{fps:.1f} Hz")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        n_transients = results['transient_results']['n_transients']
        st.metric("Transients Detected", f"{n_transients}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add debug info
    with st.expander("🔧 Debug Information", expanded=False):
        st.write("**Results structure:**")
        st.write(f"- Processed time points: {len(results.get('processed_time', []))}")
        st.write(f"- DFF values: {len(results.get('dff', []))}")
        st.write(f"- Events found: {len(results.get('events_df', pd.DataFrame()))}")
        st.write(f"- PSTH results: {list(results.get('psth_results', {}).keys())}")
        st.write(f"- Transients detected: {results.get('transient_results', {}).get('n_transients', 0)}")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Raw & Processed Signals", 
        "🎯 Event Analysis", 
        "🔍 Transient Detection", 
        "📋 Data Tables", 
        "💾 Download Results"
    ])
    
    with tab1:
        display_signals_tab(results)
    
    with tab2:
        display_events_tab(results)
    
    with tab3:
        display_transients_tab(results)
    
    with tab4:
        display_data_tables_tab(results)
    
    with tab5:
        display_download_tab(results)

def display_signals_tab(results):
    """Display raw and processed signals"""
    st.markdown("### Raw and Processed Signals")
    
    try:
        time_vec = results['processed_time']
        
        # Check if we have data
        if len(time_vec) == 0:
            st.error("No processed time data available")
            return
        
        st.info(f"Displaying {len(time_vec):,} data points over {time_vec[-1]:.1f} seconds")
        
        # Raw Signals Plot
        st.markdown("#### Raw Fluorescence Signals")
        fig_raw = go.Figure()
        
        fig_raw.add_trace(
            go.Scatter(
                x=time_vec, 
                y=results['processed_470'], 
                name="470nm (Signal)", 
                line=dict(color="#1f77b4", width=1)
            )
        )
        
        if results['processed_410'] is not None:
            fig_raw.add_trace(
                go.Scatter(
                    x=time_vec, 
                    y=results['processed_410'], 
                    name="410nm (Control)", 
                    line=dict(color="#ff7f0e", width=1)
                )
            )
        
        fig_raw.update_layout(
            title="Raw Fluorescence Signals",
            xaxis_title="Time (s)",
            yaxis_title="Fluorescence",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_raw, use_container_width=True)
        
        # ΔF/F Plot
        st.markdown("#### ΔF/F Signal")
        fig_dff = go.Figure()
        
        fig_dff.add_trace(
            go.Scatter(
                x=time_vec, 
                y=results['dff'], 
                name="ΔF/F", 
                line=dict(color="#2ca02c", width=1)
            )
        )
        
        # Add event markers if available
        if not results['events_df'].empty:
            for _, event in results['events_df'].iterrows():
                event_time = event['timestamp']
                event_label = event.get('label', 'Event')
                fig_dff.add_vline(
                    x=event_time, 
                    line_dash="dash", 
                    line_color="red", 
                    opacity=0.7,
                    annotation_text=event_label,
                    annotation_position="top"
                )
        
        fig_dff.update_layout(
            title="ΔF/F Signal with Event Markers",
            xaxis_title="Time (s)",
            yaxis_title="ΔF/F",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_dff, use_container_width=True)
        
        # Z-score Plot
        st.markdown("#### Z-Score Signals")
        fig_z = go.Figure()
        
        fig_z.add_trace(
            go.Scatter(
                x=time_vec, 
                y=results['zscore_std'], 
                name="Z-score (Standard)", 
                line=dict(color="#d62728", width=1)
            )
        )
        
        fig_z.add_trace(
            go.Scatter(
                x=time_vec, 
                y=results['zscore_baseline'], 
                name="Z-score (Baseline)", 
                line=dict(color="#9467bd", width=1)
            )
        )
        
        fig_z.update_layout(
            title="Z-Score Normalized Signals",
            xaxis_title="Time (s)",
            yaxis_title="Z-Score",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_z, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying signals: {str(e)}")
        st.exception(e)

def display_events_tab(results):
    """Display event analysis results"""
    st.markdown("### Event-Related Analysis (PSTH)")
    
    try:
        psth_results = results['psth_results']
        
        if not psth_results:
            st.info("No event analysis results available.")
            return
        
        # Show events available
        st.info(f"Events analyzed: {', '.join(psth_results.keys())}")
        
        # Event selection
        event_options = [event for event, result in psth_results.items() if result['n_trials'] > 0]
        
        if not event_options:
            st.warning("No events found with trials for analysis.")
            return
        
        selected_event = st.selectbox("Select Event Type", event_options)
        
        result = psth_results[selected_event]
        
        # Display event statistics
        col1, col2, col3 = st.columns(3)
        
        stats = result['summary_stats']
        
        with col1:
            st.metric("Number of Trials", result['n_trials'])
            st.metric("Mean AUC", f"{stats['mean_auc']:.3f}")
        
        with col2:
            st.metric("Peak Amplitude", f"{stats['mean_peak_amplitude']:.3f} ± {stats['sem_peak_amplitude']:.3f}")
            st.metric("Peak Latency", f"{stats['mean_peak_latency']:.2f} ± {stats['sem_peak_latency']:.2f} s")
        
        with col3:
            st.metric("AUC Range", f"{stats['auc_range'][0]:.3f} - {stats['auc_range'][1]:.3f}")
            st.metric("Amplitude Range", f"{stats['amplitude_range'][0]:.3f} - {stats['amplitude_range'][1]:.3f}")
        
        # PSTH plot
        st.markdown("#### PSTH Plot")
        fig = go.Figure()
        
        psth_time = result['psth_time']
        psth_mean = result['psth_mean']
        psth_sem = result['psth_sem']
        
        # Mean trace with error ribbon
        fig.add_trace(
            go.Scatter(
                x=psth_time,
                y=psth_mean + psth_sem,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=psth_time,
                y=psth_mean - psth_sem,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.2)',
                showlegend=False,
                hoverinfo='skip'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=psth_time,
                y=psth_mean,
                mode='lines',
                name=f'{selected_event} (Mean ± SEM)',
                line=dict(color='#1f77b4', width=2)
            )
        )
        
        # Add event marker at time 0
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Event Onset")
        
        fig.update_layout(
            title=f"PSTH for {selected_event} ({result['n_trials']} trials)",
            xaxis_title="Time from event (s)",
            yaxis_title="ΔF/F",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap of individual trials
        if result['n_trials'] > 1:
            st.markdown("#### Trial-by-Trial Heatmap")
            
            try:
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=result['trials'],
                    x=psth_time,
                    y=[f"Trial {i+1}" for i in range(result['n_trials'])],
                    colorscale='RdBu_r',
                    zmid=0
                ))
                
                fig_heatmap.update_layout(
                    title=f"Individual Trials Heatmap - {selected_event}",
                    xaxis_title="Time from event (s)",
                    yaxis_title="Trial Number",
                    height=min(400, result['n_trials'] * 20 + 100)
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating heatmap: {str(e)}")
        
    except Exception as e:
        st.error(f"Error displaying events: {str(e)}")
        st.exception(e)

def display_transients_tab(results):
    """Display transient detection results"""
    st.markdown("### Transient Detection")
    
    try:
        transient_results = results['transient_results']
        
        if transient_results['n_transients'] == 0:
            st.info("No transients detected in the signal.")
            st.info("Try adjusting the detection threshold in the sidebar to detect more transients.")
            return
        
        # Transient statistics
        col1, col2, col3 = st.columns(3)
        
        stats = transient_results['summary_stats']
        
        with col1:
            st.metric("Total Transients", transient_results['n_transients'])
            st.metric("Frequency", f"{transient_results['frequency_per_hour']:.1f} events/hour")
        
        with col2:
            st.metric("Mean Amplitude", f"{stats['mean_amplitude']:.3f} ± {stats['std_amplitude']:.3f}")
            st.metric("Mean Duration", f"{stats['mean_duration']:.2f} ± {stats['std_duration']:.2f} s")
        
        with col3:
            st.metric("Amplitude Range", f"{stats['amplitude_range'][0]:.3f} - {stats['amplitude_range'][1]:.3f}")
            st.metric("Duration Range", f"{stats['duration_range'][0]:.2f} - {stats['duration_range'][1]:.2f} s")
        
        # Transient detection plot
        st.markdown("#### ΔF/F Signal with Detected Transients")
        
        time_vec = results['processed_time']
        dff = results['dff']
        
        fig = go.Figure()
        
        # ΔF/F signal
        fig.add_trace(
            go.Scatter(
                x=time_vec,
                y=dff,
                mode='lines',
                name='ΔF/F Signal',
                line=dict(color='#1f77b4', width=1)
            )
        )
        
        # Detected transients
        transients = transient_results['transients']
        if len(transients) > 0:
            transients_df = pd.DataFrame(transients)
            
            fig.add_trace(
                go.Scatter(
                    x=transients_df['peak_time'],
                    y=transients_df['amplitude'],
                    mode='markers',
                    name='Detected Transients',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='diamond'
                    ),
                    hovertemplate='<b>Transient</b><br>' +
                                 'Time: %{x:.2f} s<br>' +
                                 'Amplitude: %{y:.3f}<br>' +
                                 '<extra></extra>'
                )
            )
        
        fig.update_layout(
            title=f"Transient Detection Results ({transient_results['n_transients']} detected)",
            xaxis_title="Time (s)",
            yaxis_title="ΔF/F",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Transient properties histograms
        if len(transients) > 5:  # Only show histograms if we have enough data
            st.markdown("#### Transient Properties Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_amp = px.histogram(
                    transients_df, 
                    x='amplitude', 
                    title='Transient Amplitude Distribution',
                    nbins=min(20, len(transients_df)//2)
                )
                fig_amp.update_layout(xaxis_title='Amplitude (ΔF/F)', yaxis_title='Count')
                st.plotly_chart(fig_amp, use_container_width=True)
            
            with col2:
                fig_dur = px.histogram(
                    transients_df, 
                    x='duration', 
                    title='Transient Duration Distribution',
                    nbins=min(20, len(transients_df)//2)
                )
                fig_dur.update_layout(xaxis_title='Duration (s)', yaxis_title='Count')
                st.plotly_chart(fig_dur, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying transients: {str(e)}")
        st.exception(e)

def display_data_tables_tab(results):
    """Display data tables"""
    st.markdown("### Data Tables")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Timeseries", "Events", "PSTH Summary", "Transients"])
    
    with tab1:
        st.markdown("#### Processed Timeseries Data")
        
        # Create timeseries dataframe
        timeseries_data = {
            'time': results['processed_time'],
            'signal_470': results['processed_470'],
            'dff': results['dff'],
            'zscore_std': results['zscore_std'],
            'zscore_baseline': results['zscore_baseline'],
            'zscore_modified': results['zscore_modified']
        }
        
        if results['processed_410'] is not None:
            timeseries_data['signal_410'] = results['processed_410']
        
        timeseries_df = pd.DataFrame(timeseries_data)
        st.dataframe(timeseries_df, use_container_width=True)
    
    with tab2:
        st.markdown("#### Events Data")
        if not results['events_df'].empty:
            st.dataframe(results['events_df'], use_container_width=True)
        else:
            st.info("No events data available.")
    
    with tab3:
        st.markdown("#### PSTH Summary Statistics")
        
        summary_data = []
        for event_label, result in results['psth_results'].items():
            if result['n_trials'] > 0:
                stats = result['summary_stats']
                summary_data.append({
                    'Event': event_label,
                    'N_Trials': result['n_trials'],
                    'Mean_AUC': stats['mean_auc'],
                    'SEM_AUC': stats['sem_auc'],
                    'Mean_Peak_Amplitude': stats['mean_peak_amplitude'],
                    'SEM_Peak_Amplitude': stats['sem_peak_amplitude'],
                    'Mean_Peak_Latency': stats['mean_peak_latency'],
                    'SEM_Peak_Latency': stats['sem_peak_latency']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No PSTH summary data available.")
    
    with tab4:
        st.markdown("#### Detected Transients")
        
        transients = results['transient_results']['transients']
        if len(transients) > 0:
            transients_df = pd.DataFrame(transients)
            st.dataframe(transients_df, use_container_width=True)
        else:
            st.info("No transients detected.")

def display_download_tab(results):
    """Display download options"""
    st.markdown("### Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Data Files")
        
        # Timeseries CSV
        timeseries_data = {
            'time': results['processed_time'],
            'signal_470': results['processed_470'],
            'dff': results['dff'],
            'zscore_std': results['zscore_std'],
            'zscore_baseline': results['zscore_baseline'],
            'zscore_modified': results['zscore_modified']
        }
        
        if results['processed_410'] is not None:
            timeseries_data['signal_410'] = results['processed_410']
        
        timeseries_df = pd.DataFrame(timeseries_data)
        csv_data = timeseries_df.to_csv(index=False)
        
        st.download_button(
            label="📈 Download Timeseries CSV",
            data=csv_data,
            file_name="flair_timeseries.csv",
            mime="text/csv"
        )
        
        # Events CSV
        if not results['events_df'].empty:
            events_csv = results['events_df'].to_csv(index=False)
            st.download_button(
                label="🎯 Download Events CSV",
                data=events_csv,
                file_name="flair_events.csv",
                mime="text/csv"
            )
        
        # Transients CSV
        transients = results['transient_results']['transients']
        if len(transients) > 0:
            transients_df = pd.DataFrame(transients)
            transients_csv = transients_df.to_csv(index=False)
            st.download_button(
                label="🔍 Download Transients CSV",
                data=transients_csv,
                file_name="flair_transients.csv",
                mime="text/csv"
            )
    
    with col2:
        st.markdown("#### 📋 Reports")
        
        # Configuration file
        config_yaml = yaml.dump(st.session_state.config, default_flow_style=False)
        st.download_button(
            label="⚙️ Download Configuration YAML",
            data=config_yaml,
            file_name="flair_config.yaml",
            mime="text/yaml"
        )
        
        # Analysis summary JSON
        analysis_summary = {
            'metadata': results['metadata'],
            'processing_log': results['processing_log'],
            'psth_summary': {
                event: {
                    'n_trials': res['n_trials'],
                    'summary_stats': res['summary_stats']
                } for event, res in results['psth_results'].items()
            },
            'transient_summary': {
                'n_transients': results['transient_results']['n_transients'],
                'frequency_per_hour': results['transient_results']['frequency_per_hour'],
                'summary_stats': results['transient_results']['summary_stats']
            }
        }
        
        summary_json = json.dumps(analysis_summary, indent=2, default=str)
        st.download_button(
            label="📊 Download Analysis Summary JSON",
            data=summary_json,
            file_name="flair_analysis_summary.json",
            mime="application/json"
        )

def render_footer():
    """Render the footer"""
    st.markdown(
        '<div class="footer">'
        'FLAIR - Fiber-photometry Live Analysis and Intuitive Results<br>'
        'Product Owner: Elsa Karam | Developer: Chadi Abi Fadel<br>'
        'Built with Streamlit and the Fiber Photometry Analysis Pipeline'
        '</div>',
        unsafe_allow_html=True
    )

def main():
    """Main application function"""
    initialize_session_state()
    
    render_header()
    
    # Sidebar configuration
    render_sidebar()
    
    # Main content
    uploaded_file = upload_file()
    
    if uploaded_file is not None:
        if not st.session_state.analysis_complete:
            run_analysis(uploaded_file)
        else:
            # Option to start new analysis
            st.info("✅ Analysis completed! You can view results below or upload a new file to start over.")
            if st.button("🔄 Start New Analysis"):
                st.session_state.analysis_complete = False
                st.session_state.results = {}
                st.rerun()
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete:
        display_results()
    
    render_footer()

if __name__ == "__main__":
    main()