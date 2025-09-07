"""
CSV Parser for fiber-photometry data files.

Handles the custom CSV format with semicolon-separated JSON metadata
and standard CSV data columns.
"""

import json
import re
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np


class FluorescenceParser:
    """Robust parser for fiber-photometry CSV files with embedded metadata."""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.metadata: Dict[str, Any] = {}
        self.timeseries: Optional[pd.DataFrame] = None
        self.events: Optional[pd.DataFrame] = None
        
    def parse(self) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        """
        Parse the CSV file and extract metadata, timeseries, and events.
        
        Returns:
            Tuple of (metadata_dict, timeseries_df, events_df)
        """
        self._extract_metadata()
        self._extract_timeseries()
        self._extract_events()
        self._validate_data()
        
        return self.metadata, self.timeseries, self.events
    
    def _extract_metadata(self) -> None:
        """Extract and parse the JSON-like metadata from the first line."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        # Convert semicolon-separated format to proper JSON
        json_str = self._semicolon_to_json(first_line)
        
        try:
            self.metadata = json.loads(json_str)
        except json.JSONDecodeError as e:
            warnings.warn(f"Failed to parse metadata as JSON: {e}")
            # Fallback: manual parsing
            self.metadata = self._manual_metadata_parse(first_line)
    
    def _semicolon_to_json(self, semicolon_str: str) -> str:
        """Convert semicolon-separated pseudo-JSON to valid JSON."""
        # Replace semicolons with commas, but not inside quoted strings
        in_quotes = False
        result = []
        i = 0
        
        while i < len(semicolon_str):
            char = semicolon_str[i]
            
            if char == '"' and (i == 0 or semicolon_str[i-1] != '\\'):
                in_quotes = not in_quotes
                result.append(char)
            elif char == ';' and not in_quotes:
                result.append(',')
            else:
                result.append(char)
            i += 1
        
        return ''.join(result)
    
    def _manual_metadata_parse(self, line: str) -> Dict[str, Any]:
        """Fallback manual parsing for malformed metadata."""
        metadata = {}
        
        # Extract FPS
        fps_match = re.search(r'"Fps":\s*([0-9.]+)', line)
        if fps_match:
            metadata['Fps'] = float(fps_match.group(1))
        
        # Extract LED settings
        led_pattern = r'"Led(\d+)Enable":\s*(true|false)'
        led_matches = re.findall(led_pattern, line)
        if led_matches:
            metadata['Light'] = {}
            for wavelength, enabled in led_matches:
                metadata['Light'][f'Led{wavelength}Enable'] = enabled == 'true'
        
        # Extract LED values
        led_val_pattern = r'"Led(\d+)Value":\s*([0-9.]+)'
        led_val_matches = re.findall(led_val_pattern, line)
        if led_val_matches:
            if 'Light' not in metadata:
                metadata['Light'] = {}
            for wavelength, value in led_val_matches:
                metadata['Light'][f'Led{wavelength}Value'] = float(value)
        
        # Extract channel information
        channel_pattern = r'"Name":\s*"(CH\d+)"'
        channels = re.findall(channel_pattern, line)
        if channels:
            metadata['Channels'] = [{'Name': ch} for ch in channels]
        
        # Extract AllMarking events (simplified)
        events_pattern = r'"name":\s*"([^"]+)"'
        event_names = re.findall(events_pattern, line)
        if event_names:
            metadata['AllMarking'] = [{'name': name} for name in event_names]
        
        return metadata
    
    def _extract_timeseries(self) -> None:
        """Extract the time-series data starting from line 2."""
        # Read CSV data starting from line 2 (skip metadata header)
        self.timeseries = pd.read_csv(
            self.filepath, 
            skiprows=1,
            encoding='utf-8'
        )
        
        # Clean column names (remove trailing commas, spaces)
        self.timeseries.columns = [col.strip().rstrip(',') for col in self.timeseries.columns]
        
        # Convert timestamp to seconds if in milliseconds
        if 'TimeStamp' in self.timeseries.columns:
            # Check if timestamps are in milliseconds (>1000 suggests ms)
            if self.timeseries['TimeStamp'].max() > 1000:
                self.timeseries['TimeStamp'] = self.timeseries['TimeStamp'] / 1000.0
            
            # Create a time column starting from 0
            self.timeseries['time'] = self.timeseries['TimeStamp'] - self.timeseries['TimeStamp'].iloc[0]
        
        # Identify signal channels
        signal_channels = [col for col in self.timeseries.columns 
                          if 'CH' in col and any(wl in col for wl in ['410', '470', '560'])]
        
        if not signal_channels:
            raise ValueError("No signal channels (CH1-410, CH1-470, etc.) found in data")
        
        # Store channel information
        self.metadata['detected_channels'] = signal_channels
        
    def _extract_events(self) -> None:
        """Extract event markers from the Events column."""
        events_list = []
        
        if 'Events' in self.timeseries.columns:
            # Find rows with non-empty Events
            event_rows = self.timeseries[self.timeseries['Events'].notna() & 
                                      (self.timeseries['Events'] != '')]
            
            for idx, row in event_rows.iterrows():
                events_list.append({
                    'timestamp': row['time'] if 'time' in row else row['TimeStamp'],
                    'event_label': row['Events'],
                    'trial_index': len(events_list)  # Sequential numbering
                })
        
        self.events = pd.DataFrame(events_list)
        
        # If no events found in Events column, create from metadata
        if self.events.empty and 'AllMarking' in self.metadata:
            # Create placeholder events based on metadata definitions
            available_events = [event['name'] for event in self.metadata['AllMarking'] 
                              if isinstance(event, dict) and 'name' in event]
            
            # Note: Actual event timestamps would need to be populated from 
            # external source or user input
            self.events = pd.DataFrame(columns=['timestamp', 'event_label', 'trial_index'])
            self.metadata['available_event_types'] = available_events
    
    def _validate_data(self) -> None:
        """Validate the parsed data for consistency."""
        # Check FPS consistency
        if 'time' in self.timeseries.columns and len(self.timeseries) > 1:
            time_diff = np.diff(self.timeseries['time'].values)
            actual_fps = 1.0 / np.median(time_diff)
            expected_fps = self.metadata.get('Fps', 100.0)
            
            if abs(actual_fps - expected_fps) > 5:  # 5 Hz tolerance
                warnings.warn(
                    f"FPS mismatch: Expected {expected_fps:.1f} Hz, "
                    f"got {actual_fps:.1f} Hz from timestamps"
                )
        
        # Check for missing data
        for col in self.timeseries.columns:
            if col.startswith('CH') and self.timeseries[col].isna().sum() > 0:
                warnings.warn(f"Missing data detected in channel {col}")
        
        # Validate channel presence
        has_410 = any('410' in col for col in self.timeseries.columns)
        has_470 = any('470' in col for col in self.timeseries.columns)
        
        if not has_470:
            warnings.warn("No 470nm signal channel detected")
        if not has_410:
            warnings.warn("No 410nm control channel detected - single channel analysis will be used")
        
        self.metadata['dual_channel'] = has_410 and has_470
        
    def get_channel_data(self, wavelength: str) -> Optional[np.ndarray]:
        """Get data for a specific wavelength channel."""
        channel_col = None
        for col in self.timeseries.columns:
            if wavelength in col:
                channel_col = col
                break
        
        if channel_col is None:
            return None
        
        return self.timeseries[channel_col].values
    
    def get_time_vector(self) -> np.ndarray:
        """Get the time vector in seconds."""
        if 'time' in self.timeseries.columns:
            return self.timeseries['time'].values
        elif 'TimeStamp' in self.timeseries.columns:
            timestamps = self.timeseries['TimeStamp'].values
            if timestamps.max() > 1000:  # Convert from ms
                timestamps = timestamps / 1000.0
            return timestamps - timestamps[0]  # Start from 0
        else:
            # Fallback: generate time vector from FPS
            fps = self.metadata.get('Fps', 100.0)
            return np.arange(len(self.timeseries)) / fps
    
    def summary(self) -> Dict[str, Any]:
        """Return a summary of the parsed data."""
        time_vec = self.get_time_vector()
        
        summary = {
            'file_path': str(self.filepath),
            'duration_seconds': time_vec[-1] - time_vec[0] if len(time_vec) > 0 else 0,
            'n_samples': len(self.timeseries) if self.timeseries is not None else 0,
            'fps': self.metadata.get('Fps', 'Unknown'),
            'channels': self.metadata.get('detected_channels', []),
            'dual_channel': self.metadata.get('dual_channel', False),
            'n_events': len(self.events) if self.events is not None else 0,
            'available_event_types': self.metadata.get('available_event_types', []),
            'led_settings': self.metadata.get('Light', {})
        }
        
        return summary