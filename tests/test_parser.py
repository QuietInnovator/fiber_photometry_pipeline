"""
Unit tests for the CSV parser module.
"""

import pytest
import tempfile
import json
from pathlib import Path
import pandas as pd
import numpy as np

from fiber_photometry.parser import FluorescenceParser


class TestFluorescenceParser:
    """Test cases for FluorescenceParser."""
    
    def create_test_csv(self, metadata_dict, data_rows):
        """Helper to create test CSV files."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        # Convert metadata dict to semicolon-separated string
        metadata_str = json.dumps(metadata_dict).replace(':', ':').replace(',', ';')
        
        # Write header and data
        temp_file.write(metadata_str + '\n')
        temp_file.write('TimeStamp,Events,CH1-410,CH1-470,\n')
        
        for row in data_rows:
            temp_file.write(f"{row[0]},,{row[1]},{row[2]},\n")
        
        temp_file.close()
        return temp_file.name
    
    def test_parse_basic_metadata(self):
        """Test parsing of basic metadata."""
        metadata = {
            "Light": {"Led410Enable": True, "Led470Enable": True, "Led410Value": 25.0, "Led470Value": 60.0},
            "Fps": 100.0,
            "Channels": [{"Name": "CH1"}],
            "AllMarking": [
                {"name": "cocaine", "type": 1},
                {"name": "foot shock", "type": 1}
            ]
        }
        
        data_rows = [
            (0.0, 18.0, 28.0),
            (20.0, 18.1, 28.1),
            (40.0, 18.2, 28.2)
        ]
        
        csv_path = self.create_test_csv(metadata, data_rows)
        
        try:
            parser = FluorescenceParser(csv_path)
            parsed_metadata, timeseries, events = parser.parse()
            
            # Test metadata parsing
            assert parsed_metadata['Fps'] == 100.0
            assert parsed_metadata['Light']['Led410Enable'] is True
            assert parsed_metadata['Light']['Led470Enable'] is True
            assert len(parsed_metadata['AllMarking']) == 2
            
            # Test timeseries parsing
            assert len(timeseries) == 3
            assert 'time' in timeseries.columns
            assert 'CH1-410' in timeseries.columns
            assert 'CH1-470' in timeseries.columns
            
            # Test time vector creation
            time_vec = parser.get_time_vector()
            assert len(time_vec) == 3
            assert time_vec[0] == 0.0
            
        finally:
            Path(csv_path).unlink()  # Clean up
    
    def test_get_channel_data(self):
        """Test channel data extraction."""
        metadata = {"Fps": 100.0}
        data_rows = [
            (0.0, 18.0, 28.0),
            (20.0, 18.1, 28.1),
            (40.0, 18.2, 28.2)
        ]
        
        csv_path = self.create_test_csv(metadata, data_rows)
        
        try:
            parser = FluorescenceParser(csv_path)
            parser.parse()
            
            # Test channel data extraction
            data_470 = parser.get_channel_data('470')
            data_410 = parser.get_channel_data('410')
            data_nonexistent = parser.get_channel_data('560')
            
            assert data_470 is not None
            assert len(data_470) == 3
            assert np.allclose(data_470, [28.0, 28.1, 28.2])
            
            assert data_410 is not None
            assert len(data_410) == 3
            assert np.allclose(data_410, [18.0, 18.1, 18.2])
            
            assert data_nonexistent is None
            
        finally:
            Path(csv_path).unlink()
    
    def test_millisecond_timestamps(self):
        """Test handling of millisecond timestamps."""
        metadata = {"Fps": 100.0}
        data_rows = [
            (0.0, 18.0, 28.0),
            (1000.0, 18.1, 28.1),  # 1000ms = 1s
            (2000.0, 18.2, 28.2)   # 2000ms = 2s
        ]
        
        csv_path = self.create_test_csv(metadata, data_rows)
        
        try:
            parser = FluorescenceParser(csv_path)
            parser.parse()
            
            time_vec = parser.get_time_vector()
            
            # Should be converted to seconds and start from 0
            assert np.allclose(time_vec, [0.0, 1.0, 2.0])
            
        finally:
            Path(csv_path).unlink()
    
    def test_malformed_metadata_fallback(self):
        """Test fallback parsing for malformed metadata."""
        # Create a CSV with malformed JSON-like metadata
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        # Malformed metadata (missing quotes, etc.)
        malformed_metadata = '{"Fps": 100.0; "Led470Enable": true; malformed}'
        
        temp_file.write(malformed_metadata + '\n')
        temp_file.write('TimeStamp,Events,CH1-410,CH1-470,\n')
        temp_file.write('0.0,,18.0,28.0,\n')
        temp_file.write('20.0,,18.1,28.1,\n')
        
        temp_file.close()
        
        try:
            parser = FluorescenceParser(temp_file.name)
            
            # Should not raise exception, but use fallback parsing
            metadata, timeseries, events = parser.parse()
            
            # Should still extract some basic information
            assert len(timeseries) == 2
            assert 'CH1-410' in timeseries.columns
            assert 'CH1-470' in timeseries.columns
            
        finally:
            Path(temp_file.name).unlink()
    
    def test_summary_generation(self):
        """Test summary information generation."""
        metadata = {
            "Fps": 100.0,
            "Light": {"Led410Enable": True, "Led470Enable": True}
        }
        data_rows = [(i * 10.0, 18.0 + i * 0.1, 28.0 + i * 0.1) for i in range(10)]
        
        csv_path = self.create_test_csv(metadata, data_rows)
        
        try:
            parser = FluorescenceParser(csv_path)
            parser.parse()
            
            summary = parser.summary()
            
            assert summary['fps'] == 100.0
            assert summary['n_samples'] == 10
            assert summary['dual_channel'] is True
            assert 'CH1-410' in summary['channels']
            assert 'CH1-470' in summary['channels']
            assert summary['duration_seconds'] > 0
            
        finally:
            Path(csv_path).unlink()
    
    def test_single_channel_detection(self):
        """Test detection of single channel data."""
        metadata = {"Fps": 100.0}
        
        # Create CSV with only 470nm channel
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write(json.dumps(metadata).replace(',', ';') + '\n')
        temp_file.write('TimeStamp,Events,CH1-470,\n')
        temp_file.write('0.0,,28.0,\n')
        temp_file.write('20.0,,28.1,\n')
        temp_file.close()
        
        try:
            parser = FluorescenceParser(temp_file.name)
            parser.parse()
            
            summary = parser.summary()
            
            assert summary['dual_channel'] is False
            assert len(summary['channels']) == 1
            assert 'CH1-470' in summary['channels']
            
            # 410nm data should be None
            assert parser.get_channel_data('410') is None
            assert parser.get_channel_data('470') is not None
            
        finally:
            Path(temp_file.name).unlink()
    
    def test_empty_events(self):
        """Test handling of empty events."""
        metadata = {"Fps": 100.0}
        data_rows = [(0.0, 18.0, 28.0), (20.0, 18.1, 28.1)]
        
        csv_path = self.create_test_csv(metadata, data_rows)
        
        try:
            parser = FluorescenceParser(csv_path)
            metadata_parsed, timeseries, events = parser.parse()
            
            # Events should be empty DataFrame
            assert len(events) == 0
            assert 'timestamp' in events.columns or len(events.columns) == 0
            
        finally:
            Path(csv_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__])