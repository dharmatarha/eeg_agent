import json
import pytest
from unittest.mock import patch, MagicMock
from src.tools.metadata_extractor import metadata_extractor

@patch("os.path.exists")
def test_metadata_extractor_file_not_found(mock_exists):
    mock_exists.return_value = False
    result_json = metadata_extractor.invoke({"file_path": "missing.fif"})
    result = json.loads(result_json)
    assert "error" in result
    assert "File not found" in result["error"]

@patch("os.path.exists")
@patch("mne.io.read_raw")
def test_metadata_extractor_success(mock_read_raw, mock_exists):
    mock_exists.return_value = True
    
    mock_raw = MagicMock()
    mock_raw.ch_names = ["EEG1", "EEG2"]
    mock_raw.info = {
        "sfreq": 250.0,
        "highpass": 0.1,
        "lowpass": 50.0
    }
    
    mock_annotations = MagicMock()
    mock_annotations.description = ["Stimulus/1", "Stimulus/2", "Stimulus/1"]
    mock_raw.annotations = mock_annotations
    
    mock_read_raw.return_value = mock_raw
    
    result_json = metadata_extractor.invoke({"file_path": "test.fif"})
    result = json.loads(result_json)
    
    assert result["ch_names"] == ["EEG1", "EEG2"]
    assert result["n_channels"] == 2
    assert result["sfreq"] == 250.0
    assert result["highpass"] == 0.1
    assert result["lowpass"] == 50.0
    assert set(result["annotations"]) == {"Stimulus/1", "Stimulus/2"}

@patch("os.path.exists")
@patch("mne.io.read_raw")
def test_metadata_extractor_exception(mock_read_raw, mock_exists):
    mock_exists.return_value = True
    mock_read_raw.side_effect = Exception("Read failed")
    
    result_json = metadata_extractor.invoke({"file_path": "test.fif"})
    result = json.loads(result_json)
    
    assert "error" in result
    assert "Failed to extract metadata: Read failed" in result["error"]
