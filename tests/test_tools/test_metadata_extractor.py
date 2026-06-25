import json
import os
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

@patch("os.path.exists")
@patch("mne.io.read_raw")
def test_metadata_extractor_channel_types(mock_read_raw, mock_exists):
    mock_exists.return_value = True
    
    mock_raw = MagicMock()
    mock_raw.ch_names = ["EEG1", "EEG2", "EOG1", "Status"]
    mock_raw.info = {"sfreq": 250.0}
    mock_raw.get_channel_types.return_value = ["eeg", "eeg", "eog", "stim"]
    mock_raw.annotations = None
    mock_read_raw.return_value = mock_raw
    
    result_json = metadata_extractor.invoke({"file_path": "test.fif"})
    result = json.loads(result_json)
    
    assert result["channel_types"] == {"eeg": 2, "eog": 1, "stim": 1}
    assert result["eeg_channels"] == ["EEG1", "EEG2"]
    assert result["eog_channels"] == ["EOG1"]
    assert "Status" in result["stim_channels"]

@patch("os.path.exists")
@patch("mne.io.read_raw")
@patch("mne.find_events")
def test_metadata_extractor_digital_events(mock_find_events, mock_read_raw, mock_exists):
    mock_exists.return_value = True
    
    mock_raw = MagicMock()
    mock_raw.ch_names = ["EEG1", "Status"]
    mock_raw.info = {"sfreq": 250.0}
    mock_raw.get_channel_types.return_value = ["eeg", "stim"]
    mock_raw.annotations = None
    mock_read_raw.return_value = mock_raw
    
    import numpy as np
    # Mock mne.find_events to return 3 events: two of ID 10, one of ID 20
    mock_find_events.return_value = np.array([
        [100, 0, 10],
        [200, 0, 20],
        [300, 0, 10]
    ])
    
    result_json = metadata_extractor.invoke({"file_path": "test.fif"})
    result = json.loads(result_json)
    
    assert result["digital_triggers"] == {"10": 2, "20": 1}

def test_metadata_extractor_brainvision_sidecar(tmp_path):
    vhdr = tmp_path / "subject.vhdr"
    vmrk = tmp_path / "subject.vmrk"
    
    vhdr.write_text("BrainVision Header file")
    vmrk.write_text("""[Common Infos]
Codepage=UTF-8
[Marker Infos]
Mk1=New Segment,,1,1,0,20260625061217000000
Mk2=Stimulus,S  1,100,1,0
Mk3=Stimulus,S  2,250,1,0
Mk4=Stimulus,S  1,400,1,0
""")
    
    # We patch mne.io.read_raw to return a raw object with no annotations/stim channels
    # so it falls back to parsing the .vmrk file
    with patch("os.path.exists", return_value=True), \
         patch("mne.io.read_raw") as mock_read_raw:
        
        mock_raw = MagicMock()
        mock_raw.ch_names = ["EEG1"]
        mock_raw.info = {"sfreq": 250.0}
        mock_raw.get_channel_types.return_value = ["eeg"]
        mock_raw.annotations = None
        mock_read_raw.return_value = mock_raw
        
        # We need to simulate that both os.path.exists("subject.vmrk") is True
        # and vmrk_path points to the actual temporary file
        original_exists = os.path.exists
        def mock_exists_side_effect(path):
            if path == str(vmrk):
                return True
            return original_exists(path)
            
        with patch("os.path.exists", side_effect=mock_exists_side_effect):
            result_json = metadata_extractor.invoke({"file_path": str(vhdr)})
            result = json.loads(result_json)
            
            assert set(result["annotations"]) == {"", "S  1", "S  2"}

