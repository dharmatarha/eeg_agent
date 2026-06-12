import json
import pytest
from unittest.mock import patch, MagicMock, mock_open
from src.tools.bids_inspector import bids_inspector

@patch("os.path.exists")
def test_bids_inspector_dir_not_found(mock_exists):
    mock_exists.return_value = False
    result_json = bids_inspector.invoke({"bids_root": "missing_dir"})
    result = json.loads(result_json)
    assert "error" in result
    assert "BIDS root directory not found" in result["error"]

@patch("os.path.exists")
@patch("os.path.isdir")
@patch("os.walk")
@patch("builtins.open", new_callable=mock_open, read_data='{"Name": "Test BIDS Dataset"}')
@patch("mne.io.read_raw")
def test_bids_inspector_success(mock_read_raw, mock_open_file, mock_walk, mock_isdir, mock_exists):
    mock_exists.return_value = True
    mock_isdir.return_value = True
    
    # Mock os.walk structure: (root, dirs, files)
    mock_walk.return_value = [
        ("/data/bids", ["sub-01", "sub-02", "derivatives"], ["dataset_description.json"]),
        ("/data/bids/sub-01", ["ses-01"], []),
        ("/data/bids/sub-01/ses-01", ["eeg"], []),
        ("/data/bids/sub-01/ses-01/eeg", [], ["sub-01_ses-01_task-P300_run-1_eeg.fif"]),
        ("/data/bids/sub-02", ["ses-01"], []),
        ("/data/bids/sub-02/ses-01", ["eeg"], []),
        ("/data/bids/sub-02/ses-01/eeg", [], ["sub-02_ses-01_task-P300_run-1_eeg.fif"])
    ]
    
    # Mock MNE raw object
    mock_raw = MagicMock()
    mock_raw.ch_names = ["EEG1", "EEG2"]
    mock_raw.info = {
        "sfreq": 250.0,
        "highpass": 0.1,
        "lowpass": 50.0
    }
    mock_raw.annotations = None
    mock_read_raw.return_value = mock_raw
    
    result_json = bids_inspector.invoke({"bids_root": "/data/bids"})
    result = json.loads(result_json)
    
    assert result["dataset_name"] == "Test BIDS Dataset"
    assert result["n_subjects"] == 2
    assert result["subjects"] == ["sub-01", "sub-02"]
    assert result["sessions"] == ["ses-01"]
    assert result["tasks"] == ["P300"]
    assert result["runs"] == ["1"]
    
    rep_meta = result["representative_metadata"]
    assert rep_meta["ch_names"] == ["EEG1", "EEG2"]
    assert rep_meta["n_channels"] == 2
    assert rep_meta["sfreq"] == 250.0
