import os
import json
import pytest
from unittest.mock import patch, MagicMock
from src.tools.dataset_explorer import dataset_explorer

def test_dataset_explorer_path_not_exists():
    result_json = dataset_explorer.invoke({"action": "list", "path": "/nonexistent/path"})
    result = json.loads(result_json)
    assert "error" in result
    assert "Path does not exist" in result["error"]

def test_dataset_explorer_list(tmp_path):
    # Create temp files
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "README.md").write_text("dataset readme")
    (d / "sub-01_task-P300_eeg.fif").write_text("raw eeg data")
    (d / "sub-02_task-P300_eeg.fif").write_text("raw eeg data")
    (d / "other_file.txt").write_text("other")

    # List all
    res_all_json = dataset_explorer.invoke({"action": "list", "path": str(d)})
    res_all = json.loads(res_all_json)
    assert res_all["action"] == "list"
    assert res_all["total_matches"] == 4
    assert "README.md" in res_all["files"]
    assert "sub-01_task-P300_eeg.fif" in res_all["files"]

    # List with pattern
    res_pattern_json = dataset_explorer.invoke({"action": "list", "path": str(d), "pattern": "*.fif"})
    res_pattern = json.loads(res_pattern_json)
    assert res_pattern["total_matches"] == 2
    assert "sub-01_task-P300_eeg.fif" in res_pattern["files"]
    assert "README.md" not in res_pattern["files"]

def test_dataset_explorer_read(tmp_path):
    f = tmp_path / "README.md"
    f.write_text("This is a mock EEG dataset description file.")

    # Read full
    res_read_json = dataset_explorer.invoke({"action": "read", "path": str(f)})
    res_read = json.loads(res_read_json)
    assert res_read["action"] == "read"
    assert "mock EEG dataset" in res_read["content"]
    assert res_read["truncated"] is False

    # Read truncated
    res_trunc_json = dataset_explorer.invoke({"action": "read", "path": str(f), "max_bytes": 10})
    res_trunc = json.loads(res_trunc_json)
    assert len(res_trunc["content"]) == 10
    assert res_trunc["truncated"] is True

@patch("mne.io.read_raw")
def test_dataset_explorer_verify_consistency_consistent(mock_read_raw, tmp_path):
    d = tmp_path / "dataset"
    d.mkdir()
    f1 = d / "sub-01_eeg.fif"
    f2 = d / "sub-02_eeg.fif"
    f1.write_text("dummy")
    f2.write_text("dummy")

    # Mock raw object returned by mne.io.read_raw
    mock_raw = MagicMock()
    mock_raw.ch_names = ["EEG 1", "EEG 2", "Cz"]
    mock_raw.info = {"sfreq": 500.0}
    mock_read_raw.return_value = mock_raw

    res_json = dataset_explorer.invoke({"action": "verify_consistency", "path": str(d)})
    res = json.loads(res_json)

    assert res["action"] == "verify_consistency"
    assert res["is_consistent"] is True
    assert res["total_files_found"] == 2
    assert res["files_checked"] == 2
    assert "discrepancies" not in res

@patch("mne.io.read_raw")
def test_dataset_explorer_verify_consistency_inconsistent(mock_read_raw, tmp_path):
    d = tmp_path / "dataset"
    d.mkdir()
    f1 = d / "sub-01_eeg.fif"
    f2 = d / "sub-02_eeg.fif"
    f1.write_text("dummy")
    f2.write_text("dummy")

    # Set up mock to return different metadata for the two files
    mock_raw_1 = MagicMock()
    mock_raw_1.ch_names = ["EEG 1", "EEG 2", "Cz"]
    mock_raw_1.info = {"sfreq": 500.0}

    mock_raw_2 = MagicMock()
    mock_raw_2.ch_names = ["EEG 1", "EEG 2"] # Missing Cz
    mock_raw_2.info = {"sfreq": 250.0} # Different sfreq

    mock_read_raw.side_effect = [mock_raw_1, mock_raw_2]

    res_json = dataset_explorer.invoke({"action": "verify_consistency", "path": str(d)})
    res = json.loads(res_json)

    assert res["action"] == "verify_consistency"
    assert res["is_consistent"] is False
    assert "discrepancies" in res
    assert "sfreq" in res["discrepancies"]
    assert "channel_count" in res["discrepancies"]
    assert "channel_names" in res["discrepancies"]
