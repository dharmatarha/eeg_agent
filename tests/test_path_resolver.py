import os
from unittest.mock import patch
from src.utils.path_resolver import resolve_host_path

def test_resolve_host_path_empty_or_none():
    assert resolve_host_path("") == ""
    assert resolve_host_path(None) is None

def test_resolve_host_path_unchanged():
    # Paths not matching /mnt/data should be returned as-is
    assert resolve_host_path("/other/path/file.fif") == "/other/path/file.fif"
    assert resolve_host_path("relative/path.vhdr") == "relative/path.vhdr"

def test_resolve_host_path_with_env_var():
    with patch.dict(os.environ, {"EEG_DATA_DIR": "/custom/host/data"}):
        result = resolve_host_path("/mnt/data/sub-01/eeg.fif")
        # Should resolve /mnt/data to /custom/host/data
        assert result == "/custom/host/data/sub-01/eeg.fif"

def test_resolve_host_path_without_env_var():
    with patch.dict(os.environ, {}):
        if "EEG_DATA_DIR" in os.environ:
            del os.environ["EEG_DATA_DIR"]
            
        result = resolve_host_path("/mnt/data/ds004408")
        
        # Expected path is project_root + /data/ds004408
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        expected = os.path.abspath(os.path.join(project_root, "data", "ds004408"))
        assert result == expected

def test_resolve_host_path_exact_mnt_data():
    with patch.dict(os.environ, {"EEG_DATA_DIR": "/custom/host/data"}):
        result = resolve_host_path("/mnt/data")
        assert result == "/custom/host/data"
