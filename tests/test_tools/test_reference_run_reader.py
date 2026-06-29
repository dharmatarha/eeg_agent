import pytest
import os
import json
from unittest.mock import patch, mock_open
from src.tools.reference_run_reader import read_reference_run_file

def test_read_reference_run_file_unsupported():
    result_str = read_reference_run_file.invoke({
        "filename": "unsupported.txt",
        "reference_thread_id": "run_123"
    })
    result = json.loads(result_str)
    assert "error" in result
    assert "Unsupported file name" in result["error"]

@patch("os.path.exists")
def test_read_reference_run_file_not_found(mock_exists):
    mock_exists.return_value = False
    
    result_str = read_reference_run_file.invoke({
        "filename": "analysis_pipeline.py",
        "reference_thread_id": "run_123"
    })
    result = json.loads(result_str)
    assert "error" in result
    assert "File not found" in result["error"]

def test_read_reference_run_file_security_check():
    result_str = read_reference_run_file.invoke({
        "filename": "analysis_pipeline.py",
        "reference_thread_id": "../../../etc/passwd"
    })
    result = json.loads(result_str)
    assert "error" in result
    assert "Access denied" in result["error"]

@patch("os.path.exists")
def test_read_reference_run_file_success(mock_exists):
    mock_exists.return_value = True
    file_content = "import mne\nprint('success')"
    
    with patch("builtins.open", mock_open(read_data=file_content)):
        result_str = read_reference_run_file.invoke({
            "filename": "analysis_pipeline.py",
            "reference_thread_id": "run_123"
        })
        
    result = json.loads(result_str)
    assert result["filename"] == "analysis_pipeline.py"
    assert result["reference_thread_id"] == "run_123"
    assert result["content"] == file_content

@patch("os.path.exists")
def test_read_reference_run_file_exception(mock_exists):
    mock_exists.return_value = True
    
    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = IOError("Permission denied")
        result_str = read_reference_run_file.invoke({
            "filename": "analysis_pipeline.py",
            "reference_thread_id": "run_123"
        })
        
    result = json.loads(result_str)
    assert "error" in result
    assert "Permission denied" in result["error"]
