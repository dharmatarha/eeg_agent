import pytest
from unittest.mock import patch, MagicMock
from src.tools.web_search import web_search

@patch("src.tools.web_search.DuckDuckGoSearchRun")
def test_web_search_success(mock_ddg_class):
    mock_ddg_instance = MagicMock()
    mock_ddg_instance.run.return_value = "This is a search result."
    mock_ddg_class.return_value = mock_ddg_instance
    
    result = web_search.invoke({"query": "mne-python raw"})
    assert result == "This is a search result."
    mock_ddg_instance.run.assert_called_once_with("mne-python raw")

@patch("src.tools.web_search.DuckDuckGoSearchRun")
def test_web_search_failure(mock_ddg_class):
    mock_ddg_instance = MagicMock()
    mock_ddg_instance.run.side_effect = Exception("Connection error")
    mock_ddg_class.return_value = mock_ddg_instance
    
    result = web_search.invoke({"query": "mne-python raw"})
    assert "Error executing web search" in result
    assert "Connection error" in result
