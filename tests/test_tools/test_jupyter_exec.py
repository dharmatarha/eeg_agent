import json
import pytest
from unittest.mock import patch, MagicMock
import websocket
from src.tools.jupyter_exec import get_or_create_kernel, stateful_jupyter_exec
import urllib.error

@patch("urllib.request.urlopen")
def test_get_or_create_kernel_existing(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps([{"id": "test-kernel-123"}]).encode('utf-8')
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response
    
    kid = get_or_create_kernel()
    assert kid == "test-kernel-123"

@patch("urllib.request.urlopen")
def test_get_or_create_kernel_new(mock_urlopen):
    # First call (GET) raises URLError simulating no kernels
    # Second call (POST) returns new kernel
    def side_effect(req, *args, **kwargs):
        if req.get_method() == "GET":
            raise urllib.error.URLError("Not found")
        
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"id": "new-kernel-456"}).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        return mock_response
        
    mock_urlopen.side_effect = side_effect
    
    kid = get_or_create_kernel()
    assert kid == "new-kernel-456"

@patch("src.tools.jupyter_exec.get_or_create_kernel")
@patch("websocket.create_connection")
def test_stateful_jupyter_exec_success(mock_create_connection, mock_get_kernel):
    mock_get_kernel.return_value = "fake-kernel-id"
    
    mock_ws = MagicMock()
    mock_create_connection.return_value = mock_ws
    
    # We need a predictable msg_id to test matching parent_id
    # We patch uuid to return a specific ID
    with patch("uuid.uuid4") as mock_uuid:
        mock_uuid.return_value.hex = "test-msg-id"
        
        # Simulate stream, display_data, and execute_reply
        mock_ws.recv.side_effect = [
            json.dumps({
                "header": {"msg_type": "stream"},
                "parent_header": {"msg_id": "test-msg-id"},
                "content": {"text": "hello\n"}
            }),
            json.dumps({
                "header": {"msg_type": "display_data"},
                "parent_header": {"msg_id": "test-msg-id"},
                "content": {"data": {"image/png": "base64data"}}
            }),
            json.dumps({
                "header": {"msg_type": "execute_reply"},
                "parent_header": {"msg_id": "test-msg-id"},
                "content": {"status": "ok"}
            })
        ]
        
        result_json = stateful_jupyter_exec.invoke("print('hello')")
        result = json.loads(result_json)
        
        assert result["logs"] == "hello\n"
        assert result["error"] is False
        assert result["images"] == ["base64data"]
        mock_ws.send.assert_called_once()
        mock_ws.close.assert_called_once()

@patch("src.tools.jupyter_exec.get_or_create_kernel")
@patch("websocket.create_connection")
def test_stateful_jupyter_exec_error(mock_create_connection, mock_get_kernel):
    mock_get_kernel.return_value = "fake-kernel-id"
    
    mock_ws = MagicMock()
    mock_create_connection.return_value = mock_ws
    
    with patch("uuid.uuid4") as mock_uuid:
        mock_uuid.return_value.hex = "test-msg-id"
        
        mock_ws.recv.side_effect = [
            json.dumps({
                "header": {"msg_type": "error"},
                "parent_header": {"msg_id": "test-msg-id"},
                "content": {"traceback": ["Traceback line 1", "Traceback line 2"]}
            }),
            json.dumps({
                "header": {"msg_type": "execute_reply"},
                "parent_header": {"msg_id": "test-msg-id"},
                "content": {"status": "error"}
            })
        ]
        
        result_json = stateful_jupyter_exec.invoke("1/0")
        result = json.loads(result_json)
        
        assert "Traceback line 1\nTraceback line 2" in result["logs"]
        assert result["error"] is True

@patch("src.tools.jupyter_exec.get_or_create_kernel")
@patch("websocket.create_connection")
@patch("src.config.get_val")
def test_stateful_jupyter_exec_timeout(mock_get_config, mock_create_connection, mock_get_kernel):
    mock_get_kernel.return_value = "fake-kernel-id"
    mock_get_config.side_effect = lambda key, default=None: 0.01 if key == "sandbox.timeout" else default
    
    mock_ws = MagicMock()
    mock_create_connection.return_value = mock_ws
    
    with patch("uuid.uuid4") as mock_uuid:
        mock_uuid.return_value.hex = "test-msg-id"
        
        # Simulate websocket timeout exception
        mock_ws.recv.side_effect = websocket.WebSocketTimeoutException("Timeout")
        
        result_json = stateful_jupyter_exec.invoke("print('slow')")
        result = json.loads(result_json)
        
        assert "timed out" in result["logs"]
        assert result["error"] is True

