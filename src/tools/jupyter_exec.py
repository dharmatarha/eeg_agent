import json
import urllib.request
import urllib.error
import websocket
import uuid
import os
from langchain_core.tools import tool

# Gateway configuration (can be overridden by environment variables)
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8888")
WS_GATEWAY_URL = os.environ.get("WS_GATEWAY_URL", "ws://localhost:8888")
TOKEN = os.environ.get("JUPYTER_TOKEN", "eeg_adk_sandbox_token")

kernel_id = None

def get_or_create_kernel():
    global kernel_id
    headers = {"Authorization": f"token {TOKEN}"} if TOKEN else {}
    
    # Try to list existing kernels
    try:
        req = urllib.request.Request(f"{GATEWAY_URL}/api/kernels", headers=headers)
        with urllib.request.urlopen(req) as response:
            kernels = json.loads(response.read().decode())
            if kernels:
                kernel_id = kernels[0]['id']
                return kernel_id
    except urllib.error.URLError:
        pass
        
    # Create new kernel if none exist
    req = urllib.request.Request(f"{GATEWAY_URL}/api/kernels", headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            kernel_info = json.loads(response.read().decode())
            kernel_id = kernel_info['id']
            return kernel_id
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to connect to Jupyter Kernel Gateway: {e}")

@tool
def stateful_jupyter_exec(code_string: str) -> str:
    """
    Execute Python code in the stateful Docker container.
    Use this to run MNE-Python commands.
    Data loaded in previous turns remains in memory.
    Returns a JSON string containing the execution 'logs', boolean 'error' flag, and 'images' (list of base64 strings).
    """
    try:
        kid = get_or_create_kernel()
    except Exception as e:
        return json.dumps({"logs": str(e), "error": True, "images": []})
        
    ws_url = f"{WS_GATEWAY_URL}/api/kernels/{kid}/channels"
    if TOKEN:
        ws_url += f"?token={TOKEN}"
        
    try:
        ws = websocket.create_connection(ws_url, timeout=60)
    except Exception as e:
        return json.dumps({"logs": f"WebSocket connection failed: {e}", "error": True, "images": []})
    
    msg_id = uuid.uuid4().hex
    msg = {
        "header": {
            "msg_id": msg_id,
            "username": "eeg_agent",
            "session": uuid.uuid4().hex,
            "msg_type": "execute_request",
            "version": "5.0"
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code_string,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False
        }
    }
    
    ws.send(json.dumps(msg))
    
    outputs = []
    images = []
    error_occurred = False
    
    while True:
        try:
            rsp = json.loads(ws.recv())
            msg_type = rsp["header"]["msg_type"]
            parent_id = rsp["parent_header"].get("msg_id")
            
            if parent_id != msg_id:
                continue
                
            if msg_type == "stream":
                outputs.append(rsp["content"]["text"])
            elif msg_type in ("display_data", "execute_result"):
                data = rsp["content"]["data"]
                if "text/plain" in data:
                    outputs.append(data["text/plain"])
                if "image/png" in data:
                    images.append(data["image/png"])
            elif msg_type == "error":
                traceback = "\n".join(rsp["content"]["traceback"])
                outputs.append(traceback)
                error_occurred = True
            elif msg_type == "execute_reply":
                if rsp["content"]["status"] == "error":
                    error_occurred = True
                break
        except websocket.WebSocketTimeoutException:
            outputs.append("\nExecution timed out.")
            error_occurred = True
            break
        except Exception as e:
            outputs.append(f"\nWebSocket Error: {e}")
            error_occurred = True
            break
            
    ws.close()
    
    response_obj = {
        "logs": "".join(outputs),
        "error": error_occurred,
        "images": images
    }
    
    return json.dumps(response_obj)
