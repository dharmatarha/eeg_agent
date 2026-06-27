import json
import urllib.request
import urllib.error
import websocket
import uuid
import os
import logging
import time
from langchain_core.tools import tool

logger = logging.getLogger("eeg_agent.tools.jupyter_exec")

from src import config

# Gateway configuration (can be overridden by environment variables)
GATEWAY_URL = config.get_val("sandbox.gateway_url", "GATEWAY_URL")
WS_GATEWAY_URL = config.get_val("sandbox.ws_gateway_url", "WS_GATEWAY_URL")
TOKEN = config.get_val("sandbox.jupyter_token", "JUPYTER_TOKEN")

kernel_id = None

def get_or_create_kernel():
    global kernel_id
    headers = {"Authorization": f"token {TOKEN}"} if TOKEN else {}
    
    logger.debug("Checking for active Jupyter kernels at %s...", GATEWAY_URL)
    # Try to list existing kernels
    try:
        req = urllib.request.Request(f"{GATEWAY_URL}/api/kernels", headers=headers)
        with urllib.request.urlopen(req) as response:
            kernels = json.loads(response.read().decode())
            if kernels:
                kernel_id = kernels[0]['id']
                logger.info("Reusing existing Jupyter kernel: %s", kernel_id)
                return kernel_id
    except urllib.error.URLError:
        pass
        
    logger.info("Creating a new Jupyter kernel...")
    # Create new kernel if none exist
    req = urllib.request.Request(f"{GATEWAY_URL}/api/kernels", headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            kernel_info = json.loads(response.read().decode())
            kernel_id = kernel_info['id']
            logger.info("Successfully created new Jupyter kernel: %s", kernel_id)
            return kernel_id
    except urllib.error.URLError as e:
        logger.error("Failed to connect to Jupyter Kernel Gateway: %s", e)
        raise ConnectionError(f"Failed to connect to Jupyter Kernel Gateway: {e}")

@tool
def stateful_jupyter_exec(code_string: str) -> str:
    """
    Execute Python code in the stateful Docker container.
    Use this to run MNE-Python commands.
    Data loaded in previous turns remains in memory.
    Returns a JSON string containing the execution 'logs', boolean 'error' flag, and 'images' (list of base64 strings).
    """
    logger.info("Received request to execute code in stateful Jupyter sandbox.")
    try:
        kid = get_or_create_kernel()
    except Exception as e:
        logger.error("Kernel lookup/creation failed: %s", e)
        return json.dumps({"logs": str(e), "error": True, "images": []})
        
    ws_url = f"{WS_GATEWAY_URL}/api/kernels/{kid}/channels"
    if TOKEN:
        ws_url += f"?token={TOKEN}"
        
    logger.debug("Connecting to WebSocket gateway at %s...", WS_GATEWAY_URL)
    try:
        # Set socket connection timeout to 60 seconds
        ws = websocket.create_connection(ws_url, timeout=60)
    except Exception as e:
        logger.error("WebSocket connection failed: %s", e)
        return json.dumps({"logs": f"WebSocket connection failed: {e}", "error": True, "images": []})
    
    # Configure total execution timeout
    execution_timeout = config.get_val("sandbox.timeout", "SANDBOX_TIMEOUT")
    if execution_timeout is None:
        execution_timeout = 300.0
    else:
        try:
            execution_timeout = float(execution_timeout)
        except ValueError:
            execution_timeout = 300.0
            
    # Set short read timeout to prevent blocking indefinitely when receiving heartbeats/status messages
    ws.settimeout(5.0)
    
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
    
    logger.debug("Sending execute request message (id: %s)...", msg_id)
    ws.send(json.dumps(msg))
    
    outputs = []
    images = []
    error_occurred = False
    
    logger.debug("Awaiting execution reply and outputs from kernel...")
    start_time = time.time()
    while True:
        try:
            # Check if total execution time limit is exceeded
            elapsed = time.time() - start_time
            if elapsed > execution_timeout:
                logger.error("Jupyter sandbox execution exceeded total timeout limit of %ss.", execution_timeout)
                outputs.append(f"\nExecution timed out (exceeded total limit of {execution_timeout}s).")
                error_occurred = True
                break
                
            try:
                rsp_raw = ws.recv()
                rsp = json.loads(rsp_raw)
            except websocket.WebSocketTimeoutException:
                # Socket timed out but total execution limit not reached yet.
                # Continue loop to check elapsed time and wait again.
                continue
                
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
                    logger.info("Received base64-encoded plot from sandbox.")
            elif msg_type == "error":
                traceback = "\n".join(rsp["content"]["traceback"])
                outputs.append(traceback)
                error_occurred = True
                logger.warning("Code execution error received from Jupyter sandbox.")
            elif msg_type == "execute_reply":
                if rsp["content"]["status"] == "error":
                    error_occurred = True
                break
        except Exception as e:
            logger.error("Jupyter sandbox communication error: %s", e)
            outputs.append(f"\nWebSocket Error: {e}")
            error_occurred = True
            break
            
    ws.close()
    logger.info("Jupyter sandbox execution complete. Success=%s.", not error_occurred)
    
    response_obj = {
        "logs": "".join(outputs),
        "error": error_occurred,
        "images": images
    }
    
    return json.dumps(response_obj)
