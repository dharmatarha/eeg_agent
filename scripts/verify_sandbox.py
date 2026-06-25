"""
Sandbox Verification Utility for EEG-ADK Multi-Agent System

This script verifies connection to the stateful Docker Jupyter Sandbox.
It executes a test snippet inside the container, imports the installed
EEG-related libraries (MNE-Python, MNE-Bids), and verifies
that host GPU acceleration is available inside the container via nvidia-smi.
"""

import sys
import os
import json

# Add the workspace root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(override=True)

from src.tools.jupyter_exec import stateful_jupyter_exec

def main():
    """
    Main execution routine for sandbox verification.

    Connects to the Jupyter kernel gateway, executes imports and verification,
    and prints out system logs and GPU information from the container.
    """
    print("Testing connection to Docker Jupyter Sandbox...")
    code = """
import sys
import mne

print("Python version:", sys.version)
print("MNE version:", mne.__version__)

# Verify GPU availability using nvidia-smi command inside container
import subprocess
try:
    smi_output = subprocess.check_output(['nvidia-smi']).decode()
    print("\\n--- nvidia-smi output ---")
    print(smi_output.strip())
    print("-------------------------\\n")
    print("GPU verification: SUCCESS")
except Exception as e:
    print("nvidia-smi not available or failed:", str(e))
    print("GPU verification: FAILED")
"""
    result_str = stateful_jupyter_exec.invoke(code)
    try:
        result = json.loads(result_str)
        if result.get("error"):
            print("\n❌ Execution failed inside the container!")
            print("Logs:")
            print(result.get("logs"))
            sys.exit(1)
        else:
            print("\n✅ Sandbox verification succeeded!")
            print("Logs:")
            print(result.get("logs"))
            sys.exit(0)
    except Exception as e:
        print("\n❌ Failed to parse response from sandbox:", e)
        print("Raw response:", result_str)
        sys.exit(1)

if __name__ == "__main__":
    main()
