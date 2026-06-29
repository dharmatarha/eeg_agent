import os
import json
import logging
from langchain_core.tools import tool

logger = logging.getLogger("eeg_agent.tools.reference_run_reader")

@tool
def read_reference_run_file(filename: str, reference_thread_id: str) -> str:
    """
    Read the contents of a text file from a previous reference run's output directory.
    Supported files are: 'run_memory.json', 'analysis_pipeline.py', and 'final_report.md'.
    
    Parameters:
      - filename (str): The name of the file to read (e.g., 'analysis_pipeline.py', 'final_report.md', or 'run_memory.json').
      - reference_thread_id (str): The unique thread ID of the reference run (e.g., 'run_YYYYMMDD_HHMMSS_uuid').
    """
    logger.info("Reading file %s from reference run %s", filename, reference_thread_id)
    if filename not in ["run_memory.json", "analysis_pipeline.py", "final_report.md"]:
        return json.dumps({"error": f"Unsupported file name: {filename}. Only 'run_memory.json', 'analysis_pipeline.py', and 'final_report.md' are supported."})
        
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    file_path = os.path.abspath(os.path.join(project_root, "output", reference_thread_id, filename))
    
    # Security check: ensure path is within the output directory
    output_dir = os.path.abspath(os.path.join(project_root, "output"))
    if not file_path.startswith(output_dir):
        return json.dumps({"error": "Access denied: Path is outside the output directory."})
        
    if not os.path.exists(file_path):
        return json.dumps({"error": f"File not found: {filename} in reference run {reference_thread_id}"})
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return json.dumps({
            "filename": filename,
            "reference_thread_id": reference_thread_id,
            "content": content
        }, indent=2)
    except Exception as e:
        logger.error("Failed to read reference run file: %s", e)
        return json.dumps({"error": f"Failed to read file: {e}"})
