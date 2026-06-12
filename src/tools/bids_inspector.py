import os
import json
import logging
import re
import mne
from langchain_core.tools import tool

logger = logging.getLogger("eeg_agent.tools.bids_inspector")

@tool
def bids_inspector(bids_root: str) -> str:
    """
    Inspects a BIDS dataset directory to extract its structure and parameters.
    Returns a JSON string containing the dataset name, subjects list, sessions list,
    detected tasks, and metadata (channels, sampling frequency, annotations) from a representative subject.
    """
    logger.info("Starting BIDS inspection for root: %s", bids_root)
    if not os.path.exists(bids_root) or not os.path.isdir(bids_root):
        logger.error("BIDS root directory not found: %s", bids_root)
        return json.dumps({"error": f"BIDS root directory not found: {bids_root}"})
    
    # 1. Parse dataset_description.json
    dataset_name = "Unknown Dataset"
    desc_path = os.path.join(bids_root, "dataset_description.json")
    if os.path.exists(desc_path):
        try:
            with open(desc_path, "r", encoding="utf-8") as f:
                desc = json.load(f)
                dataset_name = desc.get("Name", dataset_name)
        except Exception as e:
            logger.warning("Failed to parse dataset_description.json: %s", e)

    # 2. Find subjects and sessions
    subjects = []
    sessions = set()
    tasks = set()
    runs = set()
    eeg_files = []
    
    for root, dirs, files in os.walk(bids_root):
        # Find unique subjects
        for d in dirs:
            if d.startswith("sub-"):
                subjects.append(d)
            elif d.startswith("ses-"):
                sessions.add(d)
                
        # Find unique tasks and EEG data files
        for f in files:
            task_match = re.search(r"_task-([a-zA-Z0-9]+)_", f)
            if task_match:
                tasks.add(task_match.group(1))
            
            run_match = re.search(r"_run-([a-zA-Z0-9]+)_", f)
            if run_match:
                runs.add(run_match.group(1))
                
            if f.endswith((".fif", ".set", ".edf", ".vhdr", ".bdf")):
                if "derivatives" not in root and "tmp" not in root:
                    eeg_files.append(os.path.join(root, f))
                    
    subjects = sorted(list(set(subjects)))
    
    metadata = {
        "dataset_name": dataset_name,
        "bids_root": bids_root,
        "n_subjects": len(subjects),
        "subjects": subjects[:10],  # list first 10
        "sessions": sorted(list(sessions)),
        "tasks": sorted(list(tasks)),
        "runs": sorted(list(runs)),
        "representative_metadata": {}
    }
    
    # 3. Read header of a representative EEG file if any found
    if eeg_files:
        rep_file = eeg_files[0]
        logger.info("Reading representative metadata from: %s", rep_file)
        try:
            raw = mne.io.read_raw(rep_file, preload=False, verbose='ERROR')
            rep_meta = {
                "file_path": rep_file,
                "ch_names": raw.ch_names[:20],  # list first 20 channels to avoid clutter
                "n_channels": len(raw.ch_names),
                "sfreq": raw.info['sfreq'],
                "highpass": raw.info.get('highpass', None),
                "lowpass": raw.info.get('lowpass', None),
                "annotations": []
            }
            if raw.annotations:
                rep_meta["annotations"] = list(set(raw.annotations.description))
            metadata["representative_metadata"] = rep_meta
        except Exception as e:
            logger.error("Failed to extract representative metadata from %s: %s", rep_file, e)
            metadata["representative_metadata"] = {"error": f"Failed to read file header: {str(e)}"}
    else:
        logger.warning("No raw EEG data files found in BIDS root.")
        metadata["representative_metadata"] = {"warning": "No raw EEG files (.fif, .set, .edf, .vhdr) found."}
        
    return json.dumps(metadata, indent=2)
