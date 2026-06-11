import mne
import json
import os
import logging
from langchain_core.tools import tool

logger = logging.getLogger("eeg_agent.tools.metadata_extractor")

@tool
def metadata_extractor(file_path: str) -> str:
    """
    Extracts metadata from a raw EEG file (e.g., .fif, .set, .edf).
    Returns a JSON string containing channel names, sampling frequency, and annotations (triggers).
    """
    logger.info("Starting metadata extraction for file: %s", file_path)
    if not os.path.exists(file_path):
        logger.error("File not found for metadata extraction: %s", file_path)
        return json.dumps({"error": f"File not found: {file_path}"})
    
    try:
        # Preload=False ensures we don't load data into memory, just headers
        raw = mne.io.read_raw(file_path, preload=False, verbose='ERROR')
        
        metadata = {
            "file_path": file_path,
            "ch_names": raw.ch_names,
            "n_channels": len(raw.ch_names),
            "sfreq": raw.info['sfreq'],
            "highpass": raw.info.get('highpass', None),
            "lowpass": raw.info.get('lowpass', None),
            "annotations": []
        }
        
        # Extract unique annotations (triggers)
        if raw.annotations:
            unique_desc = list(set(raw.annotations.description))
            metadata["annotations"] = unique_desc
            
        logger.info(
            "Metadata extracted successfully: %d channels, sfreq=%s.",
            metadata["n_channels"],
            metadata["sfreq"]
        )
        return json.dumps(metadata, indent=2)
        
    except Exception as e:
        logger.error("Failed to extract metadata from %s: %s", file_path, e)
        return json.dumps({"error": f"Failed to extract metadata: {str(e)}"})
