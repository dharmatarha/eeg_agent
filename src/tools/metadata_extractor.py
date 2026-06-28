"""
Metadata Extractor Utility for EEG-ADK Multi-Agent System

This module implements the `metadata_extractor` LangChain tool, which peeks into
EEG file headers (supporting .fif, .set, .edf, .bdf, .vhdr) using MNE-Python to extract
metadata (channel count/names, sfreq, filter settings, classification, events)
without preloading large binary arrays.
"""

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
    from src.utils.path_resolver import resolve_host_path
    resolved_file_path = resolve_host_path(file_path)
    
    if not os.path.exists(resolved_file_path):
        logger.error("File not found for metadata extraction: %s", resolved_file_path)
        return json.dumps({"error": f"File not found: {file_path}"})
    
    try:
        # Preload=False ensures we don't load data into memory, just headers
        raw = mne.io.read_raw(resolved_file_path, preload=False, verbose='ERROR')
        
        ch_types_dict = {}
        eeg_channels = []
        eog_channels = []
        stim_channels = []
        
        if hasattr(raw, "get_channel_types"):
            try:
                ch_types = raw.get_channel_types()
                for ch_type in set(ch_types):
                    ch_types_dict[ch_type] = ch_types.count(ch_type)
                
                eeg_channels = [name for name, ch_type in zip(raw.ch_names, ch_types) if ch_type == 'eeg']
                eog_channels = [name for name, ch_type in zip(raw.ch_names, ch_types) if ch_type == 'eog']
                stim_channels = [name for name, ch_type in zip(raw.ch_names, ch_types) if ch_type == 'stim']
            except Exception as e:
                logger.debug("Failed to get channel types: %s", e)

        # Detect potential stim channels from names if they aren't typed as stim in headers
        common_stim_names = ['Status', 'STI 014', 'STI014', 'TRIG', 'trigger', 'TRIGGER']
        potential_stim = [name for name in raw.ch_names if name in common_stim_names]
        stim_channels = list(set(stim_channels + potential_stim))
        
        # Extract events via digital trigger channels if present
        events_dict = {}
        if stim_channels:
            try:
                import numpy as np
                for stim_ch in stim_channels:
                    evs = mne.find_events(raw, stim_channel=stim_ch, verbose='ERROR', min_duration=0.002)
                    if evs is not None and len(evs) > 0:
                        unique_ids, counts = np.unique(evs[:, 2], return_counts=True)
                        for uid, cnt in zip(unique_ids, counts):
                            events_dict[str(uid)] = int(cnt)
                        break
            except Exception as event_err:
                logger.debug("Failed to extract digital events: %s", event_err)

        # Extract unique annotations (triggers)
        unique_desc = []
        if raw.annotations:
            try:
                unique_desc = list(set(raw.annotations.description))
            except Exception as e:
                logger.debug("Failed to parse annotations description: %s", e)
            
        # Handle BrainVision .vmrk sidecar parsing if MNE annotations are empty
        if resolved_file_path.lower().endswith((".vhdr", ".vmrk")) and not unique_desc and not events_dict:
            vmrk_path = resolved_file_path if resolved_file_path.lower().endswith(".vmrk") else resolved_file_path[:-5] + ".vmrk"
            if os.path.exists(vmrk_path):
                try:
                    markers = []
                    with open(vmrk_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if line.startswith("Mk") and "=" in line:
                                parts = line.split("=", 1)[1].split(",")
                                if len(parts) >= 2:
                                    markers.append(parts[1].strip())
                    if markers:
                        unique_desc = list(set(markers))
                except Exception as vmrk_err:
                    logger.warning("Failed to parse sidecar vmrk file: %s", vmrk_err)

        metadata = {
            "file_path": file_path,
            "file_format": os.path.splitext(file_path)[1].upper(),
            "ch_names": raw.ch_names,
            "n_channels": len(raw.ch_names),
            "sfreq": raw.info.get('sfreq', None),
            "highpass": raw.info.get('highpass', None),
            "lowpass": raw.info.get('lowpass', None),
            "channel_types": ch_types_dict,
            "eeg_channels": eeg_channels[:20],
            "eog_channels": eog_channels,
            "stim_channels": stim_channels,
            "annotations": unique_desc,
            "digital_triggers": events_dict
        }
            
        logger.info(
            "Metadata extracted successfully: %d channels, sfreq=%s.",
            metadata["n_channels"],
            metadata["sfreq"]
        )
        return json.dumps(metadata, indent=2)
        
    except Exception as e:
        logger.error("Failed to extract metadata from %s: %s", file_path, e)
        return json.dumps({"error": f"Failed to extract metadata: {str(e)}"})
