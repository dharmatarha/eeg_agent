"""
Dataset Explorer Utility for EEG-ADK Multi-Agent System

This module implements the `dataset_explorer` LangChain tool, which allows
autonomous agents to discover dataset file structures, read documentation/sidecar files,
and verify header consistency (channel counts, names, sampling rates) across EEG files.
"""

import os
import json
import fnmatch
import mne
import logging
from langchain_core.tools import tool

logger = logging.getLogger("eeg_agent.tools.dataset_explorer")

@tool
def dataset_explorer(action: str, path: str, pattern: str = None, max_bytes: int = 8192) -> str:
    """
    Explore dataset directories, read metadata/README files, and verify dataset consistency.

    Parameters:
      - action (str): One of:
          * 'list': List files recursively in `path`, filtered by `pattern` (e.g. '*.fif', '*README*').
          * 'read': Read the contents of a text file (e.g. README, JSON, TSV, CSV) up to `max_bytes`.
          * 'verify_consistency': Scan MNE headers of all EEG files matching `pattern` in `path`
                                  and check if they share consistent channel counts, names, and sfreq.
      - path (str): Target directory or file path.
      - pattern (str, optional): Glob pattern used for filtering lists or selecting files to verify.
      - max_bytes (int, optional): Maximum characters/bytes to return when reading a text file (default: 8192).
    """
    logger.info("Dataset explorer running action: %s on path: %s", action, path)
    from src.utils.path_resolver import resolve_host_path
    resolved_path = resolve_host_path(path)
    
    if not os.path.exists(resolved_path):
        logger.error("Path does not exist: %s", resolved_path)
        return json.dumps({"error": f"Path does not exist: {path}"})

    if action == "list":
        if not os.path.isdir(resolved_path):
            logger.error("Path is not a directory: %s", resolved_path)
            return json.dumps({"error": f"Path is not a valid directory: {path}"})
        
        pattern = pattern or "*"
        matching_files = []
        
        for root, _, files in os.walk(resolved_path):
            for f in files:
                rel_path = os.path.relpath(os.path.join(root, f), resolved_path)
                if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(rel_path, pattern):
                    matching_files.append(rel_path)
                    
        matching_files = sorted(matching_files)
        total_matches = len(matching_files)
        
        logger.info("Found %d matching files under %s for pattern: %s", total_matches, resolved_path, pattern)
        return json.dumps({
            "action": "list",
            "directory": path,
            "pattern": pattern,
            "total_matches": total_matches,
            "files": matching_files[:100],  # Limit output size to prevent context bloat
            "truncated": total_matches > 100
        }, indent=2)

    elif action == "read":
        if not os.path.isfile(resolved_path):
            logger.error("Path is not a file: %s", resolved_path)
            return json.dumps({"error": f"Path is not a valid file: {path}"})
            
        try:
            with open(resolved_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(max_bytes)
            
            file_size = os.path.getsize(resolved_path)
            logger.info("Successfully read text file %s (%d bytes).", resolved_path, len(content))
            return json.dumps({
                "action": "read",
                "file_path": path,
                "size_bytes": file_size,
                "read_bytes": len(content),
                "content": content,
                "truncated": file_size > len(content)
            }, indent=2)
        except Exception as e:
            logger.error("Failed to read text file %s: %s", resolved_path, e)
            return json.dumps({"error": f"Failed to read file: {str(e)}"})

    elif action == "verify_consistency":
        if not os.path.isdir(resolved_path):
            logger.error("Path is not a directory: %s", resolved_path)
            return json.dumps({"error": f"Path is not a valid directory: {path}"})
            
        eeg_files = []
        extensions = (".fif", ".set", ".edf", ".vhdr", ".bdf")
        
        for root, _, files in os.walk(resolved_path):
            # Skip derivatives and temporary directories relative to the search path
            rel_root = os.path.relpath(root, resolved_path)
            parts = rel_root.split(os.sep)
            if "derivatives" in parts or "tmp" in parts:
                continue
            for f in files:
                if pattern:
                    if fnmatch.fnmatch(f, pattern):
                        eeg_files.append(os.path.join(root, f))
                else:
                    if f.endswith(extensions):
                        eeg_files.append(os.path.join(root, f))
                        
        eeg_files = sorted(eeg_files)
        if not eeg_files:
            logger.warning("No EEG data files found in %s.", resolved_path)
            return json.dumps({"warning": f"No EEG files found in '{path}' matching pattern '{pattern or '*'}'"})
            
        # Limit checking to first 50 files for performance
        files_to_check = eeg_files[:50]
        results = {}
        errors = {}
        
        logger.info("Verifying header consistency for %d files in %s...", len(files_to_check), resolved_path)
        for f_path in files_to_check:
            rel_name = os.path.relpath(f_path, resolved_path)
            try:
                # read with preload=False is fast and memory safe
                raw = mne.io.read_raw(f_path, preload=False, verbose='ERROR')
                results[rel_name] = {
                    "sfreq": raw.info['sfreq'],
                    "n_channels": len(raw.ch_names),
                    "ch_names": sorted(raw.ch_names)
                }
            except Exception as e:
                errors[rel_name] = str(e)
                
        if not results:
            logger.error("Failed to extract headers from any EEG files.")
            return json.dumps({
                "error": "Failed to read headers from any detected files.",
                "details": errors
            }, indent=2)
            
        reference_file = list(results.keys())[0]
        ref_meta = results[reference_file]
        
        inconsistent_sfreq = {}
        inconsistent_channels_count = {}
        inconsistent_channels_names = {}
        
        for f_name, meta in results.items():
            if meta["sfreq"] != ref_meta["sfreq"]:
                inconsistent_sfreq[f_name] = meta["sfreq"]
            if meta["n_channels"] != ref_meta["n_channels"]:
                inconsistent_channels_count[f_name] = meta["n_channels"]
            if meta["ch_names"] != ref_meta["ch_names"]:
                extra = list(set(meta["ch_names"]) - set(ref_meta["ch_names"]))
                missing = list(set(ref_meta["ch_names"]) - set(meta["ch_names"]))
                inconsistent_channels_names[f_name] = {
                    "extra_channels": extra,
                    "missing_channels": missing
                }
                
        is_consistent = (
            len(inconsistent_sfreq) == 0 and
            len(inconsistent_channels_count) == 0 and
            len(inconsistent_channels_names) == 0
        )
        
        summary = {
            "action": "verify_consistency",
            "directory": path,
            "total_files_found": len(eeg_files),
            "files_checked": len(files_to_check),
            "is_consistent": is_consistent,
            "reference_file": reference_file,
            "reference_metadata": {
                "sfreq": ref_meta["sfreq"],
                "n_channels": ref_meta["n_channels"]
            }
        }
        
        if not is_consistent:
            summary["discrepancies"] = {}
            if inconsistent_sfreq:
                summary["discrepancies"]["sfreq"] = {
                    "reference": ref_meta["sfreq"],
                    "mismatches": inconsistent_sfreq
                }
            if inconsistent_channels_count:
                summary["discrepancies"]["channel_count"] = {
                    "reference": ref_meta["n_channels"],
                    "mismatches": inconsistent_channels_count
                }
            if inconsistent_channels_names:
                summary["discrepancies"]["channel_names"] = {
                    "reference_file": reference_file,
                    "mismatches": inconsistent_channels_names
                }
                
        if errors:
            summary["failed_to_read"] = errors
            
        if len(eeg_files) > 50:
            summary["note"] = "Only the first 50 files were checked for consistency."
            
        logger.info("Consistency check completed. Consistent=%s.", is_consistent)
        return json.dumps(summary, indent=2)

    else:
        logger.error("Unknown action passed to dataset_explorer: %s", action)
        return json.dumps({"error": f"Unknown action: {action}"})
