import os
import sys
import logging
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from src.graph.workflow import build_workflow
from src.tools.metadata_extractor import metadata_extractor
from src.utils.logging_config import setup_logging

logger = logging.getLogger("eeg_agent.main")

def main():
    load_dotenv(override=True)
    setup_logging()
    
    print("=== EEG-ADK Multi-Agent System ===")
    
    # The Docker container only has access to the mapped 'data' directory (configured dynamically via EEG_DATA_DIR)
    data_dir = os.environ.get("EEG_DATA_DIR")
    if data_dir:
        data_dir = os.path.abspath(data_dir)
    else:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"\nNote: The Docker Sandbox only has access to files inside:\n{data_dir}")
    print("Please ensure your EEG data is placed there.")
    rel_path = input("\nEnter the filename or folder path relative to the data directory (e.g., 'sample.fif' or 'bids_dataset'): ").strip()
    
    data_path = os.path.join(data_dir, rel_path)
    
    if not os.path.exists(data_path):
        logger.error("Path '%s' does not exist.", data_path)
        sys.exit(1)
        
    # We must pass the container-side path to the Planner so it writes correct code
    container_data_path = f"/mnt/data/{rel_path}"
        
    directive = input("Enter high-level analysis directive (or path to a text/md file): ").strip()
    expanded_path = os.path.abspath(os.path.expanduser(directive))
    if os.path.isfile(expanded_path):
        logger.info("Reading analysis directive from file: %s", expanded_path)
        try:
            with open(expanded_path, "r", encoding="utf-8") as f:
                directive = f.read().strip()
        except Exception as e:
            logger.error("Failed to read directive file: %s", e)
            sys.exit(1)
            
    ref_run_id = input("\nEnter a previous Run ID (Thread ID) to reference (optional, press ENTER to skip): ").strip()
    reference_run_memory = None
    if ref_run_id:
        ref_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output", ref_run_id))
        ref_memory_path = os.path.join(ref_dir, "run_memory.json")
        if os.path.exists(ref_memory_path):
            try:
                with open(ref_memory_path, "r", encoding="utf-8") as f_ref:
                    reference_run_memory = json.load(f_ref)
                logger.info("Loaded reference memory from %s", ref_memory_path)
                print(f"Successfully loaded memory from reference run: {ref_run_id}")
            except Exception as e:
                logger.error("Failed to load reference run memory: %s", e)
                print(f"Error loading reference run memory: {e}")
                sys.exit(1)
        else:
            logger.error("Reference run memory file not found at %s", ref_memory_path)
            print(f"Reference run memory file not found at {ref_memory_path}")
            sys.exit(1)
    
    is_bids = False
    if os.path.isdir(data_path):
        desc_exists = os.path.exists(os.path.join(data_path, "dataset_description.json"))
        has_sub_dirs = any(d.startswith("sub-") for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)))
        if desc_exists or has_sub_dirs:
            is_bids = True

    if is_bids:
        logger.info("BIDS dataset directory detected at %s. Invoking bids_inspector...", data_path)
        from src.tools.bids_inspector import bids_inspector
        raw_metadata = bids_inspector.invoke({"bids_root": data_path})
    elif os.path.isdir(data_path):
        logger.info("Directory detected at %s. Scanning contents...", data_path)
        files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
        rep_file = next((f for f in files if f.endswith((".fif", ".set", ".edf", ".vhdr", ".bdf"))), None)
        if rep_file:
            logger.info("Extracting representative metadata from %s...", rep_file)
            rep_meta_str = metadata_extractor.invoke({"file_path": os.path.join(data_path, rep_file)})
            raw_metadata = json.dumps({
                "directory_path": container_data_path,
                "files": files,
                "representative_file": rep_file,
                "representative_metadata": json.loads(rep_meta_str)
            }, indent=2)
        else:
            raw_metadata = json.dumps({
                "directory_path": container_data_path,
                "files": files,
                "warning": "No readable EEG files found in the directory."
            }, indent=2)
    else:
        logger.info("Extracting metadata for %s...", data_path)
        raw_metadata = metadata_extractor.invoke({"file_path": data_path})
    
    initial_state = {
        "user_directive": directive,
        "data_path": container_data_path,
        "raw_metadata": raw_metadata,
        "reference_run": reference_run_memory,
        "analysis_plan": "",
        "execution_logs": [],
        "generated_plots": [],
        "error_count": 0,
        "critic_feedback": "",
        "is_approved": False,
        "rag_history": [],
        "executed_code_blocks": []
    }
    
    logger.info("Building Orchestration Graph...")
    app = build_workflow()
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_uuid = str(uuid.uuid4())[:8]
    thread_id = f"run_{run_timestamp}_{run_uuid}"
    config = {"configurable": {"thread_id": thread_id}}
    logger.info("Initialized session with unique Thread ID: %s", thread_id)
    print(f"\nInitialized session with unique Thread ID: {thread_id}")
    print(f"To inspect the session history later: python scripts/inspect_run.py -t {thread_id}\n")
    
    logger.info("Invoking Planner Agent...")
    
    # Stream until interrupt
    for event in app.stream(initial_state, config=config):
        for k, v in event.items():
            if k == "planner":
                print("\n" + "="*40)
                print("         PROPOSED ANALYSIS PLAN")
                print("="*40)
                print(v.get("analysis_plan", "No plan generated."))
                print("="*40 + "\n")
    
    # State is now paused before 'executor'
    print("[HITL] State execution paused.")
    print("Review the plan above.")
    action = input("Press ENTER to approve and execute, or type feedback to adjust: ").strip()
    
    if action:
        logger.info("Adding user feedback to plan.")
        current_state = app.get_state(config)
        current_plan = current_state.values.get("analysis_plan", "")
        app.update_state(config, {"analysis_plan": f"USER FEEDBACK: {action}\n\n" + current_plan})
        
    logger.info("Resuming execution (Executor -> Critic)...")
    
    final_state = None
    for event in app.stream(None, config=config):
        for k, v in event.items():
            if k == "executor":
                print("\n" + "="*40)
                print("         EXECUTOR ACTION")
                print("="*40)
                code_blocks = v.get("executed_code_blocks", [])
                plots = v.get("generated_plots", [])
                
                print(f"Executed {len(code_blocks)} code block(s) in sandbox.")
                for idx, block in enumerate(code_blocks):
                    status = "❌ Failed" if block.get("error", False) else "✅ Succeeded"
                    print(f"\n--- Code Block {idx + 1} ({status}) ---")
                    print(block.get("code", "").strip())
                    if block.get("error", False):
                        print(f"\nError logs:\n{block.get('logs', '').strip()}")
                print(f"\nGenerated {len(plots)} plot(s).")
                print("="*40 + "\n")
                
            elif k == "critic":
                print("\n" + "="*40)
                print("         CRITIC REVIEW")
                print("="*40)
                feedback = v.get("critic_feedback", "")
                is_approved = "APPROVE" in feedback.upper()
                verdict = "✅ APPROVED" if is_approved else "❌ REJECTED"
                print(f"Verdict: {verdict}")
                print(f"\nFeedback:\n{feedback}")
                print("="*40 + "\n")
                
        final_state = event
        
    print("\n=== Workflow Completed ===")
    
    from src.web.finalize import finalize_run
    result = finalize_run(app, config, thread_id, directive)
    
    logger.info("All output artifacts saved to %s", result["output_dir"])
    print(f"Output saved to: {result['output_dir']}")
    if result["pipeline_path"]:
        print(f"  Pipeline script: {result['pipeline_path']}")
    print(f"  Report: {result['report_path']}")
    print(f"  Plots: {len(result['plot_files'])} saved")
    print(f"  Run memory: {result['memory_path']}")

if __name__ == "__main__":
    main()
