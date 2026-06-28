import os
import sys
import logging
import base64
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph
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
    if os.path.isfile(directive):
        logger.info("Reading analysis directive from file: %s", directive)
        try:
            with open(directive, "r", encoding="utf-8") as f:
                directive = f.read().strip()
        except Exception as e:
            logger.error("Failed to read directive file: %s", e)
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
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    os.makedirs(output_dir, exist_ok=True)
    
    state_data = app.get_state(config).values
    
    # Save base64 generated plots to disk
    generated_plots = state_data.get("generated_plots", [])
    plot_files = []
    for idx, img_b64 in enumerate(generated_plots):
        try:
            img_data = base64.b64decode(img_b64)
            filename = f"plot_{idx + 1}.png"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "wb") as f_img:
                f_img.write(img_data)
            plot_files.append(filename)
            logger.info("Saved generated plot: %s", filepath)
        except Exception as e:
            logger.error("Failed to save plot %d: %s", idx, e)
            
    # 1. Compile successful code blocks into output/analysis_pipeline.py
    executed_code_blocks = state_data.get("executed_code_blocks", [])
    successful_code = []
    
    for idx, block in enumerate(executed_code_blocks):
        if not block.get("error", False):
            code = block.get("code", "").strip()
            if code:
                successful_code.append(f"# --- Code Block {idx + 1} ---\n{code}\n")
                
    if successful_code:
        pipeline_path = os.path.join(output_dir, "analysis_pipeline.py")
        with open(pipeline_path, "w") as f_py:
            f_py.write("#!/usr/bin/env python\n")
            f_py.write('"""\nGenerated EEG Analysis Pipeline\n')
            f_py.write(f"Session Thread ID: {thread_id}\n")
            f_py.write(f"User Directive: {directive}\n")
            f_py.write('"""\n\n')
            f_py.write("import mne\n")
            f_py.write("mne.set_config('MNE_MEMMAP_MIN_SIZE', '10M')\n\n")
            f_py.write("\n".join(successful_code))
        logger.info("Saved successful analysis pipeline script to %s", pipeline_path)

    report_path = os.path.join(output_dir, "final_report.md")
    with open(report_path, "w") as f:
        f.write(f"# Final Analysis Report - Session `{thread_id}`\n\n")
        
        f.write("## 1. User Directive & Raw Metadata\n")
        f.write(f"**Directive:** {directive}\n\n")
        f.write("### Extracted Metadata:\n")
        f.write("```json\n")
        f.write(state_data.get("raw_metadata", "{}"))
        f.write("\n```\n\n")
        
        f.write("## 2. Plan Executed\n")
        f.write(state_data.get("analysis_plan", "No plan generated."))
        f.write("\n\n")
        
        # RAG retrieval audit log
        rag_history = state_data.get("rag_history", [])
        if rag_history:
            f.write("## 3. RAG Retrieval Audit Log\n")
            f.write("The following scientific findings and API references were retrieved during the session:\n\n")
            for idx, item in enumerate(rag_history):
                f.write(f"### Query {idx + 1}: `{item.get('query', '')}`\n")
                f.write(f"- **Paradigm**: {item.get('paradigm', 'N/A')}\n")
                f.write(f"- **Target Collection**: {item.get('target', 'both')}\n\n")
                f.write("<details>\n<summary>Click to view retrieved reference text</summary>\n\n")
                f.write(item.get("results", "No results returned."))
                f.write("\n\n</details>\n\n")
        
        # Detailed code execution trace
        if executed_code_blocks:
            f.write("## 4. Code Execution Trace\n")
            f.write("Below is the sequence of Python scripts executed inside the Docker Sandbox:\n\n")
            for idx, block in enumerate(executed_code_blocks):
                status_str = "❌ Failed" if block.get("error", False) else "✅ Success"
                f.write(f"### Block {idx + 1} ({status_str})\n")
                f.write("```python\n")
                f.write(block.get("code", ""))
                f.write("\n```\n")
                f.write("#### Output logs:\n")
                f.write("```\n")
                f.write(block.get("logs", "").strip() or "No output.")
                f.write("\n```\n\n")
                
        f.write("## 5. Critic Feedback & Quality Assurance\n")
        f.write(state_data.get("critic_feedback", "No critic feedback registered."))
        
        if plot_files:
            f.write("\n\n## 6. Visual Artifacts\n")
            for filename in plot_files:
                f.write(f"![{filename}]({filename})\n\n")
        
    logger.info("Report saved to %s", report_path)

if __name__ == "__main__":
    main()
