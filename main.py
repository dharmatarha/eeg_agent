import os
import sys
import logging
import base64
import json
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
    
    # The Docker container only has access to the mapped 'data' directory
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
        
    directive = input("Enter high-level analysis directive: ").strip()
    
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
        "is_approved": False
    }
    
    logger.info("Building Orchestration Graph...")
    app = build_workflow()
    
    config = {"configurable": {"thread_id": "1"}}
    
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
        # For simplicity, we just inject it into the plan or state here and proceed
        app.update_state(config, {"analysis_plan": f"USER FEEDBACK: {action}\n\n" + initial_state["analysis_plan"]})
        
    logger.info("Resuming execution (Executor -> Critic)...")
    
    final_state = None
    for event in app.stream(None, config=config):
        for k, v in event.items():
            logger.info("Node completed: %s", k.upper())
            if "critic_feedback" in v:
                print(f"\nCritic Feedback:\n{v['critic_feedback']}")
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
            
    report_path = os.path.join(output_dir, "final_report.md")
    with open(report_path, "w") as f:
        f.write("# Final Analysis Report\n\n")
        f.write("## Plan Executed\n")
        f.write(state_data.get("analysis_plan", ""))
        f.write("\n\n## Critic Feedback & Results\n")
        f.write(state_data.get("critic_feedback", ""))
        
        if plot_files:
            f.write("\n\n## Visual Artifacts\n")
            for filename in plot_files:
                f.write(f"![{filename}]({filename})\n\n")
        
    logger.info("Report saved to %s", report_path)

if __name__ == "__main__":
    main()
