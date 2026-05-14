import os
import sys
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from src.graph.workflow import build_workflow
from src.tools.metadata_extractor import metadata_extractor

def main():
    load_dotenv()
    
    print("=== EEG-ADK Multi-Agent System ===")
    
    # The Docker container only has access to the mapped 'data' directory
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"\nNote: The Docker Sandbox only has access to files inside:\n{data_dir}")
    print("Please ensure your EEG data is placed there.")
    rel_path = input("\nEnter the filename or path relative to the data directory (e.g., 'sample.fif'): ").strip()
    
    data_path = os.path.join(data_dir, rel_path)
    
    if not os.path.exists(data_path):
        print(f"Error: Path '{data_path}' does not exist.")
        sys.exit(1)
        
    # We must pass the container-side path to the Planner so it writes correct code
    container_data_path = f"/mnt/data/{rel_path}"
        
    directive = input("Enter high-level analysis directive: ").strip()
    
    print("\n[System] Extracting metadata...")
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
    
    print("\n[System] Building Orchestration Graph...")
    app = build_workflow()
    
    config = {"configurable": {"thread_id": "1"}}
    
    print("\n[System] Invoking Planner Agent...")
    
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
        print("[System] Adding feedback. In a full app, this would route back to the planner.")
        # For simplicity, we just inject it into the plan or state here and proceed
        app.update_state(config, {"analysis_plan": f"USER FEEDBACK: {action}\n\n" + initial_state["analysis_plan"]})
        
    print("\n[System] Resuming execution (Executor -> Critic)...")
    
    final_state = None
    for event in app.stream(None, config=config):
        for k, v in event.items():
            print(f"\n[{k.upper()} COMPLETED]")
            if "critic_feedback" in v:
                print(f"Critic Feedback:\n{v['critic_feedback']}")
        final_state = event
        
    print("\n=== Workflow Completed ===")
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    os.makedirs(output_dir, exist_ok=True)
    
    state_data = app.get_state(config).values
    with open(os.path.join(output_dir, "final_report.md"), "w") as f:
        f.write("# Final Analysis Report\n\n")
        f.write("## Plan Executed\n")
        f.write(state_data.get("analysis_plan", ""))
        f.write("\n\n## Critic Feedback & Results\n")
        f.write(state_data.get("critic_feedback", ""))
        
    print(f"\nReport saved to {output_dir}/final_report.md")

if __name__ == "__main__":
    main()
