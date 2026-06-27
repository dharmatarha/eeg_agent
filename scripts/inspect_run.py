#!/usr/bin/env python3
import os
import sys
import sqlite3
import argparse
from langgraph.checkpoint.sqlite import SqliteSaver

def main():
    parser = argparse.ArgumentParser(description="Inspect EEG-ADK Multi-Agent run history and checkpoints.")
    parser.add_argument("-t", "--thread-id", help="The Thread ID of the run to inspect.")
    parser.add_argument("-l", "--list", action="store_true", help="List all available runs in the database.")
    parser.add_argument("-c", "--show-code", action="store_true", help="Print the executed Python code blocks and logs.")
    parser.add_argument("-p", "--show-plan", action="store_true", help="Print the generated Analysis Plan.")
    parser.add_argument("-d", "--db-path", default="logs/checkpoints.sqlite", help="Path to the checkpoints SQLite database.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db_path):
        print(f"Error: Database file not found at '{args.db_path}'")
        sys.exit(1)
        
    conn = sqlite3.connect(args.db_path)
    saver = SqliteSaver(conn)
    
    # Fetch all thread IDs
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        thread_ids = [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError as e:
        print(f"Error querying database: {e}")
        sys.exit(1)
        
    if not thread_ids:
        print("No runs found in the checkpoints database.")
        sys.exit(0)
        
    # Sort thread_ids. Our format is run_YYYYMMDD_HHMMSS_uuid, which sorts chronologically.
    # We sort in descending order (most recent first).
    thread_ids.sort(reverse=True)
    
    # Resolve thread ID if not provided
    selected_thread_id = args.thread_id
    
    if not selected_thread_id or args.list:
        print("\n=== Available Runs (Most Recent First) ===")
        runs = []
        for idx, tid in enumerate(thread_ids):
            config = {"configurable": {"thread_id": tid}}
            checkpoint_tuple = saver.get(config)
            directive = "N/A"
            if checkpoint_tuple:
                channel_values = checkpoint_tuple.get("channel_values", {})
                directive = channel_values.get("user_directive", "N/A")
                # Clean up if it's too long
                if len(directive) > 60:
                    directive = directive[:57] + "..."
            runs.append((tid, directive))
            print(f"[{idx + 1}] Thread: {tid}")
            print(f"    Directive: {directive}")
            print("-" * 50)
            
        if args.list:
            sys.exit(0)
            
        # If user didn't specify a thread-id, prompt them to choose or default to the most recent
        user_choice = input(f"\nSelect a run number (1-{len(thread_ids)}) [default: 1]: ").strip()
        if not user_choice:
            selected_thread_id = thread_ids[0]
        else:
            try:
                choice_idx = int(user_choice) - 1
                if 0 <= choice_idx < len(thread_ids):
                    selected_thread_id = thread_ids[choice_idx]
                else:
                    print("Invalid run number. Defaulting to most recent.")
                    selected_thread_id = thread_ids[0]
            except ValueError:
                print("Invalid input. Defaulting to most recent.")
                selected_thread_id = thread_ids[0]
                
    # Inspect the selected thread
    config = {"configurable": {"thread_id": selected_thread_id}}
    checkpoint_tuple = saver.get(config)
    
    if not checkpoint_tuple:
        print(f"Error: No checkpoint data found for thread ID '{selected_thread_id}'")
        sys.exit(1)
        
    channel_values = checkpoint_tuple.get("channel_values", {})
    
    print("\n" + "=" * 60)
    print(f"         RUN OVERVIEW: {selected_thread_id}")
    print("=" * 60)
    
    print(f"\n* User Directive:")
    print(f"  {channel_values.get('user_directive', 'N/A')}")
    
    print(f"\n* Data Path:")
    print(f"  {channel_values.get('data_path', 'N/A')}")
    
    # Format metadata representation
    raw_meta = channel_values.get("raw_metadata", "{}")
    print(f"\n* Raw Metadata:")
    print(f"  {raw_meta.strip()}")
    
    # Analysis Plan Summary
    plan = channel_values.get("analysis_plan", "")
    plan_status = "Generated" if plan else "Not generated yet"
    print(f"\n* Analysis Plan ({plan_status}):")
    if plan:
        if args.show_plan:
            print("-" * 40)
            print(plan.strip())
            print("-" * 40)
        else:
            # Print preview (first few lines/chars)
            lines = plan.strip().split("\n")
            preview = "\n".join(lines[:5])
            print("-" * 40)
            print(preview)
            if len(lines) > 5:
                print(f"... ({len(lines) - 5} more lines. Use -p or --show-plan to see the full plan) ...")
            print("-" * 40)
            
    # Executed Code Blocks Summary
    code_blocks = channel_values.get("executed_code_blocks", [])
    print(f"\n* Code Execution Summary:")
    print(f"  - Total blocks executed: {len(code_blocks)}")
    
    succeeded = sum(1 for b in code_blocks if not b.get("error", False))
    failed = sum(1 for b in code_blocks if b.get("error", False))
    print(f"  - Succeeded blocks: {succeeded}")
    print(f"  - Failed blocks: {failed}")
    
    if code_blocks and (args.show_code or input("\nShow executed code blocks and outputs? (y/n) [n]: ").strip().lower() == "y"):
        for idx, block in enumerate(code_blocks):
            status = "❌ Failed" if block.get("error", False) else "✅ Succeeded"
            print(f"\n--- Code Block {idx + 1} ({status}) ---")
            print(block.get("code", "").strip())
            if block.get("error", False) or block.get("logs", "").strip():
                print(f"\nLogs/Errors:")
                print(block.get("logs", "").strip())
            print("-" * 40)
            
    # Generated Plots
    plots = channel_values.get("generated_plots", [])
    print(f"\n* Generated Plots: {len(plots)}")
    
    # Critic feedback
    feedback = channel_values.get("critic_feedback", "")
    approved = channel_values.get("is_approved", False)
    verdict = "✅ APPROVED" if approved else "❌ REJECTED"
    print(f"\n* Critic Verdict: {verdict}")
    if feedback:
        print(f"  Feedback:")
        feedback_lines = feedback.strip().split("\n")
        preview_feedback = "\n".join(feedback_lines[:5])
        print("  " + preview_feedback.replace("\n", "\n  "))
        if len(feedback_lines) > 5:
            print(f"  ... ({len(feedback_lines) - 5} more lines of feedback) ...")
            
    print(f"\n* Error Count: {channel_values.get('error_count', 0)}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
