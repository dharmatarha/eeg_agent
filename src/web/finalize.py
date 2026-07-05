"""
Shared post-processing logic for completed EEG agent runs.

Extracts the finalization code from main.py into a reusable function
so both the CLI entry point and the web server can generate the same
output artifacts (final report, pipeline script, plots, run memory).
"""

import os
import base64
import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("eeg_agent.finalize")


def finalize_run(
    app,
    config: dict,
    thread_id: str,
    directive: str,
    output_base_dir: Optional[str] = None,
) -> dict:
    """
    Generate all output artifacts for a completed graph run.

    Args:
        app: The compiled LangGraph application (used to read final state).
        config: The LangGraph config dict (with thread_id in configurable).
        thread_id: The unique run identifier (e.g., "run_20260704_123456_abc12345").
        directive: The original user directive string.
        output_base_dir: Base directory for outputs. Defaults to <project_root>/output.

    Returns:
        A dict with:
        - output_dir: Absolute path to the run output directory.
        - report_path: Path to final_report.md.
        - pipeline_path: Path to analysis_pipeline.py (or None).
        - plot_files: List of saved plot filenames.
        - memory_path: Path to run_memory.json.
    """
    if output_base_dir is None:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        output_base_dir = os.path.join(project_root, "output")

    output_dir = os.path.join(output_base_dir, thread_id)
    os.makedirs(output_dir, exist_ok=True)

    state_data = app.get_state(config).values

    # --- Save Base64 plots to disk ---
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

    # --- Compile successful code blocks into analysis_pipeline.py ---
    executed_code_blocks = state_data.get("executed_code_blocks", [])
    successful_code = []

    for idx, block in enumerate(executed_code_blocks):
        if not block.get("error", False):
            code = block.get("code", "").strip()
            if code:
                successful_code.append(f"# --- Code Block {idx + 1} ---\n{code}\n")

    pipeline_path = None
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

    # --- Generate final_report.md ---
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
            f.write(
                "The following scientific findings and API references were "
                "retrieved during the session:\n\n"
            )
            for idx, item in enumerate(rag_history):
                f.write(f"### Query {idx + 1}: `{item.get('query', '')}`\n")
                f.write(f"- **Paradigm**: {item.get('paradigm', 'N/A')}\n")
                f.write(f"- **Target Collection**: {item.get('target', 'both')}\n\n")
                f.write(
                    "<details>\n<summary>Click to view retrieved reference text"
                    "</summary>\n\n"
                )
                f.write(item.get("results", "No results returned."))
                f.write("\n\n</details>\n\n")

        # Detailed code execution trace
        if executed_code_blocks:
            f.write("## 4. Code Execution Trace\n")
            f.write(
                "Below is the sequence of Python scripts executed inside "
                "the Docker Sandbox:\n\n"
            )
            for idx, block in enumerate(executed_code_blocks):
                status_str = (
                    "❌ Failed" if block.get("error", False) else "✅ Success"
                )
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

    # --- Save structured run memory ---
    memory_path = os.path.join(output_dir, "run_memory.json")
    try:
        try:
            raw_meta_parsed = json.loads(state_data.get("raw_metadata", "{}"))
        except Exception:
            raw_meta_parsed = state_data.get("raw_metadata", "")

        container_data_path = state_data.get("data_path", "")
        run_memory = {
            "thread_id": thread_id,
            "timestamp": datetime.now().isoformat(),
            "user_directive": directive,
            "data_path": container_data_path,
            "raw_metadata": raw_meta_parsed,
            "analysis_plan": state_data.get("analysis_plan", ""),
            "is_approved": state_data.get("is_approved", False),
            "error_count": state_data.get("error_count", 0),
            "critic_feedback": state_data.get("critic_feedback", ""),
            "artifacts": {
                "pipeline_script": (
                    f"output/{thread_id}/analysis_pipeline.py"
                    if successful_code
                    else None
                ),
                "report": f"output/{thread_id}/final_report.md",
                "plots": [f"output/{thread_id}/{fn}" for fn in plot_files],
            },
        }
        with open(memory_path, "w", encoding="utf-8") as f_mem:
            json.dump(run_memory, f_mem, indent=2)
        logger.info("Saved run memory to %s", memory_path)
    except Exception as e:
        logger.error("Failed to save run memory: %s", e)

    return {
        "output_dir": output_dir,
        "report_path": report_path,
        "pipeline_path": pipeline_path,
        "plot_files": plot_files,
        "memory_path": memory_path,
    }
