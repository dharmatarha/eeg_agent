"""
FastAPI bridge server for the EEG-ADK Multi-Agent System.

Wraps the existing LangGraph workflow with HTTP + WebSocket endpoints
so the assistant-ui React frontend can drive the analysis pipeline.

Usage:
    uvicorn src.web.server:app --host 127.0.0.1 --port 8000

The server binds to localhost only (no auth required).
Supports one active run at a time (single-session design).
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.graph.workflow import build_workflow
from src.tools.metadata_extractor import metadata_extractor
from src.utils.logging_config import setup_logging
from src.web.finalize import finalize_run

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

load_dotenv(override=True)
setup_logging()

logger = logging.getLogger("eeg_agent.web")

app = FastAPI(
    title="EEG-ADK Web Bridge",
    description="Bridge server connecting the assistant-ui frontend to the EEG multi-agent graph.",
    version="0.1.0",
)

# CORS: allow the Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.environ.get("EEG_DATA_DIR") or os.path.join(PROJECT_ROOT, "data")
DATA_DIR = os.path.abspath(DATA_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# EEG file extensions we surface in the browser
EEG_EXTENSIONS = {".fif", ".set", ".edf", ".bdf", ".vhdr"}

# ---------------------------------------------------------------------------
# Single-session state
# ---------------------------------------------------------------------------

active_run: Optional[dict] = None  # Holds run metadata while a graph is executing


class RunPhase(str, Enum):
    PLANNING = "planner"
    AWAITING_HITL = "awaiting_hitl"
    EXECUTING = "executor"
    REVIEWING = "critic"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CreateRunRequest(BaseModel):
    data_path: str = Field(
        ..., description="Path relative to the data/ directory (e.g. 'sample.fif')"
    )
    directive: str = Field(
        ..., description="High-level analysis directive"
    )
    reference_run_id: Optional[str] = Field(
        None, description="Thread ID of a past run to use as reference"
    )


class CreateRunResponse(BaseModel):
    run_id: str
    data_path: str
    container_data_path: str
    is_bids: bool


class RunSummary(BaseModel):
    run_id: str
    timestamp: str
    directive: str
    is_approved: Optional[bool] = None


class FileTreeNode(BaseModel):
    name: str
    path: str  # relative to data/
    type: str  # "file" or "directory"
    size: Optional[int] = None  # bytes, for files
    is_bids: bool = False
    children: Optional[list] = None  # for directories


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _build_file_tree(base_path: str, rel_prefix: str = "") -> list[dict]:
    """Recursively build a file tree from base_path, filtering to EEG-relevant items."""
    entries = []
    try:
        items = sorted(os.listdir(base_path))
    except PermissionError:
        return entries

    for item in items:
        # Skip hidden files/dirs
        if item.startswith("."):
            continue

        full_path = os.path.join(base_path, item)
        rel_path = os.path.join(rel_prefix, item) if rel_prefix else item

        if os.path.isdir(full_path):
            children = _build_file_tree(full_path, rel_path)
            # Check BIDS markers
            is_bids = os.path.exists(
                os.path.join(full_path, "dataset_description.json")
            ) or any(
                d.startswith("sub-")
                for d in os.listdir(full_path)
                if os.path.isdir(os.path.join(full_path, d))
            )
            entries.append(
                {
                    "name": item,
                    "path": rel_path,
                    "type": "directory",
                    "is_bids": is_bids,
                    "children": children,
                }
            )
        elif os.path.isfile(full_path):
            _, ext = os.path.splitext(item)
            if ext.lower() in EEG_EXTENSIONS:
                entries.append(
                    {
                        "name": item,
                        "path": rel_path,
                        "type": "file",
                        "size": os.path.getsize(full_path),
                    }
                )

    return entries


def _load_reference_memory(ref_run_id: str) -> Optional[dict]:
    """Load run_memory.json for a previous run, or None if not found."""
    ref_dir = os.path.join(OUTPUT_DIR, ref_run_id)
    ref_memory_path = os.path.join(ref_dir, "run_memory.json")
    if not os.path.exists(ref_memory_path):
        return None
    with open(ref_memory_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_metadata(data_path: str, container_data_path: str) -> tuple[str, bool]:
    """
    Extract metadata from the data path, mirroring main.py's pre-flight logic.
    Returns (raw_metadata_json_string, is_bids).
    """
    is_bids = False

    if os.path.isdir(data_path):
        desc_exists = os.path.exists(
            os.path.join(data_path, "dataset_description.json")
        )
        has_sub_dirs = any(
            d.startswith("sub-")
            for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        )
        if desc_exists or has_sub_dirs:
            is_bids = True

    if is_bids:
        logger.info("BIDS dataset detected at %s. Invoking bids_inspector...", data_path)
        from src.tools.bids_inspector import bids_inspector

        raw_metadata = bids_inspector.invoke({"bids_root": data_path})
    elif os.path.isdir(data_path):
        logger.info("Directory detected at %s. Scanning contents...", data_path)
        files = [
            f
            for f in os.listdir(data_path)
            if os.path.isfile(os.path.join(data_path, f))
        ]
        rep_file = next(
            (f for f in files if f.endswith((".fif", ".set", ".edf", ".vhdr", ".bdf"))),
            None,
        )
        if rep_file:
            logger.info("Extracting representative metadata from %s...", rep_file)
            rep_meta_str = metadata_extractor.invoke(
                {"file_path": os.path.join(data_path, rep_file)}
            )
            raw_metadata = json.dumps(
                {
                    "directory_path": container_data_path,
                    "files": files,
                    "representative_file": rep_file,
                    "representative_metadata": json.loads(rep_meta_str),
                },
                indent=2,
            )
        else:
            raw_metadata = json.dumps(
                {
                    "directory_path": container_data_path,
                    "files": files,
                    "warning": "No readable EEG files found in the directory.",
                },
                indent=2,
            )
    else:
        logger.info("Extracting metadata for %s...", data_path)
        raw_metadata = metadata_extractor.invoke({"file_path": data_path})

    return raw_metadata, is_bids


def _list_past_runs() -> list[dict]:
    """List past runs from the output/ directory by reading their run_memory.json."""
    runs = []
    if not os.path.isdir(OUTPUT_DIR):
        return runs

    for entry in sorted(os.listdir(OUTPUT_DIR), reverse=True):
        run_dir = os.path.join(OUTPUT_DIR, entry)
        memory_path = os.path.join(run_dir, "run_memory.json")
        if os.path.isdir(run_dir) and os.path.exists(memory_path):
            try:
                with open(memory_path, "r", encoding="utf-8") as f:
                    mem = json.load(f)
                runs.append(
                    {
                        "run_id": mem.get("thread_id", entry),
                        "timestamp": mem.get("timestamp", ""),
                        "directive": mem.get("user_directive", ""),
                        "is_approved": mem.get("is_approved"),
                    }
                )
            except Exception:
                runs.append(
                    {
                        "run_id": entry,
                        "timestamp": "",
                        "directive": "(unable to read run memory)",
                        "is_approved": None,
                    }
                )
    return runs


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/data/browse")
async def browse_data():
    """Return a recursive directory tree of the data/ directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tree = _build_file_tree(DATA_DIR)
    return JSONResponse(content={"data_dir": DATA_DIR, "tree": tree})


@app.get("/api/runs")
async def list_runs():
    """List past completed runs from the output/ directory."""
    runs = _list_past_runs()
    return JSONResponse(content={"runs": runs})


@app.post("/api/runs", response_model=CreateRunResponse)
async def create_run(req: CreateRunRequest):
    """
    Create a new analysis run.

    Validates the data path, extracts metadata, and prepares the initial state.
    The actual graph execution starts when the client connects via WebSocket.
    Returns 409 if a run is already active.
    """
    global active_run

    if active_run is not None:
        raise HTTPException(
            status_code=409,
            detail=f"A run is already active: {active_run['run_id']}. "
            "Wait for it to complete or cancel it.",
        )

    # Validate data path
    data_path = os.path.join(DATA_DIR, req.data_path)
    data_path = os.path.abspath(data_path)

    # Security: ensure the resolved path stays inside DATA_DIR
    if not data_path.startswith(os.path.abspath(DATA_DIR)):
        raise HTTPException(status_code=400, detail="Data path escapes the data directory.")

    if not os.path.exists(data_path):
        raise HTTPException(
            status_code=400,
            detail=f"Path does not exist: {req.data_path}",
        )

    # Container-side path mapping (for Docker sandbox)
    container_data_path = f"/mnt/data/{req.data_path}"

    # Extract metadata (blocking call — runs synchronously)
    try:
        raw_metadata, is_bids = await asyncio.to_thread(
            _extract_metadata, data_path, container_data_path
        )
    except Exception as e:
        logger.error("Metadata extraction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Metadata extraction failed: {e}")

    # Load reference run memory if provided
    reference_run_memory = None
    if req.reference_run_id:
        reference_run_memory = _load_reference_memory(req.reference_run_id)
        if reference_run_memory is None:
            raise HTTPException(
                status_code=400,
                detail=f"Reference run '{req.reference_run_id}' not found or has no run_memory.json.",
            )

    # Generate thread ID
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_uuid = str(uuid.uuid4())[:8]
    thread_id = f"run_{run_timestamp}_{run_uuid}"

    # Prepare initial graph state
    initial_state = {
        "user_directive": req.directive,
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
        "executed_code_blocks": [],
    }

    # Register the active run
    active_run = {
        "run_id": thread_id,
        "directive": req.directive,
        "data_path": data_path,
        "container_data_path": container_data_path,
        "is_bids": is_bids,
        "initial_state": initial_state,
        "phase": RunPhase.PLANNING,
        "graph_app": None,
        "graph_config": None,
    }

    logger.info("Created run %s for data path: %s", thread_id, req.data_path)

    return CreateRunResponse(
        run_id=thread_id,
        data_path=req.data_path,
        container_data_path=container_data_path,
        is_bids=is_bids,
    )


@app.get("/api/runs/{run_id}/state")
async def get_run_state(run_id: str):
    """Get the current state of a run (for reconnection after page refresh)."""
    # Check active run first
    if active_run and active_run["run_id"] == run_id:
        result = {"run_id": run_id, "phase": active_run["phase"]}
        if active_run["graph_app"] and active_run["graph_config"]:
            try:
                state = active_run["graph_app"].get_state(active_run["graph_config"])
                result["state"] = {
                    k: v
                    for k, v in state.values.items()
                    if k not in ("reference_run",)  # Skip large fields
                }
            except Exception:
                pass
        return JSONResponse(content=result)

    # Check completed runs
    memory_path = os.path.join(OUTPUT_DIR, run_id, "run_memory.json")
    if os.path.exists(memory_path):
        with open(memory_path, "r", encoding="utf-8") as f:
            memory = json.load(f)
        return JSONResponse(
            content={
                "run_id": run_id,
                "phase": RunPhase.COMPLETED,
                "memory": memory,
            }
        )

    raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")


@app.get("/api/runs/{run_id}/report")
async def get_run_report(run_id: str):
    """Return the final_report.md content for a completed run."""
    report_path = os.path.join(OUTPUT_DIR, run_id, "final_report.md")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found.")
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    return JSONResponse(content={"report": content})


@app.get("/api/runs/{run_id}/plots/{filename}")
async def get_run_plot(run_id: str, filename: str):
    """Serve a saved plot PNG file for a run."""
    # Security: prevent path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    filepath = os.path.join(OUTPUT_DIR, run_id, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Plot not found.")
    return FileResponse(filepath, media_type="image/png")


@app.delete("/api/runs/{run_id}")
async def cancel_run(run_id: str):
    """Cancel the active run (releases the single-session lock)."""
    global active_run
    if active_run and active_run["run_id"] == run_id:
        logger.info("Cancelling active run: %s", run_id)
        active_run = None
        return JSONResponse(content={"status": "cancelled", "run_id": run_id})
    raise HTTPException(status_code=404, detail="Run not found or not active.")


# ---------------------------------------------------------------------------
# WebSocket: Real-time graph execution stream
# ---------------------------------------------------------------------------


@app.websocket("/api/runs/{run_id}/stream")
async def stream_run(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint that drives graph execution and streams events to the client.

    Protocol:
    - Server → Client: JSON events (see ServerEvent types in the implementation plan)
    - Client → Server: JSON commands (hitl_response, cancel)
    """
    global active_run

    await websocket.accept()

    # Validate that this run exists and is the active run
    if not active_run or active_run["run_id"] != run_id:
        await websocket.send_json(
            {"type": "error", "message": f"Run '{run_id}' is not active."}
        )
        await websocket.close()
        return

    run = active_run

    try:
        # Build the graph
        logger.info("Building workflow for run %s...", run_id)
        graph_app = build_workflow()
        config = {"configurable": {"thread_id": run_id}}
        run["graph_app"] = graph_app
        run["graph_config"] = config

        # --- Phase 1: Run Planner (streams until interrupt_before=["executor"]) ---
        await websocket.send_json(
            {"type": "status", "phase": "planner", "message": "Planner agent is generating the analysis plan..."}
        )
        run["phase"] = RunPhase.PLANNING

        # Run the graph until it hits the interrupt_before=["executor"]
        plan_text = ""
        await asyncio.to_thread(
            _run_graph_phase_1, graph_app, run["initial_state"], config
        )

        # Read the state after planner completes
        state_after_plan = graph_app.get_state(config)
        plan_text = state_after_plan.values.get("analysis_plan", "")

        await websocket.send_json({"type": "plan_ready", "plan": plan_text})

        # --- Phase 2: HITL — wait for user decision ---
        run["phase"] = RunPhase.AWAITING_HITL
        await websocket.send_json(
            {"type": "hitl_required", "plan": plan_text}
        )

        # Wait for the client's HITL response
        hitl_decision = await _wait_for_hitl(websocket)

        if hitl_decision is None:
            # Client disconnected or sent cancel
            logger.info("Run %s: HITL cancelled by client.", run_id)
            active_run = None
            return

        if hitl_decision.get("decision") == "reject":
            await websocket.send_json(
                {"type": "status", "phase": "planner", "message": "Run rejected by user."}
            )
            await websocket.send_json(
                {"type": "completed", "thread_id": run_id}
            )
            active_run = None
            return

        # If user provided feedback, prepend it to the plan
        feedback = hitl_decision.get("feedback", "").strip()
        if feedback:
            logger.info("User provided feedback on plan.")
            current_plan = state_after_plan.values.get("analysis_plan", "")
            graph_app.update_state(
                config,
                {"analysis_plan": f"USER FEEDBACK: {feedback}\n\n{current_plan}"},
            )

        # --- Phase 3: Executor → Critic loop ---
        await websocket.send_json(
            {"type": "status", "phase": "executor", "message": "Executor agent is running code in the sandbox..."}
        )
        run["phase"] = RunPhase.EXECUTING

        # Resume execution (stream None to continue from interrupt)
        events = await asyncio.to_thread(
            _run_graph_phase_2, graph_app, config
        )

        # Send events to client
        for node_name, node_data in events:
            if node_name == "executor":
                code_blocks = node_data.get("executed_code_blocks", [])
                plots = node_data.get("generated_plots", [])

                for idx, block in enumerate(code_blocks):
                    await websocket.send_json(
                        {
                            "type": "code_block",
                            "index": idx,
                            "code": block.get("code", ""),
                            "logs": block.get("logs", ""),
                            "error": block.get("error", False),
                        }
                    )

                for idx, b64 in enumerate(plots):
                    await websocket.send_json(
                        {"type": "plot", "index": idx, "base64": b64}
                    )

                await websocket.send_json(
                    {
                        "type": "status",
                        "phase": "executor",
                        "message": f"Executed {len(code_blocks)} code block(s), generated {len(plots)} plot(s).",
                    }
                )

            elif node_name == "critic":
                feedback_text = node_data.get("critic_feedback", "")
                is_approved = node_data.get("is_approved", False)
                run["phase"] = RunPhase.REVIEWING

                await websocket.send_json(
                    {
                        "type": "critic_verdict",
                        "approved": is_approved,
                        "feedback": feedback_text,
                    }
                )

                if not is_approved:
                    # The graph may loop back to executor internally
                    await websocket.send_json(
                        {
                            "type": "status",
                            "phase": "executor",
                            "message": "Critic rejected. Executor is retrying...",
                        }
                    )

        # --- Phase 4: Finalize ---
        run["phase"] = RunPhase.COMPLETED
        await websocket.send_json(
            {"type": "status", "phase": "completed", "message": "Generating output artifacts..."}
        )

        result = await asyncio.to_thread(
            finalize_run, graph_app, config, run_id, run["directive"]
        )

        await websocket.send_json(
            {
                "type": "completed",
                "thread_id": run_id,
                "artifacts": {
                    "report": result.get("report_path"),
                    "pipeline": result.get("pipeline_path"),
                    "plots": result.get("plot_files", []),
                },
            }
        )

        logger.info("Run %s completed successfully.", run_id)

    except WebSocketDisconnect:
        logger.warning("Client disconnected during run %s.", run_id)
    except Exception as e:
        logger.error("Run %s failed: %s", run_id, e, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        active_run = None


# ---------------------------------------------------------------------------
# Graph execution helpers (run in thread via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _run_graph_phase_1(graph_app, initial_state: dict, config: dict) -> None:
    """
    Run the graph from initial state until it hits the interrupt_before=["executor"].
    This executes the planner node.
    """
    for event in graph_app.stream(initial_state, config=config):
        # The stream yields events for each node. We just consume them;
        # the state is persisted in the checkpointer.
        pass


def _run_graph_phase_2(graph_app, config: dict) -> list[tuple[str, dict]]:
    """
    Resume graph execution after HITL approval (stream(None, config)).
    Collects all node events (executor, critic) and returns them.
    """
    events = []
    for event in graph_app.stream(None, config=config):
        for node_name, node_data in event.items():
            events.append((node_name, node_data))
    return events


async def _wait_for_hitl(websocket: WebSocket) -> Optional[dict]:
    """
    Wait for a HITL response from the client.
    Returns the parsed JSON command, or None if cancelled/disconnected.
    """
    try:
        while True:
            data = await websocket.receive_text()
            try:
                command = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON in command."}
                )
                continue

            cmd_type = command.get("type")
            if cmd_type == "hitl_response":
                return command
            elif cmd_type == "cancel":
                return None
            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown command type: {cmd_type}"}
                )
    except WebSocketDisconnect:
        return None
