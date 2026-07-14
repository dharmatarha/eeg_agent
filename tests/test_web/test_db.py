import os
import json
import sqlite3
import pytest
from src.web.db import init_db, insert_run, update_run_status, list_runs, sync_past_runs

@pytest.fixture
def temp_db_path(tmp_path):
    """Fixture returning a path to a temporary SQLite database."""
    return os.path.join(tmp_path, "test_checkpoints.sqlite")

def test_init_db_creates_table(temp_db_path):
    """Verify that init_db creates the runs_index table with correct columns."""
    assert not os.path.exists(temp_db_path)
    
    init_db(temp_db_path)
    assert os.path.exists(temp_db_path)
    
    # Check that table and schema exist
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.execute("PRAGMA table_info(runs_index)")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()
    
    assert "run_id" in columns
    assert "timestamp" in columns
    assert "directive" in columns
    assert "status" in columns
    assert "is_approved" in columns
    assert "data_path" in columns

def test_insert_and_list_runs(temp_db_path):
    """Verify runs can be inserted and listed chronologically."""
    init_db(temp_db_path)
    
    # Insert two runs
    insert_run(
        db_path=temp_db_path,
        run_id="run_1",
        timestamp="2026-07-14T12:00:00",
        directive="Clean N400 data",
        status="planning",
        is_approved=None,
        data_path="/mnt/data/subj1.fif"
    )
    
    insert_run(
        db_path=temp_db_path,
        run_id="run_2",
        timestamp="2026-07-14T13:00:00",
        directive="Verify ERP",
        status="completed",
        is_approved=True,
        data_path="/mnt/data/subj2.fif"
    )
    
    runs = list_runs(temp_db_path)
    assert len(runs) == 2
    
    # Should be sorted desc by timestamp (run_2 first)
    assert runs[0]["run_id"] == "run_2"
    assert runs[0]["directive"] == "Verify ERP"
    assert runs[0]["status"] == "completed"
    assert runs[0]["is_approved"] is True
    assert runs[0]["data_path"] == "/mnt/data/subj2.fif"
    
    assert runs[1]["run_id"] == "run_1"
    assert runs[1]["directive"] == "Clean N400 data"
    assert runs[1]["status"] == "planning"
    assert runs[1]["is_approved"] is None
    assert runs[1]["data_path"] == "/mnt/data/subj1.fif"

def test_update_run_status(temp_db_path):
    """Verify run status and approval state can be updated."""
    init_db(temp_db_path)
    
    insert_run(
        db_path=temp_db_path,
        run_id="run_1",
        timestamp="2026-07-14T12:00:00",
        directive="Clean N400 data",
        status="planning",
        is_approved=None,
        data_path="/mnt/data/subj1.fif"
    )
    
    # Update status only
    update_run_status(temp_db_path, "run_1", "executing")
    runs = list_runs(temp_db_path)
    assert runs[0]["status"] == "executing"
    assert runs[0]["is_approved"] is None
    
    # Update status and approval
    update_run_status(temp_db_path, "run_1", "reviewing", is_approved=False)
    runs = list_runs(temp_db_path)
    assert runs[0]["status"] == "reviewing"
    assert runs[0]["is_approved"] is False

def test_sync_past_runs(temp_db_path, tmp_path):
    """Verify past runs can be synchronized from the filesystem output dir."""
    init_db(temp_db_path)
    
    # Set up mock output directories
    output_dir = os.path.join(tmp_path, "output")
    os.makedirs(output_dir)
    
    # Run 1: has run_memory.json
    run1_dir = os.path.join(output_dir, "run_20260701_100000_abc")
    os.makedirs(run1_dir)
    run1_mem = {
        "thread_id": "run_20260701_100000_abc",
        "timestamp": "2026-07-01T10:00:00",
        "user_directive": "Inspect subject 1",
        "is_approved": True,
        "data_path": "/mnt/data/subj1.fif"
    }
    with open(os.path.join(run1_dir, "run_memory.json"), "w") as f:
        json.dump(run1_mem, f)
        
    # Run 2: is already in db, should not be synced/overwritten
    run2_dir = os.path.join(output_dir, "run_20260702_100000_def")
    os.makedirs(run2_dir)
    run2_mem = {
        "thread_id": "run_20260702_100000_def",
        "timestamp": "2026-07-02T10:00:00",
        "user_directive": "Directive to ignore",
        "is_approved": True,
        "data_path": "/mnt/data/subj2.fif"
    }
    with open(os.path.join(run2_dir, "run_memory.json"), "w") as f:
        json.dump(run2_mem, f)
        
    # Insert run 2 manually beforehand to verify it's skipped
    insert_run(
        db_path=temp_db_path,
        run_id="run_20260702_100000_def",
        timestamp="2026-07-02T10:00:00",
        directive="Already in DB",
        status="completed",
        is_approved=True,
        data_path="/mnt/data/subj2.fif"
    )
    
    # Run sync
    sync_past_runs(temp_db_path, output_dir)
    
    runs = list_runs(temp_db_path)
    assert len(runs) == 2
    
    # Verify run 1 was synced
    run1_db = next(r for r in runs if r["run_id"] == "run_20260701_100000_abc")
    assert run1_db["directive"] == "Inspect subject 1"
    assert run1_db["status"] == "completed"
    assert run1_db["is_approved"] is True
    
    # Verify run 2 was NOT overwritten
    run2_db = next(r for r in runs if r["run_id"] == "run_20260702_100000_def")
    assert run2_db["directive"] == "Already in DB"
