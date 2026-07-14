import os
import sqlite3
import json
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger("eeg_agent.web.db")

def _get_connection(db_path: str) -> sqlite3.Connection:
    """Return a database connection."""
    return sqlite3.connect(db_path, check_same_thread=False)

def init_db(db_path: str) -> None:
    """Initialize the runs index database and create schema if needed."""
    db_dir = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(db_dir, exist_ok=True)
    
    with _get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs_index (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                directive TEXT NOT NULL,
                status TEXT NOT NULL,
                is_approved INTEGER, -- NULL, 0 (False), or 1 (True)
                data_path TEXT NOT NULL
            )
        """)
        logger.info("Database schema initialized at %s", db_path)

def insert_run(
    db_path: str,
    run_id: str,
    timestamp: str,
    directive: str,
    status: str,
    is_approved: Optional[bool],
    data_path: str
) -> None:
    """Insert a new run record into the database."""
    approved_val = None
    if is_approved is not None:
        approved_val = 1 if is_approved else 0
        
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs_index 
            (run_id, timestamp, directive, status, is_approved, data_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, timestamp, directive, status, approved_val, data_path)
        )
        logger.info("Inserted/updated run %s into index with status: %s", run_id, status)

def update_run_status(
    db_path: str,
    run_id: str,
    status: str,
    is_approved: Optional[bool] = None
) -> None:
    """Update status and optionally is_approved for a run."""
    with _get_connection(db_path) as conn:
        if is_approved is not None:
            approved_val = 1 if is_approved else 0
            conn.execute(
                "UPDATE runs_index SET status = ?, is_approved = ? WHERE run_id = ?",
                (status, approved_val, run_id)
            )
        else:
            conn.execute(
                "UPDATE runs_index SET status = ? WHERE run_id = ?",
                (status, run_id)
            )
        logger.info("Updated run %s status to: %s", run_id, status)

def list_runs(db_path: str) -> List[Dict[str, Any]]:
    """Retrieve all runs from the database sorted by timestamp descending."""
    runs = []
    try:
        with _get_connection(db_path) as conn:
            cursor = conn.execute(
                "SELECT run_id, timestamp, directive, status, is_approved, data_path FROM runs_index ORDER BY timestamp DESC"
            )
            for row in cursor.fetchall():
                approved_val = row[4]
                is_approved = None
                if approved_val is not None:
                    is_approved = True if approved_val == 1 else False
                    
                runs.append({
                    "run_id": row[0],
                    "timestamp": row[1],
                    "directive": row[2],
                    "status": row[3],
                    "is_approved": is_approved,
                    "data_path": row[5]
                })
    except sqlite3.OperationalError as e:
        logger.error("Failed to query runs_index: %s", e)
    return runs

def get_run_by_id(db_path: str, run_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single run record by its ID, or None if not found."""
    try:
        with _get_connection(db_path) as conn:
            cursor = conn.execute(
                "SELECT run_id, timestamp, directive, status, is_approved, data_path "
                "FROM runs_index WHERE run_id = ?",
                (run_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None

            approved_val = row[4]
            is_approved = None
            if approved_val is not None:
                is_approved = True if approved_val == 1 else False

            return {
                "run_id": row[0],
                "timestamp": row[1],
                "directive": row[2],
                "status": row[3],
                "is_approved": is_approved,
                "data_path": row[5],
            }
    except sqlite3.OperationalError as e:
        logger.error("Failed to query run %s: %s", run_id, e)
        return None


def sync_past_runs(db_path: str, output_dir: str) -> None:
    """Synchronize past completed runs from the filesystem output directory into SQLite."""
    if not os.path.isdir(output_dir):
        return
        
    # Get set of run_ids currently indexed
    indexed_runs = set()
    try:
        with _get_connection(db_path) as conn:
            cursor = conn.execute("SELECT run_id FROM runs_index")
            indexed_runs = {row[0] for row in cursor.fetchall()}
    except sqlite3.OperationalError:
        # Table might not exist yet if init_db wasn't called
        return

    logger.info("Starting synchronization of past runs from %s", output_dir)
    sync_count = 0
    
    for entry in os.listdir(output_dir):
        run_dir = os.path.join(output_dir, entry)
        if not os.path.isdir(run_dir):
            continue
            
        # Skip if already in the index database
        if entry in indexed_runs:
            continue
            
        memory_path = os.path.join(run_dir, "run_memory.json")
        if os.path.exists(memory_path):
            try:
                with open(memory_path, "r", encoding="utf-8") as f:
                    mem = json.load(f)
                    
                run_id = mem.get("thread_id", entry)
                timestamp = mem.get("timestamp", "")
                directive = mem.get("user_directive", "")
                is_approved = mem.get("is_approved", True)
                data_path = mem.get("data_path", "")
                
                # Check status: default to completed
                status = "completed"
                # If error_count is present and we want to categorize as failed, we can do:
                # if mem.get("error_count", 0) >= max_retries and not is_approved...
                
                insert_run(
                    db_path=db_path,
                    run_id=run_id,
                    timestamp=timestamp,
                    directive=directive,
                    status=status,
                    is_approved=is_approved,
                    data_path=data_path
                )
                sync_count += 1
            except Exception as e:
                logger.error("Failed to synchronize run directory %s: %s", entry, e)
                
    if sync_count > 0:
        logger.info("Successfully synchronized %d past runs to the database index.", sync_count)
