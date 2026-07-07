"""
Tests for src.web.server — the FastAPI bridge server endpoints.

These tests cover REST endpoints (data browsing, run management, artifacts).
WebSocket streaming tests are kept minimal since they require mocking the full
LangGraph execution pipeline.
"""

import asyncio
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def data_dir(tmp_path):
    """Create a realistic data directory with EEG files."""
    # Create some test files
    (tmp_path / "sample.fif").write_bytes(b"fake fif data")
    (tmp_path / "recording.edf").write_bytes(b"fake edf data")
    (tmp_path / "notes.txt").write_text("not an EEG file")

    # Create a BIDS-like directory
    bids_dir = tmp_path / "auditory_bids"
    bids_dir.mkdir()
    (bids_dir / "dataset_description.json").write_text('{"Name": "Auditory"}')
    sub_dir = bids_dir / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)
    (sub_dir / "sub-01_task-aud_eeg.set").write_bytes(b"fake set data")

    # Create a plain directory with EEG files
    plain_dir = tmp_path / "recordings"
    plain_dir.mkdir()
    (plain_dir / "run1.vhdr").write_bytes(b"fake vhdr")
    (plain_dir / "run2.bdf").write_bytes(b"fake bdf")

    return tmp_path


@pytest.fixture
def output_dir(tmp_path):
    """Create a realistic output directory with past runs."""
    run_dir = tmp_path / "run_20260701_120000_abc12345"
    run_dir.mkdir()
    (run_dir / "final_report.md").write_text("# Test Report\nContent here.")
    (run_dir / "run_memory.json").write_text(
        json.dumps(
            {
                "thread_id": "run_20260701_120000_abc12345",
                "timestamp": "2026-07-01T12:00:00",
                "user_directive": "Analyze auditory ERPs",
                "is_approved": True,
            }
        )
    )

    # Create a plot file
    (run_dir / "plot_1.png").write_bytes(b"\x89PNG fake plot data")

    return tmp_path


@pytest.fixture
def client(data_dir, output_dir):
    """Create a test client with patched directories."""
    with patch("src.web.server.DATA_DIR", str(data_dir)), \
         patch("src.web.server.OUTPUT_DIR", str(output_dir)), \
         patch("src.web.server.active_run", None):
        from src.web.server import app
        yield TestClient(app)


class TestBrowseData:
    """Tests for GET /api/data/browse."""

    def test_returns_file_tree(self, client, data_dir):
        """Browse endpoint returns a recursive tree of EEG files."""
        response = client.get("/api/data/browse")
        assert response.status_code == 200

        data = response.json()
        assert "tree" in data

        tree = data["tree"]
        names = [item["name"] for item in tree]

        # Should include EEG files
        assert "sample.fif" in names
        assert "recording.edf" in names

        # Should NOT include non-EEG files
        assert "notes.txt" not in names

    def test_detects_bids_directories(self, client, data_dir):
        """BIDS directories are flagged with is_bids=True."""
        response = client.get("/api/data/browse")
        tree = response.json()["tree"]

        bids_entry = next(
            (item for item in tree if item["name"] == "auditory_bids"), None
        )
        assert bids_entry is not None
        assert bids_entry["type"] == "directory"
        assert bids_entry["is_bids"] is True

    def test_includes_nested_files(self, client, data_dir):
        """Nested EEG files appear as children in the tree."""
        response = client.get("/api/data/browse")
        tree = response.json()["tree"]

        recordings = next(
            (item for item in tree if item["name"] == "recordings"), None
        )
        assert recordings is not None
        child_names = [c["name"] for c in recordings.get("children", [])]
        assert "run1.vhdr" in child_names
        assert "run2.bdf" in child_names

    def test_file_entries_have_size(self, client, data_dir):
        """File entries include their size in bytes."""
        response = client.get("/api/data/browse")
        tree = response.json()["tree"]

        fif_entry = next(
            (item for item in tree if item["name"] == "sample.fif"), None
        )
        assert fif_entry is not None
        assert "size" in fif_entry
        assert fif_entry["size"] > 0


class TestListRuns:
    """Tests for GET /api/runs."""

    def test_lists_past_runs(self, client, output_dir):
        """List runs endpoint returns past runs from output/."""
        response = client.get("/api/runs")
        assert response.status_code == 200

        runs = response.json()["runs"]
        assert len(runs) >= 1

        run = runs[0]
        assert run["run_id"] == "run_20260701_120000_abc12345"
        assert run["directive"] == "Analyze auditory ERPs"
        assert run["is_approved"] is True

    def test_empty_output_directory(self, client, tmp_path):
        """Returns empty list when no past runs exist."""
        with patch("src.web.server.OUTPUT_DIR", str(tmp_path / "nonexistent")):
            response = client.get("/api/runs")
            assert response.status_code == 200
            assert response.json()["runs"] == []


class TestCreateRun:
    """Tests for POST /api/runs."""

    def test_rejects_nonexistent_path(self, client):
        """Returns 400 for a data path that doesn't exist."""
        response = client.post(
            "/api/runs",
            json={
                "data_path": "nonexistent_file.fif",
                "directive": "Test directive",
            },
        )
        assert response.status_code == 400
        assert "does not exist" in response.json()["detail"]

    def test_rejects_path_traversal(self, client):
        """Returns 400 for path traversal attempts."""
        response = client.post(
            "/api/runs",
            json={
                "data_path": "../../etc/passwd",
                "directive": "Hack attempt",
            },
        )
        assert response.status_code == 400
        assert "escapes" in response.json()["detail"]

    def test_rejects_concurrent_run(self, client, data_dir):
        """Returns 409 when a run is already active."""
        with patch("src.web.server.active_run", {"run_id": "existing_run"}):
            response = client.post(
                "/api/runs",
                json={
                    "data_path": "sample.fif",
                    "directive": "Test",
                },
            )
            assert response.status_code == 409
            assert "already active" in response.json()["detail"]

    def test_rejects_invalid_reference_run(self, client, data_dir):
        """Returns 400 for a reference run that doesn't exist."""
        with patch("src.web.server._extract_metadata", return_value=("{}", False)):
            response = client.post(
                "/api/runs",
                json={
                    "data_path": "sample.fif",
                    "directive": "Test",
                    "reference_run_id": "nonexistent_run",
                },
            )
            assert response.status_code == 400
            assert "not found" in response.json()["detail"]


class TestRunReport:
    """Tests for GET /api/runs/{run_id}/report."""

    def test_returns_report_content(self, client, output_dir):
        """Returns the report markdown content."""
        response = client.get("/api/runs/run_20260701_120000_abc12345/report")
        assert response.status_code == 200

        data = response.json()
        assert "# Test Report" in data["report"]

    def test_404_for_missing_report(self, client):
        """Returns 404 for a run without a report."""
        response = client.get("/api/runs/nonexistent_run/report")
        assert response.status_code == 404


class TestRunPlots:
    """Tests for GET /api/runs/{run_id}/plots/{filename}."""

    def test_serves_plot_file(self, client, output_dir):
        """Returns the plot PNG file."""
        response = client.get(
            "/api/runs/run_20260701_120000_abc12345/plots/plot_1.png"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_404_for_missing_plot(self, client, output_dir):
        """Returns 404 for a nonexistent plot."""
        response = client.get(
            "/api/runs/run_20260701_120000_abc12345/plots/nonexistent.png"
        )
        assert response.status_code == 404

    def test_rejects_path_traversal_in_filename(self, client):
        """Returns 400 for path traversal in filename."""
        response = client.get(
            "/api/runs/run_20260701_120000_abc12345/plots/../../etc/passwd"
        )
        # FastAPI may normalize this, but our handler should catch it
        assert response.status_code in (400, 404)


class TestCancelRun:
    """Tests for DELETE /api/runs/{run_id}."""

    def test_cancels_active_run(self, client):
        """Cancelling the active run clears the lock and cancels the task."""
        mock_task = MagicMock()
        with patch("src.web.server.active_run", {"run_id": "active_run_123", "task": mock_task}):
            response = client.delete("/api/runs/active_run_123")
            assert response.status_code == 200
            assert response.json()["status"] == "cancelled"
            mock_task.cancel.assert_called_once()

    def test_404_for_inactive_run(self, client):
        """Returns 404 when trying to cancel a non-active run."""
        response = client.delete("/api/runs/nonexistent_run")
        assert response.status_code == 404


class TestRunState:
    """Tests for GET /api/runs/{run_id}/state."""

    def test_returns_completed_run_state(self, client, output_dir):
        """Returns state for a completed run from its run_memory.json."""
        response = client.get("/api/runs/run_20260701_120000_abc12345/state")
        assert response.status_code == 200

        data = response.json()
        assert data["phase"] == "completed"
        assert "memory" in data

    def test_404_for_unknown_run(self, client):
        """Returns 404 for a run that doesn't exist."""
        response = client.get("/api/runs/nonexistent_run/state")
        assert response.status_code == 404


class TestBackgroundExecution:
    """Tests to verify the background run_execution_loop and WebSocket state management."""

    @pytest.mark.asyncio
    async def test_run_execution_loop_basic(self):
        """Verify run_execution_loop sets phase, runs graph phases, and cleans up active_run."""
        import src.web.server as server
        from src.web.server import run_execution_loop, RunPhase

        run_id = "test_bg_run"
        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()

        # Mock hitl_event so it does not block on .wait()
        mock_hitl_event = MagicMock()
        mock_hitl_event.wait = AsyncMock()

        # Initialize mock active_run structure
        server.active_run = {
            "run_id": run_id,
            "directive": "Analyze data",
            "data_path": "/fake/path",
            "container_data_path": "/mnt/data/path",
            "is_bids": False,
            "initial_state": {"user_directive": "Analyze data"},
            "phase": RunPhase.PLANNING,
            "graph_app": None,
            "graph_config": None,
            "task": None,
            "event_log": [],
            "websockets": {mock_ws},
            "hitl_event": mock_hitl_event,
            "hitl_decision": {"decision": "approve", "feedback": ""},
            "error": None,
        }

        # Mock dependencies of run_execution_loop
        with patch("src.web.server.build_workflow") as mock_build, \
             patch("src.web.server._run_graph_phase_1") as mock_phase_1, \
             patch("src.web.server._run_graph_phase_2", return_value=[("executor", {"executed_code_blocks": [], "generated_plots": []})]) as mock_phase_2, \
             patch("src.web.server.finalize_run", return_value={"report_path": "report.md"}) as mock_finalize:

            mock_graph = MagicMock()
            mock_state = MagicMock()
            mock_state.values = {"analysis_plan": "Mock Plan"}
            mock_graph.get_state.return_value = mock_state
            mock_build.return_value = mock_graph

            # Execute loop
            await run_execution_loop(run_id)

            # Assert lock is released (active_run is cleared)
            assert server.active_run is None

            # Assert internal execution steps were triggered
            mock_build.assert_called_once()
            mock_phase_1.assert_called_once()
            mock_phase_2.assert_called_once()
            mock_finalize.assert_called_once()

            # Assert that WS broadcast was called with expected phases
            sent_types = [call.args[0]["type"] for call in mock_ws.send_json.call_args_list]
            assert "plan_ready" in sent_types
            assert "completed" in sent_types


# AsyncMock helper for Python < 3.8 or custom mock setups
class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)

