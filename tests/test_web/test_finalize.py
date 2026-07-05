"""
Tests for src.web.finalize — the shared post-processing / output generation logic.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from src.web.finalize import finalize_run


@pytest.fixture
def mock_app_and_state():
    """Create a mock graph app with realistic state data."""
    state_values = {
        "user_directive": "Perform ERP analysis on auditory oddball data",
        "data_path": "/mnt/data/sample.fif",
        "raw_metadata": json.dumps({"n_channels": 64, "sfreq": 256}),
        "analysis_plan": "## Plan\n1. Load data\n2. Filter\n3. Epoch\n4. Average",
        "execution_logs": ["Loaded data successfully", "Applied bandpass filter"],
        "generated_plots": [],  # No plots by default
        "error_count": 0,
        "critic_feedback": "APPROVE: Analysis looks correct. Methods section follows.",
        "is_approved": True,
        "rag_history": [
            {
                "query": "N400 ERP filter settings",
                "paradigm": "ERP",
                "target": "methods",
                "results": "Use 0.1-30 Hz bandpass filter.",
            }
        ],
        "executed_code_blocks": [
            {
                "code": "import mne\nraw = mne.io.read_raw_fif('/mnt/data/sample.fif', preload=False)",
                "logs": "Reading raw data...\nReady.",
                "error": False,
            },
            {
                "code": "raw.filter(0.1, 30)",
                "logs": "Filtering...\nDone.",
                "error": False,
            },
        ],
    }

    mock_state = MagicMock()
    mock_state.values = state_values

    mock_app = MagicMock()
    mock_app.get_state.return_value = mock_state

    return mock_app, state_values


class TestFinalizeRun:
    """Tests for the finalize_run function."""

    def test_creates_output_directory(self, mock_app_and_state, tmp_path):
        """finalize_run creates the output directory structure."""
        app, _ = mock_app_and_state
        config = {"configurable": {"thread_id": "test_run_001"}}

        result = finalize_run(
            app, config, "test_run_001", "Test directive", output_base_dir=str(tmp_path)
        )

        assert os.path.isdir(result["output_dir"])
        assert result["output_dir"] == str(tmp_path / "test_run_001")

    def test_generates_report(self, mock_app_and_state, tmp_path):
        """finalize_run generates a final_report.md with all sections."""
        app, _ = mock_app_and_state
        config = {"configurable": {"thread_id": "test_run_002"}}

        result = finalize_run(
            app, config, "test_run_002", "ERP analysis", output_base_dir=str(tmp_path)
        )

        assert os.path.exists(result["report_path"])

        with open(result["report_path"], "r") as f:
            content = f.read()

        # Check report sections
        assert "# Final Analysis Report - Session `test_run_002`" in content
        assert "ERP analysis" in content  # directive
        assert "## 2. Plan Executed" in content
        assert "## 3. RAG Retrieval Audit Log" in content
        assert "N400 ERP filter settings" in content  # RAG query
        assert "## 4. Code Execution Trace" in content
        assert "import mne" in content  # code block
        assert "## 5. Critic Feedback" in content
        assert "APPROVE" in content

    def test_generates_pipeline_script(self, mock_app_and_state, tmp_path):
        """finalize_run compiles successful code blocks into analysis_pipeline.py."""
        app, _ = mock_app_and_state
        config = {"configurable": {"thread_id": "test_run_003"}}

        result = finalize_run(
            app, config, "test_run_003", "Test", output_base_dir=str(tmp_path)
        )

        assert result["pipeline_path"] is not None
        assert os.path.exists(result["pipeline_path"])

        with open(result["pipeline_path"], "r") as f:
            content = f.read()

        assert "#!/usr/bin/env python" in content
        assert "import mne" in content
        assert "MNE_MEMMAP_MIN_SIZE" in content
        assert "Code Block 1" in content
        assert "Code Block 2" in content

    def test_skips_failed_code_blocks_in_pipeline(self, mock_app_and_state, tmp_path):
        """Failed code blocks are NOT included in analysis_pipeline.py."""
        app, state = mock_app_and_state
        state["executed_code_blocks"].append(
            {"code": "broken_code()", "logs": "NameError", "error": True}
        )
        config = {"configurable": {"thread_id": "test_run_004"}}

        result = finalize_run(
            app, config, "test_run_004", "Test", output_base_dir=str(tmp_path)
        )

        with open(result["pipeline_path"], "r") as f:
            content = f.read()

        assert "broken_code()" not in content

    def test_no_pipeline_when_no_successful_code(self, mock_app_and_state, tmp_path):
        """If all code blocks failed, no pipeline script is generated."""
        app, state = mock_app_and_state
        state["executed_code_blocks"] = [
            {"code": "fail()", "logs": "Error", "error": True}
        ]
        config = {"configurable": {"thread_id": "test_run_005"}}

        result = finalize_run(
            app, config, "test_run_005", "Test", output_base_dir=str(tmp_path)
        )

        assert result["pipeline_path"] is None

    def test_generates_run_memory(self, mock_app_and_state, tmp_path):
        """finalize_run saves a valid run_memory.json."""
        app, _ = mock_app_and_state
        config = {"configurable": {"thread_id": "test_run_006"}}

        result = finalize_run(
            app, config, "test_run_006", "Memory test", output_base_dir=str(tmp_path)
        )

        assert os.path.exists(result["memory_path"])

        with open(result["memory_path"], "r") as f:
            memory = json.load(f)

        assert memory["thread_id"] == "test_run_006"
        assert memory["user_directive"] == "Memory test"
        assert memory["is_approved"] is True
        assert memory["error_count"] == 0
        assert "report" in memory["artifacts"]
        assert memory["artifacts"]["pipeline_script"] is not None

    def test_saves_plots_to_disk(self, mock_app_and_state, tmp_path):
        """finalize_run decodes and saves base64 plot images."""
        import base64

        app, state = mock_app_and_state
        # Create a minimal valid PNG (1x1 pixel)
        # This is the smallest valid PNG file
        tiny_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        state["generated_plots"] = [base64.b64encode(tiny_png).decode()]
        config = {"configurable": {"thread_id": "test_run_007"}}

        result = finalize_run(
            app, config, "test_run_007", "Plot test", output_base_dir=str(tmp_path)
        )

        assert len(result["plot_files"]) == 1
        assert result["plot_files"][0] == "plot_1.png"
        assert os.path.exists(os.path.join(result["output_dir"], "plot_1.png"))

    def test_report_includes_plot_references(self, mock_app_and_state, tmp_path):
        """When plots exist, the report includes image references."""
        import base64

        app, state = mock_app_and_state
        tiny_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50  # Not a real PNG but will decode
        state["generated_plots"] = [base64.b64encode(tiny_png).decode()]
        config = {"configurable": {"thread_id": "test_run_008"}}

        result = finalize_run(
            app, config, "test_run_008", "Test", output_base_dir=str(tmp_path)
        )

        with open(result["report_path"], "r") as f:
            content = f.read()

        assert "## 6. Visual Artifacts" in content
        assert "![plot_1.png]" in content

    def test_handles_empty_state(self, tmp_path):
        """finalize_run handles a minimal/empty state without crashing."""
        state_values = {
            "raw_metadata": "{}",
            "analysis_plan": "",
            "execution_logs": [],
            "generated_plots": [],
            "error_count": 0,
            "critic_feedback": "",
            "is_approved": False,
            "rag_history": [],
            "executed_code_blocks": [],
            "data_path": "/mnt/data/test",
        }

        mock_state = MagicMock()
        mock_state.values = state_values
        mock_app = MagicMock()
        mock_app.get_state.return_value = mock_state
        config = {"configurable": {"thread_id": "test_run_009"}}

        result = finalize_run(
            mock_app, config, "test_run_009", "", output_base_dir=str(tmp_path)
        )

        assert os.path.exists(result["report_path"])
        assert result["pipeline_path"] is None
        assert result["plot_files"] == []
