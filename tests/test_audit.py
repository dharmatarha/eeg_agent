import os
import sqlite3
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, ToolMessage
from src.graph.workflow import extract_tool_trace, build_workflow
from langgraph.checkpoint.sqlite import SqliteSaver

def test_extract_tool_trace():
    # Mock messages
    ai_msg_rag = MagicMock(spec=AIMessage)
    ai_msg_rag.type = "ai"
    ai_msg_rag.tool_calls = [{
        "id": "call_1",
        "name": "scientific_rag",
        "args": {"query": "N400 filter", "paradigm": "N400", "target": "methods"}
    }]
    
    tool_msg_rag = MagicMock(spec=ToolMessage)
    tool_msg_rag.type = "tool"
    tool_msg_rag.name = "scientific_rag"
    tool_msg_rag.tool_call_id = "call_1"
    tool_msg_rag.content = "Retrieved filter window: 0.1 to 30 Hz"
    
    ai_msg_exec = MagicMock(spec=AIMessage)
    ai_msg_exec.type = "ai"
    ai_msg_exec.tool_calls = [{
        "id": "call_2",
        "name": "stateful_jupyter_exec",
        "args": {"code_string": "print('hello')"}
    }]
    
    tool_msg_exec = MagicMock(spec=ToolMessage)
    tool_msg_exec.type = "tool"
    tool_msg_exec.name = "stateful_jupyter_exec"
    tool_msg_exec.tool_call_id = "call_2"
    tool_msg_exec.content = '{"logs": "hello\\n", "error": false}'
    
    messages = [ai_msg_rag, tool_msg_rag, ai_msg_exec, tool_msg_exec]
    rag_history, executed_code_blocks = extract_tool_trace(messages)
    
    assert len(rag_history) == 1
    assert rag_history[0]["query"] == "N400 filter"
    assert rag_history[0]["paradigm"] == "N400"
    assert rag_history[0]["target"] == "methods"
    assert rag_history[0]["results"] == "Retrieved filter window: 0.1 to 30 Hz"
    
    assert len(executed_code_blocks) == 1
    assert executed_code_blocks[0]["code"] == "print('hello')"
    assert executed_code_blocks[0]["logs"] == "hello\n"
    assert executed_code_blocks[0]["error"] is False

def test_sqlite_saver_workflow(tmp_path):
    # Test compilation and checkpointer database creation
    db_file = tmp_path / "test_checkpoints.sqlite"
    conn = sqlite3.connect(str(db_file))
    saver = SqliteSaver(conn)
    
    # Verify saver creation
    assert saver is not None
    
    # Verify that build_workflow compiles successfully with custom checkpointer
    app = build_workflow(checkpointer=saver)
    assert app is not None
    assert os.path.exists(str(db_file))
