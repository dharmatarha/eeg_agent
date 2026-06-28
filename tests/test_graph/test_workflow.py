import pytest
from unittest.mock import patch, MagicMock
from src.graph.workflow import planner_node, executor_node, critic_node, critic_router
from langgraph.graph import END

@patch("src.graph.workflow.get_planner_agent")
def test_planner_node(mock_get_planner):
    mock_agent = MagicMock()
    mock_get_planner.return_value = mock_agent
    
    mock_message = MagicMock()
    mock_message.content = "Mocked Plan that is long enough to pass validation because it needs to be at least 100 characters to be considered a valid analysis plan by the check."
    mock_agent.invoke.return_value = {"messages": [mock_message]}
    
    state = {"user_directive": "Clean data", "data_path": "/data/test.fif"}
    result = planner_node(state)
    
    assert result == {"analysis_plan": mock_message.content, "rag_history": []}

@patch("src.graph.workflow.get_planner_agent")
def test_planner_node_retry_on_malformed(mock_get_planner):
    mock_agent = MagicMock()
    mock_get_planner.return_value = mock_agent
    
    # First message: response_metadata says MALFORMED_FUNCTION_CALL
    msg1 = MagicMock()
    msg1.content = "}"
    msg1.response_metadata = {"finish_reason": "MALFORMED_FUNCTION_CALL"}
    
    # Second message: too short
    msg2 = MagicMock()
    msg2.content = "short plan"
    msg2.response_metadata = {}
    
    # Third message: valid
    msg3 = MagicMock()
    msg3.content = "This is a valid plan that is long enough to pass the length constraint of 100 characters successfully."
    msg3.response_metadata = {}
    
    mock_agent.invoke.side_effect = [
        {"messages": [msg1]},
        {"messages": [msg2]},
        {"messages": [msg3]},
    ]
    
    state = {"user_directive": "Clean data", "data_path": "/data/test.fif"}
    result = planner_node(state)
    
    assert mock_agent.invoke.call_count == 3
    assert result == {"analysis_plan": msg3.content, "rag_history": []}

@patch("src.graph.workflow.get_planner_agent")
def test_planner_node_retry_exhausted(mock_get_planner):
    mock_agent = MagicMock()
    mock_get_planner.return_value = mock_agent
    
    msg = MagicMock()
    msg.content = "}"
    msg.response_metadata = {"finish_reason": "MALFORMED_FUNCTION_CALL"}
    
    mock_agent.invoke.return_value = {"messages": [msg]}
    
    state = {"user_directive": "Clean data", "data_path": "/data/test.fif"}
    result = planner_node(state)
    
    assert mock_agent.invoke.call_count == 3
    assert result == {"analysis_plan": "}", "rag_history": []}

@patch("src.graph.workflow.get_executor_agent")
def test_executor_node(mock_get_executor):
    mock_agent = MagicMock()
    mock_get_executor.return_value = mock_agent
    
    mock_tool_message = MagicMock()
    mock_tool_message.type = "tool"
    mock_tool_message.content = '{"logs": "Executed print()", "images": ["base64"], "error": false}'
    
    mock_agent.invoke.return_value = {"messages": [mock_tool_message]}
    
    state = {"analysis_plan": "Do something"}
    result = executor_node(state)
    
    assert result["execution_logs"] == ["Executed print()"]
    assert result["generated_plots"] == ["base64"]
    assert result["error_count"] == 0
    assert result["rag_history"] == []
    assert result["executed_code_blocks"] == []

@patch("src.graph.workflow.get_executor_agent")
def test_executor_node_accumulates_errors(mock_get_executor):
    mock_agent = MagicMock()
    mock_get_executor.return_value = mock_agent
    
    mock_tool_message = MagicMock()
    mock_tool_message.type = "tool"
    mock_tool_message.content = '{"logs": "Error backtrace", "images": [], "error": true}'
    
    mock_agent.invoke.return_value = {"messages": [mock_tool_message]}
    
    state = {"analysis_plan": "Do something", "error_count": 2}
    result = executor_node(state)
    
    assert result["error_count"] == 3

@patch("src.graph.workflow.get_critic_agent")
def test_critic_node_approve(mock_get_critic):
    mock_agent = MagicMock()
    mock_agent.return_value = "I APPROVE this."
    mock_get_critic.return_value = mock_agent
    
    state = {"execution_logs": [], "generated_plots": []}
    result = critic_node(state)
    
    assert result["is_approved"] is True
    assert result["critic_feedback"] == "I APPROVE this."

@patch("src.graph.workflow.get_critic_agent")
def test_critic_node_reject(mock_get_critic):
    mock_agent = MagicMock()
    mock_agent.return_value = "I REJECT this."
    mock_get_critic.return_value = mock_agent
    
    state = {"execution_logs": [], "generated_plots": []}
    result = critic_node(state)
    
    assert result["is_approved"] is False

def test_critic_router():
    assert critic_router({"is_approved": True}) == END
    assert critic_router({"is_approved": False, "error_count": 5}) == END
    assert critic_router({"is_approved": False, "error_count": 0}) == "executor"

def test_detect_repetition():
    from src.graph.workflow import detect_repetition
    
    # Clean text: no repetition
    assert not detect_repetition("Hello world\nThis is a plan\nIt should be good.")
    
    # 3 consecutive repeats of a line > 20 chars
    consecutive = (
        "This is a line that is longer than twenty characters.\n"
        "This is a line that is longer than twenty characters.\n"
        "This is a line that is longer than twenty characters.\n"
        "This is a line that is longer than twenty characters."
    )
    assert detect_repetition(consecutive)
    
    # 5 repeats of a line > 30 chars anywhere
    repeated_anywhere = (
        "This is a very long line that will be repeated many times in the document.\n"
        "Some intermediate text.\n"
        "This is a very long line that will be repeated many times in the document.\n"
        "Another text.\n"
        "This is a very long line that will be repeated many times in the document.\n"
        "More text.\n"
        "This is a very long line that will be repeated many times in the document.\n"
        "Almost done.\n"
        "This is a very long line that will be repeated many times in the document."
    )
    assert detect_repetition(repeated_anywhere)

@patch("src.graph.workflow.get_planner_agent")
def test_planner_node_retry_on_repetition_and_length(mock_get_planner):
    mock_agent = MagicMock()
    mock_get_planner.return_value = mock_agent
    
    # First response: too long (26000 chars)
    msg1 = MagicMock()
    msg1.content = "A" * 26000
    msg1.response_metadata = {}
    
    # Second response: has repetition
    msg2 = MagicMock()
    msg2.content = (
        "Let's check if there are other channels in the raw data that we didn't see.\n"
        "Let's check if there are other channels in the raw data that we didn't see.\n"
        "Let's check if there are other channels in the raw data that we didn't see.\n"
        "Let's check if there are other channels in the raw data that we didn't see."
    )
    msg2.response_metadata = {}
    
    # Third response: valid
    msg3 = MagicMock()
    msg3.content = "This is a valid plan that is long enough to pass the length constraint of 100 characters successfully without any repetition."
    msg3.response_metadata = {}
    
    mock_agent.invoke.side_effect = [
        {"messages": [msg1]},
        {"messages": [msg2]},
        {"messages": [msg3]},
    ]
    
    state = {"user_directive": "Clean data", "data_path": "/data/test.fif"}
    result = planner_node(state)
    
    assert mock_agent.invoke.call_count == 3
    assert result == {"analysis_plan": msg3.content, "rag_history": []}

