import pytest
from unittest.mock import patch, MagicMock
from src.graph.workflow import planner_node, executor_node, critic_node, critic_router
from langgraph.graph import END

@patch("src.graph.workflow.get_planner_agent")
def test_planner_node(mock_get_planner):
    mock_agent = MagicMock()
    mock_get_planner.return_value = mock_agent
    
    mock_message = MagicMock()
    mock_message.content = "Mocked Plan"
    mock_agent.invoke.return_value = {"messages": [mock_message]}
    
    state = {"user_directive": "Clean data", "data_path": "/data/test.fif"}
    result = planner_node(state)
    
    assert result == {"analysis_plan": "Mocked Plan", "rag_history": []}

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
