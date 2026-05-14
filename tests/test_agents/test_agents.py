import pytest
from unittest.mock import patch, MagicMock
from src.agents.planner import get_planner_agent
from src.agents.executor import get_executor_agent
from src.agents.critic import get_critic_agent

@patch("src.agents.planner.create_react_agent")
@patch("src.agents.planner.get_llm")
def test_get_planner_agent(mock_get_llm, mock_create):
    mock_get_llm.return_value = MagicMock()
    mock_create.return_value = "FakeAgent"
    agent = get_planner_agent()
    assert agent == "FakeAgent"
    mock_get_llm.assert_called_once_with(agent_type="text", temperature=0.1)

@patch("src.agents.executor.create_react_agent")
@patch("src.agents.executor.get_llm")
def test_get_executor_agent(mock_get_llm, mock_create):
    mock_get_llm.return_value = MagicMock()
    mock_create.return_value = "FakeAgent"
    agent = get_executor_agent()
    assert agent == "FakeAgent"
    mock_get_llm.assert_called_once_with(agent_type="text", temperature=0.2)

@patch("src.agents.critic.get_llm")
def test_get_critic_agent(mock_get_llm):
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    agent_func = get_critic_agent()
    assert callable(agent_func)
    mock_get_llm.assert_called_once_with(agent_type="multimodal", temperature=0.1)
    
    # Test critic execution
    mock_llm.invoke.return_value = MagicMock(content="APPROVE")
    response = agent_func({"execution_logs": "success", "generated_plots": ["base64img"]})
    assert response == "APPROVE"
    mock_llm.invoke.assert_called_once()
