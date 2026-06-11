from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState
from src.agents.planner import get_planner_agent
from src.agents.executor import get_executor_agent
from src.agents.critic import get_critic_agent
import json
import logging

logger = logging.getLogger("eeg_agent.workflow")

def planner_node(state: AgentState):
    logger.info("Planner Node: Starting plan generation...")
    planner = get_planner_agent()
    prompt = f"User Directive: {state['user_directive']}\nData Path: {state['data_path']}\n\nPlease generate the Analysis Plan."
    result = planner.invoke({"messages": [HumanMessage(content=prompt)]})
    
    final_message = result["messages"][-1].content
    logger.info("Planner Node: Plan generation completed.")
    return {"analysis_plan": final_message}

def executor_node(state: AgentState):
    logger.info("Executor Node: Starting code generation and execution in Docker Sandbox...")
    executor = get_executor_agent()
    
    prompt = f"Here is the Analysis Plan to execute:\n{state['analysis_plan']}\n\n"
    if state.get("critic_feedback"):
        logger.info("Executor Node: Incorporating Critic feedback from previous iteration.")
        prompt += f"CRITIC FEEDBACK FROM PREVIOUS RUN: {state['critic_feedback']}\nPlease adjust your code accordingly.\n"
        
    result = executor.invoke({"messages": [HumanMessage(content=prompt)]})
    
    logs = []
    images = []
    error_occurred = False
    
    for m in result["messages"]:
        if m.type == "tool":
            try:
                res = json.loads(m.content)
                if "logs" in res:
                    logs.append(res["logs"])
                if "images" in res:
                    images.extend(res["images"])
                if res.get("error", False):
                    error_occurred = True
            except:
                pass

    logger.info(
        "Executor Node: Finished execution. Success=%s. Generated %d plots.",
        not error_occurred,
        len(images)
    )
    return {
        "execution_logs": logs,
        "generated_plots": images,
        "error_count": 1 if error_occurred else 0 # Simplified error tracking
    }

def critic_node(state: AgentState):
    logger.info("Critic Node: Invoking QA / review agent...")
    critic = get_critic_agent()
    feedback = critic(state)
    
    is_approved = "APPROVE" in feedback.upper()
    logger.info("Critic Node: QA completed. Approved=%s.", is_approved)
    return {
        "critic_feedback": feedback,
        "is_approved": is_approved
    }

def critic_router(state: AgentState):
    is_approved = state.get("is_approved", False)
    error_count = state.get("error_count", 0)
    
    logger.info("Critic Router: checking approvals. approved=%s, error_count=%d", is_approved, error_count)
    if is_approved:
        logger.info("Critic Router: Approved. Ending workflow.")
        return END
    else:
        # Check recursion limit
        from src import config
        max_retries = int(config.get_val("executor.max_retries"))
        if error_count >= max_retries:
            logger.warning("Critic Router: Max retry limit (%d) reached. Ending workflow with unresolved issues.", max_retries)
            return END
        logger.info("Critic Router: Rejected. Routing back to Executor Node.")
        return "executor"

from langgraph.checkpoint.memory import MemorySaver

def build_workflow():
    logger.info("Building StateGraph workflow...")
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("critic", critic_node)
    
    workflow.set_entry_point("planner")
    
    # The interrupt_before flag will pause execution here for HITL
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "critic")
    
    workflow.add_conditional_edges("critic", critic_router, {
        "executor": "executor",
        END: END
    })
    
    memory = MemorySaver()
    return workflow.compile(interrupt_before=["executor"], checkpointer=memory)
