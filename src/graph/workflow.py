from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState
from src.agents.planner import get_planner_agent
from src.agents.executor import get_executor_agent
from src.agents.critic import get_critic_agent
import json

def planner_node(state: AgentState):
    planner = get_planner_agent()
    prompt = f"User Directive: {state['user_directive']}\nData Path: {state['data_path']}\n\nPlease generate the Analysis Plan."
    result = planner.invoke({"messages": [HumanMessage(content=prompt)]})
    
    final_message = result["messages"][-1].content
    return {"analysis_plan": final_message}

def executor_node(state: AgentState):
    executor = get_executor_agent()
    
    prompt = f"Here is the Analysis Plan to execute:\n{state['analysis_plan']}\n\n"
    if state.get("critic_feedback"):
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

    return {
        "execution_logs": logs,
        "generated_plots": images,
        "error_count": 1 if error_occurred else 0 # Simplified error tracking
    }

def critic_node(state: AgentState):
    critic = get_critic_agent()
    feedback = critic(state)
    
    is_approved = "APPROVE" in feedback.upper()
    return {
        "critic_feedback": feedback,
        "is_approved": is_approved
    }

def critic_router(state: AgentState):
    if state.get("is_approved", False):
        return END
    else:
        # Check recursion limit
        if state.get("error_count", 0) >= 5:
            return END
        return "executor"

from langgraph.checkpoint.memory import MemorySaver

def build_workflow():
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
