from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState
from src.agents.planner import get_planner_agent
from src.agents.executor import get_executor_agent
from src.agents.critic import get_critic_agent
import json
import logging
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

logger = logging.getLogger("eeg_agent.workflow")

def extract_tool_trace(messages):
    rag_history = []
    executed_code_blocks = []
    
    # Map to hold tool calls by their unique call ID for matching with execution outputs
    tool_calls = {}
    
    for msg in messages:
        if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls[tc["id"]] = tc
        elif msg.type == "tool":
            call_id = getattr(msg, "tool_call_id", None)
            tool_call = tool_calls.get(call_id)
            tool_name = getattr(msg, "name", "") or (tool_call["name"] if tool_call else "")
            
            if tool_name == "scientific_rag":
                query = tool_call["args"].get("query", "") if tool_call else "Unknown query"
                paradigm = tool_call["args"].get("paradigm", "") if tool_call else ""
                target = tool_call["args"].get("target", "both") if tool_call else "both"
                rag_history.append({
                    "query": query,
                    "paradigm": paradigm,
                    "target": target,
                    "results": msg.content
                })
            elif tool_name == "stateful_jupyter_exec":
                code = tool_call["args"].get("code_string", "") if tool_call else ""
                try:
                    res = json.loads(msg.content)
                    logs = res.get("logs", "")
                    error = res.get("error", False)
                except Exception:
                    logs = msg.content
                    error = True
                
                executed_code_blocks.append({
                    "code": code,
                    "logs": logs,
                    "error": error
                })
                
    return rag_history, executed_code_blocks

def normalize_content(content) -> str:
    """Normalize message content to a clean string, resolving lists of blocks."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return "".join(text_parts)
    return str(content)

def detect_repetition(text: str) -> bool:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return False
    
    # 1. Check for consecutive identical lines (e.g. 3 repeats)
    consecutive_repeats = 0
    for idx in range(len(lines) - 1):
        if lines[idx] == lines[idx + 1] and len(lines[idx]) > 20:
            consecutive_repeats += 1
            if consecutive_repeats >= 3:
                return True
        else:
            consecutive_repeats = 0
            
    # 2. Check for overall frequency of long lines (e.g. repeated 5+ times anywhere)
    from collections import Counter
    line_counts = Counter(lines)
    for line, count in line_counts.items():
        if len(line) > 30 and count >= 5:
            return True
            
    return False

def planner_node(state: AgentState, config=None):
    logger.info("Planner Node: Starting plan generation...")
    thread_id = config["configurable"].get("thread_id") if config else None
    planner = get_planner_agent(thread_id=thread_id)
    
    prompt = ""
    ref_run = state.get("reference_run")
    if ref_run:
        prompt += f"REFERENCE RUN MEMORY (use for consistency/parameters if compatible):\n{json.dumps(ref_run, indent=2)}\n\n"
        
    planner_feedback = state.get("planner_feedback")
    previous_plan = state.get("analysis_plan")
    
    if planner_feedback and previous_plan:
        logger.info("Planner Node: Incorporating user feedback for plan revision.")
        prompt += (
            f"User Directive: {state['user_directive']}\n"
            f"Data Path: {state['data_path']}\n\n"
            f"CURRENT ANALYSIS PLAN:\n{previous_plan}\n\n"
            f"USER FEEDBACK / REQUESTED REVISIONS:\n{planner_feedback}\n\n"
            f"Please revise the current Analysis Plan incorporating the user's feedback."
        )
    else:
        prompt += f"User Directive: {state['user_directive']}\nData Path: {state['data_path']}\n\nPlease generate the Analysis Plan."
    
    max_attempts = 3
    final_message = ""
    result = None
    
    for attempt in range(1, max_attempts + 1):
        logger.info("Planner Node: Plan generation attempt %d/%d...", attempt, max_attempts)
        result = planner.invoke({"messages": [HumanMessage(content=prompt)]})
        
        if not result or "messages" not in result or not result["messages"]:
            logger.warning("Planner Node: Attempt %d returned empty messages.", attempt)
            continue
            
        last_msg = result["messages"][-1]
        final_message = last_msg.content
        plan_str = normalize_content(final_message)
        
        # Check metadata for malformed finish reason
        metadata = getattr(last_msg, "response_metadata", {}) or {}
        finish_reason = metadata.get("finish_reason", "")
        
        is_malformed = (
            finish_reason == "MALFORMED_FUNCTION_CALL" or
            plan_str.strip() == "}" or
            len(plan_str.strip()) < 100 or
            len(plan_str.strip()) > 25000 or
            detect_repetition(plan_str)
        )
        
        if not is_malformed:
            logger.info("Planner Node: Plan generation completed successfully on attempt %d.", attempt)
            break
        else:
            logger.warning(
                "Planner Node: Attempt %d produced malformed plan (finish_reason: %s, len: %d). Plan snippet: %s",
                attempt, finish_reason, len(plan_str), repr(plan_str[:50])
            )
            
    logger.info("Planner Node: Finalizing plan...")
    rag_history = []
    if result and "messages" in result:
        rag_history, _ = extract_tool_trace(result["messages"])
        
    return {
        "analysis_plan": normalize_content(final_message),
        "rag_history": rag_history,
        "planner_feedback": ""
    }


def executor_node(state: AgentState, config=None):
    logger.info("Executor Node: Starting code generation and execution in Docker Sandbox...")
    thread_id = config["configurable"].get("thread_id") if config else None
    executor = get_executor_agent(thread_id=thread_id)
    
    prompt = ""
    ref_run = state.get("reference_run")
    if ref_run:
        prompt += f"REFERENCE RUN MEMORY:\n{json.dumps(ref_run, indent=2)}\n\n"
        
    prompt += f"Here is the Analysis Plan to execute:\n{state['analysis_plan']}\n\n"
    if state.get("critic_feedback"):
        logger.info("Executor Node: Incorporating Critic feedback from previous iteration.")
        prompt += f"CRITIC FEEDBACK FROM PREVIOUS RUN: {state['critic_feedback']}\nPlease adjust your code accordingly.\n"
        
    result = executor.invoke({"messages": [HumanMessage(content=prompt)]})
    
    # Extract tool traces
    rag_history, executed_code_blocks = extract_tool_trace(result["messages"])
    
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
            except Exception:
                pass

    logger.info(
        "Executor Node: Finished execution. Success=%s. Generated %d plots.",
        not error_occurred,
        len(images)
    )
    current_errors = state.get("error_count", 0)
    return {
        "execution_logs": logs,
        "generated_plots": images,
        "error_count": current_errors + 1 if error_occurred else current_errors,
        "rag_history": rag_history,
        "executed_code_blocks": executed_code_blocks
    }


def critic_node(state: AgentState, config=None):
    logger.info("Critic Node: Invoking QA / review agent...")
    thread_id = config["configurable"].get("thread_id") if config else None
    critic = get_critic_agent(thread_id=thread_id)
    feedback = normalize_content(critic(state))
    
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

def approval_gate_node(state: AgentState, config=None):
    logger.info("Approval Gate Node: Evaluating plan approval state...")
    return {}

def approval_router(state: AgentState):
    is_approved = state.get("is_approved", False)
    logger.info("Approval Router: checking plan approval. approved=%s", is_approved)
    if is_approved:
        return "executor"
    else:
        return "planner"

def build_workflow(checkpointer=None):
    logger.info("Building StateGraph workflow...")
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("approval_gate", approval_gate_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("critic", critic_node)
    
    workflow.set_entry_point("planner")
    
    # Transition to approval_gate, and interrupt before it to get user feedback
    workflow.add_edge("planner", "approval_gate")
    workflow.add_conditional_edges("approval_gate", approval_router, {
        "planner": "planner",
        "executor": "executor"
    })
    
    workflow.add_edge("executor", "critic")
    
    workflow.add_conditional_edges("critic", critic_router, {
        "executor": "executor",
        END: END
    })
    
    if checkpointer is None:
        # Setup persistent SQLite checkpoints
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        db_path = os.path.join(logs_dir, "checkpoints.sqlite")
        
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        
    return workflow.compile(interrupt_before=["approval_gate"], checkpointer=checkpointer)

