import operator
from typing import TypedDict, Annotated, List, Any

class AgentState(TypedDict):
    user_directive: str
    data_path: str
    raw_metadata: str
    analysis_plan: str
    reference_run: Any
    
    # Executor state
    execution_logs: List[str]
    generated_plots: List[str] # Base64 images
    error_count: int
    
    # Critic state
    critic_feedback: str
    is_approved: bool

    # Planner feedback (HITL plan revisions)
    planner_feedback: str

    # Tracking & Audit
    rag_history: Annotated[List[dict], operator.add]
    executed_code_blocks: Annotated[List[dict], operator.add]
