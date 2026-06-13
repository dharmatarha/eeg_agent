import operator
from typing import TypedDict, Annotated, List, Any

class AgentState(TypedDict):
    user_directive: str
    data_path: str
    raw_metadata: str
    analysis_plan: str
    
    # Executor state
    execution_logs: Annotated[List[str], operator.add]
    generated_plots: Annotated[List[str], operator.add] # Base64 images
    error_count: int
    
    # Critic state
    critic_feedback: str
    is_approved: bool

    # Tracking & Audit
    rag_history: Annotated[List[dict], operator.add]
    executed_code_blocks: Annotated[List[dict], operator.add]
