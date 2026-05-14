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
