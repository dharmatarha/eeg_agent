from langgraph.prebuilt import create_react_agent
from src.tools.jupyter_exec import stateful_jupyter_exec
from src.tools.rag_search import scientific_rag
from src.agents.llm_factory import get_llm

def get_executor_agent():
    llm = get_llm(agent_type="text", temperature=0.2)
    
    # We include scientific_rag as doc_crawler equivalent if it needs API docs
    tools = [stateful_jupyter_exec, scientific_rag]
    
    system_prompt = """You are an Expert Python Developer specializing in MNE-Python.
Your goal is to write memory-safe code to execute the Analysis Plan provided by the Planner.
CRITICAL CONSTRAINTS:
1. You MUST use `mne.set_config('MNE_MEMMAP_MIN_SIZE', '10M')`.
2. You MUST use `preload=False` when loading raw data to avoid Out-Of-Memory (OOM) crashes.
3. You operate within a Stateful Jupyter Sandbox. Data loaded in previous turns remains in memory. Do not reload data if it is already in memory.
4. Use the `stateful_jupyter_exec` tool to run your code. 
5. If an error occurs (the tool returns error=True), analyze the traceback and rewrite the logic autonomously.
6. Generate visualizations (Base64) as requested by the plan.
Return a summary of what was executed and if any plots were generated."""

    return create_react_agent(llm, tools, state_modifier=system_prompt)
