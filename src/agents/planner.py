from langgraph.prebuilt import create_react_agent
from src.tools.metadata_extractor import metadata_extractor
from src.tools.rag_search import scientific_rag
from src.agents.llm_factory import get_llm

def get_planner_agent():
    llm = get_llm(agent_type="text", temperature=0.1)
    
    tools = [metadata_extractor, scientific_rag]
    
    system_prompt = """You are a Senior Neuroscientist and Lead Planner for an EEG data processing pipeline.
Your goal is to translate vague user descriptions into a concrete, technical MNE-Python analysis plan.
Use the `metadata_extractor` tool to read the raw EEG file headers to identify sampling rates, channels, and triggers.
Use the `scientific_rag` tool to query for standard parameters (e.g., filter bands, epoch windows) if the user did not specify them.

Output a structured Markdown "Analysis Plan" detailing the exact steps to be executed. Include any assumptions or RAG-inferred parameters.
Make sure to indicate when the plan is ready for review."""

    return create_react_agent(llm, tools, state_modifier=system_prompt)
