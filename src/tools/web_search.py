import logging
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

logger = logging.getLogger("eeg_agent.tools.web_search")

@tool
def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo to retrieve API documentation, function syntax,
    or coding examples for Python packages (e.g., pandas, scipy, scikit-learn, numpy, matplotlib, etc.)
    or to troubleshoot error tracebacks.
    
    Args:
        query: The search query (e.g., "sklearn.preprocessing.RobustScaler example", "pandas dataframe insert column").
    """
    logger.info("Web search query received: %s", query)
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        logger.info("Web search complete.")
        return result
    except Exception as e:
        logger.error("Error executing web search: %s", e, exc_info=True)
        return f"Error executing web search: {str(e)}. Please proceed with local heuristics or scientific_rag."
