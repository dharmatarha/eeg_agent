import os
import logging
from langchain_core.tools import tool
from langchain_chroma import Chroma
try:
    from langchain.storage import LocalFileStore, create_kv_docstore
except ModuleNotFoundError:
    from langchain_classic.storage import LocalFileStore, create_kv_docstore
try:
    from langchain.retrievers import ParentDocumentRetriever
except ModuleNotFoundError:
    from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.agents.llm_factory import get_embeddings
from src import config

logger = logging.getLogger("eeg_agent.tools.rag_search")

@tool
def scientific_rag(query: str, paradigm: str = "", target: str = "both") -> str:
    """
    Search the offline Vector DB for standard EEG processing parameters and MNE API documentation.
    
    Args:
        query: The search query (e.g., "N400 bandpass filter", "mne.Epochs parameters")
        paradigm: Optional paradigm name (e.g., "P300", "N400") to narrow search context.
        target: Which knowledge base to search. Use "methods" for scientific papers/parameters, 
                "api" for MNE-Python code/functions, or "both" (default).
    """
    logger.info(
        "RAG search query received. target: %s, paradigm: %s, query: %s",
        target, paradigm, query
    )
    try:
        embeddings = get_embeddings()
        db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chroma_data"))
        
        search_query = f"{paradigm} {query}".strip()
        results_text = []

        # 1. Search Scientific Methods (Layout Chunked + Summaries)
        if target in ["methods", "both"]:
            logger.debug("Querying 'neuroimage_methods' collection for: %s", search_query)
            methods_store = Chroma(
                collection_name="neuroimage_methods",
                embedding_function=embeddings,
                persist_directory=db_dir
            )
            # Try to fetch
            methods_k = int(config.get_val("retrieval.methods_k"))
            methods_results = methods_store.similarity_search(search_query, k=methods_k)
            logger.info("Found %d documents in methods collection.", len(methods_results))
            if methods_results:
                results_text.append("=== Scientific Methods Findings ===")
                for i, doc in enumerate(methods_results):
                    summary = doc.metadata.get('global_summary', 'No summary available.')
                    results_text.append(f"--- Document {i+1} ---\nGlobal Summary: {summary}\nRelevant Excerpt:\n{doc.page_content}\n")

        # 2. Search API Documentation (Parent-Child Hierarchical)
        if target in ["api", "both"]:
            logger.debug("Querying 'neuroimage_api' collection for: %s", search_query)
            api_vectorstore = Chroma(
                collection_name="neuroimage_api",
                embedding_function=embeddings,
                persist_directory=db_dir
            )
            store_dir = os.path.join(db_dir, "docstore")
            
            if os.path.exists(store_dir):
                fs = LocalFileStore(store_dir)
                store = create_kv_docstore(fs)
                child_chunk_size = int(config.get_val("ingestion.api_docs.child_chunk_size"))
                child_chunk_overlap = int(config.get_val("ingestion.api_docs.child_chunk_overlap"))
                retriever = ParentDocumentRetriever(
                    vectorstore=api_vectorstore,
                    docstore=store,
                    child_splitter=RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap),
                    parent_splitter=None,
                )
                api_results = retriever.invoke(search_query)
                logger.info("Found %d parent documents in API collection.", len(api_results))
                # Since ParentDocumentRetriever returns full parent docs, we limit to conserve context
                api_k = int(config.get_val("retrieval.api_k"))
                if api_results:
                    results_text.append("=== MNE-Python API Documentation ===")
                    for i, doc in enumerate(api_results[:api_k]):
                        results_text.append(f"--- API Reference {i+1} ---\n{doc.page_content}\n")
            else:
                logger.warning("API docstore path does not exist: %s", store_dir)

        if not results_text:
            logger.info("No RAG results found matching the query.")
            return "No relevant documents found in the Vector DB for this query. Proceed with standard heuristics or ask the user."
            
        return "\n".join(results_text)
        
    except Exception as e:
        logger.error("Error querying the Vector DB: %s", e, exc_info=True)
        return f"Error querying the Vector DB: {str(e)}. Proceed with standard heuristics."
