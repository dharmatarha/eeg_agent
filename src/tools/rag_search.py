import os
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain.storage import LocalFileStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.agents.llm_factory import get_embeddings

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
    try:
        embeddings = get_embeddings()
        db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chroma_data"))
        
        search_query = f"{paradigm} {query}".strip()
        results_text = []

        # 1. Search Scientific Methods (Layout Chunked + Summaries)
        if target in ["methods", "both"]:
            methods_store = Chroma(
                collection_name="neuroimage_methods",
                embedding_function=embeddings,
                persist_directory=db_dir
            )
            # Try to fetch
            methods_results = methods_store.similarity_search(search_query, k=2)
            if methods_results:
                results_text.append("=== Scientific Methods Findings ===")
                for i, doc in enumerate(methods_results):
                    summary = doc.metadata.get('global_summary', 'No summary available.')
                    results_text.append(f"--- Document {i+1} ---\nGlobal Summary: {summary}\nRelevant Excerpt:\n{doc.page_content}\n")

        # 2. Search API Documentation (Parent-Child Hierarchical)
        if target in ["api", "both"]:
            api_vectorstore = Chroma(
                collection_name="neuroimage_api",
                embedding_function=embeddings,
                persist_directory=db_dir
            )
            store_dir = os.path.join(db_dir, "docstore")
            
            if os.path.exists(store_dir):
                store = LocalFileStore(store_dir)
                retriever = ParentDocumentRetriever(
                    vectorstore=api_vectorstore,
                    docstore=store,
                    child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
                    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
                )
                api_results = retriever.invoke(search_query)
                # Since ParentDocumentRetriever returns full parent docs, we limit to 1-2 to save context
                if api_results:
                    results_text.append("=== MNE-Python API Documentation ===")
                    for i, doc in enumerate(api_results[:2]):
                        results_text.append(f"--- API Reference {i+1} ---\n{doc.page_content}\n")

        if not results_text:
            return "No relevant documents found in the Vector DB for this query. Proceed with standard heuristics or ask the user."
            
        return "\n".join(results_text)
        
    except Exception as e:
        return f"Error querying the Vector DB: {str(e)}. Proceed with standard heuristics."
