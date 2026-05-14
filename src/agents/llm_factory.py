import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(agent_type="text", temperature=0.1):
    """
    Factory function to get the appropriate LLM based on environment configuration.
    Supports 'vllm' (default, OpenAI-compatible) and 'gemini'.
    """
    provider = os.environ.get("LLM_PROVIDER", "vllm").lower()
    
    if provider == "gemini":
        # Gemini handles multimodal transparently with the same model (e.g. gemini-1.5-pro)
        model = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
    else:
        # Default to vLLM (OpenAI compatible)
        base_url = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
        api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
        
        if agent_type == "multimodal":
            model = os.environ.get("VLM_MODEL", "llava-hf/llava-1.5-7b-hf")
        else:
            model = os.environ.get("VLLM_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
            
        return ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature
        )

def get_embeddings():
    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", os.environ.get("LLM_PROVIDER", "vllm")).lower()
    
    if embedding_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ.get("GOOGLE_API_KEY"))
    elif embedding_provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        # BAAI/bge-small-en-v1.5 is a fast, highly-rated open source embedding model
        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        return HuggingFaceEmbeddings(model_name=model_name)
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            base_url=os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1"),
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        )
