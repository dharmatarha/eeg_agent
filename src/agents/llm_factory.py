import os
import torch
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src import config

def get_llm(agent_type="text", temperature=None):
    """
    Factory function to get the appropriate LLM based on configuration.
    Supports 'vllm' (default, OpenAI-compatible) and 'gemini'.
    """
    provider = config.get_val("llm_provider", "LLM_PROVIDER").lower()
    
    if temperature is None:
        temperature = float(config.get_val("planner.temperature"))
        
    if provider == "gemini":
        # Gemini handles multimodal transparently with the same model
        model = config.get_val("gemini_model", "GEMINI_MODEL")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
    else:
        # Default to vLLM (OpenAI compatible)
        base_url = config.get_val("vllm_api_base", "VLLM_API_BASE")
        api_key = config.get_val("vllm_api_key", "VLLM_API_KEY")
        
        if agent_type == "multimodal":
            model = config.get_val("vlm_model", "VLM_MODEL")
        else:
            model = config.get_val("vllm_model", "VLLM_MODEL")
            
        return ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature
        )

def get_embeddings():
    embedding_provider = config.get_val("embedding_provider", "EMBEDDING_PROVIDER").lower()
    model_name = config.get_val("embedding_model", "EMBEDDING_MODEL")
    
    if embedding_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=os.environ.get("GOOGLE_API_KEY"))
    elif embedding_provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Determine optimal device and precision options
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"device": device, "trust_remote_code": True}
        
        if device == "cuda":
            # Force half-precision (bfloat16 or float16) to conserve GPU memory
            if torch.cuda.is_bf16_supported():
                model_kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
            else:
                model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        
        # Keep batch_size low to prevent CUDA OutOfMemory on large texts/books
        encode_kwargs = {"batch_size": 4}
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            base_url=config.get_val("vllm_api_base", "VLLM_API_BASE"),
            api_key=config.get_val("vllm_api_key", "VLLM_API_KEY"),
            model=model_name
        )
