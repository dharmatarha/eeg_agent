import os
import sys
import json

# Locate project root (assuming src/ is at project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")

# Default configurations (matching the original code defaults to ensure test compatibility)
DEFAULT_CONFIG = {
    "llm_provider": "vllm",
    "gemini_model": "gemini-1.5-pro",
    "vllm_api_base": "http://localhost:8000/v1",
    "vllm_api_key": "EMPTY",
    "vllm_model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "vlm_model": "llava-hf/llava-1.5-7b-hf",
    "embedding_provider": "vllm",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "ingestion": {
        "articles": {
            "chunk_size": 1500,
            "chunk_overlap": 300,
            "summary_max_chars": 10000
        },
        "books": {
            "chunk_size": 2000,
            "chunk_overlap": 300
        },
        "api_docs": {
            "parent_chunk_size": 2000,
            "parent_chunk_overlap": 200,
            "child_chunk_size": 400,
            "child_chunk_overlap": 50
        }
    },
    "retrieval": {
        "methods_k": 2,
        "api_k": 2
    },
    "sandbox": {
        "gateway_url": "http://localhost:8888",
        "ws_gateway_url": "ws://localhost:8888",
        "jupyter_token": "eeg_adk_sandbox_token"
    },
    "planner": {
        "temperature": 0.0
    },
    "executor": {
        "max_retries": 5
    }
}

config = DEFAULT_CONFIG.copy()

# Detect if we are running unit tests
is_testing = (
    "pytest" in sys.modules or
    "PYTEST_CURRENT_TEST" in os.environ or
    any("pytest" in arg for arg in sys.argv)
)

if not is_testing and os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            user_config = json.load(f)
            def merge_configs(base, update):
                for k, v in update.items():
                    if isinstance(v, dict) and k in base:
                        merge_configs(base[k], v)
                    else:
                        base[k] = v
            merge_configs(config, user_config)
    except Exception as e:
        pass

def get_val(key, env_var=None):
    """
    Get a configuration value, checking environment variables first for overrides.
    """
    if env_var and os.environ.get(env_var) is not None:
        return os.environ.get(env_var)
        
    if "." in key:
        parts = key.split(".")
        current = config
        for p in parts:
            if isinstance(current, dict) and p in current:
                current = current[p]
            else:
                return DEFAULT_CONFIG.get(key)
        return current
        
    return config.get(key)
