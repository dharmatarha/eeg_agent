# Configuration Guide

The EEG-ADK Multi-Agent System loads settings from `config.json` at the root of the project. If a setting is omitted or the file is missing, the system falls back to default values defined in `src/config.py`.

## Core Settings

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `llm_provider` | `string` | `"vllm"` | Choices: `"gemini"` or `"vllm"`. Sets the default backend for planning, coding, and review agents. |
| `gemini_model` | `string` | `"gemini-1.5-pro"` | The Gemini model name to use when the provider is `"gemini"`. |
| `vllm_api_base` | `string` | `"http://localhost:8000/v1"` | The API base URL for local open-source LLM inferences (vLLM instance). |
| `vllm_api_key` | `string` | `"EMPTY"` | API authorization token for the vLLM instance. |
| `vllm_model` | `string` | `"mistralai/Mixtral-8x7B-Instruct-v0.1"` | Model name targeted on the vLLM API server. |
| `vlm_model` | `string` | `"llava-hf/llava-1.5-7b-hf"` | Vision-Language Model name (VLM) for the Critic agent. |
| `embedding_provider` | `string` | `"vllm"` | Choices: `"gemini"` or `"vllm"`. Sets the provider for text embedding calculations. |
| `embedding_model` | `string` | `"BAAI/bge-small-en-v1.5"` | Embedding model identifier used for RAG database indexing and searches. |

---

## RAG Ingestion Settings (`ingestion`)

Controls the chunking parameters for populating the ChromaDB vector collections under `rag_docs/`.

* **`articles`** (Scientific Papers)
  * `chunk_size` (default: `1500`): Character chunk size.
  * `chunk_overlap` (default: `300`): Chunk overlap size.
  * `summary_max_chars` (default: `10000`): Size limit for generated paper summaries.
* **`books`** (Reference Textbooks)
  * `chunk_size` (default: `2000`): Character chunk size.
  * `chunk_overlap` (default: `300`): Chunk overlap size.
* **`api_docs`** (MNE-Python API Documentation)
  * `parent_chunk_size` (default: `2000`): Large parent document chunk size.
  * `parent_chunk_overlap` (default: `200`): Parent document overlap.
  * `child_chunk_size` (default: `400`): Small child search chunk size.
  * `child_chunk_overlap` (default: `50`): Child chunk overlap.

---

## Retrieval Settings (`retrieval`)

* `methods_k` (default: `2`): Number of scientific paper chunks to retrieve for context in the Planner.
* `api_k` (default: `2`): Number of API reference documentation chunks to retrieve for debugging in the Executor.

---

## Sandbox Settings (`sandbox`)

* `gateway_url` (default: `"http://localhost:8888"`): Address of the Jupyter Gateway service.
* `ws_gateway_url` (default: `"ws://localhost:8888"`): WebSocket address of the Jupyter Gateway.
* `jupyter_token` (default: `"eeg_adk_sandbox_token"`): Token to authorize commands to the Jupyter Gateway.

---

## Agent Specific Settings

* **`planner`**
  * `temperature` (default: `0.0`): LLM sampling temperature for plan proposals.
* **`executor`**
  * `max_retries` (default: `5`): Maximum retry count for self-debugging loops.
