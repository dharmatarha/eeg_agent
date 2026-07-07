# EEG-ADK Multi-Agent System: Agent Roles and Orchestration

This document outlines the architecture, roles, and interactions of the autonomous agents within the EEG-ADK Multi-Agent System, and how they interface with the web-based runtime and frontend client.

The system is built on a **State-Graph orchestration framework** (powered by LangGraph) combined with Google ADK methodology. This ensures memory-safe operations, recursive debugging, and verifiable quality assurance for complex MNE-Python workflows.

---

## 1. Core Agents

The system delegates specialized tasks to three primary agents, each configured with highly specific system prompts and tools.

```
       +-------------------------------------------------------------+
       |                  EEG-ADK Analysis Studio (UI)               |
       +------------------------------+------------------------------+
                                      |
                         REST & WebSocket Protocol
                                      v
       +-------------------------------------------------------------+
       |                     FastAPI Bridge Server                   |
       |                   (src/web/server.py Node)                  |
       +------------------------------+------------------------------+
                                      |
                            LangGraph State Machine
                                      |
      +-------------------------------+-------------------------------+
      |                               |                               |
      v                               v                               v
🧠 Lead Planner                💻 Executor                     👁️ Critic
(src/agents/planner.py)       (src/agents/executor.py)        (src/agents/critic.py)
      |                               |                               |
      +--> metadata_extractor         +--> stateful_jupyter_exec      +--> Visual SNR
      +--> bids_inspector             +--> web_search                 +--> Matplotlib checks
      +--> scientific_rag             +--> scientific_rag             +--> Audit methods
```

### 🧠 Lead Planner (Strategist)
- **Role:** Neuroinformatics & Workflow Design
- **Source File:** `src/agents/planner.py`
- **Core Function:** Translates vague user descriptions (e.g., "clean the N400 data") into concrete, technical MNE-Python analysis plans.
- **Key Behaviors:** 
  - Scans raw data file headers using the `metadata_extractor` tool (for single files) or the `bids_inspector` tool (for BIDS datasets/directories) to understand subjects, tasks, channels, and triggers without fully loading the data.
  - Queries the offline Vector Database (`scientific_rag`) to automatically populate missing standard parameters (like standard filter cutoffs).
  - Outlines multi-subject loop and grand average architectures for group analyses.
  - Outputs a structured Analysis Plan that is presented to the user for Human-in-the-Loop (HITL) approval (via terminal inputs or custom Next.js UI cards) before any code executes.

### 💻 Executor (Programmer)
- **Role:** Code Generation & Execution
- **Source File:** `src/agents/executor.py`
- **Core Function:** Writes and runs MNE-Python scripts to fulfill the Analysis Plan.
- **Key Behaviors:**
  - Operates safely inside a **Stateful Jupyter Sandbox** via the `stateful_jupyter_exec` WebSocket tool. Data loaded in previous turns is retained in memory.
  - Adheres strictly to memory-safe constraints (`preload=False` and `MNE_MEMMAP_MIN_SIZE='10M'`) to prevent Out-Of-Memory (OOM) crashes on local hardware.
  - Recursively debugs and self-corrects: If an execution returns an error traceback, it analyzes the error, consults RAG or queries the internet via `web_search`, and rewrites the logic autonomously.

### 👁️ Critic (Reviewer & QA)
- **Role:** Quality Assurance & Reviewer
- **Source File:** `src/agents/critic.py`
- **Core Function:** Acts as a Multimodal Vision-Language Model (VLM) reviewer that validates the Executor's results, code correctness, and plan adherence.
- **Key Behaviors:**
  - Reviews the original `user_directive`, the `analysis_plan`, the exact `executed_code_blocks`, the `execution_logs`, and visually inspects generated Base64 plots.
  - Performs critical checks:
    - **Adherence to Plan:** Ensures the executed code conforms to the planned methodology.
    - **Memory Safety & Sandbox constraints:** Checks that `preload=False`, `MNE_MEMMAP_MIN_SIZE` configuration, and loop-level memory management (`gc.collect()`, Matplotlib closure) are followed.
    - **Library Auditing:** Audits imports to ensure only supported libraries (`mne`, `mne-bids`, `mne-connectivity`) are used.
    - **SNR & Artifact Validation:** Audits plots for ocular, muscle, and bad-channel artifacts.
  - Returns `REJECT` alongside detailed feedback if any code checks fail or artifacts remain, routing control back to the Executor.
  - Returns `APPROVE` when acceptable, synthesizing a final, manuscript-ready Methods & Results section based on the actual executed code.

---

## 2. Agent Tools

The agents interact with the environment and external knowledge bases using seven specialized tools:

1. **`metadata_extractor` (`src/tools/metadata_extractor.py`)**: A fast utility that peeks into raw EEG file headers (supporting `.fif`, `.set`, `.edf`, `.bdf`, `.vhdr`) to extract critical dimensions, classify channel types (EEG vs EOG vs Stimulus), and extract normalized events/annotations without loading large binary arrays.
2. **`bids_inspector` (`src/tools/bids_inspector.py`)**: A directory-scanning microservice that parses BIDS dataset formats (`dataset_description.json`, `sub-*` folders), extracts subject lists, and captures representative EEG headers.
3. **`scientific_rag` (`src/tools/rag_search.py`)**: An offline RAG system powered by ChromaDB. It searches dual collections: scientific neuroimaging methods (including methods sections from EEG papers and textbooks for best-practice processing conventions) and the MNE-Python API documentation (for syntax).
4. **`dataset_explorer` (`src/tools/dataset_explorer.py`)**: A dataset utility that lists directories recursively matching patterns, reads text configuration files (READMEs, JSON/TSV sidecars) up to size limits, and performs automatic consistency checks across files (validating matching sampling rates, channel counts, and names).
5. **`stateful_jupyter_exec` (`src/tools/jupyter_exec.py`)**: A WebSocket client that sends code strings to a persistent Dockerized Jupyter Kernel gateway, allowing sequential analysis steps while capturing text outputs, error tracebacks, and Base64 encoded plots.
6. **`web_search` (`src/tools/web_search.py`)**: A web search utility powered by DuckDuckGo that allows the Executor agent to retrieve API documentation, function syntax, and coding examples for third-party libraries (e.g. pandas, scikit-learn, scipy) and debug sandbox errors.
7. **`read_reference_run_file` (`src/tools/reference_run_reader.py`)**: A utility that allows the Planner and Executor agents to inspect previous reference runs' code execution scripts, final reports, and run memories.

---

## 3. Orchestration & State Graph

The interaction between the agents is strictly governed by a LangGraph state machine (`src/graph/workflow.py`). 

1. **Initialization:** The state initializes with the user's `user_directive`, extracted `raw_metadata`, and an optional `reference_run` memory object parsed from a past session.
2. **Planning:** The Planner formulates the `analysis_plan`.
3. **Human-in-the-Loop (HITL):** Execution explicitly pauses here. The user must review the plan:
   * **In CLI Mode:** The user is prompted in the shell to approve or type corrective feedback.
   * **In Web UI Mode:** The state is set to `awaiting_hitl`. The FastAPI backend streams this state change over WebSockets. The frontend UI renders an interactive **Plan Review Card** containing `Approve` and `Request Changes` (with a text area for comments) buttons. 
   * **Plan Revision Loop:** When the user clicks `Request Changes` and provides natural language feedback, the backend updates the graph state with `planner_feedback` and loops back to the **Lead Planner** agent to generate a revised analysis plan. When `Approve` is clicked, the state is marked as approved, the revision loop exits, and the execution proceeds to the **Executor**.
4. **Execution Loop:** 
   - The Executor generates and runs the code.
   - The Critic reviews the outputs.
   - If the Critic rejects the output (e.g., poor data quality or unhandled errors), control loops back to the Executor. This recursive loop has a hardcoded limit (e.g., 5 retries) to prevent infinite loops.
5. **Finalization:** Once the Critic approves, the graph terminates, and the final state (containing the exact methodology and plots) is saved as a complete report along with a structured `run_memory.json` memory file.

---

## 4. LLM Factory Architecture

To maintain flexibility between local deployment and cloud scalability, agent initialization is routed through an LLM Factory (`src/agents/llm_factory.py`).

By configuring environment variables (e.g., `LLM_PROVIDER`), the agents can seamlessly switch between:
- **Local / Open-Source Backend:** vLLM instances running models like Mixtral (for Planner/Executor) and LLaVA (for the Multimodal Critic).
- **Cloud Backend:** Google Gemini endpoints (e.g., `gemini-1.5-pro` or `gemini-3.5-flash`), natively supporting both text and multimodal prompts across all agents.

---

## 5. State Persistence & Audit Trails

To ensure maximum transparency, reproducibility, and auditability, the system replaces ephemeral in-memory checkpointing with persistent session logging:

1. **Persistent Checkpoints (`logs/checkpoints.sqlite`):** All execution states, variable bounds, and agent decisions are serialized into a local SQLite database using LangGraph's `SqliteSaver`. Each execution session is isolated using a unique `thread_id` (e.g., `run_YYYYMMDD_HHMMSS_uuid`).
2. **Web Session Hydration:** When a client opens a session page (e.g. `http://localhost:3000/run/{thread_id}`), the FastAPI backend queries the SQLite checkpoints, parses the latest state dictionary, and hydates the frontend client to restore the entire conversation thread, current phase, and all generated plots.
3. **RAG Provenance Auditing:** All scientific queries, retrieval contexts, and documentation excerpts populated by `scientific_rag` are tracked inside the `rag_history` list. This enables researchers to verify the origin of all inferred parameters.
4. **Executable Code Aggregation (`output/analysis_pipeline.py`):** Successful Python code blocks executed in the stateful Jupyter sandbox are compiled into a standalone, reproducible MNE-Python script at the end of the execution run.
5. **Comprehensive Audit Reporting (`output/final_report.md`):** Consolidates the original user directive, extracted raw metadata, RAG logs (with collapsible source blocks), full sandbox code execution traces, and critic reviews into a unified report.
