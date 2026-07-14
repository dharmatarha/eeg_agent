# **Blueprint: Actionable EEG-ADK Multi-Agent System**

**Version:** 5.0 (Containerized Microservices & Web Interface)  
**Target:** High-Performance Local Workstation (License-Free / CPU or GPU)  
**Orchestration Paradigm:** State-Graph Multi-Agent Orchestration (powered by LangGraph) integrated with Google ADK, exposed via FastAPI, and controlled through a React/Next.js Web UI.

---

## **1\. Core Objectives: EEG-ADK Methodology**

The primary goal of this multi-agent architecture is to mitigate the inherent instability and reproducibility crisis in conventional EEG workflows. Key drivers include:

1. **Synthesizing Procedural Automation:** Reducing researcher overhead by automating the generation of boilerplate code for ingestion, signal filtering, epoching, and visualization.  
2. **Intelligent Parameter Inference:** Leveraging scientific RAG to function as an expert assistant, autonomously populating standardized research parameters (such as P300 filter specifications) when user input is underspecified.  
3. **Adaptive Script Rectification:** Maintaining execution flow through iterative self-debugging; the system dynamically resolves syntax errors or memory bottlenecks rather than halting at the first failure.  
4. **Verified Auditability & Persistence:** Ensuring every computational decision and inference is persisted to SQLite (`logs/checkpoints.sqlite`), indexing active and completed runs in the database to enable instant sidebar navigation, and producing a standalone, runnable script (`output/analysis_pipeline.py`) alongside a manuscript-ready final audit report (`output/final_report.md`) that documents exact RAG queries, code blocks executed, and validation steps.
5. **Zero-Setup Containerization:** Utilizing Docker Compose to bundle the frontend, backend, vector database, and sandbox environment into a single, portable application requiring zero configuration.

---

## **2\. System Architecture & Containerized Services**

The system operates as a decoupled microservices architecture coordinated via a Docker network:

* **`frontend` (Next.js client-server on port 3000):** A custom dark-themed dashboard written in TypeScript/React using `next/font/google` for optimized font rendering and `assistant-ui`'s `ExternalStoreRuntime`.
* **`backend` (FastAPI bridge on port 8000):** Coordinates the LangGraph state machine execution, WebSocket streaming of state transitions, REST API endpoints for browsing files, and state database checkpoints hydration.
* **`sandbox` (Jupyter Kernel Gateway on port 8888):** Provides a stateful, isolated workspace containing MNE-Python, MNE-BIDS, and MNE-Connectivity. Retains variables and loaded datasets in RAM across multiple execution steps.
* **`chromadb` (Vector Store on port 8001):** Holds the RAG knowledge collections for paper methods and API syntax. The backend connects directly to its local persist directory, while standalone services can query it.

---

## **3\. System Architecture: State-Graph Orchestration**

The orchestration is built on a **State-Graph framework**, where a shared "State" dictionary is maintained and updated by each node, while Google ADK defines the individual agents and their tools.

* **State Object:** Maintains fields for `user_directive`, `data_path`, `raw_metadata`, `analysis_plan`, `execution_logs`, `generated_plots`, `error_count`, `critic_feedback`, `is_approved`, as well as tracking fields `rag_history` (RAG citations) and `executed_code_blocks` (sandbox execution logs).  
* **Persistence & Session Isolation:** Uses LangGraph `SqliteSaver` to maintain state checkpoints across workflow interruptions, along with a sqlite-backed `runs_index` table to register run statuses instantly for the client sidebar. Each execution uses a unique timestamped thread ID (`run_YYYYMMDD_HHMMSS_uuid`) for clean session isolation.
* **Recursion Limits:** Hardcoded execution caps (e.g., maximum 5 retries for the Executor to fix an error) to prevent burning compute indefinitely. If the limit is reached, the graph halts and requests Human-in-the-Loop (HITL) intervention.

---

## **4\. Multi-Agent Architecture (Core Agents)**

| Agent Role | Primary Responsibilities | Key Behaviors & Constraints |
| :---- | :---- | :---- |
| **Lead Planner (Strategist)** | Neuroinformatics & Workflow Design | Translates user methods into a technical pipeline; halts for user approval before execution; uses offline RAG to infer missing parameters. |
| **Executor (Programmer)** | Code Generation & Execution | Writes MNE-Python scripts; operates within a Stateful Jupyter Sandbox; implements memory-safe loading; self-corrects based on error logs and web searches. |
| **Critic (Reviewer)** | Quality Assurance & Reviewer (VLM) | Must be a Multimodal Vision-Language Model. Reviews user directive, plan, code, logs, and plots. Validates results against the plan, checks memory-safety constraints, audits libraries, and inspects plots for SNR/artifacts; synthesizes Methods and Results. |

---

## **5\. Refined Tool Specifications**

The following tools are registered via Google ADK to facilitate high-fidelity EEG processing.

| Tool Name | Input Parameters | Realistic Implementation Strategy |
| :---- | :---- | :---- |
| **scientific\_rag & doc\_crawler** | query (str), paradigm (str) | Build an offline Vector Database (ChromaDB/FAISS) pre-populated with standard Neuroimage methods papers and the official MNE-Python API documentation. Avoids live API rate limits and hallucinations. |
| **stateful\_jupyter\_exec** | code\_string (str) | Connects to a Dockerized Kernel Gateway via WebSocket. Returns text/plain (logs) and image/png (Base64 data). Runs data-loading in "Turn 1" and filtering in "Turn 2" while data remains safely in RAM. |
| **metadata\_extractor** | file\_path (str) | A pre-written Python microservice using mne.io.read\_raw(..., preload=False).info to immediately dump channels, sampling frequency, and annotations to JSON. |
| **bids\_inspector** | bids\_root (str) | A directory-scanning microservice that crawls subject structures, parses dataset configurations, and extracts representative MNE raw headers. |
| **web\_search** | query (str) | DuckDuckGo search API client to query API docs or troubleshoot error messages inside the sandbox. |
| **read\_reference\_run\_file** | run\_id (str), filename (str) | Inspects previous reference runs' code execution scripts and final reports to reuse parameters. |

---

## **6\. Operational Workflow Execution**

The practical deployment of the system follows a **Human-in-the-Loop (HITL)** state-graph trajectory as detailed below:

**Step 1: Data & Task Ingestion**
* The researcher opens the web dashboard, browses files mounted under `data/`, and selects a file or BIDS root.
* The researcher types a high-level natural language directive and (optionally) selects a past Run ID to reference.

**Step 2: Analysis Planning & RAG Research**
* The **Planner Agent** scans file metadata for triggers, channel lists, and sampling rates.
* If bandpass/parameter requirements are missing, the system queries the local vector DB to integrate standards into a formal Analysis Plan.
* A structured markdown Analysis Plan is generated.

**Step 3: Human-in-the-Loop Validation**
* State execution pauses and transitions to `awaiting_hitl`.
* The Web UI displays the plan along with a **Plan Review** component.
* The researcher evaluates the plan and can click **Approve** (to resume execution) or **Request Changes** (typing corrective comments, which updates the graph state and triggers the Planner to generate a revised plan).

**Step 4: Stateful Computation & Recursive Debugging**
* The **Executor Agent** initiates MNE-Python operations within the stateful Jupyter kernel sandbox.
* If a runtime error occurs, the Executor captures the traceback, queries MNE syntax docs via RAG or Web Search, and rewrites the logic autonomously up to 5 times.

**Step 5: VLM-Based Quality Assurance**
* Jupyter outputs Base64 ERP and Topomap visualizations, which are transmitted to the **Critic Agent**.
* The Critic visually audits the plots. If artifacts persist, the Critic rejects the state and demands a re-run with revised processing options.

**Step 6: Reporting & Synthesis**
* Upon Critic approval, the graph consolidates all actions into a final data package.
* The researcher receives processed datasets, visual plots, a standalone compiled Python script (`output/analysis_pipeline.py`), and a comprehensive audit report (`output/final_report.md`).
* Active and past runs can be inspected or navigated at any time in the Web UI sidebar or via the `scripts/inspect_run.py` CLI utility.

---

## **7\. Edge Cases & Safety Protocols**

* **Memory Limits:** The Executor's system prompt MUST explicitly require the use of mne.set\_config('MNE\_MEMMAP\_MIN_SIZE', '10M') and preload=False memory mapping techniques to ensure the local Docker container avoids Out-Of-Memory (OOM) crashes.  
* **Corrupt Triggers:** If the metadata\_extractor finds 0 triggers, the Planner must dynamically shift strategy to prompt the user for an external events file (e.g., .vmrk or .csv).
* **Network Failures:** In the event of temporary network or WebSocket disconnects between the browser and FastAPI backend, the backend persists execution checkpoints in SQLite, allowing the client to fully hydrate and resume the run immediately on reload.

---

## **8\. Technical Setup & Deployment Guide**

### **1. Configure Environment Credentials**
Create a `.env` file at the root of the project:
```env
GOOGLE_API_KEY=your_google_api_key_here
EEG_DATA_DIR=./data
```

### **2. Prepare Data and RAG Vector Database**
Before running the services, you must prepare the data folder and ingest the neuroinformatics reference knowledge base into Chroma DB on your host:

* **Ingest RAG Documents**:
  Make sure your `.env` contains the Gemini API key, then run:
  ```bash
  pip install -e .
  python scripts/ingest_rag_data.py
  ```
  This creates the `./chroma_data` directory populated with layouts, summaries, and hierarchical indexes of standard neuroimaging papers and textbooks.
* **Organize EEG Data**:
  Place your EEG raw recordings under `./data`. Organize BIDS datasets under subfolders (e.g., `./data/ds004408/`).

### **3. Deploy with Docker Compose**
Start all containerized microservices:
```bash
docker compose -f docker/docker-compose.yml up -d --build
```
* **Web Dashboard (Next.js)**: `http://localhost:3000`
* **API Bridge (FastAPI)**: `http://localhost:8000`
* **Sandbox Gateway (Jupyter)**: Runs internally on port `8888`
* **Chroma DB Port**: Host-mapped to port `8001`

### **4. Run Locally (Workstation Development Mode)**
If running outside containers:
```bash
# Install core, web, and test dependencies
pip install -e .
pip install -r requirements-web.txt
pip install -r requirements-dev.txt

# Start only the sandbox container
cd docker && docker compose up -d sandbox && cd ..

# Run FastAPI backend
python -m uvicorn src.web.server:app --reload --host 127.0.0.1 --port 8000

# Start UI
cd ui && npm install && npm run dev
```