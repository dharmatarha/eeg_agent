# **Blueprint: Actionable EEG-ADK Multi-Agent System**

**Version:** 4.0 (Unified Architecture)  
**Target:** High-Performance Local Workstation (License-Free)  
**Orchestration Paradigm:** State-Graph Multi-Agent Orchestration (e.g., LangGraph) integrated with Google ADK

## **1\. Core Objectives: EEG-ADK Methodology**

The primary goal of this multi-agent architecture is to mitigate the inherent instability and reproducibility crisis in conventional EEG workflows. Key drivers include:

1. **Synthesizing Procedural Automation:** Reducing researcher overhead by automating the generation of boilerplate code for ingestion, signal filtering, and visualization.  
2. **Intelligent Parameter Inference:** Leveraging scientific RAG to function as an expert assistant, autonomously populating standardized research parameters (such as P300 filter specifications) when user input is underspecified.  
3. **Adaptive Script Rectification:** Maintaining execution flow through iterative self-debugging; the system dynamically resolves syntax errors or memory bottlenecks rather than halting at the first failure.  
4. **Verified Auditability:** Ensuring every computational decision and inference is persisted to SQLite (`logs/checkpoints.sqlite`), producing a standalone, runnable script (`output/analysis_pipeline.py`) alongside a manuscript-ready final audit report (`output/final_report.md`) that documents exact RAG queries, code blocks executed, and validation steps.

## **2\. System Architecture: State-Graph Orchestration**

Simple sequential agent communication is insufficient and prone to infinite loops when debugging complex Python code. The orchestration is built on a **State-Graph framework**, where a shared "State" dictionary is maintained and updated by each node, while Google ADK defines the individual agents and their tools.

* **State Object:** Maintains fields for `user_directive`, `data_path`, `raw_metadata`, `analysis_plan`, `execution_logs`, `generated_plots`, `error_count`, `critic_feedback`, `is_approved`, as well as tracking fields `rag_history` (RAG citations) and `executed_code_blocks` (sandbox execution logs).  
* **Persistence & Session Isolation:** Uses LangGraph `SqliteSaver` to maintain state checkpoints across workflow interruptions. Each execution uses a unique timestamped thread ID (`run_YYYYMMDD_HHMMSS_uuid`) for clean session isolation.
* **Recursion Limits:** Hardcoded execution caps (e.g., maximum 5 retries for the Executor to fix an error) to prevent burning compute indefinitely. If the limit is reached, the graph halts and requests Human-in-the-Loop (HITL) intervention.

## **3\. Multi-Agent Architecture (Core Agents)**

| Agent Role | Primary Responsibilities | Key Behaviors & Constraints |
| :---- | :---- | :---- |
| **Lead Planner (Strategist)** | Neuroinformatics & Workflow Design | Translates user methods into a technical pipeline; halts for user approval before execution; uses offline RAG to infer missing parameters. |
| **Executor (Programmer)** | Code Generation & Execution | Writes MNE-Python scripts; operates within a Stateful Jupyter Sandbox; implements memory-safe loading; self-corrects based on error logs. |
| **Critic (Reviewer)** | Quality Assurance & Reporting (VLM) | Must be a Multimodal Vision-Language Model. Validates Base64 plots for SNR and anomalies (e.g., eye-blinks); synthesizes final Methods and Results. |

## **4\. Refined Tool Specifications (Local & Realistic)**

The following tools are registered via Google ADK to facilitate high-fidelity EEG processing.

| Tool Name | Input Parameters | Realistic Implementation Strategy |
| :---- | :---- | :---- |
| **scientific\_rag & doc\_crawler** | query (str), paradigm (str) | Build an offline Vector Database (ChromaDB/FAISS) pre-populated with standard Neuroimage methods papers and the official MNE-Python API documentation. Avoids live API rate limits and hallucinations. |
| **stateful\_jupyter\_exec** | code\_string (str) | Connects to a Dockerized Kernel Gateway via WebSocket. Returns text/plain (logs) and image/png (Base64 data). Runs data-loading in "Turn 1" and filtering in "Turn 2" while data remains safely in RAM. |
| **metadata\_extractor** | file\_path (str) | A pre-written Python microservice using mne.io.read\_raw(..., preload=False).info to immediately dump channels, sampling frequency, and annotations to JSON. |
| **bids\_inspector** | bids\_root (str) | A directory-scanning microservice that crawls subject structures, parses dataset configurations, and extracts representative MNE raw headers. |

## **5\. Operational Workflow Execution**

The practical deployment of the system follows a Human-in-the-Loop (HITL) state-graph trajectory:

1. **Data & Task Ingestion:** The user points the workstation to a local directory containing raw datasets and inputs a high-level natural language directive.  
2. **Analysis Planning & RAG Research:** The Planner scans file metadata. If bandpass requirements are missing, it queries the local vector DB to integrate standards into a formal Analysis Plan.  
3. **Human-in-the-Loop Validation:** State execution pauses. The user evaluates the proposed workflow and can approve it or provide corrective feedback.  
4. **Stateful Computation & Recursive Debugging:** The Executor initiates MNE-Python operations. Data remains preloaded in RAM. If an error occurs, the Executor analyzes the traceback, consults API docs, and rewrites logic autonomously.  
5. **VLM-Based Quality Assurance:** The Jupyter kernel outputs Base64 visualizations, transmitted to the Critic. If artifacts persist, the Critic rejects the state and demands a re-run with tightened thresholds.  
6. **Reporting & Synthesis:** Upon Critic approval, the graph consolidates all actions. The researcher receives processed datasets, statistical CSVs, high-fidelity plots, and a manuscript-ready Methods section.

## **6\. Python Implementation (Google ADK \+ LangGraph)**

Below is a boilerplate code snippet for registering the agents and tools using the Python ADK library, designed to be plugged into a LangGraph state machine.  
`from google.adk import Agent, Tool`  
`from langgraph.graph import StateGraph, END`

`# 1. Define the Tools`  
`research_tool = Tool(`  
    `name="scientific_rag",`  
    `description="Search offline Vector DB for standard EEG processing parameters",`  
    `func=my_local_rag_function`  
`)`

`sandbox_tool = Tool(`  
    `name="stateful_jupyter_exec",`  
    `description="Execute Python code in a stateful Docker container",`  
    `func=my_jupyter_websocket_client`  
`)`

# 2. Define the Agents via ADK  
planner = Agent(  
    name="Planner",  
    system_instruction="You are a Senior Neuroscientist. Translate vague descriptions...",  
    tools=[research_tool, metadata_extractor_tool, bids_inspector_tool]  
)

executor = Agent(  
    name="Executor",  
    system_instruction="You are a Python developer. Write memory-safe code...",  
    tools=[sandbox_tool, doc_crawler_tool]  
)

`# Note: In a LangGraph setup, these agents act as nodes processing the global State object.`

## **7\. Edge Cases & Safety Protocols**

* **Memory Limits:** The Executor's system prompt MUST explicitly require the use of mne.set\_config('MNE\_MEMMAP\_MIN\_SIZE', '10M') and preload=False memory mapping techniques to ensure the local Docker container avoids Out-Of-Memory (OOM) crashes.  
* **Corrupt Triggers:** If the metadata\_extractor finds 0 triggers, the Planner must dynamically shift strategy to prompt the user for an external events file (e.g., .vmrk or .csv).

## **8\. Operational Workflow Execution**

The practical deployment of the system follows a **Human-in-the-Loop (HITL)** state-graph trajectory as detailed below:

**Step 1: Data & Task Ingestion**

* Point the workstation to a local directory containing raw .set or .fif assets.  
* Input a high-level natural language directive, such as: *"Execute N400 analysis: apply ICA cleaning, epoch at word onset, and perform cluster permutation statistics."*

**Step 2: Analysis Planning & RAG Research**

* The **Planner Agent** scans file metadata for triggers and sampling rates, cross-referencing these against the user's request.  
* *Technical Logic:* If bandpass requirements are missing for the N400, the system queries the local vector DB, identifies the 0.1–30 Hz standard, and integrates it into the formal plan.  
* A structured markdown Analysis Plan is generated for final review.

**Step 3: Human-in-the-Loop Validation**

* State execution pauses, allowing the researcher to evaluate the proposed workflow.  
* The user can approve the graph or provide corrective feedback like, *"Adjust high-pass threshold to 0.5 Hz,"* prompting a state update.

**Step 4: Stateful Computation & Recursive Debugging**

* The **Executor Agent** initiates MNE-Python operations within the stateful Jupyter kernel environment.  
* Data remains preloaded in RAM as the system applies filtering and ICA transformations.  
* *Recursion Workflow:* Should a "Channel Missing" error occur, the Executor analyzes the traceback, consults the API docs, and rewrites the interpolation logic autonomously.

**Step 5: VLM-Based Quality Assurance**

* Jupyter outputs Base64 ERP and Topomap visualizations, which are transmitted to the **Critic Agent**.  
* The Critic visually identifies anomalies. If eye-blink artifacts persist in frontal sensors, the Critic rejects the state and demands an ICA re-run with tightened variance thresholds.

**Step 6: Reporting & Synthesis**

* Upon Critic approval, the graph consolidates all actions into a final data package.  
* The researcher receives processed datasets, statistical CSVs, high-fidelity plots, a standalone compiled Python script (`output/analysis_pipeline.py`), and a comprehensive audit report (`output/final_report.md`) detailing the user directive, file metadata, RAG retrieval audit log, sandbox code execution traces, and critic QA feedback.

## **9\. Technical Setup Guide: EEG-ADK Local Environment**

**Deployment Strategy:** Docker Containerization with vLLM Backend (License-Free Approach)

### **Local Packaging & CLI Configuration**
To configure proper package resolution and install CLI scripts, perform a local editable installation:
```bash
pip install -e .
```
This registers the executable script `eeg-agent` to boot the orchestrator.


### **License-Free Docker Sandbox (Dockerfile)**

This configuration uses the MATLAB Runtime (MCR) to run compiled EEGLAB and MNE-Python for modern processing.  
`# Use NVIDIA CUDA base for GPU acceleration`  
`FROM nvidia/cuda:12.1.1-devel-ubuntu22.04`

`# Agree to MATLAB Runtime License`  
`ENV AGREE_TO_MATLAB_RUNTIME_LICENSE=yes`

`# Install System Dependencies`  
`RUN apt-get update && apt-get install -y \`  
    `python3-pip python3-dev git wget unzip \`  
    `libgl1-mesa-glx libqt5gui5 \`  
    `&& rm -rf /var/lib/apt/lists/*`

`# Install MATLAB Runtime (MCR) R2024a`  
`RUN mkdir /mcr_install && \`  
    `cd /mcr_install && \`  
    `wget https://ssd.mathworks.com/supportfiles/downloads/R2024a/Release/0/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2024a_glnxa64.zip && \`  
    `unzip MATLAB_Runtime_R2024a_glnxa64.zip && \`  
    `./install -mode silent -agreeToLicense yes && \`  
    `cd / && rm -rf /mcr_install`

`# Install Compiled EEGLAB`  
`RUN mkdir /opt/eeglab && \`  
    `wget https://sccn.ucsd.edu/eeglab/download/eeglab_compiled.zip && \`  
    `unzip eeglab_compiled.zip -d /opt/eeglab`

`# Set Environment Variables for MCR`  
`ENV LD_LIBRARY_PATH="/usr/local/MATLAB/MATLAB_Runtime/v241/runtime/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v241/bin/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v241/sys/os/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v241/extern/bin/glnxa64"`

`# Install Python Neuroimaging Stack`  
`RUN pip3 install --upgrade pip`  
`RUN pip3 install \`  
    `mne mne-bids mne-connectivity \`  
    `scikit-learn pandas matplotlib vllm jupyter_client`

### **vLLM Backend Configuration**

To run the agent logic locally using vLLM with high context support to ensure it can process research RAG:  
`python3 -m vllm.entrypoints.openai.api_server \`  
    `--model "mistralai/Mixtral-8x7B-Instruct-v0.1" \`  
    `--gpu-memory-utilization 0.9 \`  
    `--max-model-len 32768 \`  
    `--tensor-parallel-size 1`

### **Workflow Governance**

* **Read-Only Data:** Mount your EEG data as read-only. The host path is defined dynamically via the `EEG_DATA_DIR` environment variable, which defaults to `./data` and maps to `/mnt/data:ro` in the sandbox container. All outputs should go to a separate `/output` directory.