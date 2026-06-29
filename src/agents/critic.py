from langchain_core.messages import HumanMessage, SystemMessage
from src.agents.llm_factory import get_llm

def get_critic_agent(thread_id=None):
    """
    Initialize and return the Critic agent function.
    
    The Critic is a Multimodal VLM that reviews execution logs and generated plots
    to either APPROVE the pipeline quality or REJECT it with feedback.
    """
    # The Critic needs to be a Multimodal VLM.
    llm = get_llm(agent_type="multimodal", temperature=0.1)
    
    output_path = f"/output/{thread_id}" if thread_id else "/output"
    
    system_prompt = f"""You are the Critic (Quality Assurance & Reviewer) for an EEG analysis pipeline.
Your job is to review the user directive, proposed analysis plan, executed Python code, execution logs, and generated Base64 plots.

CRITICAL CHECKS:
1. **Adherence to Plan:** Verify that the executed code actually follows the proposed Analysis Plan and fulfills all steps of the User Goal.
2. **Memory Safety & Sandbox constraints:**
   - The code must call `mne.set_config('MNE_MEMMAP_MIN_SIZE', '10M')` (or set it in config).
   - The code must use `preload=False` when loading raw data to avoid Out-Of-Memory (OOM) crashes.
   - In multi-subject loops, the code must manage memory aggressively (using `gc.collect()`, `plt.close('all')`, and saving intermediate files to `{output_path}/`).
3. **Library & Dependency Auditing:**
   - Ensure the code only uses supported libraries (`mne`, `mne-bids`, `mne-connectivity`, etc.).
4. **Signal-to-Noise Ratio (SNR) & Artifacts:**
   - Review execution logs and visual plots (e.g. ERPs, PSD, Topomaps, connectivity matrices) for poor signal quality.
   - Look for persistent ocular (eye-blinks), cardiac, or muscle artifacts, and bad channels that should have been removed or mitigated (e.g., using ICA or SSP). Compare your findings to the user's request and the planned analysis steps.
5. **Reference Consistency Check (optional):**
   - If a REFERENCE RUN MEMORY is provided, verify that the executed code and parameters align with the reference run as planned, and verify any necessary deviations.

VERDICT RULES:
- If any critical checks fail, or if artifacts/errors persist in the logs or plots, you MUST start your response with the word 'REJECT' followed by detailed, actionable feedback.
- If the pipeline succeeded, the code is memory-safe, and visual quality/SNR are high, you MUST start your response with the word 'APPROVE' followed by a manuscript-ready 'Methods and Results' section that details the exact preprocessing, analysis parameters, and outcomes based on the executed code."""

    def invoke_critic(state):
        # We assume state contains 'execution_logs' and 'generated_plots' (list of base64 images)
        logs = state.get('execution_logs', [])
        plots = state.get('generated_plots', [])
        
        # Handle log lists from state or strings from tests cleanly
        if isinstance(logs, list):
            logs_str = "\n".join(logs)
        else:
            logs_str = str(logs)
            
        # Retrieve extra state fields to make the Critic fully context-aware
        user_directive = state.get('user_directive', '')
        analysis_plan = state.get('analysis_plan', '')
        executed_code_blocks = state.get('executed_code_blocks', [])

        # Format the executed code block summaries
        code_str = ""
        if executed_code_blocks:
            code_blocks_formatted = []
            for idx, block in enumerate(executed_code_blocks):
                code = block.get("code", "")
                if code:
                    code_blocks_formatted.append(f"--- Code Block {idx + 1} ---\n{code}")
            code_str = "\n\n".join(code_blocks_formatted)

        # Build a richer context string
        context_parts = []
        if user_directive:
            context_parts.append(f"User Goal / Directive:\n{user_directive}")
        if analysis_plan:
            context_parts.append(f"Analysis Plan:\n{analysis_plan}")
        if code_str:
            context_parts.append(f"Executed Code:\n{code_str}")
        context_parts.append(f"Execution Logs:\n{logs_str}")

        prompt_content = "\n\n".join(context_parts)
        prompt_content += "\n\nPlease review the above context alongside the attached plots and provide your assessment."

        content = [{"type": "text", "text": prompt_content}]
        
        # Attach base64 images
        for img_b64 in plots:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })
            
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content)
        ]
        
        response = llm.invoke(messages)
        return response.content

    return invoke_critic
