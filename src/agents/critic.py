from langchain_core.messages import HumanMessage, SystemMessage
from src.agents.llm_factory import get_llm

def get_critic_agent():
    # The Critic needs to be a Multimodal VLM.
    llm = get_llm(agent_type="multimodal", temperature=0.1)
    
    system_prompt = """You are the Critic (Quality Assurance & Reviewer) for an EEG analysis pipeline.
Your job is to review the generated Base64 plots and execution logs to validate SNR and identify anomalies (e.g., eye-blinks).
If artifacts persist, you should output 'REJECT' along with your reasoning, demanding a re-run with tightened thresholds.
If the quality is acceptable, output 'APPROVE' and synthesize a final manuscript-ready 'Methods and Results' section based on the executed code."""

    def invoke_critic(state):
        # We assume state contains 'execution_logs' and 'generated_plots' (list of base64 images)
        logs = state.get('execution_logs', '')
        plots = state.get('generated_plots', [])
        
        content = [{"type": "text", "text": f"Execution Logs:\n{logs}\n\nPlease review the attached plots and provide your assessment."}]
        
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
