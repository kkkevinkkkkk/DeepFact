import argparse
import os
import threading

from dotenv import load_dotenv
from huggingface_hub import login
from baselines.deep_research_hf.scripts.text_inspector_tool import TextInspectorTool
from baselines.deep_research_hf.scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from baselines.deep_research_hf.scripts.visual_qa import visualizer

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    # InferenceClientModel,
    LiteLLMModel,
    ToolCallingAgent,
)
import json

load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "question", type=str, help="for example: 'How many studio albums did Mercedes Sosa release before 2007?'"
    # )
    parser.add_argument("--model-id", type=str, default="o1")
    return parser.parse_args()


custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent(model_id="o1" ,verbosity_level=2, return_full_results=False, return_managed_agents=False):
    model_params = {
        "model_id": model_id,
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": 8192,
    }
    if "bedrock" in model_id:
        model_params["num_retries"] = 6

    if model_id == "o1":
        model_params["reasoning_effort"] = "high"
    model = LiteLLMModel(**model_params)

    text_limit = 100000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    WEB_TOOLS = [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=verbosity_level,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, text_limit)],
        max_steps=6,
        verbosity_level=verbosity_level,
        additional_authorized_imports=["*"],
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
        return_full_result=return_full_results
    )
    if return_managed_agents:
        return manager_agent, text_webbrowser_agent

    return manager_agent

template = '''You will be given a sentence along with its surrounding context to help clarify its meaning. Your task is to determine whether the sentence is Supported, Inconclusive, or Contradictory based on available evidence. To do this, search the internet and consult reliable, up-to-date sources. You may use your own general knowledge and reasoning ability, but rely primarily on credible, verifiable information to make your determination.

If a sentence contains multiple claims, the sentence-level label is determined by the least-supported claim:

- If all claims are Supported → the sentence is Supported.
- If any claim is Contradictory → the sentence is Contradictory.
- If no claim is Contradictory but at least one is Inconclusive → the sentence is Inconclusive.

1. Supported
Definition:
The claim is fully and unambiguously entailed—either directly or through clear, bounded reasoning—by at least one reliable source. No equally or more reliable source contradicts the claim.

Requirements:

- The evidence covers all key elements of the claim.
- If inference is used, it must follow transparent logic (e.g., arithmetic, taxonomic, temporal, causal).
- No reliable source refutes the claim or introduces reasonable doubt.

2. Contradictory
Definition:
The claim is directly contradicted by at least one reliable source, and no equally strong or stronger source supports it.

Requirements:

- You find a credible source that states the opposite of the claim.
- No other evidence of equal or higher credibility supports the claim.

3. Inconclusive
Definition:
The claim is not clearly supported or contradicted by available evidence. Either the information is missing, conflicting, or the reasoning needed to infer the claim is too speculative or unclear.



In your response, provide a clear explanation summarizing the key information you found and how it led to your conclusion, followed by your final verdict. Be concise, factual, and cite sources when appropriate.

Context: {context}

Claim: {claim}
'''
def main():
    args = parse_args()
    model_id = args.model_id
    model_id = "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    # model_id = "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0"
    agent = create_agent(model_id=model_id, verbosity_level=2)

    question = "Is the following claim supported, contradict, or in conclusive? You need to find the evidence online and make your own judgements. Claim: In fact-based question answering, the answer should not reflect the model’s own hypotheses."
    question = "You will be given a sentence along with its surrounding context to help clarify its meaning. Your task is to determine whether the sentence is Supported, Inconclusive, or Contradictory based on available evidence. To do this, search the internet and consult reliable, up-to-date sources. You may use your own general knowledge and reasoning ability, but rely primarily on credible, verifiable information to make your determination.\n\nIf a sentence contains multiple claims, the sentence-level label is determined by the least-supported claim:\n\n- If all claims are Supported → the sentence is Supported.\n- If any claim is Contradictory → the sentence is Contradictory.\n- If no claim is Contradictory but at least one is Inconclusive → the sentence is Inconclusive.\n\n1. Supported\nDefinition:\nThe claim is fully and unambiguously entailed—either directly or through clear, bounded reasoning—by at least one reliable source. No equally or more reliable source contradicts the claim.\n\nRequirements:\n\n- The evidence covers all key elements of the claim.\n- If inference is used, it must follow transparent logic (e.g., arithmetic, taxonomic, temporal, causal).\n- No reliable source refutes the claim or introduces reasonable doubt.\n\n2. Contradictory\nDefinition:\nThe claim is directly contradicted by at least one reliable source, and no equally strong or stronger source supports it.\n\nRequirements:\n\n- You find a credible source that states the opposite of the claim.\n- No other evidence of equal or higher credibility supports the claim.\n\n3. Inconclusive\nDefinition:\nThe claim is not clearly supported or contradicted by available evidence. Either the information is missing, conflicting, or the reasoning needed to infer the claim is too speculative or unclear.\n\n\n\nIn your response, provide a clear explanation summarizing the key information you found and how it led to your conclusion, followed by your final verdict. Be concise, factual, and cite sources when appropriate.\n\nContext: Notably, CR-DPO’s chain-of-thought training outperformed the rule-based CTPC approach on advanced models – **GPT-4 with SCR+CR-DPO surpassed any CTPC-based rule performance**[arxiv.org](https://arxiv.org/html/2410.14675v1#:~:text=confidence%20scores%20rather%20than%20the,based%20answer%20with)[arxiv.org](https://arxiv.org/html/2410.14675v1#:~:text=In%20contrast%2C%20SCR%20avoids%20these,the%20model%20operates%20more%20effectively). In summary, allowing the model to *learn* when to trust context (via preference optimization on confidence reasoning) leads to more generalizable conflict resolution than fixed calibration rules. **Knowledge-Aware Preference Optimization: KnowPO (2024)** Another leap beyond CTPC came from training LLMs to actively prefer truthful evidence using **preference optimization**. **KnowPO** (Zhang *et al.*, 2024) introduced a *Knowledge-aware Preference Optimization* strategy that fine-tunes models on specially constructed knowledge-conflict examples[arxiv.org](https://arxiv.org/html/2408.03297v2#:~:text=absence%20of%20explicit%20negative%20signals,strategy%20and%20data%20ratio%20optimization)[arxiv.org](https://arxiv.org/html/2408.03297v2#:~:text=ignorance%20and%20contextual%20overinclusion,knowledge%20conflicts%20by%20over%2037).\n\nClaim: Another leap beyond CTPC came from training LLMs to actively prefer truthful evidence using **preference optimization**.\n"
    claim = "Another leap beyond CTPC came from training LLMs to actively prefer truthful evidence using **preference optimization**."
    report_path = "data/visualization/data_yukun_v0/Handling-Contradictory-Evidence-in-QA:-Recent-Advances-Post‑2023.json"

    claim = "- **Self-Think Module**: Iteratively synthesizes insights from aligned and original contexts, enabling the model to resolve conflicts through structured reasoning."
    claim = "This step minimizes incorrect-match errors by filtering out irrelevant or misleading information."
    claim = "This approach addresses over-confidence errors by ensuring the model does not stop retrieval prematurely when the context is incomplete."
    claim = "EDC2-RAG’s experiments on NQ and WebQ demonstrate significant improvements in faithfulness metrics."
    report_path = "data/visualization/data_yukun_v0/Synthesis-of-Methods-to-Enhance-Faithfulness-in-RAG-Long-Form-Generation-Without-Compromising-Coverage.json"

    with open(report_path, 'r') as f:
        context = json.load(f)["response"]

    question = template.format(context=context, claim=claim)
    answer = agent.run(question)
    print(f"Got this answer: {answer}")


if __name__ == "__main__":
    main()
