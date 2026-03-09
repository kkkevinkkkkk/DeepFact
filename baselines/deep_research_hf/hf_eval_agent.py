#!/usr/bin/env python3
"""
Token-aware CodeAgent that tracks token usage from both main agent and managed agents.
"""

from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from deep_fact.evaluators.utils.llm_client import (
    TokenUsage as EvalTokenUsage,
    single_shot_llm_call,
)

try:
    from smolagents import CodeAgent
    from smolagents.memory import TokenUsage as SmolTokenUsage
except ModuleNotFoundError:
    CodeAgent = Any  # type: ignore[misc,assignment]

    class SmolTokenUsage:
        """Fallback token usage shape for local/offline testing without smolagents."""

        def __init__(self, input_tokens: int = 0, output_tokens: int = 0):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

        def dict(self) -> Dict[str, int]:
            return {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
            }


class TokenAwareCodeAgent:
    """
    Wrapper around CodeAgent that tracks comprehensive token usage including managed agents.
    """
    
    def __init__(self, main_agent: CodeAgent, managed_agents: List = None):
        self.main_agent = main_agent
        self.managed_agents = managed_agents or []
        self.last_token_breakdown = {}
        
    def run(self, task: str, **kwargs) -> Any:
        """
        Run the agent and track all token usage.
        
        Returns:
            The agent's output result
        """
        # Reset all monitors before starting
        if hasattr(self.main_agent, 'monitor'):
            self.main_agent.monitor.reset()

        # Reset monitors of currently registered managed agents on the main agent
        for agent in getattr(self.main_agent, 'managed_agents', {}).values():
            if hasattr(agent, 'monitor'):
                agent.monitor.reset()
        
        # Run the main agent
        result = self.main_agent.run(task, **kwargs)
        
        # Collect token usage from all agents after execution
        self._collect_token_usage()
        
        return result
    
    def _collect_token_usage(self):
        """Collect token usage from all agents and store breakdown."""
        # Get main agent token usage
        main_tokens = SmolTokenUsage(input_tokens=0, output_tokens=0)
        if hasattr(self.main_agent, 'monitor'):
            main_tokens = self.main_agent.monitor.get_total_token_counts()
        
        # Get managed agents token usage (source of truth: main_agent.managed_agents)
        managed_tokens = {}
        total_managed_input = 0
        total_managed_output = 0

        for agent in getattr(self.main_agent, 'managed_agents', {}).values():
            if hasattr(agent, 'monitor'):
                agent_name = getattr(agent, 'name', None) or agent.__class__.__name__
                agent_tokens = agent.monitor.get_total_token_counts()
                managed_tokens[agent_name] = agent_tokens.dict()
                total_managed_input += agent_tokens.input_tokens
                total_managed_output += agent_tokens.output_tokens
        
        # Calculate totals
        total_input = main_tokens.input_tokens + total_managed_input
        total_output = main_tokens.output_tokens + total_managed_output
        
        # Store breakdown
        self.last_token_breakdown = {
            "total": {
                "input_tokens": total_input,
                "output_tokens": total_output, 
                "total_tokens": total_input + total_output
            },
            "main_agent": main_tokens.dict(),
            "managed_agents": managed_tokens
        }
    
    def get_token_usage(self) -> SmolTokenUsage:
        """Get total token usage from last run."""
        total_data = self.last_token_breakdown.get("total", {})
        return SmolTokenUsage(
            input_tokens=total_data.get("input_tokens", 0),
            output_tokens=total_data.get("output_tokens", 0)
        )
    
    def get_token_breakdown(self) -> Dict[str, Any]:
        """Get detailed token usage breakdown."""
        return self.last_token_breakdown.copy()
    
    @property
    def token_usage(self) -> SmolTokenUsage:
        """Property access to token usage (for compatibility)."""
        return self.get_token_usage()


def create_token_aware_agent(model_id: str = "openai/gpt-4.1-mini", 
                           verbosity_level: int = -1,
                           return_full_results: bool = True) -> TokenAwareCodeAgent:
    """
    Create a token-aware agent that tracks usage from managed agents.
    
    Args:
        model_id: The model to use
        verbosity_level: Logging verbosity level
        return_full_results: Whether to return full results
        
    Returns:
        TokenAwareCodeAgent instance
    """
    from baselines.deep_research_hf.run import create_agent
    
    # Create the main agent (it already holds its managed_agents internally)
    main_agent, _managed = create_agent(
        model_id=model_id,
        verbosity_level=verbosity_level,
        return_full_results=return_full_results,
        return_managed_agents=True
    )

    # Wrap in token-aware wrapper; rely on main_agent.managed_agents for aggregation
    return TokenAwareCodeAgent(main_agent)


# Convenience function for single sentence verification
def verify_sentence_with_token_tracking(sentence_idx: int, 
                                      sentence: str, 
                                      report: str, 
                                      model_id: str = "openai/gpt-4.1-mini") -> Dict[str, Any]:
    """
    Verify a sentence with comprehensive token tracking.
    
    Args:
        sentence_idx: Index of the sentence
        sentence: The sentence to verify
        report: The full report for context
        model_id: Model to use for verification
        
    Returns:
        Dictionary with verification results and token breakdown
    """
    from deep_fact.utils.llm_tools import extract_claim_context
    
    # Import templates
    hf_evaluator_template = '''You will be given a sentence along with its surrounding context to help clarify its meaning. Your task is to determine whether the sentence is Supported, Inconclusive, or Contradictory based on available evidence. To do this, search the internet and consult reliable, up-to-date sources. You may use your own general knowledge and reasoning ability, but rely primarily on credible, verifiable information to make your determination.

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

    template_extract_final_verdict = '''You will be given an evaluator's judgment regarding the verdict of a claim.
Your task is to extract the final verdict from the evaluator's response.

Only return one of the following labels as your output: "supported", "contradictory", or "inconclusive". Do not include any additional text.
'''
    
    # Create token-aware agent
    agent = create_token_aware_agent(model_id=model_id, verbosity_level=-1)
    
    # Extract context and create question
    context = extract_claim_context(report, sentence, model_id)
    question = hf_evaluator_template.format(context=context, claim=sentence)
    
    # Run the agent with token tracking
    outputs = agent.run(question)
    answer = outputs.output
    # Extract final verdict
    message = f"Evaluator's judgement: {answer}"
    verdict = single_shot_llm_call(
        model=model_id,
        system_prompt=template_extract_final_verdict,
        message=message
    )
    
    # Get comprehensive token usage
    token_usage = agent.get_token_usage()
    token_breakdown = agent.get_token_breakdown()
    
    return {
        "answer": answer,
        "sentence_idx": sentence_idx,
        "verdict": verdict,
        "context": context,
        "input_tokens": token_usage.input_tokens,
        "output_tokens": token_usage.output_tokens,
        "token_breakdown": token_breakdown
    }


def _extract_verified_sentences(
    data: Dict[str, Any],
    require_human_verified: bool = False,
    skip_sampled_labels: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """Collect candidate sentences for HF verification."""
    skip_labels = skip_sampled_labels or set()
    verified: List[Dict[str, Any]] = []
    for sentence_idx, sentence in enumerate(data.get("sentences_info", [])):
        if sentence.get("human_verdict") is None:
            continue
        if require_human_verified and sentence.get("human_verified") is not True:
            continue
        if sentence.get("sampled") in skip_labels:
            continue
        verified.append(sentence | {"sentence_idx": sentence_idx})
    return verified


def _normalize_verdict(verdict: Any) -> str:
    normalized = str(verdict or "").strip().lower()
    if normalized == "contradicted":
        return "contradictory"
    return normalized


def _aggregate_token_usage(results: List[Dict[str, Any]], model_id: str) -> EvalTokenUsage:
    token_usage = EvalTokenUsage(input_tokens=0, output_tokens=0)
    for result in results:
        if "error" in result:
            continue
        token_usage.add_model_usage(
            model=model_id,
            input_tokens=int(result.get("input_tokens", 0) or 0),
            output_tokens=int(result.get("output_tokens", 0) or 0),
        )
    return token_usage


def _aggregate_token_breakdown(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_main_input = 0
    total_main_output = 0
    total_managed_input = 0
    total_managed_output = 0
    managed_agents: Dict[str, Dict[str, int]] = {}

    for result in results:
        if "error" in result:
            continue
        breakdown = result.get("token_breakdown", {}) or {}
        main_tokens = breakdown.get("main_agent", {}) or {}
        total_main_input += int(main_tokens.get("input_tokens", 0) or 0)
        total_main_output += int(main_tokens.get("output_tokens", 0) or 0)

        managed = breakdown.get("managed_agents", {}) or {}
        for agent_name, usage in managed.items():
            input_tokens = int((usage or {}).get("input_tokens", 0) or 0)
            output_tokens = int((usage or {}).get("output_tokens", 0) or 0)
            if agent_name not in managed_agents:
                managed_agents[agent_name] = {"input_tokens": 0, "output_tokens": 0}
            managed_agents[agent_name]["input_tokens"] += input_tokens
            managed_agents[agent_name]["output_tokens"] += output_tokens
            total_managed_input += input_tokens
            total_managed_output += output_tokens

    total_input = total_main_input + total_managed_input
    total_output = total_main_output + total_managed_output
    return {
        "total": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
        },
        "main_agent": {
            "input_tokens": total_main_input,
            "output_tokens": total_main_output,
            "total_tokens": total_main_input + total_main_output,
        },
        "managed_agents": managed_agents,
    }


def evaluate_reports(
    data: Dict[str, Any],
    model_id: str = "openai/gpt-4.1",
    max_workers: int = 8,
    retries: int = 1,
    retry_delay: float = 1.0,
    require_human_verified: bool = False,
    skip_sampled_labels: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate all eligible sentences in one report using the HF agent.

    Returns:
        {
            "evaluation": list[dict],
            "token_usage": TokenUsage,
            "token_breakdown": dict,
        }
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if retries < 1:
        raise ValueError("retries must be >= 1")

    report = data.get("response", "")
    annotated_sentences = _extract_verified_sentences(
        data=data,
        require_human_verified=require_human_verified,
        skip_sampled_labels=skip_sampled_labels,
    )
    if len(annotated_sentences) == 0:
        return {
            "evaluation": [],
            "token_usage": EvalTokenUsage(input_tokens=0, output_tokens=0),
            "token_breakdown": {
                "total": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "main_agent": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "managed_agents": {},
            },
        }

    def worker(sentence_idx: int, claim: str) -> Dict[str, Any]:
        for attempt in range(1, retries + 1):
            try:
                result = verify_sentence_with_token_tracking(
                    sentence_idx=sentence_idx,
                    sentence=claim,
                    report=report,
                    model_id=model_id,
                )
                return {
                    "sentence_idx": sentence_idx,
                    "claim": claim,
                    "context": result.get("context", ""),
                    "rationale": result.get("answer", ""),
                    "verdict": _normalize_verdict(result.get("verdict", "")),
                    "input_tokens": int(result.get("input_tokens", 0) or 0),
                    "output_tokens": int(result.get("output_tokens", 0) or 0),
                    "token_breakdown": result.get("token_breakdown", {}),
                }
            except Exception as err:
                if attempt >= retries:
                    return {
                        "sentence_idx": sentence_idx,
                        "claim": claim,
                        "error": str(err),
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                time.sleep(retry_delay)
        return {
            "sentence_idx": sentence_idx,
            "claim": claim,
            "error": "unexpected_retry_exit",
            "input_tokens": 0,
            "output_tokens": 0,
        }

    results_by_idx: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker, item["sentence_idx"], item["sentence"]): item["sentence_idx"]
            for item in annotated_sentences
        }
        for future in as_completed(futures):
            sentence_idx = futures[future]
            try:
                result = future.result()
            except Exception as err:
                result = {
                    "sentence_idx": sentence_idx,
                    "claim": next(
                        (item["sentence"] for item in annotated_sentences if item["sentence_idx"] == sentence_idx),
                        "",
                    ),
                    "error": str(err),
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            results_by_idx[sentence_idx] = result

    ordered_results: List[Dict[str, Any]] = []
    for item in annotated_sentences:
        sentence_idx = item["sentence_idx"]
        ordered_results.append(
            results_by_idx.get(
                sentence_idx,
                {
                    "sentence_idx": sentence_idx,
                    "claim": item.get("sentence", ""),
                    "error": "missing_result",
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            )
        )

    token_usage = _aggregate_token_usage(ordered_results, model_id=model_id)
    token_breakdown = _aggregate_token_breakdown(ordered_results)
    return {
        "evaluation": ordered_results,
        "token_usage": token_usage,
        "token_breakdown": token_breakdown,
    }



def reflect_sentence_verification(sentence_idx: int,
                                sentence: str,
                                report: str,
                                previous_verdict: str,
                                previous_rationale: str = "",
                                others_verdict: str = "",
                                others_rationale: str = "",
                                model_id: str = "openai/gpt-4.1-mini") -> Dict[str, Any]:
    """
    Reflect on a previous sentence verification and potentially revise the verdict.

    Args:
        sentence_idx: Index of the sentence
        sentence: The sentence to verify
        report: The full report for context
        previous_verdict: The prior verdict to reflect on
        model_id: Model to use for reflection

    Returns:
        Dictionary with reflection results and token breakdown
    """
    from deep_fact.utils.llm_tools import extract_claim_context

    # Import templates
    hf_reflector_template = '''You are an expert fact-checker responsible for reassessing the verification of a claim. You will be provided with:
- the original claim and its context
- Your prior verdict (Supported, Inconclusive, or Contradictory) and your rationale
- A second expert’s verdict and their reasoning

Your tasks: 

Review all provided materials and decide whether to uphold or revise your original verdict.

In making your decision, you should:
1. Carefully reflect on your initial reasoning and verdict
2. Critically evaluate the second expert’s reasoning
3. Apply your own independent judgment, incorporating: the quality of the evidence and logic on both sides, whether any verdict better satisfies the definitions and criteria below

Verdict Definitions:

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


You need to search the internet and might use general factual knowledge, and rely primarily on credible and verifiable evidence. You need to provide a clear explanation summarizing the key information you found and how it led to your conclusion, followed by your final verdict. Be concise, factual, and cite sources when appropriate.


Context: {context}
    
Claim: {claim}

Previous Verdict: {previous_verdict}

Previous Rationale: {previous_rationale}

Other Expert's Verdict: {others_verdict}

Other Expert's Rationale: {others_rationale}
'''
    template_extract_final_verdict = '''You will be given an evaluator's judgment regarding the verdict of a claim.
    Your task is to extract the final verdict from the evaluator's response.

    Only return one of the following labels as your output: "supported", "contradictory", or "inconclusive". Do not include any additional text.
    '''

    agent = create_token_aware_agent(model_id=model_id, verbosity_level=-1)
    context = extract_claim_context(report, sentence, model_id)
    question = hf_reflector_template.format(
        context=context,
        claim=sentence,
        previous_verdict=previous_verdict,
        previous_rationale=previous_rationale,
        others_verdict=others_verdict ,
        others_rationale=others_rationale
    )
    outputs = agent.run(question)
    answer = outputs.output
    message = f"Reflector's judgement: {answer}"
    verdict = single_shot_llm_call(
        model=model_id,
        system_prompt=template_extract_final_verdict,
        message=message,
    )
    token_usage = agent.get_token_usage()
    token_breakdown = agent.get_token_breakdown()
    return {
        "answer": answer,
        "sentence_idx": sentence_idx,
        "verdict": verdict,
        "context": context,
        "input_tokens": token_usage.input_tokens,
        "output_tokens": token_usage.output_tokens,
        "token_breakdown": token_breakdown
    }


def justify_sentence_verdict(sentence_idx: int,
                           sentence: str,
                           report: str,
                           verdict: str,
                           model_id: str = "openai/gpt-4.1-mini") -> Dict[str, Any]:
    """
    Justify a sentence verdict with comprehensive token tracking.

    Args:
        sentence_idx: Index of the sentence
        sentence: The sentence to justify
        report: The full report for context
        verdict: The verdict to justify
        model_id: Model to use for justification

    Returns:
        Dictionary with justification results and token breakdown
    """
    from deep_fact.utils.llm_tools import extract_claim_context

    # Import templates
    hf_justifier_template = '''You are an expert fact-checker tasked with providing a clear and concise rationale for a given verdict on a claim. You will be provided with the original claim, its surrounding context, and the verdict (Supported, Inconclusive, or Contradictory). Your job is to explain the reasoning behind the verdict, citing credible sources and evidence that support your conclusion.

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


You need to search the internet and might use general factual knowledge, and rely primarily on credible and verifiable evidence. You need to provide a clear explanation summarizing the key information you found and how it led to your conclusion. Be concise, factual, and cite sources when appropriate.

Context: {context}

Claim: {claim}

Verdict: {verdict}

Return a concise rationale explaining the reasoning behind the verdict in markdown format.
'''
    agent = create_token_aware_agent(model_id=model_id, verbosity_level=-1)
    context = extract_claim_context(report, sentence, model_id)
    question = hf_justifier_template.format(
        context=context,
        claim=sentence,
        verdict=verdict
    )
    outputs = agent.run(question)
    answer = outputs.output
    token_usage = agent.get_token_usage()
    token_breakdown = agent.get_token_breakdown()
    return {
        "answer": answer,
        "sentence_idx": sentence_idx,
        "verdict": verdict,
        "context": context,
        "input_tokens": token_usage.input_tokens,
        "output_tokens": token_usage.output_tokens,
        "token_breakdown": token_breakdown
    }
