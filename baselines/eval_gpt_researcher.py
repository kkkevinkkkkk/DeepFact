import json
import logging
import asyncio
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

# Import GPT Researcher components
from deep_fact.utils.llm_tools import factuality_evaluation_prompt, extract_claim_context, factuality_evaluation_no_context_prompt
from deep_fact.evaluators.utils.llm_client import single_shot_llm_call, TokenUsage
import sys
import os
BASELINES_DIR = Path(__file__).resolve().parent
GPT_RESEARCHER_ROOT = BASELINES_DIR / "gpt-researcher"
if str(GPT_RESEARCHER_ROOT) not in sys.path:
    sys.path.append(str(GPT_RESEARCHER_ROOT))

from gpt_researcher import GPTResearcher
from gpt_researcher.utils.enum import ReportType, ReportSource, Tone


os.environ.setdefault("RETRIEVER", "serper")
os.environ.setdefault("SCRAPER", "crawl4ai")
os.environ.setdefault("STRATEGIC_LLM", "openai:gpt-4.1")
os.environ.setdefault("SMART_LLM", "openai:gpt-4.1")
os.environ.setdefault("FAST_LLM", "openai:gpt-4.1-mini")
# os.environ["STRATEGIC_LLM"] = "openai:gpt-4.1-mini"
# os.environ["SMART_LLM"] = "openai:gpt-4.1-mini"
# os.environ["FAST_LLM"] = "openai:gpt-4.1-mini"
os.environ.setdefault("DEEP_RESEARCH_BREADTH", "5")
# default depth is 3
os.environ.setdefault("DEEP_RESEARCH_DEPTH", "3")
# os.environ["DEEP_RESEARCH_DEPTH"] = "4"

def extract_verdict_from_report(evaluation_report: str, model_id: str, token_usage: TokenUsage | None = None) -> str:
    """
    Extract the verdict from an evaluator report using an LLM.
    
    Args:
        evaluation_report: The full evaluation report text
        model_id: The model to use for verdict extraction
        
    Returns:
        String verdict: "Supported", "Contradictory", "Inconclusive", or "Unknown"
    """

    if not evaluation_report:
        return ""
    
    # Template for verdict extraction (similar to hf_eval_agent.py)
    template_extract_final_verdict = '''You will be given an evaluator's judgment regarding the verdict of a claim.
Your task is to extract the final verdict from the evaluator's response.

Only return one of the following labels as your output: "supported", "contradictory", or "inconclusive". Do not include any additional text.'''
    
    try:
        # Use LLM to extract verdict
        message = f"Evaluator's judgement: {evaluation_report}"
        verdict = single_shot_llm_call(
            model=model_id,
            system_prompt=template_extract_final_verdict,
            message=message,
            token_usage=token_usage
        )
        
        # Normalize the verdict to match expected format
        verdict_lower = verdict.lower().strip()
        if "supported" in verdict_lower:
            return "supported"
        elif "contradictory" in verdict_lower:
            return "contradictory"
        elif "inconclusive" in verdict_lower:
            return "inconclusive"
        else:
            return ""  # Default fallback
            
    except Exception as e:
        logging.error(f"Failed to extract verdict using LLM: {str(e)}")
        return ""


async def evaluate_single_claim_async(claim: str, report_text: str = None, config: dict = None, model_id: str = "openai/gpt-4.1-mini") -> dict:
    """
    Evaluate a single claim using GPT Researcher's evaluator report functionality.
    
    Args:
        claim: The claim to evaluate
        report_text: Optional context from the original report
        config: Configuration options for GPT Researcher
        model_id: The model to use for verdict extraction
        
    Returns:
        Dictionary containing evaluation results
    """
    report_context = ""
    verdict_token_usage = TokenUsage()
    try:
        if report_text is None:
            # If no report context is provided, use a different prompt
            query = factuality_evaluation_no_context_prompt.format(claim=claim)
        else:
            report_context = extract_claim_context(report_text, claim, model_id=model_id)
            query = factuality_evaluation_prompt.format(context=report_context, claim=claim)

        config_kwargs = dict(config or {})
        verbose = bool(config_kwargs.pop("verbose", False))
        # Initialize GPT Researcher with evaluator report type
        researcher = GPTResearcher(
            query=query,
            # report_type=ReportType.EvaluatorReport.value,
            report_type=ReportType.DeepResearch.value,
            report_source=ReportSource.Web.value,
            tone=Tone.Objective,
            verbose=verbose,
            **config_kwargs
        )
        
        # Conduct research to gather evidence
        await researcher.conduct_research()
        
        # Generate the evaluator report
        evaluation_report = await researcher.write_report()
        
        # Extract verdict from the report using the utility function (off-thread to avoid blocking)
        verdict = await asyncio.to_thread(
            extract_verdict_from_report, evaluation_report, model_id, verdict_token_usage
        )
        
        return {
            "claim": claim,
            "evaluation": evaluation_report,
            "verdict": verdict,
            "sources": researcher.get_source_urls(),
            "context": report_context,
            "input_tokens": researcher.get_total_input_tokens(),
            "output_tokens": researcher.get_total_output_tokens(),
            "per_model_token_usage": getattr(researcher, "per_model_token_usage", {}),
            "verdict_token_usage": verdict_token_usage.dict(),
        }
        
    except Exception as e:
        logging.error(f"Failed to evaluate claim: {str(e)}")
        return {
            "claim": claim,
            "evaluation": None,
            "verdict": "Error",
            "sources": [],
            "research_costs": 0.0,
            "context": report_context,
            "error": str(e)
        }


def evaluate_single_claim_sync(claim_data: tuple) -> dict:
    """
    Synchronous wrapper for claim evaluation to work with ThreadPoolExecutor.
    
    Args:
        claim_data: Tuple of (idx, claim, report_context, config, model_id)
        
    Returns:
        Dictionary containing evaluation results with index
    """
    idx, claim, report_context, config, model_id = claim_data
    
    # Get or create event loop for this thread
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run the async evaluation
    result = loop.run_until_complete(
        evaluate_single_claim_async(claim, report_context, config, model_id)
    )
    
    # Add index for ordering
    result["idx"] = idx
    
    return result


async def _bounded_eval(
    idx: int,
    claim: str,
    report_text: str | None,
    config: dict,
    sem: asyncio.Semaphore,
    timeout_per_claim: float | None = None,
) -> dict:
    fast_llm = os.environ.get("FAST_LLM", "openai:gpt-4.1-mini")
    model_id = fast_llm.replace("openai:", "openai/")
    async with sem:
        try:
            coro = evaluate_single_claim_async(claim, report_text, config, model_id)
            if timeout_per_claim:
                result = await asyncio.wait_for(coro, timeout=timeout_per_claim)
            else:
                result = await coro
            result["idx"] = idx
            return result
        except Exception as e:
            logging.error(f"Claim {idx+1} evaluation failed: {e}")
            return {
                "idx": idx,
                "claim": claim,
                "evaluation": None,
                "verdict": "Error",
                "sources": [],
                "context": "",
                "error": str(e),
            }


async def _run_all_async(
    claims: list[str],
    report_text: str | None,
    config: dict,
    max_concurrency: int = 4,
    timeout_per_claim: float | None = None,
) -> list[dict]:
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [
        asyncio.create_task(
            _bounded_eval(i, claim, report_text, config or {}, sem, timeout_per_claim)
        )
        for i, claim in enumerate(claims)
    ]

    results_by_idx: dict[int, dict] = {}
    with tqdm(total=len(tasks), desc="Evaluating claims", unit="claim") as pbar:
        for task in asyncio.as_completed(tasks):
            res = await task
            results_by_idx[res["idx"]] = res
            pbar.update(1)
            pbar.set_postfix({
                'verdict': res.get('verdict', 'Unknown'),
                'in_tokens': res.get('input_tokens', 0),
                'out_tokens': res.get('output_tokens', 0),
            })

    return [results_by_idx[i] for i in range(len(claims)) if i in results_by_idx]

def evaluate_report_gpt_researcher(
    report_path: str,
    max_workers: int = 4,  # Reduced default to avoid rate limits
) -> list:
    """
    Evaluate multiple claims using GPT Researcher's evaluator report functionality.
    
    Args:
        report_path: Path to the report JSON file
        max_workers: Maximum number of worker threads for parallel evaluation
        model_id: The model to use for verdict extraction
        
    Returns:
        List of evaluation results for each claim
    """
    
    # config = {"scraper": "craw4ai"}
    config = None

    with open(report_path, 'r') as f:
        report_data = json.load(f)


    report = report_data["response"]
    annotated_sentences = [s | {"sentence_idx": i} for i, s in enumerate(report_data["sentences_info"]) if
                           s.get("human_verified") is True and s.get("human_verdict") is not None and s.get("sampled") is not None]
    # annotated_sentences = annotated_sentences[:4]
    claims  = [s["sentence"] for s in annotated_sentences]

    # Extract report information for context
    report_text = report_data.get("response") if report_data else None

    
    logging.info(f"Evaluating {len(claims)} claims using GPT Researcher evaluator reports")
    logging.info(f"Report context available: {bool(report_text)}")
    logging.info(f"Using {max_workers} workers for parallel processing")
    
    # Execute claims evaluation with a single asyncio loop and bounded concurrency
    results = asyncio.run(
        _run_all_async(
            claims=claims,
            report_text=report_text,
            config=config or {},
            max_concurrency=max_workers,
            timeout_per_claim=None,  # set a float (seconds) if desired
        )
    )
    
    total_input_tokens = sum(result.get("input_tokens", 0.0) for result in results)
    total_output_tokens = sum(result.get("output_tokens", 0.0) for result in results)
    # Aggregate GPTResearcher research token usage per model
    token_usage_by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0, "total": 0})
    for result in results:
        research_per_model = result.get("per_model_token_usage", {}) or {}
        for model, usage in research_per_model.items():
            inp = int(usage.get("input", 0))
            out = int(usage.get("output", 0))
            token_usage_by_model[model]["input"] += inp
            token_usage_by_model[model]["output"] += out
            token_usage_by_model[model]["total"] += (inp + out)
    
    # Log summary statistics
    successful_evaluations = len([r for r in results if r.get("verdict") != "Error"])
    verdict_counts = {}
    for result in results:
        verdict = result.get("verdict", "")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1


    
    logging.info(f"Evaluation completed: {successful_evaluations}/{len(claims)} successful")
    logging.info(f"Verdict distribution: {verdict_counts}")
    logging.info(f"Total input tokens: {total_input_tokens}, Total output tokens: {total_output_tokens}")
    if token_usage_by_model:
        logging.info(f"Research token usage by model: {dict(token_usage_by_model)}")

    # ---------------- Map back to original data ----------------
    processed_count = max(len(results), 1)
    per_model_usage_formatted: dict[str, dict[str, float]] = {}
    for model, stats in token_usage_by_model.items():
        total_input = int(stats.get("input", 0))
        total_output = int(stats.get("output", 0))
        per_model_usage_formatted[model] = {
            "avg_input": total_input / processed_count,
            "avg_output": total_output / processed_count,
            "total_input": total_input,
            "total_output": total_output,
        }

    # model_name = model_id.split("/")
    model_name = os.environ.get("STRATEGIC_LLM").split(":")[-1]
    depth = os.environ.get("DEEP_RESEARCH_DEPTH", "3")
    depth_suffix = f"-d{depth}" if depth.isdigit() and int(depth) != 3 else ""
    method_name = f"gr-{model_name}{depth_suffix}"

    # Map each evaluated claim back to its original sentence by original idx
    for res in results:
        claim_idx = int(res.get("idx", -1))
        if claim_idx < 0 or claim_idx >= len(annotated_sentences):
            continue
        orig_idx = annotated_sentences[claim_idx]["sentence_idx"]
        sinfo = report_data["sentences_info"][orig_idx]
        sinfo[f"{method_name}_verdict"] = res.get("verdict", "")
        sinfo[f"{method_name}_reason"] = res.get("evaluation", "") or ""
        sinfo[f"{method_name}_per_model_usage"] = per_model_usage_formatted

    relative_path = report_path.split("data/", 1)[-1] if "data/" in report_path else os.path.basename(report_path)
    save_path = f"results/{method_name}/{relative_path}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)


    return results

if __name__ == "__main__":

    report_dir = "/home/ubuntu/fact_eval/data/final"
    report_paths = [os.path.join(report_dir, fname) for fname in os.listdir(report_dir)
                    if fname.endswith(".json")]
    for report_path in report_paths:
    # Evaluate the claims
        results = evaluate_report_gpt_researcher(
            report_path,
            max_workers=8,  # Use fewer workers for testing
        )

    

    
