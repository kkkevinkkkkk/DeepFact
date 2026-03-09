import argparse
import asyncio
import json
import os
import glob
import random
from pathlib import Path

from deep_fact.evaluators import create_agent
from deep_fact.utils.llm_tools import (
    filter_verifiable_sentences,
    get_key_summary,
    rate_relevance,
    split_report_into_sentences,
)

DEFAULT_CONFIG_NAME = "deep_fact_eval_lite_gpt-4-1_gs5"
DEFAULT_REPORT_DIR = "data/deep_fact_bench/test_reports_split"
DEFAULT_RESULTS_ROOT = "results/deep_fact_score"
DEFAULT_EVALUATION_MODEL_ID = "openai/gpt-4.1"
DEFAULT_MAX_SENTENCE_NUM = 40
DEFAULT_MAX_VERIFICATION_WORKERS = 10
DEFAULT_EVALUATOR_AGENT_TYPE = "agent"


def calculate_deep_fact_score(report_path, model_id="openai/gpt-4.1", max_sentence_num=40, output_dir=None, evaluator_agent_type="tg_deep_evaluator", max_verification_workers=8, config_path=None):
    with open(report_path, "r") as f:
        data = json.load(f)
    report = data["response"]
    question = data.get("question", "")

    method_name = evaluator_agent_type

    # Step 1: Split the report into sentences
    print("Step 1: Splitting report into sentences...")
    sentences = split_report_into_sentences(report, model_id=model_id)
    print(f"Found {len(sentences)} sentences")

    # Step 2: Get the key thesis/summary of the report
    print("Step 2: Getting key thesis/summary...")
    key_summary = get_key_summary(question, report, model_name="openai/gpt-4.1-mini")
    print(f"Key thesis: {key_summary[:100]}...")

    # Step 3: Get verifiable sentences first
    print("Step 3: Filtering verifiable sentences...")
    verifiability_result = filter_verifiable_sentences(report, sentences, model_id)
    verifiable_sentences_basic = []
    
    for sentence_info in verifiability_result["sentences"]:
        idx = int(sentence_info["sentence_idx"])
        if sentence_info["label"] == "verifiable":
            # Get context (previous and next sentences)
            context_start = max(0, idx-5)
            context_end = min(len(sentences), idx+5)
            context = " ".join(sentences[context_start:context_end])
            
            verifiable_sentences_basic.append({
                "index": idx,
                "sentence": sentences[idx],
                "context": context,
                "verifiable": True,
                "verifiable_reason": sentence_info["reason"]
            })
    
    print(f"Found {len(verifiable_sentences_basic)} verifiable sentences")

    # Step 4: Rate the relevance of only the verifiable sentences using multi-threading
    print("Step 4: Rating relevance of verifiable sentences...")
    
    async def rate_single_relevance(args, semaphore):
        """Async worker to rate relevance of a single sentence."""
        sentence_info, question, key_summary = args
        async with semaphore:
            try:
                relevance_result = await asyncio.to_thread(
                    rate_relevance,
                    question,
                    key_summary,
                    sentence_info["context"],
                    sentence_info["sentence"],
                    "openai/gpt-4.1",
                )
                return sentence_info["index"], relevance_result
            except Exception as e:
                print(f"Error rating relevance for sentence {sentence_info['index']}: {e}")
                return sentence_info["index"], {"relevance": "unknown", "relevance_reason": f"Error: {str(e)}"}
    
    # Prepare arguments for multithreading
    relevance_args = [(sentence_info, question, key_summary) for sentence_info in verifiable_sentences_basic]
    
    relevance_results_dict = {}
    if verifiable_sentences_basic:
        max_relevance_workers = min(max_verification_workers, len(verifiable_sentences_basic))
        max_relevance_workers = max(max_relevance_workers, 1)
        print(f"Using {max_relevance_workers} async workers for parallel relevance rating...")

        async def run_relevance_tasks():
            semaphore = asyncio.Semaphore(max_relevance_workers)
            tasks = [asyncio.create_task(rate_single_relevance(args, semaphore)) for args in relevance_args]
            return await asyncio.gather(*tasks, return_exceptions=True)

        relevance_results = asyncio.run(run_relevance_tasks())
        for res in relevance_results:
            if isinstance(res, Exception):
                # Should not normally happen because we catch inside worker, but keep safe.
                print(f"Unexpected exception during relevance tasks: {res}")
                continue
            original_idx, relevance_result = res
            relevance_results_dict[original_idx] = relevance_result
    
    # Combine verifiable sentences with their relevance scores
    verifiable_sentences = []
    for sentence_info in verifiable_sentences_basic:
        idx = sentence_info["index"]
        relevance_result = relevance_results_dict.get(idx, {"relevance": "unknown", "relevance_reason": "No result"})
        
        verifiable_sentences.append({
            **sentence_info,
            "relevance_score": relevance_result.get("relevance", "unknown"),
            "relevance_reason": relevance_result.get("relevance_reason", "")
        })
    
    print(f"Completed relevance rating for {len(verifiable_sentences)} verifiable sentences")

    # Step 5: Choose verifiable sentences to verify with priority based on relevance
    print("Step 5: Prioritizing sentences for verification...")
    
    # Sort by relevance score, but shuffle sentences with the same relevance score
    relevance_priority = {"5": 5, "4": 4, "3": 3, "2": 2, "1": 1, "unknown": 0}
    
    # First, add a random value to break ties for sentences with the same relevance
    for sentence in verifiable_sentences:
        sentence["_random_tie_breaker"] = random.random()
    
    # Sort by relevance score (descending), then by random tie breaker for same relevance
    verifiable_sentences.sort(key=lambda x: (relevance_priority.get(str(x["relevance_score"]), 0), x["_random_tie_breaker"]), reverse=True)
    
    # Remove the temporary tie breaker field
    for sentence in verifiable_sentences:
        del sentence["_random_tie_breaker"]
    
    # Limit to max_sentence_num
    sentences_to_verify = verifiable_sentences[:max_sentence_num]
    # sort back to original order
    sentences_to_verify.sort(key=lambda x: x["index"])
    print(f"Will verify {len(sentences_to_verify)} sentences (limited by max_sentence_num={max_sentence_num})")

    # Step 6: Verify the sentences using the batched evaluator's evaluate_report
    print("Step 6: Verifying sentences...")

    verification_results_dict = {}  # Map original indices to results
    verification_results = []

    if sentences_to_verify:
        # Choose config for evaluator
        eval_config_path = config_path or "src/deep_fact/configs/batched_cached_gpt-4-1_gs10_v2_1.yaml"
        try:
            evaluator_instance = create_agent(eval_config_path, return_instance=True)
            report_payload = {"response": report, "thesis": key_summary}
            claims = [s["sentence"] for s in sentences_to_verify]
            eval_results = evaluator_instance.evaluate_report(
                report_data=report_payload,
                claims=claims,
                max_workers=max_verification_workers,
                clear_cache=True,
            )
        except Exception as e:
            print(f"✗ Error during batched verification: {e}")
            eval_results = [{"error": str(e)} for _ in sentences_to_verify]

        # Align results with the sampled sentences
        for sentence_info, res in zip(sentences_to_verify, eval_results):
            original_idx = sentence_info["index"]
            verdict = ""
            rationale = ""
            token_usage = {
                "input_tokens": res.get("input_tokens", 0) if isinstance(res, dict) else 0,
                "output_tokens": res.get("output_tokens", 0) if isinstance(res, dict) else 0,
            }

            if isinstance(res, dict):
                # Try to parse structured evaluation if present
                if "evaluation" in res and isinstance(res["evaluation"], str):
                    try:
                        parsed_eval = json.loads(res["evaluation"])
                        rationale = parsed_eval.get("rationale", "")
                        verdict = parsed_eval.get("verdict", "")
                    except Exception:
                        rationale = res["evaluation"]
                rationale = rationale or res.get("rationale", "") or res.get("error", "")
                verdict = (verdict or res.get("verdict", "")).lower()
            else:
                rationale = str(res)

            # Heuristic fallback to determine verdict
            verdict_lower = verdict.lower() if isinstance(verdict, str) else ""
            if verdict_lower not in {"supported", "contradictory", "inconclusive"}:
                if "supported" in rationale.lower():
                    verdict_lower = "supported"
                elif "contradictory" in rationale.lower() or "refute" in rationale.lower():
                    verdict_lower = "contradictory"
                elif "inconclusive" in rationale.lower():
                    verdict_lower = "inconclusive"
                else:
                    verdict_lower = "inconclusive"

            verification_result = {
                **sentence_info,
                "verification_result": rationale,
                "verdict": verdict_lower,
                "token_usage": token_usage,
            }

            verification_results_dict[original_idx] = verification_result
            verification_results.append(verification_result)

    print(f"✅ Completed verification of {len(verification_results)}/{len(sentences_to_verify)} sentences")

    # Step 7: Calculate the fact score
    print("Step 7: Calculating fact score...")
    
    supported_count = sum(1 for result in verification_results if result["verdict"] == "supported")
    total_verified = len([result for result in verification_results if result["verdict"] in ["supported", "contradictory", "inconclusive"]])
    
    if total_verified > 0:
        fact_score = supported_count / total_verified
    else:
        fact_score = 0.0
    
    print(f"Fact Score: {fact_score:.3f} ({supported_count}/{total_verified})")

    print("Preparing comprehensive sentence information...")

    verifiability_lookup = {int(s["sentence_idx"]): s for s in verifiability_result["sentences"]}
    relevance_lookup = relevance_results_dict  # Maps original_idx -> relevance result for verifiable sentences only
    verification_lookup = verification_results_dict  # Use the dict directly since it already maps original_idx -> result
    sampled_indices = {s["index"] for s in sentences_to_verify}

    sentences_output = []
    for i, sentence in enumerate(sentences):
        verifiability = "not_evaluated"
        verifiability_reason = ""
        if i in verifiability_lookup:
            verif_data = verifiability_lookup[i]
            verifiability = verif_data.get("label", "not_evaluated")
            verifiability_reason = verif_data.get("reason", "")

        relevance = "unknown"
        relevance_reason = ""
        if i in relevance_lookup:
            relevance_data = relevance_lookup[i]
            relevance = relevance_data.get("relevance", "unknown")
            relevance_reason = relevance_data.get("relevance_reason", "")

        sampled = i in sampled_indices
        sentence_entry = {
            "sentence_idx": i,
            "sentence": sentence,
            "relevance": relevance,
            "relevance_reason": relevance_reason,
            "verifiability": verifiability,
            "verifiability_reason": verifiability_reason,
            "sampled": sampled,
        }

        if sampled and i in verification_lookup:
            verif_result = verification_lookup[i]
            sentence_entry[f"{method_name}_verdict"] = verif_result.get("verdict")
            sentence_entry[f"{method_name}_reason"] = verif_result.get("verification_result")
            sentence_entry[f"{method_name}_token_usage"] = verif_result.get("token_usage")

        sentences_output.append(sentence_entry)

    # Mutate original data with the new fields while preserving existing content.
    data["thesis"] = key_summary
    data["deep_fact_score"] = fact_score
    data["supported_count"] = supported_count
    data["contradictory_count"] = sum(1 for result in verification_results if result["verdict"] == "contradictory")
    data["inconclusive_count"] = sum(1 for result in verification_results if result["verdict"] == "inconclusive")
    data["error_count"] = sum(1 for result in verification_results if result["verdict"] == "error")
    data["sentences_info"] = sentences_output
    data["verifiable_sentences_count"] = len(verifiable_sentences)
    data["verified_sentences_count"] = len(verification_results)

    # Save the results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # filename = os.path.basename(report_path).replace(".json", "_fact_score.json")
        filename = os.path.basename(report_path)
        output_path = os.path.join(output_dir, filename)
    else:
        output_path = report_path.replace(".json", "_fact_score.json")
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    
    return data


def process_report_wrapper(args):
    """Wrapper function for multithreading"""
    report_path, model_id, max_sentence_num, output_dir, evaluator_agent_type, max_verification_workers, config_path = args
    try:
        print(f"Processing {report_path}...")
        result = calculate_deep_fact_score(report_path, model_id, max_sentence_num, output_dir, evaluator_agent_type, max_verification_workers, config_path)
        print(f"✓ Completed {report_path}")
        return result
    except Exception as e:
        print(f"✗ Error processing {report_path}: {e}")
        return None


def process_model_reports(model_name, report_dir, output_dir, model_id="openai/gpt-4.1", max_sentence_num=40, max_workers=1, evaluator_agent_type="hf_deep_evaluator", max_verification_workers=4, max_reports=None, config_path=None):
    """Process all reports for a given model using multithreading"""
    print(f"\n🔄 Processing reports for model: {model_name}")
    print(f"Report directory: {report_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all JSON files in the report directory
    report_paths = glob.glob(os.path.join(report_dir, "*.json"))
    # sort report paths
    report_paths.sort()
    if max_reports:
        # sort the report_paths by the file name
        report_paths.sort(key=lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0]))
        report_paths = report_paths[:max_reports]
    
    if not report_paths:
        print(f"⚠️  No JSON files found in {report_dir}")
        return []
    
    print(f"Found {len(report_paths)} reports to process")
    
    # Process reports sequentially for predictable behavior
    results = []
    for report_path in report_paths[::-1]:
        # if output file already exists, skip
        filename = os.path.basename(report_path)
        output_path = os.path.join(output_dir, filename)
        if os.path.exists(output_path):
            print(f"Result already exists at {output_path}, skipping...")
            continue
        result = process_report_wrapper(
            (report_path, model_id, max_sentence_num, output_dir, evaluator_agent_type, max_verification_workers, config_path)
        )
        if result:
            results.append(result)

    print(f"✅ Completed processing {len(results)}/{len(report_paths)} reports for {model_name}")
    return results


def _resolve_config_path(config: str) -> Path:
    requested = Path(config)
    if requested.exists():
        return requested

    config_dir = Path("src/deep_fact/configs")
    if requested.suffix == ".yaml":
        candidate = config_dir / requested.name
    else:
        candidate = config_dir / f"{config}.yaml"

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Config not found: '{config}'. Tried '{requested}' and '{candidate}'."
    )


def _collect_report_paths(report_dir: Path | None, report_path: Path | None) -> list[Path]:
    if report_path is not None:
        if not report_path.exists():
            raise FileNotFoundError(f"Report path does not exist: {report_path}")
        if report_path.suffix.lower() != ".json":
            raise ValueError(f"Report path must be a .json file: {report_path}")
        return [report_path]

    if report_dir is None:
        raise ValueError("Either report_dir or report_path must be provided.")
    if not report_dir.exists():
        raise FileNotFoundError(f"Report directory does not exist: {report_dir}")

    return sorted(path for path in report_dir.iterdir() if path.suffix.lower() == ".json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate Deep Fact score for report JSON files.")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_NAME,
        help="Config name or path. Example: batched_cached_gpt-4-1_gs10_v2_1",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(DEFAULT_REPORT_DIR),
        help="Directory containing report .json files.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Calculate score for a specific report .json path.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(DEFAULT_RESULTS_ROOT),
        help="Root output directory for processed reports.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_EVALUATION_MODEL_ID,
        help="Model id used for sentence split/relevance steps.",
    )
    parser.add_argument(
        "--max-sentence-num",
        type=int,
        default=DEFAULT_MAX_SENTENCE_NUM,
        help="Maximum number of verifiable sentences to verify.",
    )
    parser.add_argument(
        "--max-verification-workers",
        type=int,
        default=DEFAULT_MAX_VERIFICATION_WORKERS,
        help="Parallel worker count for sentence verification/relevance.",
    )
    parser.add_argument(
        "--evaluator-agent-type",
        type=str,
        default=DEFAULT_EVALUATOR_AGENT_TYPE,
        help="Field prefix for saved verdict/reason/token-usage fields.",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=-1,
        help="Maximum number of reports to process. -1 means no limit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result files. Default skips existing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = _resolve_config_path(args.config)
    model_prefix = config_path.stem

    report_paths = _collect_report_paths(args.report_dir, args.report_path)
    if len(report_paths) == 0:
        print("No report files found to process.")
        return

    processed = 0
    skipped = 0

    for report_path in report_paths:
        if args.max_report > -1 and processed >= args.max_report:
            print(f"Reached max_report={args.max_report}. Stopping.")
            break

        result_path = args.results_root / model_prefix / report_path.name
        if result_path.exists() and not args.overwrite:
            skipped += 1
            print(f"Result already exists at {result_path}, skipping (resume mode).")
            continue

        print(f"Processing {report_path} ...")
        try:
            calculate_deep_fact_score(
                report_path=str(report_path),
                model_id=args.model_id,
                max_sentence_num=args.max_sentence_num,
                output_dir=str(result_path.parent),
                evaluator_agent_type=args.evaluator_agent_type,
                max_verification_workers=args.max_verification_workers,
                config_path=str(config_path),
            )
            processed += 1
        except Exception as e:
            print(f"Error processing {report_path}: {e}")

    print(
        f"Done. processed={processed}, skipped={skipped}, total_candidates={len(report_paths)}, "
        f"config='{config_path}', model_prefix='{model_prefix}'"
    )


if __name__ == "__main__":
    main()
