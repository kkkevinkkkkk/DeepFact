import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from deep_fact.evaluators import create_agent
from deep_fact.evaluators.utils.logging import AgentLogger
from deep_fact.utils.metric import calculate_scores

logging = AgentLogger("evaluate_report")
def _extract_verified_sentences(data: dict[str, Any]) -> list[dict[str, Any]]:
    verified = [
        sentence | {"sentence_idx": sentence_idx}
        for sentence_idx, sentence in enumerate(data["sentences_info"])
        # if sentence.get("human_verified") is True and sentence.get("human_verdict") is not None
        if sentence.get("human_verdict",None) is not None
    ]
    return verified


def evaluate_report(
    data: dict[str, Any],
    config_path: str,
    max_workers: int = 8,
) -> dict[str, Any]:
    logging.info(f"Using configuration from file: {config_path}")

    report = data["response"]
    annotated_sentences = _extract_verified_sentences(data)
    claims = [item["sentence"] for item in annotated_sentences]

    evaluator = create_agent(config_path, return_instance=True)
    report_data = {"response": report, "thesis": data.get("thesis", "")}

    raw_results = evaluator.evaluate_report(
        report_data=report_data,
        claims=claims,
        max_workers=max_workers,
        clear_cache=True,
    )
    indexed_results = [result | {"idx": idx} for idx, result in enumerate(raw_results)]

    formatted_results: list[dict[str, Any]] = []
    for result in indexed_results:
        sentence_idx = annotated_sentences[result["idx"]]["sentence_idx"]
        if "error" in result:
            formatted_results.append(
                {
                    "sentence_idx": sentence_idx,
                    "claim": result["claim"],
                    "error": result["error"],
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            )
            print(result['error'])
            continue

        rationale = result.get("rationale", "")
        verdict = result.get("verdict", "")

        formatted_results.append(
            {
                "sentence_idx": sentence_idx,
                "claim": result["claim"],
                "context": result.get("context", ""),
                "rationale": rationale,
                "verdict": str(verdict).lower(),
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
            }
        )

    return {"evaluation": formatted_results, "token_usage": evaluator.token_usage}


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


DEFAULT_CONFIG_NAME = "deep_fact_eval_lite_gpt-4-1_gs5"
DEFAULT_REPORT_DIR = "data/deep_fact_bench/test_reports_split"
DEFAULT_RESULTS_ROOT = "results/deep_fact_bench"
DEFAULT_FIELD_PREFIX = "agent"

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deep-fact benchmark reports.")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_NAME,
        help="Config name or path. Example: deep_fact_eval_lite_gpt-4-1_gs5",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Max worker count used by evaluator.",
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
        help="Overwrite existing result files. Default behavior resumes by skipping existing results.",
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
        help="Evaluate a specific report .json path.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(DEFAULT_RESULTS_ROOT),
        help="Root output directory for evaluated reports.",
    )
    parser.add_argument(
        "--field-prefix",
        type=str,
        default=DEFAULT_FIELD_PREFIX,
        help="Prefix for saved sentence-level fields (default: agent).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = _resolve_config_path(args.config)
    model_prefix = config_path.stem
    field_prefix = args.field_prefix
    max_reports = args.max_report

    report_paths = _collect_report_paths(args.report_dir, args.report_path)
    if len(report_paths) == 0:
        logging.warning("No report files found to process.")
        return

    processed = 0
    skipped = 0

    for report_path in tqdm(report_paths):
        if max_reports > -1 and processed >= max_reports:
            logging.info(f"Reached max_report={max_reports}. Stopping.")
            break

        result_path = args.results_root / model_prefix / report_path.name
        if result_path.exists() and not args.overwrite:
            skipped += 1
            logging.info(f"Result already exists at {result_path}, skipping (resume mode).")
            continue

        with open(report_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        logging.info(f"Loaded report from {report_path}")

        results = evaluate_report(
            data,
            config_path=str(config_path),
            max_workers=args.max_workers,
        )
        evaluation_results = results["evaluation"]
        token_usage = results["token_usage"]
        num_claims = len(evaluation_results)

        avg_input_tokens = token_usage.input_tokens / num_claims if num_claims > 0 else 0.0
        avg_output_tokens = token_usage.output_tokens / num_claims if num_claims > 0 else 0.0
        logging.info(f"Avg input tokens: {avg_input_tokens:.2f}")
        logging.info(f"Avg output tokens: {avg_output_tokens:.2f}")

        per_model_avg_usage: dict[str, dict[str, float | int]] = {}
        if token_usage.per_model_usage and num_claims > 0:
            logging.info("Per-model token usage:")
            for model in token_usage.get_all_models():
                model_usage = token_usage.get_model_usage(model)
                model_total = model_usage["input"] + model_usage["output"]
                avg_model_input = model_usage["input"] / num_claims
                avg_model_output = model_usage["output"] / num_claims
                per_model_avg_usage[model] = {
                    "avg_input": avg_model_input,
                    "avg_output": avg_model_output,
                    "total_input": model_usage["input"],
                    "total_output": model_usage["output"],
                }
                logging.info(
                    f"  {model}: {model_usage['input']} input + {model_usage['output']} output = {model_total} total"
                )

        for result in evaluation_results:
            sentence_idx = result["sentence_idx"]
            if result.get("verdict") is None:
                logging.warning(f"Verdict is None for sentence_idx={sentence_idx}")
            data["sentences_info"][sentence_idx][f"{field_prefix}_verdict"] = result.get("verdict")
            data["sentences_info"][sentence_idx][f"{field_prefix}_reason"] = result.get("rationale")
            data["sentences_info"][sentence_idx][f"{field_prefix}_avg_input_tokens"] = avg_input_tokens
            data["sentences_info"][sentence_idx][f"{field_prefix}_avg_output_tokens"] = avg_output_tokens
            data["sentences_info"][sentence_idx][f"{field_prefix}_per_model_usage"] = per_model_avg_usage

        result_path.parent.mkdir(parents=True, exist_ok=True)

        scored_sentences = _extract_verified_sentences(data)
        eval_result_by_sentence_idx = {
            result["sentence_idx"]: result
            for result in evaluation_results
        }
        human_verdicts = []
        model_verdicts = []
        for sentence in scored_sentences:
            sentence_idx = sentence["sentence_idx"]
            human_verdicts.append(sentence.get("human_verdict"))
            eval_result = eval_result_by_sentence_idx.get(sentence_idx, {})
            model_verdict = eval_result.get("verdict")
            if model_verdict is None:
                logging.warning(
                    f"Missing verdict for sentence_idx={sentence_idx} "
                    f"error={eval_result.get('error')}"
                )

            model_verdicts.append(model_verdict)
        scores = calculate_scores(human_verdicts, model_verdicts)
        logging.info(f"Evaluation Scores: {scores}")

        with open(result_path, "w", encoding="utf-8") as file:
            json.dump(data, file)
        logging.info(f"Saved to {result_path} (evaluated {len(evaluation_results)} claims)")
        processed += 1

    logging.info(
        f"Done. processed={processed}, skipped={skipped}, total_candidates={len(report_paths)}, "
        f"config='{config_path}', model_prefix='{model_prefix}', field_prefix='{field_prefix}'"
    )


if __name__ == "__main__":
    main()
