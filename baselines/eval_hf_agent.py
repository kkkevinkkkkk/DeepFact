import argparse
import json
from pathlib import Path
from typing import Any

from baselines.deep_research_hf.hf_eval_agent import evaluate_reports
from deep_fact.evaluators.utils.logging import AgentLogger
from deep_fact.utils.metric import calculate_scores

logging = AgentLogger("evaluate_hf_agent")


def _extract_verified_sentences(
    data: dict[str, Any],
    require_human_verified: bool = False,
) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for sentence_idx, sentence in enumerate(data.get("sentences_info", [])):
        if sentence.get("human_verdict") is None:
            continue
        if require_human_verified and sentence.get("human_verified") is not True:
            continue
        verified.append(sentence | {"sentence_idx": sentence_idx})
    return verified


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


def evaluate_report(
    data: dict[str, Any],
    model_id: str,
    max_workers: int = 8,
    retries: int = 1,
    retry_delay: float = 1.0,
    require_human_verified: bool = False,
    skip_sampled_labels: set[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a single report with HF backend using evaluate_report.py-compatible shape."""
    return evaluate_reports(
        data=data,
        model_id=model_id,
        max_workers=max_workers,
        retries=retries,
        retry_delay=retry_delay,
        require_human_verified=require_human_verified,
        skip_sampled_labels=skip_sampled_labels,
    )


DEFAULT_MODEL_ID = "openai/gpt-4.1"
DEFAULT_REPORT_DIR = "data/deep_fact_bench/test_reports_split"
DEFAULT_RESULTS_ROOT = "results/deep_fact_bench_hf"
DEFAULT_FIELD_PREFIX = "hf"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reports with HF agent backend.")
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Model id for HF agent (default: {DEFAULT_MODEL_ID}).",
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
        help=f"Prefix for saved sentence-level fields (default: {DEFAULT_FIELD_PREFIX}).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retry count per claim when HF evaluation fails.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Delay between retries in seconds.",
    )
    parser.add_argument(
        "--require-human-verified",
        action="store_true",
        help="If set, only evaluate sentences where human_verified is true and human_verdict is present.",
    )
    parser.add_argument(
        "--skip-sampled-labels",
        type=str,
        nargs="*",
        default=[],
        help="Optional sampled labels to skip (example: citation negative).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    report_paths = _collect_report_paths(args.report_dir, args.report_path)
    if len(report_paths) == 0:
        logging.warning("No report files found to process.")
        return

    model_prefix = args.model_id.replace("/", "__").replace(":", "_")
    field_prefix = args.field_prefix
    max_reports = args.max_report
    skip_labels = set(args.skip_sampled_labels)

    processed = 0
    skipped = 0

    for report_path in report_paths:
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
            data=data,
            model_id=args.model_id,
            max_workers=args.max_workers,
            retries=args.retries,
            retry_delay=args.retry_delay,
            require_human_verified=args.require_human_verified,
            skip_sampled_labels=skip_labels,
        )
        evaluation_results = results["evaluation"]
        token_usage = results["token_usage"]
        token_breakdown = results["token_breakdown"]
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
            data["sentences_info"][sentence_idx][f"{field_prefix}_verdict"] = result.get("verdict")
            data["sentences_info"][sentence_idx][f"{field_prefix}_reason"] = result.get("rationale", "")
            data["sentences_info"][sentence_idx][f"{field_prefix}_context"] = result.get("context", "")
            data["sentences_info"][sentence_idx][f"{field_prefix}_error"] = result.get("error")
            data["sentences_info"][sentence_idx][f"{field_prefix}_input_tokens"] = result.get("input_tokens", 0)
            data["sentences_info"][sentence_idx][f"{field_prefix}_output_tokens"] = result.get("output_tokens", 0)
            data["sentences_info"][sentence_idx][f"{field_prefix}_avg_input_tokens"] = avg_input_tokens
            data["sentences_info"][sentence_idx][f"{field_prefix}_avg_output_tokens"] = avg_output_tokens
            data["sentences_info"][sentence_idx][f"{field_prefix}_per_model_usage"] = per_model_avg_usage
            if "token_breakdown" in result:
                data["sentences_info"][sentence_idx][f"{field_prefix}_token_breakdown"] = result["token_breakdown"]

        scored_sentences = _extract_verified_sentences(
            data,
            require_human_verified=args.require_human_verified,
        )
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

        data[f"{field_prefix}_report_token_breakdown"] = token_breakdown
        data[f"{field_prefix}_scores"] = scores

        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as file:
            json.dump(data, file)
        logging.info(f"Saved to {result_path} (evaluated {len(evaluation_results)} claims)")
        processed += 1

    logging.info(
        f"Done. processed={processed}, skipped={skipped}, total_candidates={len(report_paths)}, "
        f"model_id='{args.model_id}', field_prefix='{field_prefix}'"
    )


if __name__ == "__main__":
    main()
