import argparse
import json
from pathlib import Path
from typing import Any

from deep_fact.evaluators import create_agent
from deep_fact.evaluators.utils.logging import AgentLogger

logging = AgentLogger("evaluate_claims")

DEFAULT_CONFIG_NAME = "deep_fact_eval_lite_gpt-4-1_gs5"

DEMO_CLAIMS_ONLY = [
    "The Eiffel Tower grows about 15 centimeters taller in summer due to thermal expansion.",
    "Australia is home to more than 10,000 beaches.",
    "The shortest war in recorded history lasted less than an hour.",
    "Bananas are botanically classified as berries, while strawberries are not.",
    "Japan has more pet dogs and cats than children under age 15.",
]

DEMO_SHARED_CONTEXT = (
    "Apple announced the iPhone 17e on March 2, 2026. Apple said it is the most "
    "affordable model in the iPhone 17 family, starts at $599, comes with 256GB "
    "starting storage, uses the A19 chip, has a 6.1-inch Super Retina XDR display, "
    "includes a 48MP Fusion camera, supports MagSafe, and opens for pre-order on "
    "March 4 with availability on March 11."
)

DEMO_CLAIMS_WITH_CONTEXT = [
    "Apple announced the iPhone 17e as the most affordable member of the iPhone 17 family.",
    "The iPhone 17e starts at 128GB of storage.",
    "The iPhone 17e starts at a price of $599.",
    "The iPhone 17e uses Apple's A18 chip.",
    "The phone includes a 48MP Fusion camera.",
    "The iPhone 17e has a 6.7-inch display.",
    "The device supports MagSafe accessories.",
    "Apple said pre-orders begin on March 11.",
    "The iPhone 17e is available in black, white, and soft pink.",
    "The iPhone 17e does not include satellite features.",
]


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


def _format_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for idx, result in enumerate(results):
        formatted.append(
            {
                "idx": idx,
                "claim": result.get("claim", ""),
                "verdict": str(result.get("verdict", "")).lower(),
                "rationale": result.get("rationale", result.get("error", "")),
                "context": result.get("context", ""),
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "error": result.get("error"),
            }
        )
    return formatted


def run_demo(config_path: str, max_workers: int = 20) -> dict[str, Any]:
    evaluator = create_agent(config_path, return_instance=True)

    logging.info("Running demo: claims only (no context)")
    claims_only_results = evaluator.evaluate_claims(
        context=None,
        claims=DEMO_CLAIMS_ONLY,
        max_workers=max_workers,
    )

    logging.info("Running demo: claims with shared context")
    claims_with_context_results = evaluator.evaluate_claims(
        context=DEMO_SHARED_CONTEXT,
        claims=DEMO_CLAIMS_WITH_CONTEXT,
        max_workers=max_workers,
    )

    return {
        "claims_only": _format_results(claims_only_results),
        "claims_with_shared_context": _format_results(claims_with_context_results),
        "token_usage": {
            "input_tokens": evaluator.token_usage.input_tokens,
            "output_tokens": evaluator.token_usage.output_tokens,
            "per_model_usage": evaluator.token_usage.per_model_usage,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo DeepFact claim evaluation with and without shared context."
    )
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
        "--output",
        type=str,
        default="",
        help="Optional path to save JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = _resolve_config_path(args.config)
    results = run_demo(str(config_path), max_workers=args.max_workers)
    rendered = json.dumps(results, indent=2, ensure_ascii=False)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        logging.info(f"Demo results written to {output_path}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
