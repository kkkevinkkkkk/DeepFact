from typing import Any

import yaml
import asyncio
# from fact_eval.deep_research_tg.together_open_deep_research import DeepResearcher

# from deep_fact.evaluators.core.deep_fact_eval import DeepFactEvaluator
from deep_fact.evaluators.core.deep_fact_eval_lite import DeepFactEvaluatorLite
from typing import List

def load_config(config_path: str):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def create_agent(config: str, return_instance: bool = False, **kwargs) -> Any:
    """
    Factory method to create an agent with specified configuration.
    """

    config_dict = load_config(config)

    agent_config = config_dict.get("agent")
    if not agent_config:
        raise ValueError(f"Missing 'agent' section in config: {config}")

    agent_config = dict(agent_config)
    agent_type = agent_config.pop("type")


    if agent_type.startswith("deep_evaluator"):
        if "max_steps" in agent_config:
            agent_config["budget"] = agent_config.pop("max_steps")

        # if agent_type == "deep_evaluator_advanced_cached":
        #     researcher = DeepFactEvaluator(**agent_config)

        if agent_type == "deep_fact_eval_lite":
            researcher = DeepFactEvaluatorLite(**agent_config)
            if "group_size" in kwargs:
                researcher.group_size = kwargs["group_size"]
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        if return_instance:
            return researcher

        def research_wrapper(goal: str):
            return asyncio.run(researcher.research_topic(goal))
        
        def research_wrapper_batched(report: str, claims: List[str]):
            return asyncio.run(researcher.evaluate_report_claims(report, claims))

        if agent_type == "deep_fact_eval_lite":
            return research_wrapper_batched
        else:
            return research_wrapper


    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
