from typing import Any, Optional

import tenacity
from litellm import acompletion, completion
from dataclasses import dataclass, field
from tenacity import retry, wait_exponential, RetryCallState
import re
VLLM_HOST_URL = "http://0.0.0.0:20002/v1/"

@dataclass
class TokenUsage:
    """
    Contains the token usage information for a given step or run.
    Supports both single model usage (backward compatibility) and multi-model usage.
    """

    input_tokens: int = field(default=0)
    output_tokens: int = field(default=0)
    total_tokens: int = field(init=False, default=0)
    # Dictionary to track per-model usage: {model_name: {"input": int, "output": int}}
    per_model_usage: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        self.total_tokens = self.input_tokens + self.output_tokens

    def add_model_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Add token usage for a specific model"""
        if model not in self.per_model_usage:
            self.per_model_usage[model] = {"input": 0, "output": 0}
        
        self.per_model_usage[model]["input"] += input_tokens
        self.per_model_usage[model]["output"] += output_tokens
        
        # Update total counters
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens = self.input_tokens + self.output_tokens

    def get_model_usage(self, model: str) -> dict[str, int]:
        """Get token usage for a specific model"""
        return self.per_model_usage.get(model, {"input": 0, "output": 0})

    def get_all_models(self) -> list[str]:
        """Get list of all models that have been used"""
        return list(self.per_model_usage.keys())

    def dict(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "per_model_usage": self.per_model_usage,
        }

    def __str__(self):
        base_str = f"input_tokens={self.input_tokens}, output_tokens={self.output_tokens}, total_tokens={self.total_tokens}"
        if self.per_model_usage:
            model_details = []
            for model, usage in self.per_model_usage.items():
                model_details.append(f"{model}: {usage['input']}+{usage['output']}={usage['input']+usage['output']}")
            base_str += f", per_model=[{', '.join(model_details)}]"
        return base_str

    # ---------- addition semantics ----------
    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented
        
        result = TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )
        
        # Merge per-model usage
        result.per_model_usage = self.per_model_usage.copy()
        for model, usage in other.per_model_usage.items():
            if model in result.per_model_usage:
                result.per_model_usage[model]["input"] += usage["input"]
                result.per_model_usage[model]["output"] += usage["output"]
            else:
                result.per_model_usage[model] = usage.copy()
        
        return result

    # Optional: support ``sum([...])`` and ``0 + a`` idioms
    def __radd__(self, other):
        if other == 0:  # ``sum`` starts with 0
            return self
        return self.__add__(other)

    # Optional: in-place ``a += b`` (mutates ``a``)
    def __iadd__(self, other: "TokenUsage"):
        if not isinstance(other, TokenUsage):
            return NotImplemented
        
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens = self.input_tokens + self.output_tokens
        
        # Merge per-model usage
        for model, usage in other.per_model_usage.items():
            if model in self.per_model_usage:
                self.per_model_usage[model]["input"] += usage["input"]
                self.per_model_usage[model]["output"] += usage["output"]
            else:
                self.per_model_usage[model] = usage.copy()
        
        return self

    @classmethod
    def from_str(cls, s: str) -> "TokenUsage":
        """
        Re‑create a ``TokenUsage`` instance from the string produced by ``__str__``.

        Expected format (exactly what ``__str__`` emits):
            'input_tokens=<int>, output_tokens=<int>, total_tokens=<int>[, per_model=[...]]'
        """
        # Handle both old and new format for backward compatibility
        base_match = re.search(
            r"input_tokens=(\d+),\s*output_tokens=(\d+),\s*total_tokens=(\d+)", s.strip()
        )
        if not base_match:
            raise ValueError(
                "String does not match TokenUsage.__str__ format "
                f"(got {s!r})"
            )
        
        in_tok, out_tok, tot_tok = map(int, base_match.groups())

        # Defensive check – catches accidental mismatches.
        if in_tok + out_tok != tot_tok:  # pragma: no cover
            raise ValueError(
                "total_tokens field is inconsistent with input+output "
                f"({in_tok}+{out_tok} != {tot_tok})"
            )

        result = cls(input_tokens=in_tok, output_tokens=out_tok)
        
        # Try to parse per-model usage if present
        per_model_match = re.search(r"per_model=\[(.*?)\]", s)
        if per_model_match:
            model_details = per_model_match.group(1)
            # Parse individual model entries like "model_name: 100+200=300"
            for model_entry in model_details.split(", "):
                if ":" in model_entry:
                    model_name, usage_str = model_entry.split(":", 1)
                    model_name = model_name.strip()
                    # Parse usage like "100+200=300"
                    usage_match = re.match(r"(\d+)\+(\d+)=\d+", usage_str.strip())
                    if usage_match:
                        model_input, model_output = map(int, usage_match.groups())
                        result.per_model_usage[model_name] = {
                            "input": model_input,
                            "output": model_output
                        }
        
        return result

def dynamic_stop(rs: RetryCallState) -> bool:
    # Pull `model` from kwargs (or args[0] if you like positional)
    model = rs.kwargs.get("model", "")
    max_attempts = 5 if "bedrock" in model or "gemini" in model else 3
    return rs.attempt_number > max_attempts            # → True ⇒ stop

def dynamic_wait(rs: RetryCallState) -> float:
    model = rs.kwargs.get("model", "")
    if "bedrock" in model or "gemini" in model:               # slower queue, back off longer
        return min(60, 15 * 2 ** (rs.attempt_number - 1))

    # default: 4, 8, 16 … capped at 15 s (same shape as wait_exponential)
    return min(15, 4 * 2 ** (rs.attempt_number - 1))

@retry(stop=dynamic_stop, wait=dynamic_wait, reraise=True)
# @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=4, max=15))
async def asingle_shot_llm_call(
    model: str,
    system_prompt: str,
    message: str,
    response_format: Optional[dict[str, str | dict[str, Any]]] = None,
    max_completion_tokens: int | None = None,
    token_usage: TokenUsage | None = None,
) -> str:
    api_base = None if "hosted_vllm" not in model else VLLM_HOST_URL
    timeout = 600 if "hosted_vllm" not in model else 2400

    temperature = 0.0
    if "gpt-5" in model:
        temperature = 1.0

    req_kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": message}],
        temperature=temperature,
        response_format=response_format,
        # NOTE: max_token is deprecated per OpenAI API docs, use max_completion_tokens instead if possible
        # NOTE: max_completion_tokens is not currently supported by Together AI, so we use max_tokens instead
        max_tokens=max_completion_tokens,
        api_base=api_base,
        timeout=timeout,
    )
    if "qwen3" in model.lower():
        req_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    response = await acompletion(**req_kwargs)
    content = response.choices[0].message["content"]
    
    # Track token usage per model
    if token_usage:
        token_usage.add_model_usage(
            model=model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
    
    return content  # type: ignore


# @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=4, max=15))
@retry(stop=dynamic_stop, wait=dynamic_wait, reraise=True)
def single_shot_llm_call(
    model: str,
    system_prompt: str,
    message: str,
    response_format: Optional[dict[str, str | dict[str, Any]]] = None,
    max_completion_tokens: int | None = None,
    token_usage: TokenUsage | None = None,
) -> str:
    api_base = None if "hosted_vllm" not in model else VLLM_HOST_URL
    timeout = 600 if "hosted_vllm" not in model else 2400

    temperature = 0.0
    if "gpt-5" in model:
        temperature = 1.0

    req_kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": message}],
        temperature=temperature,
        response_format=response_format,
        # NOTE: max_token is deprecated per OpenAI API docs, use max_completion_tokens instead if possible
        # NOTE: max_completion_tokens is not currently supported by Together AI, so we use max_tokens instead
        max_tokens=max_completion_tokens,
        api_base=api_base,
        timeout=timeout,
    )
    if "qwen3" in model.lower():
        req_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    response = completion(**req_kwargs)
    
    # Track token usage per model
    if token_usage:
        token_usage.add_model_usage(
            model=model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
    
    return response.choices[0].message["content"]  # type: ignore



def generate_toc_image(prompt: str, planning_model: str, topic: str, token_usage: TokenUsage | None = None) -> str:
    """Generate a table of contents image"""
    from together import Together

    image_generation_prompt = single_shot_llm_call(
        model=planning_model, 
        system_prompt=prompt, 
        message=f"Research Topic: {topic}",
        token_usage=token_usage
    )

    if image_generation_prompt is None:
        raise ValueError("Image generation prompt is None")

    # HERE WE CALL THE TOGETHER API SINCE IT'S AN IMAGE GENERATION REQUEST
    client = Together()
    imageCompletion = client.images.generate(
        model="black-forest-labs/FLUX.1-dev",
        width=1024,
        height=768,
        steps=28,
        prompt=image_generation_prompt,
    )

    return imageCompletion.data[0].url  # type: ignore
