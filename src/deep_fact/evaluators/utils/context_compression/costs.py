import os
from typing import Any, Sequence


OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

ENCODING_MODEL = "o200k_base"
EMBEDDING_COST = 0.02 / 1000000


def estimate_embedding_cost(model: str, docs: Sequence[Any]) -> float:
    try:
        import tiktoken
    except ImportError:
        return 0.0

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        try:
            encoding = tiktoken.get_encoding(ENCODING_MODEL)
        except Exception:
            return 0.0

    total_tokens = sum(len(encoding.encode(str(doc))) for doc in docs)
    return total_tokens * EMBEDDING_COST
