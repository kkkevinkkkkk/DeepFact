"""Lightweight context compression helper for fact_eval."""

from __future__ import annotations

from typing import Any, Callable, Optional


from deep_fact.evaluators.utils.context_compression.compression import ContextCompressor
from deep_fact.evaluators.utils.context_compression.costs import OPENAI_EMBEDDING_MODEL
from deep_fact.evaluators.utils.context_compression.prompt_family import PromptFamily

try:  # langchain-openai is an optional dep in some installs
    from langchain_openai import OpenAIEmbeddings  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    OpenAIEmbeddings = None  # type: ignore


class ContextManager:
    """Manages query-focused context compression for retrieved pages."""

    def __init__(
        self,
        embeddings: Any | None = None,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        cost_callback: Optional[Callable[[float], None]] = None,
        **kwargs: Any,
    ):
        """
        Create a lightweight context manager that does not depend on a
        GPT Researcher installation. If no embeddings are provided, defaults
        to `OpenAIEmbeddings` with `OPENAI_EMBEDDING_MODEL`.
        """
        self.embeddings = embeddings or self._default_embeddings()
        self.prompt_family = prompt_family
        self.cost_callback = cost_callback
        self.kwargs = kwargs

    def _default_embeddings(self):
        if OpenAIEmbeddings is None:
            raise ImportError(
                "OpenAIEmbeddings is unavailable. Install langchain-openai or pass an embeddings instance."
            )
        return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

    async def get_similar_content_by_query(self, query, pages, max_results: int = 10, return_docs: bool = False) -> Any:
        context_compressor = ContextCompressor(
            documents=pages,
            embeddings=self.embeddings,
            prompt_family=self.prompt_family,
            **self.kwargs
        )
        return await context_compressor.async_get_context(
            query=query, max_results=max_results, cost_callback=self.cost_callback, return_docs=return_docs
        )


__all__ = ["ContextManager"]


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Quick test for ContextManager.get_similar_content_by_query")
    parser.add_argument("--query", type=str, default="What is context compression?", help="Query to test")
    parser.add_argument("--max-results", type=int, default=5, help="Max chunks to return")

    args = parser.parse_args()

    # Sample pages for a smoke test
    sample_pages = [
        {
            "raw_content": '''With the rising popularity of Transformer-based large language models (LLMs), reducing their high inference costs has become a significant research focus. One effective approach is to compress the long input contexts. Existing methods typically leverage the self-attention mechanism of the LLM itself for context compression. While these methods have achieved notable results, the compression process still involves quadratic time complexity, which limits their applicability.
To mitigate this limitation, we propose the In-Context Former (IC-Former). Unlike previous methods, IC-Former does not depend on the target LLMs. Instead, it leverages the cross-attention mechanism and a small number of learnable digest tokens to directly condense information from the contextual word embeddings. This approach significantly reduces inference time, which achieves linear growth in time complexity within the compression range.
Experimental results indicate that our method requires only 1/32 of the floating-point operations of the baseline during compression and improves processing speed by 68 to 112 times while achieving over 90% of the baseline performance on evaluation metrics. Overall, our model effectively reduces compression costs and makes real-time compression scenarios feasible.''',
            "title": "Context Compression",
            "url": "https://example.com/context-compression",
        },
        {
            "raw_content": '''“Context compression” is any technique that shrinks what you feed into an LLM (the “context”: prompt + retrieved docs + chat history) while trying to keep the information needed to answer well.

What it’s doing (intuitively)

You often have too much text to fit in the model’s context window (or you want to save tokens/cost/latency). Compression tries to keep the signal and drop the noise.

Common forms of context compression
	•	Summarization: Turn long text into a shorter summary (sometimes with structure like bullets, sections, key facts).
	•	Extraction / “salient spans”: Keep only the most relevant sentences/paragraphs from documents.
	•	Distillation into structured memory: Convert text into facts, triples, tables, or “notes” (e.g., {"person":..., "date":..., "claim":...}).
	•	Query-focused rewriting: Rewrite the context specifically for the current question (keeps what matters for this query).
	•	Hierarchical compression: Summarize chunks → summarize summaries (useful for very long sources).
	•	Semantic compression: Replace verbose text with shorter paraphrases while preserving meaning (harder than it sounds).
	•	Representation-level compression (advanced): Store information in embeddings / latent vectors and only regenerate text when needed (more “agent system” style).

Why it matters
	•	Fits longer histories / more docs into the limited context window.
	•	Reduces token cost and latency.
	•	Can improve quality by removing irrelevant or distracting material.
	•	But it can also lose details or introduce summary errors, which hurts factuality—so good systems often keep links/quotes to original sources.''',
            "title": "RAG Overview",
            "url": "https://example.com/rag-overview",
        },
    ]

    # Optional offline embeddings for testing without API calls
    embeddings = None


    async def _run():
        cm = ContextManager()
        compressed = await cm.get_similar_content_by_query(
            query=args.query,
            pages=sample_pages,
            max_results=args.max_results,
        )
        print("Compressed context:\n")
        print(compressed)

    asyncio.run(_run())
