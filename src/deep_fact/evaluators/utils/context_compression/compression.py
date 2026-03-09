import asyncio
import os
from typing import Any

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

from deep_fact.evaluators.utils.context_compression.costs import (
    OPENAI_EMBEDDING_MODEL,
    estimate_embedding_cost,
)
from deep_fact.evaluators.utils.context_compression.embeddings_filter import (
    EmbeddingsFilter,
)
from deep_fact.evaluators.utils.context_compression.prompt_family import PromptFamily


class SearchAPIRetriever(BaseRetriever):
    """Retriever that wraps already-fetched search pages."""

    pages: list[dict] = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [
            Document(
                page_content=page.get("raw_content", ""),
                metadata={
                    "title": page.get("title", ""),
                    "source": page.get("url", ""),
                },
            )
            for page in self.pages
        ]


class ContextCompressor:
    def __init__(
        self,
        documents,
        embeddings,
        max_results=5,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        self.max_results = max_results
        self.documents = documents
        self.kwargs = kwargs
        self.embeddings = embeddings
        self.similarity_threshold = float(os.environ.get("SIMILARITY_THRESHOLD", 0.35))
        self.prompt_family = prompt_family

    def __get_contextual_retriever(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        relevance_filter = EmbeddingsFilter(embeddings=self.embeddings,
                                            similarity_threshold=self.similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, relevance_filter]
        )
        base_retriever = SearchAPIRetriever(
            pages=self.documents
        )
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )
        return contextual_retriever

    async def async_get_context(self, query, max_results=5, cost_callback=None, return_docs: bool = False) -> Any:
        compressed_docs = self.__get_contextual_retriever()
        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))
        relevant_docs = await asyncio.to_thread(compressed_docs.invoke, query, **self.kwargs)
        if return_docs:
            return relevant_docs[:max_results]
        return self.prompt_family.pretty_print_docs(relevant_docs, max_results)
