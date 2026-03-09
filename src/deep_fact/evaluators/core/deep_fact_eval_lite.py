import asyncio
import hashlib
import pickle
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

import yaml
from filelock import FileLock
from deep_fact.evaluators.models.types import (
    DeepResearchResult,
    DeepResearchResults,
    ResearchPlan,
    SourceList,
    UserCommunication,
    FactualVerdict,
    FactualVerdictList,
    DocumentDetailAssessment,
    VerificationAspects,
    SentencePairs,
)
from pydantic import BaseModel, Field
from deep_fact.evaluators.utils.llm_client import asingle_shot_llm_call, TokenUsage
from deep_fact.evaluators.utils.logging import AgentLogger
from tenacity import retry, stop_after_attempt, wait_exponential

from collections import defaultdict
import nest_asyncio
import re
import json

logging = AgentLogger("deep_fact_eval.lite")

ClaimResult = dict[str, Any]
AssessmentPayload = dict[str, Any]
SearchBackend = Callable[[str], Awaitable[Any]]
T = TypeVar("T")


@lru_cache(maxsize=None)
def _load_prompt_assets(filename: str) -> dict[str, str]:
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / filename
    with open(prompt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_claims_context_async(*args, **kwargs):
    from deep_fact.utils.llm_tools import extract_claims_context_async

    return extract_claims_context_async(*args, **kwargs)


def _create_context_manager():
    from deep_fact.evaluators.utils.context_compression.context_manager import ContextManager

    return ContextManager()


class MultipleVerificationAspects(BaseModel):
    claims_aspects: list[VerificationAspects] = Field(
        description="List of key verification aspects that need detailed evidence for each claim")


class DeepFactEvaluatorLite:
    """Evaluate factual claims using multi-step search, evidence extraction, and verdict generation."""

    def __init__(
            self,
            budget: int = 6,
            remove_thinking_tags: bool = False,
            max_queries: int = -1,
            max_sources: int = -1,
            max_completion_tokens: int = 4096,
            user_timeout: float = 30.0,
            interactive: bool = False,
            planning_model: str = "openai/gpt-4.1-mini",
            summarization_model: str = "openai/gpt-4.1-mini",
            json_model: str = "openai/gpt-4.1-mini",
            answer_model: str = "openai/gpt-4.1-mini",
            debug_file_path: str | None = None,
            cache_dir: str | None = None,
            use_cache: bool = False,
            observer: Callable | None = None,
            max_doc_len: int = -1,
            search_tool: str = "serper",
            group_size: int = 5,
            compress_doc: bool = False,
            deep_queries_group_size: int = 10,
            verification_queries_group_size: int = 5,
            answer_generation_group_size: int = -1,
    ):
        self.budget = budget
        self.current_spending = 0
        self.remove_thinking_tags = remove_thinking_tags
        self.max_queries = max_queries
        self.max_sources = max_sources
        self.max_completion_tokens = max_completion_tokens
        self.user_timeout = user_timeout
        self.interactive = interactive
        self.planning_model = planning_model
        self.summarization_model = summarization_model
        self.json_model = json_model
        self.answer_model = answer_model
        self.debug_file_path = debug_file_path
        self.communication = UserCommunication()
        self.use_cache = use_cache
        self.max_doc_len = max_doc_len
        self.search_tool, self._search_backend = self._resolve_search_backend(search_tool)
        self.token_usage = TokenUsage(0, 0)
        self.group_size = group_size
        self.deep_queries_group_size = deep_queries_group_size
        self.verification_queries_group_size = verification_queries_group_size
        self.answer_generation_group_size = answer_generation_group_size

        self.compress_doc = compress_doc

        # this is a little hack to make the observer optional
        self.observer = observer if observer is not None else lambda *args, **kwargs: None

        if self.use_cache:
            model_name = self.summarization_model.split("/")[-1].replace(".", "-").strip()
            use_compress_suffix = "_compressed" if self.compress_doc else ""
            self.cache_dir = Path(
                cache_dir) if cache_dir else Path.home() / f".cache_evaluator_batched_cached_{model_name}_gs{self.group_size}{use_compress_suffix}"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Create a locks directory for the file locks
            self.locks_dir = self.cache_dir / ".locks"
            self.locks_dir.mkdir(parents=True, exist_ok=True)

        self.prompts = dict(_load_prompt_assets("deep_fact_eval_lite.yaml"))

    @staticmethod
    def _resolve_search_backend(search_tool: str) -> tuple[str, SearchBackend]:
        normalized = search_tool.strip().lower()
        if normalized == "serper":
            from deep_fact.evaluators.search.serper import aserper_search_results

            return normalized, aserper_search_results
        if normalized == "tavily":
            from deep_fact.evaluators.search.tavily import atavily_search_results

            return normalized, atavily_search_results
        raise ValueError(f"Unsupported search_tool: {search_tool}. Expected one of: serper, tavily")

    def _clear_cache_file(self):
        """Clear all cache files"""
        if self.use_cache:
            # Clear cache directory
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                    # logging.info(f"Deleted cache file: {cache_file}")
                except Exception as e:
                    logging.warning(f"Failed to delete cache file {cache_file}: {str(e)}")

    def __call__(self, report: str, claims: list[str], summarize_topic: str | None = None) -> list[ClaimResult]:
        """
        Evaluate multiple claims from a report. Extracts context and runs verification per claim concurrently.
        """
        try:
            running_loop = asyncio.get_running_loop()  # raises RuntimeError if none
            in_running_loop = running_loop.is_running()
        except RuntimeError:
            running_loop, in_running_loop = None, False

        if in_running_loop:
            # Jupyter’s loop rejects nested run_until_complete calls unless patched
            nest_asyncio.apply()

            # Schedule the coroutine and wait synchronously for its result
            return running_loop.run_until_complete(self.evaluate_report_claims(report, claims, summarize_topic))
        loop = asyncio.new_event_loop()
        try:
            answers = loop.run_until_complete(self.evaluate_report_claims(report, claims, summarize_topic))

            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(asyncio.wait(pending, timeout=10))

            return answers
        finally:
            loop.close()

    async def generate_research_queries(self, topic: str) -> list[str]:
        planning_prompt = self.prompts["planning_prompt"]

        plan = await asingle_shot_llm_call(
            model=self.planning_model, system_prompt=planning_prompt, message=f"Research Task: {topic}",
            token_usage=self.token_usage
        )

        logging.info(f"\n\nGenerated deep research plan for task: {topic}\n\nPlan: {plan}\n\n")

        plan_parsing_prompt = self.prompts["plan_parsing_prompt"]

        response_json = await asingle_shot_llm_call(
            model=self.json_model,
            system_prompt=plan_parsing_prompt,
            message=f"Plan to be parsed: {plan}",
            response_format=ResearchPlan,
            token_usage=self.token_usage
        )

        plan = json.loads(response_json)

        return plan["queries"]

    def _get_cache_path(self, query: str, summarize_topic: str = None) -> Path:
        """Generate a cache file path for a given query using its hash"""
        # Include summarize_topic in cache key for better cache sharing
        cache_key = query
        if summarize_topic:
            cache_key = f"{query}|{summarize_topic}"
        query_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"{self.search_tool}_{query_hash}.pkl"

    def _get_doc_cache_path(self, doc_url: str, summarize_topic: str = None) -> Path:
        """Generate a cache file path for a given document URL and summarize topic using its hash"""
        # Include summarize_topic in cache key for better cache sharing
        cache_key = doc_url
        if summarize_topic:
            cache_key = f"{doc_url}|{summarize_topic}"
        doc_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"doc_{self.search_tool}_{doc_hash}.pkl"

    def _get_lock_path(self, cache_path: Path) -> Path:
        """Generate a lock file path for a given cache file"""
        return self.locks_dir / f"{cache_path.name}.lock"

    @contextmanager
    def _cache_lock(self, query: str, summarize_topic: str = None):
        """Context manager for thread-safe cache operations"""
        cache_path = self._get_cache_path(query, summarize_topic)
        lock_path = self._get_lock_path(cache_path)
        lock = FileLock(str(lock_path))
        try:
            with lock:
                yield cache_path
        finally:
            # Clean up lock file if it's stale
            if lock_path.exists() and not lock.is_locked:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

    def _save_to_cache(self, query: str, results: DeepResearchResults, summarize_topic: str = None):
        """
        Save search results to cache in a thread-safe manner - both query and document level.

        This method performs unified caching:
        1. Saves complete query results: (query, summarize_topic) -> DeepResearchResults
        2. Saves individual document summaries: (doc_url, summarize_topic) -> summary string

        This ensures both levels of cache are populated in one operation.
        """
        if not self.use_cache:
            return

        # Save complete query results (primary cache)
        # Create a copy of results without detailed_content for caching
        results_to_cache = DeepResearchResults(results=[
            DeepResearchResult(
                title=result.title,
                link=result.link,
                content=result.content,
                raw_content=result.raw_content,
                filtered_raw_content=result.filtered_raw_content,
                detailed_content=""  # Don't cache detailed_content
            ) for result in results.results
        ])

        with self._cache_lock(query, summarize_topic) as cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump(results_to_cache, f)

        # Save individual document summaries (secondary cache)
        # Note: We only cache filtered_raw_content, not detailed_content
        for result in results.results:
            if result.link and result.filtered_raw_content:
                try:
                    doc_cache_path = self._get_doc_cache_path(result.link, summarize_topic)
                    lock_path = self._get_lock_path(doc_cache_path)
                    lock = FileLock(str(lock_path))

                    with lock:
                        with open(doc_cache_path, "wb") as f:
                            # Only cache the filtered_raw_content (summary), not detailed_content
                            pickle.dump(result.filtered_raw_content, f)
                except Exception as e:
                    logging.warning(f"Failed to save doc cache for '{result.link}': {e}")

    def _load_from_cache(self, query: str, summarize_topic: str = None) -> DeepResearchResults | None:
        """Load search results from cache if they exist in a thread-safe manner"""
        if not self.use_cache:
            return None
        try:
            with self._cache_lock(query, summarize_topic) as cache_path:
                if cache_path.exists():
                    with open(cache_path, "rb") as f:
                        return pickle.load(f)
        except Exception as e:
            logging.warning(f"Failed to load cache for query '{query}': {e}")
        return None

    def _load_doc_from_cache(self, doc_url: str, summarize_topic: str = None) -> str | None:
        """Load document summary from cache if it exists in a thread-safe manner"""
        if not self.use_cache:
            return None

        try:
            doc_cache_path = self._get_doc_cache_path(doc_url, summarize_topic)
            lock_path = self._get_lock_path(doc_cache_path)
            lock = FileLock(str(lock_path))

            with lock:
                if doc_cache_path.exists():
                    with open(doc_cache_path, "rb") as f:
                        return pickle.load(f)
        except Exception as e:
            logging.warning(f"Failed to load doc cache for '{doc_url}': {e}")
        return None

    async def _compress_content_async(self, raw_content: str, queries: list[str]) -> str:
        async def fetch_relevant_chunks(raw_text: str, verification_query: str) -> str:
            cm = _create_context_manager()
            compressed = await cm.get_similar_content_by_query(
                query=verification_query,
                pages=[{"raw_content": raw_text}],
                max_results=10,
                return_docs=True
            )
            return compressed

        compressed_contents = await asyncio.gather(
            *[fetch_relevant_chunks(raw_content, query) for query in queries]
        )
        compressed_chunks = defaultdict(str)
        for compressed in compressed_contents:
            for chunk in compressed:
                if chunk.page_content.strip():
                    compressed_chunks[chunk.metadata["chunk_id"]] = chunk.page_content

        compressed_content = ""
        sorted_chunks_ids = sorted(compressed_chunks.keys(), key=lambda x: int(x.split("-")[1]))
        for chunk_id in sorted_chunks_ids:
            chunk_id_ = chunk_id.split("-")[1]
            compressed_content += f"\n\nChunk {chunk_id_}: {compressed_chunks[chunk_id]}"
        return compressed_content

    async def _extract_detailed_content_grouped_async(
        self, doc: DeepResearchResult, verification_queries: list[str]
    ) -> str:
        if not verification_queries:
            return ""

        grouped_queries = await self.group_items(
            verification_queries,
            group_size=self.verification_queries_group_size,
            verbose=False,
        )
        verification_tasks = [self._extract_detailed_content_async(doc, group) for group in grouped_queries]

        detailed_contents = await asyncio.gather(*verification_tasks, return_exceptions=True)
        collected_contents: list[str] = []
        for detailed_content in detailed_contents:
            if isinstance(detailed_content, Exception):
                logging.warning(f"Failed to extract grouped detailed content: {detailed_content}")
                continue
            if detailed_content:
                collected_contents.append(detailed_content)

        return "\n\n".join(collected_contents)

    async def _extract_detailed_content_async(self, doc: DeepResearchResult, verification_queries: list[str]) -> str:
        """
        Extract detailed content from raw document based on verification queries.

        Args:
            doc: The document containing content to summarize.
            verification_queries: Queries used to guide extraction.

        Returns:
            Extracted detailed content
        """
        if not verification_queries:
            return ""

        content = doc.raw_content
        if self.compress_doc:
            content = await self._compress_content_async(content, verification_queries)
            if not content.strip():
                return ""

        details_summarizer_prompt = (
            self.prompts["details_summarizer_prompt"]
            if not self.compress_doc
            else self.prompts["details_chunk_summarizer_prompt"]
        )

        queries_text = "\n".join([f"- {vq}" for vq in verification_queries])
        message_content = (
            f"<Raw Content>{content}</Raw Content>\n\n"
            f"<Verification Queries>\n{queries_text}\n</Verification Queries>"
        )

        result = await asingle_shot_llm_call(
            model=self.summarization_model,
            system_prompt=details_summarizer_prompt,
            message=message_content,
            token_usage=self.token_usage
        )

        return result

    async def _enhance_with_detailed_content_for_claims(
        self,
        results: list[DeepResearchResult],
        main_claims: str | None = None,
        deep_queries: list[str] | None = None,
    ) -> list[DeepResearchResult]:
        """
        Generate important verification aspects and extract detailed content from documents.

        Args:
            results: List of DeepResearchResult objects with summaries
            main_claims: The main claims text being fact-checked

        Returns:
            Enhanced list of DeepResearchResult objects with detailed content
        """
        # Generate document-specific queries based on verification aspects.
        document_queries_map = await self._generate_document_specific_queries_for_claims(
            deep_queries, results, main_claims
        )

        # Step 3: Extract detailed content for each document
        detail_extraction_tasks = []
        task_indices = []

        for doc_idx, queries_for_doc in document_queries_map.items():
            if doc_idx < len(results):
                result = results[doc_idx]
                if result.raw_content:
                    task = self._extract_detailed_content_grouped_async(result, queries_for_doc)
                    detail_extraction_tasks.append(task)
                    task_indices.append(doc_idx)

        if detail_extraction_tasks:
            detailed_contents = await asyncio.gather(*detail_extraction_tasks, return_exceptions=True)

            # Apply detailed content to the appropriate results
            for task_idx, detailed_content in zip(task_indices, detailed_contents):
                result = results[task_idx]
                if isinstance(detailed_content, Exception):
                    logging.warning(f"Failed to extract detailed content for '{result.title}': {detailed_content}")
                    detailed_content = ""

                if detailed_content:
                    if result.detailed_content:
                        detailed_content = result.detailed_content + "\n\n" + detailed_content

                    # Create enhanced result with detailed content
                    enhanced_result = DeepResearchResult(
                        title=result.title,
                        link=result.link,
                        content=result.content,
                        raw_content=result.raw_content,
                        filtered_raw_content=result.filtered_raw_content,
                        detailed_content=detailed_content
                    )
                    results[task_idx] = enhanced_result
                    logging.info(f"Enhanced '{result.title}' with detailed content")

        return results

    async def _generate_verification_aspects_for_claims(
        self, main_claims: str, current_evidence: str | None = None
    ) -> list[str]:
        """
        Generate important aspects/sub-queries/checklist for multiple claims.

        Args:
            main_claims: The main claims text being fact-checked

        Returns:
            List of verification aspects
        """
        verification_aspects_prompt = (
            self.prompts["details_verification_prompt"]
            if not current_evidence
            else self.prompts["details_verification_with_evidence_prompt"]
        )

        message = f"<Claims>{main_claims}</Claims>" if not current_evidence else f"<Current External Information>{current_evidence}</Current External Information>\n\n<Claims>{main_claims}</Claims>"

        aspects_response = await asingle_shot_llm_call(
            model=self.planning_model,
            system_prompt=verification_aspects_prompt,
            message=message,
            response_format=MultipleVerificationAspects,
            token_usage=self.token_usage
        )
        aspects_response = json.loads(aspects_response)

        logging.info(f"Generated verification aspects for claims: {aspects_response}")

        # Extract aspects for each claim
        claims_aspects = aspects_response.get("claims_aspects", [])
        result_aspects = []

        for claim_aspect in claims_aspects:
            if isinstance(claim_aspect, dict) and "aspects" in claim_aspect:
                result_aspects.extend(claim_aspect["aspects"])
            else:
                continue

        return result_aspects

    async def _generate_document_specific_queries_for_claims(
        self,
        deep_queries: list[str] | None,
        results: list[DeepResearchResult],
        main_claims: str | None,
    ) -> dict[int, list[str]]:
        """
        Generate document-specific detail queries based on verification aspects for multiple claims.

        Args:
            deep_queries: List of aspects/queries that need detailed verification
            results: List of DeepResearchResult objects with summaries
            main_claims: The main claims text being fact-checked

        Returns:
            Dictionary mapping document index to list of specific queries for that document
        """
        if not deep_queries:
            return {}

        doc_queries_map = {}

        # Generate queries for each document
        query_generation_tasks = []
        for i, result in enumerate(results):
            task = self._generate_grouped_queries_for_document_claims(main_claims, deep_queries, result, i)
            query_generation_tasks.append(task)

        # Execute all query generation tasks in parallel
        document_assessments = await asyncio.gather(*query_generation_tasks, return_exceptions=True)

        # Process results
        for i, assessment_result in enumerate(document_assessments):
            if isinstance(assessment_result, Exception):
                logging.warning(f"Failed to generate queries for document {i}: {assessment_result}")
                continue

            try:
                assessment = DocumentDetailAssessment(**assessment_result)
                if assessment.relevant and not assessment.sufficient and assessment.queries:
                    doc_queries_map[i] = assessment.queries
                    logging.info(f"Generated {len(assessment.queries)} queries for '{results[i].title}'")
            except Exception as e:
                logging.warning(f"Failed to parse assessment for document {i}: {e}")

        return doc_queries_map

    async def _generate_grouped_queries_for_document_claims(
        self,
        main_claims: str,
        deep_queries: list[str],
        result: DeepResearchResult,
        doc_index: int,
    ) -> AssessmentPayload:
        """
        Generate grouped queries for a single document based on claims.
        """
        # group the deep queries into groups of size self.max_deep_queries_input_per_doc
        # in this way, we get better detail extraction queries for each document
        deep_queries_grouped = await self.group_items(
            deep_queries, group_size=self.deep_queries_group_size, verbose=False
        )
        query_generation_tasks = []
        
        for group in deep_queries_grouped:
            aspects_text = "\n".join([f"- {aspect}" for aspect in group])
            task = self._generate_queries_for_document_claims(main_claims, aspects_text, result, doc_index)
            query_generation_tasks.append(task)

        document_assessments = await asyncio.gather(*query_generation_tasks, return_exceptions=True)
        document_assessment = DocumentDetailAssessment(relevant=False, sufficient=True, queries=[])
        for assessment in document_assessments:
            if isinstance(assessment, Exception):
                logging.warning(f"Failed to generate queries for document {doc_index}: {assessment}")
                continue
            try:
                assessment = DocumentDetailAssessment(**assessment)
                document_assessment += assessment
            except Exception as e:
                logging.warning(f"Failed to parse assessment for document {doc_index}: {e}")

        return document_assessment.model_dump()

    async def _generate_queries_for_document_claims(
        self, main_claims: str, aspects_text: str, result: DeepResearchResult, doc_index: int
    ) -> AssessmentPayload:
        """
        Generate specific queries for a single document based on claims.

        Args:
            main_claims: The main claims text being fact-checked
            aspects_text: Formatted text of verification aspects
            result: The document result with summary
            doc_index: Index of the document

        Returns:
            Dictionary with assessment results
        """
        query_generator_prompt = self.prompts["details_query_generator_prompt"]
        current_summary = result.filtered_raw_content
        if result.detailed_content:
            current_summary += f"\n\nCurrent Detailed Content: {result.detailed_content}"

        message_content = (
            f"<Main Claims>\n{main_claims}\n</Main Claims>\n\n"
            f"<Key Verification Aspects>\n{aspects_text}\n</Key Verification Aspects>\n\n"
            f"<Document>\n"
            f"Title: {result.title}\n"
            f"URL: {result.link}\n"
            f"Current Summary: {current_summary}\n"
            f"</Document>"
        )

        response = await asingle_shot_llm_call(
            model=self.planning_model,
            system_prompt=query_generator_prompt,
            message=message_content,
            response_format=DocumentDetailAssessment,
            token_usage=self.token_usage
        )

        return json.loads(response)

    def extract_main_claims(self, claims: str) -> tuple[str, str]:
        if "<Sentence 0>" in claims:
            context = claims.split("<sentences>")[1].split("<Sentence 0>")[0].strip()
            claims = "<sentences>\n" + "<Sentence 0>" + claims.split("<Sentence 0>")[1]

        elif "<sentence 0>" in claims:
            context = claims.split("<sentences>")[1].split("<sentence 0>")[0].strip()
            claims = "<sentences>\n" + "<sentence 0>" + claims.split("<sentence 0>")[1]

        else:
            claims = claims
            context = ""
        return claims, context


    async def extract_pairs(self, text: str) -> list[tuple[int, str, str]]:
        pair_pattern = re.compile(
            r"<Sentence\s+(?P<i>\d+)>\s*(?P<sent>.*?)\s*</Sentence\s+(?P<ci>\d+)>\s*"
            r"<Rephrased\s+Sentence\s+(?P<j>\d+)>\s*(?P<reph>.*?)\s*</Rephrased\s+Sentence\s+(?P<cj>\d+)>",
            flags=re.DOTALL | re.IGNORECASE,
        )
        pairs = []
        error = False
        for m in pair_pattern.finditer(text):
            i, ci, j, cj = map(int, (m.group("i"), m.group("ci"), m.group("j"), m.group("cj")))
            # Ensure tags are consistent: <Sentence k> ... </Sentence k> and same k for rephrased
            if not (i == ci == j == cj):
                error = True
                continue  # or raise ValueError(...) if you prefer strictness

            sent = m.group("sent").strip()
            reph = m.group("reph").strip()
            pairs.append((i, sent, reph))

        pairs.sort(key=lambda x: x[0])
        if error:
            logging.warning(f"Error extracting claims from topic: {text}")
            answer = await asingle_shot_llm_call(
                model=self.answer_model,
                system_prompt=self.prompts["template_extract_claims"],
                message=text,
                max_completion_tokens=self.max_completion_tokens,
                response_format=SentencePairs,
                token_usage=self.token_usage
            )
            pairs = [(pair.sentence_idx, pair.sentence, pair.rephrased_sentence) for pair in answer.pairs]
        return pairs

    async def calculate_claims_num(self, text: str) -> int:
        claims, _ = self.extract_main_claims(text)
        pairs = await self.extract_pairs(claims)
        return len(pairs)


    async def search_all_queries(
        self,
        queries: list[str],
        main_claims: str | None = None,
        summarize_topic: str | None = None,
        existing_results: DeepResearchResults | None = None,
        deep_queries: list[str] | None = None,
    ) -> DeepResearchResults:
        """
        Execute searches for all queries in parallel, then summarize all results with two-level caching.

        Two-Level Caching Strategy:
        1. Primary Cache: (query, summarize_topic) -> Complete DeepResearchResults
           - Fastest: Direct hit returns complete processed results

        2. Secondary Cache: (doc_url, summarize_topic) -> Document summary string
           - Partial hit: Reuse document summaries across different queries
           - Different queries finding same documents can share summaries

        Cache Flow:
        1. Check primary cache for (query, summarize_topic) - if hit, return complete results
        2. If miss, perform search to get raw documents
        3. For each document, check secondary cache (doc_url, summarize_topic)
        4. If document cached, reuse summary; if not, summarize
        5. Save both query results and document summaries in unified cache operation

        This maximizes cache reuse when different queries find overlapping documents.
        """
        tasks = []
        per_query_results: list[tuple[str, Any]] = []

        # Stage 1: Check primary cache (query, summarize_topic) level
        queries_to_search = []
        for query in queries:
            cached_result = self._load_from_cache(query, summarize_topic)
            if cached_result is not None:
                logging.info(f"Using cached complete results for query: {query}")
                per_query_results.append((query, cached_result))
            else:
                queries_to_search.append(query)

        # Stage 2: For queries not in primary cache, fetch raw search results
        if queries_to_search:
            for query in queries_to_search:
                tasks.append(self._search_engine_call(query))

            fetched_raw_results = await asyncio.gather(*tasks)

            # Combine queries with their raw results
            for query, raw_results in zip(queries_to_search, fetched_raw_results):
                per_query_results.append((query, raw_results))

        # Stage 3: Process results with two-level caching for summarization
        raw_content_summarizer_prompt = self.prompts["raw_content_summarizer_prompt"]

        formatted_results: list[DeepResearchResult] = []
        summarization_tasks = []
        result_info = []  # list of tuples (SearchResult-like, query, doc_url)
        query_to_results_map = {}  # Track which results belong to which query

        for query, results_obj in per_query_results:
            # If cached object is already DeepResearchResults, reuse directly
            if isinstance(results_obj, DeepResearchResults):
                # get the cached docs
                for cached_result in results_obj.results:
                    formatted_results.append(cached_result)
                continue

            # Initialize results list for this query
            if query not in query_to_results_map:
                query_to_results_map[query] = []

            # Process SearchResults-like objects with document-level caching
            for result in getattr(results_obj, "results", []):
                raw_content = getattr(result, "raw_content", None)
                doc_url = getattr(result, "link", "")

                if raw_content is None or not doc_url:
                    continue

                # Check document-level cache first
                cached_summary = self._load_doc_from_cache(doc_url, summarize_topic)
                if cached_summary is not None:
                    logging.info(f"Using cached document summary for: {doc_url}")
                    deep_result = DeepResearchResult(
                        title=getattr(result, "title", ""),
                        link=doc_url,
                        content=getattr(result, "content", ""),
                        raw_content=raw_content,
                        filtered_raw_content=cached_summary,
                        detailed_content=""
                    )
                    formatted_results.append(deep_result)
                    query_to_results_map[query].append(deep_result)
                    continue

                # Document not cached, need to summarize
                # trim the input doc if necessary
                if self.max_doc_len > 0:
                    from litellm.utils import encode, decode
                    model_name = self.summarization_model
                    ids = encode(model=model_name, text=raw_content)
                    if len(ids) > self.max_doc_len:
                        ids = ids[:self.max_doc_len]
                        raw_content = decode(model=model_name, tokens=ids)
                        logging.warning(f"Truncating the document for {getattr(result, 'title', '')}, url: {doc_url}")

                # Create summarization query based on summarize_topic or original query
                summarization_query = summarize_topic if summarize_topic else query
                task = self._summarize_content_async(raw_content, summarization_query, raw_content_summarizer_prompt)
                summarization_tasks.append(task)
                result_info.append((result, query, doc_url))

        if summarization_tasks:
            summarized_contents = await asyncio.gather(*summarization_tasks, return_exceptions=True)
            for (result, query, doc_url), summarized_content in zip(result_info, summarized_contents):
                if isinstance(summarized_content, Exception):
                    logging.warning(f"Failed to summarize content for '{doc_url}': {summarized_content}")
                    continue
                deep_result = DeepResearchResult(
                    title=getattr(result, "title", ""),
                    link=doc_url,
                    content=getattr(result, "content", ""),
                    raw_content=getattr(result, "raw_content", None),
                    filtered_raw_content=summarized_content,
                    detailed_content=""
                )
                formatted_results.append(deep_result)

                # Track which query this result belongs to
                if query not in query_to_results_map:
                    query_to_results_map[query] = []
                query_to_results_map[query].append(deep_result)

        # Cache complete results for queries that were processed (unified caching)
        for query, results_for_query in query_to_results_map.items():
            query_results = DeepResearchResults(results=results_for_query)
            self._save_to_cache(query, query_results, summarize_topic)
            logging.info(f"Cached query and document results for: {query} ({len(results_for_query)} documents)")

        formatted_results = DeepResearchResults(results=formatted_results).dedup().results

        # Copy over existing detailed_content for matching URLs from previous iterations.
        if existing_results:
            existing_detailed_map = {
                r.link: r.detailed_content
                for r in existing_results.results
                if r.link and r.detailed_content
            }
            for result_idx, result in enumerate(formatted_results):
                if result.link in existing_detailed_map and not result.detailed_content:
                    # Create new result with existing detailed_content preserved
                    formatted_results[result_idx] = DeepResearchResult(
                        title=result.title,
                        link=result.link,
                        content=result.content,
                        raw_content=result.raw_content,
                        filtered_raw_content=result.filtered_raw_content,
                        detailed_content=existing_detailed_map[result.link]
                    )
                    logging.info(f"Reused existing detailed_content for '{result.title}'")

        # Generate verification queries based on summaries and extract detailed content.
        if main_claims and formatted_results:
            try:
                formatted_results = await self._enhance_with_detailed_content_for_claims(formatted_results, main_claims,
                                                                                         deep_queries=deep_queries)
                logging.info(f"Enhanced {len(formatted_results)} results with detailed content")
            except Exception as e:
                logging.warning(f"Failed to enhance results with detailed content: {e}")

        return DeepResearchResults(results=formatted_results)

    async def _search_engine_call(self, query: str) -> Any:
        """Perform a single search and return raw results without summarization."""

        if len(query) > 400:
            # NOTE: we are truncating the query to 400 characters to avoid Tavily Search issues
            query = query[:400]
            logging.info(f"Truncated query to 400 characters: {query}")

        response = await self._search_backend(query)

        logging.info(f"{self.search_tool} Search Called.")

        # Return raw SearchResults-like object. Summarization happens after all searches.
        return response

    async def _summarize_content_async(self, raw_content: str, query: str, prompt: str) -> str:
        """Summarize content asynchronously using the LLM"""
        logging.info("Summarizing content asynchronously using the LLM")

        result = await asingle_shot_llm_call(
            model=self.summarization_model,
            system_prompt=prompt,
            message=f"<Raw Content>{raw_content}</Raw Content>\n\n<Research Topic>{query}</Research Topic>",
            token_usage=self.token_usage
        )

        return result

    async def evaluate_research_completeness(
        self, topic: str, results: DeepResearchResults, queries: list[str]
    ) -> list[str]:
        """
        Evaluate if the current search results are sufficient or if more research is needed.
        Returns an empty list if research is complete, or a list of additional queries if more research is needed.
        """

        # Format the search results for the LLM
        formatted_results = str(results)
        evaluation_prompt = self.prompts["evaluation_prompt"]

        evaluation = await asingle_shot_llm_call(
            model=self.planning_model,
            system_prompt=evaluation_prompt,
            message=(
                f"<Research Topic>{topic}</Research Topic>\n\n"
                f"<Search Queries Used>{queries}</Search Queries Used>\n\n"
                f"<Current Search Results>{formatted_results}</Current Search Results>"
            ),
            token_usage=self.token_usage
        )

        logging.info(f"Evaluation: {evaluation}")

        evaluation_parsing_prompt = self.prompts["evaluation_parsing_prompt"]

        response_json = await asingle_shot_llm_call(
            model=self.json_model,
            system_prompt=evaluation_parsing_prompt,
            message=f"Evaluation to be parsed: {evaluation}",
            response_format=ResearchPlan,
            token_usage=self.token_usage
        )

        evaluation = json.loads(response_json)
        return evaluation["queries"]

    async def filter_results(self, topic: str, results: DeepResearchResults) -> tuple[DeepResearchResults, SourceList]:
        """Filter the search results based on the research plan"""

        # Format the search results for the LLM, without the raw content
        formatted_results = str(results)

        filter_prompt = self.prompts["filter_prompt"]

        filter_response = await asingle_shot_llm_call(
            model=self.planning_model,
            system_prompt=filter_prompt,
            message=(
                f"<Evaluation Task>{topic}</Evaluation Task>\n\n"
                f"<Current Search Results>{formatted_results}</Current Search Results>"
            ),
            # NOTE: This is the max_token parameter for the LLM call on Together AI, may need to be changed for other providers
            max_completion_tokens=4096,
            token_usage=self.token_usage
        )

        logging.info(f"Filter response: {filter_response}")

        filter_parsing_prompt = self.prompts["filter_parsing_prompt"]

        response_json = await asingle_shot_llm_call(
            model=self.json_model,
            system_prompt=filter_parsing_prompt,
            message=f"Filter response to be parsed: {filter_response}",
            response_format=SourceList,
            token_usage=self.token_usage
        )

        sources = json.loads(response_json)["sources"]

        logging.info(f"Filtered sources: {sources}")

        if self.max_sources != -1:
            sources = sources[: self.max_sources]

        # Filter the results based on the source list
        filtered_results = [results.results[i - 1] for i in sources if i - 1 < len(results.results)]

        return DeepResearchResults(results=filtered_results), sources

    async def generate_research_answer(self, topic: str, results: DeepResearchResults,
                                       remove_thinking_tags: bool = False):
        """
        Generate a comprehensive answer to the research topic based on the search results.
        Returns a detailed response that synthesizes information from all search results.
        """

        formatted_results = str(results)
        answer_prompt = self.prompts["answer_prompt"]

        answer = await asingle_shot_llm_call(
            model=self.answer_model,
            system_prompt=answer_prompt,
            message=f"Research Topic: {topic}\n\nSearch Results:\n{formatted_results}",
            # NOTE: This is the max_token parameter for the LLM call on Together AI, may need to be changed for other providers
            max_completion_tokens=self.max_completion_tokens,
            response_format=FactualVerdict,
            token_usage=self.token_usage
        )

        if remove_thinking_tags:
            answer = self._remove_thinking_tags(answer)

        return answer

    def _remove_thinking_tags(self, answer: str) -> str:
        """Remove content within <think> tags"""
        while "<think>" in answer and "</think>" in answer:
            start = answer.find("<think>")
            end = answer.find("</think>") + len("</think>")
            answer = answer[:start] + answer[end:]
        return answer

    async def run_pipeline_with_claims(
        self, clarified_topic: str, summarize_topic: str | None = None
    ) -> DeepResearchResults:
        """Run planning, searching, evaluation, and filtering with claims for detailed content; return filtered results for a given topic."""
        # Step 1: Generate initial queries
        main_claims, _ = self.extract_main_claims(clarified_topic)
        summarize_topic = clarified_topic if not summarize_topic else summarize_topic

        self.observer(0.15, "Generating research queries")
        queries = await self.generate_research_queries(clarified_topic)
        deep_queries = await self._generate_verification_aspects_for_claims(main_claims)
        queries = queries[: self.max_queries - 1]
        all_queries = queries.copy()
        logging.info(f"Initial queries: {queries}")
        self.observer(0.2, "Research queries generated")

        if len(queries) == 0:
            logging.error("No initial queries generated")
            return DeepResearchResults(results=[])

        # Step 2: Perform initial search with claims for detailed content
        self.observer(0.25, "Performing initial search")
        results = await self.search_all_queries(queries, main_claims, summarize_topic, deep_queries=deep_queries)
        logging.info(f"Initial search complete, found {len(results.results)} results")
        self.observer(0.3, "Initial search complete")

        # Step 3: Conduct iterative research within budget
        total_iterations = self.budget - self.current_spending
        for iteration in range(self.current_spending, self.budget):
            current_iteration = iteration - self.current_spending + 1
            progress = 0.3 + (0.4 * (current_iteration / total_iterations))
            self.observer(progress, f"Conducting research iteration {current_iteration}/{total_iterations}")
            logging.info(f"Conducting research iteration {current_iteration}/{total_iterations}")

            # Evaluate if more research is needed
            additional_queries = await self.evaluate_research_completeness(clarified_topic, results, all_queries)
            additional_deep_queries = await self._generate_verification_aspects_for_claims(main_claims,
                                                                                           current_evidence=str(
                                                                                               results))

            # Filter out empty strings and check if any queries remain
            additional_queries = [q for q in additional_queries if q]
            if not additional_queries:
                logging.info("No need for additional research")
                self.observer(progress + 0.05, "Research complete - no additional queries needed")
                break

            additional_queries = additional_queries[: self.max_queries]
            logging.info(f"Additional queries: {additional_queries}")

            # Expand research with new queries (with claims for detailed content)
            # Pass existing results so we can reuse detailed_content for matching URLs
            self.observer(progress + 0.02, f"Searching {len(additional_queries)} additional queries")
            new_results = await self.search_all_queries(additional_queries, main_claims, summarize_topic,
                                                        existing_results=results, deep_queries=additional_deep_queries)
            logging.info(f"Follow-up search complete, found {len(new_results.results)} results")
            self.observer(progress + 0.05, f"Found {len(new_results.results)} additional results")
            results = results + new_results
            results = results.dedup()

            all_queries.extend(additional_queries)

        # Step 4: Filter and process results
        self.observer(0.7, "Filtering and processing results")
        logging.info(f"Filtering results for topic: {clarified_topic[:500]}")
        results = results.dedup()
        logging.info(f"Deduplication complete, kept {len(results.results)} results")
        filtered_results, _ = await self.filter_results(clarified_topic, results)
        logging.info(f"LLM Filtering complete, kept {len(filtered_results.results)} results")

        return filtered_results

    async def generate_grouped_research_answer(
        self, topic: str, results: DeepResearchResults, max_group_size: int = 5
    ) -> dict[str, list[ClaimResult]]:
        """Generate per-claim verdicts for a group using a single LLM call."""
        # extract claims from the topic
        main_claims, context = self.extract_main_claims(topic)
        # extract claims by regular expression
        claim_pairs = await self.extract_pairs(main_claims)
        if max_group_size > 0:
            grouped_claims_pairs = await self.group_items(claim_pairs, max_group_size)
            tasks = []
            for group in grouped_claims_pairs:
                claims_str = "\n".join(
                    [
                        f"<Sentence {c[0]}> {c[1]} </Sentence {c[0]}>\n"
                        f"<Rephrased Sentence {c[0]}> {c[2]} </Rephrased Sentence {c[0]}>"
                        for c in group
                    ]
                )
                claims_with_context = f"{context}\n{claims_str}"
                new_topic = self.prompts["template_evaluate_group"].format(sentences=claims_with_context)
                tasks.append(self._generate_grouped_research_answer(new_topic, results))

            answers = await asyncio.gather(*tasks, return_exceptions=True)
            final_answers = {"results": []}
            for answer in answers:
                if isinstance(answer, Exception):
                    logging.warning(f"Failed grouped answer generation for topic chunk: {answer}")
                    continue
                final_answers["results"].extend(answer.get("results", []))
            if len(claim_pairs) != len(final_answers["results"]):
                raise ValueError(
                    f"Number of claims and answers do not match {len(claim_pairs)} vs {len(final_answers['results'])}"
                )

            return final_answers
        final_answer = await self._generate_grouped_research_answer(topic, results)
        if len(claim_pairs) != len(final_answer.get("results", [])):
            logging.warning(
                f"Number of claims and answers do not match {len(claim_pairs)} vs {len(final_answer['results'])}, retrying with group size 1"
            )
            return await self.generate_grouped_research_answer(topic, results, max_group_size=1)
        return final_answer

    # retry
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    async def _generate_grouped_research_answer(
        self, topic: str, results: DeepResearchResults
    ) -> dict[str, list[ClaimResult]]:
        """Generate per-claim verdicts for a group using a single LLM call."""
        formatted_results = str(results)

        grouped_answer_prompt = self.prompts.get("grouped_answer_prompt")

        answer = await asingle_shot_llm_call(
            model=self.answer_model,
            system_prompt=grouped_answer_prompt,
            message=f"Research Topic: {topic}\n\nSearch Results:\n{formatted_results}",
            max_completion_tokens=self.max_completion_tokens,
            response_format=FactualVerdictList,
            token_usage=self.token_usage
        )
        return json.loads(answer)

    async def group_items(self, claims: list[T], group_size: int | None = None, verbose: bool = True) -> list[list[T]]:
        """Group claims by contiguous indices using a tunable group size (0-based)."""
        size = self.group_size if group_size is None else group_size
        if size <= 0:
            return [claims]
        num_full, left_over = len(claims) // size, len(claims) % size

        # if the left over is less than the half of the group size, group the remaining claims with the previous group
        if left_over < size / 2 and num_full > 0:
            groups = [claims[i: i + size] for i in range(0, (num_full - 1) * size, size)] + [
                claims[(num_full - 1) * size:]]
            if verbose:
                logging.info(
                    f"Grouping remaining claims with the previous group, left over: {left_over}, Total groups: {len(groups)}")
            return groups

        num_group = num_full if left_over == 0 else num_full + 1
        if verbose:
            logging.info(f"Grouping claims into {num_group} groups of size {size}")
        return [claims[i:i + size] for i in range(0, len(claims), size)]

    async def extract_context(self, report: str, claim: str) -> str:
        """Extract the most relevant context from the report for a given claim."""
        text_input = f"Deep research report: {report}\n\nClaim: {claim}"
        context = await asingle_shot_llm_call(
            model=self.summarization_model,
            system_prompt=self.prompts["extract_prompt"],
            message=text_input,
            token_usage=self.token_usage
        )
        return context

    def _build_claims_with_optional_context(self, claims: list[str], context: str | None = None) -> str:
        """Build sentence payload for grouped claim evaluation with optional shared context."""
        sentence_blocks = "\n".join(
            [
                f"<Sentence {idx}> {claim} </Sentence {idx}>\n"
                f"<Rephrased Sentence {idx}> {claim} </Rephrased Sentence {idx}>"
                for idx, claim in enumerate(claims)
            ]
        )
        if context:
            return f"<Context> {context} </Context>\n{sentence_blocks}"
        return sentence_blocks

    async def _process_claim_group_with_context(
        self, context: str | None, claims_in_group: list[str]
    ) -> list[ClaimResult]:
        """Process a claim group using provided optional context (no context extraction)."""
        sentences_text = self._build_claims_with_optional_context(claims_in_group, context=context)
        group_topic = self.prompts["template_evaluate_group"].format(sentences=sentences_text)

        filtered_results = await self.run_pipeline_with_claims(group_topic)

        max_answer_group_size = (
            min(len(claims_in_group), self.answer_generation_group_size)
            if self.answer_generation_group_size > -1
            else -1
        )
        answer = await self.generate_grouped_research_answer(
            group_topic, filtered_results, max_group_size=max_answer_group_size
        )

        items = answer.get("results", [])
        final_group: list[ClaimResult] = []
        output_context = context if context is not None else ""
        for c, it in zip(claims_in_group, items):
            it.setdefault("claim", c)
            it.setdefault("context", output_context)
            final_group.append(it)
        return final_group

    async def _evaluate_claims_async(
        self, context: str | None, claims: list[str], concurrency: int = 20
    ) -> list[ClaimResult]:
        """Evaluate claims with/without shared context."""
        grouped_claim_texts = await self.group_items(claims, group_size=self.group_size)
        semaphore = asyncio.Semaphore(concurrency)

        async def process_claim_group(claim_group: list[str]) -> list[ClaimResult]:
            async with semaphore:
                return await self._process_claim_group_with_context(context, claim_group)

        group_tasks = [process_claim_group(claim_group) for claim_group in grouped_claim_texts]
        group_results = await asyncio.gather(*group_tasks, return_exceptions=True)

        final_results: list[ClaimResult] = []
        for claim_group, res in zip(grouped_claim_texts, group_results):
            if isinstance(res, Exception):
                for c in claim_group:
                    final_results.append({"claim": c, "error": str(res)})
                    logging.error(f"Error evaluating claim group: {str(res)}")
            else:
                final_results.extend(res)
        return final_results

    def evaluate_claims(
        self, context: str | None, claims: list[str], max_workers: int = 20
    ) -> list[ClaimResult]:
        """Sync entrypoint for evaluating claims with/without shared context."""
        try:
            running_loop = asyncio.get_running_loop()  # raises RuntimeError if none
            in_running_loop = running_loop.is_running()
        except RuntimeError:
            running_loop, in_running_loop = None, False

        if in_running_loop:
            nest_asyncio.apply()
            return running_loop.run_until_complete(
                self._evaluate_claims_async(context, claims, concurrency=max_workers)
            )

        loop = asyncio.new_event_loop()
        try:
            answers = loop.run_until_complete(
                self._evaluate_claims_async(context, claims, concurrency=max_workers)
            )

            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(asyncio.wait(pending, timeout=10))

            return answers
        finally:
            loop.close()

    async def _process_claim_group(
        self, report: str, claims_in_group: list[str], summarize_topic: str | None = None
    ) -> list[ClaimResult]:
        """Process a group of claims with a single research pipeline and return per-claim results."""
        claims_with_context = await _extract_claims_context_async(
            report, claims_in_group, model_id=self.answer_model
        )

        sentences_text = f"{str(claims_with_context)}"
        group_topic = self.prompts["template_evaluate_group"].format(sentences=sentences_text)

        # 3) Run the search pipeline ONCE for the whole group
        filtered_results = await self.run_pipeline_with_claims(group_topic, summarize_topic)

        # 4) Generate grouped answer structured output
        max_answer_group_size = (
            min(len(claims_in_group), self.answer_generation_group_size)
            if self.answer_generation_group_size > -1
            else -1
        )
        answer = await self.generate_grouped_research_answer(
            group_topic, filtered_results, max_group_size=max_answer_group_size
        )

        items = answer.get("results", [])
        final_group: list[ClaimResult] = []
        for c, it in zip(claims_in_group, items):
            it.setdefault("claim", c)
            it.setdefault("context", claims_with_context)
            final_group.append(it)
        return final_group

    async def evaluate_report_claims(
        self, report: str, claims: list[str], summarize_topic: str | None = None, concurrency: int = 20
    ) -> list[ClaimResult]:
        """Group claims and evaluate each group with a single research pass; flatten per-claim results."""

        # Group claims by contiguous indices with configured size (0-based)
        grouped_claim_texts = await self.group_items(claims, group_size=self.group_size)

        # set asyncio concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def process_claim_group(claim_group: list[str]) -> list[ClaimResult]:
            async with semaphore:
                return await self._process_claim_group(report, claim_group, summarize_topic=summarize_topic)

        group_tasks = [process_claim_group(claim_group) for claim_group in grouped_claim_texts]
        group_results = await asyncio.gather(*group_tasks, return_exceptions=True)

        # Flatten, aligning errors if a group failed
        final_results: list[ClaimResult] = []
        for claim_group, res in zip(grouped_claim_texts, group_results):
            if isinstance(res, Exception):
                for c in claim_group:
                    final_results.append({"claim": c, "error": str(res)})
            else:
                final_results.extend(res)
        return final_results

    def evaluate_report(
        self,
        report_data: str | dict[str, Any],
        claims: list[str],
        max_workers: int = 20,
        clear_cache: bool = True,
    ) -> list[ClaimResult]:
        if clear_cache:
            self._clear_cache_file()

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if isinstance(report_data, str):
            report, thesis = report_data, ""
        else:
            report, thesis = report_data["response"], report_data.get("thesis", "")
        summarize_topic = "You need to collect information that can help fact-check a report with summary: " + thesis
        return loop.run_until_complete(
            self.evaluate_report_claims(
                report, claims, concurrency=max_workers, summarize_topic=summarize_topic
            )
        )
