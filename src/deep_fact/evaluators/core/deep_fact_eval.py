import asyncio
import hashlib
import json
import os
import pickle
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Any

import yaml
from dotenv import load_dotenv
from filelock import FileLock
from deep_fact.evaluators.models.types import (
    DeepResearchResult,
    DeepResearchResults,
    ResearchPlan,
    SourceList,
    UserCommunication,
    FactualVerdict,
    DocumentDetailAssessment,
    VerificationAspects,
)
from deep_fact.evaluators.search.serper import aserper_search_results
from deep_fact.evaluators.utils.llm_client import asingle_shot_llm_call, TokenUsage
from deep_fact.evaluators.utils.logging import AgentLogger
from deep_fact.evaluators.search.tavily import atavily_search_results
import nest_asyncio

from concurrent.futures import ThreadPoolExecutor, as_completed

logging = AgentLogger("deep_fact_eval")

TIME_LIMIT_MULTIPLIER = 5


@lru_cache(maxsize=None)
def _load_prompt_assets(filename: str) -> dict[str, str]:
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / filename
    with open(prompt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_llm_tools():
    from deep_fact.utils import llm_tools

    return llm_tools

class DeepFactEvaluator:
    def __init__(
        self,
        budget: int = 6,
        remove_thinking_tags: bool = False,
        max_queries: int = -1,
        max_sources: int = -1,
        max_completion_tokens: int = 4096,
        user_timeout: float = 30.0,
        interactive: bool = False,
        planning_model: str = "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        summarization_model: str = "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        json_model: str = "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        answer_model: str = "together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        debug_file_path: str | None = None,
        cache_dir: str | None = None,
        use_cache: bool = False,
        observer: Callable | None = None,
        max_doc_len: int = -1,
        search_tool: str = "serper"
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
        self.search_tool = search_tool
        self.token_usage = TokenUsage(0,0)
        self.max_detail_queries_per_doc = 5  # limit number of detail queries per document to avoid excessive processing

        # this is a little hack to make the observer optional
        self.observer = observer if observer is not None else lambda *args, **kwargs: None

        if self.use_cache:
            model_name = self.summarization_model.split("/")[-1].replace(".", "-").strip()
            self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / f".cache_tg_advanced_cached_evaluator_{model_name}"
            logging.info(f"Using cache directory: {self.cache_dir}")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Create a locks directory for the file locks
            self.locks_dir = self.cache_dir / ".locks"
            self.locks_dir.mkdir(parents=True, exist_ok=True)

        self.prompts = dict(_load_prompt_assets("deep_fact_eval.yaml"))

    def __call__(self, report_data: dict = None, claims: list = None, topic: str = None) -> str:
        """
        Makes the DeepResearcher instance callable.
        Runs research on the given report_data and claims, or a single topic.

        Args:
            report_data: Dictionary containing report information with 'response' and 'thesis' keys, or None
            claims: List of claims to be verified, or None
            topic: Single research topic or question (for backward compatibility)

        Returns:
            The research answer as a string or list of results
        """
        try:
            running_loop = asyncio.get_running_loop()  # raises RuntimeError if none
            in_running_loop = running_loop.is_running()
        except RuntimeError:
            running_loop, in_running_loop = None, False

        if in_running_loop:
            # Jupyter's loop rejects nested run_until_complete calls unless patched
            nest_asyncio.apply()

            # Schedule the coroutine and wait synchronously for its result
            if claims is not None:
                return self.evaluate_report(report_data=report_data, claims=claims)
            else:
                return running_loop.run_until_complete(self.research_topic(topic=topic))
        loop = asyncio.new_event_loop()
        try:
            if claims is not None:
                answer = self.evaluate_report(report_data=report_data, claims=claims)
            else:
                answer = loop.run_until_complete(self.research_topic(topic=topic))

            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(asyncio.wait(pending, timeout=10))

            return answer
        finally:
            loop.close()

    async def research_topic(self, topic: str, summarize_topic: str = None) -> str:
        """Main method to conduct research on a topic"""

        self.observer(0, "Starting research")

        # Use topic as summarize_topic if not provided
        if summarize_topic is None:
            summarize_topic = topic

        clarified_topic = topic
        main_claim = "<Context>" + topic.split("<Context>")[-1]

        logging.info(f"Topic: {clarified_topic}")

        # Step 1: Generate initial queries
        self.observer(0.15, "Generating research queries")
        queries = await self.generate_research_queries(clarified_topic)


        # queries = [clarified_topic] + queries[: self.max_queries - 1]
        queries = queries[: self.max_queries - 1]
        all_queries = queries.copy()
        logging.info(f"Initial queries: {queries}")
        self.observer(0.2, "Research queries generated")

        if len(queries) == 0:
            logging.error("No initial queries generated")
            return "No initial queries generated"

        # Step 2: Perform initial search
        self.observer(0.25, "Performing initial search")
        results = await self.search_all_queries(queries, main_claim, summarize_topic)
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

            # Filter out empty strings and check if any queries remain
            additional_queries = [q for q in additional_queries if q]
            if not additional_queries:
                logging.info("No need for additional research")
                self.observer(progress + 0.05, "Research complete - no additional queries needed")
                break

            # for debugging purposes we limit the number of queries
            additional_queries = additional_queries[: self.max_queries]
            logging.info(f"Additional queries: {additional_queries}")

            # Expand research with new queries
            self.observer(progress + 0.02, f"Searching {len(additional_queries)} additional queries")
            new_results = await self.search_all_queries(additional_queries, main_claim, summarize_topic)
            logging.info(f"Follow-up search complete, found {len(new_results.results)} results")
            self.observer(progress + 0.05, f"Found {len(new_results.results)} additional results")

            results = results + new_results
            all_queries.extend(additional_queries)

        # Step 4: Generate final answer with feedback loop
        self.observer(0.7, "Filtering and processing results")
        logging.info(f"Generating final answer for topic: {clarified_topic}")
        results = results.dedup()
        logging.info(f"Deduplication complete, kept {len(results.results)} results")
        filtered_results, sources = await self.filter_results(clarified_topic, results)
        logging.info(f"LLM Filtering complete, kept {len(filtered_results.results)} results")
        self.observer(0.8, f"Results filtered: kept {len(filtered_results.results)} sources")

        if self.debug_file_path:
            with open(self.debug_file_path, "w") as f:
                f.write(f"{results}\n\n\n\n{filtered_results}")
                logging.info(f"Debug file (web search results and sources) saved to {self.debug_file_path}")

        # Generate final answer
        self.observer(0.9, "Generating final research report")
        while True:
            answer = await self.generate_research_answer(clarified_topic, filtered_results, self.remove_thinking_tags)

            if not self.interactive or self.current_spending >= self.budget:
                self.observer(0.95, "Research complete")
                return answer

            logging.info(f"Answer: {answer}")
            user_feedback = await self.communication.get_input_with_timeout(
                "\nAre you satisfied with this answer? (yes/no) If no, please provide feedback: ",
                self.user_timeout * TIME_LIMIT_MULTIPLIER,
            )

            if user_feedback.lower() == "yes" or not user_feedback or user_feedback == "":
                return answer

            # Regenerate answer with user feedback
            clarified_topic = f"{clarified_topic}\n\nReport:{answer}\n\nAdditional Feedback: {user_feedback}"
            logging.info(f"Regenerating answer with feedback: {user_feedback}")
            self.current_spending += 1

    def _clear_cache_file(self):
        if self.use_cache:
            # Clear cache directory
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete cache file {cache_file}: {str(e)}")

    async def evaluate_claim_with_context(self, context: str, claim: str) -> dict:
        if context is None:
            return await self.evaluate_claim_without_context(claim=claim)
        llm_tools = _load_llm_tools()
        topic = llm_tools.factuality_evaluation_prompt.format(context=context, claim=claim)

        try:
            result = await self.research_topic(topic=topic)
            return json.loads(result)
        except Exception as e:
            logging.error(f"Failed to evaluate claim with context: {str(e)}")
            return {
                "verdict": "",
                "rationale": "",
                "error": str(e),
            }
    async def evaluate_claim_without_context(self, claim: str) -> dict:
        llm_tools = _load_llm_tools()
        topic = llm_tools.factuality_evaluation_no_context_prompt.format(claim=claim)

        try:
            result = await self.research_topic(topic=topic)
            return json.loads(result)
        except Exception as e:
            logging.error(f"Failed to evaluate claim without context: {str(e)}")
            return {
                "verdict": "",
                "rationale": "",
                "error": str(e),
            }


    def evaluate_report(self, report_data: dict, claims: list, max_workers: int = 16, clear_cache: bool = False) -> list:
        """
        Evaluate multiple claims within the context of a report using multi-threading.
        
        Args:
            report_data: Dictionary containing report information with 'response' and 'thesis' keys, or None
            claims: List of claims to be verified
            max_workers: Maximum number of worker threads for parallel evaluation
            
        Returns:
            List of evaluation results for each claim
        """


        if clear_cache and self.use_cache:
            self._clear_cache_file()
        
        # Extract report information
        report_text = report_data.get("response") if report_data else None
        report_thesis = report_data.get("thesis") if report_data else None
        
        logging.info(f"Evaluating {len(claims)} claims with report context: {bool(report_text)} (max_workers: {max_workers})")
        
        def evaluate_single_claim(claim_data):
            """Worker function to evaluate a single claim using the current evaluator instance"""
            idx, claim, full_context = claim_data
            llm_tools = _load_llm_tools()

            # Extract context for the claim if report is available
            if full_context:
                # Use a default model for context extraction - this could be made configurable
                model_name = self.planning_model

                context = llm_tools.extract_claim_context(
                    report=full_context,
                    sentence=claim,
                    model_id=model_name,
                    token_usage=self.token_usage,
                )
                topic = llm_tools.factuality_evaluation_prompt.format(context=context, claim=claim)
            else:
                # No report context, evaluate claim directly
                topic = f"Verify the factual accuracy of this claim: {claim}"
                context = None

            # Use report thesis as summarize_topic for better cache sharing
            summarize_topic = report_thesis

            # Use the current evaluator instance to evaluate the claim
            # We need to call research_topic directly since __call__ might interfere with threading
            import asyncio

            # Get or create event loop for this thread
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Call research_topic asynchronously
            evaluation_result = loop.run_until_complete(
                self.research_topic(topic=topic, summarize_topic=summarize_topic)
            )

            # Note: Token usage will be accumulated in the main instance
            result = {
                "idx": idx,
                "claim": claim,
                "context": context,
                "evaluation": evaluation_result,
            }

            logging.info(f"Completed evaluation for claim {idx+1}/{len(claims)}")
            return result
                

        
        # Execute claims evaluation in parallel
        results_by_idx = {}
        claim_full_context = report_text
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all claims for processing
            futures = {executor.submit(evaluate_single_claim, (i, claim,claim_full_context)): i for i, claim in enumerate(claims)}
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results_by_idx[result["idx"]] = result
                except Exception as e:
                    idx = futures[future]
                    logging.error(f"Thread execution failed for claim {idx+1}: {str(e)}")
                    results_by_idx[idx] = {
                        "idx": idx,
                        "claim": claims[idx],
                        "context": None,
                        "evaluation": None,
                        "error": str(e),
                    }
        
        # Order results by original claim order
        results = [results_by_idx[i] for i in range(len(claims)) if i in results_by_idx]
        
        logging.info(f"Completed evaluation of {len(results)}/{len(claims)} claims")
        return results

    async def generate_research_queries(self, topic: str) -> list[str]:
        PLANNING_PROMPT = self.prompts["planning_prompt"]

        plan= await asingle_shot_llm_call(
            model=self.planning_model, system_prompt=PLANNING_PROMPT, message=f"Research Task: {topic}", token_usage=self.token_usage
        )


        logging.info(f"\n\nGenerated deep research plan for task: {topic}\n\nPlan: {plan}\n\n")

        SEARCH_PROMPT = self.prompts["plan_parsing_prompt"]

        response_json = await asingle_shot_llm_call(
            model=self.json_model,
            system_prompt=SEARCH_PROMPT,
            message=f"Plan to be parsed: {plan}",
            # response_format={"type": "json_object", "schema": ResearchPlan.model_json_schema()},
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
            if hasattr(result, 'link') and hasattr(result, 'filtered_raw_content') and result.link and result.filtered_raw_content:
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

    async def search_all_queries(self, queries: List[str], main_claim: str = None, summarize_topic: str = None) -> DeepResearchResults:
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
        RAW_CONTENT_SUMMARIZER_PROMPT = self.prompts["raw_content_summarizer_prompt"]

        formatted_results: list[DeepResearchResult] = []
        summarization_tasks = []
        result_info = []  # list of tuples (SearchResult-like, query, doc_url)
        query_to_results_map = {}  # Track which results belong to which query

        for query, results_obj in per_query_results:
            # If cached object is already DeepResearchResults, reuse directly
            if isinstance(results_obj, DeepResearchResults):
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
                
                task = self._summarize_content_async(raw_content, summarization_query, RAW_CONTENT_SUMMARIZER_PROMPT)
                summarization_tasks.append(task)
                result_info.append((result, query, doc_url))

        if summarization_tasks:
            summarized_contents = await asyncio.gather(*summarization_tasks, return_exceptions=True)
            summarized_contents = [res for res in summarized_contents if not isinstance(res, Exception)]

            # Process summarized contents
            for (result, query, doc_url), summarized_content in zip(result_info, summarized_contents):
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


        # Stage 3: Generate verification queries based on summaries and extract detailed content
        if main_claim and formatted_results:
            try:
                formatted_results = await self._enhance_with_detailed_content(formatted_results, main_claim)
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

        if self.search_tool == "tavily":
            response = await atavily_search_results(query)
        else:
            response = await aserper_search_results(query)

        # logging.info("Tavily Search Called.")
        logging.info(f"{self.search_tool} Search Called.")

        # Return raw SearchResults-like object. Summarization happens after all searches.
        return response

    async def _summarize_content_async(self, raw_content: str, query: str, prompt: str) -> str:
        """Summarize content asynchronously using the LLM"""
        logging.info("Summarizing content asynchronously using the LLM")

        message_content = f"<Raw Content>{raw_content}</Raw Content>\n\n<Research Topic>{query}</Research Topic>"
        try:
            result = await asingle_shot_llm_call(
                model=self.summarization_model,
                system_prompt=prompt,
                message=message_content,
                token_usage=self.token_usage
            )
        except Exception as e:
            logging.warning(f"Summarization failed: {e}")
            raise e

        return result

    async def _enhance_with_detailed_content(self, results: List[DeepResearchResult], main_claim: str) -> List[DeepResearchResult]:
        """
        Generate important verification aspects and extract detailed content from documents.
        
        Args:
            results: List of DeepResearchResult objects with summaries
            main_claim: The main claim being fact-checked
            
        Returns:
            Enhanced list of DeepResearchResult objects with detailed content
        """
        # Step 1: Generate important aspects/sub-queries/checklist for verification
        verification_aspects = await self._generate_verification_aspects(main_claim)
        
        # Step 2: Generate document-specific detail queries based on aspects
        document_queries_map = await self._generate_document_specific_queries(verification_aspects, results, main_claim)
        
        # Step 3: Extract detailed content for each document
        detail_extraction_tasks = []
        task_indices = []
        
        for i, result in enumerate(results):
            if result.raw_content and i in document_queries_map and document_queries_map[i]:
                queries_for_doc = document_queries_map[i]
                task = self._extract_detailed_content_async(result.raw_content, queries_for_doc)
                detail_extraction_tasks.append(task)
                task_indices.append(i)
        
        # Execute all detail extraction tasks in parallel
        enhanced_results = list(results)  # Start with original results
        
        if detail_extraction_tasks:
            detailed_contents = await asyncio.gather(*detail_extraction_tasks, return_exceptions=True)
            
            # Apply detailed content to the appropriate results
            for task_idx, detailed_content in zip(task_indices, detailed_contents):
                result = results[task_idx]
                if isinstance(detailed_content, Exception):
                    logging.warning(f"Failed to extract detailed content for '{result.title}': {detailed_content}")
                    continue
                
                try:
                    if result.detailed_content:
                        detailed_content = result.detailed_content + "\n\n" + detailed_content
                    enhanced_result = DeepResearchResult(
                        title=result.title,
                        link=result.link,
                        content=result.content,
                        raw_content=result.raw_content,
                        filtered_raw_content=result.filtered_raw_content,
                        detailed_content=detailed_content
                    )
                    enhanced_results[task_idx] = enhanced_result
                    logging.info(f"Enhanced '{result.title}' with detailed content")
                except Exception as e:
                    logging.warning(f"Failed to create enhanced result for '{result.title}': {e}")
        
        return enhanced_results

    async def _generate_verification_aspects(self, main_claim: str) -> List[str]:
        """
        Generate important aspects/sub-queries/checklist that need special attention to details.
        
        Args:
            results: List of DeepResearchResult objects with summaries
            main_claim: The main claim being fact-checked
            
        Returns:
            List of important verification aspects
        """
        VERIFICATION_ASPECTS_PROMPT = self.prompts["details_verification_prompt"]

        aspects_response = await asingle_shot_llm_call(
            model=self.planning_model,
            system_prompt=VERIFICATION_ASPECTS_PROMPT,
            message=f"<Main Claim>{main_claim}</Main Claim>",
            response_format=VerificationAspects,
            token_usage=self.token_usage
        )
        
        logging.info(f"Generated verification aspects: {aspects_response}")

        aspects_response = json.loads(aspects_response)
        aspects = aspects_response["aspects"]
        
        return aspects

    async def _generate_document_specific_queries(self, verification_aspects: List[str], results: List[DeepResearchResult], main_claim: str) -> dict:
        """
        Generate document-specific detail queries based on verification aspects.
        
        Args:
            verification_aspects: List of important aspects that need detailed verification
            results: List of DeepResearchResult objects with summaries
            main_claim: The main claim being fact-checked
            
        Returns:
            Dictionary mapping document index to list of specific queries for that document
        """
        if not verification_aspects:
            return {}
        
        doc_queries_map = {}
        aspects_text = "\n".join([f"- {aspect}" for aspect in verification_aspects])
        
        # Generate queries for each document
        query_generation_tasks = []
        for i, result in enumerate(results):
            task = self._generate_queries_for_document(main_claim, aspects_text, result, i)
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
                    queries = assessment.queries[:self.max_detail_queries_per_doc]
                    doc_queries_map[i] = queries
                    logging.info(f"Generated {len(queries)} queries for '{results[i].title}'")
            except Exception as e:
                logging.warning(f"Failed to parse assessment for document {i}: {e}")
        
        return doc_queries_map

    async def _generate_queries_for_document(self, main_claim: str, aspects_text: str, result: DeepResearchResult, doc_index: int) -> dict:
        """
        Generate specific queries for a single document.
        
        Args:
            main_claim: The main claim being fact-checked
            aspects_text: Formatted text of verification aspects
            result: The document result with summary
            doc_index: Index of the document
            
        Returns:
            Dictionary with assessment results
        """
        QUERY_GENERATOR_PROMPT = self.prompts["details_query_generator_prompt"]
        current_summary = result.filtered_raw_content
        if result.detailed_content:
            current_summary += f"\n\nCurrent Detailed Content: {result.detailed_content}"
        
        message_content = (
            f"<Main Claim>{main_claim}</Main Claim>\n\n"
            f"<Key Verification Aspects>\n{aspects_text}\n</Key Verification Aspects>\n\n"
            f"<Document>\n"
            f"Title: {result.title}\n"
            f"URL: {result.link}\n"
            f"Current Summary: {current_summary}\n"
            f"</Document>"
        )
        
        response = await asingle_shot_llm_call(
            model=self.planning_model,
            system_prompt=QUERY_GENERATOR_PROMPT,
            message=message_content,
            response_format=DocumentDetailAssessment,
            token_usage=self.token_usage
        )
        
        return json.loads(response)

    async def _extract_detailed_content_async(self, raw_content: str, verification_queries: List[str]) -> str:
        """
        Extract detailed content from raw document based on verification queries.
        
        Args:
            raw_content: The raw content of the document
            verification_queries: List of specific queries to guide extraction
            
        Returns:
            Extracted detailed content
        """
        if not verification_queries:
            return ""
        
        DETAILS_SUMMARIZER_PROMPT = self.prompts["details_summarizer_prompt"]
        
        queries_text = "\n".join([f"- {vq}" for vq in verification_queries])
        message_content = (
            f"<Raw Content>{raw_content}</Raw Content>\n\n"
            f"<Verification Queries>\n{queries_text}\n</Verification Queries>"
        )
        
        result = await asingle_shot_llm_call(
            model=self.summarization_model,
            system_prompt=DETAILS_SUMMARIZER_PROMPT,
            message=message_content,
            token_usage=self.token_usage
        )
        
        return result

   
    async def evaluate_research_completeness(self, topic: str, results: DeepResearchResults, queries: List[str]) -> list[str]:
        """
        Evaluate if the current search results are sufficient or if more research is needed.
        Returns an empty list if research is complete, or a list of additional queries if more research is needed.
        """

        # Format the search results for the LLM
        formatted_results = str(results)
        EVALUATION_PROMPT = self.prompts["evaluation_prompt"]

        evaluation = await asingle_shot_llm_call(
            model=self.planning_model,
            system_prompt=EVALUATION_PROMPT,
            message=(
                f"<Research Topic>{topic}</Research Topic>\n\n"
                f"<Search Queries Used>{queries}</Search Queries Used>\n\n"
                f"<Current Search Results>{formatted_results}</Current Search Results>"
            ),
            token_usage=self.token_usage
        )

        logging.info(f"Evaluation: {evaluation}")

        EVALUATION_PARSING_PROMPT = self.prompts["evaluation_parsing_prompt"]

        response_json = await asingle_shot_llm_call(
            model=self.json_model,
            system_prompt=EVALUATION_PARSING_PROMPT,
            message=f"Evaluation to be parsed: {evaluation}",
            # response_format={"type": "json_object", "schema": ResearchPlan.model_json_schema()},
            response_format=ResearchPlan,
            token_usage=self.token_usage
        )

        evaluation = json.loads(response_json)
        return evaluation["queries"]

    async def filter_results(self, topic: str, results: DeepResearchResults) -> tuple[DeepResearchResults, SourceList]:
        """Filter the search results based on the research plan"""

        # Format the search results for the LLM, without the raw content
        formatted_results = str(results)

        FILTER_PROMPT = self.prompts["filter_prompt"]

        filter_response = await asingle_shot_llm_call(
            model=self.planning_model,
            system_prompt=FILTER_PROMPT,
            message=(
                f"<Evaluation Task>{topic}</Evaluation Task>\n\n"
                f"<Current Search Results>{formatted_results}</Current Search Results>"
            ),
            # NOTE: This is the max_token parameter for the LLM call on Together AI, may need to be changed for other providers
            max_completion_tokens=4096,
            token_usage=self.token_usage
        )

        logging.info(f"Filter response: {filter_response}")

        FILTER_PARSING_PROMPT = self.prompts["filter_parsing_prompt"]

        response_json = await asingle_shot_llm_call(
            model=self.json_model,
            system_prompt=FILTER_PARSING_PROMPT,
            message=f"Filter response to be parsed: {filter_response}",
            # response_format={"type": "json_object", "schema": SourceList.model_json_schema()},
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

    async def generate_research_answer(self, topic: str, results: DeepResearchResults, remove_thinking_tags: bool = False):
        """
        Generate a comprehensive answer to the research topic based on the search results.
        Returns a detailed response that synthesizes information from all search results.
        """

        formatted_results = str(results)
        ANSWER_PROMPT = self.prompts["answer_prompt"]

        answer = await asingle_shot_llm_call(
            model=self.answer_model,
            system_prompt=ANSWER_PROMPT,
            message=f"Research Topic: {topic}\n\nSearch Results:\n{formatted_results}",
            # NOTE: This is the max_token parameter for the LLM call on Together AI, may need to be changed for other providers
            max_completion_tokens=self.max_completion_tokens,
            response_format=FactualVerdict,
            token_usage=self.token_usage
        )

        # this is just to avoid typing complaints
        # if answer is None or not isinstance(answer, str):
        #     logging.error("No answer generated")
        #     return "No answer generated"
        #
        # if remove_thinking_tags:
        #     # Remove content within <think> tags
        #     answer = self._remove_thinking_tags(answer)
        #
        # # Remove markdown code block markers if they exist at the beginning
        # if answer.lstrip().startswith("```"):
        #     # Find the first line break after the opening backticks
        #     first_linebreak = answer.find("\n", answer.find("```"))
        #     if first_linebreak != -1:
        #         # Remove everything up to and including the first line break
        #         answer = answer[first_linebreak + 1 :]
        #
        #     # Remove closing code block if it exists
        #     if answer.rstrip().endswith("```"):
        #         answer = answer.rstrip()[:-3].rstrip()

        return answer

    def _remove_thinking_tags(self, answer: str) -> str:
        """Remove content within <think> tags"""
        while "<think>" in answer and "</think>" in answer:
            start = answer.find("<think>")
            end = answer.find("</think>") + len("</think>")
            answer = answer[:start] + answer[end:]
        return answer




def evaluate_report(data: dict, config_path: str, max_workers: int = 16) -> dict:
    from deep_fact.evaluators import create_agent
    logging.info(f"Using configuration from file: {config_path}")
    report = data["response"]
    annotated_sentences = [s | {"sentence_idx": i} for i, s in enumerate(data["sentences_info"]) if
                           s.get("human_verified") is True and s.get("human_verdict") is not None]
    sentences_only = [s["sentence"] for s in annotated_sentences]

    # Use the new evaluate_report method with multi-threading
    instance = create_agent(config_path, return_instance=True)

    # Prepare report data for the new interface
    report_data = {
        "response": report,
        "thesis": data.get("thesis", ""),
    }

    # Evaluate all claims using the new multi-threaded interface
    results = instance.evaluate_report(report_data=report_data, claims=sentences_only, max_workers=max_workers)

    # Convert results to the expected format for compatibility
    formatted_results = []
    for result in results:
        if "error" in result:
            formatted_results.append({
                "idx": result["idx"],
                "claim": result["claim"],
                "error": result["error"],
                "input_tokens": 0,
                "output_tokens": 0,
            })
        else:
            # Parse the evaluation JSON if it's a string
            evaluation = result["evaluation"]
            if isinstance(evaluation, str):
                try:
                    evaluation_json = json.loads(evaluation)
                    rationale = evaluation_json.get("rationale", "")
                    verdict = evaluation_json.get("verdict", "")
                except (json.JSONDecodeError, TypeError):
                    rationale = str(evaluation)
                    verdict = ""
            else:
                rationale = str(evaluation)
                verdict = ""

            formatted_results.append({
                "idx": result["idx"],
                "claim": result["claim"],
                "context": result.get("context", ""),
                "rationale": rationale,
                "verdict": verdict,
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
            })

    final_results = {
        "evaluation": formatted_results,
        "token_usage": instance.token_usage
    }
    return final_results



