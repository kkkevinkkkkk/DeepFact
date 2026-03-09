import asyncio
import select
import sys
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

@dataclass(frozen=True, kw_only=True)
class SearchResult:
    title: str
    link: str
    content: str
    raw_content: Optional[str] = None

    def __str__(self, include_raw: bool = True) -> str:
        result = f"Title: {self.title}\n" f"Link: {self.link}\n" f"Content: {self.content}"
        if include_raw and self.raw_content:
            result += f"\nRaw Content: {self.raw_content}"
        return result

    def short_str(self) -> str:
        return self.__str__(include_raw=False)


@dataclass(frozen=True, kw_only=True)
class SearchResults:
    results: list[SearchResult]

    def __str__(self, short: bool = False) -> str:
        if short:
            result_strs = [result.short_str() for result in self.results]
        else:
            result_strs = [str(result) for result in self.results]
        return "\n\n".join(f"[{i + 1}] {result_str}" for i, result_str in enumerate(result_strs))

    def __add__(self, other: "SearchResults") -> "SearchResults":
        return SearchResults(results=self.results + other.results)

    def short_str(self) -> str:
        return self.__str__(short=True)


class ResearchPlan(BaseModel):
    queries: list[str] = Field(
        description="A list of search queries to thoroughly research the topic")


class SourceList(BaseModel):
    sources: list[int] = Field(
        description="A list of source numbers from the search results")


class VerificationSource(BaseModel):
    title: str = Field(description="Title of the source")
    url: str = Field(description="URL of the source")
    content: str = Field(description="Content of the source relevant to verifying factuality")
class VerificationGuidance(BaseModel):
    sources: list[VerificationSource] = Field(
        description="A list of sources to verify the factuality of the claim")
    guidance: str = Field(description="Key points or angles the expert should focus on when assessing the claim’s verdict.")
    information_gap: str = Field(description="Any important details or evidence not covered by the gathered sources that the expert may need to investigate further. ")



class FactualVerdict(BaseModel):
    rationale: str = Field(
        description="A rational explanation for the factual verdict of the claim.")
    verdict: str = Field(
        description="The factual verdict of the claim, which can only be supported, inconclusive, or contradictory")

    def __str__(self):
        return f"Rationale: {self.rationale}\nVerdict: {self.verdict}"


class FactualVerdictItem(BaseModel):
    claim: str = Field(description="The original claim being evaluated")
    context: str = Field(description="The extracted context used to interpret the claim")
    rationale: str = Field(description="A rational explanation for the factual verdict of the claim.")
    verdict: str = Field(description="The factual verdict of the claim, which can only be supported, inconclusive, or contradictory")


class FactualVerdictList(BaseModel):
    results: list[FactualVerdictItem] = Field(description="List of verdicts for multiple claims")

    def __add__(self, other):
        return FactualVerdictList(results=self.results + other.results)


class VerificationQuery(BaseModel):
    query: str = Field(description="The verification query content")
    relevant_docs: list[int] = Field(description="List of document indices that might contain information for this query")


class VerificationQuerySet(BaseModel):
    queries: list[VerificationQuery] = Field(description="List of verification queries with their relevant documents")


class DocumentVerificationQueries(BaseModel):
    doc_index: int = Field(description="Index of the document")
    doc_title: str = Field(description="Title of the document")
    queries: list[str] = Field(description="List of verification queries specific to this document")


class DocumentDetailAssessment(BaseModel):
    relevant: bool = Field(description="Whether the document is highly relevant to claim verification")
    sufficient: bool = Field(description="Whether the current summary contains sufficient detail")
    queries: list[str] = Field(description="Specific queries to extract missing details", default_factory=list)

    def __add__(self, other):
        # any of the relevant is true means true
        relevant = self.relevant or other.relevant
        # any of the sufficient is false means false
        sufficient = self.sufficient and other.sufficient
        queries = self.queries + other.queries
        return DocumentDetailAssessment(relevant=relevant, sufficient=sufficient, queries=queries)

class SentencePair(BaseModel):
    sentence_idx: int = Field(description="Sentence index")
    sentence: str = Field(description="Sentence text")
    rephrased_sentence: str = Field(description="Rephrased sentence text")

class SentencePairs(BaseModel):
    pairs: list[SentencePair] = Field(description="List of sentence pairs")

class VerificationAspects(BaseModel):
    aspects: list[str] = Field(description="List of key verification aspects that need detailed evidence")

class UserCommunication:
    """Handles user input/output interactions with timeout functionality."""

    @staticmethod
    async def get_input_with_timeout(prompt: str, timeout: float = 30.0) -> str:
        """
        Get user input with a timeout.
        Returns empty string if timeout occurs or no input is provided.

        Args:
            prompt: The prompt to display to the user
            timeout: Number of seconds to wait for user input (default: 30.0)

        Returns:
            str: User input or empty string if timeout occurs
        """
        print(prompt, end="", flush=True)

        # Different implementation for Windows vs Unix-like systems
        if sys.platform == "win32":
            # Windows implementation
            try:
                # Run input in an executor to make it async
                loop = asyncio.get_event_loop()
                user_input = await asyncio.wait_for(loop.run_in_executor(None, input), timeout)
                return user_input.strip()
            except TimeoutError:
                print("\nTimeout reached, continuing...")
                return ""
        else:
            # Unix-like implementation
            i, _, _ = select.select([sys.stdin], [], [], timeout)
            if i:
                return sys.stdin.readline().strip()
            else:
                print("\nTimeout reached, continuing...")
                return ""


@dataclass(frozen=True, kw_only=True)
class DeepResearchResult(SearchResult):
    """Wrapper on top of SearchResults to adapt it to the DeepResearch.

    This class extends the basic SearchResult by adding a filtered version of the raw content
    that has been processed and refined for the specific research context. It maintains
    the original search result while providing additional research-specific information.

    Attributes:
        filtered_raw_content: A processed version of the raw content that has been filtered
                             and refined for relevance to the research topic
        detailed_content: Additional detailed content extracted based on verification queries
    """

    filtered_raw_content: str
    detailed_content: str = ""

    def __str__(self):
        base_str = f"Title: {self.title}\n" f"Link: {self.link}\n" f"##Refined Content: {self.filtered_raw_content[:10000]}"
        if self.detailed_content:
            base_str += f"\n\n##Detailed Evidence: {self.detailed_content[:5000]}"
        return base_str

    def short_str(self):
        return f"Title: {self.title}\nLink: {self.link}\nRaw Content: {self.content[:10000]}"


@dataclass(frozen=True, kw_only=True)
class DeepResearchResults(SearchResults):
    results: list[DeepResearchResult]

    def __add__(self, other):
        return DeepResearchResults(results=self.results + other.results)

    def dedup(self):
        def deduplicate_by_link(results):
            # Track best result and first occurrence index for each link
            link_to_best = {}  # link -> (first_index, best_result)

            for idx, result in enumerate(results):
                current_detailed = getattr(result, 'detailed_content', '') or ''
                
                if result.link not in link_to_best:
                    link_to_best[result.link] = (idx, result)
                else:
                    first_idx, best_result = link_to_best[result.link]
                    best_detailed = getattr(best_result, 'detailed_content', '') or ''
                    
                    # Keep the one with more detailed_content, preserve original index
                    if len(current_detailed) > len(best_detailed):
                        link_to_best[result.link] = (first_idx, result)

            # Sort by original index to maintain order, then extract results
            sorted_items = sorted(link_to_best.values(), key=lambda x: x[0])
            return [item[1] for item in sorted_items]

        return DeepResearchResults(results=deduplicate_by_link(self.results))
