from deep_fact.evaluators.models.types import SearchResult, SearchResults


def limit_results(results: SearchResults, max_results: int) -> SearchResults:
    """Return a truncated SearchResults list while preserving type."""
    return SearchResults(results=results.results[:max_results])


def ensure_result_fields(result: SearchResult) -> SearchResult:
    """Normalize optional string fields to empty strings."""
    return SearchResult(
        title=result.title or "",
        link=result.link or "",
        content=result.content or "",
        raw_content=result.raw_content or "",
    )
