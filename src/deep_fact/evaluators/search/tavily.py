import asyncio
import os

from deep_fact.evaluators.models.types import SearchResult, SearchResults


def extract_tavily_results(response) -> SearchResults:
    """Extract key information from Tavily search results."""
    results = []
    for item in response.get("results", []):
        results.append(
            SearchResult(
                title=item.get("title", ""),
                link=item.get("url", ""),
                content=item.get("content", ""),
                raw_content=item.get("raw_content", ""),
            )
        )
    return SearchResults(results=results)


def tavily_search(query: str, max_results: int = 3, include_raw: bool = True) -> SearchResults:
    """Perform a search using the Tavily Search API client."""
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    client = TavilyClient(api_key)
    response = client.search(query=query, search_depth="basic", max_results=max_results, include_raw_content=include_raw)

    return extract_tavily_results(response)


async def atavily_search_results(query: str, max_results: int = 3, include_raw: bool = True) -> SearchResults:
    """Perform asynchronous search using the Tavily Search API client."""
    from tavily import AsyncTavilyClient

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    client = AsyncTavilyClient(api_key)
    response = await client.search(query=query, search_depth="basic", max_results=max_results, include_raw_content=include_raw)

    return extract_tavily_results(response)


if __name__ == "__main__":
    print(asyncio.run(atavily_search_results("What is the capital of France?")))
