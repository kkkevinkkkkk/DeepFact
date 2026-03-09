import asyncio
import os
import random
from typing import Optional

import aiohttp

from deep_fact.evaluators.models.types import SearchResult, SearchResults

_SERPER_API_KEY = os.getenv("SERPER_API_KEY")
_SERPER_URL: str = "https://google.serper.dev/search"


async def _post_serper(session: aiohttp.ClientSession, query: str) -> dict:
    """Low-level POST; raises for HTTP errors."""
    headers = {
        "X-API-KEY": _SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"q": query}

    async with session.post(_SERPER_URL, headers=headers, json=payload) as resp:
        resp.raise_for_status()
        return await resp.json()


async def _post_with_retry(
    session: aiohttp.ClientSession,
    query: str,
    attempts: int = 4,
    backoff_base: float = 1.6,
) -> dict:
    """Retry wrapper for `_post_serper`."""
    last_exc: Optional[Exception] = None
    for n in range(1, attempts + 1):
        try:
            return await _post_serper(session, query)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_exc = exc
            if n == attempts:
                break
            sleep_for = backoff_base ** (n - 1) + random.random()
            await asyncio.sleep(sleep_for)
    raise last_exc


async def _to_result(item: dict, include_raw: bool) -> SearchResult:
    return SearchResult(
        title=item.get("title", ""),
        link=item.get("link", ""),
        content=item.get("snippet", ""),
        raw_content=await get_raw_content(item.get("link", "")) if include_raw else None,
    )


async def authenticate_with_retry(client, email, retries: int = 3, backoff: int = 1):
    for attempt in range(retries):
        try:
            await client.authenticate(email)
            return
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(backoff * 2 ** attempt)


async def get_raw_content(link: str):
    from deep_fact.utils.eval_citations import get_markdown

    backend = os.getenv("CRAWL4AI_BACKEND", "api").lower()
    if backend == "docker":
        from crawl4ai.docker_client import Crawl4aiDockerClient

        async with Crawl4aiDockerClient(base_url="http://localhost:11235", verbose=False) as client:
            await authenticate_with_retry(client, "me@example.com")
            markdown, _canonical_url = await get_markdown(link, client)
            return markdown
    from crawl4ai import AsyncWebCrawler, BrowserConfig

    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as client:
        markdown, _canonical_url = await get_markdown(link, client)
        return markdown


async def aserper_search_results(
    query: str,
    max_results: int = 3,
    include_raw: bool = True,
) -> SearchResults:
    """Asynchronously query Serper.dev and return top results."""
    if not _SERPER_API_KEY:
        raise RuntimeError("Environment variable SERPER_API_KEY is not set or empty.")

    async with aiohttp.ClientSession() as session:
        data = await _post_with_retry(session, query)

    organic = data.get("organic", [])
    return SearchResults(results=[await _to_result(hit, include_raw) for hit in organic[:max_results]])


if __name__ == "__main__":
    print(asyncio.run(aserper_search_results("What is the capital of France?")))
