import re
import json
# from blingfire import text_to_sentences

import asyncio
import hashlib
import pathlib
from pathlib import Path
from crawl4ai.docker_client import Crawl4aiDockerClient
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from crawl4ai import CacheMode, CrawlerRunConfig, LLMConfig, BrowserConfig, AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy
from collections import defaultdict

from deep_fact.utils.mdconvert import MarkdownConverter
from deep_fact.utils import read_jsonl
import logging
logger = logging.getLogger(__name__)
from litellm import completion
import os

EVAL_CITATIONS_MODEL = os.getenv("EVAL_CITATIONS_MODEL", "openai/gpt-4.1")


def _llm_invoke(text_input: str) -> str:
    response = completion(
        model=EVAL_CITATIONS_MODEL,
        messages=[{"role": "user", "content": text_input}],
    )
    return response.choices[0].message["content"]

_cache_dir_env = os.getenv("CRAWL_CACHE_DIR")
CACHE_DIR = (
    pathlib.Path(_cache_dir_env).expanduser()
    if _cache_dir_env
    else pathlib.Path.home() / "fact_eval" / ".crawl_cache"
)
BROWSER_DOWNLOADS = CACHE_DIR / "browser_downloads"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BROWSER_DOWNLOADS.mkdir(parents=True, exist_ok=True)

USE_TAVILY = True
if USE_TAVILY:
    from tavily import AsyncTavilyClient


prompt_extract_citations = """You will be provided with a research report. The body of the report will contain some citations to references.

Citations in the main text may appear in the following forms:
1. A segment of text + space + number, for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels 15"
2. A segment of text + [number], for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels[15]"
3. A segment of text + [number†(some line numbers, etc.)], for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels[15†L10][5L23][7†summary]"
4. [Citation Source](Citation Link), for example: "According to [ChinaFile: A Guide to Social Class in Modern China](https://www.chinafile.com/reporting-opinion/media/guide-social-class-modern-china)'s classification, Chinese society can be divided into nine strata"

Please identify **all** instances where references are cited in the main text, and extract (fact, ref_idx, url, sentence_idx) entries. When extracting, pay attention to the following:
1. Since these facts will need to be verified later, you should consider the surrounding context—both before and after the citation and its corresponding sentence—to ensure that each extracted fact is complete, self-contained, and understandable, rather than a fragment or overly brief expression.
2. If a fact cites multiple references, then it should correspond to multiple entries: one per reference, with the same fact and sentence index but different ref_idx/url.
3. The report will be split into individual sentences, each starting with a sentence index in the form of <sentence_idx>. You should retain this index and include it in the output as the "sentence_idx" field.
4. For the fourth form of citation (i.e., where the citation source and link appear directly in the text), the ref_idx should be uniformly set to 0.
5. If the main text does not specify the exact location of the citation (e.g., only a reference list at the end), return an empty list.

You should return a JSON list format, where each item in the list is a triplet, for example:
[
    {{
        "fact": "A rephrased fact claim based on the corresponding sentence in the original document. The fact should be rewritten to be **self-contained and understandable on its own, incorporating any necessary surrounding context to ensure clarity and completeness**. Do not simply copy the original sentence. Add a single backslash before any English quotation marks to ensure the string is JSON-compatible in Python."
        "ref_idx": "The index of the cited reference in the reference list for this text segment.",
        "url": "The URL of the cited reference for this text segment (extracted from the reference list at the end of the research report or from the parentheses at the citation point).",
        "sentence_idx": "The index of the sentence this citation was extracted from."
    }}
]

Here is the main text of the research report:
{report_text}

Please begin the extraction now. Output only the JSON list directly, without any chitchat or explanations."""

statement_template = '''
<statement_{idx}>
{statement}
</statement_{idx}>
'''

template_validate = """You will be provided with a reference and some statements. Please determine whether each statement is 'supported', 'unsupported', or 'unknown' with respect to the reference.

Please note:
First, assess whether the reference contains any valid content. If the reference contains no valid information, such as a 'page not found' message, then all statements should be considered 'unknown'.
If the reference is valid, for each statement: if the facts or data it contains can be found entirely or partially within the reference, it is considered 'supported' (data accepts rounding); if all facts and data in the statement cannot be found in the reference, it is considered 'unsupported'.

You should return the result in a JSON list format, where each item in the list contains the statement's index and the judgment result, for example:
[
    {{
        "idx": 0,
        "result": "supported",
        "rationale": "Explain why the statement is supported by the reference."
    }},
    {{
        "idx": 1,
        "result": "unsupported",
        "rationale": "Explain why the statement is not supported by the reference."
    }}
]

Below are the reference and statements:
<reference>
{reference}
</reference>

<statements>
{statements}
</statements>

Begin the assessment now. Output only the JSON list, without any conversational text or explanations."""


def clean_urls(input_text):
    # match [title](url) format
    pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

    def repl(match):
        title = match.group(1)
        url = match.group(2)
        # truncate #:~:text= and its content
        cut_idx = url.find('#:~:text=')
        if cut_idx != -1:
            url = url[:cut_idx]
        return f'[{title}]({url})'

    return pattern.sub(repl, input_text)


def remove_urls(input_text):
    # match [title](url) format, only remove the content in the parentheses, keep [title]
    pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    # replace [title](url) with [title]
    return pattern.sub(r'[\1]', input_text)


def clean_escape(input_text):
    # replace illegal escape characters
    input_text = input_text.replace("\\>", ">")
    input_text = input_text.replace("\\<", "<")
    input_text = input_text.replace("\\+", "+")
    input_text = input_text.replace("\\~", "~")
    return input_text

def get_citations(row):
    # text_input = prompt_template_old.format(report_text=report)

    sentences = row["sentences"]
    report = [f"<{i}> {sentence}" for i, sentence in enumerate(sentences)]
    report = "\n".join(report)

    text_input = prompt_extract_citations.format(report_text=report)

    retry_num = 3
    for i in range(retry_num):
        try:
            response = _llm_invoke(text_input)
            citations = json.loads(response.replace("```json", "").replace("```", ""))
            for c in citations:
                continue
            break
        except Exception as e:
            if i == retry_num - 1:
                raise e
            time.sleep(3)

    row["citations"] = citations
    for c in row['citations']:
        c['fact'] = remove_urls(c['fact'])
    row["sentences"] = sentences
    return row



def canonicalize_url(url: str) -> str:
    """
    Strip fragment (#…), chop tracking query params, sort the rest,
    lowercase scheme/host, and return the rebuilt URL.
    """
    _TRACKING_RE = re.compile(r"^(utm_[a-z0-9_]+|fbclid|gclid)$", re.I)
    parsed = urlparse(url.strip())
    # 1) drop fragment – browsers don't send it to the server anyway
    parsed = parsed._replace(fragment="")
    # 2) clean & sort query string
    q = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if not _TRACKING_RE.match(k)]
    q.sort()
    parsed = parsed._replace(query=urlencode(q, doseq=True))
    # 3) lowercase scheme + hostname for consistency
    parsed = parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower())
    return urlunparse(parsed)

def url_to_cache_path(url: str) -> pathlib.Path:
    """
    <domain>_<sha1>.md  — deterministic & filesystem-safe
    """
    can = canonicalize_url(url)
    h = hashlib.sha1(can.encode()).hexdigest()[:32]
    domain = urlparse(can).netloc.replace(":", "_")
    return CACHE_DIR / f"{domain}_{h}.md"

async def get_markdown_tavily(url: str):
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    client = AsyncTavilyClient(api_key)
    try:
        result = await client.extract(url)
    except Exception as e:
        print(f"Error extracting from {url}: {e}")
        return ""
    if len(result["results"]) == 0:
        logger.warning(f"✗ warning: empty tavily result whn fetching {url}")
        return ""
    return result["results"][0]['raw_content']

async def get_markdown(url: str, client) -> tuple[str, str]:
    """
    Return Markdown for *url*, hitting the Docker server only if the
    canonical page is not cached locally.
    """
    cache_path = url_to_cache_path(url)
    canonical = canonicalize_url(url)
    error_message = "Application error: a client-side exception has occurred (see the browser console for more information)"
    if cache_path.exists():
        cached_md = cache_path.read_text(encoding="utf-8").strip()
        if cached_md != "" and error_message not in cached_md:
            logger.info(f"✓ cache loaded for {url}")
            return cache_path.read_text(encoding="utf-8"), canonical
    max_retries = 1
    base_delay = 0
    # Resolve duplicates BEFORE the network call
    markdown = ""
    for attempt in range(1, max_retries + 1):
        try:
            # result = await client.crawl(urls=[canonical], crawler_config=crawl_cfg, browser_config=BrowserConfig(headless=False))
            if canonical.lower().endswith("pdf") or "arxiv.org/pdf" in canonical or '/pdf/' in canonical:
                md_converter = MarkdownConverter()
                output = md_converter.convert_url(canonical)
                markdown = output.text_content
            else:
                # Support both API crawler (AsyncWebCrawler.arun) and Docker client (crawl)
                if hasattr(client, "arun"):
                    result = await client.arun(url=canonical)
                    markdown = (getattr(result, "markdown", None) or "").strip()
                else:
                    result = await client.crawl(urls=[canonical])
                    markdown = (getattr(result, "markdown", None) or "").strip()

            if not markdown:
                raise ValueError("Empty markdown returned")

            cache_path.write_text(markdown, encoding="utf-8")
            logger.info(f"✓ fetched & cached → {cache_path.name}, url: {canonical}")
            return markdown, canonical

        except Exception as exc:
            if attempt == max_retries:
                if USE_TAVILY:
                    markdown = await get_markdown_tavily(canonical)
                    if markdown != "":
                        cache_path.write_text(markdown, encoding="utf-8")
                        logger.info(f"✓ fetched & cached → {cache_path.name}, url: {canonical}")
                        return markdown, canonical

                logger.warning(f"✗ warning: fail to fetch {url}, {exc}")
                return markdown, canonical
            wait = base_delay
            await asyncio.sleep(wait)

    return markdown, canonical

async def fetch_urls_markdown(urls):
    backend = os.getenv("CRAWL4AI_BACKEND", "api").lower()
    markdowns = []
    if backend == "docker":
        async with Crawl4aiDockerClient(
            base_url="http://localhost:11235",
            verbose=False,
        ) as client:
            # comment out if JWT is disabled in your config.yml
            await client.authenticate("me@example.com")
            for url in urls:
                markdown, canonical_url = await get_markdown(url, client)
                markdowns.append({"markdown": markdown, "canonical_url": canonical_url, "url": url})
            return markdowns
    else:
        async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as client:
            for url in urls:
                markdown, canonical_url = await get_markdown(url, client)
                markdowns.append({"markdown": markdown, "canonical_url": canonical_url, "url": url})
            return markdowns



def eval_citations(row):
    row = get_citations(row)
    citations = row["citations"]
    urls = [c['url'] for c in citations]
    markdowns = asyncio.run(fetch_urls_markdown(urls))
    urls_markdown = defaultdict(str)
    urls_statement_idx = defaultdict(list)
    for i, m in enumerate(markdowns):
        urls_markdown[m['canonical_url']] = m['markdown']
        urls_statement_idx[m['canonical_url']].append(i)

    sentences = row["sentences"]

    def _validate_statements(statements, reference):

        statements_text = []
        for k, statement_info in enumerate(statements):
            statement = statement_info["fact"]
            j = int(statement_info["sentence_idx"])
            context1 = " ".join(sentences[max(0, j - 3):j])
            sentence = f"{sentences[j].strip()}"
            context2 = " ".join(sentences[j + 1:j + 2])
            snippet = f"{context1.strip()} {sentence.strip()} {context2.strip()}".strip()
            snippet = ""
            statements_text.append(statement_template.format(idx=k, statement=statement))
        statements_text = "\n".join(statements_text)

        text_input = template_validate.format(reference=reference, statements=statements_text)

        retries = 0
        validate_res = []
        while retries < 3:
            try:
                response = _llm_invoke(text_input)
                validate_res = json.loads(response.replace("```json", "").replace("```", ""))
                for j, res in enumerate(validate_res):
                    validate_res[j]["sentence_idx"] = statements[j]['sentence_idx']
                    validate_res[j]["statement_idx"] = statements[j]['statement_idx']
                    validate_res[j]["statement"] = statements[j]['fact']
                return validate_res

            except Exception as e:
                time.sleep(3)
                retries += 1

        return validate_res

    tasks = []
    for url, md in urls_markdown.items():
        if md.strip() == "":
            for statement_idx in urls_statement_idx[url]:
                citations[statement_idx]["citation_verdict"] = "unknown"
                citations[statement_idx]["citation_rationale"] = "Fail to scrape the cited website"

        else:
            statements = [citations[statement_idx] | {"statement_idx": statement_idx} for statement_idx in urls_statement_idx[url]]
            tasks.append([statements, md])


    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = [
            pool.submit(_validate_statements, *item)
            for item in tasks]
    results = []
    for fut in futures:
        results.append(fut.result())

    for results_per_link in results:
        for validate_res in results_per_link:
            statement_idx = int(validate_res["statement_idx"])
            citations[statement_idx]["citation_verdict"] = validate_res["result"]
            citations[statement_idx]["citation_rationale"] = validate_res["rationale"]

    row["citations"] = citations

    for c in citations:
        sentence_idx = int(c["sentence_idx"])
        row["sentences_info"][sentence_idx]["citation_verdict"] = c['citation_verdict']
        row["sentences_info"][sentence_idx]["citation_rationale"] = c['citation_rationale']


    return row





