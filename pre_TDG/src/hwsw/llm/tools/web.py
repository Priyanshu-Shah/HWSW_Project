"""`web_search` (DuckDuckGo, no API key) and `fetch_page` (main text only), cached on disk."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import trafilatura
from ddgs import DDGS

from hwsw.llm.tools.tool import Tool, object_schema
from hwsw.repo_paths import CACHE_DIR

MAX_RESULTS = 5
MAX_PAGE_CHARS = 16_000  # about 4k tokens
_USER_AGENT = "Mozilla/5.0 (X11; Linux aarch64) hwsw-pre-tdg"


def web_search_tool(cache_dir: Path = CACHE_DIR / "web_search") -> Tool:
    def web_search(query: str) -> str:
        return _cached(cache_dir, query, lambda: _search(query))

    return Tool(
        name="web_search",
        description=(
            "Search the web. Returns titles, URLs and snippets. Use it for facts such as a "
            "neural network's GFLOPs or input size; cite the URL of what you use."
        ),
        parameters=object_schema({"query": {"type": "string"}}),
        run=web_search,
    )


def fetch_page_tool(cache_dir: Path = CACHE_DIR / "pages") -> Tool:
    def fetch_page(url: str) -> str:
        return _cached(cache_dir, url, lambda: _fetch(url))

    return Tool(
        name="fetch_page",
        description="The main text of a web page (cut to about 4000 tokens).",
        parameters=object_schema({"url": {"type": "string"}}),
        run=fetch_page,
    )


def _search(query: str) -> str:
    results = DDGS().text(query, max_results=MAX_RESULTS)
    if not results:
        return "no results"
    return json.dumps(
        [{"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in results], indent=1
    )


def _fetch(url: str) -> str:
    response = httpx.get(
        url, follow_redirects=True, timeout=30, headers={"User-Agent": _USER_AGENT}
    )
    response.raise_for_status()
    text = trafilatura.extract(response.text) or ""
    if not text:
        return "the page has no readable main text"
    return text[:MAX_PAGE_CHARS]


def _cached(cache_dir: Path, key: str, compute: Callable[[], str]) -> str:
    """Same query or URL, same answer: makes runs repeatable and saves network time."""
    path = cache_dir / (hashlib.sha256(key.encode()).hexdigest()[:32] + ".json")
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))["result"]
    result = compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "result": result}), encoding="utf-8")
    return result
