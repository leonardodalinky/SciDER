"""WebSearchTool — search the web via a pluggable backend.

Backend is selected at call time by the ``WEB_SEARCH_VERSION`` env var
(see ``scider/core/constant.py``):
- ``"duckduckgo"`` (default): uses the ``ddgs`` package — no API key needed.
- ``"tavily"``: uses ``tavily-python`` — LLM-optimized snippets + better
  relevance, but requires ``TAVILY_API_KEY``.

We read the env var on every call (not at import time) so runtime changes
(e.g. tests, the streamlit client toggling it) take effect without a
restart. This mirrors ``CODING_AGENT_VERSION``.
"""

from __future__ import annotations

from urllib.parse import urlparse

from loguru import logger
from pydantic import BaseModel, Field

from scider.core import constant

from ...base import BaseTool, ToolContext

# Default timeout for the underlying HTTP requests (seconds).
_TIMEOUT = 30


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query (min 2 characters)", min_length=2)
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return (default 10)",
        ge=1,
        le=30,
    )
    allowed_domains: list[str] | None = Field(
        default=None,
        description="Only include results from these domains (optional)",
    )
    blocked_domains: list[str] | None = Field(
        default=None,
        description="Exclude results from these domains (optional)",
    )


class WebSearchTool(BaseTool):
    name = "WebSearch"
    description = (
        "Search the web and return results with titles, URLs, and snippets. "
        "Backend is DuckDuckGo by default; set WEB_SEARCH_VERSION=tavily "
        "(+ TAVILY_API_KEY) to switch to Tavily. "
        "Use allowed_domains/blocked_domains to filter results by domain. "
        "Always include a Sources section with URLs in your response."
    )
    input_schema = WebSearchInput
    _always_read_only = True
    max_result_size_chars = 100_000
    prompt = (
        "# WebSearch tool usage\n"
        "- Use for looking up documentation, APIs, current events, or anything not in the codebase.\n"
        "- When presenting search results to the user, ALWAYS include a 'Sources:' section "
        "with markdown links at the end.\n"
        "- Use `allowed_domains` to restrict to specific sites (e.g., docs.python.org).\n"
    )

    def call(
        self,
        context: ToolContext,
        *,
        query: str,
        max_results: int = 10,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> str:
        version = (constant.WEB_SEARCH_VERSION or "duckduckgo").lower()
        try:
            if version == "tavily":
                return _search_tavily(query, max_results, allowed_domains, blocked_domains)
            if version in ("duckduckgo", "ddg", "ddgs"):
                return _search_duckduckgo(query, max_results, allowed_domains, blocked_domains)
            logger.warning("Unknown WEB_SEARCH_VERSION={!r}; falling back to duckduckgo", version)
            return _search_duckduckgo(query, max_results, allowed_domains, blocked_domains)
        except Exception as e:
            return f"Error performing web search (backend={version}): {e}"


# --------------------------------------------------------------------------- #
# Backends                                                                    #
# --------------------------------------------------------------------------- #


def _search_duckduckgo(
    query: str,
    max_results: int,
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: ddgs package not installed. Run `pip install ddgs`."

    results = DDGS().text(query, max_results=max_results)
    # DDGS has no native domain filter — apply client-side.
    results = _filter_by_domain(
        results,
        allowed_domains,
        blocked_domains,
        url_key="href",
    )
    if not results:
        return f'No results found for query: "{query}" (backend=duckduckgo)'

    lines = [f'Web search results for query: "{query}" (backend=duckduckgo)\n\nLinks:']
    for r in results:
        title = r.get("title", "")
        url = r.get("href", "")
        snippet = r.get("body", "")
        snippet_part = f": {snippet}" if snippet else ""
        lines.append(f"  - [{title}]({url}){snippet_part}")
    return "\n".join(lines)


def _search_tavily(
    query: str,
    max_results: int,
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
) -> str:
    if not constant.TAVILY_API_KEY:
        return (
            "Error: WEB_SEARCH_VERSION=tavily but TAVILY_API_KEY is empty. "
            "Set TAVILY_API_KEY in your .env, or switch to WEB_SEARCH_VERSION=duckduckgo."
        )
    try:
        from tavily import TavilyClient
    except ImportError:
        return "Error: tavily-python not installed. Run `pip install tavily-python`."

    client = TavilyClient(api_key=constant.TAVILY_API_KEY)
    # Tavily supports native domain filters — pass them through (capped at 300
    # domains per the API). It also clamps max_results to 20.
    kwargs: dict = {
        "query": query,
        "max_results": min(max_results, 20),
        "search_depth": "basic",
    }
    if allowed_domains:
        kwargs["include_domains"] = list(allowed_domains)
    if blocked_domains:
        kwargs["exclude_domains"] = list(blocked_domains)

    resp = client.search(**kwargs)
    results = resp.get("results") or []
    if not results:
        return f'No results found for query: "{query}" (backend=tavily)'

    lines = [f'Web search results for query: "{query}" (backend=tavily)\n\nLinks:']
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = (r.get("content") or "").strip().replace("\n", " ")
        score = r.get("score")
        score_part = f" (score={score:.2f})" if isinstance(score, (int, float)) else ""
        snippet_part = f": {snippet}" if snippet else ""
        lines.append(f"  - [{title}]({url}){score_part}{snippet_part}")
    return "\n".join(lines)


def _filter_by_domain(
    results: list[dict],
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
    *,
    url_key: str = "href",
) -> list[dict]:
    """Filter results by allowed/blocked domain lists."""
    if not allowed_domains and not blocked_domains:
        return results

    filtered = []
    for r in results:
        try:
            hostname = urlparse(r.get(url_key, "")).hostname or ""
        except Exception:
            continue

        if allowed_domains and not any(
            hostname == d or hostname.endswith("." + d) for d in allowed_domains
        ):
            continue
        if blocked_domains and any(
            hostname == d or hostname.endswith("." + d) for d in blocked_domains
        ):
            continue

        filtered.append(r)
    return filtered
