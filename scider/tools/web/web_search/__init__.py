"""WebSearchTool — search the web via DuckDuckGo (ddgs SDK).

Uses the ``ddgs`` package (https://github.com/deedy5/ddgs) for web search.
No API key required.
"""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import BaseModel, Field

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
        "Search the web using DuckDuckGo and return results with titles, URLs, and snippets. "
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
        try:
            from ddgs import DDGS

            results = DDGS().text(query, max_results=max_results)

            # Client-side domain filtering
            results = _filter_by_domain(results, allowed_domains, blocked_domains)

            if not results:
                return f'No results found for query: "{query}"'

            lines = [f'Web search results for query: "{query}"\n\nLinks:']
            for r in results:
                title = r.get("title", "")
                url = r.get("href", "")
                snippet = r.get("body", "")
                snippet_part = f": {snippet}" if snippet else ""
                lines.append(f"  - [{title}]({url}){snippet_part}")

            return "\n".join(lines)

        except ImportError:
            return "Error: ddgs package not installed. Run `pip install ddgs`."
        except Exception as e:
            return f"Error performing web search: {e}"


def _filter_by_domain(
    results: list[dict],
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
) -> list[dict]:
    """Filter results by allowed/blocked domain lists."""
    if not allowed_domains and not blocked_domains:
        return results

    filtered = []
    for r in results:
        try:
            hostname = urlparse(r.get("href", "")).hostname or ""
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
