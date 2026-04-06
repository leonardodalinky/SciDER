"""WebSearchTool — search the web via Bing HTML scraping.

Modeled after Claude Code's BingSearchAdapter. Fetches Bing search pages
and extracts results using regex on raw HTML. No external search API needed.
"""

from __future__ import annotations

import base64
import re
from html import unescape
from urllib.parse import quote, urljoin

import requests
from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext

FETCH_TIMEOUT = 30

# Browser-like headers to avoid Bing's anti-bot JS-rendered response.
# Mimics Microsoft Edge on macOS to get full HTML search results.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Ch-Ua": '"Microsoft Edge";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query (min 2 characters)", min_length=2)
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
        "Search the web using Bing and return results with titles, URLs, and snippets. "
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
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> str:
        try:
            url = f"https://www.bing.com/search?q={quote(query)}&setmkt=en-US"

            response = requests.get(url, headers=BROWSER_HEADERS, timeout=FETCH_TIMEOUT)
            response.raise_for_status()

            results = _extract_bing_results(response.text)

            # Client-side domain filtering
            results = _filter_by_domain(results, allowed_domains, blocked_domains)

            if not results:
                return f'No results found for query: "{query}"'

            # Format as markdown links (Claude Code style)
            lines = [f'Web search results for query: "{query}"\n\nLinks:']
            for r in results:
                snippet_part = f": {r['snippet']}" if r.get("snippet") else ""
                lines.append(f"  - [{r['title']}]({r['url']}){snippet_part}")

            return "\n".join(lines)

        except requests.exceptions.Timeout:
            return f"Error: Search timed out after {FETCH_TIMEOUT} seconds"
        except Exception as e:
            return f"Error performing web search: {e}"


def _extract_bing_results(html: str) -> list[dict]:
    """Extract organic search results from Bing HTML.

    Bing results live in <li class="b_algo"> blocks within <ol id="b_results">.
    """
    results = []

    block_regex = re.compile(r'<li\s+class="b_algo"[^>]*>([\s\S]*?)</li>', re.IGNORECASE)
    h2_link_regex = re.compile(
        r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>([\s\S]*?)</a>', re.IGNORECASE
    )

    for block_match in block_regex.finditer(html):
        block = block_match.group(1)

        link_match = h2_link_regex.search(block)
        if not link_match:
            continue

        raw_url = unescape(link_match.group(1))
        title_html = link_match.group(2)

        url = _resolve_bing_url(raw_url)
        if not url:
            continue

        title = unescape(re.sub(r"<[^>]+>", "", title_html).strip())
        snippet = _extract_snippet(block)

        results.append({"title": title, "url": url, "snippet": snippet})

    return results


def _extract_snippet(block: str) -> str | None:
    """Extract snippet text from a Bing result block."""
    # 1. Try <p class="b_lineclamp...">
    m = re.search(r'<p[^>]*class="b_lineclamp[^"]*"[^>]*>([\s\S]*?)</p>', block, re.IGNORECASE)
    if m:
        return unescape(re.sub(r"<[^>]+>", "", m.group(1)).strip())

    # 2. Try <p> inside b_caption
    m = re.search(
        r'<div[^>]*class="b_caption[^"]*"[^>]*>[\s\S]*?<p[^>]*>([\s\S]*?)</p>',
        block,
        re.IGNORECASE,
    )
    if m:
        return unescape(re.sub(r"<[^>]+>", "", m.group(1)).strip())

    # 3. Fallback: any text inside b_caption <div>
    m = re.search(r'<div[^>]*class="b_caption[^"]*"[^>]*>([\s\S]*?)</div>', block, re.IGNORECASE)
    if m:
        text = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        if text:
            return unescape(text)

    return None


def _resolve_bing_url(raw_url: str) -> str | None:
    """Resolve a Bing redirect URL to the actual target URL.

    Bing uses URLs like: https://www.bing.com/ck/a?...&u=a1aHR0cHM6Ly9leGFtcGxlLmNvbQ...
    The `u` query parameter is a base64-encoded URL prefixed with a1 (https) or a0 (http).
    """
    if raw_url.startswith("/") or raw_url.startswith("#"):
        return None

    # Try to extract the `u` parameter from Bing redirect URLs
    u_match = re.search(r"[?&]u=([a-zA-Z0-9+/_=-]+)", raw_url)
    if u_match:
        encoded = u_match.group(1)
        if len(encoded) >= 3:
            b64 = encoded[2:]  # Skip prefix (a1/a0)
            try:
                padded = b64.replace("-", "+").replace("_", "/")
                # Add padding if needed
                padded += "=" * (-len(padded) % 4)
                decoded = base64.b64decode(padded).decode("utf-8")
                if decoded.startswith("http"):
                    return decoded
            except Exception:
                pass

    # Direct external URL (not a Bing-internal page)
    if "bing.com" not in raw_url:
        return raw_url

    return None


def _filter_by_domain(
    results: list[dict],
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
) -> list[dict]:
    """Filter results by allowed/blocked domain lists."""
    if not allowed_domains and not blocked_domains:
        return results

    from urllib.parse import urlparse

    filtered = []
    for r in results:
        try:
            hostname = urlparse(r["url"]).hostname or ""
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
