"""WebFetchTool — fetch and extract content from a web URL.

Modeled after Claude Code's WebFetchTool. Key features:
- HTTP→HTTPS upgrade
- HTML→clean text conversion via BeautifulSoup
- In-memory LRU cache (15 min TTL)
- Redirect detection with safety checks
- URL validation
- Content size limits
"""

from __future__ import annotations

import time
from collections import OrderedDict
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from ...base import BaseTool, ToolContext

# Limits (matching Claude Code)
MAX_URL_LENGTH = 2000
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
FETCH_TIMEOUT = 60
MAX_REDIRECTS = 10
MAX_MARKDOWN_LENGTH = 100_000

# Cache: 15 min TTL, max 50 entries
CACHE_TTL_SECONDS = 15 * 60
MAX_CACHE_ENTRIES = 50

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Binary content types to reject
BINARY_TYPES = {
    "pdf",
    "zip",
    "rar",
    "7z",
    "tar",
    "gz",
    "bz2",
    "xz",
    "exe",
    "dmg",
    "pkg",
    "deb",
    "rpm",
    "mp3",
    "mp4",
    "avi",
    "mkv",
    "mov",
    "flac",
    "wav",
    "ogg",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "webp",
    "bmp",
    "tiff",
}


class _URLCache:
    """Simple LRU cache with TTL for URL content."""

    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES, ttl: int = CACHE_TTL_SECONDS):
        self._cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._max = max_entries
        self._ttl = ttl

    def get(self, url: str) -> dict | None:
        if url in self._cache:
            ts, entry = self._cache[url]
            if time.time() - ts < self._ttl:
                self._cache.move_to_end(url)
                return entry
            else:
                del self._cache[url]
        return None

    def set(self, url: str, entry: dict) -> None:
        self._cache[url] = (time.time(), entry)
        self._cache.move_to_end(url)
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)


_url_cache = _URLCache()


class WebFetchInput(BaseModel):
    url: str = Field(description="The URL to fetch content from")
    prompt: str | None = Field(
        default=None,
        description="Optional instructions for how to process the content (e.g., 'extract the API reference')",
    )


class WebFetchTool(BaseTool):
    name = "WebFetch"
    description = (
        "Fetch and extract content from a web URL. Converts HTML to clean text. "
        "HTTP URLs are automatically upgraded to HTTPS. "
        "Results are cached for 15 minutes. "
        "Use the prompt parameter to specify what information to extract."
    )
    input_schema = WebFetchInput
    _always_read_only = True
    max_result_size_chars = 100_000
    prompt = (
        "# WebFetch tool usage\n"
        "- Fetch content from a specific URL. Use after WebSearch to read a page in detail.\n"
        "- Use the `prompt` parameter to focus extraction (e.g., 'extract the API reference').\n"
        "- Results are cached for 15 minutes. Subsequent fetches of the same URL are instant.\n"
    )

    def call(self, context: ToolContext, *, url: str, prompt: str | None = None) -> str:
        # Validate URL
        error = _validate_url(url)
        if error:
            return error

        # Check cache
        cached = _url_cache.get(url)
        if cached:
            content = cached["content"]
            if prompt:
                return f"[Cached] URL: {url}\nPrompt: {prompt}\n\n{content}"
            return f"[Cached] URL: {url}\n\n{content}"

        # Upgrade HTTP to HTTPS
        parsed = urlparse(url)
        fetch_url = url
        if parsed.scheme == "http":
            fetch_url = "https" + url[4:]

        try:
            result = _fetch_with_redirects(fetch_url)

            if isinstance(result, dict) and result.get("type") == "redirect":
                return (
                    f"Redirect detected:\n"
                    f"  Original URL: {result['original_url']}\n"
                    f"  Redirects to: {result['redirect_url']}\n"
                    f"  Status: {result['status_code']}\n\n"
                    f"Call get_url_content again with the redirect URL if you want to follow it."
                )

            response = result
            content_type = response.headers.get("content-type", "").lower()

            # Reject binary content
            if any(bt in content_type for bt in BINARY_TYPES):
                return (
                    f"Error: Binary content type ({content_type}). "
                    f"Only text content is supported."
                )

            # Convert HTML to clean text
            if "text/html" in content_type:
                content = _html_to_text(response.content, url)
            elif any(t in content_type for t in ["text/", "application/json", "application/xml"]):
                content = response.text
            else:
                content = response.text[:MAX_MARKDOWN_LENGTH]

            # Truncate if needed
            if len(content) > MAX_MARKDOWN_LENGTH:
                content = content[:MAX_MARKDOWN_LENGTH] + "\n\n[Content truncated]"

            # Cache the result
            _url_cache.set(url, {"content": content, "code": response.status_code})

            # Build output
            parts = [f"URL: {url}", f"Status: {response.status_code}", ""]
            if prompt:
                parts.insert(2, f"Prompt: {prompt}")
            parts.append(content)

            return "\n".join(parts)

        except requests.exceptions.Timeout:
            return f"Error: Request timed out after {FETCH_TIMEOUT} seconds for URL: {url}"
        except requests.exceptions.ConnectionError as e:
            return f"Error: Could not connect to {url}: {e}"
        except requests.exceptions.RequestException as e:
            return f"Error fetching URL {url}: {e}"
        except Exception as e:
            return f"Unexpected error processing URL {url}: {e}"


def _validate_url(url: str) -> str | None:
    """Validate URL format. Returns error message or None if valid."""
    if len(url) > MAX_URL_LENGTH:
        return f"Error: URL too long ({len(url)} chars, max {MAX_URL_LENGTH})"

    try:
        parsed = urlparse(url)
    except Exception:
        return f"Error: Invalid URL format: {url}"

    if not parsed.scheme or not parsed.netloc:
        return f"Error: Invalid URL format: {url}"

    if parsed.username or parsed.password:
        return "Error: URLs with embedded credentials are not supported"

    hostname = parsed.hostname or ""
    if len(hostname.split(".")) < 2:
        return f"Error: Invalid hostname: {hostname}"

    return None


def _is_permitted_redirect(original_url: str, redirect_url: str) -> bool:
    """Check if a redirect is safe to follow.

    Allows: same domain (with/without www), path/query changes.
    Blocks: different protocol, port, or hostname changes.
    """
    try:
        orig = urlparse(original_url)
        redir = urlparse(redirect_url)

        if redir.scheme != orig.scheme:
            return False
        if redir.port != orig.port:
            return False
        if redir.username or redir.password:
            return False

        strip_www = lambda h: h.removeprefix("www.")
        return strip_www(orig.hostname or "") == strip_www(redir.hostname or "")
    except Exception:
        return False


def _fetch_with_redirects(url: str, depth: int = 0) -> requests.Response | dict:
    """Fetch URL with manual redirect handling for safety."""
    if depth > MAX_REDIRECTS:
        raise requests.exceptions.TooManyRedirects(f"Too many redirects (exceeded {MAX_REDIRECTS})")

    response = requests.get(
        url,
        timeout=FETCH_TIMEOUT,
        allow_redirects=False,
        headers={
            "Accept": "text/html, text/markdown, */*",
            "User-Agent": USER_AGENT,
        },
        stream=False,
    )

    if response.status_code in (301, 302, 307, 308):
        redirect_url = response.headers.get("Location", "")
        if not redirect_url:
            raise requests.exceptions.RequestException("Redirect missing Location header")

        # Resolve relative URLs
        redirect_url = requests.compat.urljoin(url, redirect_url)

        if _is_permitted_redirect(url, redirect_url):
            return _fetch_with_redirects(redirect_url, depth + 1)
        else:
            return {
                "type": "redirect",
                "original_url": url,
                "redirect_url": redirect_url,
                "status_code": response.status_code,
            }

    response.raise_for_status()
    return response


def _html_to_text(content: bytes, url: str) -> str:
    """Convert HTML to clean text using BeautifulSoup."""
    soup = BeautifulSoup(content, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text().strip() if title_tag else ""

    # Extract main content
    text = soup.get_text(separator="\n")

    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    text = "\n".join(line for line in lines if line)

    # Collapse multiple blank lines
    import re

    text = re.sub(r"\n{3,}", "\n\n", text)

    parts = []
    if title:
        parts.append(f"# {title}\n")
    parts.append(text)

    return "\n".join(parts)
