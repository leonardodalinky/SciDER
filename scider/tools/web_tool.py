"""
Toolset for web search and web access.
"""

from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from jinja2 import Template

from .registry import register_tool, register_toolset_desc

register_toolset_desc(
    "web",
    "Web toolset for searching the web and fetching content from URLs. Should be only used if it is indeed needed.",
)

WEB_SEARCH_TEMPLATE: Template = Template(
    """\
Results for web search query '{{ query }}':

{% for result in results %}
=== Web Result {{ loop.index }} ===
    {% for key, value in result.items() %}
{{ key }}: {{ value }}
    {%- endfor %}
{% endfor %}
"""
)


@register_tool(
    "web",
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Perform a general web search and return the top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to perform"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
)
def web_search(query: str, max_results: int = 5) -> str:
    """
    Perform a general web search and return the top results.
    """
    from ddgs import DDGS

    try:
        results: list[dict[str, Any]] = DDGS().text(query, max_results=max_results)
        output_text = WEB_SEARCH_TEMPLATE.render(query=query, results=results)
    except Exception as e:
        output_text = "web_search tool error: " + str(e)
    return output_text


@register_tool(
    "web",
    {
        "type": "function",
        "function": {
            "name": "get_url_content",
            "description": "Fetch and extract textual content from a web URL. Supports HTML and other textual content, but excludes binary files like PDFs, ZIP files, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch content from"},
                    "timeout": {
                        "type": "integer",
                        "description": "Request timeout in seconds",
                        "default": 10,
                    },
                },
                "required": ["url"],
            },
        },
    },
)
def get_url_content(url: str, timeout: int = 10) -> str:
    """
    Fetch textual content from a web URL.

    Args:
        url: The URL to fetch content from
        timeout: Request timeout in seconds

    Returns:
        The textual content of the web page
    """
    try:
        # Parse URL to validate it
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return f"Error: Invalid URL format: {url}"

        # Set headers to mimic a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # Make the request
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        # Check content type to ensure it's textual
        content_type = response.headers.get("content-type", "").lower()

        # Reject binary file types
        binary_types = ["pdf", "zip", "rar", "7z", "tar", "gz", "exe", "dmg", "pkg", "deb", "rpm"]
        if any(binary_type in content_type for binary_type in binary_types):
            return f"Error: Binary content type detected ({content_type}). Only textual content is supported."

        # Handle HTML content
        if "text/html" in content_type:
            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract title
            title = soup.find("title")
            title_text = title.get_text().strip() if title else "No title"

            # Extract main content
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            return f"Title: {title_text}\n\nContent:\n{text}"

        # Handle other textual content types
        elif any(
            text_type in content_type
            for text_type in ["text/", "application/json", "application/xml"]
        ):
            return f"Content from {url}:\n\n{response.text}"

        else:
            return f"Warning: Unknown content type ({content_type}). Attempting to extract as text:\n\n{response.text[:5000]}{'...' if len(response.text) > 5000 else ''}"

    except requests.exceptions.Timeout:
        return f"Error: Request timeout after {timeout} seconds for URL: {url}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL {url}: {str(e)}"
    except Exception as e:
        return f"Unexpected error processing URL {url}: {str(e)}"
