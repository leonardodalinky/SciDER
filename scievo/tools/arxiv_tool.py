import time
import urllib.parse

import feedparser

from scievo.core.types import GraphState

from ..core.utils import wrap_dict_to_toon
from .registry import register_tool, register_toolset_desc

register_toolset_desc("arxiv", "ArXiv paper search toolset.")


@register_tool(
    "arxiv",
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": "Search ArXiv papers by query keyword",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword for paper titles",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
)
def search_arxiv(graph_state: GraphState, query: str, max_results: int = 10) -> str:
    try:
        # Build API URL
        base_url = "http://export.arxiv.org/api/query?"
        search_query = urllib.parse.quote(query)

        # Set API parameters
        params = {
            "search_query": f"ti:{search_query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        # Build complete query URL
        query_url = base_url + urllib.parse.urlencode(params)

        # Send request and parse results
        response = feedparser.parse(query_url)

        # Extract paper information
        papers = []
        for entry in response.entries:
            paper = {
                "title": entry.title,
                "author": [author.name for author in entry.authors],
                "published": entry.published,
                "summary": entry.summary,
                "url": entry.link,
                "pdf_url": next(
                    link.href for link in entry.links if link.type == "application/pdf"
                ),
            }
            papers.append(paper)

            # Follow API rate limit
            time.sleep(0.5)

        return wrap_dict_to_toon(papers)
    except Exception as e:
        return f"Error searching ArXiv: {e}"
