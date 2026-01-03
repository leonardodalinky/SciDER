import json
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import feedparser
import requests
from loguru import logger

from ..core.utils import wrap_dict_to_toon
from .registry import register_tool, register_toolset_desc

register_toolset_desc("paper_search", "Search for academic papers across multiple repositories.")


@dataclass
class Paper:
    title: str
    authors: List[str]
    published: str
    summary: str
    url: str
    pdf_url: str
    source: str


class PaperRepository:
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        raise NotImplementedError


class ArXivRepository(PaperRepository):
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        try:
            base_url = "http://export.arxiv.org/api/query?"

            # Fix: Build query string correctly for arXiv API
            # arXiv API expects: all:"query terms" or all:term1+term2
            # Don't pre-encode the query, let urlencode handle it
            search_terms = query.strip().split()
            search_query_str = "all:" + "+".join(search_terms)

            params = {
                "search_query": search_query_str,  # Let urlencode handle encoding
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }

            query_url = base_url + urllib.parse.urlencode(params)
            logger.debug(f"arXiv search URL: {query_url}")
            response = feedparser.parse(query_url)

            # Check for parsing errors
            if hasattr(response, "bozo") and response.bozo:
                logger.warning(f"arXiv API parsing error: {response.bozo_exception}")

            papers = []

            # Check if we have entries
            if not hasattr(response, "entries") or not response.entries:
                logger.warning(f"No papers found for query: {query}")
                return papers

            for entry in response.entries:
                try:
                    # Fix: Safely get PDF URL (avoid StopIteration exception)
                    pdf_url = ""
                    for link in entry.links:
                        if link.type == "application/pdf":
                            pdf_url = link.href
                            break

                    # If no PDF found, pdf_url will be empty string (acceptable)

                    paper = Paper(
                        title=entry.title,
                        authors=[author.name for author in entry.authors],
                        published=entry.published,
                        summary=entry.summary,
                        url=entry.link,
                        pdf_url=pdf_url,  # May be empty string if no PDF available
                        source="arXiv",
                    )
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Error parsing paper entry: {e}")
                    continue  # Skip problematic entries, continue with others

                if len(papers) >= max_results:
                    break

                time.sleep(0.3)  # Rate limiting (reduced delay)

            logger.info(f"Found {len(papers)} papers for query: {query}")
            return papers
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            raise Exception(f"Error searching arXiv: {e}")


class BioRxivRepository(PaperRepository):
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        try:
            base_url = "https://api.biorxiv.org/details/biorxiv/"
            params = {"query": query, "limit": max_results, "format": "json"}

            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            papers = []
            for item in data.get("collection", [])[:max_results]:
                paper = Paper(
                    title=item.get("title", ""),
                    authors=item.get("authors", "").split("; "),
                    published=item.get("date", ""),
                    summary=item.get("abstract", ""),
                    url=f"https://doi.org/{item.get('doi', '')}",
                    pdf_url=item.get("jatsxml", "").replace(".article-meta.xml", ".full.pdf"),
                    source="bioRxiv",
                )
                papers.append(paper)

            return papers
        except Exception as e:
            raise Exception(f"Error searching bioRxiv: {e}")


class MedRxivRepository(PaperRepository):
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        try:
            base_url = "https://api.medrxiv.org/details/medrxiv/"
            params = {"query": query, "limit": max_results, "format": "json"}

            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            papers = []
            for item in data.get("collection", [])[:max_results]:
                paper = Paper(
                    title=item.get("title", ""),
                    authors=item.get("authors", "").split("; "),
                    published=item.get("date", ""),
                    summary=item.get("abstract", ""),
                    url=f"https://doi.org/{item.get('doi', '')}",
                    pdf_url=item.get("jatsxml", "").replace(".article-meta.xml", ".full.pdf"),
                    source="medRxiv",
                )
                papers.append(paper)

            return papers
        except Exception as e:
            raise Exception(f"Error searching medRxiv: {e}")


class SemanticScholarRepository(PaperRepository):
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        try:
            base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,authors,year,abstract,url,openAccessPdf,publicationDate",
            }

            headers = {"Accept": "application/json"}

            response = requests.get(base_url, params=params, headers=headers)

            # Handle rate limiting gracefully
            if response.status_code == 429:
                raise Exception(
                    "Semantic Scholar API rate limit exceeded. Please try again later or use only arXiv."
                )

            response.raise_for_status()
            data = response.json()

            papers = []
            for item in data.get("data", [])[:max_results]:
                # Extract author names
                authors = [author.get("name", "") for author in item.get("authors", [])]

                # Get PDF URL if available
                pdf_url = ""
                if item.get("openAccessPdf"):
                    pdf_url = item.get("openAccessPdf", {}).get("url", "")

                # Get publication date
                pub_date = item.get("publicationDate", "") or str(item.get("year", ""))

                paper = Paper(
                    title=item.get("title", ""),
                    authors=authors,
                    published=pub_date,
                    summary=item.get("abstract", "") or "No abstract available",
                    url=item.get("url", "")
                    or f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
                    pdf_url=pdf_url,
                    source="Semantic Scholar",
                )
                papers.append(paper)

            return papers
        except Exception as e:
            raise Exception(f"Error searching Semantic Scholar: {e}")


class PaperSearch:
    def __init__(self):
        self.repositories = {
            "arxiv": ArXivRepository(),
            "biorxiv": BioRxivRepository(),
            "medrxiv": MedRxivRepository(),
            "semanticscholar": SemanticScholarRepository(),
        }

    def search(self, query: str, sources: List[str] = None, max_results: int = 10) -> List[Paper]:
        if sources is None:
            sources = ["arxiv"]  # Default to arXiv if no sources specified

        all_papers = []
        for source in sources:
            if source.lower() in self.repositories:
                try:
                    papers = self.repositories[source].search(query, max_results)
                    all_papers.extend(papers)
                except Exception as e:
                    # Silently skip sources that fail (e.g., rate limiting)
                    # The error message is already descriptive in the exception
                    continue

        # Sort by published date (newest first)
        all_papers.sort(key=lambda x: x.published, reverse=True)
        return all_papers[:max_results]


# Register the tool with the framework
@register_tool(
    "paper_search",
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search for academic papers across multiple repositories",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for paper titles/abstracts",
                    },
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["arxiv", "biorxiv", "medrxiv", "semanticscholar"],
                            "description": "List of repositories to search (arxiv, biorxiv, medrxiv, semanticscholar)",
                        },
                        "default": ["arxiv"],
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return per source",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
)
def search_papers(query: str, sources: List[str] = None, max_results: int = 10) -> str:
    """
    Search for academic papers across multiple repositories.

    Args:
        query: Search query for paper titles/abstracts
        sources: List of repositories to search (arxiv, biorxiv, medrxiv, semanticscholar)
        max_results: Maximum number of results to return per source

    Returns:
        str: JSON string containing the search results
    """
    try:
        searcher = PaperSearch()
        papers = searcher.search(query, sources, max_results)

        # Convert Paper objects to dictionaries
        result = [
            {
                "title": paper.title,
                "authors": paper.authors,
                "published": paper.published,
                "summary": paper.summary,
                "url": paper.url,
                "pdf_url": paper.pdf_url,
                "source": paper.source,
            }
            for paper in papers
        ]

        return wrap_dict_to_toon(result)
    except Exception as e:
        logger.exception("Error searching papers")
        # Return error in TOON format to avoid parsing errors
        return wrap_dict_to_toon({"error": f"Error searching papers: {e}", "papers": []})
