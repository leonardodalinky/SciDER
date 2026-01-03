"""
Ideation Toolset - Tools for research ideation through literature review.

This toolset provides capabilities to:
1. Search for relevant academic papers
2. Read and extract content from papers
3. Analyze papers for research ideas
"""

from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from ..core.utils import unwrap_dict_from_toon, wrap_dict_to_toon
from .arxiv_tool import PaperSearch, search_papers
from .registry import register_tool, register_toolset_desc

register_toolset_desc(
    "ideation",
    "Tools for research ideation through literature review. Search papers, read content, and generate research ideas.",
)


@register_tool(
    "ideation",
    {
        "type": "function",
        "function": {
            "name": "search_literature",
            "description": "Search for academic papers and literature relevant to a research topic. Returns paper metadata including title, authors, abstract, and URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for finding relevant papers (e.g., 'machine learning', 'neural networks', 'transformer models')",
                    },
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["arxiv", "biorxiv", "medrxiv", "semanticscholar"],
                        },
                        "description": "List of repositories to search",
                        "default": ["arxiv"],
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of papers to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
)
def search_literature(query: str, sources: List[str] = None, max_results: int = 10) -> str:
    """
    Search for academic papers relevant to a research topic.

    Args:
        query: Search query for finding relevant papers
        sources: List of repositories to search (arxiv, biorxiv, medrxiv, semanticscholar)
        max_results: Maximum number of papers to return

    Returns:
        JSON string containing paper metadata (title, authors, abstract, url, pdf_url, source)
    """
    try:
        # Use the existing search_papers function
        result = search_papers(query=query, sources=sources, max_results=max_results)

        # Check if result is an error message (starts with "Error")
        if isinstance(result, str) and result.startswith("Error"):
            # Return error in TOON format
            return wrap_dict_to_toon({"error": result, "papers": []})

        # Normalize the result: convert 'summary' to 'abstract' for consistency

        try:
            papers = unwrap_dict_from_toon(result)
        except (ValueError, Exception) as parse_error:
            # If TOON parsing fails, it might be an error message
            logger.warning("Failed to parse search result as TOON: {}", parse_error)
            if isinstance(result, str) and ("Error" in result or "error" in result.lower()):
                return wrap_dict_to_toon({"error": result, "papers": []})
            # Re-raise if it's not an error message
            raise

        if isinstance(papers, list):
            # Normalize each paper: ensure 'abstract' field exists
            normalized_papers = []
            for paper in papers:
                normalized_paper = paper.copy()
                # If paper has 'summary' but no 'abstract', copy it
                if "summary" in normalized_paper and "abstract" not in normalized_paper:
                    normalized_paper["abstract"] = normalized_paper["summary"]
                # If paper has neither, set empty string
                elif "abstract" not in normalized_paper:
                    normalized_paper["abstract"] = normalized_paper.get(
                        "summary", "No abstract available"
                    )
                normalized_papers.append(normalized_paper)
            return wrap_dict_to_toon(normalized_papers)

        # If result is not a list, return as is (shouldn't happen, but be defensive)
        return result
    except Exception as e:
        logger.exception("Error searching literature")
        return wrap_dict_to_toon({"error": f"Error searching literature: {e}", "papers": []})


@register_tool(
    "ideation",
    {
        "type": "function",
        "function": {
            "name": "read_paper_abstract",
            "description": "Read the abstract and metadata of a paper from its URL. For arXiv papers, extracts the abstract from the paper page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_url": {
                        "type": "string",
                        "description": "URL of the paper (arXiv URL, DOI, or other academic paper URL)",
                    },
                },
                "required": ["paper_url"],
            },
        },
    },
)
def read_paper_abstract(paper_url: str) -> str:
    """
    Read the abstract and metadata of a paper from its URL.

    Args:
        paper_url: URL of the paper (arXiv URL, DOI, or other academic paper URL)

    Returns:
        Paper abstract and metadata
    """
    try:
        # Handle arXiv URLs
        if "arxiv.org" in paper_url:
            # Extract arXiv ID
            arxiv_id = None
            if "/abs/" in paper_url:
                arxiv_id = paper_url.split("/abs/")[-1]
            elif "/pdf/" in paper_url:
                arxiv_id = paper_url.split("/pdf/")[-1].replace(".pdf", "")
            elif "arxiv.org/abs/" in paper_url:
                arxiv_id = paper_url.split("arxiv.org/abs/")[-1]

            if arxiv_id:
                # Use arXiv API to get abstract
                import urllib.parse

                import feedparser

                api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
                response = feedparser.parse(api_url)

                if response.entries:
                    entry = response.entries[0]
                    result = {
                        "title": entry.title,
                        "authors": [author.name for author in entry.authors],
                        "published": entry.published,
                        "abstract": entry.summary,
                        "url": entry.link,
                        "source": "arXiv",
                    }
                    return wrap_dict_to_toon(result)

        # For other URLs, try to fetch and parse HTML
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(paper_url, headers=headers, timeout=10)
        response.raise_for_status()

        # Try to extract abstract from HTML (basic extraction)
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(response.content, "html.parser")
        # Look for common abstract patterns
        abstract = ""
        for tag in soup.find_all(["div", "section", "p"]):
            if "abstract" in tag.get("class", []) or "abstract" in tag.get("id", ""):
                abstract = tag.get_text(strip=True)
                break

        if not abstract:
            # Fallback: return first few paragraphs
            paragraphs = soup.find_all("p")
            abstract = " ".join([p.get_text(strip=True) for p in paragraphs[:3]])

        result = {
            "title": soup.find("title").get_text(strip=True) if soup.find("title") else "Unknown",
            "abstract": abstract[:2000] if abstract else "Could not extract abstract",
            "url": paper_url,
            "source": "web",
        }
        return wrap_dict_to_toon(result)

    except Exception as e:
        logger.exception("Error reading paper abstract")
        return wrap_dict_to_toon({"error": f"Error reading paper abstract: {e}", "url": paper_url})


@register_tool(
    "ideation",
    {
        "type": "function",
        "function": {
            "name": "analyze_papers_for_ideas",
            "description": "Analyze a collection of papers to identify research gaps, opportunities, and potential research directions. Takes a list of paper summaries and generates research ideas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "papers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "abstract": {"type": "string"},
                                "authors": {"type": "array", "items": {"type": "string"}},
                                "published": {"type": "string"},
                            },
                        },
                        "description": "List of paper objects with title, abstract, authors, and published date",
                    },
                    "research_domain": {
                        "type": "string",
                        "description": "The research domain or topic of interest (e.g., 'machine learning', 'natural language processing')",
                    },
                },
                "required": ["papers", "research_domain"],
            },
        },
    },
)
def analyze_papers_for_ideas(papers: List[Dict[str, Any]], research_domain: str) -> str:
    """
    Analyze papers to identify research gaps and opportunities.

    Args:
        papers: List of paper objects with title, abstract, authors, published date
        research_domain: The research domain or topic of interest

    Returns:
        Analysis of research gaps and potential research directions
    """
    try:
        if not papers:
            return "No papers provided for analysis."

        # Format papers for analysis (without abstracts)
        papers_text = "\n\n".join(
            [
                f"Paper {i+1}: {p.get('title', 'Unknown')}\n"
                f"Authors: {', '.join(p.get('authors', [])[:5])}\n"
                f"Published: {p.get('published', 'Unknown')}\n"
                f"URL: {p.get('url', 'N/A')}"
                for i, p in enumerate(papers[:20])  # Limit to 20 papers
            ]
        )

        result = {
            "research_domain": research_domain,
            "papers_analyzed": len(papers),
            "summary": f"Analyzed {len(papers)} papers in the domain of {research_domain}. "
            "Use the LLM to generate detailed research ideas based on these papers.",
            "papers_text": papers_text,
        }

        return wrap_dict_to_toon(result)

    except Exception as e:
        logger.exception("Error analyzing papers")
        return wrap_dict_to_toon({"error": f"Error analyzing papers: {e}", "papers_analyzed": 0})
