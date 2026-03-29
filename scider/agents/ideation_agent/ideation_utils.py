"""
Ideation Toolset - Internal functions for research ideation through literature review.

These are utility functions used directly by the ideation agent's execution nodes,
not registered as LLM-callable tools.
"""

import json
from typing import Any, Dict, List

import requests
from loguru import logger

from scider.agents.data_agent.paper_subagent.paper_search import search_papers


def search_literature(query: str, sources: List[str] = None, max_results: int = 10) -> str:
    """
    Search for academic papers relevant to a research topic.

    Args:
        query: Search query for finding relevant papers
        sources: List of repositories to search (arxiv, semanticscholar). Defaults dynamically.
        max_results: Maximum number of papers to return

    Returns:
        JSON string containing paper metadata (title, authors, abstract, url, pdf_url, source)
    """
    try:
        result = search_papers(query=query, sources=sources, max_results=max_results)

        if isinstance(result, str) and result.startswith("Error"):
            return json.dumps({"error": result, "papers": []})

        if isinstance(result, str):
            try:
                papers = json.loads(result)
            except json.JSONDecodeError as parse_error:
                logger.warning("Failed to parse search result as JSON: {}", parse_error)
                if "Error" in result or "error" in result.lower():
                    return json.dumps({"error": result, "papers": []})
                raise
        else:
            papers = result

        if isinstance(papers, list):
            normalized_papers = []
            for paper in papers:
                normalized_paper = paper.copy()
                if "summary" in normalized_paper and "abstract" not in normalized_paper:
                    normalized_paper["abstract"] = normalized_paper["summary"]
                elif "abstract" not in normalized_paper:
                    normalized_paper["abstract"] = normalized_paper.get(
                        "summary", "No abstract available"
                    )
                normalized_papers.append(normalized_paper)
            return json.dumps(normalized_papers)

        return json.dumps(papers)
    except Exception as e:
        logger.exception("Error searching literature")
        return json.dumps({"error": f"Error searching literature: {e}", "papers": []})


def read_paper_abstract(paper_url: str) -> str:
    """
    Read the abstract and metadata of a paper from its URL.

    Args:
        paper_url: URL of the paper (arXiv URL, DOI, or other academic paper URL)

    Returns:
        Paper abstract and metadata as JSON string
    """
    try:
        if "arxiv.org" in paper_url:
            arxiv_id = None
            if "/abs/" in paper_url:
                arxiv_id = paper_url.split("/abs/")[-1]
            elif "/pdf/" in paper_url:
                arxiv_id = paper_url.split("/pdf/")[-1].replace(".pdf", "")
            elif "arxiv.org/abs/" in paper_url:
                arxiv_id = paper_url.split("arxiv.org/abs/")[-1]

            if arxiv_id:
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
                    return json.dumps(result)

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(paper_url, headers=headers, timeout=10)
        response.raise_for_status()

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(response.content, "html.parser")
        abstract = ""
        for tag in soup.find_all(["div", "section", "p"]):
            if "abstract" in tag.get("class", []) or "abstract" in tag.get("id", ""):
                abstract = tag.get_text(strip=True)
                break

        if not abstract:
            paragraphs = soup.find_all("p")
            abstract = " ".join([p.get_text(strip=True) for p in paragraphs[:3]])

        result = {
            "title": soup.find("title").get_text(strip=True) if soup.find("title") else "Unknown",
            "abstract": abstract[:2000] if abstract else "Could not extract abstract",
            "url": paper_url,
            "source": "web",
        }
        return json.dumps(result)

    except Exception as e:
        logger.exception("Error reading paper abstract")
        return json.dumps({"error": f"Error reading paper abstract: {e}", "url": paper_url})


def analyze_papers_for_ideas(papers: List[Dict[str, Any]], research_domain: str) -> str:
    """
    Analyze papers to identify research gaps and opportunities.

    Args:
        papers: List of paper objects with title, abstract, authors, published date
        research_domain: The research domain or topic of interest

    Returns:
        Analysis of research gaps and potential research directions as JSON string
    """
    try:
        if not papers:
            return "No papers provided for analysis."

        papers_text = "\n\n".join(
            [
                f"Paper {i+1}: {p.get('title', 'Unknown')}\n"
                f"Authors: {', '.join(p.get('authors', [])[:5])}\n"
                f"Published: {p.get('published', 'Unknown')}\n"
                f"URL: {p.get('url', 'N/A')}"
                for i, p in enumerate(papers[:20])
            ]
        )

        result = {
            "research_domain": research_domain,
            "papers_analyzed": len(papers),
            "summary": f"Analyzed {len(papers)} papers in the domain of {research_domain}. "
            "Use the LLM to generate detailed research ideas based on these papers.",
            "papers_text": papers_text,
        }

        return json.dumps(result)

    except Exception as e:
        logger.exception("Error analyzing papers")
        return json.dumps({"error": f"Error analyzing papers: {e}", "papers_analyzed": 0})
