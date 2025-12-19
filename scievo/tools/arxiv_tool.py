import time
import urllib.parse
import json
from typing import List, Dict, Any, Optional
import feedparser
import requests
from dataclasses import dataclass

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
            search_query = urllib.parse.quote(query)
            
            params = {
                "search_query": f"ti:{search_query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
            
            query_url = base_url + urllib.parse.urlencode(params)
            response = feedparser.parse(query_url)
            
            papers = []
            for entry in response.entries:
                paper = Paper(
                    title=entry.title,
                    authors=[author.name for author in entry.authors],
                    published=entry.published,
                    summary=entry.summary,
                    url=entry.link,
                    pdf_url=next(link.href for link in entry.links if link.type == "application/pdf"),
                    source="arXiv"
                )
                papers.append(paper)
                time.sleep(0.5)  # Rate limiting
                
            return papers
        except Exception as e:
            raise Exception(f"Error searching arXiv: {e}")

class BioRxivRepository(PaperRepository):
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        try:
            base_url = "https://api.biorxiv.org/details/biorxiv/"
            params = {
                "query": query,
                "limit": max_results,
                "format": "json"
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for item in data.get('collection', [])[:max_results]:
                paper = Paper(
                    title=item.get('title', ''),
                    authors=item.get('authors', '').split('; '),
                    published=item.get('date', ''),
                    summary=item.get('abstract', ''),
                    url=f"https://doi.org/{item.get('doi', '')}",
                    pdf_url=item.get('jatsxml', '').replace('.article-meta.xml', '.full.pdf'),
                    source="bioRxiv"
                )
                papers.append(paper)
                
            return papers
        except Exception as e:
            raise Exception(f"Error searching bioRxiv: {e}")

class MedRxivRepository(PaperRepository):
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        try:
            base_url = "https://api.medrxiv.org/details/medrxiv/"
            params = {
                "query": query,
                "limit": max_results,
                "format": "json"
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for item in data.get('collection', [])[:max_results]:
                paper = Paper(
                    title=item.get('title', ''),
                    authors=item.get('authors', '').split('; '),
                    published=item.get('date', ''),
                    summary=item.get('abstract', ''),
                    url=f"https://doi.org/{item.get('doi', '')}",
                    pdf_url=item.get('jatsxml', '').replace('.article-meta.xml', '.full.pdf'),
                    source="medRxiv"
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
                "fields": "title,authors,year,abstract,url,openAccessPdf,publicationDate"
            }
            
            headers = {
                "Accept": "application/json"
            }
            
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for item in data.get('data', [])[:max_results]:
                # Extract author names
                authors = [author.get('name', '') for author in item.get('authors', [])]
                
                # Get PDF URL if available
                pdf_url = ''
                if item.get('openAccessPdf'):
                    pdf_url = item.get('openAccessPdf', {}).get('url', '')
                
                # Get publication date
                pub_date = item.get('publicationDate', '') or str(item.get('year', ''))
                
                paper = Paper(
                    title=item.get('title', ''),
                    authors=authors,
                    published=pub_date,
                    summary=item.get('abstract', '') or 'No abstract available',
                    url=item.get('url', '') or f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
                    pdf_url=pdf_url,
                    source="Semantic Scholar"
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
            "semanticscholar": SemanticScholarRepository()
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
                    print(f"Warning: {e}")
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
                            "description": "List of repositories to search (arxiv, biorxiv, medrxiv, semanticscholar)"
                        },
                        "default": ["arxiv"]
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
        result = [{
            "title": paper.title,
            "authors": paper.authors,
            "published": paper.published,
            "summary": paper.summary,
            "url": paper.url,
            "pdf_url": paper.pdf_url,
            "source": paper.source
        } for paper in papers]
        
        return wrap_dict_to_toon(result)
    except Exception as e:
        return f"Error searching papers: {e}"
