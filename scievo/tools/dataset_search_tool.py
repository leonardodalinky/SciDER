"""
Toolset for searching academic datasets across multiple repositories.
"""

import time
import urllib.parse
from dataclasses import dataclass
from typing import List, Optional

import requests

from ..core.utils import wrap_dict_to_toon
from .registry import register_tool, register_toolset_desc

register_toolset_desc(
    "dataset_search", "Search for academic datasets across multiple repositories."
)


@dataclass
class Dataset:
    name: str
    description: str
    domain: str  # e.g., "computer vision", "NLP", "speech"
    size: str  # e.g., "1.2M samples", "50GB"
    url: str
    source: str
    paper_url: Optional[str] = None
    download_url: Optional[str] = None
    license: Optional[str] = None


class DatasetRepository:
    def search(
        self, query: str, domain: Optional[str] = None, max_results: int = 10
    ) -> List[Dataset]:
        raise NotImplementedError


class PapersWithCodeRepository(DatasetRepository):
    """Search datasets from Papers with Code."""

    def search(
        self, query: str, domain: Optional[str] = None, max_results: int = 10
    ) -> List[Dataset]:
        try:
            # Papers with Code API endpoint
            base_url = "https://paperswithcode.com/api/v1/datasets/"
            params = {"search": query, "page_size": min(max_results, 50)}

            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            datasets = []
            for item in data.get("results", [])[:max_results]:
                # Extract domain/task from tags
                tags = item.get("tags", [])
                domain_str = (
                    ", ".join([tag.get("name", "") for tag in tags[:3]]) if tags else "general"
                )

                dataset = Dataset(
                    name=item.get("name", ""),
                    description=(
                        item.get("description", "")[:500]
                        if item.get("description")
                        else "No description"
                    ),
                    domain=domain_str,
                    size=item.get("size", "Unknown"),
                    url=f"https://paperswithcode.com/dataset/{item.get('name', '').lower().replace(' ', '-')}",
                    source="Papers with Code",
                    paper_url=item.get("paper", {}).get("url") if item.get("paper") else None,
                )
                datasets.append(dataset)
                time.sleep(0.3)  # Rate limiting

            return datasets
        except Exception as e:
            raise Exception(f"Error searching Papers with Code: {e}")


class HuggingFaceRepository(DatasetRepository):
    """Search datasets from Hugging Face."""

    def search(
        self, query: str, domain: Optional[str] = None, max_results: int = 10
    ) -> List[Dataset]:
        try:
            # Hugging Face API endpoint
            base_url = "https://huggingface.co/api/datasets"
            params = {"search": query, "limit": min(max_results, 50)}

            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            datasets = []
            for item in data[:max_results]:
                # Extract domain from tags
                tags = item.get("tags", [])
                domain_str = ", ".join(tags[:3]) if tags else "general"

                # Get dataset info
                dataset_id = item.get("id", "")
                if not dataset_id:
                    continue

                # Try to get more details
                try:
                    detail_url = f"https://huggingface.co/api/datasets/{dataset_id}"
                    detail_response = requests.get(detail_url, timeout=5)
                    if detail_response.status_code == 200:
                        detail_data = detail_response.json()
                        description = (
                            detail_data.get("description", "")[:500]
                            if detail_data.get("description")
                            else "No description"
                        )
                    else:
                        description = "No description available"
                except Exception:
                    description = "No description available"

                dataset = Dataset(
                    name=item.get("id", ""),
                    description=description,
                    domain=domain_str,
                    size=item.get("downloads", "Unknown"),
                    url=f"https://huggingface.co/datasets/{dataset_id}",
                    source="Hugging Face",
                    download_url=f"https://huggingface.co/datasets/{dataset_id}",
                    license=item.get("license", "Unknown"),
                )
                datasets.append(dataset)
                time.sleep(0.3)  # Rate limiting

            return datasets
        except Exception as e:
            raise Exception(f"Error searching Hugging Face: {e}")


class UCIRepository(DatasetRepository):
    """Search datasets from UCI ML Repository (using web search as fallback)."""

    def search(
        self, query: str, domain: Optional[str] = None, max_results: int = 10
    ) -> List[Dataset]:
        try:
            # UCI ML Repository doesn't have a public API, so we use web search
            # This is a simplified implementation
            base_url = "https://archive.ics.uci.edu/ml/datasets.php"
            # For now, return empty list as UCI doesn't have easy API access
            # In a full implementation, you could scrape or use their search page
            return []
        except Exception as e:
            raise Exception(f"Error searching UCI Repository: {e}")


class DatasetSearch:
    def __init__(self):
        self.repositories = {
            "paperswithcode": PapersWithCodeRepository(),
            "huggingface": HuggingFaceRepository(),
            "uci": UCIRepository(),
        }

    def search(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        domain: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dataset]:
        if sources is None:
            sources = ["paperswithcode", "huggingface"]  # Default sources

        all_datasets = []
        for source in sources:
            if source.lower() in self.repositories:
                try:
                    datasets = self.repositories[source].search(query, domain, max_results)
                    all_datasets.extend(datasets)
                except Exception as e:
                    # Silently skip sources that fail
                    continue

        # Sort by relevance (could be improved with scoring)
        return all_datasets[:max_results]


# Register the tool with the framework
@register_tool(
    "dataset_search",
    {
        "type": "function",
        "function": {
            "name": "search_datasets",
            "description": "Search for academic datasets across multiple repositories",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for dataset names/descriptions",
                    },
                    "domain": {
                        "type": "string",
                        "description": "Optional domain filter (e.g., 'computer vision', 'NLP', 'speech')",
                    },
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["paperswithcode", "huggingface", "uci"],
                            "description": "List of repositories to search",
                        },
                        "default": ["paperswithcode", "huggingface"],
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
def search_datasets(
    query: str,
    domain: Optional[str] = None,
    sources: Optional[List[str]] = None,
    max_results: int = 10,
) -> str:
    """
    Search for academic datasets across multiple repositories.

    Args:
        query: Search query for dataset names/descriptions
        domain: Optional domain filter (e.g., 'computer vision', 'NLP')
        sources: List of repositories to search (paperswithcode, huggingface, uci)
        max_results: Maximum number of results to return per source

    Returns:
        str: TOON-formatted string containing the search results
    """
    try:
        searcher = DatasetSearch()
        datasets = searcher.search(query, sources, domain, max_results)

        # Convert Dataset objects to dictionaries
        result = [
            {
                "name": dataset.name,
                "description": dataset.description,
                "domain": dataset.domain,
                "size": dataset.size,
                "url": dataset.url,
                "source": dataset.source,
                "paper_url": dataset.paper_url,
                "download_url": dataset.download_url,
                "license": dataset.license,
            }
            for dataset in datasets
        ]

        return wrap_dict_to_toon(result)
    except Exception as e:
        return wrap_dict_to_toon({"error": f"Error searching datasets: {e}"})
