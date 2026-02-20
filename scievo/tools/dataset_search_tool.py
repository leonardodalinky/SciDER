"""
Toolset for searching academic datasets across multiple repositories.
Supports searching based on data agent's dataset analysis.
"""

import json
import re
import time
from dataclasses import dataclass
from typing import List, Optional

import requests
from loguru import logger

# Delay import of ModelRegistry to avoid circular import
# from ..core.llms import ModelRegistry  # Moved to function level
# Message is imported here but only used in functions that have lazy ModelRegistry import
from ..core.types import Message
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
            # Papers with Code API endpoint - try different formats
            # The API might require different endpoint or parameters
            base_url = "https://paperswithcode.com/api/v1/datasets/"
            params = {"search": query, "page_size": min(max_results, 50)}

            # Add headers to mimic browser request
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "application/json",
            }

            logger.debug(f"Papers with Code search URL: {base_url} with params: {params}")
            response = requests.get(base_url, params=params, headers=headers, timeout=10)

            # Check response status
            if response.status_code != 200:
                logger.warning(
                    f"Papers with Code API returned status {response.status_code}: {response.text[:200]}"
                )
                return []

            # Check if response is empty
            if not response.text or not response.text.strip():
                logger.warning("Papers with Code API returned empty response")
                return []

            # Check content type - Papers with Code API might return HTML
            content_type = response.headers.get("content-type", "").lower()
            if "application/json" not in content_type and "text/json" not in content_type:
                # Papers with Code API appears to have changed or requires authentication
                # Return empty list gracefully
                logger.warning(
                    f"Papers with Code API returned HTML instead of JSON. "
                    f"This may indicate the API endpoint has changed or requires authentication. "
                    f"Status: {response.status_code}, Content-Type: {content_type}"
                )
                # Don't log the full HTML response as it's too large
                return []

            # Check if response is valid JSON
            try:
                data = response.json()
            except ValueError as e:
                logger.error(f"Invalid JSON response from Papers with Code: {e}")
                logger.debug(f"Response text (first 500 chars): {response.text[:500]}")
                return []

            # Check if data is a dictionary with "results" key
            if not isinstance(data, dict):
                logger.warning(f"Unexpected response format from Papers with Code: {type(data)}")
                return []

            results = data.get("results", [])
            if not isinstance(results, list):
                logger.warning(f"Unexpected results format: {type(results)}")
                return []

            if not results:
                logger.warning(f"No datasets found in Papers with Code for query: {query}")
                return []

            datasets = []
            for item in results[:max_results]:
                try:
                    # Extract domain/task from tags
                    tags = item.get("tags", [])
                    domain_str = (
                        ", ".join([tag.get("name", "") for tag in tags[:3]]) if tags else "general"
                    )

                    # Filter by domain if specified
                    if domain and domain.lower() not in domain_str.lower():
                        continue

                    name = item.get("name", "")
                    if not name:
                        continue  # Skip items without name

                    dataset = Dataset(
                        name=name,
                        description=(
                            item.get("description", "")[:500]
                            if item.get("description")
                            else "No description"
                        ),
                        domain=domain_str,
                        size=item.get("size", "Unknown"),
                        url=f"https://paperswithcode.com/dataset/{name.lower().replace(' ', '-')}",
                        source="Papers with Code",
                        paper_url=item.get("paper", {}).get("url") if item.get("paper") else None,
                    )
                    datasets.append(dataset)
                except Exception as e:
                    logger.warning(f"Error parsing dataset item from Papers with Code: {e}")
                    continue  # Skip problematic items

                time.sleep(0.2)  # Rate limiting (reduced delay)

            logger.info(f"Found {len(datasets)} datasets from Papers with Code for query: {query}")
            return datasets
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error searching Papers with Code: {e}")
            return []  # Return empty list instead of raising exception
        except Exception as e:
            logger.error(f"Error searching Papers with Code: {e}")
            return []  # Return empty list instead of raising exception


class HuggingFaceRepository(DatasetRepository):
    """Search datasets from Hugging Face."""

    def search(
        self, query: str, domain: Optional[str] = None, max_results: int = 10
    ) -> List[Dataset]:
        try:
            # Hugging Face API endpoint
            # Try multiple query formats for better results
            base_url = "https://huggingface.co/api/datasets"

            # Add headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "application/json",
            }

            # Try different query formats - split complex queries into simpler ones
            queries_to_try = []

            # 1. Original query
            queries_to_try.append(query)

            # 2. Lowercase version
            queries_to_try.append(query.lower())

            # 3. Extract key terms (remove common words)
            words = query.lower().split()
            stop_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "as",
                "is",
                "are",
                "was",
                "were",
                "neural",
                "networks",
            }
            key_terms = [w for w in words if w not in stop_words and len(w) > 2]
            if key_terms and " ".join(key_terms) not in queries_to_try:
                queries_to_try.append(" ".join(key_terms))

            # 4. Single most important term (usually the first non-stop word)
            if key_terms:
                queries_to_try.append(key_terms[0])

            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in queries_to_try:
                if q and q not in seen:
                    seen.add(q)
                    unique_queries.append(q)
            queries_to_try = unique_queries

            all_datasets = []

            # Try up to 3 query formats to maximize results
            for search_query in queries_to_try[:3]:
                params = {"search": search_query, "limit": min(max_results * 2, 50)}

                logger.debug(f"Hugging Face search URL: {base_url} with params: {params}")
                response = requests.get(base_url, params=params, headers=headers, timeout=10)

                # Check response status
                if response.status_code != 200:
                    logger.debug(
                        f"Hugging Face API returned status {response.status_code} for query '{search_query}'"
                    )
                    continue  # Try next query format

                # Check if response is empty
                if not response.text or not response.text.strip():
                    logger.debug(
                        f"Hugging Face API returned empty response for query '{search_query}'"
                    )
                    continue

                # Check content type
                content_type = response.headers.get("content-type", "").lower()
                if "application/json" not in content_type and "text/json" not in content_type:
                    logger.debug(
                        f"Hugging Face API returned non-JSON content type: {content_type} for query '{search_query}'"
                    )
                    continue

                # Check if response is valid JSON
                try:
                    data = response.json()
                except ValueError as e:
                    logger.debug(
                        f"Invalid JSON response from Hugging Face for query '{search_query}': {e}"
                    )
                    continue

                # Check if data is a list
                if not isinstance(data, list):
                    logger.debug(f"Unexpected response format from Hugging Face: {type(data)}")
                    continue

                if data:
                    logger.info(
                        f"Found {len(data)} datasets from Hugging Face with query '{search_query}'"
                    )
                    all_datasets.extend(data)
                    if len(all_datasets) >= max_results * 2:
                        break  # Got enough results
                else:
                    logger.debug(f"No datasets found in Hugging Face for query '{search_query}'")

            if not all_datasets:
                logger.warning(
                    f"No datasets found in Hugging Face for any query variant of: {query}"
                )
                return []

            # Deduplicate by dataset ID
            seen_ids = set()
            unique_datasets = []
            for item in all_datasets:
                dataset_id = item.get("id", "")
                if dataset_id and dataset_id not in seen_ids:
                    seen_ids.add(dataset_id)
                    unique_datasets.append(item)
                    if len(unique_datasets) >= max_results * 2:
                        break

            datasets = []
            for item in unique_datasets:
                try:
                    # Extract domain from tags
                    tags = item.get("tags", [])
                    if not isinstance(tags, list):
                        tags = []
                    domain_str = ", ".join(tags[:3]) if tags else "general"

                    # Filter by domain if specified
                    if domain and domain.lower() not in domain_str.lower():
                        continue

                    # Get dataset info
                    dataset_id = item.get("id", "")
                    if not dataset_id:
                        continue

                    # Try to get more details (but don't fail if it doesn't work)
                    description = "No description available"
                    try:
                        detail_url = f"https://huggingface.co/api/datasets/{dataset_id}"
                        detail_response = requests.get(detail_url, timeout=5)
                        if detail_response.status_code == 200:
                            detail_data = detail_response.json()
                            if isinstance(detail_data, dict):
                                description = (
                                    detail_data.get("description", "")[:500]
                                    if detail_data.get("description")
                                    else "No description"
                                )
                    except Exception as e:
                        logger.debug(f"Could not fetch details for {dataset_id}: {e}")
                        # Use basic description from search result if available
                        description = (
                            item.get("description", "")[:500]
                            if item.get("description")
                            else "No description available"
                        )

                    dataset = Dataset(
                        name=dataset_id,
                        description=description,
                        domain=domain_str,
                        size=str(item.get("downloads", "Unknown")),
                        url=f"https://huggingface.co/datasets/{dataset_id}",
                        source="Hugging Face",
                        download_url=f"https://huggingface.co/datasets/{dataset_id}",
                        license=item.get("license") or "Unknown",
                    )
                    datasets.append(dataset)

                    if len(datasets) >= max_results:
                        break

                    time.sleep(0.2)  # Rate limiting (reduced delay)
                except Exception as e:
                    logger.warning(f"Error parsing dataset item from Hugging Face: {e}")
                    continue  # Skip problematic items

            logger.info(f"Found {len(datasets)} datasets from Hugging Face for query: {query}")
            return datasets
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error searching Hugging Face: {e}")
            return []  # Return empty list instead of raising exception
        except Exception as e:
            logger.error(f"Error searching Hugging Face: {e}")
            return []  # Return empty list instead of raising exception


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


def extract_dataset_features_from_summary(data_summary: str) -> dict:
    """
    Extract dataset features from data agent's analysis summary.

    Returns a dictionary with:
    - task_type: classification, regression, etc.
    - columns: list of column names
    - data_types: types of data
    - domain: inferred domain
    - keywords: key terms for search
    """
    if not data_summary:
        return {}

    features = {
        "task_type": None,
        "columns": [],
        "data_types": [],
        "domain": None,
        "keywords": [],
    }

    summary_lower = data_summary.lower()

    # Extract task type
    if any(term in summary_lower for term in ["classification", "classify", "class"]):
        features["task_type"] = "classification"
    elif any(term in summary_lower for term in ["regression", "predict", "forecast"]):
        features["task_type"] = "regression"
    elif any(term in summary_lower for term in ["clustering", "cluster"]):
        features["task_type"] = "clustering"

    # Extract column names (look for patterns like "columns:", "features:", etc.)
    column_patterns = [
        r"columns?[:\s]+([^\n]+)",
        r"features?[:\s]+([^\n]+)",
        r"variables?[:\s]+([^\n]+)",
    ]
    for pattern in column_patterns:
        matches = re.findall(pattern, data_summary, re.IGNORECASE)
        if matches:
            # Split by comma, semicolon, or newline
            cols = re.split(r"[,;\n]", matches[0])
            features["columns"] = [col.strip() for col in cols if col.strip()][:10]  # Limit to 10
            break

    # Extract data types
    if "image" in summary_lower or "picture" in summary_lower or "photo" in summary_lower:
        features["data_types"].append("image")
    if "text" in summary_lower or "nlp" in summary_lower or "language" in summary_lower:
        features["data_types"].append("text")
    if "audio" in summary_lower or "sound" in summary_lower or "speech" in summary_lower:
        features["data_types"].append("audio")
    if "video" in summary_lower:
        features["data_types"].append("video")
    if "tabular" in summary_lower or "csv" in summary_lower or "table" in summary_lower:
        features["data_types"].append("tabular")

    # Extract domain keywords
    domain_keywords = []
    if any(
        term in summary_lower for term in ["vision", "image", "object detection", "segmentation"]
    ):
        features["domain"] = "computer vision"
        domain_keywords.extend(["vision", "image", "cv"])
    if any(term in summary_lower for term in ["nlp", "language", "text", "translation", "bert"]):
        features["domain"] = "nlp"
        domain_keywords.extend(["nlp", "language", "text"])
    if any(term in summary_lower for term in ["speech", "audio", "voice"]):
        features["domain"] = "speech"
        domain_keywords.extend(["speech", "audio"])

    # Build keywords from columns and task type
    keywords = []
    if features["task_type"]:
        keywords.append(features["task_type"])
    keywords.extend(domain_keywords)
    # Add first few column names as keywords (if they're meaningful)
    for col in features["columns"][:3]:
        # Skip generic column names
        if col.lower() not in ["id", "index", "target", "label", "y", "x"]:
            keywords.append(col.lower())

    features["keywords"] = list(set(keywords))  # Remove duplicates

    return features


def build_search_query_from_data_summary(
    data_summary: str, user_query: Optional[str] = None
) -> str:
    """
    Build a search query from data summary using LLM to extract relevant dataset characteristics.

    This generates a query that describes the dataset characteristics rather than ML keywords.
    """
    if not data_summary:
        return user_query or ""

    # Extract features
    features = extract_dataset_features_from_summary(data_summary)

    # Use LLM to generate a dataset-focused search query
    prompt = f"""Based on the following data analysis summary, generate a search query to find similar academic datasets.

Data Analysis Summary:
{data_summary[:2000]}  # Limit to avoid token limits

Extracted Features:
- Task Type: {features.get('task_type', 'unknown')}
- Columns: {', '.join(features.get('columns', [])[:5])}
- Data Types: {', '.join(features.get('data_types', []))}
- Domain: {features.get('domain', 'unknown')}

Generate a concise search query (2-5 key terms) that describes the dataset characteristics.
Focus on:
1. The type of data (e.g., "iris flowers", "house prices", "image classification")
2. The task domain (e.g., "computer vision", "NLP", "tabular data")
3. Key characteristics (e.g., "multiclass classification", "regression")

Return ONLY the search query text, no explanation:"""

    try:
        # Lazy import to avoid circular dependency
        from ..core.llms import ModelRegistry

        # Try to use dataset_search model, fallback to data model
        try:
            model_name = "dataset_search"
            ModelRegistry.instance().get_model_params(model_name)
        except ValueError:
            try:
                model_name = "data"
                ModelRegistry.instance().get_model_params(model_name)
            except ValueError:
                logger.warning(
                    "No suitable model found for query generation, using feature-based query"
                )
                if features["keywords"]:
                    return " ".join(features["keywords"][:5])
                return user_query or ""

        msg = ModelRegistry.completion(
            model_name,
            [Message(role="user", content=prompt)],
            system_prompt="You are an expert at generating dataset search queries. Return only the query text.",
            agent_sender="dataset_search",
            tools=None,
        )

        query = msg.content.strip().strip('"').strip("'")

        # Fallback to feature-based query if LLM fails
        if not query or len(query) < 3:
            if features["keywords"]:
                query = " ".join(features["keywords"][:5])
            else:
                query = user_query or ""

        logger.info(f"Generated dataset search query from data summary: {query}")
        return query
    except Exception as e:
        logger.warning(f"Failed to generate query from data summary: {e}")
        # Fallback to feature-based query
        if features["keywords"]:
            return " ".join(features["keywords"][:5])
        return user_query or ""


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
        data_summary: Optional[str] = None,
    ) -> List[Dataset]:
        # If data_summary is provided, use it to build a better query
        if data_summary:
            query = build_search_query_from_data_summary(data_summary, query)
            logger.info(f"Using data-summary-based query: {query}")

        if sources is None:
            sources = ["paperswithcode", "huggingface"]  # Default sources

        all_datasets = []
        for source in sources:
            if source.lower() in self.repositories:
                try:
                    datasets = self.repositories[source].search(query, domain, max_results)
                    if datasets:
                        all_datasets.extend(datasets)
                        logger.debug(f"Source {source} returned {len(datasets)} datasets")
                    else:
                        logger.debug(f"Source {source} returned 0 datasets")
                except Exception as e:
                    # Log error but continue with other sources
                    logger.warning(f"Error searching {source}: {e}")
                    continue

        # Sort by relevance (could be improved with scoring)
        logger.info(f"Total datasets found: {len(all_datasets)} from {len(sources)} sources")
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
                        "description": "Search query for dataset names/descriptions (fallback if data_summary not provided)",
                    },
                    "data_summary": {
                        "type": "string",
                        "description": "Optional data analysis summary from data agent. If provided, will extract dataset features and search for similar datasets.",
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
    data_summary: Optional[str] = None,
) -> str:
    """
    Search for academic datasets across multiple repositories.

    If data_summary is provided, extracts dataset features from the data analysis
    and searches for similar datasets based on characteristics rather than ML keywords.

    Args:
        query: Search query for dataset names/descriptions (used as fallback)
        domain: Optional domain filter (e.g., 'computer vision', 'NLP')
        sources: List of repositories to search (paperswithcode, huggingface, uci)
        max_results: Maximum number of results to return per source
        data_summary: Optional data analysis summary from data agent. If provided,
                     will extract dataset features and generate a better search query.

    Returns:
        str: TOON-formatted string containing the search results
    """
    try:
        searcher = DatasetSearch()
        datasets = searcher.search(query, sources, domain, max_results, data_summary)

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

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Error searching datasets: {e}"})
