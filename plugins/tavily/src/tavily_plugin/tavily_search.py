import os
from typing import Any

from openjiuwen.core.foundation.tool import tool
from tavily import TavilyClient


class TavilySearch:
    """
    A class for performing search operations using the Tavily Search API.

    Args:
        api_key (str): The API key for accessing the Tavily Search API.
        project_id (str, optional): The project ID for tracking and analytics.

    Methods:
        search: Retrieves search results from the Tavily Search API.
    """

    def __init__(self, api_key: str | None = None, project_id: str | None = None) -> None:
        self.client = TavilyClient(api_key=api_key, project_id=project_id)

    def search(self, params: dict[str, Any]) -> dict:
        """
        Retrieves search results from the Tavily Search API.

        Args:
            params (Dict[str, Any]): The search parameters.

        Returns:
            dict: The search results.
        """
        if "api_key" in params:
            del params["api_key"]
        if "project_id" in params:
            del params["project_id"]
        processed_params = self._process_params(params)
        return self.client.search(**processed_params)

    @staticmethod
    def _process_params(params: dict[str, Any]) -> dict:
        """
        Processes and validates the search parameters.

        Args:
            params (Dict[str, Any]): The search parameters.

        Returns:
            dict: The processed parameters.
        """
        processed_params = {}
        for key, value in params.items():
            if value is None or value == "None" or value == "not_specified":
                continue
            if key in ["include_domains", "exclude_domains"]:
                if isinstance(value, str):
                    processed_params[key] = [domain.strip() for domain in value.replace(",", " ").split()]
                else:
                    processed_params[key] = value
            elif key in [
                "include_images",
                "include_image_descriptions",
                "auto_parameters",
                "include_favicon",
                "include_usage",
            ]:
                if isinstance(value, str):
                    processed_params[key] = value.lower() == "true"
                else:
                    processed_params[key] = bool(value)
            elif key == "include_answer":
                if isinstance(value, str):
                    if value.lower() in ["false", ""]:
                        continue
                    elif value.lower() in ["true", "basic"]:
                        processed_params[key] = "basic"
                    elif value.lower() == "advanced":
                        processed_params[key] = "advanced"
                    else:
                        processed_params[key] = value.lower() == "true"
                elif value:
                    processed_params[key] = "basic"
            elif key == "include_raw_content":
                if isinstance(value, str):
                    if value.lower() in ["false", ""]:
                        continue
                    elif value.lower() in ["true", "markdown"]:
                        processed_params[key] = "markdown"
                    elif value.lower() == "text":
                        processed_params[key] = "text"
                    else:
                        processed_params[key] = value.lower() == "true"
                elif value:
                    processed_params[key] = "markdown"
            elif key in ["max_results", "chunks_per_source"]:
                if isinstance(value, str):
                    processed_params[key] = int(value)
                else:
                    processed_params[key] = value
            elif key in [
                "search_depth",
                "topic",
                "query",
                "time_range",
                "country",
                "start_date",
                "end_date",
            ]:
                processed_params[key] = value
            else:
                pass
        processed_params.setdefault("search_depth", "basic")
        processed_params.setdefault("topic", "general")
        processed_params.setdefault("max_results", 5)
        if processed_params.get("search_depth") == "advanced":
            processed_params.setdefault("chunks_per_source", 3)
        return processed_params


def _format_results_as_text(search_results: dict, tool_parameters: dict[str, Any]) -> str:
    """
    Formats the search results into markdown text based on user-selected parameters.

    Args:
        search_results (dict): The search results.
        tool_parameters (dict): The tool parameters selected by the user.

    Returns:
        str: The formatted markdown text.
    """
    output_lines: list[str] = []
    include_answer = tool_parameters.get("include_answer", False)
    if include_answer and include_answer not in [False, "false"] and search_results.get("answer"):
        output_lines.append(f"**Answer:** {search_results['answer']}\n")

    if "results" in search_results:
        for idx, result in enumerate(search_results["results"], 1):
            title = result.get("title", "No Title")
            url = result.get("url", "")
            content = result.get("content", "")
            published_date = result.get("published_date", "")
            score = result.get("score", "")
            output_lines.append(f"# Result {idx}: [{title}]({url})\n")
            if tool_parameters.get("topic") == "news" and published_date:
                output_lines.append(f"**Published Date:** {published_date}\n")
            output_lines.append(f"**URL:** {url}\n")
            if score:
                output_lines.append(f"**Relevance Score:** {score}\n")
            if tool_parameters.get("include_favicon", False) and result.get("favicon"):
                output_lines.append(f"**Favicon:** ![Favicon]({result['favicon']})\n")
            if content:
                output_lines.append(f"**Content:**\n{content}\n")
            include_raw_content = tool_parameters.get("include_raw_content", False)
            if include_raw_content and include_raw_content not in [False, "false"] and result.get("raw_content"):
                output_lines.append(f"**Raw Content:**\n{result['raw_content']}\n")
            output_lines.append("---\n")

    if tool_parameters.get("include_images", False) and search_results.get("images"):
        output_lines.append("**Images:**\n")
        for image in search_results["images"]:
            if isinstance(image, dict):
                image_url = image.get("url")
                description = image.get("description", "Tavily search result image")
            else:
                image_url = image
                description = "Tavily search result image"
            if image_url:
                output_lines.append(f"![{description}]({image_url})\n")
        output_lines.append("\n")

    return "\n".join(output_lines)


@tool(
    name="tavily_search",
    description=(
        "专为 AI 代理构建的搜索引擎，提供实时、准确的结果。"
        "A search engine built for AI agents (LLMs), delivering real-time, accurate, and factual results."
    ),
    input_params={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query you want to execute with Tavily."},
            "search_depth": {
                "type": "string",
                "description": (
                    "Depth of search: 'basic' (standard), 'advanced' (comprehensive), 'fast', 'ultra-fast'."
                ),
                "enum": ["basic", "advanced", "fast", "ultra-fast"],
                "default": "basic",
            },
            "topic": {
                "type": "string",
                "description": "Category of the search: 'general', 'news', or 'finance'.",
                "enum": ["general", "news", "finance"],
                "default": "general",
            },
            "time_range": {
                "type": "string",
                "description": "Time range to filter results: 'not_specified', 'day', 'week', 'month', 'year'.",
                "enum": ["not_specified", "day", "week", "month", "year"],
                "default": "not_specified",
            },
            "start_date": {"type": "string", "description": "Start date for filtering results (YYYY-MM-DD)."},
            "end_date": {"type": "string", "description": "End date for filtering results (YYYY-MM-DD)."},
            "country": {
                "type": "string",
                "description": (
                    "Boost results from a specific country (e.g. 'china', 'us'). " "Only when topic is general."
                ),
            },
            "auto_parameters": {
                "type": "boolean",
                "description": "Automatically configure search parameters based on query. Default false.",
                "default": False,
            },
            "include_images": {
                "type": "boolean",
                "description": "Include query-related images in the response. Default false.",
                "default": False,
            },
            "include_image_descriptions": {
                "type": "boolean",
                "description": "When include_images is true, add descriptions for each image. Default false.",
                "default": False,
            },
            "include_favicon": {
                "type": "boolean",
                "description": "Include favicon URL for each result. Default false.",
                "default": False,
            },
            "include_answer": {
                "type": "string",
                "description": "Include a short answer: false, true/basic, or advanced for detailed answer.",
                "enum": ["false", "true", "basic", "advanced"],
                "default": "false",
            },
            "include_raw_content": {
                "type": "string",
                "description": "Include raw content per result: false, true/markdown, or text.",
                "enum": ["false", "true", "markdown", "text"],
                "default": "false",
            },
            "include_domains": {
                "type": "string",
                "description": "Comma- or space-separated list of domains to include in search.",
            },
            "exclude_domains": {
                "type": "string",
                "description": "Comma- or space-separated list of domains to exclude from search.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results to return (1-20). Default 5.",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
            },
            "chunks_per_source": {
                "type": "integer",
                "description": "Max content chunks per source (1-3). Only when search_depth is advanced. Default 3.",
                "minimum": 1,
                "maximum": 3,
                "default": 3,
            },
            "include_usage": {
                "type": "boolean",
                "description": "Include API usage (token counts, credits) in the response. Default false.",
                "default": False,
            },
            "project_id": {
                "type": "string",
                "description": "Optional project ID for tracking. Sent as X-Project-ID header.",
            },
        },
        "required": ["query"],
    },
)
def tavily_search(params: dict[str, Any] | None = None, **kwargs) -> dict:
    """
    Invokes the Tavily Search tool with the given parameters.

    Args:
        params (Dict[str, Any]): The parameters for the Tavily Search tool.
            - query: Required. The search query.
            - search_depth: Optional. basic / advanced / fast / ultra-fast.
            - topic: Optional. general / news / finance.
            - time_range, start_date, end_date, country: Optional filters.
            - auto_parameters, include_images, include_image_descriptions, include_favicon: Optional booleans.
            - include_answer: Optional. false / true|basic / advanced.
            - include_raw_content: Optional. false / true|markdown / text.
            - include_domains, exclude_domains: Optional. Comma/space-separated domains.
            - max_results, chunks_per_source, include_usage: Optional.
    """
    params = params or kwargs
    query = params.get("query", "")
    if not query or not str(query).strip():
        return {"error": "Please input a query."}

    tavily_api_key = os.getenv("tavily_api_key")
    if not tavily_api_key:
        return {"error": "Tavily API key is missing. Please set tavily_api_key in the environment."}
    project_id = params.get("project_id")
    client = TavilySearch(api_key=tavily_api_key, project_id=project_id)

    try:
        search_results = client.search(params)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Error occurred while searching: {str(e)}"}

    if not search_results.get("results"):
        return {
            "error": f"No results found for '{query}' in Tavily.",
            "raw": search_results,
        }

    report_text = _format_results_as_text(search_results, params)
    return {"report": report_text, "raw": search_results}
