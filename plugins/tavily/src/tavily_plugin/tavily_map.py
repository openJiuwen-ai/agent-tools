import os
from typing import Any

from openjiuwen.core.foundation.tool import tool
from tavily import TavilyClient


class TavilyMap:
    """
    A class for mapping website structure using the Tavily Map API.

    Args:
        api_key (str): The API key for accessing the Tavily Map API.
        project_id (str, optional): The project ID for tracking and analytics.

    Methods:
        map: Retrieves a list of URLs from a website starting from a root URL.
    """

    def __init__(self, api_key: str | None = None, project_id: str | None = None) -> None:
        self.client = TavilyClient(api_key=api_key, project_id=project_id)

    def map(self, params: dict[str, Any]) -> dict:
        """
        Maps a website structure starting from a root URL.

        Args:
            params (Dict[str, Any]): The mapping parameters, which may include:
                - url: Required. The root URL to begin the mapping.
                - instructions: Optional string. Natural language guidance for crawling.
                - max_depth: Optional integer. Exploration distance from base URL (1-5).
                - max_breadth: Optional integer. Links followed per page level (1-500).
                - limit: Optional integer. Total links processed before halting.
                - select_paths: Optional list. Regex patterns to select specific paths.
                - select_domains: Optional list. Regex patterns to filter domains.
                - exclude_paths: Optional list. Regex patterns to exclude paths.
                - exclude_domains: Optional list. Regex patterns to exclude domains.
                - allow_external: Optional boolean. Include external domain links.
                - timeout: Optional float. Maximum wait duration (10-150 seconds).
                - include_usage: Optional boolean. Return credit consumption data.

        Returns:
            dict: The mapping results containing discovered URLs.
        """
        processed_params = self._process_params(params)
        return self.client.map(**processed_params)

    @staticmethod
    def _process_params(params: dict[str, Any]) -> dict:
        """
        Processes and validates the mapping parameters.

        Args:
            params (Dict[str, Any]): The mapping parameters.

        Returns:
            dict: The processed parameters.
        """
        processed_params = {}

        # Required parameter: url
        if "url" in params and params["url"]:
            processed_params["url"] = params["url"].strip()
        else:
            raise ValueError("The 'url' parameter is required.")

        # Optional string parameter: instructions
        if "instructions" in params and params["instructions"]:
            processed_params["instructions"] = params["instructions"]

        # Optional integer parameters with bounds
        if "max_depth" in params and params["max_depth"] is not None:
            max_depth = params["max_depth"]
            if isinstance(max_depth, str):
                max_depth = int(max_depth)
            if max_depth < 1 or max_depth > 5:
                raise ValueError("max_depth must be between 1 and 5")
            processed_params["max_depth"] = max_depth

        if "max_breadth" in params and params["max_breadth"] is not None:
            max_breadth = params["max_breadth"]
            if isinstance(max_breadth, str):
                max_breadth = int(max_breadth)
            if max_breadth < 1 or max_breadth > 500:
                raise ValueError("max_breadth must be between 1 and 500")
            processed_params["max_breadth"] = max_breadth

        if "limit" in params and params["limit"] is not None:
            limit = params["limit"]
            if isinstance(limit, str):
                limit = int(limit)
            if limit < 1:
                raise ValueError("limit must be at least 1")
            processed_params["limit"] = limit

        # Optional list parameters (comma-separated strings or lists)
        for key in ["select_paths", "select_domains", "exclude_paths", "exclude_domains"]:
            if key in params and params[key]:
                value = params[key]
                if isinstance(value, str):
                    processed_params[key] = [item.strip() for item in value.split(",") if item.strip()]
                elif isinstance(value, list):
                    processed_params[key] = value

        # Optional boolean parameter: allow_external
        if "allow_external" in params and params["allow_external"] is not None:
            value = params["allow_external"]
            if isinstance(value, str):
                processed_params["allow_external"] = value.lower() == "true"
            else:
                processed_params["allow_external"] = bool(value)

        # Optional float parameter: timeout
        if "timeout" in params and params["timeout"] is not None:
            timeout = params["timeout"]
            if isinstance(timeout, str):
                timeout = float(timeout)
            if timeout < 10 or timeout > 150:
                raise ValueError("timeout must be between 10 and 150 seconds")
            processed_params["timeout"] = timeout

        # Optional boolean parameter: include_usage
        if "include_usage" in params and params["include_usage"] is not None:
            value = params["include_usage"]
            if isinstance(value, str):
                processed_params["include_usage"] = value.lower() == "true"
            else:
                processed_params["include_usage"] = bool(value)

        return processed_params


def _format_results_as_text(map_results: dict, tool_params: dict[str, Any]) -> str:
    """
    Formats the mapping results into markdown text.

    Args:
        map_results (dict): The mapping results.
        tool_params (dict): The tool parameters (e.g. for base_url fallback).

    Returns:
        str: The formatted markdown text.
    """
    output_lines: list[str] = []

    base_url = map_results.get("base_url", tool_params.get("url", ""))
    output_lines.append(f"# Website Map for: {base_url}\n")

    results = map_results.get("results", [])
    output_lines.append(f"**Total URLs discovered:** {len(results)}\n")

    if map_results.get("response_time") is not None:
        try:
            output_lines.append(f"**Response time:** {float(map_results['response_time']):.2f} seconds\n")
        except (TypeError, ValueError):
            pass

    if map_results.get("usage") is not None:
        output_lines.append(f"**API Credits used:** {map_results['usage']}\n")

    output_lines.append("\n## Discovered URLs:\n")

    for idx, url in enumerate(results, 1):
        output_lines.append(f"{idx}. {url}\n")

    return "\n".join(output_lines)


@tool(
    name="tavily_map",
    description=(
        "使用 Tavily Map 从根 URL 开始发现所有 URL，映射网站结构。"
        "Map a website's structure by discovering all URLs starting from a root URL."
    ),
    input_params={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The root URL to begin the website mapping from."},
            "instructions": {
                "type": "string",
                "description": (
                    "Natural language guidance for crawling behavior. "
                    "Note that using instructions doubles API credit consumption."
                ),
            },
            "max_depth": {
                "type": "integer",
                "description": "Exploration distance from base URL. Range is 1-5. Default is 1.",
                "minimum": 1,
                "maximum": 5,
                "default": 1,
            },
            "max_breadth": {
                "type": "integer",
                "description": "Number of links followed per page level. Range is 1-500. Default is 20.",
                "minimum": 1,
                "maximum": 500,
                "default": 20,
            },
            "limit": {
                "type": "integer",
                "description": "Total links processed before halting. Minimum is 1. Default is 50.",
                "minimum": 1,
                "default": 50,
            },
            "select_paths": {
                "type": "string",
                "description": (
                    "Comma-separated regex patterns to filter URLs by path. "
                    "Only URLs matching these patterns will be included."
                ),
            },
            "select_domains": {
                "type": "string",
                "description": (
                    "Comma-separated regex patterns to filter URLs by domain. "
                    "Only URLs from matching domains will be included."
                ),
            },
            "exclude_paths": {
                "type": "string",
                "description": (
                    "Comma-separated regex patterns to exclude URLs by path. "
                    "URLs matching these patterns will be omitted."
                ),
            },
            "exclude_domains": {
                "type": "string",
                "description": (
                    "Comma-separated regex patterns to exclude URLs by domain. "
                    "URLs from matching domains will be blocked."
                ),
            },
            "allow_external": {
                "type": "boolean",
                "description": "When set to true, includes external domain links in results. Default is true.",
                "default": True,
            },
            "timeout": {
                "type": "number",
                "description": "Maximum wait duration in seconds. Range is 10-150. Default is 150.",
                "minimum": 10,
                "maximum": 150,
                "default": 150,
            },
            "include_usage": {
                "type": "boolean",
                "description": "When set to true, includes API credit usage information in the response.",
                "default": False,
            },
            "project_id": {
                "type": "string",
                "description": (
                    "Optional project ID for usage tracking and analytics. Will be sent as X-Project-ID header."
                ),
            },
        },
        "required": ["url"],
    },
)
def tavily_map(params: dict[str, Any] | None = None, **kwargs) -> dict:
    """
    Invokes the Tavily Map tool with the given parameters.

    Args:
        params (Dict[str, Any]): The parameters for the Tavily Map tool.
            - url: Required. The root URL to begin the mapping.
            - instructions: Optional. Natural language guidance for crawling.
            - max_depth: Optional. Exploration distance (1-5).
            - max_breadth: Optional. Links per page level (1-500).
            - limit: Optional. Total links before halting.
            - select_paths, select_domains, exclude_paths, exclude_domains: Optional. Regex filters.
            - allow_external: Optional. Include external links.
            - timeout: Optional. Max wait in seconds (10-150).
            - include_usage: Optional. Return credit usage.
    """
    params = params or kwargs

    url = params.get("url", "")
    if not url or not str(url).strip():
        return {"error": "Please input a URL to map."}

    # Remove credentials / project_id from params passed to client; keep project_id for constructor
    project_id = params.get("project_id")

    tavily_api_key = os.getenv("tavily_api_key")
    if not tavily_api_key:
        return {"error": "Tavily API key is missing. Please set tavily_api_key in the environment."}

    client = TavilyMap(api_key=tavily_api_key, project_id=project_id)

    try:
        map_results = client.map(params)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Error occurred while mapping website: {str(e)}"}

    if not map_results.get("results"):
        return {"error": f"No URLs could be discovered from '{url}'.", "raw": map_results}

    report_text = _format_results_as_text(map_results, params)
    return {"report": report_text, "raw": map_results}
