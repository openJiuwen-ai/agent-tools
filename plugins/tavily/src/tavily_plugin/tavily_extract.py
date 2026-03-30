import os
from typing import Any

from openjiuwen.core.foundation.tool import tool
from tavily import TavilyClient


class TavilyExtract:
    """
    A class for extracting content from web pages using the Tavily Extract API.

    Args:
        api_key (str): The API key for accessing the Tavily Extract API.
        project_id (str, optional): The project ID for tracking and analytics.

    Methods:
        extract: Retrieves extracted content from the Tavily Extract API.
    """

    def __init__(self, api_key: str | None = None, project_id: str | None = None) -> None:
        self.client = TavilyClient(api_key=api_key, project_id=project_id)

    def extract(self, params: dict[str, Any]) -> dict:
        """
        Retrieves extracted content from the Tavily Extract API.

        Args:
            params (Dict[str, Any]): The extraction parameters, which may include:
                - urls: Required. A string or list of URLs to extract content from.
                - query: Optional string. User intent for reranking extracted content chunks.
                - include_images: Optional boolean. Whether to include images in the response.
                - include_favicon: Optional boolean. Whether to include favicon URLs.
                - extract_depth: Optional string. 'basic' or 'advanced'. Default is 'basic'.
                - format: Optional string. 'markdown' or 'text'. Default is 'markdown'.
                - chunks_per_source: Optional integer. Max content chunks per source (1-5).
                - timeout: Optional float. Request timeout in seconds (1.0-60.0).
                - include_usage: Optional boolean. Whether to include credit usage info.

        Returns:
            dict: The extracted content with results.
        """
        processed_params = self._process_params(params)
        return self.client.extract(**processed_params)

    @staticmethod
    def _process_params(params: dict[str, Any]) -> dict:
        """
        Processes and validates the extraction parameters.

        Args:
            params (Dict[str, Any]): The extraction parameters.

        Returns:
            dict: The processed parameters.
        """
        processed_params = {}

        if "urls" in params:
            urls = params["urls"]
            if isinstance(urls, str):
                url_list = [url.strip() for url in urls.split(",") if url.strip()]
                processed_params["urls"] = url_list
            elif isinstance(urls, list):
                processed_params["urls"] = urls
        else:
            raise ValueError("The 'urls' parameter is required.")

        if not processed_params.get("urls"):
            raise ValueError("At least one valid URL must be provided.")

        # Optional boolean parameters
        for key in ["include_images", "include_favicon", "include_usage"]:
            if key in params and params[key] is not None:
                value = params[key]
                if isinstance(value, str):
                    processed_params[key] = value.lower() == "true"
                else:
                    processed_params[key] = bool(value)

        if "extract_depth" in params and params["extract_depth"]:
            extract_depth = params["extract_depth"]
            if extract_depth not in ["basic", "advanced"]:
                raise ValueError("extract_depth must be either 'basic' or 'advanced'")
            processed_params["extract_depth"] = extract_depth

        if "format" in params and params["format"]:
            format_value = params["format"]
            if format_value not in ["markdown", "text"]:
                raise ValueError("format must be either 'markdown' or 'text'")
            processed_params["format"] = format_value

        if "query" in params and params.get("query"):
            processed_params["query"] = params["query"]

        if "chunks_per_source" in params and params["chunks_per_source"] is not None:
            chunks = params["chunks_per_source"]
            if isinstance(chunks, str):
                chunks = int(chunks)
            if chunks < 1 or chunks > 5:
                raise ValueError("chunks_per_source must be between 1 and 5")
            processed_params["chunks_per_source"] = chunks

        if "timeout" in params and params.get("timeout") is not None:
            timeout = params["timeout"]
            if isinstance(timeout, str):
                timeout = float(timeout)
            if timeout < 1.0 or timeout > 60.0:
                raise ValueError("timeout must be between 1.0 and 60.0 seconds")
            processed_params["timeout"] = timeout

        return processed_params


def _format_results_as_text(extract_results: dict) -> str:
    """
    Formats the extraction results into markdown text.

    Args:
        extract_results (dict): The extraction results.

    Returns:
        str: The formatted markdown text.
    """
    output_lines: list[str] = []
    for idx, result in enumerate(extract_results.get("results", []), 1):
        url = result.get("url", "")
        raw_content = result.get("raw_content", "")
        output_lines.append(f"# Extracted Content {idx}: {url}\n")

        if result.get("favicon"):
            output_lines.append(f"**Favicon:** ![Favicon]({result['favicon']})\n")

        output_lines.append(f"**Raw Content:**\n{raw_content}\n")

        if result.get("images"):
            output_lines.append("**Images:**\n")
            for image_url in result["images"]:
                output_lines.append(f"![Image]({image_url})\n")

        output_lines.append("---\n")

    if extract_results.get("failed_results"):
        output_lines.append("# Failed URLs:\n")
        for failed in extract_results["failed_results"]:
            url = failed.get("url", "")
            error = failed.get("error", "Unknown error")
            output_lines.append(f"- {url}: {error}\n")

    return "\n".join(output_lines)


@tool(
    name="tavily_extract",
    description=(
        "使用 Tavily Extract 从一个或多个指定 URL 提取网页内容。"
        "Extract raw content from web pages with basic or advanced extraction modes."
    ),
    input_params={
        "type": "object",
        "properties": {
            "urls": {
                "type": "string",
                "description": "One or more URLs to extract content from (comma-separated if multiple).",
            },
            "query": {
                "type": "string",
                "description": (
                    "Optional query string to rerank extracted content chunks based on relevance to user intent."
                ),
            },
            "include_images": {
                "type": "boolean",
                "description": "Include images from the URLs in the response. Default is false.",
                "default": False,
            },
            "include_favicon": {
                "type": "boolean",
                "description": "When set to true, includes the favicon URL for each result.",
                "default": False,
            },
            "extract_depth": {
                "type": "string",
                "description": "Extraction depth - 'basic' (default, faster) or 'advanced' (more data, higher cost).",
                "enum": ["basic", "advanced"],
                "default": "basic",
            },
            "format": {
                "type": "string",
                "description": "Format of extracted content - 'markdown' (default) or 'text' (plain text).",
                "enum": ["markdown", "text"],
                "default": "markdown",
            },
            "chunks_per_source": {
                "type": "integer",
                "description": "Maximum number of relevant content chunks (1-5) returned per source. Default is 3.",
                "minimum": 1,
                "maximum": 5,
                "default": 3,
            },
            "timeout": {
                "type": "number",
                "description": "Request timeout in seconds (1.0-60.0). Default is 10s for basic, 30s for advanced.",
                "minimum": 1,
                "maximum": 60,
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
        "required": ["urls"],
    },
)
def tavily_extract(params: dict[str, Any] | None = None, **kwargs) -> dict:
    """
    Invokes the Tavily Extract tool with the given parameters.

    Args:
        params (Dict[str, Any]): The parameters for the Tavily Extract tool.
            - urls: Required. One or more URLs (comma-separated string or list).
            - query: Optional. User intent for reranking content chunks.
            - include_images: Optional. Include images in the response.
            - include_favicon: Optional. Include favicon URL per result.
            - extract_depth: Optional. 'basic' or 'advanced'.
            - format: Optional. 'markdown' or 'text'.
            - chunks_per_source: Optional. 1-5 chunks per source.
            - timeout: Optional. 1.0-60.0 seconds.
            - include_usage: Optional. Include credit usage info.
    """
    params = params or kwargs

    urls = params.get("urls", "")
    if not urls:
        return {"error": "Please input at least one URL to extract."}
    if isinstance(urls, str) and not urls.strip():
        return {"error": "Please input at least one URL to extract."}
    if isinstance(urls, list) and not urls:
        return {"error": "Please input at least one URL to extract."}

    project_id = params.get("project_id")

    tavily_api_key = os.getenv("tavily_api_key")
    if not tavily_api_key:
        return {"error": "Tavily API key is missing. Please set tavily_api_key in the environment."}

    client = TavilyExtract(api_key=tavily_api_key, project_id=project_id)

    try:
        extract_results = client.extract(params)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Error occurred while extracting content: {str(e)}"}

    if not extract_results.get("results"):
        return {
            "error": "No content could be extracted from the provided URLs.",
            "raw": extract_results,
        }

    report_text = _format_results_as_text(extract_results)
    return {"report": report_text, "raw": extract_results}
