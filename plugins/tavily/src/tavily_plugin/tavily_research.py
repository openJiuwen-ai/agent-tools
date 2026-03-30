import os
import time
from typing import Any

from openjiuwen.core.foundation.tool import tool
from tavily import TavilyClient


class TavilyResearch:
    """
    A class for conducting deep research using the Tavily Research API.

    Args:
        api_key (str): The API key for accessing the Tavily Research API.
        project_id (str, optional): The project ID for tracking and analytics.

    Methods:
        research: Creates a research task.
        get_research: Gets research results by request_id.
    """

    def __init__(self, api_key: str | None = None, project_id: str | None = None) -> None:
        self.client = TavilyClient(api_key=api_key, project_id=project_id)

    def research(self, params: dict[str, Any]) -> dict:
        """
        Creates a research task.

        Args:
            params (Dict[str, Any]): The research parameters, which may include:
                - input: Required. The research task or question to investigate.
                - model: Optional string. The model to use ('mini', 'pro', or 'auto').
                - citation_format: Optional string. Citation format ('numbered', 'mla', 'apa', 'chicago').

        Returns:
            dict: Response containing request_id, created_at, status, input, and model.
        """
        processed_params = self._process_params(params)
        return self.client.research(**processed_params)

    def get_research(self, request_id: str) -> dict:
        """
        Gets research results by request_id.

        Args:
            request_id: The research request ID.

        Returns:
            dict: Research response containing request_id, created_at, completed_at, status, content, and sources.
        """
        return self.client.get_research(request_id)

    @staticmethod
    def _process_params(params: dict[str, Any]) -> dict:
        """
        Processes and validates the research parameters.

        Args:
            params (Dict[str, Any]): The research parameters.

        Returns:
            dict: The processed parameters.
        """
        processed_params = {}

        # Required parameter: input
        if "input" in params and params["input"]:
            processed_params["input"] = params["input"].strip()
        else:
            raise ValueError("The 'input' parameter is required.")

        # Optional parameter: model
        if "model" in params and params["model"]:
            model = params["model"]
            if model not in ["mini", "pro", "auto"]:
                raise ValueError("model must be 'mini', 'pro', or 'auto'")
            processed_params["model"] = model

        # Optional parameter: citation_format
        if "citation_format" in params and params["citation_format"]:
            citation_format = params["citation_format"]
            if citation_format not in ["numbered", "mla", "apa", "chicago"]:
                raise ValueError("citation_format must be 'numbered', 'mla', 'apa', or 'chicago'")
            processed_params["citation_format"] = citation_format

        return processed_params


@tool(
    name="tavily_research",
    description=(
        "使用 Tavily Research API 对任何主题进行深度研究，返回带引用的综合报告。"
        "Conduct deep, comprehensive research on any topic, returning detailed reports with citations and sources."
    ),
    input_params={
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": (
                    "The research task or question to investigate. Be specific and detailed for better results."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Research model: 'mini' (quick, targeted), 'pro' (comprehensive), 'auto' (automatic selection)."
                ),
                "enum": ["auto", "mini", "pro"],
                "default": "auto",
            },
            "citation_format": {
                "type": "string",
                "description": "Citation format for sources: numbered, MLA, APA, or Chicago style.",
                "enum": ["numbered", "mla", "apa", "chicago"],
                "default": "numbered",
            },
            "project_id": {
                "type": "string",
                "description": "Optional project ID for tracking and analytics. Sent as X-Project-ID header.",
            },
        },
        "required": ["input"],
    },
)
def tavily_research(params: dict[str, Any] | None = None, **kwargs) -> dict:
    """
    Invokes the Tavily Research tool with the given parameters.

    Args:
        params (Dict[str, Any]): The parameters for the Tavily Research tool.
            - input: Required. The research task or question to investigate.
            - model: Optional. 'mini', 'pro', or 'auto'.
            - citation_format: Optional. 'numbered', 'mla', 'apa', or 'chicago'.
    """
    params = params or kwargs
    research_input = params.get("input", "")
    if not research_input or not str(research_input).strip():
        return {"error": "Please input a research question or task."}

    tavily_api_key = os.getenv("tavily_api_key")
    if not tavily_api_key:
        return {"error": "Tavily API key is missing. Please set tavily_api_key in the environment."}

    project_id = params.get("project_id")
    tavily_research_client = TavilyResearch(api_key=tavily_api_key, project_id=project_id)

    # Step 1: Create research task
    try:
        create_response = tavily_research_client.research(params)
    except Exception as e:
        return {"error": f"Failed to create research task: {str(e)}"}

    request_id = create_response.get("request_id")
    if not request_id:
        return {
            "error": "Failed to create research task: No request_id returned.",
            "raw": create_response,
        }

    # Step 2: Poll for results (similar logic to TavilyResearchTool in reference)
    default_poll_interval_seconds = 3
    max_poll_interval_seconds = 10

    start_time = time.time()
    poll_interval = default_poll_interval_seconds

    while True:
        try:
            result = tavily_research_client.get_research(request_id)
        except Exception as e:
            return {"error": f"Error polling research status: {str(e)}"}

        status = result.get("status", "unknown")

        if status == "completed":
            break
        if status == "failed":
            error_msg = result.get("error", "Unknown error")
            return {"error": f"Research task failed: {error_msg}", "raw": result}
        if status in ["pending", "in_progress"]:
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.2, max_poll_interval_seconds)
        else:
            # Unknown status, keep polling but avoid tight loop
            time.sleep(poll_interval)

    # Step 3: Format and return results
    report_text = _format_results_as_text(result)
    elapsed = time.time() - start_time

    # Attach elapsed time info if not present
    if "response_time" not in result:
        result["response_time"] = elapsed

    return {
        "report": report_text,
        "raw": result,
    }


def _format_results_as_text(result: dict) -> str:
    """
    Formats the research results into markdown text.

    Args:
        result (dict): The research result from get_research.

    Returns:
        str: Formatted markdown with header, metadata, findings, and sources.
    """
    output_lines: list[str] = []

    # Header
    output_lines.append("# Research Report\n")

    # Metadata
    if result.get("request_id"):
        output_lines.append(f"**Request ID:** {result['request_id']}")
    if result.get("created_at"):
        output_lines.append(f"**Created:** {result['created_at']}")
    if result.get("completed_at"):
        output_lines.append(f"**Completed:** {result['completed_at']}")
    if result.get("response_time") is not None:
        try:
            output_lines.append(f"**Response Time:** {float(result['response_time']):.2f}s")
        except (TypeError, ValueError):
            pass

    output_lines.append("")  # Empty line

    # Content
    content = result.get("content", "")
    if content:
        output_lines.append("## Research Findings\n")
        output_lines.append(str(content))
        output_lines.append("")

    # Sources
    sources = result.get("sources", [])
    if sources:
        output_lines.append("## Sources\n")
        for idx, source in enumerate(sources, 1):
            if isinstance(source, dict):
                title = source.get("title", "Untitled")
                url = source.get("url", "")
                if url:
                    output_lines.append(f"{idx}. [{title}]({url})")
                else:
                    output_lines.append(f"{idx}. {title}")
            else:
                output_lines.append(f"{idx}. {source}")

    return "\n".join(output_lines)
