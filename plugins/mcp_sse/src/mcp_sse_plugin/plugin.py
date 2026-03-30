from .mcp_call_tool import mcp_call_tool
from .mcp_list_tools import mcp_list_tools


def register(context=None):
    return [
        mcp_list_tools,
        mcp_call_tool,
    ]
