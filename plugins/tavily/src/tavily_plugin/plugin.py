from tavily_plugin.tavily_search import tavily_search
from tavily_plugin.tavily_research import tavily_research
from tavily_plugin.tavily_map import tavily_map
from tavily_plugin.tavily_extract import tavily_extract


def register(context=None):
    return [tavily_search, tavily_research, tavily_map, tavily_extract]
