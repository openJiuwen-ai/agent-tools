from firecrawl_plugin.firecrawl_crawl import firecrawl_crawl
from firecrawl_plugin.firecrawl_map import firecrawl_map
from firecrawl_plugin.firecrawl_scrape import firecrawl_scrape


def register(context=None):
    return [firecrawl_scrape, firecrawl_map, firecrawl_crawl]
