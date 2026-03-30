from jina_plugin.jina_reader import jina_reader
from jina_plugin.jina_search import jina_search


def register(context=None):
    return [jina_reader, jina_search]
