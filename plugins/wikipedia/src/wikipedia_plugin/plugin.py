from wikipedia_plugin.wikipedia_search import wikipedia_search


def register(context=None):
    return [wikipedia_search]
