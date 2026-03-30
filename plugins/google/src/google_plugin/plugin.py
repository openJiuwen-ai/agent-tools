from google_plugin.google_search import google_search
from google_plugin.google_image_search import google_image_search


def register(context=None):
    return [google_search, google_image_search]
