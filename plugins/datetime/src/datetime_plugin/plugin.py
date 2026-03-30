from datetime_plugin.current_time import get_current_time


def register(context=None):
    return [get_current_time]
