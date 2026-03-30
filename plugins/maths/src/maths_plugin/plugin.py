from maths_plugin.eval_expression import eval_expression


def register(context=None):
    return [eval_expression]
