"""general_tools.py"""

from itertools import zip_longest


def _grouper(iterable, n, fill_value=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fill_value)


def list_grouper(iterable: {list, tuple}, n: int, fill_value=None):
    """
    Returns a list of lists where each sub-list is of size 'n'
    :param iterable: list or tuple
    :param n: length of each sub-list
    :param fill_value: value to be populated as an element into a sub-list that does not have 'n' elements
    :return:
    """
    g = list(_grouper(iterable, n, fill_value))
    try:
        g[-1] = [e for e in g[-1] if e is not None]
        return [list(tup) for tup in g]
    except IndexError:
        return [[]]
